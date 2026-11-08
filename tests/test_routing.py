import asyncio
import random
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from typing import Any, cast

import httpx2 as httpx
import pytest
from openai import (
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from worldwarpixel.routing import (
    MAX_RATE_LIMIT_RETRIES,
    Endpoint,
    EndpointPool,
    describe_endpoint,
    endpoint_client,
    endpoint_headers,
    modal_endpoint,
    retry_after,
)


class FakeClient:
    def __init__(
        self, fail: Exception | None = None, outcomes: list[Exception | str] | None = None
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail
        self.outcomes = outcomes or []
        self.before_fail: Callable[[], None] | None = None

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=self)

    async def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if self.fail is not None:
            if self.before_fail is not None:
                self.before_fail()
            raise self.fail
        return kw["model"]


class NoJitter(random.Random):
    def uniform(self, _a: float, _b: float) -> float:
        return 0.0


def _rate_limited(headers: dict[str, str] | None = None) -> RateLimitError:
    request = httpx.Request("POST", "https://x/v1/chat/completions")
    return RateLimitError(
        "429", response=httpx.Response(429, request=request, headers=headers), body=None
    )


def _server_error() -> InternalServerError:
    request = httpx.Request("POST", "https://x/v1/chat/completions")
    return InternalServerError("503", response=httpx.Response(503, request=request), body=None)


def _unauthorized() -> AuthenticationError:
    request = httpx.Request("POST", "https://x/v1/chat/completions")
    return AuthenticationError("401", response=httpx.Response(401, request=request), body=None)


def _pool(
    n: int,
    clock: list[float] | None = None,
    *,
    sleep: Any = asyncio.sleep,
    rng: random.Random | None = None,
    max_in_flight: int = 16,
    initial_in_flight: int | None = None,
) -> tuple[EndpointPool, list[FakeClient]]:
    clients = [FakeClient() for _ in range(n)]
    endpoints = [
        Endpoint(
            f"ws{i}",
            f"https://ws{i}/v1",
            f"m{i}",
            "modal",
            (),
            max_in_flight=max_in_flight,
            initial_in_flight=initial_in_flight,
        )
        for i in range(n)
    ]
    t = clock if clock is not None else [0.0]
    pool = EndpointPool(
        list(zip(endpoints, clients, strict=True)),
        clock=lambda: t[0],
        sleep=sleep,
        rng=rng,
    )
    return pool, clients


@pytest.mark.parametrize(
    ("nkeys", "expected"),
    [(9, [3, 3, 3]), (10, [3, 3, 4])],
)
def test_least_loaded_assignment_is_even_and_sticky(nkeys: int, expected: list[int]) -> None:
    pool, _ = _pool(3)
    routes = {f"run:{i}": pool.route(f"run:{i}") for i in range(nkeys)}
    assert sorted(Counter(routes.values()).values()) == expected
    assert all(pool.route(key) == endpoint for key, endpoint in routes.items())

    key = "run:sticky"
    route = pool.route(key)

    async def calls() -> None:
        assert cast(Any, await pool.create(key, messages=[])) == f"m{route[-1]}"
        assert cast(Any, await pool.create(key, messages=[])) == f"m{route[-1]}"

    asyncio.run(calls())
    assert pool.route(key) == route


def test_release_frees_a_run_prefix_for_new_assignments() -> None:
    pool, _ = _pool(3)
    for i in range(3):
        pool.route(f"keep:{i}")
    released_endpoint = pool.route("run1:agent")
    pool.route("keep:extra1")
    pool.route("keep:extra2")
    assert {stat["agents"] for stat in pool.public_stats().values()} == {2}

    pool.release("run1:")

    assert pool.public_stats()[released_endpoint]["agents"] == 1
    assert pool.route("run2:agent") == released_endpoint


def test_retry_after_parses_seconds_milliseconds_dates_and_clamps() -> None:
    assert retry_after(_rate_limited({"retry-after": "7"}), 15) == 7
    assert retry_after(_rate_limited({"retry-after-ms": "2500"}), 15) == 2.5
    assert retry_after(_rate_limited({"retry-after": "10000"}), 15) == 300
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert retry_after(_rate_limited({"retry-after": date}), 15) == pytest.approx(30, abs=1)
    assert retry_after(_rate_limited(), 15) == 15
    assert retry_after(_rate_limited({"retry-after": "later"}), 15) == 15
    assert retry_after(_rate_limited({"retry-after-ms": "bad", "retry-after": "7"}), 15) == 7


def test_429_waits_and_retries_the_same_endpoint() -> None:
    clock = [0.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    pool, clients = _pool(2, clock, sleep=sleep, rng=NoJitter())
    endpoint_name = pool.route("agent")
    client = clients[int(endpoint_name[-1])]
    client.outcomes = [_rate_limited({"retry-after": "7"}), "recovered"]

    result = asyncio.run(pool.create("agent", messages=[]))

    assert cast(Any, result) == "recovered"
    assert len(client.calls) == 2
    assert sum(len(other.calls) for other in clients if other is not client) == 0
    assert slept == [7]
    assert pool.stats[endpoint_name]["rate_limited"] == 1
    assert pool.public_stats()[endpoint_name]["cooling"] is False


def test_rate_limit_pauses_only_keys_assigned_to_that_endpoint() -> None:
    async def main() -> None:
        clock = [0.0]
        gate = asyncio.Event()
        sleeping = asyncio.Event()
        slept: list[float] = []

        async def sleep(seconds: float) -> None:
            slept.append(seconds)
            sleeping.set()
            await gate.wait()
            clock[0] += seconds

        pool, clients = _pool(2, clock, sleep=sleep, rng=NoJitter())
        routes = {f"key:{i}": pool.route(f"key:{i}") for i in range(20)}
        endpoint_name = routes["key:0"]
        same_endpoint = [key for key, endpoint in routes.items() if endpoint == endpoint_name]
        different_endpoint = next(
            key for key, endpoint in routes.items() if endpoint != endpoint_name
        )
        endpoint_client = clients[int(endpoint_name[-1])]
        endpoint_client.outcomes = [
            _rate_limited({"retry-after": "20"}),
            "first resumed call",
            "second resumed call",
        ]

        first = asyncio.create_task(pool.create(same_endpoint[0], messages=[]))
        await asyncio.wait_for(sleeping.wait(), timeout=1)
        second = asyncio.create_task(pool.create(same_endpoint[1], messages=[]))
        await asyncio.sleep(0)

        assert pool.public_stats()[endpoint_name]["waiting"] == 2
        assert len(endpoint_client.calls) == 1 and clock[0] == 0
        before_other_endpoint = len(slept)
        other_result = await pool.create(different_endpoint, messages=[])
        assert cast(Any, other_result) == f"m{pool.route(different_endpoint)[-1]}"
        assert len(slept) == before_other_endpoint

        gate.set()
        results = await asyncio.gather(first, second)
        assert [cast(Any, result) for result in results] == [
            "first resumed call",
            "second resumed call",
        ]
        assert len(endpoint_client.calls) == 3
        assert pool.public_stats()[endpoint_name]["waiting"] == 0
        assert clock[0] >= 20

    asyncio.run(main())


def test_rate_limit_gives_up_after_maximum_and_attributes_error() -> None:
    clock = [0.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    pool, clients = _pool(1, clock, sleep=sleep, rng=NoJitter())
    clients[0].outcomes = [
        _rate_limited({"retry-after": "1"}) for _ in range(MAX_RATE_LIMIT_RETRIES)
    ]
    endpoint = pool.order("agent")[0][0]

    with pytest.raises(RateLimitError) as exc:
        asyncio.run(pool.create("agent", messages=[]))

    assert len(clients[0].calls) == MAX_RATE_LIMIT_RETRIES
    assert cast(Any, exc.value).endpoint == endpoint
    assert pool.stats[endpoint.name]["rate_limited"] == MAX_RATE_LIMIT_RETRIES
    assert len(slept) == MAX_RATE_LIMIT_RETRIES - 1


def test_5xx_still_fails_over_and_cools_endpoint() -> None:
    clock = [0.0]
    pool, clients = _pool(2, clock)
    first = pool.route("a")
    failing_client = clients[int(first[-1])]
    failing_client.fail = _server_error()
    other = "ws1" if first == "ws0" else "ws0"

    assert cast(Any, asyncio.run(pool.create("a"))) == "m" + other[-1]
    assert pool.stats[first] == {"calls": 1, "errors": 1, "failovers": 1, "rate_limited": 0}
    assert pool.route("a") == other
    asyncio.run(pool.create("a"))
    assert len(failing_client.calls) == 1
    clock[0] = 100.0
    assert pool.route("a") == first


def test_5xx_failover_skips_rate_limited_endpoints() -> None:
    pool, clients = _pool(3)
    key = "agent"
    primary = pool.route(key)
    ordered = pool.order(key)
    limited_name = ordered[1][0].name
    healthy_name = ordered[2][0].name
    pool._limited_until[limited_name] = 20
    clients[int(primary[-1])].fail = _server_error()

    assert cast(Any, asyncio.run(pool.create(key))) == f"m{healthy_name[-1]}"
    assert clients[int(limited_name[-1])].calls == []
    assert len(clients[int(healthy_name[-1])].calls) == 1


def test_failover_does_not_reassign_a_key_released_in_flight() -> None:
    pool, clients = _pool(2)
    key = "run1:agent"
    retained_key = "run2:agent"
    primary = pool.route(key)
    pool.route(retained_key)
    failing_client = clients[int(primary[-1])]
    failing_client.fail = _server_error()
    failing_client.before_fail = lambda: pool.release("run1:")
    fallback = "ws1" if primary == "ws0" else "ws0"

    result = asyncio.run(pool.create(key))

    assert cast(Any, result) == f"m{fallback[-1]}"
    assert sum(stat["agents"] for stat in pool.public_stats().values()) == 1
    assert pool.public_stats()[pool.route(retained_key)]["agents"] == 1


def test_all_down_endpoints_raise_the_last_error() -> None:
    pool, clients = _pool(2)
    for client in clients:
        client.fail = _server_error()
    last_endpoint = pool.order("a")[-1][0]

    with pytest.raises(InternalServerError) as exc:
        asyncio.run(pool.create("a"))

    assert all(len(client.calls) == 1 for client in clients)
    assert cast(Any, exc.value).endpoint == last_endpoint


def test_authentication_error_is_attributed_without_failing_over() -> None:
    pool, _ = _pool(2)
    first_endpoint, first_client = pool.order("a")[0]
    first_client.fail = _unauthorized()
    other_endpoint, other_client = next(
        (endpoint, client) for endpoint, client in pool.order("a") if endpoint != first_endpoint
    )

    with pytest.raises(AuthenticationError) as exc:
        asyncio.run(pool.create("a"))

    assert cast(Any, exc.value).endpoint == first_endpoint
    assert pool.stats[first_endpoint.name] == {
        "calls": 1,
        "errors": 1,
        "failovers": 0,
        "rate_limited": 0,
    }
    assert pool.stats[other_endpoint.name] == {
        "calls": 0,
        "errors": 0,
        "failovers": 0,
        "rate_limited": 0,
    }
    assert not pool.public_stats()[other_endpoint.name]["cooling"]
    assert not other_client.calls
    assert describe_endpoint(exc.value) == f"{first_endpoint.name} ({first_endpoint.base_url})"


def test_public_stats_exposes_waiting_retry_and_agent_counts() -> None:
    clock = [0.0]
    pool, _ = _pool(2, clock)
    endpoint_name = pool.route("agent")
    pool._limited_until[endpoint_name] = 5
    pool._waiting[endpoint_name] = 2

    stats = pool.public_stats()[endpoint_name]

    assert stats["agents"] == 1
    assert stats["retry_in"] == 5
    assert stats["waiting"] == 2
    assert stats["limit"] == 4
    assert stats["in_flight"] == 0


def test_endpoint_limits_concurrent_calls_and_releases_a_waiter() -> None:
    async def main() -> None:
        class BlockingClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.entered: asyncio.Queue[int] = asyncio.Queue()
                self.releases: list[asyncio.Event] = []

            @property
            def chat(self) -> Any:
                return SimpleNamespace(completions=self)

            async def create(self, **kwargs: Any) -> str:
                index = len(self.calls)
                self.calls.append(kwargs)
                release = asyncio.Event()
                self.releases.append(release)
                await self.entered.put(index)
                await release.wait()
                return cast(str, kwargs["model"])

        client = BlockingClient()
        endpoint = Endpoint(
            "only",
            "https://only/v1",
            "m",
            "bearer",
            (),
            max_in_flight=2,
            initial_in_flight=2,
        )
        pool = EndpointPool([(endpoint, client)])
        tasks = [asyncio.create_task(pool.create(f"key:{i}")) for i in range(3)]

        assert [await asyncio.wait_for(client.entered.get(), 1) for _ in range(2)] == [0, 1]
        await asyncio.sleep(0)
        assert len(client.calls) == 2
        assert pool.public_stats()["only"]["in_flight"] == 2

        client.releases[0].set()
        assert await asyncio.wait_for(client.entered.get(), 1) == 2
        assert pool.public_stats()["only"]["in_flight"] == 2

        for release in client.releases:
            release.set()
        await asyncio.gather(*tasks)
        assert pool.public_stats()["only"]["in_flight"] == 0

    asyncio.run(main())


def test_cancelled_slot_waiter_does_not_block_the_queue_or_leak() -> None:
    async def main() -> None:
        class BlockingClient:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.entered: asyncio.Queue[int] = asyncio.Queue()
                self.releases: list[asyncio.Event] = []

            @property
            def chat(self) -> Any:
                return SimpleNamespace(completions=self)

            async def create(self, **kwargs: Any) -> str:
                index = len(self.calls)
                self.calls.append(kwargs)
                release = asyncio.Event()
                self.releases.append(release)
                await self.entered.put(index)
                await release.wait()
                return cast(str, kwargs["model"])

        client = BlockingClient()
        endpoint = Endpoint(
            "only",
            "https://only/v1",
            "m",
            "bearer",
            (),
            max_in_flight=1,
            initial_in_flight=1,
        )
        pool = EndpointPool([(endpoint, client)])
        active = asyncio.create_task(pool.create("active"))
        assert await asyncio.wait_for(client.entered.get(), 1) == 0

        cancelled = asyncio.create_task(pool.create("cancelled"))
        queued = asyncio.create_task(pool.create("queued"))
        await asyncio.sleep(0)
        assert len(pool._slot_waiters["only"]) == 2

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert len(pool._slot_waiters["only"]) == 1

        client.releases[0].set()
        assert await asyncio.wait_for(client.entered.get(), 1) == 1
        assert len(client.calls) == 2
        client.releases[1].set()
        await asyncio.gather(active, queued)
        assert pool.public_stats()["only"]["in_flight"] == 0

    asyncio.run(main())


def test_retry_after_wait_does_not_hold_an_endpoint_slot() -> None:
    async def main() -> None:
        clock = [0.0]
        sleeping = asyncio.Event()
        resume = asyncio.Event()

        async def sleep(seconds: float) -> None:
            sleeping.set()
            await resume.wait()
            clock[0] += seconds

        pool, clients = _pool(
            1,
            clock,
            sleep=sleep,
            rng=NoJitter(),
            max_in_flight=1,
            initial_in_flight=1,
        )
        clients[0].outcomes = [_rate_limited({"retry-after": "20"}), "recovered"]
        call = asyncio.create_task(pool.create("key"))
        await asyncio.wait_for(sleeping.wait(), 1)

        assert pool.public_stats()["ws0"]["in_flight"] == 0
        assert pool.public_stats()["ws0"]["waiting"] == 1

        resume.set()
        assert cast(Any, await call) == "recovered"
        assert pool.public_stats()["ws0"]["in_flight"] == 0

    asyncio.run(main())


def test_waiting_slot_rechecks_retry_after_before_calling_endpoint() -> None:
    async def main() -> None:
        clock = [0.0]
        sleep_started = asyncio.Event()
        resume_sleep = asyncio.Event()

        async def sleep(seconds: float) -> None:
            sleep_started.set()
            await resume_sleep.wait()
            clock[0] += seconds

        class LimitedFirstCall:
            def __init__(self) -> None:
                self.calls = 0
                self.first_entered = asyncio.Event()
                self.release_first = asyncio.Event()

            @property
            def chat(self) -> Any:
                return SimpleNamespace(completions=self)

            async def create(self, **kwargs: Any) -> str:
                self.calls += 1
                if self.calls == 1:
                    self.first_entered.set()
                    await self.release_first.wait()
                    raise _rate_limited({"retry-after": "20"})
                return cast(str, kwargs["model"])

        client = LimitedFirstCall()
        endpoint = Endpoint(
            "only",
            "https://only/v1",
            "m",
            "bearer",
            (),
            max_in_flight=1,
            initial_in_flight=1,
        )
        pool = EndpointPool(
            [(endpoint, client)], clock=lambda: clock[0], sleep=sleep, rng=NoJitter()
        )
        first = asyncio.create_task(pool.create("first"))
        await asyncio.wait_for(client.first_entered.wait(), 1)
        second = asyncio.create_task(pool.create("second"))
        await asyncio.sleep(0)
        assert len(pool._slot_waiters["only"]) == 1

        client.release_first.set()
        await asyncio.wait_for(sleep_started.wait(), 1)
        while pool.public_stats()["only"]["waiting"] < 2:
            await asyncio.sleep(0)
        assert client.calls == 1
        assert pool.public_stats()["only"]["in_flight"] == 0

        resume_sleep.set()
        await asyncio.gather(first, second)
        assert client.calls == 3
        assert pool.public_stats()["only"]["in_flight"] == 0

    asyncio.run(main())


def test_rate_limits_halve_once_per_retry_window_and_not_below_one() -> None:
    async def main() -> None:
        clock = [0.0]
        sleep_gate = asyncio.Event()
        sleep_started = asyncio.Event()
        sleep_count = 0

        async def sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            sleep_started.set()
            await sleep_gate.wait()
            clock[0] += seconds

        class BurstClient:
            def __init__(self) -> None:
                self.calls = 0
                self.entered: asyncio.Queue[int] = asyncio.Queue()
                self.release: list[asyncio.Event] = []
                self.fail_count = 0

            @property
            def chat(self) -> Any:
                return SimpleNamespace(completions=self)

            async def create(self, **kwargs: Any) -> str:
                index = self.calls
                self.calls += 1
                if index < 2 or self.fail_count:
                    if index >= 2:
                        self.fail_count -= 1
                        raise _rate_limited({"retry-after": "20"})
                    gate = asyncio.Event()
                    self.release.append(gate)
                    await self.entered.put(index)
                    await gate.wait()
                    raise _rate_limited({"retry-after": "20"})
                return cast(str, kwargs["model"])

        client = BurstClient()
        endpoint = Endpoint(
            "only",
            "https://only/v1",
            "m",
            "bearer",
            (),
            max_in_flight=8,
            initial_in_flight=8,
        )
        pool = EndpointPool(
            [(endpoint, client)], clock=lambda: clock[0], sleep=sleep, rng=NoJitter()
        )
        first = asyncio.create_task(pool.create("first"))
        second = asyncio.create_task(pool.create("second"))
        assert [await asyncio.wait_for(client.entered.get(), 1) for _ in range(2)] == [0, 1]

        client.release[0].set()
        await asyncio.wait_for(sleep_started.wait(), 1)
        assert pool.public_stats()["only"]["limit"] == 4
        client.release[1].set()
        while sleep_count < 2:
            await asyncio.sleep(0)
        assert pool.public_stats()["only"]["limit"] == 4

        sleep_gate.set()
        await asyncio.gather(first, second)
        assert clock[0] >= 20

        client.fail_count = 1
        assert cast(Any, await pool.create("after-window")) == "m"
        assert pool.public_stats()["only"]["limit"] == 2

        client.fail_count = MAX_RATE_LIMIT_RETRIES
        with pytest.raises(RateLimitError):
            await pool.create("continue-halving")
        assert pool.public_stats()["only"]["limit"] == 1

    asyncio.run(main())


def test_successes_increase_limit_additively_up_to_the_ceiling() -> None:
    pool, _ = _pool(1, max_in_flight=4, initial_in_flight=2)

    async def calls(count: int) -> None:
        for _ in range(count):
            await pool.create("key")

    async def main() -> None:
        await calls(2)
        assert pool.public_stats()["ws0"]["limit"] == 3
        await calls(3)
        assert pool.public_stats()["ws0"]["limit"] == 4
        await calls(8)
        assert pool.public_stats()["ws0"]["limit"] == 4

    asyncio.run(main())


def test_429_epoch_ignores_old_successes_until_four_fresh_successes() -> None:
    async def main() -> None:
        clock = [0.0]
        sleep_started = asyncio.Event()
        resume_sleep = asyncio.Event()

        async def sleep(seconds: float) -> None:
            sleep_started.set()
            await resume_sleep.wait()
            clock[0] += seconds

        class ControlledClient:
            def __init__(self) -> None:
                self.calls = 0
                self.entered: asyncio.Queue[int] = asyncio.Queue()
                self.releases: list[asyncio.Event] = []

            @property
            def chat(self) -> Any:
                return SimpleNamespace(completions=self)

            async def create(self, **kwargs: Any) -> str:
                index = self.calls
                self.calls += 1
                release = asyncio.Event()
                self.releases.append(release)
                await self.entered.put(index)
                await release.wait()
                if index == 0:
                    raise _rate_limited({"retry-after": "20"})
                return cast(str, kwargs["model"])

        client = ControlledClient()
        endpoint = Endpoint(
            "only",
            "https://only/v1",
            "m",
            "bearer",
            (),
            max_in_flight=8,
            initial_in_flight=8,
        )
        pool = EndpointPool(
            [(endpoint, client)], clock=lambda: clock[0], sleep=sleep, rng=NoJitter()
        )
        original = [asyncio.create_task(pool.create(f"old:{i}")) for i in range(5)]
        assert [await asyncio.wait_for(client.entered.get(), 1) for _ in range(5)] == list(range(5))

        client.releases[0].set()
        await asyncio.wait_for(sleep_started.wait(), 1)
        assert pool.public_stats()["only"]["limit"] == 4

        for release in client.releases[1:3]:
            release.set()
        await asyncio.gather(*original[1:3])
        assert pool.public_stats()["only"]["limit"] == 4
        assert pool._successes_since_change["only"] == 0

        resume_sleep.set()
        assert await asyncio.wait_for(client.entered.get(), 1) == 5
        for release in client.releases[3:5]:
            release.set()
        await asyncio.gather(*original[3:])
        assert pool.public_stats()["only"]["limit"] == 4
        assert pool._successes_since_change["only"] == 0

        fresh = [asyncio.create_task(pool.create(f"fresh:{i}")) for i in range(3)]
        assert [await asyncio.wait_for(client.entered.get(), 1) for _ in range(3)] == [6, 7, 8]
        for release in client.releases[5:]:
            release.set()
        await asyncio.gather(original[0], *fresh)
        assert pool.public_stats()["only"]["limit"] == 5
        assert pool._successes_since_change["only"] == 0
        assert pool.public_stats()["only"]["in_flight"] == 0

    asyncio.run(main())


def test_describe_endpoint_returns_unknown_for_unattributed_error() -> None:
    assert describe_endpoint(ValueError()) == "unknown endpoint"


def test_timeouts_do_not_fail_over() -> None:
    pool, clients = _pool(2)
    for client in clients:
        client.fail = APITimeoutError(httpx.Request("POST", "https://x"))

    with pytest.raises(APITimeoutError):
        asyncio.run(pool.create("a"))

    assert sum(len(client.calls) for client in clients) == 1


def test_endpoint_headers_need_every_secret() -> None:
    endpoint = modal_endpoint("chryztoph", "ep-x-server", "m", secret_prefix="CHRYZTOPH_")
    assert endpoint.base_url == "https://chryztoph--ep-x-server.us-west.modal.direct/v1"
    env = {"CHRYZTOPH_MODAL_KEY": "k", "CHRYZTOPH_MODAL_SECRET": "s"}
    assert endpoint_headers(endpoint, lambda name: env.get(name, "")) == {
        "Modal-Key": "k",
        "Modal-Secret": "s",
    }
    env.pop("CHRYZTOPH_MODAL_SECRET")
    assert endpoint_headers(endpoint, lambda name: env.get(name, "")) is None


def test_endpoint_clients_disable_sdk_retries(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("worldwarpixel.routing.AsyncOpenAI", client)
    modal = Endpoint("modal", "https://modal/v1", "m", "modal", ("KEY", "SECRET"))
    bearer = Endpoint("bearer", "https://bearer/v1", "m", "bearer", ("API_KEY",))

    endpoint_client(modal, lambda _: "secret", timeout=10)
    endpoint_client(bearer, lambda _: "secret", timeout=10)

    assert len(calls) == 2
    assert all(kwargs["max_retries"] == 0 for kwargs in calls)
