import asyncio
from collections import Counter
from types import SimpleNamespace
from typing import Any, cast

import httpx2 as httpx
import pytest
from openai import APITimeoutError, RateLimitError

from worldwarpixel.routing import Endpoint, EndpointPool, endpoint_headers, modal_endpoint


class FakeClient:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=self)

    async def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.fail is not None:
            raise self.fail
        return kw["model"]


def _rate_limited() -> RateLimitError:
    request = httpx.Request("POST", "https://x/v1/chat/completions")
    return RateLimitError("429", response=httpx.Response(429, request=request), body=None)


def _pool(n: int, clock: list[float] | None = None) -> tuple[EndpointPool, list[FakeClient]]:
    clients = [FakeClient() for _ in range(n)]
    eps = [Endpoint(f"ws{i}", f"https://ws{i}/v1", f"m{i}", "modal", ()) for i in range(n)]
    t = clock if clock is not None else [0.0]
    return EndpointPool(list(zip(eps, clients, strict=True)), clock=lambda: t[0]), clients


def test_keys_stick_to_one_endpoint_and_spread_evenly() -> None:
    pool, _ = _pool(3)
    routes = {f"run:{i}": pool.route(f"run:{i}") for i in range(300)}
    assert all(pool.route(k) == r for k, r in routes.items())  # stable per key
    counts = Counter(routes.values())
    assert set(counts) == {"ws0", "ws1", "ws2"}
    assert min(counts.values()) > 60


def test_losing_an_endpoint_only_moves_its_keys() -> None:
    pool, clients = _pool(3)
    eps = [Endpoint(f"ws{i}", f"https://ws{i}/v1", f"m{i}", "modal", ()) for i in range(3)]
    smaller = EndpointPool([(eps[0], clients[0]), (eps[1], clients[1])])
    for i in range(200):
        before = pool.route(f"k{i}")
        if before != "ws2":
            assert smaller.route(f"k{i}") == before


def test_call_goes_to_the_routed_endpoint_with_its_model_id() -> None:
    pool, clients = _pool(3)
    route = pool.route("agent-7")
    assert cast(Any, asyncio.run(pool.create("agent-7", messages=[]))) == "m" + route[-1]
    assert sum(len(c.calls) for c in clients) == 1


def test_rate_limit_fails_over_and_cools_the_endpoint() -> None:
    clock = [0.0]
    pool, clients = _pool(2, clock)
    first = pool.route("a")
    bad = clients[int(first[-1])]
    bad.fail = _rate_limited()
    other = "ws1" if first == "ws0" else "ws0"
    assert cast(Any, asyncio.run(pool.create("a"))) == "m" + other[-1]
    assert pool.stats[first] == {"calls": 1, "errors": 1, "failovers": 1}
    # While cooling, the key goes straight to the other endpoint ...
    assert pool.route("a") == other
    asyncio.run(pool.create("a"))
    assert len(bad.calls) == 1
    # ... and comes back once the cooldown has passed (cache affinity restored).
    clock[0] = 100.0
    assert pool.route("a") == first


def test_all_endpoints_failing_raises_the_last_error() -> None:
    pool, clients = _pool(2)
    for c in clients:
        c.fail = _rate_limited()
    with pytest.raises(RateLimitError):
        asyncio.run(pool.create("a"))
    assert all(len(c.calls) == 1 for c in clients)


def test_timeouts_do_not_fail_over() -> None:
    pool, clients = _pool(2)
    for c in clients:
        c.fail = APITimeoutError(httpx.Request("POST", "https://x"))
    with pytest.raises(APITimeoutError):
        asyncio.run(pool.create("a"))
    assert sum(len(c.calls) for c in clients) == 1


def test_endpoint_headers_need_every_secret() -> None:
    ep = modal_endpoint("chryztoph", "ep-x-server", "m", secret_prefix="CHRYZTOPH_")
    assert ep.base_url == "https://chryztoph--ep-x-server.us-west.modal.direct/v1"
    env = {"CHRYZTOPH_MODAL_KEY": "k", "CHRYZTOPH_MODAL_SECRET": "s"}
    assert endpoint_headers(ep, lambda n: env.get(n, "")) == {"Modal-Key": "k", "Modal-Secret": "s"}
    env.pop("CHRYZTOPH_MODAL_SECRET")
    assert endpoint_headers(ep, lambda n: env.get(n, "")) is None
