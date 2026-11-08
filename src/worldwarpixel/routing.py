"""Spread one model's calls over several endpoints that serve it.

Each caller passes a sticky ``key`` (one per agent, one per narrator). New keys go to the
least-loaded endpoint, with rendezvous hashing as a deterministic tie-breaker, so agents keep
their server's KV cache affinity while spreading evenly. Rate-limited endpoints pause their
assigned callers; endpoints that are down fail over and cool down for a while.
"""

import asyncio
import hashlib
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One OpenAI-compatible server for a model, plus the secrets needed to call it."""

    name: str
    """Short unique label (usually the Modal workspace), shown in stats."""
    base_url: str
    model: str
    """Model id this endpoint serves."""
    auth: Literal["modal", "bearer"]
    secrets: tuple[str, ...]
    """Env var names: modal = (key, secret) sent as Modal-Key/Modal-Secret; bearer = (api key,)."""


def describe_endpoint(e: BaseException) -> str:
    """'name (base_url)' of the endpoint that raised e, or 'unknown endpoint'."""
    endpoint = getattr(e, "endpoint", None)
    if not isinstance(endpoint, Endpoint):
        return "unknown endpoint"
    return f"{endpoint.name} ({endpoint.base_url})" if endpoint.base_url else endpoint.name


def modal_endpoint(workspace: str, server: str, model: str, *, secret_prefix: str) -> Endpoint:
    """A Modal ``<workspace>--<server>.us-west.modal.direct`` endpoint with proxy auth from
    ``<secret_prefix>MODAL_KEY`` / ``<secret_prefix>MODAL_SECRET``."""
    return Endpoint(
        name=workspace,
        base_url=f"https://{workspace}--{server}.us-west.modal.direct/v1",
        model=model,
        auth="modal",
        secrets=(f"{secret_prefix}MODAL_KEY", f"{secret_prefix}MODAL_SECRET"),
    )


def endpoint_headers(endpoint: Endpoint, secret: Callable[[str], str]) -> dict[str, str] | None:
    """Auth headers for a modal endpoint, or None when its secrets are missing."""
    values = [secret(name) for name in endpoint.secrets]
    if not all(values):
        return None
    if endpoint.auth == "modal":
        return {"Modal-Key": values[0], "Modal-Secret": values[1]}
    return {}


def endpoint_client(
    endpoint: Endpoint, secret: Callable[[str], str], timeout: float
) -> AsyncOpenAI:
    if endpoint.auth == "modal":
        return AsyncOpenAI(
            api_key="none",
            base_url=endpoint.base_url,
            timeout=timeout,
            default_headers=endpoint_headers(endpoint, secret) or None,
            max_retries=0,
        )
    return AsyncOpenAI(
        api_key=secret(endpoint.secrets[0]),
        base_url=endpoint.base_url,
        timeout=timeout,
        max_retries=0,
    )


def _score(key: str, endpoint: str) -> int:
    digest = hashlib.blake2b(f"{key}\0{endpoint}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError)
"""Errors worth trying another endpoint for (429, 5xx such as Modal's cold 503, unreachable)."""

MAX_RATE_LIMIT_RETRIES = 3


def retry_after(e: RateLimitError, default: float) -> float:
    """Seconds to wait for a 429, bounded to a useful range."""
    headers = getattr(getattr(e, "response", None), "headers", {})
    wait: float | None = None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            value = float(retry_after_ms) / 1000
            if math.isfinite(value):
                wait = value
        except (TypeError, ValueError):
            pass

    retry_after_seconds = headers.get("retry-after")
    if wait is None and retry_after_seconds is not None:
        try:
            value = float(retry_after_seconds)
            if math.isfinite(value):
                wait = value
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(retry_after_seconds)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                value = (retry_at - datetime.now(UTC)).total_seconds()
                if math.isfinite(value):
                    wait = value
            except (TypeError, ValueError, OverflowError):
                pass

    if wait is None:
        wait = default if math.isfinite(default) else 1.0
    return min(300.0, max(1.0, wait))


class EndpointPool:
    """Routes chat completions for one model across its endpoints (see module docstring)."""

    def __init__(
        self,
        endpoints: Sequence[tuple[Endpoint, Any]],
        cooldown: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not endpoints:
            raise ValueError("an endpoint pool needs at least one endpoint")
        self._endpoints = list(endpoints)
        self._cooldown = cooldown
        self._clock = clock
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._cooling_until: dict[str, float] = {}
        self._limited_until: dict[str, float] = {}
        self._waiting: dict[str, int] = {}
        self._assigned: dict[str, str] = {}
        self._assigned_counts: dict[str, int] = {e.name: 0 for e, _ in self._endpoints}
        self.stats: dict[str, dict[str, int]] = {
            e.name: {"calls": 0, "errors": 0, "failovers": 0, "rate_limited": 0}
            for e, _ in self._endpoints
        }

    @classmethod
    def single(cls, client: Any, model: str) -> "EndpointPool":
        """Wrap one ready client (tests, or a caller that does its own routing)."""
        endpoint = Endpoint(name="default", base_url="", model=model, auth="bearer", secrets=())
        return cls([(endpoint, client)])

    @property
    def model(self) -> str:
        return self._endpoints[0][0].model

    def _assign(self, key: str) -> str:
        assigned = self._assigned.get(key)
        if assigned is None:
            least_loaded = min(self._assigned_counts.values())
            assigned = max(
                (name for name, count in self._assigned_counts.items() if count == least_loaded),
                key=lambda name: _score(key, name),
            )
            self._assigned[key] = assigned
            self._assigned_counts[assigned] += 1
        return assigned

    def _ranked(self, key: str, primary: str) -> list[tuple[Endpoint, Any]]:
        """Assigned endpoint first, then least-loaded endpoints with deterministic tie-breaks."""
        first = next(ec for ec in self._endpoints if ec[0].name == primary)
        rest = [ec for ec in self._endpoints if ec[0].name != primary]
        rest.sort(key=lambda ec: (self._assigned_counts[ec[0].name], -_score(key, ec[0].name)))
        return [first, *rest]

    def order(self, key: str) -> list[tuple[Endpoint, Any]]:
        """Assign a sticky endpoint if needed, then rank the pool for this key."""
        return self._ranked(key, self._assign(key))

    def release(self, prefix: str) -> None:
        """Forget all sticky keys beginning with ``prefix``."""
        for key, name in list(self._assigned.items()):
            if key.startswith(prefix):
                del self._assigned[key]
                self._assigned_counts[name] -= 1

    def route(self, key: str) -> str:
        """Name of the endpoint the key's next call will try first."""
        return self._candidates(key, set(), self._assign(key))[0][0].name

    def _candidates(
        self, key: str, excluded: set[str] | None, primary_name: str
    ) -> list[tuple[Endpoint, Any]]:
        ranked = self._ranked(key, primary_name)
        excluded = excluded if excluded is not None else set()
        now = self._clock()
        primary = ranked[0]
        others = [
            ec
            for ec in ranked[1:]
            if ec[0].name not in excluded and self._limited_until.get(ec[0].name, 0.0) <= now
        ]
        if primary[0].name not in excluded and self._limited_until.get(primary[0].name, 0.0) > now:
            healthy = [ec for ec in others if self._cooling_until.get(ec[0].name, 0.0) <= now]
            cooling = [ec for ec in others if ec not in healthy]
            return [primary, *healthy, *cooling]

        eligible = [
            ec
            for ec in ranked
            if ec[0].name not in excluded and self._limited_until.get(ec[0].name, 0.0) <= now
        ]
        healthy = [ec for ec in eligible if self._cooling_until.get(ec[0].name, 0.0) <= now]
        # Cooling endpoints stay as a last resort rather than failing the call outright.
        return healthy + [ec for ec in eligible if ec not in healthy]

    async def _wait_for_limit(self, endpoint: str) -> None:
        while (remaining := self._limited_until.get(endpoint, 0.0) - self._clock()) > 0:
            self._waiting[endpoint] = self._waiting.get(endpoint, 0) + 1
            try:
                await self._sleep(remaining + self._rng.uniform(0, 1))
            finally:
                self._waiting[endpoint] -= 1

    async def create(self, key: str, **kwargs: Any) -> ChatCompletion:
        """Call the assigned endpoint; pause on 429 and fail over on 5xx/connect."""
        primary = self._assign(key)
        attempted: set[str] = set()
        last_error: Exception | None = None
        rate_limit_retries = 0

        while True:
            candidates = self._candidates(key, attempted, primary)
            if not candidates:
                if last_error is not None:
                    raise last_error
                raise AssertionError("no endpoint available")
            endpoint, client = candidates[0]
            if (
                endpoint.name != primary
                and self._limited_until.get(endpoint.name, 0.0) > self._clock()
            ):
                attempted.add(endpoint.name)
                continue

            stats = self.stats[endpoint.name]
            while True:
                await self._wait_for_limit(endpoint.name)
                stats["calls"] += 1
                try:
                    response: ChatCompletion = await client.chat.completions.create(
                        model=endpoint.model, **kwargs
                    )
                    return response
                except Exception as e:
                    stats["errors"] += 1
                    cast(Any, e).endpoint = endpoint
                    if isinstance(e, RateLimitError):
                        stats["rate_limited"] += 1
                        rate_limit_retries += 1
                        now = self._clock()
                        retry = retry_after(e, self._cooldown)
                        self._limited_until[endpoint.name] = max(
                            self._limited_until.get(endpoint.name, 0.0), now + retry
                        )
                        wait = self._limited_until[endpoint.name] - now
                        log.warning(
                            "endpoint %s rate limited; retrying in %.1fs",
                            endpoint.name,
                            wait,
                        )
                        if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                            raise
                        await self._wait_for_limit(endpoint.name)
                        continue
                    if not isinstance(e, RETRYABLE):
                        raise
                    if isinstance(e, APITimeoutError):
                        raise  # slow, not down: retrying elsewhere would only double the wait
                    self._cooling_until[endpoint.name] = self._clock() + self._cooldown
                    attempted.add(endpoint.name)
                    last_error = e
                    next_candidates = self._candidates(key, attempted, primary)
                    if not next_candidates:
                        raise
                    next_endpoint = next_candidates[0][0]
                    log.warning(
                        "endpoint %s failed (%s: %s); failing over to %s",
                        endpoint.name,
                        type(e).__name__,
                        e,
                        next_endpoint.name,
                    )
                    stats["failovers"] += 1
                    break

    def public_stats(self) -> dict[str, Any]:
        now = self._clock()
        return {
            name: {
                **counts,
                "cooling": self._cooling_until.get(name, 0.0) > now,
                "retry_in": round(max(0.0, self._limited_until.get(name, 0.0) - now), 1),
                "waiting": self._waiting.get(name, 0),
                "agents": self._assigned_counts[name],
            }
            for name, counts in self.stats.items()
        }


def as_pool(client: Any, model: str) -> EndpointPool:
    return client if isinstance(client, EndpointPool) else EndpointPool.single(client, model)
