"""Spread one model's calls over several endpoints that serve it.

Each caller passes a sticky ``key`` (one per agent, one per narrator). Keys are mapped to
endpoints with rendezvous hashing, so an agent keeps talking to the same server and its
growing prompt prefix stays in that server's KV cache; adding or losing an endpoint only
moves the keys that hashed to it. When the chosen endpoint is rate-limited or down, the
call fails over to the key's next endpoint and the bad one cools down for a while.
"""

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion


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
        )
    return AsyncOpenAI(
        api_key=secret(endpoint.secrets[0]), base_url=endpoint.base_url, timeout=timeout
    )


def _score(key: str, endpoint: str) -> int:
    digest = hashlib.blake2b(f"{key}\0{endpoint}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError)
"""Errors worth trying another endpoint for (429, 5xx such as Modal's cold 503, unreachable)."""


class EndpointPool:
    """Routes chat completions for one model across its endpoints (see module docstring)."""

    def __init__(
        self,
        endpoints: Sequence[tuple[Endpoint, Any]],
        cooldown: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoints:
            raise ValueError("an endpoint pool needs at least one endpoint")
        self._endpoints = list(endpoints)
        self._cooldown = cooldown
        self._clock = clock
        self._cooling_until: dict[str, float] = {}
        self.stats: dict[str, dict[str, int]] = {
            e.name: {"calls": 0, "errors": 0, "failovers": 0} for e, _ in self._endpoints
        }

    @classmethod
    def single(cls, client: Any, model: str) -> "EndpointPool":
        """Wrap one ready client (tests, or a caller that does its own routing)."""
        endpoint = Endpoint(name="default", base_url="", model=model, auth="bearer", secrets=())
        return cls([(endpoint, client)])

    @property
    def model(self) -> str:
        return self._endpoints[0][0].model

    def order(self, key: str) -> list[tuple[Endpoint, Any]]:
        """This key's endpoints, most preferred first (stable for a given endpoint set)."""
        return sorted(self._endpoints, key=lambda ec: _score(key, ec[0].name), reverse=True)

    def route(self, key: str) -> str:
        """Name of the endpoint the key's next call will try first."""
        return self._candidates(key)[0][0].name

    def _candidates(self, key: str) -> list[tuple[Endpoint, Any]]:
        ranked = self.order(key)
        now = self._clock()
        healthy = [ec for ec in ranked if self._cooling_until.get(ec[0].name, 0.0) <= now]
        # Cooling endpoints stay as a last resort rather than failing the call outright.
        return healthy + [ec for ec in ranked if ec not in healthy]

    async def create(self, key: str, **kwargs: Any) -> ChatCompletion:
        """``chat.completions.create`` on the key's endpoint, failing over on 429/5xx/connect."""
        candidates = self._candidates(key)
        for i, (endpoint, client) in enumerate(candidates):
            stats = self.stats[endpoint.name]
            stats["calls"] += 1
            try:
                response: ChatCompletion = await client.chat.completions.create(
                    model=endpoint.model, **kwargs
                )
                return response
            except RETRYABLE as e:
                stats["errors"] += 1
                if isinstance(e, APITimeoutError):
                    raise  # slow, not down: retrying elsewhere would only double the wait
                self._cooling_until[endpoint.name] = self._clock() + self._cooldown
                if i == len(candidates) - 1:
                    raise
                stats["failovers"] += 1
        raise AssertionError("unreachable")

    def public_stats(self) -> dict[str, Any]:
        now = self._clock()
        return {
            name: {**counts, "cooling": self._cooling_until.get(name, 0.0) > now}
            for name, counts in self.stats.items()
        }


def as_pool(client: Any, model: str) -> EndpointPool:
    return client if isinstance(client, EndpointPool) else EndpointPool.single(client, model)
