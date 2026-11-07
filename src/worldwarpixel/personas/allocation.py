"""Largest-remainder (Hamilton) apportionment of agent seats to countries."""

from __future__ import annotations

import math
from collections.abc import Mapping


def allocate_seats(weights: Mapping[str, float], seats: int) -> dict[str, int]:
    """Split ``seats`` across keys in proportion to ``weights``.

    Each key first gets the floor of its exact quota; the leftover seats go to the keys with the
    largest fractional remainders (ties: larger weight, then key order). Keys with zero seats are
    omitted. Deterministic, so the same population table always yields the same country mix.
    """
    if seats < 0:
        raise ValueError("seats must be non-negative")
    if seats == 0:
        return {}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    quotas = {k: w * seats / total for k, w in weights.items() if w > 0}
    floors = {k: math.floor(q) for k, q in quotas.items()}
    leftover = seats - sum(floors.values())
    by_remainder = sorted(quotas, key=lambda k: (-(quotas[k] - floors[k]), -weights[k], str(k)))
    for k in by_remainder[:leftover]:
        floors[k] += 1
    return {k: n for k, n in floors.items() if n > 0}
