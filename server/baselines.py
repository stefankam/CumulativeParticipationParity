"""Baseline registry for fair federated-learning experiments.

Selection and aggregation are deliberately separate: scheduling baselines choose
participants, while fair-FL baselines change the aggregation weights.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class BaselineClient:
    """Telemetry snapshot consumed by non-CPP benchmark schedulers."""
    client_id: str
    availability: float = 1.0
    estimated_availability: float = 1.0
    selections: int = 0
    labels: tuple[int, ...] = ()


STANDARD = ("fedavg_random", "uniform_available", "fedprox")
FAIR_FL = ("q_ffl", "php_fl", "fairfedcs", "fedfv", "afl")
SCHEDULING = (
    "round_robin", "least_selected", "deficit_based", "inverse_availability",
    "oracle_availability", "estimated_availability",
)
ALL_BASELINES = STANDARD + FAIR_FL + SCHEDULING


@dataclass
class BaselineState:
    selections: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cursor: int = 0


def select_clients(
    name: str,
    clients: Sequence[BaselineClient],
    count: int,
    state: BaselineState,
    *,
    rng: random.Random,
) -> list[str]:
    """Select clients according to a named, availability-respecting baseline."""
    if name not in ALL_BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(ALL_BASELINES)}")
    available = [client for client in clients if client.availability > 0]
    count = min(max(count, 0), len(available))
    if not count:
        return []

    # FL optimization baselines use conventional uniform-among-available sampling;
    # their distinguishing behavior is implemented by aggregation_weights below.
    if name in STANDARD + FAIR_FL and name != "fedavg_random":
        chosen = rng.sample(available, count)
    elif name == "fedavg_random":
        chosen = rng.sample(list(clients), min(count, len(clients)))
        chosen = [client for client in chosen if client.availability > 0]
    elif name == "round_robin":
        ordered = sorted(available, key=lambda client: client.client_id)
        chosen = [ordered[(state.cursor + i) % len(ordered)] for i in range(count)]
        state.cursor = (state.cursor + count) % len(ordered)
    elif name == "least_selected":
        chosen = sorted(available, key=lambda client: (state.selections[client.client_id], client.client_id))[:count]
    elif name == "deficit_based":
        rounds = max(1, sum(state.selections.values()) // max(1, count) + 1)
        chosen = sorted(available, key=lambda c: (state.selections[c.client_id] - rounds, c.client_id))[:count]
    else:
        availability = {
            "inverse_availability": lambda c: c.estimated_availability,
            "oracle_availability": lambda c: c.availability,
            "estimated_availability": lambda c: c.estimated_availability,
        }[name]
        reverse = name == "estimated_availability"
        chosen = sorted(available, key=lambda c: (availability(c), c.client_id), reverse=reverse)[:count]

    ids = [client.client_id for client in chosen]
    for client_id in ids:
        state.selections[client_id] += 1
    return ids


def aggregation_weights(
    name: str,
    client_ids: Sequence[str],
    *,
    sample_counts: Mapping[str, int] | None = None,
    losses: Mapping[str, float] | None = None,
    q: float = 1.0,
) -> dict[str, float]:
    """Return normalized server aggregation weights for a baseline.

    FedProx's proximal local objective must be applied by the client; its server
    aggregation remains sample-weighted FedAvg. FedFV additionally requires its
    gradient-conflict projection before applying these uniform coefficients.
    """
    if not client_ids:
        return {}
    sample_counts = sample_counts or {}
    losses = losses or {}
    raw: dict[str, float] = {}
    for client_id in client_ids:
        samples = max(1, sample_counts.get(client_id, 1))
        loss = max(1e-12, losses.get(client_id, 1.0))
        if name == "q_ffl":
            raw[client_id] = samples * loss ** q
        elif name in ("php_fl", "fairfedcs", "afl"):
            raw[client_id] = samples * loss
        elif name == "fedfv":
            raw[client_id] = 1.0
        else:
            raw[client_id] = float(samples)
    total = sum(raw.values())
    return {client_id: value / total for client_id, value in raw.items()}
