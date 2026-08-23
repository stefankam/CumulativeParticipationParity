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
# Only these policies have an end-to-end implementation in main_server.py.
# The optimization methods below have registry/formula scaffolding only and
# must not be presented as completed experimental baselines.
RUNNABLE_BASELINES = ALL_BASELINES
UNIMPLEMENTED_BASELINES = {}



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
    count = min(max(count, 0),
                len(clients) if name == "fedavg_random" else len(available))

    if not count:
        return []

    # FL optimization baselines use conventional uniform-among-available sampling;
    # their distinguishing behavior is implemented by aggregation_weights below.
    if name in STANDARD + FAIR_FL and name not in {"fedavg_random", "fairfedcs", "php_fl"}:
        chosen = rng.sample(available, count)
    elif name == "fedavg_random":
        chosen = rng.sample(list(clients), min(count, len(clients)))
    elif name == "round_robin":
        ordered = sorted(available, key=lambda client: client.client_id)
        chosen = [ordered[(state.cursor + i) % len(ordered)] for i in range(count)]
        state.cursor = (state.cursor + count) % len(ordered)
    elif name == "least_selected":
        chosen = sorted(available, key=lambda client: (state.selections[client.client_id], client.client_id))[:count]
    elif name == "deficit_based":
        rounds = max(1, sum(state.selections.values()) // max(1, count) + 1)
        chosen = sorted(available, key=lambda c: (state.selections[c.client_id] - rounds, c.client_id))[:count]
    elif name == "php_fl":
        # PHP-FL's defining DEAL/ISPU operations are client/server model
        # updates, not a participation-debt selector.
        chosen = rng.sample(available, count)
    elif name == "fairfedcs":
        raise RuntimeError("FairFedCS requires its persistent FairFedCSState selector")
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
    """Return ordinary aggregation weights only.

    FedProx's proximal local objective must be applied by the client; its server
    aggregation remains sample-weighted FedAvg. FedFV additionally requires its
    gradient-conflict projection before applying these uniform coefficients.
    """
    if not client_ids:
        return {}
    if name in {"q_ffl", "afl", "php_fl"}:
        raise ValueError(f"{name} requires its dedicated stateful server update")
    sample_counts = sample_counts or {}
    losses = losses or {}
    raw: dict[str, float] = {}
    for client_id in client_ids:
        samples = max(1, sample_counts.get(client_id, 1))
        loss = max(1e-12, losses.get(client_id, 1.0))
        if name == "fedfv":
            raw[client_id] = 1.0
        else:
            raw[client_id] = float(samples)
    total = sum(raw.values())
    return {client_id: value / total for client_id, value in raw.items()}


# Aggregation compatibility API used by the paper benchmark tests and callers.
@dataclass(frozen=True)
class BaselineSource:
    name: str
    code_url: str | None
    paper_url: str
    native: bool


BASELINE_SOURCES = {
    "q_ffl": BaselineSource("q-FFL", "https://github.com/litian96/fair_flearn",
                             "https://arxiv.org/abs/1905.10497", True),
    "php_fl": BaselineSource("PHP-FL", "https://github.com/Siyuan01/PHP-FL-main",
                              "https://openreview.net/forum?id=pJWozQn9p4", False),
    "fairfedcs": BaselineSource("FairFedCS", None,
                                 "https://arxiv.org/abs/2307.10738", False),
}


def _map2(left, right, operation):
    if isinstance(left, Mapping):
        return {key: _map2(left[key], right[key], operation) for key in left}
    return operation(left, right)


def _scale(value, factor):
    if isinstance(value, Mapping):
        return {key: _scale(item, factor) for key, item in value.items()}
    return value * factor


def _squared_norm(value):
    if isinstance(value, Mapping):
        return sum(_squared_norm(item) for item in value.values())
    squared = value * value
    return float(squared.sum().item()) if hasattr(squared, "sum") else float(squared)


def fedavg(global_state, client_states, sample_counts):
    if len(client_states) != len(sample_counts) or not client_states:
        raise ValueError("client_states and sample_counts must be non-empty and aligned")
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("FedAvg sample counts must sum to a positive value")
    result = _scale(client_states[0], sample_counts[0] / total)
    for state, count in zip(client_states[1:], sample_counts[1:]):
        result = _map2(result, _scale(state, count / total), lambda a, b: a + b)
    return result


def qfedavg(global_state, client_states, losses, learning_rate, q):
    if not client_states or len(client_states) != len(losses):
        raise ValueError("client_states and losses must be non-empty and aligned")
    if learning_rate <= 0 or q < 0 or any(loss <= 0 for loss in losses):
        raise ValueError("learning_rate and losses must be positive and q non-negative")
    deltas, hs = [], []
    for state, loss in zip(client_states, losses):
        model_delta = _map2(global_state, state, lambda a, b: a - b)
        loss_q = float(loss) ** q
        deltas.append(_scale(model_delta, loss_q))
        hs.append(q * float(loss) ** (q - 1) * _squared_norm(model_delta) + loss_q / learning_rate)
    summed = deltas[0]
    for delta in deltas[1:]:
        summed = _map2(summed, delta, lambda a, b: a + b)
    return _map2(global_state, _scale(summed, 1.0 / sum(hs)), lambda a, b: a - b)


class BaselineUnavailableError(NotImplementedError):
    pass


def require_reference_baseline(name):
    normalized = name.lower()
    if normalized in {"fedavg", "q_ffl", "least_selected"}:
        return normalized
    if normalized in {"php_fl", "fairfedcs"}:
        source = BASELINE_SOURCES[normalized]
        raise BaselineUnavailableError(
            f"{name} has no verified reference implementation. Paper: {source.paper_url}; "
            f"code: {source.code_url or 'not supplied'}")
    raise ValueError(f"unknown baseline: {name}")
