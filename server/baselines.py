"""Complete native baseline update rules.

Only algorithms implemented end-to-end belong here.  In particular q-FFL is an
objective/aggregation algorithm, not a loss-ranked client scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSource:
    name: str
    code_url: str | None
    paper_url: str
    native: bool


BASELINE_SOURCES = {
    "q_ffl": BaselineSource(
        "q-FFL", "https://github.com/litian96/fair_flearn",
        "https://arxiv.org/abs/1905.10497", True),
    "php_fl": BaselineSource(
        "PHP-FL", "https://github.com/Siyuan01/PHP-FL-main",
        "https://openreview.net/forum?id=pJWozQn9p4", False),
    "fairfedcs": BaselineSource(
        "FairFedCS", None, "https://arxiv.org/abs/2307.10738", False),
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
    # Works for scalars, NumPy arrays, and torch tensors.
    squared = value * value
    return float(squared.sum().item()) if hasattr(squared, "sum") else float(squared)


def fedavg(global_state, client_states, sample_counts):
    """Canonical sample-count-weighted FedAvg model average."""
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
    """q-FedAvg server update from the q-FFL algorithm.

    For each client, Delta_k=L_k^q(w-w_k) and
    h_k=q L_k^(q-1)||w-w_k||^2 + L_k^q/eta.  The returned state is
    w-sum(Delta_k)/sum(h_k).
    """
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
    update = _scale(summed, 1.0 / sum(hs))
    return _map2(global_state, update, lambda a, b: a - b)


class BaselineUnavailableError(NotImplementedError):
    """Raised rather than silently substituting an invented baseline."""


def require_reference_baseline(name):
    """Reject methods for which no verified end-to-end implementation is vendored."""
    normalized = name.lower()
    if normalized in {"fedavg", "q_ffl", "least_selected"}:
        return normalized
    if normalized in {"php_fl", "fairfedcs"}:
        source = BASELINE_SOURCES[normalized]
        raise BaselineUnavailableError(
            f"{name} has no verified reference implementation in this repository; "
            f"refusing to run the former heuristic placeholder. Paper: {source.paper_url}; "
            f"code: {source.code_url or 'not supplied'}")
    raise ValueError(f"unknown baseline: {name}")
