import random
import pytest
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "server"))

from baselines import ALL_BASELINES, BaselineClient, BaselineState, aggregation_weights, select_clients
from fairness import (AvailabilityEstimator, CumulativeUtilityParityScheduler,
                      UtilityTracker, cumulative_utility_parity_rates,
                      fairness_metrics)


def test_annotations_are_postponed_for_python_38_compatibility():
    assert BaselineClient.__annotations__["labels"] == "tuple[int, ...]"
    assert select_clients.__annotations__["return"] == "list[str]"


def clients():
    return [
        BaselineClient("rare", 1.0, 0.1, 0, (0,)),
        BaselineClient("common", 1.0, 0.9, 0, (1,)),
        BaselineClient("offline", 0.0, 0.1, 0, (2,)),
    ]


def test_availability_estimator_uses_telemetry_only():
    estimator = AvailabilityEstimator(window_size=2)
    estimator.observe({"rare": True, "common": True})
    estimator.observe({"rare": False, "common": True})
    assert estimator.estimate("rare") == 0.5
    assert estimator.estimate("common") == 1.0


def test_cpp_never_selects_offline_clients():
    scheduler = CumulativeUtilityParityScheduler(seed=0)
    selected = scheduler.select(["rare", "common", "offline"], 3,
        availability={"rare": True, "common": True, "offline": False},
        pi_hat={"rare": .5, "common": 1.0, "offline": .5},
        mu_hat={"rare": 1.0, "common": 1.0, "offline": 1.0})
    assert "offline" not in selected


def test_cup_rates_satisfy_expected_budget():
    tau, rates = cumulative_utility_parity_rates({"a": .5, "b": 1.0}, {"a": 1.0, "b": 2.0}, 1.0)
    assert tau > 0
    assert abs(sum(pi * rates[k] for k, pi in {"a": .5, "b": 1.0}.items()) - 1.0) < 1e-12


def test_every_baseline_is_runnable_and_availability_safe():
    for method in ALL_BASELINES:
        if method in {"fairfedcs", "fedavg_random"}:
            continue
        selected = select_clients(method, clients(), 1, BaselineState(), rng=random.Random(4))
        assert "offline" not in selected


def test_fedavg_random_records_scheduler_intent_before_availability_gate():
    selected = select_clients(
        "fedavg_random", [BaselineClient("offline", 0.0)], 1,
        BaselineState(), rng=random.Random(4))
    assert selected == ["offline"]


def test_stateful_methods_cannot_use_scalar_aggregation_weights():
    for method in ("q_ffl", "afl", "php_fl"):
        with pytest.raises(ValueError, match="dedicated stateful server update"):
            aggregation_weights(method, ["a", "b"])


def test_utility_only_accrues_on_participation():
    tracker = UtilityTracker("auc")
    assert tracker.observe("a", accuracy=.8, loss=.2, participated=False) == 0
    assert tracker.cumulative("a") == 0


def test_fairness_metrics():
    assert fairness_metrics({"a": 2, "b": 2}, {"a": 1, "b": 1})["gini_coefficient"] == 0
