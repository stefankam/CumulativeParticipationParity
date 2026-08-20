import random
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
        selected = select_clients(method, clients(), 1, BaselineState(), rng=random.Random(4))
        assert "offline" not in selected


def test_qffl_upweights_high_loss_clients():
    weights = aggregation_weights("q_ffl", ["a", "b"], losses={"a": 1.0, "b": 3.0})
    assert weights["b"] > weights["a"]
    assert sum(weights.values()) == 1.0


def test_utility_only_accrues_on_participation():
    tracker = UtilityTracker("auc")
    assert tracker.observe("a", accuracy=.8, loss=.2, participated=False) == 0
    assert tracker.cumulative("a") == 0


def test_fairness_metrics():
    assert fairness_metrics({"a": 2, "b": 2}, {"a": 1, "b": 1})["gini_coefficient"] == 0
