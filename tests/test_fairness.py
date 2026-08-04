import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from fairness import (AvailabilityEstimator, ProbabilisticInverseAvailabilityScheduler,
    RoundCoordinator, UtilityTracker, fairness_metrics, surrogate_weight, summarize_seeds)


def test_availability_window_uses_telemetry_only():
    estimator = AvailabilityEstimator(3)
    for value in [0, 1, 1, 0]: estimator.observe({"k": value})
    assert estimator.estimate("k") == 2 / 3


def test_probabilistic_scheduler_is_without_replacement_and_seeded():
    pi = {"a": .1, "b": .5, "c": 1.0}
    one = ProbabilisticInverseAvailabilityScheduler(7).select(list(pi), 3, pi)
    two = ProbabilisticInverseAvailabilityScheduler(7).select(list(pi), 3, pi)
    assert one == two and len(set(one)) == 3


def test_auc_is_sum_and_fairness_metrics_complete():
    tracker = UtilityTracker("auc")
    tracker.observe("a", accuracy=.4, loss=1); tracker.observe("a", accuracy=.6, loss=.8)
    assert tracker.cumulative("a") == 1.0
    assert set(fairness_metrics({"a": 1, "b": 2}, {"a": 1, "b": 3})) == {
        "utility_cv", "utility_jain_index", "selection_gap", "gini_coefficient",
        "worst_client_utility", "mean_utility"}


def test_participation_is_availability_times_selection_and_logs(tmp_path):
    coordinator = RoundCoordinator(["a", "b"], 2, output_dir=tmp_path)
    selected, participated = coordinator.begin_round(0, {"a": True, "b": False})
    assert set(selected) == {"a", "b"} and participated == ["a"]
    coordinator.log_round(0, {"a": True, "b": False}, selected, {"a": (.5, 1), "b": (.4, 2)})
    with (tmp_path / "participation.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["participation"] for r in rows] == ["1", "0"]


def test_decay_and_seed_summary(tmp_path):
    assert math.isclose(surrogate_weight(2, .1, 3), 2 * math.exp(-.3))
    out = tmp_path / "results_summary.csv"
    summary = summarize_seeds([{"seed": i, "accuracy": i} for i in range(5)], out)
    assert summary[0]["mean"] == 2 and out.exists()
