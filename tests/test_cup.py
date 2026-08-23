import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from cup import (CumulativeUtilityParity, dependent_round_sample,
                 fixed_size_inclusion_probabilities, oracle_maxmin_rates)


def make_cup(tmp_path, monkeypatch, clients=("h0", "h1", "h2"), capacity=2):
    monkeypatch.setenv("CUP_EPSILON", "0.1")
    monkeypatch.setenv("CUP_DEBT_LAMBDA", "0.5")
    return CumulativeUtilityParity(
        clients, capacity, seed=3, output_path=tmp_path / "cup.csv")


def test_availability_selection_and_participation_are_distinct(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch)
    availability = {"h0": False, "h1": True, "h2": True}
    cup.observe_external_selection(availability, 0, ["h0", "h1"])
    rows = cup.end_round(0, availability, ["h0", "h1"],
                         {"h0": 1, "h1": 1, "h2": 1}, [])
    by_client = {row["client_id"]: row for row in rows}
    assert [by_client[f"h{i}"]["participated"] for i in range(3)] == [0, 1, 0]
    assert [cup.states[f"h{i}"].selection_count for i in range(3)] == [1, 1, 0]
    assert [cup.states[f"h{i}"].participation_count for i in range(3)] == [0, 1, 0]
    assert [cup.states[f"h{i}"].availability_count for i in range(3)] == [0, 1, 1]


def test_availability_estimate_uses_only_telemetry(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=("h0", "h1"), capacity=1)
    traces = {"h0": [1, 0, 1, 0], "h1": [1, 1, 1, 0]}
    for round_index in range(4):
        availability = {client: bool(trace[round_index]) for client, trace in traces.items()}
        cup.observe_external_selection(availability, round_index, ["h0"] if round_index == 0 else [])
        cup.end_round(round_index, availability, [], {"h0": 0, "h1": 0}, [])
    assert cup.states["h0"].availability_estimate == 0.5
    assert cup.states["h1"].availability_estimate == 0.75
    assert cup.states["h0"].participation_count == 0


def test_realized_utility_only_accumulates_when_p_is_one(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=("h0",), capacity=1)
    # Establish the previous accuracy, then gains are [0.2, 100.0, 0.3].
    accuracies = [0.0, 0.2, 100.2, 100.5]
    selections = [[], ["h0"], [], ["h0"]]
    for round_index, (accuracy, selected) in enumerate(zip(accuracies, selections)):
        availability = {"h0": True}
        cup.observe_external_selection(availability, round_index, selected)
        cup.end_round(round_index, availability, selected, {"h0": accuracy}, [])
    assert math.isclose(cup.states["h0"].utility, 0.5)


def test_reactive_score_and_probability_normalization(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=("h0", "h1"), capacity=1)
    cup.states["h0"].availability_estimate = 0.5
    cup.states["h1"].availability_estimate = 0.9
    cup.states["h0"].participation_debt = 2
    cup.states["h1"].participation_debt = 0
    scores = cup.reactive_scores()
    assert math.isclose(scores["h0"], 1 / 0.6 * (1 + 0.5 * 2))
    assert math.isclose(scores["h1"], 1 / 1.0)
    probabilities = fixed_size_inclusion_probabilities(scores, 1)
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["h0"] > probabilities["h1"]


def test_participation_debt_is_sum_of_one_minus_p(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=("h0",), capacity=1)
    for round_index, participated in enumerate([1, 0, 0, 1]):
        availability = {"h0": bool(participated)}
        selected = ["h0"]
        cup.observe_external_selection(availability, round_index, selected)
        cup.end_round(round_index, availability, selected, {"h0": round_index}, [])
    assert cup.states["h0"].participation_debt == 2


def test_oracle_maxmin_rates_equalize_conditional_utility_rate():
    pi = {"a": 0.5, "b": 1.0}
    mu = {"a": 1.0, "b": 2.0}
    tau, rates = oracle_maxmin_rates(pi, mu, budget=0.5)
    assert math.isclose(tau, 0.5)
    assert math.isclose(rates["a"] * mu["a"], rates["b"] * mu["b"])
    assert math.isclose(sum(pi[k] * rates[k] for k in pi), 0.5)


def test_fixed_size_dependent_rounding_preserves_budget():
    probabilities = fixed_size_inclusion_probabilities({"a": 8, "b": 2, "c": 1}, 2)
    import random
    selected = dependent_round_sample(probabilities, random.Random(2))
    assert len(selected) == 2
    assert math.isclose(sum(probabilities.values()), 2.0)


def test_surrogate_is_utility_only_and_never_changes_participation(tmp_path, monkeypatch):
    monkeypatch.setenv("CUP_SURROGATE", "true")
    monkeypatch.setenv("CUP_SURROGATE_DECAY", "0.5")
    cup = make_cup(tmp_path, monkeypatch, clients=("h0",), capacity=1)
    # Establish accuracy, realize a positive gain, then miss two rounds.
    scenarios = [(True, [], 0.0), (True, ["h0"], 1.0),
                 (False, ["h0"], 1.0), (False, ["h0"], 1.0)]
    weights = []
    for round_index, (available, selected, accuracy) in enumerate(scenarios):
        telemetry = {"h0": available}
        cup.observe_external_selection(telemetry, round_index, selected)
        row = cup.end_round(round_index, telemetry, selected, {"h0": accuracy}, [])[0]
        if row["surrogate_used"]:
            weights.append(row["surrogate_weight"])
            assert row["participated"] == 0
    assert weights[1] < weights[0]
    assert cup.states["h0"].utility == 1.0
    assert cup.states["h0"].surrogate_utility > 0


def test_empty_participation_round_still_logs_and_updates_debt(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=("h0", "h1"), capacity=1)
    availability = {"h0": False, "h1": False}
    selected = cup.select_clients(availability, 0)
    rows = cup.end_round(0, availability, selected, {"h0": 0, "h1": 0}, [])
    assert len(selected) == 1
    assert not any(row["participated"] for row in rows)
    assert all(state.participation_debt == 1 for state in cup.states.values())
    with (tmp_path / "cup.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_deterministic_smoke_has_selected_unavailable_clients(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=tuple(f"h{i}" for i in range(5)), capacity=2)
    trace = [
        {f"h{i}": False for i in range(5)},
        {f"h{i}": i % 2 == 0 for i in range(5)},
    ] * 5
    saw_distinct = False
    for round_index, availability in enumerate(trace):
        selected = cup.select_clients(availability, round_index)
        participated = cup.realize_participation(availability, selected)
        saw_distinct |= set(selected) != set(participated)
        cup.end_round(round_index, availability, selected,
                      {f"h{i}": float(round_index + i) for i in range(5)}, [])
    assert saw_distinct


def test_metrics_use_availability_normalized_utility_and_budget_target(tmp_path, monkeypatch):
    cup = make_cup(tmp_path, monkeypatch, clients=("h0", "h1"), capacity=1)
    cup.states["h0"].availability_estimate = 0.5
    cup.states["h1"].availability_estimate = 1.0
    cup.states["h0"].utility = 1.0
    cup.states["h1"].utility = 2.0
    cup.states["h0"].selection_count = 2
    cup.states["h1"].selection_count = 0
    metrics = cup.metrics(2)
    assert metrics["utility_jain_index"] == 1.0  # normalized utilities are [2,2]
    assert metrics["selection_gap"] == 1.0  # target mT/N is one each


def test_method_name_does_not_change_shared_availability_trace(tmp_path, monkeypatch):
    trace = [{"h0": bool(round_index % 2), "h1": bool((round_index + 1) % 3)}
             for round_index in range(5)]
    observed = {}
    for method in ("select_fair_nodes", "q_ffl", "afl", "fairfedcs",
                   "php_fl", "fedavg_random", "fedprox", "uniform_available"):
        cup = CumulativeUtilityParity(
            ("h0", "h1"), 1, seed=0,
            output_path=tmp_path / f"{method}.csv")
        for round_index, availability in enumerate(trace):
            cup.observe_external_selection(availability, round_index, [])
        observed[method] = [cup.states[client].availability_count
                            for client in ("h0", "h1")]
    assert len({tuple(counts) for counts in observed.values()}) == 1
