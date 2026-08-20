import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from benchmark_suite import METRICS, benchmark_matrix, produce_graphs
from experiment_suite import (DEFAULT_EXPERIMENT_METHODS, completed_metric_rounds,
                              last_accuracy, run_case, write_benchmark_results)


def test_default_suite_runs_the_complete_requested_comparison():
    assert set(DEFAULT_EXPERIMENT_METHODS) == {
        "select_fair_nodes", "fedavg_random", "uniform_available", "fedprox",
        "q_ffl", "php_fl", "fairfedcs", "fedfv", "afl", "round_robin",
        "least_selected", "deficit_based", "inverse_availability",
        "oracle_availability", "estimated_availability",
    }


def test_matrix_has_requested_datasets_seeds_baselines_and_ablations():
    rows = list(benchmark_matrix())
    assert {row["dataset"] for row in rows} == {"cifar10", "cifar100", "femnist"}
    assert {row["seed"] for row in rows} == set(range(5))
    assert {row["method"] for row in rows} >= {
        "fedavg_random", "uniform_available", "round_robin", "least_selected", "deficit_based",
        "inverse_availability", "oracle_availability",
        "estimated_availability", "cpp",
    }
    unavailable = list(benchmark_matrix(include_unavailable=True))
    assert {row["implementation"] for row in unavailable} == {"native"}
    assert {row["ablation"] for row in rows if row["method"] == "cpp"} == {
        "no_surrogate", "surrogate", "normalization_only", "full_method"}


def test_graphs_are_generated_only_from_supplied_results(tmp_path):
    source = tmp_path / "results.csv"
    fields = ["dataset", "seed", "method", "availability_model", "ablation", *METRICS]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in range(5):
            writer.writerow({"dataset": "cifar10", "seed": seed, "method": "fedavg_random",
                "availability_model": "independent", "ablation": "no_surrogate",
                **{metric: seed + 1 for metric in METRICS}})
    graphs = produce_graphs(source, tmp_path / "graphs")
    assert len(graphs) == len(METRICS)
    assert all(path.suffix == ".pdf" for path in graphs)
    assert all(path.read_bytes().startswith(b"%PDF-1.4") for path in graphs)
    graph = graphs[0].read_bytes()
    assert b"FedAvg-random" in graph and b"Method" in graph
    assert b"Fairness-aware FL" not in graph and b"Client scheduling" not in graph
    assert b"Bar = seed mean; whisker = sample SD" in graph
    coverage = (tmp_path / "graphs" / "coverage.txt").read_text()
    assert "Methods present" in coverage and "Methods without result rows" in coverage


def test_partial_final_metrics_populate_benchmark_and_graphs(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("round,accuracy\n0,\n")
    assert completed_metric_rounds(metrics) == 1
    assert last_accuracy(metrics) is None

    final = tmp_path / "final_metrics.csv"
    final.write_text(
        "Round,global_accuracy,mean_client_accuracy,worst_10_percent_utility,"
        "Utility CV (No Surrogate),Jain (Utility) (No Surrogate),"
        "Sel. Gap (No Surrogate),runtime_seconds\n"
        "1,,0.0,0.0,0.0,0.0,1,2.5\n")
    run = {
        "return_code": 2, "completed_rounds": 1, "expected_rounds": 50,
        "final_metrics_path": str(final), "dataset": "cifar10", "seed": 0,
        "selector": "select_fair_nodes", "availability_model": "independent",
        "ablation": "no_surrogate",
    }
    destination = tmp_path / "benchmark_results.csv"
    rows = write_benchmark_results([run], destination, include_in_progress=True)
    assert len(rows) == 1
    assert rows[0]["global_accuracy"] == ""
    graphs = produce_graphs(destination, tmp_path / "graphs")
    assert len(graphs) == len(METRICS)


def test_fair_fl_baselines_are_separate_legend_entries_and_bars(tmp_path):
    source = tmp_path / "fair.csv"
    fields = ["dataset", "seed", "method", "availability_model", "ablation", *METRICS]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, method in enumerate(("q_ffl", "php_fl", "fairfedcs", "fedfv", "afl"), 1):
            writer.writerow({
                "dataset": "cifar10", "seed": 0, "method": method,
                "availability_model": "independent", "ablation": "no_surrogate",
                **{metric: index for metric in METRICS},
            })
    graph = produce_graphs(source, tmp_path / "graphs")[0].read_bytes()
    for label in (b"q-FFL", b"PHP-FL", b"FairFedCS", b"FedFV", b"AFL"):
        assert label in graph
    assert b"Fairness-aware FL" not in graph


def test_run_case_routes_final_metrics_to_per_run_file(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        returncode = 1
        stdout = []

        def __init__(self, cmd, **kwargs):
            captured.update(kwargs["env"])

        def wait(self):
            return None

    monkeypatch.setattr("experiment_suite.RUN_DIR", tmp_path)
    monkeypatch.setattr("experiment_suite.ROOT", tmp_path)
    monkeypatch.setattr("experiment_suite.MAIN_SERVER", tmp_path / "main_server.py")
    monkeypatch.setattr("experiment_suite.subprocess.Popen", FakeProcess)
    run_case({"NUM_ROUNDS": 1}, "seed0")
    assert captured["METRICS_LOG_PATH"] == str(tmp_path / "metrics_seed0.csv")
    assert captured["FINAL_METRICS_PATH"] == str(tmp_path / "final_seed0.csv")
