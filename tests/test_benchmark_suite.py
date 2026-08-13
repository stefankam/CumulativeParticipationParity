import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from benchmark_suite import METRICS, benchmark_matrix, produce_graphs
from experiment_suite import completed_metric_rounds, last_accuracy, run_case, write_benchmark_results


def test_matrix_has_requested_datasets_seeds_baselines_and_ablations():
    rows = list(benchmark_matrix())
    assert {row["dataset"] for row in rows} == {"cifar10", "cifar100", "femnist"}
    assert {row["seed"] for row in rows} == set(range(5))
    assert {row["method"] for row in rows} >= {"fedavg", "q_ffl", "least_selected"}
    unavailable = list(benchmark_matrix(include_unavailable=True))
    assert {row["method"] for row in unavailable if row["implementation"] == "external_required"} == {
        "php_fl", "fairfedcs"}
    assert {row["ablation"] for row in rows if row["method"] == "full_method"} == {
        "no_surrogate", "surrogate", "normalization_only", "full_method"}


def test_graphs_are_generated_only_from_supplied_results(tmp_path):
    source = tmp_path / "results.csv"
    fields = ["dataset", "seed", "method", "availability_model", "ablation", *METRICS]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in range(5):
            writer.writerow({"dataset": "cifar10", "seed": seed, "method": "fedavg",
                "availability_model": "independent", "ablation": "no_surrogate",
                **{metric: seed + 1 for metric in METRICS}})
    graphs = produce_graphs(source, tmp_path / "graphs")
    assert len(graphs) == len(METRICS)
    assert all(path.read_text().startswith("<svg") for path in graphs)


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
