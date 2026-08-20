import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from experiment_suite import (BENCHMARK_FIELDS, completed_metric_rounds,
                              last_awpsp_accuracy, write_benchmark_results)


def test_last_awpsp_accuracy_uses_latest_nonempty_measurement(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Round", "AWPSP_Accuracy"])
        writer.writeheader()
        writer.writerow({"Round": 0, "AWPSP_Accuracy": "71.5"})
        writer.writerow({"Round": 1, "AWPSP_Accuracy": ""})
    assert last_awpsp_accuracy(path) == 71.5


def test_last_awpsp_accuracy_returns_none_when_no_update_completed(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("Round,AWPSP_Accuracy\n0,\n")
    assert last_awpsp_accuracy(path) is None


def test_completed_metric_rounds_counts_distinct_persisted_rounds(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("Round,AWPSP_Accuracy\n0,10\n1,11\n1,11\n")
    assert completed_metric_rounds(path) == 2


def test_metric_helpers_accept_paths_returned_by_run_case_as_strings(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("Round,AWPSP_Accuracy\n0,71.5\n")
    assert completed_metric_rounds(str(path)) == 1
    assert last_awpsp_accuracy(str(path)) == 71.5


def test_run_suite_reports_completed_rounds_in_result_schema():
    source = (Path(__file__).parents[1] / "server" / "experiment_suite.py").read_text()
    assert '"completed_rounds"' in source
    assert '"expected_rounds"' in source
    assert 'completed_rounds={completed_rounds}/{rounds_per_run}' in source
    assert 'child exited normally after persisting {completed_rounds}/{expected_rounds}' in source
    assert 'default="all"' in source
    assert 'run_suite(live_graphs=args.action == "all")' in source


def test_write_benchmark_results_converts_complete_final_metrics(tmp_path):
    final_path = tmp_path / "final.csv"
    with final_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "global_accuracy", "mean_client_accuracy", "worst_10_percent_utility",
            "Utility CV (No Surrogate)", "Jain (Utility) (No Surrogate)",
            "Sel. Gap (No Surrogate)", "runtime_seconds",
        ])
        writer.writeheader()
        writer.writerow({
            "global_accuracy": 75, "mean_client_accuracy": 70,
            "worst_10_percent_utility": 2, "Utility CV (No Surrogate)": .2,
            "Jain (Utility) (No Surrogate)": .9, "Sel. Gap (No Surrogate)": 1,
            "runtime_seconds": 12,
        })
    output = tmp_path / "benchmark.csv"
    rows = write_benchmark_results([{
        "return_code": 0, "completed_rounds": 50, "expected_rounds": 50,
        "final_metrics_path": str(final_path), "dataset": "cifar10", "seed": 0,
        "selector": "awpsp", "availability_model": "independent",
        "ablation": "no_surrogate",
    }], output)
    assert len(rows) == 1
    with output.open() as handle:
        assert csv.DictReader(handle).fieldnames == BENCHMARK_FIELDS


def test_write_benchmark_results_exposes_latest_in_progress_round(tmp_path):
    final_path = tmp_path / "final.csv"
    final_path.write_text(
        "global_accuracy,mean_client_accuracy,worst_10_percent_utility,"
        "Utility CV (No Surrogate),Jain (Utility) (No Surrogate),"
        "Sel. Gap (No Surrogate),runtime_seconds\n"
        "60,55,1.5,0.3,0.8,2,9\n"
    )
    output = tmp_path / "benchmark.csv"
    rows = write_benchmark_results([{
        "return_code": 0, "completed_rounds": 1, "expected_rounds": 50,
        "final_metrics_path": str(final_path), "dataset": "cifar10", "seed": 0,
        "selector": "awpsp", "availability_model": "independent",
        "ablation": "no_surrogate",
    }], output, include_in_progress=True)
    assert len(rows) == 1
