import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from benchmark_suite import METRICS, benchmark_matrix, produce_graphs


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
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for seed in range(5):
            writer.writerow({"dataset": "cifar10", "seed": seed, "method": "fedavg",
                "availability_model": "independent", "ablation": "no_surrogate",
                **{metric: seed + 1 for metric in METRICS}})
    graphs = produce_graphs(source, tmp_path / "graphs")
    assert len(graphs) == len(METRICS) and all(path.read_text().startswith("<svg") for path in graphs)
