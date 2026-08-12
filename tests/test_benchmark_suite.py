diff --git a/tests/test_benchmark_suite.py b/tests/test_benchmark_suite.py
index dd4d7159a52574115efb6fa2938955ae3135ae9d..b8acfae5e70f5a1ecae7b344699afec32298db11 100644
--- a/tests/test_benchmark_suite.py
+++ b/tests/test_benchmark_suite.py
@@ -1,31 +1,56 @@
 import csv
 import sys
 from pathlib import Path
 
 sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
 from benchmark_suite import METRICS, benchmark_matrix, produce_graphs
+from experiment_suite import completed_metric_rounds, last_accuracy, write_benchmark_results
 
 
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
+
+
+def test_first_lowercase_round_populates_benchmark_artifacts(tmp_path):
+    metrics = tmp_path / "metrics.csv"
+    metrics.write_text("round,accuracy\n0,0.75\n")
+    assert completed_metric_rounds(metrics) == 1
+    assert last_accuracy(metrics) == 0.75
+
+    final = tmp_path / "final_metrics.csv"
+    final.write_text(
+        "Round,global_accuracy,mean_client_accuracy,worst_10_percent_utility,"
+        "Utility CV (No Surrogate),Jain (Utility) (No Surrogate),"
+        "Sel. Gap (No Surrogate),runtime_seconds\n"
+        "1,0.75,0.7,0.2,0.3,0.8,1,2.5\n")
+    run = {
+        "return_code": 0, "completed_rounds": 1, "expected_rounds": 50,
+        "final_metrics_path": str(final), "dataset": "cifar10", "seed": 0,
+        "selector": "select_fair_nodes", "availability_model": "independent",
+        "ablation": "no_surrogate",
+    }
+    destination = tmp_path / "benchmark_results.csv"
+    rows = write_benchmark_results([run], destination, include_in_progress=True)
+    assert len(rows) == 1
+    assert list(csv.DictReader(destination.open()))[0]["global_accuracy"] == "0.75"
