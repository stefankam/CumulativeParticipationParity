"""Run every requested baseline and create comparison graphs from its CSV data."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

from baselines import ALL_BASELINES
from experiment_suite import RUN_DIR, run_suite


METHODS = ("select_fair_nodes",) + ALL_BASELINES


def create_graphs(run_dir: Path = RUN_DIR, graph_dir: Path | None = None) -> list[Path]:
    """Create accuracy and participation-fairness graphs for completed runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph_dir = graph_dir or run_dir.parent / "experiment_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    series = defaultdict(lambda: defaultdict(list))
    for path in run_dir.glob("metrics_*.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                method = row["method"]
                for metric in ("accuracy", "participation_gini", "participation_variance"):
                    if row.get(metric) not in (None, "", "None"):
                        series[method][metric].append(float(row[metric]))

    outputs = []
    for metric, title in (
        ("accuracy", "Global accuracy"),
        ("participation_gini", "Participation Gini (lower is fairer)"),
        ("participation_variance", "Participation variance (lower is fairer)"),
    ):
        fig, axis = plt.subplots(figsize=(10, 6))
        for method in METHODS:
            values = series[method][metric]
            if values:
                axis.plot(range(1, len(values) + 1), values, label=method)
        axis.set(title=title, xlabel="Round", ylabel=metric)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small", ncol=2)
        fig.tight_layout()
        output = graph_dir / f"{metric}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    os.environ["EXPERIMENT_METHODS"] = ",".join(METHODS)
    run_suite()
    for graph in create_graphs():
        print(f"[benchmark] wrote {graph}")
