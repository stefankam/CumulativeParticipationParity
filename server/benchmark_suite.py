"""Paper benchmark matrix and dependency-free graph generation.

This launcher intentionally consumes completed run CSVs; it never invents missing
measurements. Training jobs can use :func:`benchmark_matrix` to obtain the exact,
reproducible comparison grid and then call :func:`produce_graphs` on their results.
"""


from __future__ import annotations

import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path



DATASETS = ("cifar10", "cifar100", "femnist")
SEEDS = (0, 1, 2, 3, 4)
METHODS = ("fedavg", "q_ffl", "php_fl", "fairfedcs", "least_selected", "full_method")
NATIVE_METHODS = {"fedavg", "q_ffl", "least_selected", "full_method"}
AVAILABILITY_MODELS = ("independent", "trace", "minority_class_dropout")
ABLATIONS = ("no_surrogate", "surrogate", "normalization_only", "full_method")
METRICS = ("global_accuracy", "mean_client_accuracy", "worst_10_percent_utility",
           "utility_cv", "utility_jain_index", "conditional_selection_gap", "runtime_seconds")


def benchmark_matrix(datasets=DATASETS, include_unavailable=False):
    """Yield every dataset/seed/method/availability/ablation configuration."""
    for dataset, seed, method, availability, ablation in itertools.product(
            datasets, SEEDS, METHODS, AVAILABILITY_MODELS, ABLATIONS):
        # Ablations apply to our method; baselines have no surrogate variant.
        if method != "full_method" and ablation != "no_surrogate":
            continue
        if method not in NATIVE_METHODS and not include_unavailable:
            continue
        yield {"dataset": dataset, "seed": seed, "method": method,
               "availability_model": availability, "ablation": ablation,
               "implementation": "native" if method in NATIVE_METHODS else "external_required"}


def validate_results(rows):
    required = {"dataset", "seed", "method", "availability_model", "ablation", *METRICS}
    for index, row in enumerate(rows, 2):
        missing = required - set(row)
        if missing:
            raise ValueError(f"results row {index} is missing: {sorted(missing)}")


def _aggregate(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        grouped[(row["dataset"], row["method"], row["availability_model"], row["ablation"])].append(number)
    return [(key, sum(values) / len(values),
             math.sqrt(sum((x - sum(values) / len(values)) ** 2 for x in values) / max(1, len(values) - 1)))
            for key, values in sorted(grouped.items())]


def _svg_chart(title, points, path):
    """Write an accessible SVG bar chart without adding plotting dependencies."""
    width, height, margin = 1200, 620, 70
    maximum = max((mean + std for _, mean, std in points), default=1.0) or 1.0
    bar_width = max(3, (width - 2 * margin) / max(1, len(points)) * .75)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img">',
              f'<title>{title}</title>', '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20">{title}</text>',
              f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>']
    spacing = (width - 2 * margin) / max(1, len(points))
    for index, (key, mean, std) in enumerate(points):
        x = margin + index * spacing + (spacing - bar_width) / 2
        bar_height = mean / maximum * (height - 2 * margin)
        y = height - margin - bar_height
        label = "/".join(key)
        chunks.extend([f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#4472c4"/>',
                       f'<title>{label}: mean={mean:.5g}, std={std:.5g}</title>',
                       f'<text x="{x + bar_width/2:.2f}" y="{height-margin+15}" text-anchor="end" transform="rotate(-45 {x + bar_width/2:.2f},{height-margin+15})" font-size="8">{label}</text>'])
    chunks.append('</svg>')
    path.write_text("\n".join(chunks), encoding="utf-8")


def produce_graphs(results_csv, output_dir="benchmark_graphs"):
    """Create one mean±SD SVG graph per required metric from real run rows."""
    with Path(results_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_results(rows)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in METRICS:
        path = output / f"{metric}.svg"
        _svg_chart(f"{metric} (five-seed mean; SD in tooltip)", _aggregate(rows, metric), path)
        paths.append(path)
    return paths



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv")
    parser.add_argument("--output-dir", default="benchmark_graphs")
    args = parser.parse_args()
    for graph in produce_graphs(args.results_csv, args.output_dir):
        print(graph)
