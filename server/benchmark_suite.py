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
STANDARD_METHODS = ("fedavg_random", "uniform_available", "fedprox")
FAIR_FL_METHODS = ("q_ffl", "php_fl", "fairfedcs", "fedfv", "afl")
SCHEDULING_METHODS = (
    "round_robin", "least_selected", "deficit_based", "inverse_availability",
    "oracle_availability", "estimated_availability",
)
METHODS = STANDARD_METHODS + FAIR_FL_METHODS + SCHEDULING_METHODS + ("cpp",)
# PHP-FL and FairFedCS require their authors' external implementations.  They
# remain in the published matrix but are never silently replaced by another
# algorithm in native benchmark runs.
NATIVE_METHODS = set(METHODS)
AVAILABILITY_MODELS = ("independent", "trace", "minority_class_dropout")
ABLATIONS = ("no_surrogate", "surrogate", "normalization_only", "full_method")
METRICS = ("global_accuracy", "mean_client_accuracy", "worst_10_percent_utility",
           "utility_cv", "utility_jain_index", "conditional_selection_gap", "runtime_seconds")


def benchmark_matrix(datasets=DATASETS, include_unavailable=False):
    """Yield every dataset/seed/method/availability/ablation configuration."""
    for dataset, seed, method, availability, ablation in itertools.product(
            datasets, SEEDS, METHODS, AVAILABILITY_MODELS, ABLATIONS):
        # Ablations apply to our method; baselines have no surrogate variant.
        if method != "cpp" and ablation != "no_surrogate":
            continue
        if method not in NATIVE_METHODS and not include_unavailable:
            continue
        yield {"dataset": dataset, "seed": seed, "method": method,
               "availability_model": availability, "ablation": ablation,
               "implementation": "native"}

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


def _pdf_escape(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_chart(title, points, path):
    """Write a vector PDF chart with publication-sized labels and legend."""
    width = max(900, 58 * len(points))
    height, left, bottom, top = 650, 85, 155, 90
    maximum = max((mean + std for _, mean, std in points), default=1.0) or 1.0
    plot_height, plot_width = height - bottom - top, width - left - 30
    spacing = plot_width / max(1, len(points))
    bar_width = max(3, spacing * .72)
    commands = ["1 1 1 rg 0 0 %s %s re f" % (width, height),
                "0 0 0 RG 1.5 w %s %s m %s %s l S" % (left, bottom, width - 30, bottom),
                "BT /F2 20 Tf %s %s Td (%s) Tj ET" % (left, height - 35, _pdf_escape(title))]
    pdf_colors = {"standard": (.27, .45, .77), "fair_fl": (.93, .49, .19),
                  "scheduling": (.44, .68, .28), "cpp": (.44, .19, .63)}
    legend = (("Standard", "standard"), ("Fair FL", "fair_fl"),
              ("Scheduling", "scheduling"), ("CPP", "cpp"))
    for legend_index, (legend_label, category) in enumerate(legend):
        legend_x = left + legend_index * 155
        r, g, b = pdf_colors[category]
        commands += [f"{r} {g} {b} rg {legend_x} {height - 68} 18 18 re f",
                     f"BT /F2 15 Tf {legend_x + 25} {height - 65} Td ({legend_label}) Tj ET"]
    for index, (key, mean, std) in enumerate(points):
        x = left + index * spacing + (spacing - bar_width) / 2
        bar_height = mean / maximum * plot_height
        dataset, method, availability, ablation = key
        label = method if availability == "independent" else f"{method}/{availability}"
        if ablation != "no_surrogate":
            label += f"/{ablation}"
        category = ("standard" if method in STANDARD_METHODS else
                    "fair_fl" if method in FAIR_FL_METHODS else
                    "scheduling" if method in SCHEDULING_METHODS else "cpp")
        r, g, b = pdf_colors[category]
        commands += [f"{r} {g} {b} rg {x:.2f} {bottom} {bar_width:.2f} {bar_height:.2f} re f",
                     f"BT /F1 13 Tf 0.819 0.574 -0.574 0.819 {x + 4:.2f} {bottom - 25} Tm ({_pdf_escape(label)}) Tj ET"]
    stream = "\n".join(commands).encode("latin-1", "replace")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode(),
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
    output, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output)); output += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(output); output += f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    output += f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(output)


def produce_graphs(results_csv, output_dir="benchmark_graphs"):
    """Create one mean±SD PDF graph per required metric from real run rows."""
    with Path(results_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_results(rows)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in METRICS:
        path = output / f"{metric}.pdf"
        _pdf_chart(f"{metric} (five-seed mean +/- SD)", _aggregate(rows, metric), path)
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
