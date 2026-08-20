"""Paper benchmark matrix and dependency-free graph generation.

This launcher intentionally consumes completed run CSVs; it never invents missing
measurements. Training jobs can use :func:`benchmark_matrix` to obtain the exact,
reproducible comparison grid and then call :func:`produce_graphs` on their results.
"""


from __future__ import annotations

import csv
import importlib.util
import itertools
import math
from collections import defaultdict
from pathlib import Path

if importlib.util.find_spec("matplotlib") is not None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:
    plt = None



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


METRIC_LABELS = {
    "global_accuracy": "Global accuracy (%)",
    "mean_client_accuracy": "Mean client utility / accuracy (%)",
    "worst_10_percent_utility": "Worst 10% mean utility",
    "utility_cv": "Utility coefficient of variation",
    "utility_jain_index": "Jain utility index",
    "conditional_selection_gap": "Conditional selection gap",
    "runtime_seconds": "Runtime (seconds)",
}
METHOD_DISPLAY = {
    "fedavg_random": "FedAvg-random", "uniform_available": "Uniform available",
    "fedprox": "FedProx", "q_ffl": "q-FFL", "php_fl": "PHP-FL",
    "fairfedcs": "FairFedCS", "fedfv": "FedFV", "afl": "AFL",
    "round_robin": "Round-robin", "least_selected": "Least-selected",
    "deficit_based": "Deficit-based", "inverse_availability": "Inverse availability",
    "oracle_availability": "Oracle availability",
    "estimated_availability": "Estimated availability", "cpp": "CPP",
}
METHOD_COLORS = {
    method: color for method, color in zip(METHODS, (
        (.12, .47, .71), (.68, .78, .91), (.20, .63, .17),
        (.89, .10, .11), (.98, .60, .60), (.58, .40, .74),
        (.77, .69, .84), (.55, .34, .29), (.89, .47, .76),
        (.50, .50, .50), (.74, .74, .13), (.09, .75, .81),
        (.65, .34, .16), (.40, .76, .65), (.40, .16, .60),
    ))
}




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


def _manual_pdf_chart(metric, points, path):
    """Write a self-contained vector PDF with axes, values, and SD bars."""
    width = max(1000, 125 * len(points))
    height, left, bottom, top = 700, 125, 135, 125
    observed_max = max((mean + std for _, mean, std in points), default=1.0)
    maximum = max(observed_max * 1.15, 1e-12)
    plot_height, plot_width = height - bottom - top, width - left - 35
    spacing = plot_width / max(1, len(points))
    bar_width = max(3, min(90, spacing * .68))
    title = METRIC_LABELS[metric]
    commands = ["1 1 1 rg 0 0 %s %s re f" % (width, height),
                "0 0 0 RG 1.5 w %s %s m %s %s l S" % (left, bottom, width - 35, bottom),
                "0 0 0 RG 1.5 w %s %s m %s %s l S" % (left, bottom, left, height - top),
                "BT /F2 22 Tf %s %s Td (%s) Tj ET" % (
                    max(left, width / 2 - len(title) * 6), height - 35, _pdf_escape(title)),
                "BT /F1 13 Tf %s %s Td (Bar = seed mean; whisker = sample SD) Tj ET" % (
                    width / 2 - 115, height - 55),
                "BT /F2 16 Tf 0 1 -1 0 28 %s Tm (%s) Tj ET" % (
                    bottom + plot_height / 2 - len(title) * 3.5, _pdf_escape(title)),
                "BT /F2 16 Tf %s 38 Td (Method) Tj ET" % (width / 2 - 28)]
    present_methods = list(dict.fromkeys(key[1] for key, _, _ in points))
    columns = min(5, max(1, len(present_methods)))
    legend_spacing = (width - left - 30) / columns
    for legend_index, method in enumerate(present_methods):
        legend_x = left + (legend_index % columns) * legend_spacing
        legend_y = height - 68 - (legend_index // columns) * 23
        r, g, b = METHOD_COLORS.get(method, (.25, .25, .25))
        commands += [f"{r} {g} {b} rg {legend_x:.2f} {legend_y} 16 16 re f",
                     f"BT /F2 11 Tf {legend_x + 22:.2f} {legend_y + 3} Td ({_pdf_escape(METHOD_DISPLAY.get(method, method))}) Tj ET"]
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom + plot_height * tick / 5
        commands += [f"0.85 0.85 0.85 RG 0.5 w {left} {y:.2f} m {width - 35} {y:.2f} l S",
                     f"0 0 0 RG 1 w {left - 5} {y:.2f} m {left} {y:.2f} l S",
                     f"BT /F1 13 Tf {left - 72} {y - 4:.2f} Td ({value:.3g}) Tj ET"]

    for index, (key, mean, std) in enumerate(points):
        x = left + index * spacing + (spacing - bar_width) / 2
        bar_height = mean / maximum * plot_height
        dataset, method, availability, ablation = key
        label = METHOD_DISPLAY.get(method, method)
        if availability != "independent":
            label += f"/{availability}"
        if ablation != "no_surrogate":
            label += f"/{ablation}"
        r, g, b = METHOD_COLORS.get(method, (.25, .25, .25))
        center, mean_y = x + bar_width / 2, bottom + bar_height
        std_height = std / maximum * plot_height
        label_x = center - min(len(label) * 3.2, spacing * .45)
        commands += [f"{r} {g} {b} rg {x:.2f} {bottom} {bar_width:.2f} {bar_height:.2f} re f",
                     f"0 0 0 RG 1.2 w {center:.2f} {max(bottom, mean_y - std_height):.2f} m {center:.2f} {mean_y + std_height:.2f} l S",
                     f"0 0 0 RG 1.2 w {center - 7:.2f} {mean_y + std_height:.2f} m {center + 7:.2f} {mean_y + std_height:.2f} l S",
                     f"BT /F2 12 Tf {center - 18:.2f} {mean_y + std_height + 7:.2f} Td ({mean:.4g}) Tj ET",
                     f"BT /F1 12 Tf {label_x:.2f} {bottom - 24} Td ({_pdf_escape(label)}) Tj ET"]
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


def _pdf_chart(metric, points, path):
    """Render with Matplotlib when installed; retain a dependency-free fallback."""
    if plt is None:
        _manual_pdf_chart(metric, points, path)
        return

    labels, means, deviations, colors = [], [], [], []
    for key, mean, std in points:
        _, method, availability, ablation = key
        label = METHOD_DISPLAY.get(method, method)
        if availability != "independent":
            label += f"\n{availability}"
        if ablation != "no_surrogate":
            label += f"\n{ablation}"
        labels.append(label)
        means.append(mean)
        deviations.append(std)
        colors.append(METHOD_COLORS.get(method, (.25, .25, .25)))

    figure_width = max(12, len(points) * 1.55)
    fig, axis = plt.subplots(figsize=(figure_width, 8.5), constrained_layout=True)
    positions = list(range(len(points)))
    bars = axis.bar(
        positions, means, yerr=deviations, capsize=6, color=colors,
        edgecolor="black", linewidth=0.8,
    )
    axis.set_title(METRIC_LABELS[metric], fontsize=22, fontweight="bold", pad=22)
    axis.set_xlabel("Method", fontsize=18, fontweight="bold", labelpad=14)
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=18, fontweight="bold", labelpad=14)
    axis.set_xticks(positions, labels, fontsize=14, rotation=25, ha="right")
    axis.tick_params(axis="y", labelsize=14)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.bar_label(bars, labels=[f"{value:.4g}" for value in means],
                   padding=8, fontsize=13, fontweight="bold")
    methods = list(dict.fromkeys(key[1] for key, _, _ in points))
    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=METHOD_COLORS.get(method, (.25, .25, .25)))
               for method in methods]
    axis.legend(
        handles, [METHOD_DISPLAY.get(method, method) for method in methods],
        title="Methods (each is a separate result series)", title_fontsize=14,
        fontsize=13, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=min(5, max(1, len(methods))), frameon=True,
    )
    axis.text(
        0.5, 1.01, "Bars show seed means; whiskers show sample standard deviation",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=13,
    )
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)



def produce_graphs(results_csv, output_dir="benchmark_graphs"):
    """Create one mean±SD PDF graph per required metric from real run rows."""
    with Path(results_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_results(rows)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    present = {row["method"] for row in rows}
    missing = [method for method in METHODS if method not in present]
    (output / "coverage.txt").write_text(
        "Graph renderer: " + ("Matplotlib" if plt is not None else "built-in PDF fallback") + "\n"
        "Graph format version: 4 (title, labeled axes, ticks, values, per-method legend)\n\n"
        "Each method is an independent result series with its own color.\n\n"
        "Methods present in benchmark_results.csv:\n  "
        + (", ".join(sorted(present)) or "none")
        + "\n\nMethods without result rows (not plotted):\n  "
        + (", ".join(missing) or "none") + "\n",
        encoding="utf-8")
    paths = []
    for metric in METRICS:
        path = output / f"{metric}.pdf"
        _pdf_chart(metric, _aggregate(rows, metric), path)
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
