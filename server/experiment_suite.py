"""Experiment orchestration for scalability + robustness studies.

Runs matrix over:
- logical population N in {100,300,1000,3000}
- selected clients m in {10,20,50,100}
- split mode in {overlap, dirichlet}
- one method selected through SELECTOR_MODE (CPP or a registered baseline)
- seeds (default 5)
- correlation noise in {0,10,20,40}

It launches main_server.py with env overrides and stores summarized outputs.
"""

import csv
import os
import subprocess
import statistics
import sys
import argparse
from collections import deque
from pathlib import Path
from baselines import RUNNABLE_BASELINES

CPP_METHOD = "select_fair_nodes"
DEFAULT_EXPERIMENT_METHODS = (CPP_METHOD,) + RUNNABLE_BASELINES



def resolve_paths():
    script_path = Path(__file__).resolve()
    # Case A: script located at repo_root/server/experiment_suite.py
    repo_root = script_path.parents[1]
    main_server = repo_root / "server" / "main_server.py"
    if main_server.exists():
        return repo_root, main_server

    # Case B: script copied/executed directly from /app/experiment_suite.py
    repo_root = script_path.parent
    main_server = repo_root / "main_server.py"
    if main_server.exists():
        return repo_root, main_server

    raise FileNotFoundError(
        f"Could not locate main_server.py relative to {script_path}. "
        "Expected either ./server/main_server.py or ./main_server.py"
    )


ROOT, MAIN_SERVER = resolve_paths()
OUT = ROOT / "experiment_results.csv"
SUMMARY = ROOT / "results_summary.csv"
RUN_DIR = ROOT / "experiment_runs"
BENCHMARK_RESULTS = ROOT / "benchmark_results.csv"
GRAPH_DIR = ROOT / "benchmark_graphs"
SUITE_FORMAT_VERSION = 2

BENCHMARK_FIELDS = [
    "dataset", "seed", "method", "availability_model", "ablation",
    "global_accuracy", "mean_client_accuracy", "worst_10_percent_utility",
    "utility_cv", "utility_jain_index", "conditional_selection_gap",
    "runtime_seconds",
]


def confidence_interval(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0
    stdev = statistics.stdev(values)
    half = 1.96 * stdev / (len(values) ** 0.5)
    return mean, half


def last_accuracy(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # A round can legitimately have no update.  Use the most recent completed
    # AW-PSP measurement rather than assuming the last CSV row is populated.
    for row in reversed(rows):
        val = row.get("accuracy")
        if val not in (None, ""):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def completed_metric_rounds(metrics_path):
    """Count distinct round rows actually persisted by the child process."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return 0
    with metrics_path.open() as handle:
        rounds = set()
        for row in csv.DictReader(handle):
            value = row.get("round", row.get("Round"))
            if value not in (None, ""):
                rounds.add(value)
    return len(rounds)


def write_benchmark_results(rows, destination=BENCHMARK_RESULTS,
                            include_in_progress=False):
    """Convert successful suite final-metric files into graph-ready rows.

    No values are synthesized: a run is omitted if it failed, is incomplete, or
    lacks any metric required by :mod:`benchmark_suite`.
    """
    benchmark_rows = []
    for run in rows:
        # Code 2 means the child completed but no round produced a global
        # accuracy.  Keep its other real measurements; process failures and
        # incomplete runs remain excluded.
        if run["return_code"] not in (0, 2):
            continue
        if (not include_in_progress
                and run["completed_rounds"] < run["expected_rounds"]):
            continue
        final_path = Path(run["final_metrics_path"])
        if not final_path.exists():
            continue
        with final_path.open(newline="") as handle:
            final_rows = list(csv.DictReader(handle))
        if not final_rows:
            continue
        final = final_rows[-1]
        use_surrogate = run["ablation"] in {"surrogate", "full_method"}
        suffix = "With Surrogate" if use_surrogate else "No Surrogate"
        candidate = {
            "dataset": run["dataset"],
            "seed": run["seed"],
            "method": "cpp" if run["selector"] == CPP_METHOD else run["selector"],
            "availability_model": run["availability_model"],
            "ablation": run["ablation"],
            "global_accuracy": final.get("global_accuracy"),
            "mean_client_accuracy": final.get("mean_client_accuracy"),
            "worst_10_percent_utility": final.get("worst_10_percent_utility"),
            "utility_cv": final.get(f"Utility CV ({suffix})"),
            "utility_jain_index": final.get(f"Jain (Utility) ({suffix})"),
            "conditional_selection_gap": final.get(f"Sel. Gap ({suffix})"),
            "runtime_seconds": final.get("runtime_seconds"),
        }
        # Preserve partial real measurements.  A round with no client update
        # legitimately has a blank global accuracy, but its utility/fairness
        # and runtime measurements are still graphable.  Requiring every value
        # used to discard the entire row and leave benchmark_results.csv empty.
        metadata_fields = BENCHMARK_FIELDS[:5]
        metric_fields = BENCHMARK_FIELDS[5:]
        if (all(candidate[field] not in (None, "") for field in metadata_fields)
                and any(candidate[field] not in (None, "") for field in metric_fields)):
            benchmark_rows.append(candidate)

    with Path(destination).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(benchmark_rows)
    return benchmark_rows


def write_experiment_results(rows, fieldnames, destination=OUT):
    """Rewrite the live suite index so every run has at most one current row."""
    with Path(destination).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_summary(rows, destination=SUMMARY):
    """Rewrite seed aggregates from all measurements available so far."""
    grouped = {}
    for row in rows:
        key = (row["N"], row["m"], row["split_mode"],
               row["labels_per_client"], row["selector"], row["noise_pct"])
        grouped.setdefault(key, []).append(row["accuracy_last"])

    fieldnames = ["N", "m", "split_mode", "labels_per_client", "selector",
                  "noise_pct", "mean", "standard_deviation", "ci95_low",
                  "ci95_high"]
    with Path(destination).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, values in grouped.items():
            clean = [value for value in values if isinstance(value, (float, int))]
            mean, ci = confidence_interval(clean)
            std = statistics.stdev(clean) if len(clean) > 1 else 0.0
            writer.writerow(dict(zip(fieldnames[:6], key), mean=mean,
                                 standard_deviation=std, ci95_low=mean - ci,
                                 ci95_high=mean + ci))



def run_case(env_overrides, run_tag, progress_callback=None):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RUN_DIR / f"metrics_{run_tag}.csv"
    final_path = RUN_DIR / f"final_{run_tag}.csv"
    log_path = RUN_DIR / f"run_{run_tag}.log"
    python_bin = os.getenv("PYTHON_BIN", sys.executable or "python3")
    cmd = [python_bin, str(MAIN_SERVER.relative_to(ROOT))]

    print(f"[suite] launching: {' '.join(cmd)}", flush=True)
    env["METRICS_LOG_PATH"] = str(metrics_path)
    # Keep each child's final metrics beside its round trace.  Without this
    # override every child writes ROOT/final_metrics.csv while the suite polls
    # experiment_runs/final_<tag>.csv, so live artifacts can never advance.
    env["FINAL_METRICS_PATH"] = str(final_path)
    env["SELECTOR_MODE"] = str(env_overrides.get("EXPERIMENT_METHOD", "select_fair_nodes"))
    env["REUSE_REGISTERED_CLIENTS"] = os.getenv("REUSE_REGISTERED_CLIENTS", "1")
    print(f"[suite] env: {env_overrides}", flush=True)
    print(f"[suite] metrics: {metrics_path}", flush=True)

    tail = deque(maxlen=4000)
    diagnostic_tail = deque(maxlen=200)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )


        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            tail.append(line)
            if any(marker in line for marker in (
                    "Error", "Exception", "Traceback", "No updates received",
                    "No fair-select updates", "No AW-PSP updates",
                    "No OORT updates", "No PSP updates",
                    "No client model updates",
                    "unavailable", "failed after")):
                diagnostic_tail.append(line)
            if progress_callback is not None:
                progress_callback(metrics_path, final_path)

    process.wait()
    merged_tail = ''.join(tail)
    acc = last_accuracy(metrics_path)
    return_code = process.returncode
    expected_rounds = int(env_overrides.get("NUM_ROUNDS", 50))
    completed_rounds = completed_metric_rounds(metrics_path)
    if return_code == 0 and completed_rounds < expected_rounds:
        return_code = 3
        diagnostic_tail.append(
            f"[suite] ERROR: child exited normally after persisting {completed_rounds}/{expected_rounds} "
            f"rounds. This is not a completed suite run; inspect the full child log at {log_path}.\n")
        diagnostic_tail.append("[suite] final child-output tail follows:\n")
        diagnostic_tail.append(merged_tail[-4000:])
    elif return_code == 0 and acc is None:
        return_code = 2
        diagnostic_tail.append(
            "[suite] ERROR: process exited successfully but produced no Accuracy; "
            f"inspect {log_path}\n")
    reported_tail = ''.join(diagnostic_tail) or merged_tail[-4000:]
    reported_tail = f"[full_log={log_path}]\n{reported_tail}"
    # Six values; keep this contract synchronized with the six targets in run_suite().
    return return_code, acc, reported_tail[-8000:], "", str(metrics_path), str(final_path)


def run_suite(live_graphs=False):
    print(
        f"[suite] format={SUITE_FORMAT_VERSION} ROOT={ROOT} MAIN_SERVER={MAIN_SERVER}",
        flush=True,
    )
    populations = [100]
    selections = [10]
    split_modes = ["overlap"]
    labels_per_client_options = [2]
    selectors = [
        item.strip() for item in os.getenv(
            "EXPERIMENT_METHODS", ",".join(DEFAULT_EXPERIMENT_METHODS)
        ).split(",") if item.strip()
    ]
    unknown = set(selectors) - set(DEFAULT_EXPERIMENT_METHODS)
    if unknown:
        raise ValueError(
            f"Unknown EXPERIMENT_METHODS entries: {sorted(unknown)}; choose from "
            f"{', '.join(DEFAULT_EXPERIMENT_METHODS)}")
    noises = [0]
    seeds = [0, 1, 2, 3, 4]
    rounds_per_run = int(os.getenv("NUM_ROUNDS", "50"))
    dataset = os.getenv("DATASET", "cifar10").lower()
    availability_model = os.getenv("AVAILABILITY_MODEL", "independent").lower()
    ablation = os.getenv("ABLATION", "no_surrogate").lower()


    rows = []
    result_fields = [
        "N", "m", "split_mode", "labels_per_client", "selector", "noise_pct", "seed",
        "dataset", "availability_model", "ablation",
        "return_code", "completed_rounds", "expected_rounds", "accuracy_last", "stdout_tail", "stderr_tail",
        "metrics_path", "final_metrics_path"
    ]
    # Create every public artifact before the first child starts. They are then
    # refreshed after each persisted federated round, not only after the suite.
    write_experiment_results(rows, result_fields)
    write_results_summary(rows)
    write_benchmark_results(rows, include_in_progress=True)

    total_runs = len(populations) * len(selections) * len(split_modes) * len(labels_per_client_options) * len(selectors) * len(noises) * len(seeds)
    total_federated_rounds = total_runs * rounds_per_run
    print(
        "[suite] matrix dimensions: "
        f"populations={len(populations)} selections={len(selections)} "
        f"split_modes={len(split_modes)} labels={len(labels_per_client_options)} "
        f"selectors={len(selectors)} noises={len(noises)} seeds={len(seeds)}; "
        f"child_runs={total_runs}; rounds_per_run={rounds_per_run}; "
        f"maximum_federated_rounds={total_federated_rounds}",
        flush=True,
    )
    done = 0
    for n in populations:
        for m in selections:
            for split in split_modes:
                for labels_per_client in labels_per_client_options:
                    for selector in selectors:
                        for noise in noises:
                            for seed in seeds:
                                run_tag = f"N{n}_m{m}_{split}_labels{labels_per_client}_{selector}_noise{noise}_seed{seed}"

                                live_row = {
                                    "N": n, "m": m, "split_mode": split,
                                    "labels_per_client": labels_per_client,
                                    "selector": selector, "noise_pct": noise,
                                    "seed": seed, "dataset": dataset,
                                    "availability_model": availability_model,
                                    "ablation": ablation, "return_code": 0,
                                    "completed_rounds": 0,
                                    "expected_rounds": rounds_per_run,
                                    "accuracy_last": None,
                                    "stdout_tail": "", "stderr_tail": "",
                                    "metrics_path": "", "final_metrics_path": "",
                                }
                                rows.append(live_row)
                                # Make the in-progress run visible immediately.
                                # A federated round can take a long time, and an
                                # empty index during that time looks like a
                                # failed suite.
                                write_experiment_results(rows, result_fields)
                                write_results_summary(rows)
                                last_persisted_round = -1

                                def persist_round(metrics_file, final_file):
                                    nonlocal last_persisted_round
                                    completed = completed_metric_rounds(metrics_file)
                                    if completed == 0 or completed == last_persisted_round:
                                        return
                                    # main_server writes the round trace before its
                                    # final-metric row. Wait until both files expose
                                    # the round so benchmark_results.csv is complete.
                                    if not final_file.exists():
                                        return
                                    with final_file.open(newline="") as handle:
                                        final_rows = list(csv.DictReader(handle))
                                    if not final_rows or int(final_rows[-1].get("Round", 0)) < completed:
                                        return
                                    last_persisted_round = completed
                                    live_row.update({
                                        "completed_rounds": completed,
                                        "accuracy_last": last_accuracy(metrics_file),
                                        "metrics_path": str(metrics_file),
                                        "final_metrics_path": str(final_file),
                                    })
                                    write_experiment_results(rows, result_fields)
                                    write_results_summary(rows)
                                    benchmark_rows = write_benchmark_results(
                                        rows, include_in_progress=True)
                                    if live_graphs and benchmark_rows:
                                        create_graphs(BENCHMARK_RESULTS, GRAPH_DIR)
                                    print(
                                        f"[suite] persisted live artifacts after round "
                                        f"{completed}/{rounds_per_run} for {run_tag}",
                                        flush=True,
                                    )

                                # run_case() returns exactly these six values.

                                code, acc, out_tail, err_tail, metrics_path, final_path = run_case(
                                    {
                                        "LOGICAL_CLIENT_COUNT": n,
                                        "LOGICAL_SELECTED_PER_ROUND": m,
                                        "LOGICAL_LABELS_PER_CLIENT": labels_per_client,
                                        "PHYSICAL_CLIENT_COUNT": os.getenv(
                                            "PHYSICAL_CLIENT_COUNT", "3"),
                                        "LOGICAL_SPLIT_MODE": split,
                                        "EXPERIMENT_METHOD": selector,
                                        "SELECTION_MODE": os.getenv("SELECTION_MODE", "cup"),
                                        "CORRELATION_NOISE_PCT": noise,
                                        "EXPERIMENT_SEED": seed,
                                        "DATASET": dataset,
                                        "AVAILABILITY_MODEL": availability_model,
                                        "ABLATION": ablation,
                                        "NUM_ROUNDS": rounds_per_run,
                                        "USE_LOGICAL_SCHEDULING": True,
                                    },
                                    run_tag,
                                    progress_callback=persist_round,
                                )
                                done += 1

                                completed_rounds = completed_metric_rounds(metrics_path)
                                print(f"[suite] child_run={done}/{total_runs} completed_rounds={completed_rounds}/{rounds_per_run} N={n} m={m} split={split} labels={labels_per_client} selector={selector} noise={noise} seed={seed} rc={code} acc={acc}", flush=True)
                                live_row.update({
                                    "return_code": code,
                                    "completed_rounds": completed_rounds,
                                    "accuracy_last": acc,
                                    "stdout_tail": out_tail.replace("\n", "\\n"),
                                    "stderr_tail": err_tail.replace("\n", "\\n"),
                                    "metrics_path": metrics_path,
                                    "final_metrics_path": final_path,
                                })
                                write_experiment_results(rows, result_fields)
                                write_results_summary(rows)
                                write_benchmark_results(
                                    rows, include_in_progress=True)

    benchmark_rows = write_benchmark_results(rows)
    print(
        f"[suite] wrote {len(benchmark_rows)} graph-ready rows to {BENCHMARK_RESULTS}",
        flush=True,
    )


    write_experiment_results(rows, result_fields)
    write_results_summary(rows)


def sweep_configurations():
    """Yield the two paper hyperparameter sweeps for experiment launchers."""
    for value in (0.05, 0.10, 0.20, 0.50):
        yield {"sweep": "lambda_decay", "LAMBDA_DECAY": value,
               "metrics": "Accuracy,Utility CV,Jain Index,Selection Gap"}
    for value in (10, 20, 50):
        yield {"sweep": "availability_window", "AVAILABILITY_WINDOW_SIZE": value,
               "metrics": "availability estimation error,Utility CV,Accuracy"}


def create_graphs(results_path=BENCHMARK_RESULTS, output_dir=GRAPH_DIR):
    from benchmark_suite import produce_graphs
    paths = produce_graphs(results_path, output_dir)
    print(f"[suite] wrote {len(paths)} graphs under {output_dir}", flush=True)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run FL experiments and/or graph completed results")
    parser.add_argument("action", choices=("run", "graphs", "all"), nargs="?", default="all")
    parser.add_argument("--results", type=Path, default=BENCHMARK_RESULTS)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    args = parser.parse_args(argv)
    if args.action in {"run", "all"}:
        run_suite(live_graphs=args.action == "all")
        print(f"[suite] wrote {OUT} and {SUMMARY} (per-run logs under {RUN_DIR})", flush=True)
    if args.action in {"graphs", "all"}:
        create_graphs(args.results, args.graph_dir)





if __name__ == "__main__":
    main()
