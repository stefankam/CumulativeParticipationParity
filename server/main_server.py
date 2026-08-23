# main_server.py

from flask import Flask, request
import torch
import io
import os
import csv
import time
import copy
import math
from typing import Dict, Tuple
from pathlib import Path
from torchvision import models
from shared_state import topology
import threading
import random
import requests
import socket
from topology_server import TopologyProvider
import shared_state
from availability import (extract_availability_vectors,
                          logical_client_availability,
                          resolve_availability_trace_path)

import numpy as np
import json
from collections import defaultdict
from baselines import (ALL_BASELINES, RUNNABLE_BASELINES, UNIMPLEMENTED_BASELINES,
                       BaselineClient, BaselineState, select_clients)
from fairness import fairness_metrics
from experiment_config import build_logical_label_map
from fl_methods import FairFedCSState
from cup import CumulativeUtilityParity


def read_proc_stat() -> Tuple[int, int]:
    with open("/proc/stat", "r") as f:
        cpu_line = f.readline().strip().split()
    values = list(map(int, cpu_line[1:]))
    idle = values[3] + values[4]  # idle + iowait
    total = sum(values)
    return total, idle


def read_meminfo() -> Dict[str, int]:
    info = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key.strip()] = int(value.strip().split()[0])
    return info


def read_diskstats() -> Dict[str, int]:
    stats = {}
    with open("/proc/diskstats", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            name = parts[2]
            if name.startswith("loop") or name.startswith("ram"):
                continue
            reads = int(parts[3])
            writes = int(parts[7])
            sectors_read = int(parts[5])
            sectors_written = int(parts[9])
            stats[name] = reads + writes + sectors_read + sectors_written
    return stats


def snapshot_system():
    total, idle = read_proc_stat()
    mem = read_meminfo()
    disk = read_diskstats()
    return total, idle, mem, disk


def summarize_system(start, end):
    total0, idle0, mem0, disk0 = start
    total1, idle1, mem1, disk1 = end
    cpu_delta = total1 - total0
    idle_delta = idle1 - idle0
    cpu_pct = 0.0 if cpu_delta == 0 else (1.0 - idle_delta / cpu_delta) * 100
    mem_used_kb = mem1.get("MemTotal", 0) - mem1.get("MemAvailable", 0)
    disk_delta = sum(disk1.values()) - sum(disk0.values())
    return cpu_pct, mem_used_kb, disk_delta


app = Flask(__name__)
current_round = 0
server_started_at = time.time()


# Global device registry
registry_path = Path(os.getenv(
    "REGISTERED_CLIENTS_CACHE",
    os.getenv("DEVICE_REGISTRY_PATH", "registered_clients.json"),
))
registry_lock = threading.Lock()


def reuse_registered_clients_enabled():
    return os.getenv("REUSE_REGISTERED_CLIENTS", "1").lower() in (
        "1", "true", "yes", "on")


def load_device_registry():
    """Restore physical endpoints shared by consecutive seed processes."""
    if not reuse_registered_clients_enabled() or not registry_path.exists():
        return {}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        restored = {}
        for device_id, endpoint in data.items():
            restored[str(device_id)] = dict(endpoint)
            restored[str(device_id)]["ip"] = str(endpoint["ip"])
            restored[str(device_id)]["port"] = int(endpoint["port"])
        return restored
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"⚠️ Ignoring invalid device registry {registry_path}: {error}")
        return {}


device_registry = load_device_registry()


def persist_device_registry():
    """Atomically persist endpoints and latest telemetry for the next run."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    temporary.write_text(json.dumps(device_registry), encoding="utf-8")
    temporary.replace(registry_path)
#topology = None  # ← define globally so update_status() can access it
#current_round = 0  # make current_round global to


# -------------------------------
# 1. REGISTRATION ENDPOINT
# -------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()

    # 🧠 Use actual sender IP, not what the client claims
#    sender_ip = request.remote_addr

    with registry_lock:
        device_id = data["device_id"]
        endpoint = {"ip": str(data["ip"]), "port": int(data["port"])}

        # The logical scheduler refers to stable physical slots (Device_0,
        # Device_1, ...). A restarted replacement must therefore register with
        # the failed worker's device_id. Reject a brand-new slot after topology
        # initialization because it would not exist in the DHT used by status
        # updates and selection.
        if shared_state.topology is not None and device_id not in device_registry:
            return (
                "UNKNOWN_DEVICE_ID: restart the replacement with the failed "
                "worker's --device_id",
                409,
            )

        previous = dict(device_registry.get(device_id, {}))
        endpoint_changed = any(
            previous.get(key) != value for key, value in endpoint.items())

        # Refresh the endpoint without erasing telemetry saved by the previous
        # seed process. This is also the live failover path: send_weights reads
        # this shared registry for every request, so later logical-client waves
        # immediately use the replacement container's address.
        device_registry.setdefault(device_id, {}).update(endpoint)
        persist_device_registry()

    print(f"📥 Registered {data['device_id']} at {data['ip']}:{data['port']}")
    if endpoint_changed and previous:
        print(
            f"🔄 Rebound physical client {device_id} from "
            f"{previous.get('ip')}:{previous.get('port')} to "
            f"{endpoint['ip']}:{endpoint['port']}"
        )
        return "REBOUND", 200
    if previous:
        return "ALREADY_REGISTERED", 200

    print(f"📥 Registered {device_id} at {endpoint['ip']}:{endpoint['port']}")
    return "OK", 200


# Distributed Hash Table
class DHT:
    def __init__(self, size=100):
        self.table = {}
        self.size = size

@app.route("/ready", methods=["GET"])
def ready():
    if shared_state.topology:
        return "ready", 200
    return "not_ready", 503

def initialize_topology(device_file="devices.txt", num_clients=None):
    if num_clients is None:
        num_clients = int(os.getenv(
            "REGISTERED_CLIENT_COUNT",
            os.getenv("PHYSICAL_CLIENT_COUNT",
                      os.getenv("PHYSICAL_CONTAINER_LIMIT", "2")),
        ))
    restored_count = len(device_registry)
    if restored_count >= num_clients:
        print(
            f"📂 Restored {restored_count} endpoint(s) from {registry_path}; "
            "no re-registration wait is required."
        )
    else:
        print(
            f"⏳ Restored {restored_count} endpoint(s) from {registry_path}; "
            "waiting only for the missing registrations."
        )
        while len(device_registry) < num_clients:
            print(f"🕒 Registered devices: {len(device_registry)} / {num_clients}")
            time.sleep(2)

    # REGISTERED_CLIENT_COUNT is a minimum, not a hard cap. Once that minimum
    # is present, keep registration open for a short quiet period so concurrently
    # starting standby containers join the topology before /ready becomes true.
    settle_seconds = float(os.getenv("PHYSICAL_REGISTRATION_SETTLE_SECONDS", "10"))
    if settle_seconds > 0:
        observed_count = len(device_registry)
        settle_deadline = time.monotonic() + settle_seconds
        print(
            f"⏳ Minimum registration count reached; waiting {settle_seconds:g}s "
            "for additional standby clients."
        )
        while time.monotonic() < settle_deadline:
            current_count = len(device_registry)
            if current_count != observed_count:
                observed_count = current_count
                settle_deadline = time.monotonic() + settle_seconds
                print(
                    f"📥 Registration pool now has {observed_count} clients; "
                    "resetting standby settle timer."
                )
            time.sleep(0.25)

    registered_count = len(device_registry)
    print(
        f"✅ Registration pool closed with {registered_count} clients. "
        "Initializing topology."
    )

    device_ids = list(device_registry.keys())

    shared_state.topology = TopologyProvider(
        device_names=device_ids,
        num_epochs=1,
        link_latency=20, 
        link_loss=5,
        model_name='resnet',
        device_registry=device_registry 
    )
    shared_state.topology.dht = DHT(size=100)  # Initialize the DHT
    for device_id in device_ids:
        cached = device_registry[device_id]
        shared_state.topology.dht.table[device_id] = {
          "latency": cached.get("latency"),
          "packet_loss": cached.get("packet_loss"),
          "last_seen": cached.get("last_seen"),
          "availability": cached.get("availability"),
          "freshness": cached.get("freshness"),
          "correlation": cached.get("correlation"),
        }
    print("✅ Topology initialized.")
    # The registry may contain endpoints left by containers that died during a
    # previous benchmark. Require only enough fresh workers to fill the active
    # pool, then collect every additional fresh reporter as a standby.
    wait_for_latency_data(int(os.getenv("PHYSICAL_CONTAINER_LIMIT", "3")))


@app.route("/status_update", methods=["POST"])
def update_status():
    global current_round
    data = request.get_json()
    node = data["device_id"]

    if shared_state.topology is None:
        print("⚠️ Topology not initialized yet. Ignoring status update.")
        return "ERROR: Topology not initialized", 503

    if node not in shared_state.topology.dht.table:
        print(f"⚠️ Node {node} not found in DHT.")
        return "ERROR: Node not found", 404

    shared_state.topology.dht.table[node]["latency"] = data["latency"]
    shared_state.topology.dht.table[node]["packet_loss"] = data["packet_loss"]
    shared_state.topology.dht.table[node]["last_seen"] = time.time()
    shared_state.topology.dht.table[node]["availability"] = data["availability"]
    shared_state.topology.dht.table[node]["freshness"] = shared_state.topology.get_freshness(node, current_round)
    shared_state.topology.dht.table[node]["correlation"] = shared_state.topology.failure_correlation.get(node, {})
    with registry_lock:
        device_registry.setdefault(node, {}).update({
            key: shared_state.topology.dht.table[node].get(key)
            for key in ("latency", "packet_loss", "last_seen", "availability",
                        "freshness", "correlation")
        })
        # defaultdict/set correlation state is not JSON serializable and is
        # recomputed by each coordinator; persist only scalar telemetry.
        device_registry[node]["correlation"] = None
        persist_device_registry()
    print(f"📶 Updated status for {node}: latency={data['latency']}, loss={data['packet_loss']}")


    return "OK", 200

# -------------------------------
# 2. HEALTH CHECK (optional)
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

# -------------------------------
# 3. FEDERATED COORDINATOR
# -------------------------------

import torch.nn as nn
from torchvision import models

def init_resnet(train_last_n_blocks=1):
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)

    # Freeze everything first
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze last N blocks + FC
    if train_last_n_blocks >= 1:
        for param in base_model.layer4.parameters():
            param.requires_grad = True
    if train_last_n_blocks >= 2:
        for param in base_model.layer3.parameters():
            param.requires_grad = True
    if train_last_n_blocks >= 3:
        for param in base_model.layer2.parameters():
            param.requires_grad = True

    # Always unfreeze fc
    for param in base_model.fc.parameters():
        param.requires_grad = True

    return base_model



def run_federated_training():
    """Run one explicitly selected policy; comparisons belong to benchmark_suite."""
    global current_round
    if shared_state.topology is None:
        raise RuntimeError("Topology has not been initialized")

    seed = int(os.getenv("EXPERIMENT_SEED", "42"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)



    logical_count = int(os.getenv("LOGICAL_CLIENT_COUNT", 100))
    selected_per_round = int(os.getenv("LOGICAL_SELECTED_PER_ROUND", 10))
    physical_limit = int(os.getenv("PHYSICAL_CONTAINER_LIMIT", 10))
    use_logical = os.getenv(
        "USE_LOGICAL_SCHEDULING", "true").lower() in ("1", "true", "yes", "on")
    labels_per_client = int(os.getenv("LOGICAL_LABELS_PER_CLIENT", 2))
    split_mode = os.getenv("LOGICAL_SPLIT_MODE", "overlap")
    selector = os.getenv("SELECTOR_MODE", "select_fair_nodes").lower()
    if selector != "select_fair_nodes" and selector not in ALL_BASELINES:
        raise ValueError(f"Unknown SELECTOR_MODE={selector!r}")

    if selector in UNIMPLEMENTED_BASELINES:
        raise NotImplementedError(
            f"SELECTOR_MODE={selector!r} is not an end-to-end implementation: "
            f"{UNIMPLEMENTED_BASELINES[selector]}. Runnable baselines: "
            f"{', '.join(RUNNABLE_BASELINES)}")
    if not use_logical:
        raise NotImplementedError(
            "Audited CUP and common retrospective baseline accounting require "
            "USE_LOGICAL_SCHEDULING=true; the legacy physical path does not "
            "provide a complete per-round A/S/P vector for the logical population.")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    weights = copy.deepcopy(model.state_dict())
    logical_labels = build_logical_label_map(
        logical_count, labels_per_client, split_mode=split_mode,
        dirichlet_alpha=float(os.getenv("DIRICHLET_ALPHA", 0.5)), seed=seed,
    )
    trace_path = resolve_availability_trace_path(
        os.getenv("AVAILABILITY_TRACE_PATH", "traces/traces.txt"))
    availability_vectors = extract_availability_vectors(
        trace_path,
        length=max(100, int(os.getenv("NUM_ROUNDS", 50))),
    )


    trace_path = resolve_availability_trace_path(
        os.getenv("AVAILABILITY_TRACE_PATH", "traces/traces.txt"))
    availability_vectors = extract_availability_vectors(
        trace_path,
        length=max(100, int(os.getenv("NUM_ROUNDS", 50))),
    )


    client_ids = [f"h{i}" for i in range(logical_count)]
    shared_state.topology.configure_logical_clients(client_ids)
    selection_counts = defaultdict(int)
    participation_counts = defaultdict(int)
    availability_seen = defaultdict(int)
    baseline_state = BaselineState()
    cup = CumulativeUtilityParity(
        client_ids, selected_per_round, seed=seed,
        output_path=os.getenv("CUP_ROUND_LOG_PATH", "cup_rounds.csv"))
    fairfedcs_state = FairFedCSState(tuple(client_ids), selected_per_round)
    metrics_path = os.getenv("METRICS_LOG_PATH", "metrics_log.csv")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)


    final_metrics_path = os.getenv("FINAL_METRICS_PATH", "final_metrics.csv")
    Path(final_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    final_fields = [
        "Round", "global_accuracy", "mean_client_accuracy",
        "Jain (Client Accuracy)",
        "utility_metric", "cup_scheduler",
        "worst_10_percent_utility", "Utility CV (No Surrogate)",
        "Jain (Utility) (No Surrogate)", "Sel. Gap (No Surrogate)",
        "Utility CV (With Surrogate)", "Jain (Utility) (With Surrogate)",
        "Sel. Gap (With Surrogate)", "runtime_seconds",
    ]
    training_started = time.perf_counter()

    with open(metrics_path, "w", newline="") as output, open(
            final_metrics_path, "w", newline="") as final_output:
        writer = csv.DictWriter(output, fieldnames=[
            "round", "method", "accuracy", "participation_gini",
            "participation_variance", "selected_clients", "utility_metric",
            "cup_scheduler",
        ])
        writer.writeheader()
        final_writer = csv.DictWriter(final_output, fieldnames=final_fields)
        final_writer.writeheader()
        for current_round in range(int(os.getenv("NUM_ROUNDS", 50))):
            physical_ids = shared_state.topology.active_physical_worker_snapshot()
            if use_logical:
                clients = []
                for client_id in client_ids:
                    # Logical availability comes from the per-device trace. A
                    # healthy physical pool is execution capacity, not evidence
                    # that every simulated logical client is online.
                    trace_available = logical_client_availability(
                        availability_vectors, client_id, current_round)
                    available = float(bool(physical_ids) and trace_available)
                    availability_seen[client_id] += int(available > 0)
                    estimate = availability_seen[client_id] / (current_round + 1)
                    clients.append(BaselineClient(
                        client_id, available, estimate,
                        selection_counts[client_id], tuple(logical_labels.get(client_id, ())),
                    ))
                telemetry = {client.client_id: bool(client.availability) for client in clients}
                if selector == "select_fair_nodes":
                    selected = cup.select_clients(
                        telemetry, current_round,
                        mu_hat={
                            client_id: max(
                                cup.states[client_id].utility
                                / max(1, cup.states[client_id].participation_count),
                                float(os.getenv("CUP_EPSILON", "1e-3")),
                            ) for client_id in client_ids
                        })
                elif selector == "fairfedcs":
                    selected = fairfedcs_state.select(
                        [client.client_id for client in clients if client.availability],
                        selected_per_round,
                    )
                else:
                    selected = select_clients(selector, clients, selected_per_round, baseline_state, rng=rng)
                if selector != "select_fair_nodes":
                    cup.observe_external_selection(telemetry, current_round, selected)
                participating = cup.realize_participation(telemetry, selected)
                if selector == "select_fair_nodes":
                    shared_state.topology.set_cup_round_context(
                        cup.aggregation_context())
                updated = shared_state.topology.run_logical_federated_round(
                    participating, physical_ids, weights)
            else:
                failures = shared_state.topology.get_correlated_failure(
                    current_round, extract_availability_vectors("traces/traces.txt"),
                    corr_threshold=0.35, num_neighbors=4,
                )


                if selector == "select_fair_nodes":
                    selected, _, _ = shared_state.topology.select_fair_nodes(
                        model, current_round, failures, shared_state.topology.label_map,
                        selected_per_round,
                    )
                else:

                    failed = {node for pair in failures for node in pair}
                    clients = [BaselineClient(
                        node,
                        0.0 if node in failed else float(meta.get("availability") or 0.0),
                        float(meta.get("availability") or 0.0),
                        selection_counts[node],
                        tuple(shared_state.topology.label_map.get(node, ())),
                    ) for node, meta in shared_state.topology.dht.table.items()]
                    selected = select_clients(selector, clients, selected_per_round, baseline_state, rng=rng)
                updated = shared_state.topology.run_federated_round(selected, weights, model)

            if updated is not None:
                weights = updated
                model.load_state_dict(weights)
                accuracy = shared_state.topology.evaluate_global_model(
                    model, selected_nodes=selected, use_selected_nodes=False,
                )
                per_client_accuracy = (
                    shared_state.topology.evaluate_logical_client_accuracy(
                        model, logical_labels)
                    if use_logical else
                    shared_state.topology.evaluate_per_client_accuracy(model, client_ids)
                )

            else:
                accuracy = None
                per_client_accuracy = {
                    client_id: (cup.states[client_id].previous_eval_accuracy or 0.0)
                    for client_id in client_ids}

                print(
                    "ERROR: No client model updates were received; metrics for "
                    "this round are unavailable. Check client /train logs and "
                    "CLIENT_TRAIN_TIMEOUT_SECONDS instead of interpreting zeros "
                    "as measured accuracy or utility.",
                    flush=True,
                )
            for client_id in selected:
                selection_counts[client_id] += 1
            for client_id in participating:
                participation_counts[client_id] += 1
            cup.end_round(
                current_round, telemetry, selected, per_client_accuracy,
                shared_state.topology.last_client_records)
            if selector == "fairfedcs":
                fairfedcs_state.on_round_end(
                    selected, shared_state.topology.last_client_records)
                print(
                    "FairFedCS state:",
                    {client_id: {
                        "reputation": fairfedcs_state.reputations[client_id],
                        "queue": fairfedcs_state.queues[client_id],
                        "csi": fairfedcs_state.suitability(client_id),
                    } for client_id in client_ids},
                )
            counts = [selection_counts[client_id] for client_id in client_ids]
            mean = sum(counts) / len(counts) if counts else 0.0
            variance = sum((value - mean) ** 2 for value in counts) / len(counts) if counts else 0.0
            round_fairness = fairness_metrics(
                {client_id: float(participation_counts[client_id]) for client_id in client_ids},
                participation_counts)
            writer.writerow({
                "round": current_round, "method": selector, "accuracy": accuracy,
                "participation_gini": round_fairness["gini_coefficient"],
                "participation_variance": variance, "selected_clients": selected,
                "utility_metric": cup.utility_metric,
                "cup_scheduler": cup.scheduler_mode,
            })
            output.flush()


            observed_utilities = [
                cup.states[client_id].normalized_utility
                if math.isfinite(cup.states[client_id].normalized_utility) else 0.0
                for client_id in client_ids]
            utility_by_client = dict(zip(client_ids, observed_utilities))
            utility_fairness = fairness_metrics(utility_by_client, selection_counts)
            cup_metrics = cup.metrics(current_round + 1)
            cup_metrics_with_surrogate = cup.metrics(
                current_round + 1, include_surrogate=True)
            accuracy_fairness = fairness_metrics(
                per_client_accuracy, selection_counts)
            tail_size = max(1, math.ceil(len(observed_utilities) * 0.1))
            final_writer.writerow({
                "Round": current_round + 1,
                "global_accuracy": accuracy,
                "mean_client_accuracy": (
                    sum(per_client_accuracy.values()) / len(per_client_accuracy)
                    if per_client_accuracy else 0.0),
                "Jain (Client Accuracy)": accuracy_fairness["utility_jain_index"],
                "utility_metric": cup.utility_metric,
                "cup_scheduler": cup.scheduler_mode,
                "worst_10_percent_utility": (
                    sum(sorted(observed_utilities)[:tail_size]) / tail_size),
                "Utility CV (No Surrogate)": cup_metrics["utility_cv"],
                "Jain (Utility) (No Surrogate)": cup_metrics["utility_jain_index"],
                "Sel. Gap (No Surrogate)": cup_metrics["selection_gap"],
                "Utility CV (With Surrogate)": cup_metrics_with_surrogate["utility_cv"],
                "Jain (Utility) (With Surrogate)": cup_metrics_with_surrogate["utility_jain_index"],
                "Sel. Gap (With Surrogate)": cup_metrics_with_surrogate["selection_gap"],
                "runtime_seconds": time.perf_counter() - training_started,
            })
            final_output.flush()

            print(f"Round {current_round + 1}: {selector} accuracy={accuracy} selected={selected}")



def fresh_latency_nodes():
    """Return workers that have reported telemetry to this coordinator process."""
    return [
        node for node, metadata in shared_state.topology.dht.table.items()
        if (metadata.get("latency") is not None
            and metadata.get("packet_loss") is not None
            and metadata.get("last_seen") is not None
            and metadata["last_seen"] >= server_started_at)
    ]


def wait_for_latency_data(num_clients=3):
    print("⏳ Waiting for latency updates from clients...")
    telemetry_settle_seconds = float(os.getenv(
        "PHYSICAL_TELEMETRY_SETTLE_SECONDS",
        os.getenv("PHYSICAL_REGISTRATION_SETTLE_SECONDS", "10"),
    ))
    while True:
        ready_nodes = fresh_latency_nodes()
        print(f"✅ Clients with fresh latency info: {len(ready_nodes)} / {num_clients}")
        if len(ready_nodes) >= num_clients:
            break
        time.sleep(2)

    # Do not use len(device_registry) here: the shared registry intentionally
    # survives across benchmark children and can contain dead endpoints. Wait
    # for a quiet period and build the pool solely from current-process reporters.
    if telemetry_settle_seconds > 0:
        observed_ready = len(ready_nodes)
        settle_deadline = time.monotonic() + telemetry_settle_seconds
        print(
            f"⏳ Active capacity reached; collecting fresh standbys for "
            f"{telemetry_settle_seconds:g}s."
        )
        while time.monotonic() < settle_deadline:
            current_ready = fresh_latency_nodes()
            if len(current_ready) != observed_ready:
                ready_nodes = current_ready
                observed_ready = len(current_ready)
                settle_deadline = time.monotonic() + telemetry_settle_seconds
                print(
                    f"📡 Fresh physical pool now has {observed_ready} workers; "
                    "resetting telemetry settle timer."
                )
            time.sleep(0.25)

    physical_limit = int(os.getenv("PHYSICAL_CONTAINER_LIMIT", "3"))
    shared_state.topology.configure_physical_worker_pool(
        ready_nodes, physical_limit)
    print("🚀 Fresh physical pool ready. Starting training.")
    run_federated_training()



if __name__ == "__main__":
    print("🚀 Starting main server HTTP API...")


    # Step 1: Start the server in a separate thread
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()

    # Step 2: Start topology initialization in the background
#    threading.Thread(target=initialize_topology).start()
    # Keep Flask in the background but run coordination on the main thread.
    # Exceptions during training must make the child process fail instead of
    # being hidden in a thread while experiment_suite reports a normal exit.
    initialize_topology()

    # Step 3: Wait for latency info from clients, then start training
    #threading.Thread(target=wait_for_latency_data).start()
