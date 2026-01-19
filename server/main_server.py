# main_server.py

from flask import Flask, request
import torch
import io
import os
import csv
import time
import copy
from torchvision import models
from shared_state import topology
import threading
import requests
import socket
from topology_server import TopologyProvider
import shared_state
from availability import extract_availability_vectors


app = Flask(__name__)
current_round = 0

# Global device registry
device_registry = {}
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

    device_registry[data["device_id"]] = {
        "ip": data["ip"],
        "port": data["port"]
    }

    print(f"📥 Registered {data['device_id']} at {data['ip']}:{data['port']}")

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

def initialize_topology(device_file="devices.txt", num_clients=3):
    print("⏳ Waiting for clients to register...")
    while len(device_registry) < num_clients:
        print(f"🕒 Registered devices: {len(device_registry)} / {num_clients}")
        time.sleep(2)

    print("✅ All clients registered. Initializing topology.")

#    print("📄 Loading devices from file...")

#    device_registry = {}
#    with open(device_file, "r") as f:
#        for i, line in enumerate(f):
#            if i >= num_clients:
#                break
#            if not line.strip():
#                continue
#            device_id, ip, port = line.strip().split()
#            device_registry[device_id] = {
#                "ip": ip,
#                "port": int(port)
#            }

#    print(f"✅ Loaded {len(device_registry)} devices.")

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
        shared_state.topology.dht.table[device_id] = {
          "latency": None,
          "packet_loss": None,
          "last_seen": None,
          "availability": None,
          "freshness": None,
          "correlation": None
        }
    print("✅ Topology initialized.")
    wait_for_latency_data()



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
    base_model = models.resnet34(weights=None)
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
    global current_round
    if shared_state.topology is None:
        raise RuntimeError("❌ Topology has not been initialized.")

    label_map = shared_state.topology.label_map

    # Load availability vectors
    availability_vectors = extract_availability_vectors("traces/traces.txt")

    awpsp_accuracy_log = []
    psp_accuracy_log = [] 
    awpsp_instant_fairness_log = []
    psp_instant_fairness_log = []
    awpsp_cumul_fairness_log = []
    psp_cumul_fairness_log = []  
    corr_failure_log = []
    awpsp_covered_labels_log = []
    psp_covered_labels_log = []
    selected_awpsp_log = []
    selected_psp_log = []
    awpsp_avg_score_log = []
    psp_avg_score_log = []
    awpsp_labels_log =[]
    psp_labels_log =[]
    awpsp_KL_log =[]
    psp_KL_log =[]
    awpsp_unseen_log =[]
    psp_unseen_log =[]
    awpsp_gini_log =[]
    psp_gini_log =[]
    accuracy_log = []
    var_u_log = []
    surrogate_log = []

    # ---------------- Initialize models ----------------
    #base_model = models.resnet18(weights=None)

    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, 10)
#    base_model = init_resnet(train_last_n_blocks=2)  # train layer3 + layer4 + fc

    # Two independent weight states
    current_weights_awpsp = base_model.state_dict()
    current_weights_psp = copy.deepcopy(current_weights_awpsp)

    num_rounds = 50
    for current_round in range(num_rounds):
        print(f"\n🌐 Federated Round {current_round + 1}")

        # Detect correlated failures
        correlated_failures = shared_state.topology.get_correlated_failure(
            current_round, availability_vectors, corr_threshold=0.35, num_neighbors=4
        )
        num_corr_failed = sum(1 for _, failed_neighbors in correlated_failures if failed_neighbors)
        corr_failure_log.append((current_round, num_corr_failed))
        print(f"🌩️ Correlated failure count: {num_corr_failed}")
        selected, var_u, total_bias_bound = shared_state.topology.select_fair_nodes(
            base_model,
            current_round,
            correlated_failures,
            num_clients=5,
            corr_threshold=0.35,
            label_map=label_map,
            lambda_=0.5,
            epsilon=1e-5,
        )
        weights_fair = shared_state.topology.run_federated_round(selected, current_weights_awpsp, base_model)
        if weights_fair is not None:
           base_model.load_state_dict(weights_fair)
           current_weights_awpsp = weights_fair
           accuracy = shared_state.topology.evaluate_global_model(base_model, use_selected_nodes=False)
           accuracy_log.append((current_round, accuracy))
           var_u_log.append((current_round, var_u))
           surrogate_log.append((current_round, total_bias_bound))
           print(f"🔁 Round {current_round + 1}: Fair-Select Acc = {accuracy:.2f}%")
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")

        # ---------------- AW-PSP branch ----------------
        # Node selection based on AW-PSP
#        awpsp_model = models.resnet18(weights=None)
        awpsp_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        awpsp_model.fc = torch.nn.Linear(awpsp_model.fc.in_features, 10)
        awpsp_model.load_state_dict(current_weights_awpsp)

        selected_awpsp, awpsp_instant_var, awpsp_cumul_var, awpsp_covered_labels, awpsp_avg_score, awpsp_labels, awpsp_KL, awpsp_unseen, awpsp_gini = \
            shared_state.topology.prioritize_available_nodes(
                awpsp_model, current_round, correlated_failures, num_clients=5, label_map=label_map
            )
        awpsp_covered_labels_log.append((current_round, len(awpsp_covered_labels)))
        selected_awpsp_log.append((current_round, selected_awpsp))

        weights_awpsp = shared_state.topology.run_federated_round(selected_awpsp, current_weights_awpsp, awpsp_model)
        if weights_awpsp is not None:
           current_weights_awpsp = weights_awpsp
           awpsp_model.load_state_dict(current_weights_awpsp)
           accuracy_awpsp = shared_state.topology.evaluate_global_model(awpsp_model, use_selected_nodes=False)
           awpsp_accuracy_log.append((current_round, accuracy_awpsp))
           awpsp_instant_fairness_log.append((current_round, awpsp_instant_var))
           awpsp_cumul_fairness_log.append((current_round, awpsp_cumul_var))
           print(f"🔁 Round {current_round + 1}: AW-PSP Acc = {accuracy_awpsp:.2f}%")
           awpsp_avg_score_log.append((current_round, awpsp_avg_score))
           awpsp_labels_log.append((current_round, awpsp_labels))
           awpsp_KL_log.append((current_round, awpsp_KL))
           awpsp_unseen_log.append((current_round, awpsp_unseen))
           awpsp_gini_log.append((current_round, awpsp_gini))
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")


        # ---------------- PSP branch ----------------
        # Use a fresh model copy so AW-PSP doesn’t pollute PSP results
        #psp_model = models.resnet18(weights=None)
        psp_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        psp_model.fc = torch.nn.Linear(psp_model.fc.in_features, 10)
        psp_model.load_state_dict(current_weights_psp)

        selected_psp, psp_instant_var, psp_cumul_var, psp_covered_labels, psp_avg_score, psp_labels, psp_KL, psp_unseen, psp_gini = \
            shared_state.topology.psp_random_selection(
                psp_model, correlated_failures, num_clients=5, label_map=label_map
            )
        psp_covered_labels_log.append((current_round, len(psp_covered_labels)))
        selected_psp_log.append((current_round, selected_psp))

        weights_psp = shared_state.topology.run_federated_round(selected_psp, current_weights_psp,psp_model)
        if weights_psp is not None:
           current_weights_psp = weights_psp
           psp_model.load_state_dict(current_weights_psp)
           accuracy_psp = shared_state.topology.evaluate_global_model(psp_model, use_selected_nodes=False)
           psp_accuracy_log.append((current_round, accuracy_psp))
           psp_instant_fairness_log.append((current_round, psp_instant_var))
           psp_cumul_fairness_log.append((current_round, psp_cumul_var))
           print(f"🔁 Round {current_round + 1}: PSP Acc = {accuracy_psp:.2f}%")
           psp_avg_score_log.append((current_round, psp_avg_score))
           psp_labels_log.append((current_round, psp_labels))
           psp_KL_log.append((current_round, psp_KL))
           psp_unseen_log.append((current_round, psp_unseen))
           psp_gini_log.append((current_round, psp_gini))
        else:
           print("⚠️ No updates received from clients. Skipping model update this round.")

        # Save logs to CSV
        with open("metrics_log.csv", "w") as f:
          writer = csv.writer(f)
          writer.writerow(["Round", "Select_Fair_Accuracy", "Select_Fair_variance", "Select_Fair_Surrogate", "AWPSP_Accuracy", "AWPSP_instant_fairness", "AWPSP_cumul_fairness","CorrelatedFailureCount", "AWPSP_CoveredLabelsCount", "PSP_Accuracy", "PSP_instant_fairness", "PSP_cumul_fairness", "PSP_CoveredLAbelsCount", "selected_awpsp", "selected_psp", "AWPSP Avg Score", "PSP Avg Score", "AWPSP labels", "PSP labels", "AWPSP KL", "PSP KL", "AWPSP unseen", "PSP unseen", "AWPSP gini", "PSP gini"])
          for i in range(num_rounds):
            print(i, accuracy_log[i][1] if i < len(accuracy_log) else None, var_u_log[i][1] if i < len(var_u_log) else None, surrogate_log[i][1] if i < len(surrogate_log) else None, awpsp_accuracy_log[i][1] if i < len(awpsp_accuracy_log) else None, awpsp_instant_fairness_log[i][1] if i < len(awpsp_instant_fairness_log) else None,  awpsp_cumul_fairness_log[i][1] if i < len(awpsp_cumul_fairness_log) else None, corr_failure_log[i][1] if i < len(corr_failure_log) else None, awpsp_covered_labels_log[i][1] if i < len(awpsp_covered_labels_log) else None, psp_accuracy_log[i][1] if i < len(psp_accuracy_log) else None, psp_instant_fairness_log[i][1] if i < len(psp_instant_fairness_log) else None, psp_cumul_fairness_log[i][1] if i < len(psp_cumul_fairness_log) else None, psp_covered_labels_log[i][1] if i < len(psp_covered_labels_log) else None, selected_awpsp_log[i][1] if i < len(selected_awpsp_log) else None, selected_psp_log[i][1] if i < len(selected_psp_log) else None, awpsp_avg_score_log[i][1] if i < len(awpsp_avg_score_log) else None, psp_avg_score_log[i][1] if i < len(psp_avg_score_log) else None, awpsp_labels_log[i][1] if i < len(awpsp_labels_log) else None, psp_labels_log[i][1] if i < len(psp_labels_log) else None, awpsp_KL_log[i][1] if i < len(awpsp_KL_log) else None, psp_KL_log[i][1] if i < len(psp_KL_log) else None, awpsp_unseen_log[i][1] if i < len(awpsp_unseen_log) else None, psp_unseen_log[i][1] if i < len(psp_unseen_log) else None, awpsp_gini_log[i][1] if i < len(awpsp_gini_log) else None, psp_gini_log[i][1] if i < len(psp_gini_log) else None)
            writer.writerow([
              awpsp_accuracy_log[i][1] if i < len(awpsp_accuracy_log) else None,
              awpsp_instant_fairness_log[i][1] if i < len(awpsp_instant_fairness_log) else None,
              awpsp_cumul_fairness_log[i][1] if i < len(awpsp_cumul_fairness_log) else None,
              corr_failure_log[i][1] if i < len(corr_failure_log) else None,
              awpsp_covered_labels_log[i][1] if i < len(awpsp_covered_labels_log) else None,
              psp_accuracy_log[i][1] if i < len(psp_accuracy_log) else None,
              psp_instant_fairness_log[i][1] if i < len(psp_instant_fairness_log) else None,
              psp_cumul_fairness_log[i][1] if i < len(psp_cumul_fairness_log) else None,
              psp_covered_labels_log[i][1] if i < len(psp_covered_labels_log) else None,
              selected_awpsp_log[i][1] if i < len(selected_awpsp_log) else None,
              selected_psp_log[i][1] if i < len(selected_psp_log) else None
        ])


def wait_for_latency_data(num_clients=3):
    print("⏳ Waiting for latency updates from clients...")
    while True:
        ready = 0
        for node, metadata in shared_state.topology.dht.table.items():
            if metadata.get("latency") is not None and metadata.get("packet_loss") is not None:
                ready += 1
        print(f"✅ Clients with latency info: {ready} / {num_clients}")
        if ready >= num_clients:
            break
        time.sleep(2)

    print("🚀 Sufficient clients reported latency. Starting training.")
    run_federated_training()



if __name__ == "__main__":
    print("🚀 Starting main server HTTP API...")


    # Step 1: Start the server in a separate thread
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8080), daemon=True).start()

    # Step 2: Start topology initialization in the background
    threading.Thread(target=initialize_topology).start()

    # Step 3: Wait for latency info from clients, then start training
    #threading.Thread(target=wait_for_latency_data).start()    
