
# topology_server.py

import sys
from flask import Flask, request
import io
import re
import requests
import threading
import copy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import hashlib
from collections import defaultdict
import time
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
import random
import json
import time
from fairness import FairnessSchedulerController, UtilityTracker, surrogate_weight
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from fairness import FairnessSchedulerController
from worker_pool import PhysicalWorkerPool, PhysicalWorkerPoolExhausted
from fl_methods import (AFLState, fedavg, normalize_records,
                        cup_importance_corrected_update,
                        php_fl_auxiliary_update, q_fedavg)


class TopologyProvider:
    def __init__(self, device_names, num_epochs, link_latency=None, link_loss=None, model_name='resnet', device_registry=None):
        self.devices = device_names 
        self.link_latency = f"{link_latency / 2}ms" if link_latency else None
        self.link_loss = (1 - np.sqrt(1 - link_loss / 100)) * 100 if link_loss else None
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.utility_definition = os.getenv("UTILITY_DEFINITION", "auc").lower()
        self.utility_tracker = UtilityTracker(
            self.utility_definition,
            max_increment=float(os.getenv("UTILITY_MAX_INCREMENT", "1.0")))
        self.utility_log = defaultdict(float)  # compatibility view of cumulative U_k(T)
        self.previous_losses = {}  # Stores last loss per client
        self.transform = self.get_transform() 
        self.fixed_indices = {}
        self.cifar_loader = self.load_cifar_data()
        self.dht = DHT(size=100)  # Initialize the DHT
        self.availability_predictor = AvailabilityPredictor(node_count=len(device_names) * len(self.devices))
        self.availability_predictor.load_history()
        self.participation_log = {}
        self.failed_nodes = []
        self.recovery_log = {worker: None for worker in self.devices}  # When a node is reintegrated
        self.node_neighbors = {} 
        self.node_losses = {}
        self.total_rounds_elapsed = 0
        self.availability_counts = defaultdict(int)
        self.last_model_states = {}  # node_id → (model_state_dict, round_number)
        self.surrogate_contributions = defaultdict(int)
        self.surrogate_staleness = defaultdict(int)
        self.device_registry = device_registry or {}
        self.physical_worker_pool = PhysicalWorkerPool()
        self.failure_correlation = defaultdict(lambda: defaultdict(set))
        self.recovery_counters = {}   # how many healthy rounds since failure
        self.probation_rounds = {}    # when node first recovered
        self.recovery_threshold = 4   # rounds to wait before marking as recovered
        self.probation_duration = 3   # rounds to weight updates less

        self.logical_labels_per_client = int(os.getenv("LOGICAL_LABELS_PER_CLIENT", "2"))
        window = int(os.getenv("AVAILABILITY_WINDOW_SIZE", "50"))
        self.selection_mode = os.getenv("SELECTION_MODE", "cup").lower()
        self.lambda_decay = float(os.getenv("LAMBDA_DECAY", "0.10"))
        self.fairness_scheduler = FairnessSchedulerController(
            self.devices, mode=self.selection_mode, window_size=window,
            seed=int(os.getenv("EXPERIMENT_SEED", "0")),
            lambda_reactive=self.lambda_decay)
        # Read-only compatibility alias; scheduling implementation lives in fairness.py.
        self.availability_estimator = self.fairness_scheduler.estimator
        self.surrogate_mode = os.getenv("SURROGATE_MODE", "accounting").lower()
        self.logical_client_ids = ()
        self.afl_state = None
        self.last_method_diagnostics = {}
        self.last_client_records = []
        self.cup_round_context = {}

    def configure_logical_clients(self, client_ids):
        """Install full-population state required by stateful server methods."""
        client_ids = tuple(client_ids)
        if client_ids != self.logical_client_ids:
            self.logical_client_ids = client_ids
            self.afl_state = AFLState(client_ids)

    def set_cup_round_context(self, context):
        self.cup_round_context = dict(context)


    def get_subset_indices(self, worker_name, dataset, subset_size=1000, seed=42):
        """
        Return non-IID training data indices per worker (by label).
        """
        from torchvision.datasets import CIFAR10
        import numpy as np

        # Extract numeric index from device name, works for 'hX' or 'Device_X'
        index = int(worker_name.replace("Device_", ""))

        # get all indices
#        all_indices = list(range(total_size))

        # Assign 2 labels per client (you can change this)
        num_labels_per_worker = 2
        total_labels = 10

        start = (index * num_labels_per_worker) % total_labels
        worker_labels = list(range(total_labels))[start:start + num_labels_per_worker]

        # Get all indices that belong to the assigned labels
#        label_indices = np.where(np.isin(all_indices, worker_labels))[0]
        labels_array = np.array(dataset.targets)
        label_indices = np.where(np.isin(labels_array, worker_labels))[0]

        # deterministic shuffle
        rng = np.random.RandomState(seed + index)
        rng.shuffle(label_indices)

        print(f"[DEBUG] {worker_name} first indices:", label_indices[:5])
        print(f"[DEBUG] {worker_name} labels in subset:", [dataset.targets[i] for i in label_indices[:20]])

        self.fixed_indices[worker_name] = label_indices[:subset_size]

        return label_indices[:subset_size]



    def get_subset_indices1(self, worker_name, total_size, subset_size=1000, seed=42):
        # Make subset selection deterministic per worker
        index = int(worker_name.replace("Device_", ""))
        random.seed(seed + index)
        all_indices = list(range(total_size))
        random.shuffle(all_indices)
        return all_indices[:subset_size]


    def load_dnn_model(self, train_loader, model = None, model_weights=None):
        """Load the DNN model for failure prediction."""

        # Only train the final fully connected layer
        model.fc.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, 10)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # If weights were passed from coordinator
        if model_weights is not None:
           model.load_state_dict(model_weights)

        # Use CrossEntropyLoss for classification
        criterion = torch.nn.CrossEntropyLoss()
        # Reinitialize optimizer for only the classifier layer
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

        # Training loop
        for epoch in range(self.num_epochs):  # Number of epochs
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for i, (inputs, labels) in enumerate(train_loader):  # Assuming train_loader is your DataLoader
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
                total_predictions += labels.size(0)  # Total predictions

            # Calculate average loss and accuracy for the epoch
            avg_loss = running_loss / len(train_loader)
            accuracy = (correct_predictions / total_predictions) * 100

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return model        


    def get_trained_model(self):
        """Return the trained model for use by each host."""
        return self.dnn_model

    def get_transform(self):
        """Get the transform needed for CIFAR-10."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2430, 0.2610])
        ])
    
    def load_cifar_data(self):
        """Load the CIFAR-10 dataset."""
        full_dataset = datasets.CIFAR10(root='data/', train=True, download=True, transform=self.transform)

        self.label_map = {}
        self.dataloaders = {}

        for worker_name in self.devices:
            indices = self.get_subset_indices(worker_name, full_dataset, subset_size=1000)
            filtered_dataset = torch.utils.data.Subset(full_dataset, indices)

            # Store dataloader per worker
            self.dataloaders[worker_name] = DataLoader(filtered_dataset, batch_size=32, shuffle=False)

            # Record which labels are present in this worker's dataset
            label_set = set()
            for i in indices:
                _, label = full_dataset[i]
                label_set.add(label)

            self.label_map[worker_name] = sorted(label_set)
            print(f"📦 {worker_name} assigned labels: {sorted(label_set)}")

        return self.dataloaders  # Optionally return if you want



    def send_weights_to_client(
        self,
        device_id,
        global_weights,
        max_retries=None,
        sync_only=False,
        logical_id=None,
        logical_labels_per_client=None,
    ):

        max_retries = 3 if max_retries is None else max(0, int(max_retries))
        # Logical IDs must be routed through run_logical_federated_round, which
        # maps them onto registered physical workers. Avoid an opaque KeyError if
        # an older caller accidentally sends a logical ID directly.
        entry = self.device_registry.get(device_id)
        if entry is None:
            print(
                f"❌ Cannot contact unregistered physical client {device_id!r}. "
                "Use run_logical_federated_round for logical client IDs."
            )
            return None
        ip = entry["ip"]
        port = entry["port"]

        # POST to client
        url = f"http://{ip}:{port}/train"

        for attempt in range(max_retries):
            try:
               print(f"📤 Sending weights to {device_id} at {url} (attempt {attempt + 1})")
               # 💡 Recreate the buffer and files inside the loop!
               buffer = io.BytesIO()
               torch.save(global_weights, buffer)
               buffer.seek(0)
               files = {"weights": ("model.pth", buffer)}
               form = {
                   "sync_only": str(bool(sync_only)),
                   "logical_id": "" if logical_id is None else str(logical_id),
                   "logical_labels_per_client": (
                       "" if logical_labels_per_client is None
                       else str(logical_labels_per_client)),
                   "method": os.getenv("SELECTOR_MODE", "fedavg_random"),
                   "fedprox_mu": os.getenv("FEDPROX_MU", "0.01"),
               }

               timeout = float(os.getenv("CLIENT_TRAIN_TIMEOUT_SECONDS", "600"))
               # Send exactly one request.  The former second POST reused the
               # already-consumed BytesIO object, omitted the logical-client
               # form fields, and overwrote the successful first response.
               # Consequently every logical update was commonly discarded and
               # all accuracy/utility metrics stayed empty or zero.
               response = requests.post(
                   url, files=files, data=form, timeout=timeout)


               if response.status_code == 200:
                  return torch.load(io.BytesIO(response.content), map_location="cpu")
               else:
                  print(f"⚠️  {device_id} unavailable (status {response.status_code}), retrying...")
            except Exception as e:
               print(f"❌ Error contacting {device_id}: {e}, retrying...")
            time.sleep(2)

        print(f"❌ Client {device_id} at {ip}:{port} failed after {max_retries} retries. Skipping.")
        return None


    def resolve_pod_url(self, node_name):
        namespace = "fl-simulation"
        pod_dns = f"http://{node_name}.{namespace}.svc.cluster.local:5000/train"
        return pod_dns


    def configure_physical_worker_pool(self, physical_ids, active_count):
        """Split healthy registered workers into stable active and standby pools."""
        self.physical_worker_pool.configure(physical_ids, active_count)
        print(
            f"🛡️ Physical worker pool: "
            f"active={self.physical_worker_pool.active_snapshot()}, "
            f"standby={self.physical_worker_pool.standby}"
        )



    def active_physical_worker_snapshot(self):
        return self.physical_worker_pool.active_snapshot()

    def quarantine_and_promote(self, failed_worker):
        """Atomically replace a failed active worker with the next standby."""
        replacement = self.physical_worker_pool.quarantine_and_promote(failed_worker)
        print(
            f"🔄 Quarantined failed physical worker {failed_worker}; "
            f"promoted standby {replacement}."
        )
        return replacement

    def run_logical_federated_round(
            self, logical_ids, physical_ids, global_weights,
            per_client_timeout=30, active_count=None):
        updated_weights = []
        if not self.active_physical_worker_snapshot():
            self.configure_physical_worker_pool(
                physical_ids,
                len(physical_ids) if active_count is None else active_count,
            )

        def train_logical_on_physical(logical_id, physical_id):
            while True:
                print(f"Sending logical client {logical_id} to physical {physical_id}")
                client_weights = self.send_weights_to_client(
                    physical_id,
                    global_weights,
                    max_retries=1,
                    sync_only=False,
                    logical_id=logical_id,
                    logical_labels_per_client=self.logical_labels_per_client,
                )
                if client_weights is not None:
                    client_weights["client_id"] = logical_id
                    return client_weights
                print(
                    f"⚠️ Logical client {logical_id} failed via {physical_id}; "
                    "promoting a standby and retrying."
                )
                physical_id = self.quarantine_and_promote(physical_id)

        active_workers = self.active_physical_worker_snapshot()
        wave_size = len(active_workers)
        for wave_start in range(0, len(logical_ids), wave_size):
            wave = logical_ids[wave_start:wave_start + wave_size]
            active_workers = self.active_physical_worker_snapshot()
            with ThreadPoolExecutor(max_workers=wave_size) as executor:
                futures = []
                for idx, logical_id in enumerate(wave):
                    physical_id = active_workers[idx % wave_size]
                    futures.append(executor.submit(train_logical_on_physical, logical_id, physical_id))

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        updated_weights.append(result)

        return self.aggregate_weights(updated_weights, global_weights)



    def run_federated_round(self, selected_hosts, global_weights, model=None):
        # Backward compatibility for the former call shape
        # run_federated_round(logical_ids, physical_ids, global_weights).  Without
        # this adapter an old server sends logical IDs such as h40 to the physical
        # registry and fails with KeyError.
        if is_legacy_logical_round_call(global_weights, model):
            print(
                "⚠️ Deprecated logical-round call detected; routing logical "
                "clients through registered physical workers."
            )
            return self.run_logical_federated_round(
                list(selected_hosts), list(global_weights), model
            )

        updated_weights = []
        print("selected_hosts: ", selected_hosts)
        for host in selected_hosts:
            client_weights = self.send_weights_to_client(host, global_weights)
            if client_weights is not None:
                updated_weights.append(client_weights)

        return self.aggregate_weights(updated_weights, global_weights)






    def aggregate_weights(self, weight_list, global_weights=None):
        if not weight_list:
           print("⚠️ No weights to aggregate (no clients participated this round).")
           self.last_client_records = []
           if os.getenv("SELECTOR_MODE", "").lower() == "select_fair_nodes":
               return copy.deepcopy(global_weights)
           return None
        method = os.getenv("SELECTOR_MODE", "fedavg_random").lower()
        records = normalize_records(weight_list)
        self.last_client_records = records
        if method == "select_fair_nodes":
            return cup_importance_corrected_update(
                global_weights, records, self.cup_round_context,
                len(self.logical_client_ids),
                eps=float(os.getenv("CUP_AGGREGATION_EPS", "1e-12")),
                correction_clip=float(os.getenv("CUP_AGGREGATION_CLIP", "100")),
            )
        if method == "q_ffl":
            if global_weights is None:
                raise ValueError("q-FFL requires the current global model")
            result, self.last_method_diagnostics = q_fedavg(global_weights, records)
            print(f"q-FFL state: {self.last_method_diagnostics}")
            return result
        if method == "afl":
            if global_weights is None or self.afl_state is None:
                raise ValueError("AFL requires configured logical clients and global weights")
            result, self.last_method_diagnostics = self.afl_state.update(global_weights, records)
            print(f"AFL state: {self.last_method_diagnostics}")
            return result
        if method == "php_fl":
            result, self.last_method_diagnostics = php_fl_auxiliary_update(records)
            print(f"PHP-FL auxiliary state: {self.last_method_diagnostics}")
            return result
        # FairFedCS changes selection, then uses standard FedAvg.
        weighted_result = fedavg(records)
        if method != "fedfv":
            return weighted_result
        weighted = [(record["state_dict"], float(record.get("sample_count", 1)))
                    for record in records]
        total_weight = sum(weight for _, weight in weighted)
        weighted = [(state, weight / total_weight) for state, weight in weighted]
        # FedFV projects conflicting client model deltas before averaging.
        if method == "fedfv" and global_weights is not None:
            deltas = [{key: state[key] - global_weights[key] for key in state
                       if torch.is_floating_point(state[key])} for state, _ in weighted]
            for i in range(len(deltas)):
                for j in range(len(deltas)):
                    if i == j: continue
                    dot = sum((deltas[i][key] * deltas[j][key]).sum() for key in deltas[i])
                    norm = sum((value * value).sum() for value in deltas[j])
                    if dot < 0 and norm > 0:
                        coefficient = dot / norm
                        for key in deltas[i]: deltas[i][key] -= coefficient * deltas[j][key]
            weighted = [({**state, **{key: global_weights[key] + delta[key] for key in delta}}, weight)
                        for (state, weight), delta in zip(weighted, deltas)]
        new_state = copy.deepcopy(weighted[0][0])
        for key in new_state:
            if torch.is_floating_point(new_state[key]):
               new_state[key] = sum(
                   state[key] * weight for state, weight in weighted)
            else:
               # Non-float types (e.g. LongTensor): just copy the first one
               new_state[key] = weighted[0][0][key]
        return new_state  




    def evaluate_global_model(self, model, selected_nodes=None, subset_size=1000, use_selected_nodes=True):
        correct = total = 0
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if use_selected_nodes and selected_nodes:
            full_dataset = datasets.CIFAR10(root='data/', train=True, download=False, transform=self.transform)

            # Combine subsets from selected nodes
            combined_datasets = []
            for node in selected_nodes:
                subset = torch.utils.data.Subset(full_dataset, self.fixed_indices[node])
                combined_datasets.append(subset)
            eval_dataset = ConcatDataset(combined_datasets)
        else:
            eval_dataset = datasets.CIFAR10(root='data/', train=False, download=False, transform=self.transform)

        test_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for images, labels in test_loader:
               outputs = model(images)
               _, predicted = torch.max(outputs, 1)
               correct += (predicted == labels).sum().item()
               total += labels.size(0)
        accuracy = 100 * correct / total

        return accuracy


    def evaluate_per_client_accuracy(self, model, nodes):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        full_dataset = datasets.CIFAR10(root='data/', train=True, download=False, transform=self.transform)
        per_client_acc = {}

        with torch.no_grad():
            for node in nodes:
                indices = self.fixed_indices.get(node, [])
                if len(indices) == 0:
                    per_client_acc[node] = None
                    continue
                subset = torch.utils.data.Subset(full_dataset, indices)
                loader = DataLoader(subset, batch_size=32, shuffle=False)
                correct = total = 0
                for images, labels in loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                per_client_acc[node] = (100 * correct / total) if total else None

        return per_client_acc

    def evaluate_logical_client_accuracy(self, model, label_map):
        """Evaluate every logical client with one shared CIFAR-10 test pass."""
        model.eval()
        dataset = datasets.CIFAR10(
            root='data/', train=False, download=False, transform=self.transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        correct = defaultdict(int); totals = defaultdict(int)
        with torch.no_grad():
            for images, labels in loader:
                predictions = model(images).argmax(dim=1)
                for label in range(10):
                    mask = labels == label
                    totals[label] += int(mask.sum())
                    correct[label] += int((predictions[mask] == labels[mask]).sum())
        return {
            client_id: (100.0 * sum(correct[label] for label in client_labels)
                        / max(1, sum(totals[label] for label in client_labels)))
            for client_id, client_labels in label_map.items()
        }


    def get_freshness(self, node, current_round):
        """Returns how long since this node was last selected."""
        last_selected_round = self.participation_log.get(node)
        if last_selected_round is None or not last_selected_round:
           freshness = current_round  # Never participated
        else:
           latest_round = max(last_selected_round)
           freshness = current_round - latest_round

        print(f"{node}: last_selected_round={last_selected_round}, current_round={current_round}, freshness={freshness}")
        return freshness


    def update_participation_log(self, selected_nodes, current_round):
        """Update participation log for fairness tracking."""
        for node in selected_nodes:
            print("node: ", node)
            if node not in self.participation_log:
                self.participation_log[node] = []
            self.participation_log[node].append(current_round)  # Log current round



    def get_correlated_failure(self, current_round, availability_vectors, corr_threshold=0.6, num_neighbors=4):
        print("corr_threshold: ", corr_threshold)
   
        # 1. Compute trace-based correlation matrix
        device_ids = list(availability_vectors.keys())
        # Pearson correlation is undefined for a constant trace.  NumPy emits
        # RuntimeWarning and returns NaN in that case; treat undefined pairs as
        # uncorrelated so they cannot become false correlated-failure edges.
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = np.corrcoef([availability_vectors[d] for d in device_ids])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(matrix, 1.0)
        device_idx = {f"h{int(device[1:]) - 1}": idx for idx, device in enumerate(device_ids)}
        print("device_idx :", device_idx)
        # 2. Compute proximity-based neighbors
        node_latencies = {
           node: metadata["latency"]
           for node, metadata in self.dht.table.items()
           if "latency" in metadata
        }


        self.node_neighbors = {}
        for node, latency in node_latencies.items():
            filtered = [(other_node, other_latency)
                    for other_node, other_latency in node_latencies.items()
                    if other_node != node]
            print(f"node {node} has latency: ", latency)
            print("filtered: ", filtered)
            sorted_neighbors = sorted(filtered, key=lambda x: abs(x[1] - latency))
            self.node_neighbors[node] = [n for n, _ in sorted_neighbors[:num_neighbors]]
 
        # 3. Update failure_correlation structure
        for node in self.failed_nodes:
            neighbors = self.node_neighbors.get(node, [])
            for neighbor in neighbors:
                if neighbor in self.failed_nodes:
                   if current_round not in self.failure_correlation[node][neighbor]:
                      self.failure_correlation[node][neighbor].add(current_round)
                   if current_round not in self.failure_correlation[neighbor][node]:
                      self.failure_correlation[neighbor][node].add(current_round)
        
        # 4. Detect current failed nodes
        print("🔎 Starting Failure Detection using latency & packet loss")
        for node, metadata in self.dht.table.items():
            latency = metadata.get("latency")
            loss = metadata.get("packet_loss")
            if latency is not None and loss is not None:
               if latency > 90 or loss >= 40:
                  if node not in self.failed_nodes:
                     print(f"❌ Node {node} failed: latency={latency}, loss={loss}")
                     self.failed_nodes.append(node)
               else:
                  # increment recovery counter
                  self.recovery_counters[node] = self.recovery_counters.get(node, 0) + 1
                  if node in self.failed_nodes and self.recovery_counters[node] >= self.recovery_threshold:
                     print(f"   ^|^e Node {node} recovered after {self.recovery_counters[node]} healthy rounds")
                     self.failed_nodes.remove(node)

                     # put node into probation phase
#                     self.probation_rounds[node] = 0

        # 5. Use trace correlation and failure correlation to detect groups
        correlated_failures = []
        for node in self.failed_nodes:
            neighbors = self.node_neighbors.get(node, [])
            for neighbor in neighbors:
#                if neighbor in self.failed_nodes:
                   normalized_node = node.lower().replace("device_", "h")
                   normalized_neighbor = neighbor.lower().replace("device_", "h")
                   print(f"device_idx[{normalized_node}]: ", device_idx[normalized_node])
                   trace_score = matrix[device_idx[normalized_node]][device_idx[normalized_neighbor]] if normalized_node in device_idx and normalized_neighbor in device_idx else 0                   
                   fail_score = len(self.failure_correlation[node][neighbor]) / (current_round + 1) if neighbor in self.failure_correlation[node] else 0
                   print(f"for node {node} and neighbor {neighbor}, trace_score is {trace_score} and fail_score is {fail_score}")
                   if trace_score >= corr_threshold or fail_score >= corr_threshold:
                       correlated_failures.append((node, neighbor))

        if correlated_failures:
           print("⚠️ Correlated failures detected:", correlated_failures)
        else:
           print("✅ No correlated failures detected.")


        return correlated_failures





    def select_fair_nodes(self, model, current_round, correlated_failures, label_map, num_clients,
                              corr_threshold=0.35, lambda_=0.5, epsilon=1e-5):
        raise RuntimeError(
            "Legacy availability-filtered CUP selection is disabled; use "
            "CumulativeUtilityParity in the logical round coordinator so A, S, "
            "and P remain distinct."
        )




# Distributed Hash Table
class DHT:
    def __init__(self, size=100):
        self.table = {}
        self.size = size

    def _hash(self, key):
        key = str(key)  # Ensure key is always string
        return int(hashlib.sha1(key.encode()).hexdigest(), 16) % self.size

    def store(self, key, value):
        h = self._hash(key)
        self.table[h] = value

    def lookup(self, key):
        h = self._hash(key)
        return self.table.get(h, None)

    def all_nodes(self):
        return list(self.table.keys())

# Availability Predictor
class AvailabilityPredictor:
    def __init__(self, node_count, window_size=5, history_file="/tmp/availability_history.json"):
        self.window_size = window_size
        self.history = {
            f'node_{i}': {"comp": [], "comm": []} for i in range(node_count)
        }
        self.beta = {f'node_{i}': 0.5 for i in range(node_count)}  # Default return probability
        self.history_file = history_file
        self.load_history()

    def update(self, node, success_comp, success_comm):
        """Predict node availability using historical data and return probability."""
        if node not in self.history:
            self.history[node] = {"comp": [], "comm": []}

        # Append new values **before** writing to file
        self.history[node]["comp"].append(success_comp)
        self.history[node]["comm"].append(success_comm)
        
        # Keep only last T rounds
        if len(self.history[node]["comp"]) > self.window_size:
            self.history[node]["comp"].pop(0)
        if len(self.history[node]["comm"]) > self.window_size:
            self.history[node]["comm"].pop(0)

        with open(self.history_file, "w") as f:
            json.dump(self.history, f)


    def predict(self, node):
        """Predict node availability using historical data and return probability."""
        if node not in self.history:
            self.history[node] = {"comp": [], "comm": []}
        history_comp = self.history[node]["comp"]
        history_comm = self.history[node]["comm"]
        if not history_comp or not history_comm:
            return 0.5

        # Compute availability from historical data
        a_comp = sum(self.history[node]["comp"]) / len(self.history[node]["comp"]) if self.history[node]["comp"] else 0
        a_comm = sum(self.history[node]["comm"]) / len(self.history[node]["comm"]) if self.history[node]["comm"] else 0
        a_i = a_comp * a_comm  # Overall availability
        print(" a_i: ", a_i)
        # Compute future availability
        future_a_i = a_i + (1 - a_i) * self.beta.get(node, 0.5)
        print("future_a_i : ", future_a_i)
        return future_a_i

    def calculate_neighbor_availability(node, topology):
        neighbors = topology.get_neighbors(node)
        availabilities = []

        for neighbor_id in neighbors:
             neighbor_data = topology.dht.lookup(neighbor_id)
             if not neighbor_data:
                   continue

             availabilities.append(neighbor_data['availability'])

        if not availabilities:
             return 0  # No neighbors = assume 0 (or safe default like 0.5)

        return sum(availabilities) / len(availabilities)

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                 with open(self.history_file, "r") as f:
                      self.history = json.load(f)
            except json.JSONDecodeError:
                 print(f"⚠️ History file corrupted. Starting fresh.")
                 self.history = {}
        else:
            self.history = {}
