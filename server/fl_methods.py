"""Stateful server algorithms kept separate from client scheduling plumbing."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

import torch


def normalize_records(updates):
    """Normalize legacy responses without probing arbitrary objects as mappings."""
    records = []
    for update in updates:
        if isinstance(update, dict) and "state_dict" in update:
            records.append(update)
        elif isinstance(update, tuple) and len(update) == 2:
            records.append({"state_dict": update[0], "sample_count": update[1]})
        elif isinstance(update, dict):
            records.append({"state_dict": update, "sample_count": 1})
        else:
            raise TypeError(f"Unsupported client update record: {type(update).__name__}")
    return records


def fedavg(records):
    total = sum(max(0.0, float(record.get("sample_count", 1))) for record in records)
    if not records or total <= 0:
        return None
    result = copy.deepcopy(records[0]["state_dict"])
    for key in result:
        if torch.is_floating_point(result[key]):
            result[key] = sum(
                record["state_dict"][key]
                * (float(record.get("sample_count", 1)) / total)
                for record in records)
    return result


def php_fl_auxiliary_update(records):
    """Aggregate only successful DEAL auxiliary-model submissions."""
    if any(record.get("record_type") != "php_fl_auxiliary" for record in records):
        raise TypeError("PHP-FL server update accepts only PHP auxiliary records")
    result = fedavg(records)
    diagnostics = {
        "active_clients": [record.get("client_id") for record in records],
        "auxiliary_updates": len(records),
        "mean_mask_fraction": sum(
            float(record.get("php_diagnostics", {}).get("mask_fraction", 0.0))
            for record in records) / len(records),
    }
    return result, diagnostics


def cup_importance_corrected_update(global_weights, records, context, population_size,
                                    *, eps=1e-12, correction_clip=100.0):
    """Horvitz-Thompson CUP delta for the uniform-client target objective.

    Scheduling inclusion and telemetry availability estimates affect only this
    correction; CUP scheduler scores are never aggregation coefficients.
    """
    result = copy.deepcopy(global_weights)
    if not records:
        return result
    for key in result:
        if not torch.is_floating_point(result[key]):
            continue
        correction = torch.zeros_like(result[key])
        for record in records:
            client_id = record["client_id"]
            inclusion = float(context[client_id]["selection_probability"])
            availability = float(context[client_id]["availability_estimate"])
            participation_probability = max(inclusion * availability, eps)
            multiplier = min(1.0 / (population_size * participation_probability),
                             correction_clip)
            correction.add_(record["state_dict"][key] - global_weights[key],
                            alpha=multiplier)
        candidate = global_weights[key] + correction
        if torch.isfinite(candidate).all():
            result[key] = candidate
    return result


def q_fedavg(global_weights, records, *, q=None, lipschitz=None, eps=None):
    """q-FedAvg server solver from Li et al., Algorithm 1.

    ``loss_at_global`` is F_k(w_t), measured before local optimization.  The
    explicit ``Q_FFL_L`` is the algorithm's Lipschitz-related L; it is not
    inferred from the client optimizer learning rate.
    """
    q = float(os.getenv("Q_FFL_Q", "1")) if q is None else float(q)
    lipschitz = float(os.getenv("Q_FFL_L", "1")) if lipschitz is None else float(lipschitz)
    eps = float(os.getenv("Q_FFL_EPS", "1e-12")) if eps is None else float(eps)
    if q < 0 or lipschitz <= 0 or eps <= 0:
        raise ValueError("q, Q_FFL_L, and Q_FFL_EPS must be non-negative, positive, and positive")

    numerator = {key: torch.zeros_like(value) for key, value in global_weights.items()
                 if torch.is_floating_point(value)}
    denominator = 0.0
    diagnostics = []
    for record in records:
        loss = max(float(record["loss_at_global"]), eps)
        if not math.isfinite(loss):
            continue
        delta = {key: global_weights[key] - record["state_dict"][key]
                 for key in numerator}
        norm_sq = sum(float(value.double().pow(2).sum()) for value in delta.values())
        loss_q = loss ** q
        # q-FedAvg Algorithm 1: Delta_k=F_k(w_t)^q(w_t-w_k),
        # h_k=q F_k(w_t)^(q-1)||w_t-w_k||^2 + L F_k(w_t)^q.
        h_k = q * loss ** (q - 1) * norm_sq + lipschitz * loss_q
        if not math.isfinite(h_k) or h_k < 0:
            continue
        for key in numerator:
            numerator[key].add_(delta[key], alpha=loss_q)
        denominator += h_k
        diagnostics.append({"client_id": record.get("client_id"), "loss": loss,
                            "delta_norm": math.sqrt(norm_sq), "h": h_k})

    if not math.isfinite(denominator) or denominator <= eps:
        return copy.deepcopy(global_weights), {"denominator": denominator, "clients": diagnostics}
    result = copy.deepcopy(global_weights)
    for key, delta_sum in numerator.items():
        candidate = global_weights[key] - delta_sum / denominator
        if not torch.isfinite(candidate).all():
            return copy.deepcopy(global_weights), {"denominator": denominator, "clients": diagnostics}
        result[key] = candidate
    return result, {"denominator": denominator, "clients": diagnostics}


@dataclass
class AFLState:
    """Persistent adversarial mixture for partial-participation AFL."""

    client_ids: tuple[str, ...]
    lambda_lr: float = field(default_factory=lambda: float(os.getenv("AFL_LAMBDA_LR", ".1")))
    model_lr: float = field(default_factory=lambda: float(os.getenv("AFL_MODEL_LR", "1")))
    lambdas: dict[str, float] = field(init=False)
    observed_losses: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.client_ids or self.lambda_lr < 0 or self.model_lr <= 0:
            raise ValueError("AFL requires clients, non-negative lambda LR, and positive model LR")
        configured = os.getenv("AFL_LAMBDA_INIT", "uniform").strip()
        if configured == "uniform":
            values = [1.0 / len(self.client_ids)] * len(self.client_ids)
        else:
            values = [float(value) for value in configured.split(",")]
            if (len(values) != len(self.client_ids) or any(value < 0 for value in values)
                    or not math.isclose(sum(values), 1.0, abs_tol=1e-9)):
                raise ValueError("AFL_LAMBDA_INIT must be 'uniform' or a full simplex vector")
        self.lambdas = dict(zip(self.client_ids, values))

    def update(self, global_weights, records):
        participants = [record for record in records if record.get("client_id") in self.lambdas]
        if not participants:
            return copy.deepcopy(global_weights), self.diagnostics({})
        before = dict(self.lambdas)
        selected_mass = sum(before[record["client_id"]] for record in participants)
        result = copy.deepcopy(global_weights)
        if selected_mass > 0:
            for key in result:
                if torch.is_floating_point(result[key]):
                    direction = sum(
                        (record["state_dict"][key] - global_weights[key])
                        * (before[record["client_id"]] / selected_mass)
                        for record in participants)
                    result[key] = global_weights[key] + self.model_lr * direction

        # AFL stochastic exponentiated-gradient ascent. Only observed clients
        # receive a loss gradient; nonparticipants receive no fabricated loss.
        losses = {}
        for record in participants:
            client_id = record["client_id"]
            loss = float(record["loss_at_global"])
            if math.isfinite(loss):
                losses[client_id] = loss
                self.observed_losses[client_id] = loss
                self.lambdas[client_id] *= math.exp(self.lambda_lr * loss)
        total = sum(self.lambdas.values())
        self.lambdas = {client_id: value / total for client_id, value in self.lambdas.items()}
        return result, self.diagnostics(losses)

    def diagnostics(self, current_losses):
        entropy = -sum(value * math.log(max(value, 1e-300)) for value in self.lambdas.values())
        leader = max(self.lambdas, key=self.lambdas.get)
        objective = sum(self.lambdas[k] * loss for k, loss in current_losses.items())
        return {"lambda_min": min(self.lambdas.values()), "lambda_max": max(self.lambdas.values()),
                "lambda_sum": sum(self.lambdas.values()), "lambda_entropy": entropy,
                "highest_lambda_client": leader, "weighted_afl_objective": objective,
                "observed_clients": sorted(current_losses)}


@dataclass
class FairFedCSState:
    """Persistent Lyapunov client-selection state for FairFedCS."""

    client_ids: tuple[str, ...]
    capacity: int
    sigma: float = field(default_factory=lambda: float(os.getenv("FAIRFEDCS_SIGMA", "1")))
    reputation_decay: float = field(default_factory=lambda: float(os.getenv("FAIRFEDCS_REPUTATION_DECAY", ".9")))
    queues: dict[str, float] = field(init=False)
    reputations: dict[str, float] = field(init=False)
    participation: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.queues = {client_id: 0.0 for client_id in self.client_ids}
        self.reputations = {client_id: 0.0 for client_id in self.client_ids}
        self.participation = {client_id: 0 for client_id in self.client_ids}

    def suitability(self, client_id):
        # FairFedCS client suitability index: Psi_i(t)=sigma*r_i(t)+Q_i(t).
        return self.sigma * self.reputations[client_id] + self.queues[client_id]

    def select(self, available, count):
        return sorted(available, key=lambda client_id: (-self.suitability(client_id), client_id))[:count]

    def on_round_end(self, selected, records):
        selected = set(selected)
        target = min(self.capacity, len(self.client_ids)) / len(self.client_ids)
        contributions = {}
        for record in records:
            start = float(record.get("loss_at_global", record.get("loss", 0.0)))
            finish = float(record.get("loss", start))
            contributions[record["client_id"]] = max(0.0, start - finish)
        for client_id in self.client_ids:
            indicator = float(client_id in selected)
            # Lyapunov virtual queue: Q_i(t+1)=[Q_i(t)+target-x_i(t)]_+.
            self.queues[client_id] = max(0.0, self.queues[client_id] + target - indicator)
            if client_id in contributions:
                contribution = contributions[client_id]
                self.reputations[client_id] = (
                    self.reputation_decay * self.reputations[client_id]
                    + (1 - self.reputation_decay) * contribution)
                self.participation[client_id] += 1

