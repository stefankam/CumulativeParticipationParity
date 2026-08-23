"""Cumulative Utility Parity state machine.

Availability (A), scheduler intent (S), and realized participation (P=A*S)
are deliberately represented and logged independently.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CUPClientState:
    availability_count: int = 0
    selection_count: int = 0
    participation_count: int = 0
    availability_estimate: float = 0.0
    utility: float = 0.0
    normalized_utility: float = 0.0
    participation_debt: int = 0
    previous_participation_accuracy: float | None = None
    previous_loss: float | None = None
    last_real_participation_round: int | None = None
    last_real_utility_increment: float = 0.0
    surrogate_utility: float = 0.0


def oracle_maxmin_rates(pi_hat, mu_hat, budget, eps=1e-12):
    """Lemma 2 canonical max-min conditional selection rates."""
    clients = tuple(pi_hat)
    if not clients or not 0 <= budget <= sum(pi_hat.values()) + eps:
        raise ValueError("budget must be in [0, sum(pi_hat)]")
    if any(pi_hat[k] <= 0 or mu_hat[k] <= 0 for k in clients):
        raise ValueError("oracle pi and mu must be strictly positive")
    tau = min(min(mu_hat.values()), budget / sum(pi_hat[k] / mu_hat[k] for k in clients))
    return tau, {k: tau / mu_hat[k] for k in clients}


def fixed_size_inclusion_probabilities(weights, capacity):
    """Water-fill positive priorities into marginals that sum to capacity."""
    keys = tuple(weights)
    capacity = min(max(int(capacity), 0), len(keys))
    if capacity == 0:
        return {key: 0.0 for key in keys}
    positive = {key: max(0.0, float(weights[key])) for key in keys}
    if sum(positive.values()) == 0:
        positive = {key: 1.0 for key in keys}
    remaining, result, budget = set(keys), {}, float(capacity)
    while remaining:
        scale = budget / sum(positive[key] for key in remaining)
        saturated = {key for key in remaining if scale * positive[key] >= 1.0}
        if not saturated:
            result.update({key: scale * positive[key] for key in remaining})
            break
        for key in saturated:
            result[key] = 1.0
        budget -= len(saturated)
        remaining -= saturated
    return {key: result.get(key, 0.0) for key in keys}


def dependent_round_sample(inclusion_probabilities, rng):
    """Pivotal dependent rounding with exact fixed size and given marginals."""
    values = dict(inclusion_probabilities)
    fractional = lambda: [key for key, value in values.items() if 1e-12 < value < 1 - 1e-12]
    while len(fractional()) >= 2:
        left, right = fractional()[:2]
        alpha = min(1 - values[left], values[right])
        beta = min(values[left], 1 - values[right])
        if rng.random() < beta / (alpha + beta):
            values[left] += alpha; values[right] -= alpha
        else:
            values[left] -= beta; values[right] += beta
    return [key for key, value in values.items() if value >= 1 - 1e-9]


class CumulativeUtilityParity:
    """Paper-aligned CUP accounting, scheduling, and surrogate state."""

    LOG_FIELDS = (
        "round", "client_id", "available", "selected", "participated",
        "availability_estimate", "availability_count", "selection_count",
        "participation_count", "utility_metric", "utility_increment", "utility",
        "normalized_utility", "participation_debt", "selection_score",
        "selection_probability", "availability_inverse_clipped",
        "surrogate_used", "surrogate_staleness",
        "surrogate_weight", "surrogate_contribution",
    )

    def __init__(self, client_ids, capacity, *, seed=0, output_path=None):
        self.client_ids = tuple(client_ids)
        self.capacity = min(int(capacity), len(self.client_ids))
        self.scheduler_mode = os.getenv("CUP_SCHEDULER", "reactive").lower()
        self.utility_metric = os.getenv("CUP_UTILITY_METRIC", "accuracy_gain").lower()
        self.utility_max_increment = float(os.getenv("CUP_UTILITY_MAX_INCREMENT", "1.0"))
        if self.utility_max_increment <= 0:
            raise ValueError("CUP_UTILITY_MAX_INCREMENT must be positive")
        self.epsilon = float(os.getenv("CUP_EPSILON", "1e-3"))
        self.inverse_clip = float(os.getenv("CUP_INVERSE_AVAILABILITY_CLIP", "100"))
        self.cup_debt_coefficient = float(os.getenv("CUP_DEBT_LAMBDA", ".1"))
        self.alpha = {client_id: 1.0 for client_id in self.client_ids}
        self.surrogate_enabled = os.getenv("CUP_SURROGATE", "false").lower() in {"1", "true", "yes", "on"}
        self.surrogate_mode = os.getenv("CUP_SURROGATE_MODE", "utility_only").lower()
        if self.surrogate_mode != "utility_only":
            raise ValueError("Only empirical CUP_SURROGATE_MODE=utility_only is implemented")
        self.surrogate_eta0 = float(os.getenv("CUP_SURROGATE_ETA0", "1"))
        self.surrogate_decay = float(os.getenv("CUP_SURROGATE_DECAY", ".1"))
        self.states = {client_id: CUPClientState() for client_id in self.client_ids}
        self.rng = random.Random(seed)
        self.last_selection_probabilities = {client_id: 0.0 for client_id in self.client_ids}
        self.last_scores = {client_id: 0.0 for client_id in self.client_ids}
        self.last_clipped = {client_id: False for client_id in self.client_ids}
        self.output_path = Path(output_path or os.getenv("CUP_ROUND_LOG_PATH", "cup_rounds.csv"))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.LOG_FIELDS).writeheader()
        config_path = Path(os.getenv(
            "CUP_RUN_CONFIG_PATH", str(self.output_path.with_suffix(".config.json"))))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "method": os.getenv("SELECTOR_MODE", "select_fair_nodes"),
            "seed": seed,
            "availability_trace_identifier": os.getenv("AVAILABILITY_TRACE_PATH", "traces/traces.txt"),
            "utility_definition": self.utility_metric,
            "utility_max_increment": self.utility_max_increment,
            "cup_scheduler_mode": self.scheduler_mode,
            "epsilon": self.epsilon,
            "inverse_availability_clip": self.inverse_clip,
            "debt_lambda": self.cup_debt_coefficient,
            "alpha_k_policy": "uniform_one",
            "surrogate_enabled": self.surrogate_enabled,
            "surrogate_mode": self.surrogate_mode,
            "surrogate_decay": self.surrogate_decay,
            "aggregation_correction": "horvitz_thompson_uniform_client_delta",
            "local_warm_start_alpha": 1.0,
            "selection_budget": self.capacity,
        }, indent=2), encoding="utf-8")

    def observe_availability(self, availability, round_index):
        for client_id in self.client_ids:
            state = self.states[client_id]
            state.availability_count += int(bool(availability[client_id]))
            state.availability_estimate = state.availability_count / (round_index + 1)

    def reactive_scores(self):
        scores = {}
        for client_id, state in self.states.items():
            raw_inverse = 1.0 / (state.availability_estimate + self.epsilon)
            inverse = min(raw_inverse, self.inverse_clip)
            self.last_clipped[client_id] = raw_inverse > self.inverse_clip
            scores[client_id] = (self.alpha[client_id] * inverse
                                 * (1 + self.cup_debt_coefficient * state.participation_debt))
        return scores

    def oracle_scores(self, mu_hat):
        pi = {client_id: max(self.states[client_id].availability_estimate, self.epsilon)
              for client_id in self.client_ids}
        budget = min(float(self.capacity), sum(pi.values()))
        _, rates = oracle_maxmin_rates(pi, mu_hat, budget)
        return rates

    def select_clients(self, availability, round_index, mu_hat=None):
        """Score and sample only clients currently eligible to participate."""
        self.observe_availability(availability, round_index)
        if self.scheduler_mode == "reactive":
            scores = self.reactive_scores()
        elif self.scheduler_mode == "oracle_maxmin":
            self.last_clipped = {client_id: False for client_id in self.client_ids}
            scores = self.oracle_scores(mu_hat or {client_id: 1.0 for client_id in self.client_ids})
        else:
            raise ValueError("CUP_SCHEDULER must be reactive or oracle_maxmin")
        available_clients = [client_id for client_id in self.client_ids
                             if bool(availability[client_id])]
        candidate_scores = {client_id: scores[client_id]
                            for client_id in available_clients}
        candidate_probabilities = fixed_size_inclusion_probabilities(
            candidate_scores, min(self.capacity, len(available_clients)))
        selected = dependent_round_sample(candidate_probabilities, self.rng)
        probabilities = {client_id: candidate_probabilities.get(client_id, 0.0)
                         for client_id in self.client_ids}
        self.last_scores, self.last_selection_probabilities = scores, probabilities
        return selected

    def observe_external_selection(self, availability, round_index, selected):
        """Apply identical CUP accounting retrospectively to a baseline."""
        self.observe_availability(availability, round_index)
        selected = set(selected)
        self.last_scores = {client_id: 0.0 for client_id in self.client_ids}
        self.last_clipped = {client_id: False for client_id in self.client_ids}
        self.last_selection_probabilities = {
            client_id: float(client_id in selected) for client_id in self.client_ids}

    def realize_participation(self, availability, selected):
        selected = set(selected)
        return [client_id for client_id in self.client_ids
                if bool(availability[client_id]) and client_id in selected]

    def _utility_increment(self, state, accuracy, loss_record):
        if self.utility_metric == "accuracy_gain":
            previous = state.previous_participation_accuracy
            state.previous_participation_accuracy = accuracy
            gain = 0.0 if previous is None else accuracy - previous
            return min(max(0.0, gain), self.utility_max_increment)
        if self.utility_metric == "loss_reduction":
            if loss_record is None:
                return 0.0
            reduction = max(
                0.0,
                float(loss_record["loss_at_global"]) - float(loss_record["loss"]),
            )
            return min(reduction, self.utility_max_increment)
        raise ValueError("CUP_UTILITY_METRIC must be accuracy_gain or loss_reduction")

    def end_round(self, round_index, availability, selected, per_client_accuracy, records):
        selected_set = set(selected)
        participated_set = set(self.realize_participation(availability, selected))
        records_by_client = {record["client_id"]: record for record in records}
        rows = []
        for client_id in self.client_ids:
            state = self.states[client_id]
            selected_bit = int(client_id in selected_set)
            participated_bit = int(client_id in participated_set)
            state.selection_count += selected_bit
            state.participation_count += participated_bit
            increment = 0.0
            if participated_bit:
                increment = self._utility_increment(
                    state,
                    float(per_client_accuracy[client_id]),
                    records_by_client.get(client_id),
                )
                state.utility += increment
                state.last_real_utility_increment = increment
                state.last_real_participation_round = round_index
            state.participation_debt += 1 - participated_bit
            if state.availability_estimate > 0:
                state.normalized_utility = state.utility / state.availability_estimate
            else:
                state.normalized_utility = math.nan

            surrogate_used = False; staleness = 0; reliability = 0.0; surrogate_contribution = 0.0
            if (self.surrogate_enabled and not participated_bit
                    and state.last_real_participation_round is not None):
                surrogate_used = True
                staleness = round_index - state.last_real_participation_round
                reliability = self.surrogate_eta0 * math.exp(-self.surrogate_decay * staleness)
                surrogate_contribution = reliability * state.last_real_utility_increment
                state.surrogate_utility += surrogate_contribution
            rows.append({
                "round": round_index, "client_id": client_id,
                "available": int(bool(availability[client_id])), "selected": selected_bit,
                "participated": participated_bit, "availability_estimate": state.availability_estimate,
                "availability_count": state.availability_count, "selection_count": state.selection_count,
                "participation_count": state.participation_count, "utility_metric": self.utility_metric,
                "utility_increment": increment, "utility": state.utility,
                "normalized_utility": state.normalized_utility,
                "participation_debt": state.participation_debt,
                "selection_score": self.last_scores[client_id],
                "selection_probability": self.last_selection_probabilities[client_id],
                "availability_inverse_clipped": int(self.last_clipped[client_id]),
                "surrogate_used": int(surrogate_used), "surrogate_staleness": staleness,
                "surrogate_weight": reliability, "surrogate_contribution": surrogate_contribution,
            })
        with self.output_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.LOG_FIELDS).writerows(rows)
        return rows

    def aggregation_context(self):
        return {
            client_id: {
                "selection_probability": self.last_selection_probabilities[client_id],
                "availability_estimate": self.states[client_id].availability_estimate,
            } for client_id in self.client_ids
        }

    def metrics(self, rounds_completed, *, include_surrogate=False):
        normalized = [
            (state.utility + (state.surrogate_utility if include_surrogate else 0.0))
            / state.availability_estimate
            for state in self.states.values() if state.availability_estimate > 0]
        mean = sum(normalized) / len(normalized) if normalized else 0.0
        variance = (sum((value - mean) ** 2 for value in normalized) / len(normalized)
                    if normalized else 0.0)
        square_sum = sum(value * value for value in normalized)
        jain = ((sum(normalized) ** 2) / (len(normalized) * square_sum)
                if square_sum else 0.0)
        target = self.capacity * rounds_completed / len(self.client_ids)
        selection_gap = max(
            abs(state.selection_count - target) for state in self.states.values())
        return {"utility_cv": math.sqrt(variance) / (mean + self.epsilon) if mean else 0.0,
                "utility_jain_index": jain, "selection_gap": selection_gap}
