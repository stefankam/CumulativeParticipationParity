"""Theory-aligned primitives for cumulative utility parity.

This module deliberately keeps telemetry (A), scheduler decisions (S), and
actual participation (P=A*S) separate. It has no ML-framework dependencies,
which also makes experiment bookkeeping independently testable.
"""
from __future__ import annotations

import csv
import math
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


class AvailabilityEstimator:
    """Estimate pi_k exclusively from observed telemetry A_k(t)."""
    def __init__(self, window_size: Optional[int] = 50):
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be positive or None")
        self.window_size = window_size
        self._observations = defaultdict(lambda: deque(maxlen=window_size) if window_size else [])

    def observe(self, availability: Mapping[str, bool]) -> None:
        for client_id, available in availability.items():
            self._observations[client_id].append(int(bool(available)))

    def estimate(self, client_id: str, default: float = 1.0) -> float:
        values = self._observations.get(client_id)
        return sum(values) / len(values) if values else default

    def estimates(self, clients: Iterable[str]) -> Dict[str, float]:
        return {client: self.estimate(client) for client in clients}

    def observation_count(self, client_id: str) -> int:
        return len(self._observations.get(client_id, ()))


class DeterministicInverseAvailabilityScheduler:
    def select(self, clients: Sequence[str], m: int, pi_hat: Mapping[str, float], **_) -> list[str]:
        return sorted(clients, key=lambda k: (-1.0 / max(pi_hat[k], 1e-12), str(k)))[:m]


class ProbabilisticInverseAvailabilityScheduler:
    def __init__(self, seed: int = 0): self.rng = random.Random(seed)
    def select(self, clients: Sequence[str], m: int, pi_hat: Mapping[str, float], **_) -> list[str]:
        remaining, chosen = list(clients), []
        for _ in range(min(m, len(remaining))):
            weights = [1.0 / max(float(pi_hat[k]), 1e-12) for k in remaining]
            pick = self.rng.choices(remaining, weights=weights, k=1)[0]
            chosen.append(pick); remaining.remove(pick)
        return chosen


class FedAvgScheduler:
    def __init__(self, seed: int = 0): self.rng = random.Random(seed)
    def select(self, clients: Sequence[str], m: int, **_) -> list[str]:
        return self.rng.sample(list(clients), min(m, len(clients)))


class LeastSelectedScheduler:
    def select(self, clients, m, selection_counts=None, **_):
        counts = selection_counts or {}
        return sorted(clients, key=lambda k: (counts.get(k, 0), str(k)))[:m]


def make_baseline(name: str, seed: int = 0):
    factories = {"fedavg": lambda: FedAvgScheduler(seed), "least_selected": LeastSelectedScheduler}
    try: return factories[name.lower()]()
    except KeyError as exc: raise ValueError(f"unknown baseline: {name}") from exc


def make_scheduler(selection_mode: str, seed: int = 0):
    if selection_mode == "deterministic": return DeterministicInverseAvailabilityScheduler()
    if selection_mode == "probabilistic": return ProbabilisticInverseAvailabilityScheduler(seed)
    if selection_mode == "cup": return CumulativeUtilityParityScheduler(seed)
    raise ValueError("selection_mode must be 'deterministic', 'probabilistic', or 'cup'")


class UtilityTracker:
    MODES = {"loss_delta", "accuracy_delta", "auc"}
    def __init__(self, mode: str = "auc", max_increment: Optional[float] = None):
        if mode not in self.MODES: raise ValueError(f"utility_definition must be one of {sorted(self.MODES)}")
        if max_increment is not None and max_increment <= 0: raise ValueError("max_increment must be positive")
        self.mode, self.max_increment = mode, max_increment
        self.utility_history, self._previous = defaultdict(list), {}

    def observe(self, client_id: str, *, accuracy: float, loss: float, participated: bool = True) -> float:
        current = accuracy if self.mode in {"accuracy_delta", "auc"} else loss
        previous = self._previous.get(client_id)
        if self.mode == "auc": metric = float(accuracy)
        elif self.mode == "accuracy_delta": metric = 0.0 if previous is None else float(accuracy - previous)
        else: metric = 0.0 if previous is None else float(previous - loss)
        metric = metric if participated else 0.0
        if participated: self._previous[client_id] = current
        metric = max(0.0, metric)
        if self.max_increment is not None: metric = min(metric, self.max_increment)
        self.utility_history[client_id].append(metric)
        return metric

    def cumulative(self, client_id: str) -> float: return float(sum(self.utility_history[client_id]))
    def normalized(self, client_id: str, pi_hat: float) -> float:
        return self.cumulative(client_id) / max(float(pi_hat), 1e-12)


def fairness_metrics(normalized_utilities: Mapping[str, float], selection_counts: Mapping[str, int]) -> dict:
    values = list(map(float, normalized_utilities.values()))
    mean = statistics.fmean(values) if values else 0.0
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    square_sum = sum(v * v for v in values)
    jain = sum(values) ** 2 / (len(values) * square_sum) if square_sum else 0.0
    abs_diffs = sum(abs(x-y) for x in values for y in values)
    gini = abs_diffs / (2*len(values)*sum(values)) if values and sum(values) > 0 else 0.0
    counts = list(selection_counts.values())
    return {"utility_cv": std/mean if mean else 0.0, "utility_jain_index": jain,
            "selection_gap": max(counts)-min(counts) if counts else 0,
            "gini_coefficient": gini, "worst_client_utility": min(values) if values else 0.0,
            "mean_utility": mean}


def conditional_selection_gap(selection_counts, availability_counts) -> float:
    rates = [selection_counts.get(k, 0)/observed for k, observed in availability_counts.items() if observed > 0]
    return max(rates)-min(rates) if rates else 0.0


def cumulative_utility_parity_rates(pi_hat, mu_hat, budget: float, epsilon: float = 1e-12):
    clients = list(pi_hat)
    if not clients or budget < 0 or budget > sum(pi_hat.values()) + epsilon:
        raise ValueError("budget must lie in [0, sum(pi_hat)]")
    if any(pi_hat[k] <= 0 or mu_hat.get(k, 0) <= 0 for k in clients):
        raise ValueError("pi_hat and mu_hat must be strictly positive")
    denominator = sum(float(pi_hat[k])/float(mu_hat[k]) for k in clients)
    tau = min(min(float(mu_hat[k]) for k in clients), budget/denominator)
    return tau, {k: tau/float(mu_hat[k]) for k in clients}


class CumulativeUtilityParityScheduler:
    def __init__(self, seed=0):
        self.rng = random.Random(seed); self.conditional_opportunities = defaultdict(int); self.conditional_selections = defaultdict(int)
    def select(self, clients, m, *, availability, pi_hat, mu_hat, budget=None, **_):
        budget = min(float(m if budget is None else budget), sum(pi_hat.values()))
        _, targets = cumulative_utility_parity_rates(pi_hat, mu_hat, budget)
        available = [k for k in clients if availability.get(k, False)]
        for k in available: self.conditional_opportunities[k] += 1
        tie = {k: self.rng.random() for k in available}
        deficits = {k: targets[k]*self.conditional_opportunities[k]-self.conditional_selections[k] for k in available}
        ranked = sorted((k for k in available if deficits[k] > 0), key=lambda k: (-deficits[k], tie[k]))
        selected = ranked[:min(m, len(ranked))]
        for k in selected: self.conditional_selections[k] += 1
        return selected


class ParticipationDebtScheduler:
    def __init__(self, clients, alpha=None, lambda_reactive=.1, epsilon=1e-6, seed=0):
        self.debt = {k: 0 for k in clients}; self.alpha = alpha or {k: 1.0 for k in clients}
        self.lambda_reactive, self.epsilon, self.rng = lambda_reactive, epsilon, random.Random(seed)
    def weights(self, pi_hat):
        raw = {k: self.alpha[k]/(pi_hat[k]+self.epsilon)*(1+self.lambda_reactive*self.debt[k]) for k in self.debt}
        total = sum(raw.values()); return {k: v/total for k,v in raw.items()}
    def select(self, clients, m, *, pi_hat, availability, **_):
        weights = self.weights(pi_hat); available = [k for k in clients if availability.get(k, False)]
        return sorted(available, key=lambda k: (-weights[k], self.rng.random()))[:min(m,len(available))]
    def end_round(self, participated):
        participated = set(participated)
        for k in self.debt: self.debt[k] += int(k not in participated)


class FairnessSchedulerController:
    def __init__(self, clients, *, mode="cup", window_size=50, seed=0, lambda_reactive=.1, epsilon=1e-6):
        self.clients, self.mode = list(clients), mode; self.estimator = AvailabilityEstimator(window_size)
        self.scheduler = ParticipationDebtScheduler(clients, lambda_reactive=lambda_reactive, epsilon=epsilon, seed=seed) if mode == "reactive" else make_scheduler(mode, seed)
    def observe_telemetry(self, telemetry): self.estimator.observe(telemetry)
    def select(self, *, telemetry, capacity, mu_hat, budget=None):
        pi_hat = self.estimator.estimates(self.clients)
        if self.mode == "cup":
            return self.scheduler.select(self.clients, capacity, availability=telemetry, pi_hat=pi_hat, mu_hat=mu_hat,
                                         budget=min(capacity,sum(pi_hat.values())) if budget is None else budget)
        if self.mode == "reactive": return self.scheduler.select(self.clients, capacity, availability=telemetry, pi_hat=pi_hat)
        return self.scheduler.select(self.clients, capacity, pi_hat)
    def end_round(self, participated):
        if self.mode == "reactive": self.scheduler.end_round(participated)


def importance_corrected_weights(objective_weights, inclusion_probabilities, selected, clip=100.0):
    raw = {k: min(float(objective_weights[k])/max(float(inclusion_probabilities[k]), 1.0/clip), clip) for k in selected}
    total=sum(raw.values()); return {k:v/total for k,v in raw.items()} if total else {}

def surrogate_bias_bound(errors, weights): return sum(max(0.0,float(weights.get(k,0.0)))*abs(float(e)) for k,e in errors.items())
def surrogate_weight(eta0, lambda_decay, staleness): return eta0*math.exp(-lambda_decay*staleness)
def aggregation_inputs(real_updates, surrogate_updates, surrogate_mode):
    if surrogate_mode == "training": return list(real_updates)+list(surrogate_updates)
    if surrogate_mode == "accounting": return list(real_updates)
    raise ValueError("surrogate_mode must be 'training' or 'accounting'")

ROUND_FIELDS = ["round","client_id","available","selected","participated","availability_estimate","utility","normalized_utility","accuracy","loss","staleness","surrogate_weight","surrogate_error","aggregate_bias","number_of_surrogates"]

class RoundLogger:
    def __init__(self, output_dir):
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True)
        self._paths={n:self.output_dir/f"{n}.csv" for n in ("availability","selection","participation")}; self.round_path=self.output_dir/"rounds.csv"
        for path,fields in [(self.round_path,ROUND_FIELDS)]+[(p,["round","client_id",n]) for n,p in self._paths.items()]:
            with path.open("w",newline="") as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    def log(self,row):
        with self.round_path.open("a",newline="") as f: csv.DictWriter(f,fieldnames=ROUND_FIELDS).writerow({k:row.get(k) for k in ROUND_FIELDS})
        for name,path in self._paths.items():
            value=row[{"availability":"available","selection":"selected","participation":"participated"}[name]]
            with path.open("a",newline="") as f: csv.DictWriter(f,fieldnames=["round","client_id",name]).writerow({"round":row["round"],"client_id":row["client_id"],name:int(value)})

@dataclass
class RuntimeOverhead:
    scheduler_runtime: float=0.0; aggregation_runtime: float=0.0; surrogate_runtime: float=0.0; availability_estimation_runtime: float=0.0; communication_bytes: int=0; server_peak_memory_bytes: int=0
    def relative_to(self,fedavg):
        def ratio(value,baseline): return value/baseline if baseline else 0.0
        return {key+"_relative_to_fedavg":ratio(getattr(self,key),getattr(fedavg,key)) for key in self.__dataclass_fields__}


class RoundCoordinator:
    def __init__(self,clients,m,*,selection_mode="cup",window_size=50,utility_definition="auc",surrogate_mode="accounting",lambda_decay=.1,eta0=1.0,seed=0,output_dir="results"):
        if surrogate_mode not in {"training","accounting"}: raise ValueError("surrogate_mode must be 'training' or 'accounting'")
        self.clients,self.m=list(clients),m; self.estimator=AvailabilityEstimator(window_size); self.scheduler=make_scheduler(selection_mode,seed); self.utilities=UtilityTracker(utility_definition)
        self.surrogate_mode,self.lambda_decay,self.eta0=surrogate_mode,lambda_decay,eta0; self.selection_counts,self.participation_counts=defaultdict(int),defaultdict(int)
        self.utility_sums,self.utility_observations=defaultdict(float),defaultdict(int); self.last_participated={}; self.logger=RoundLogger(output_dir); self.overhead=RuntimeOverhead()

    def begin_round(self,round_index,telemetry):
        start=time.perf_counter(); self.estimator.observe(telemetry); estimates=self.estimator.estimates(self.clients); self.overhead.availability_estimation_runtime += time.perf_counter()-start
        mu_hat={k:self.utility_sums[k]/self.utility_observations[k] if self.utility_observations[k] else 1.0 for k in self.clients}; start=time.perf_counter()
        if isinstance(self.scheduler,CumulativeUtilityParityScheduler):
            # Selection precedes realization of participation in this coordinator:
            # an offline selected client is logged as selected but not participated.
            # Smooth a first-round zero estimate so the CUP rate calculation remains
            # defined until that client has an observed availability opportunity.
            smoothed={k:max(value,1e-12) for k,value in estimates.items()}
            selected=self.scheduler.select(self.clients,self.m,availability={k:True for k in self.clients},pi_hat=smoothed,mu_hat=mu_hat,budget=min(self.m,sum(smoothed.values())))
        else: selected=self.scheduler.select(self.clients,self.m,estimates)
        self.overhead.scheduler_runtime += time.perf_counter()-start; participated=[k for k in selected if telemetry.get(k,False)]
        for k in selected:self.selection_counts[k]+=1
        for k in participated:self.participation_counts[k]+=1;self.last_participated[k]=round_index
        return selected,participated

    def log_round(self,round_index,telemetry,selected,metrics,surrogate_errors=None):
        surrogate_errors=surrogate_errors or {}; surrogate_weights={k:surrogate_weight(self.eta0,self.lambda_decay,round_index-self.last_participated.get(k,round_index)) for k in surrogate_errors}; aggregate_bias=surrogate_bias_bound(surrogate_errors,surrogate_weights)
        for k in self.clients:
            accuracy,loss=metrics.get(k,(0.0,0.0)); participated=k in selected and bool(telemetry.get(k,False)); utility=self.utilities.observe(k,accuracy=accuracy,loss=loss,participated=participated)
            if participated:self.utility_sums[k]+=utility;self.utility_observations[k]+=1
            pi=self.estimator.estimate(k);stale=round_index-self.last_participated.get(k,round_index)
            self.logger.log({"round":round_index,"client_id":k,"available":bool(telemetry.get(k,False)),"selected":k in selected,"participated":participated,"availability_estimate":pi,"utility":utility,"normalized_utility":self.utilities.normalized(k,pi),"accuracy":accuracy,"loss":loss,"staleness":stale,"surrogate_weight":surrogate_weights.get(k,0.0),"surrogate_error":surrogate_errors.get(k,0.0),"aggregate_bias":aggregate_bias,"number_of_surrogates":len(surrogate_errors)})


class AvailabilityModel:
    MODES={"independent","trace","bursty_markov","correlated_dropout","minority_class_dropout"}
    def __init__(self,clients,mode="independent",probabilities=None,trace=None,seed=0,transition_up=.25,transition_down=.1,minority_clients=()):
        if mode not in self.MODES:raise ValueError(f"unknown availability model: {mode}")
        self.clients,self.mode,self.rng=list(clients),mode,random.Random(seed);self.probabilities=probabilities or {k:.8 for k in clients};self.trace,self.state=trace or {},{k:True for k in clients};self.transition_up,self.transition_down=transition_up,transition_down;self.minority_clients=set(minority_clients)
    def at(self,round_index):
        if self.mode=="trace":return {k:bool(self.trace.get(k,[False])[round_index%len(self.trace.get(k,[False]))]) for k in self.clients}
        if self.mode=="correlated_dropout":
            common_up=self.rng.random()>=1.0-statistics.fmean(self.probabilities.values());return {k:common_up and self.rng.random()<self.probabilities[k] for k in self.clients}
        result={}
        for k in self.clients:
            if self.mode=="bursty_markov":
                p_change=self.transition_down if self.state[k] else self.transition_up
                if self.rng.random()<p_change:self.state[k]=not self.state[k]
                result[k]=self.state[k]
            else:
                probability=self.probabilities[k]
                if self.mode=="minority_class_dropout" and k in self.minority_clients:probability*=.5
                result[k]=self.rng.random()<probability
        return result

def apply_dynamic_data_drift(distributions,round_index,drift_round=25):return {k:list(v[1:])+list(v[:1]) if round_index>=drift_round else list(v) for k,v in distributions.items()}
def summarize_seeds(rows,output_path="results_summary.csv"):
    grouped=defaultdict(list)
    for row in rows:
        for key,value in row.items():
            if key!="seed" and isinstance(value,(int,float)):grouped[key].append(float(value))
    summary=[]
    for metric,values in grouped.items():
        mean=statistics.fmean(values);std=statistics.stdev(values) if len(values)>1 else 0.0;half=1.96*std/math.sqrt(len(values));summary.append({"metric":metric,"mean":mean,"std":std,"ci95_low":mean-half,"ci95_high":mean+half})
    with Path(output_path).open("w",newline="") as f:writer=csv.DictWriter(f,fieldnames=["metric","mean","std","ci95_low","ci95_high"]);writer.writeheader();writer.writerows(summary)
    return summary

SEEDS=(0,1,2,3,4);LAMBDA_SWEEP=(.05,.10,.20,.50);WINDOW_SWEEP=(10,20,50)
