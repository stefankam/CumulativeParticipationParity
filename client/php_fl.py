"""PHP-FL homogeneous-architecture adapter with persistent DEAL/ISPU state.

The repository has one ResNet architecture, so the shared auxiliary model and
personalized model have matching shapes.  They remain distinct models: only the
auxiliary update is returned to the server.
"""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn.functional as F


def _kl(student, teacher, temperature):
    return F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher.detach() / temperature, dim=1),
        reduction="batchmean",
    ) * temperature * temperature


def _ispu_mask(importance, ratio):
    flat = torch.cat([value.reshape(-1) for value in importance.values()])
    count = max(1, min(flat.numel(), math.ceil(flat.numel() * ratio)))
    threshold = torch.kthvalue(flat, count).values
    return {name: value <= threshold for name, value in importance.items()}


def train_php_fl(global_aux, loader, states, client_id, local_epochs):
    """Run DEAL and ISPU while retaining a personalized local model."""
    state = states.get(client_id)
    if state is None:
        local_model = copy.deepcopy(global_aux)
        importance = {name: torch.zeros_like(parameter)
                      for name, parameter in local_model.named_parameters()}
        state = {"local_model": local_model, "importance": importance,
                 "mask": {}, "rounds": 0}
        states[client_id] = state
    local_model = state["local_model"]
    auxiliary = copy.deepcopy(global_aux)

    initial_ratio = float(os.getenv("PHP_ISPU_INITIAL_RATIO", ".1"))
    ratio_growth = float(os.getenv("PHP_ISPU_RATIO_GROWTH", ".02"))
    ratio = min(1.0, initial_ratio + ratio_growth * state["rounds"])
    mask = _ispu_mask(state["importance"], ratio)
    global_parameters = dict(global_aux.named_parameters())
    with torch.no_grad():
        for name, parameter in local_model.named_parameters():
            parameter[mask[name]] = global_parameters[name][mask[name]]

    lr = float(os.getenv("CLIENT_LEARNING_RATE", ".01"))
    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    auxiliary_optimizer = torch.optim.SGD(auxiliary.parameters(), lr=lr)
    align_weight = float(os.getenv("PHP_DEAL_ALIGN_WEIGHT", "1"))
    temperature = float(os.getenv("PHP_DEAL_TEMPERATURE", "2"))
    totals = {"supervised": 0.0, "local_alignment": 0.0,
              "aux_alignment": 0.0, "batches": 0}
    before = {name: parameter.detach().clone()
              for name, parameter in local_model.named_parameters()}
    for _ in range(local_epochs):
        local_model.train(); auxiliary.train()
        for inputs, labels in loader:
            local_logits = local_model(inputs)
            auxiliary_logits = auxiliary(inputs)
            ensemble = 0.5 * (local_logits + auxiliary_logits)
            supervised = F.cross_entropy(ensemble, labels)
            local_alignment = _kl(local_logits, auxiliary_logits, temperature)
            auxiliary_alignment = _kl(auxiliary_logits, local_logits, temperature)
            loss = supervised + align_weight * (local_alignment + auxiliary_alignment)
            local_optimizer.zero_grad(); auxiliary_optimizer.zero_grad()
            loss.backward(); local_optimizer.step(); auxiliary_optimizer.step()
            totals["supervised"] += float(supervised.detach())
            totals["local_alignment"] += float(local_alignment.detach())
            totals["aux_alignment"] += float(auxiliary_alignment.detach())
            totals["batches"] += 1

    decay = float(os.getenv("PHP_ISPU_IMPORTANCE_DECAY", ".9"))
    with torch.no_grad():
        for name, parameter in local_model.named_parameters():
            change = (parameter - before[name]).abs()
            state["importance"][name].mul_(decay).add_(change, alpha=1 - decay)
    state["mask"] = {name: value.detach().clone() for name, value in mask.items()}
    state["rounds"] += 1
    batches = max(1, totals.pop("batches"))
    diagnostics = {key: value / batches for key, value in totals.items()}
    diagnostics.update({"ispu_ratio": ratio,
                        "mask_fraction": sum(int(value.sum()) for value in mask.values()) /
                        sum(value.numel() for value in mask.values())})
    return auxiliary.state_dict(), diagnostics

