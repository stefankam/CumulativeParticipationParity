
"""Five-client deterministic diagnostic run for stateful FL methods.

Run inside the server/client image where PyTorch is installed:
`python tests/tiny_method_experiment.py`.
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "client")]
from fl_methods import AFLState, FairFedCSState, php_fl_auxiliary_update, q_fedavg
from php_fl import train_php_fl


def scalar_record(client_id, global_value, local_value, loss):
    return {"client_id": client_id, "state_dict": {"w": torch.tensor([local_value])},
            "loss_at_global": loss, "loss": loss * .8, "sample_count": 1,
            "delta": global_value - local_value}


def main():
    clients = tuple(f"h{i}" for i in range(5))
    global_weights = {"w": torch.tensor([0.0])}
    afl = AFLState(clients, lambda_lr=.1)
    fair = FairFedCSState(clients, capacity=2, reputation_decay=.5)
    php_states = {}
    loader = DataLoader(TensorDataset(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])), batch_size=2)
    auxiliary = nn.Sequential(nn.Linear(2, 2))
    for round_index in range(5):
        available = [clients[(2 * round_index) % 5], clients[(2 * round_index + 1) % 5]]
        records = [scalar_record(client_id, float(global_weights["w"]),
                                 float(global_weights["w"]) + index + 1,
                                 1.0 + index + round_index / 10)
                   for index, client_id in enumerate(available)]
        q_model, q_log = q_fedavg(global_weights, records, q=1, lipschitz=1)
        afl_before = dict(afl.lambdas)
        global_weights, afl_log = afl.update(global_weights, records)
        selected = fair.select(available, 2)
        fair.on_round_end(selected, records)

        php_records = []
        for client_id in available:
            state, php_log = train_php_fl(auxiliary, loader, php_states, client_id, 1)
            php_records.append({"record_type": "php_fl_auxiliary", "client_id": client_id,
                                "state_dict": state, "sample_count": 2,
                                "php_diagnostics": php_log})
        auxiliary_state, php_server_log = php_fl_auxiliary_update(php_records)
        auxiliary.load_state_dict(auxiliary_state)
        print({"round": round_index,
               "q_ffl": {"model": float(q_model["w"]), **q_log},
               "afl": {"lambda_before": afl_before,
                       "lambda_after": dict(afl.lambdas), **afl_log},
               "fairfedcs": {"selected": selected, "Q": dict(fair.queues),
                             "reputation": dict(fair.reputations),
                             "CSI": {client: fair.suitability(client) for client in clients}},
               "php_fl": {"active": available, "client": php_log,
                          "server": php_server_log}})


if __name__ == "__main__":
    main()

