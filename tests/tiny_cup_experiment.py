"""Dependency-free deterministic N=5, m=2, T=10 CUP diagnostic."""

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from cup import CumulativeUtilityParity


def main():
    clients = tuple(f"h{i}" for i in range(5))
    trace = [
        {"h0": 0, "h1": 1, "h2": 0, "h3": 1, "h4": 1},
        {"h0": 1, "h1": 0, "h2": 1, "h3": 0, "h4": 1},
    ] * 5
    with tempfile.TemporaryDirectory() as directory:
        cup = CumulativeUtilityParity(
            clients, 2, seed=7, output_path=Path(directory) / "cup.csv")
        for round_index, availability in enumerate(trace):
            selected = cup.select_clients(availability, round_index)
            participated = cup.realize_participation(availability, selected)
            accuracies = {client: 10 + round_index + int(client[1:])
                          for client in clients}
            rows = cup.end_round(
                round_index, availability, selected, accuracies, [])
            print({
                "round": round_index,
                "A": availability,
                "pi_hat": {client: cup.states[client].availability_estimate
                           for client in clients},
                "debt": {client: cup.states[client].participation_debt
                         for client in clients},
                "selection_score": dict(cup.last_scores),
                "S": {client: int(client in selected) for client in clients},
                "P": {client: int(client in participated) for client in clients},
                "delta_u": {row["client_id"]: row["utility_increment"] for row in rows},
                "utility": {client: cup.states[client].utility for client in clients},
                "u_tilde": {client: cup.states[client].normalized_utility
                            for client in clients},
            })


if __name__ == "__main__":
    main()

