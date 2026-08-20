"""Configuration helpers shared by experiment launchers and the server."""

from __future__ import annotations

import random


def build_logical_label_map(client_count, labels_per_client, *, split_mode="overlap",
                            dirichlet_alpha=0.5, seed=0, class_count=10):
    """Return a deterministic non-IID label assignment for logical clients.

    ``overlap`` walks around the label ring, while ``dirichlet`` samples each
    client's labels without replacement from a seeded class distribution.
    """
    if client_count < 1:
        raise ValueError("client_count must be positive")
    if not 1 <= labels_per_client <= class_count:
        raise ValueError("labels_per_client must be between 1 and class_count")
    if split_mode not in {"overlap", "dirichlet"}:
        raise ValueError(f"unknown logical split mode: {split_mode}")

    if split_mode == "overlap":
        return {
            f"h{client}": tuple(
                (client + offset) % class_count
                for offset in range(labels_per_client)
            )
            for client in range(client_count)
        }

    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    rng = random.Random(seed)
    mapping = {}
    for client in range(client_count):
        # Gamma variates normalized by their sum are Dirichlet-distributed.
        weights = [rng.gammavariate(dirichlet_alpha, 1.0)
                   for _ in range(class_count)]
        available = list(range(class_count))
        labels = []
        for _ in range(labels_per_client):
            chosen = rng.choices(available,
                                 weights=[weights[label] for label in available],
                                 k=1)[0]
            labels.append(chosen)
            available.remove(chosen)
        mapping[f"h{client}"] = tuple(labels)
    return mapping
