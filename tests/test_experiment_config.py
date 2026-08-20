import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from experiment_config import build_logical_label_map


def test_overlap_label_map_is_deterministic_and_wraps():
    mapping = build_logical_label_map(11, 2, split_mode="overlap")
    assert mapping["h0"] == (0, 1)
    assert mapping["h9"] == (9, 0)
    assert mapping["h10"] == (0, 1)


def test_dirichlet_label_map_is_seeded_and_unique_per_client():
    first = build_logical_label_map(4, 3, split_mode="dirichlet", seed=7)
    second = build_logical_label_map(4, 3, split_mode="dirichlet", seed=7)
    assert first == second
    assert all(len(set(labels)) == 3 for labels in first.values())


def test_invalid_logical_split_is_rejected():
    with pytest.raises(ValueError, match="unknown logical split mode"):
        build_logical_label_map(2, 1, split_mode="unknown")
