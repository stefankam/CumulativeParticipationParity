import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from availability import (extract_availability_vectors,
                          logical_client_availability,
                          resolve_availability_trace_path)
from fairness import FairnessSchedulerController


def test_logical_clients_use_zero_based_mapping_to_one_based_trace_hosts(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "0": {"messages": "wifi_on\nbattery_charged_on\nwifi_off"},
        "1": {"messages": "battery_charged_on\nwifi_on\nwifi_off"},
    }))

    vectors = extract_availability_vectors(trace, length=3)

    assert [logical_client_availability(vectors, "h0", round_index)
            for round_index in range(3)] == [False, True, False]
    assert [logical_client_availability(vectors, "h1", round_index)
            for round_index in range(3)] == [False, True, False]
    assert logical_client_availability(vectors, "h0", 4) is True


def test_missing_logical_trace_fails_instead_of_assuming_available():
    with pytest.raises(KeyError, match="No availability trace"):
        logical_client_availability({"h1": [1]}, "h1", 0)


def test_trace_path_can_be_resolved_relative_to_server_directory():
    resolved = resolve_availability_trace_path("traces/traces.txt")
    assert resolved.name == "traces.txt"
    assert resolved.parent.name == "traces"


def test_cpp_accepts_an_initial_all_offline_trace_round():
    controller = FairnessSchedulerController(["h0", "h1"], mode="cup")
    telemetry = {"h0": False, "h1": False}
    controller.observe_telemetry(telemetry)

    assert controller.select(
        telemetry=telemetry, capacity=1, mu_hat={"h0": 1.0, "h1": 1.0}
    ) == []
