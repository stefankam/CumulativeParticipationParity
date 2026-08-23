import ast
from pathlib import Path


def test_send_weights_uses_one_post_with_logical_client_form():
    """Guard against silently overwriting a successful training response."""
    source = (Path(__file__).parents[1] / "server" / "topology_server.py").read_text()
    module = ast.parse(source)
    provider = next(node for node in module.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == "TopologyProvider")
    method = next(node for node in provider.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "send_weights_to_client")
    posts = [node for node in ast.walk(method)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "requests"
             and node.func.attr == "post"]

    assert len(posts) == 1
    assert {keyword.arg for keyword in posts[0].keywords} >= {
        "files", "data", "timeout"}


def test_stateful_fair_methods_have_no_heuristic_scalar_weight_branches():
    root = Path(__file__).parents[1]
    topology = (root / "server" / "topology_server.py").read_text()
    baselines = (root / "server" / "baselines.py").read_text()
    assert "samples * loss **" not in topology
    assert "samples / (1 + self.utility_log" not in topology
    assert "math.exp(float(os.getenv(\"AFL_ETA\"" not in topology
    assert "requires its dedicated stateful server update" in baselines
    assert "q_fedavg(global_weights, records)" in topology
    assert "php_fl_auxiliary_update(records)" in topology


def test_legacy_availability_filtered_cup_selector_is_disabled():
    source = (Path(__file__).parents[1] / "server" / "topology_server.py").read_text()
    assert "Legacy availability-filtered CUP selection is disabled" in source
    assert "CumulativeUtilityParity in the logical round coordinator" in source
