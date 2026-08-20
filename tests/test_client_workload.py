import ast
from pathlib import Path


def test_logical_client_default_shard_finishes_within_request_budget():
    source = (Path(__file__).parents[1] / "client" / "topology_client.py").read_text()
    module = ast.parse(source)
    defaults = [
        node.args[1].value
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getenv"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "LOGICAL_CLIENT_SUBSET_SIZE"
        and isinstance(node.args[1], ast.Constant)
    ]
    assert defaults == ["100"]
