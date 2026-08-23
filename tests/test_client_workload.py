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


def test_local_training_uses_fedavg_sgd_and_freezes_feature_blocks_by_default():
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
        and node.args[0].value == "LOCAL_TRAINABLE_BLOCKS"
    ]
    assert defaults == ["0"]
    assert "optim.SGD(" in source
    assert "self._set_local_training_mode()" in source
    assert "self.optimizer = self._new_optimizer()" in source
