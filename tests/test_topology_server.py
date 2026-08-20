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
