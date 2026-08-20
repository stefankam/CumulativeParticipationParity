import ast
from pathlib import Path


def _module(path):
    return ast.parse((Path(__file__).parents[1] / path).read_text())


def test_client_registers_once_and_only_status_updates_are_periodic():
    module = _module("client/main_client.py")
    main = next(node for node in module.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and getattr(node.test.left, "id", None) == "__name__")
    periodic = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "periodic_status_update")
    called_names = {node.func.id for node in ast.walk(periodic)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)}
    assert "send_status_update" in called_names
    assert "register_with_main_server" not in called_names
    assert any(isinstance(node, ast.While) for node in main.body)


def test_suite_gives_all_seed_processes_one_registry_path():
    source = (Path(__file__).parents[1] / "server" / "experiment_suite.py").read_text()
    assert 'env["DEVICE_REGISTRY_PATH"] = str(RUN_DIR / "device_registry.json")' in source
    assert 'env["REGISTERED_CLIENTS_CACHE"] = env["DEVICE_REGISTRY_PATH"]' in source


def test_server_restores_latency_instead_of_waiting_for_fresh_seed_updates():
    source = (Path(__file__).parents[1] / "server" / "main_server.py").read_text()
    assert '"latency": cached.get("latency")' in source
    assert '"packet_loss": cached.get("packet_loss")' in source
    assert "persist_device_registry()" in source


def test_server_supports_original_cache_and_client_count_environment_names():
    source = (Path(__file__).parents[1] / "server" / "main_server.py").read_text()
    assert '"REGISTERED_CLIENTS_CACHE"' in source
    assert '"REGISTERED_CLIENT_COUNT"' in source
    assert '"REUSE_REGISTERED_CLIENTS"' in source


def test_complete_cache_does_not_log_or_enter_registration_wait():
    source = (Path(__file__).parents[1] / "server" / "main_server.py").read_text()
    assert "if restored_count >= num_clients:" in source
    assert "no re-registration wait is required" in source
    assert "waiting only for the missing registrations" in source
