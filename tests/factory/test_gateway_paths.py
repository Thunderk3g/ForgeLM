"""API gateway path forwarding.

The gateway module needs the service stack to import, so the routing table and
the path function are extracted with ast and executed on their own.
"""

import ast
from pathlib import Path

import pytest

GATEWAY = Path(__file__).resolve().parents[2] / "src/microservices/api_gateway/src/main.py"


@pytest.fixture(scope="module")
def gw():
    source = GATEWAY.read_text()
    tree = ast.parse(source)
    wanted = {"SERVICE_ROUTES", "FULL_PATH_PREFIXES", "v1_target_path", "DEFAULT_PROXY_TIMEOUT_S",
              "LONG_REQUEST_TIMEOUT_S", "LONG_REQUEST_PREFIXES", "proxy_timeout_seconds"}
    nodes = []
    for node in tree.body:
        names = set()
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.FunctionDef):
            names = {node.name}
        if names & wanted:
            nodes.append(node)
    import os

    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(GATEWAY), "exec"), namespace)
    assert wanted <= set(namespace)
    return namespace


def _old_target_path(routes, path):
    """The forwarding rule before FULL_PATH_PREFIXES existed (always strip the matched prefix)."""
    full_path = f"/api/v1/{path}"
    matched = next((r for r in routes if full_path.startswith(r)), None)
    if matched:
        prefix = matched.replace("/api/v1/", "")
        if path.startswith(prefix):
            return "/" + path[len(prefix):].lstrip("/")
    return f"/{path}" if path else ""


def test_factory_requests_keep_their_full_path(gw):
    f = gw["v1_target_path"]
    assert f("factory/hardware") == "/api/v1/factory/hardware"
    assert f("factory/classifiers/router-v1/predict") == "/api/v1/factory/classifiers/router-v1/predict"
    assert f("factory") == "/api/v1/factory"


def test_a_prefix_that_merely_starts_like_factory_is_not_treated_as_factory(gw):
    assert gw["v1_target_path"]("factoryish/x") == _old_target_path(gw["SERVICE_ROUTES"], "factoryish/x")


def test_every_other_route_is_forwarded_exactly_as_before(gw):
    routes = gw["SERVICE_ROUTES"]
    checked = 0
    for route in routes:
        if not route.startswith("/api/v1/") or any(route.startswith(p) for p in gw["FULL_PATH_PREFIXES"]):
            continue
        base = route[len("/api/v1/"):]
        for suffix in ("", "/x", "/x/y"):
            path = base + suffix
            assert gw["v1_target_path"](path) == _old_target_path(routes, path), path
            checked += 1
    assert checked >= 30
    # The documented example in the gateway code.
    assert gw["v1_target_path"]("visualization/preprocessing/visualize") == "/preprocessing/visualize"


def test_classifier_training_gets_the_long_timeout(gw):
    t = gw["proxy_timeout_seconds"]
    long_t, short_t = gw["LONG_REQUEST_TIMEOUT_S"], gw["DEFAULT_PROXY_TIMEOUT_S"]
    assert long_t > short_t
    assert t("POST", gw["v1_target_path"]("factory/classifiers")) == long_t
    assert t("post", "/api/v1/factory/classifiers/") == long_t
    # Prediction, listing and every other route keep the short timeout.
    assert t("POST", "/api/v1/factory/classifiers/router-v1/predict") == short_t
    assert t("GET", "/api/v1/factory/classifiers") == short_t
    assert t("POST", "/preprocessing/visualize") == short_t
