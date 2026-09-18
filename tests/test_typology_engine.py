"""
Automated Unit & Integration Test Suite for Typology & Synthetic Engine.
Person B: Tests network generation, FATF typology detection algorithms,
adversarial hold-out evaluation, and Typology API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from data.synthetic.generate_network import generate_fatf_network
from engines.typology.fatf_rules import build_networkx_graph, detect_structuring, detect_round_tripping, detect_rapid_layering
from engines.typology.detect import detect_all_typologies, evaluate_adversarial_set, load_synthetic_network
from engines.typology.api import app


@pytest.fixture(scope="module")
def network_data():
    return load_synthetic_network()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_synthetic_network_structure(network_data):
    assert "nodes" in network_data
    assert "edges" in network_data
    assert len(network_data["nodes"]) >= 20
    assert len(network_data["edges"]) >= 100
    
    # Check node structure
    first_node = network_data["nodes"][0]
    assert "account_id" in first_node
    assert "risk_rating" in first_node
    
    # Check edge structure
    first_edge = network_data["edges"][0]
    assert "from_account" in first_edge
    assert "to_account" in first_edge
    assert "amount" in first_edge
    assert "timestamp" in first_edge
    assert first_edge["tier"] == "synthetic_network"


def test_fatf_structuring_detector(network_data):
    G = build_networkx_graph(network_data["edges"])
    flags = detect_structuring(G)
    assert len(flags) >= 1
    
    flag = flags[0]
    assert flag["typology"] == "structuring"
    assert "FATF" in flag["fatf_reference"]
    assert len(flag["involved_accounts"]) >= 4
    assert len(flag["evidence_transaction_ids"]) >= 3
    assert 0.0 <= flag["confidence"] <= 1.0


def test_fatf_round_tripping_detector(network_data):
    G = build_networkx_graph(network_data["edges"])
    flags = detect_round_tripping(G)
    assert len(flags) >= 1
    
    flag = flags[0]
    assert flag["typology"] == "round_tripping"
    assert "Beneficial Ownership" in flag["fatf_reference"]
    assert len(flag["involved_accounts"]) >= 3
    assert len(flag["evidence_transaction_ids"]) >= 3


def test_fatf_rapid_layering_detector(network_data):
    G = build_networkx_graph(network_data["edges"])
    flags = detect_rapid_layering(G)
    assert len(flags) >= 1
    
    flag = flags[0]
    assert flag["typology"] == "rapid_layering"
    assert len(flag["involved_accounts"]) >= 4
    assert len(flag["evidence_transaction_ids"]) >= 3


def test_adversarial_evaluation_accuracy():
    eval_report = evaluate_adversarial_set()
    assert eval_report["accuracy"] == 1.0
    assert eval_report["passed_cases"] == eval_report["total_test_cases"]
    for detail in eval_report["details"]:
        assert detail["passed"] is True


def test_typology_api_endpoints(client):
    # 1. Health
    r = client.get("/api/v1/typology/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

    # 2. Typology flags
    r = client.get("/api/v1/typology/flags")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) > 0

    # 3. Synthetic Network structure
    r = client.get("/api/v1/typology/network")
    assert r.status_code == 200
    assert "nodes" in r.json()
    assert "edges" in r.json()

    # 4. Graph Walk
    r = client.get("/api/v1/typology/walk/ACC-SMURF-101?depth=2")
    assert r.status_code == 200
    walk = r.json()
    assert isinstance(walk, list)
    if walk:
        assert walk[0]["tier"] == "synthetic_network"
        assert "from_account" in walk[0]

    # 5. Adversarial evaluation report
    r = client.get("/api/v1/typology/evaluation")
    assert r.status_code == 200
    assert r.json()["accuracy"] == 1.0
