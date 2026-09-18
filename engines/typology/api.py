"""
FastAPI Service for Typology & Synthetic Network Engine.
Person B: Serves FATF typology flags, multi-hop GraphWalkStep trajectories,
and interactive synthetic graph structures for the analyst dashboard.
"""

from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .detect import (
        detect_all_typologies,
        evaluate_adversarial_set,
        load_synthetic_network,
    )
except (ImportError, ValueError):
    from detect import (
        detect_all_typologies,
        evaluate_adversarial_set,
        load_synthetic_network,
    )

app = FastAPI(
    title="Verity Typology Engine API",
    description="FATF Typology Detection & Synthetic Network Graph Service",
    version="1.0.0",
)

# Enable CORS for local dev dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_NETWORK_CACHE: dict[str, Any] | None = None
_FLAGS_CACHE: list[dict[str, Any]] | None = None


def get_network() -> dict[str, Any]:
    global _NETWORK_CACHE
    if _NETWORK_CACHE is None:
        _NETWORK_CACHE = load_synthetic_network()
    return _NETWORK_CACHE


def get_flags() -> list[dict[str, Any]]:
    global _FLAGS_CACHE
    if _FLAGS_CACHE is None:
        network = get_network()
        _FLAGS_CACHE = detect_all_typologies(network)
    return _FLAGS_CACHE


@app.get("/api/v1/typology/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "typology-engine",
        "tier": "synthetic_network",
    }


@app.get("/api/v1/typology/flags")
def list_typology_flags() -> list[dict[str, Any]]:
    """
    Returns all detected TypologyFlag records on the synthetic laundering network.
    """
    return get_flags()


@app.get("/api/v1/typology/network")
def get_synthetic_network() -> dict[str, Any]:
    """
    Returns full node-edge structure and metadata of the synthetic FATF network for UI graph rendering.
    """
    return get_network()


@app.get("/api/v1/typology/walk/{account_id}")
def walk_synthetic_graph(
    account_id: str, depth: int = Query(default=2, ge=1, le=5)
) -> list[dict[str, Any]]:
    """
    Performs multi-party graph traversal starting from account_id up to specified depth.
    Returns GraphWalkStep schema compliant records.
    """
    network = get_network()
    edges = network.get("edges", [])

    clean_acc = str(account_id).strip()

    # BFS traversal up to depth
    visited_nodes = {clean_acc}
    queue = [(clean_acc, 0)]
    walk_steps: list[dict[str, Any]] = []
    step_idx = 1

    while queue:
        curr_node, curr_depth = queue.pop(0)
        if curr_depth >= depth:
            continue

        # Find outbound edges from curr_node
        outbound = [e for e in edges if e["from_account"] == curr_node]
        for e in outbound:
            target = e["to_account"]
            walk_steps.append(
                {
                    "step_index": step_idx,
                    "from_account": e["from_account"],
                    "to_account": e["to_account"],
                    "tier": "synthetic_network",
                    "amount": float(e["amount"]),
                    "timestamp": e["timestamp"],
                    "narration": e.get("raw_narration", ""),
                    "tool_call_id": f"TOOL-CALL-SYN-{step_idx:03d}",
                }
            )
            step_idx += 1

            if target not in visited_nodes:
                visited_nodes.add(target)
                queue.append((target, curr_depth + 1))

    return walk_steps


@app.get("/api/v1/typology/evaluation")
def get_adversarial_evaluation() -> dict[str, Any]:
    """
    Returns precision/recall and evaluation report against held-out adversarial test cases.
    """
    return evaluate_adversarial_set()


if __name__ == "__main__":
    import uvicorn

    print("Starting Typology Engine API on http://127.0.0.1:8003 ...")
    uvicorn.run("engines.typology.api:app", host="127.0.0.1", port=8003, reload=False)
