"""
FATF Typology Detection Rules & Network Graph Algorithms.
Person B: Implements mathematical detection algorithms for:
1. Structuring / Smurfing (FATF Recommendation 10 / Criteria 10.1 & 10.3)
2. Round-Tripping Circular Flows (FATF Beneficial Ownership Guidance)
3. Rapid Layering Pass-Throughs (FATF Money Laundering Typologies)

Output strictly conforms to the TypologyFlag schema in contracts/schemas.json.
"""

import datetime
from typing import Any

import networkx as nx
import pandas as pd


def build_networkx_graph(edges: list[dict[str, Any]]) -> nx.MultiDiGraph:
    """
    Constructs a directed multigraph from transaction edge records.
    """
    G = nx.MultiDiGraph()
    for e in edges:
        ts = e.get("timestamp")
        dt = pd.to_datetime(ts) if ts else datetime.datetime.now(datetime.timezone.utc)
        G.add_edge(
            e["from_account"],
            e["to_account"],
            id=e["id"],
            amount=float(e["amount"]),
            timestamp=ts,
            datetime=dt,
            payment_rail=e.get("payment_rail", "WIRE"),
            raw_narration=e.get("raw_narration", ""),
        )
    return G


def detect_structuring(
    G: nx.MultiDiGraph,
    threshold: float = 10000.0,
    lower_bound: float = 7500.0,
    window_hours: int = 24,
) -> list[dict[str, Any]]:
    """
    FATF Typology: Structuring / Smurfing.
    Detects multiple fan-in transactions just below reporting threshold into a single collector within window_hours.
    """
    flags: list[dict[str, Any]] = []

    for collector in G.nodes():
        in_edges = []
        for u, v, k, data in G.in_edges(collector, keys=True, data=True):
            if lower_bound <= data["amount"] < threshold:
                in_edges.append((u, data))

        if len(in_edges) < 3:
            continue

        # Sort in-edges by timestamp
        in_edges.sort(key=lambda x: x[1]["datetime"])

        # Sliding window check
        for i in range(len(in_edges)):
            window = [in_edges[i]]
            start_time = in_edges[i][1]["datetime"]

            for j in range(i + 1, len(in_edges)):
                time_diff = (
                    in_edges[j][1]["datetime"] - start_time
                ).total_seconds() / 3600.0
                if time_diff <= window_hours:
                    window.append(in_edges[j])

            unique_senders = list(dict.fromkeys([w[0] for w in window]))
            total_window_amount = sum(w[1]["amount"] for w in window)

            # If >=3 distinct senders and cumulative amount >= 2x threshold
            if len(unique_senders) >= 3 and total_window_amount >= (threshold * 1.8):
                evidence_ids = [w[1]["id"] for w in window]
                involved_accounts = unique_senders + [collector]

                # Check for duplicate flag on same collector
                already_flagged = any(
                    collector in f["involved_accounts"] for f in flags
                )
                if not already_flagged:
                    flags.append(
                        {
                            "flag_id": f"FLAG-STRUCTURING-{collector[-4:]}",
                            "typology": "structuring",
                            "fatf_reference": "FATF 40 Recommendations - Recommendation 10 & Criteria 10.1 / Sub-threshold Smurfing Patterns",
                            "involved_accounts": involved_accounts,
                            "evidence_transaction_ids": evidence_ids,
                            "confidence": 0.95,
                        }
                    )
                    break

    return flags


def detect_round_tripping(
    G: nx.MultiDiGraph,
    max_cycle_length: int = 5,
    min_volume_retention: float = 0.85,
    max_cycle_hours: int = 72,
) -> list[dict[str, Any]]:
    """
    FATF Typology: Round-Tripping (Circular Routing).
    Detects directed cycles where capital returns to originator entity preserving >= 85% volume within 72 hours.
    Uses time-forward bounded DFS for high-performance execution.
    """
    flags: list[dict[str, Any]] = []
    seen_cycles = set()
    # Defensive bound: this DFS has no memoization, so a denser real-world
    # graph (unlike the small synthetic demo network) could blow up
    # combinatorially. Cap total recursive calls per detection run instead
    # of rewriting the algorithm.
    max_recursive_calls = 50_000
    call_count = [0]

    def dfs_cycle(
        start_node: str,
        curr_node: str,
        path: list[str],
        edges_used: list[str],
        first_time: Any,
        first_amt: float,
        prev_time: Any,
        prev_amt: float,
    ):
        call_count[0] += 1
        if call_count[0] > max_recursive_calls:
            return
        if len(path) > max_cycle_length:
            return

        for _, next_node, k, edge_data in G.out_edges(curr_node, keys=True, data=True):
            curr_time = edge_data["datetime"]
            curr_amt = edge_data["amount"]
            curr_id = edge_data["id"]

            # Chronological check
            if prev_time is not None and curr_time < prev_time:
                continue

            # If returned to start node and valid length (>= 3 nodes)
            if next_node == start_node and len(path) >= 3:
                total_duration_hours = (curr_time - first_time).total_seconds() / 3600.0
                volume_retention = curr_amt / max(1.0, first_amt)

                if (
                    total_duration_hours <= max_cycle_hours
                    and volume_retention >= min_volume_retention
                ):
                    cycle_nodes = path
                    cycle_key = tuple(sorted(cycle_nodes))
                    if cycle_key not in seen_cycles:
                        seen_cycles.add(cycle_key)
                        flags.append(
                            {
                                "flag_id": f"FLAG-ROUNDTRIP-{start_node[-4:]}",
                                "typology": "round_tripping",
                                "fatf_reference": "FATF Guidance on Concealment of Beneficial Ownership (Oct 2018) - Circular Fund Routing",
                                "involved_accounts": cycle_nodes,
                                "evidence_transaction_ids": edges_used + [curr_id],
                                "confidence": 0.94,
                            }
                        )
                continue

            # Continue path if not repeating intermediate nodes and within the
            # inter-hop time gap limit
            if (
                next_node not in path
                and len(path) < max_cycle_length
                and (curr_time - first_time).total_seconds() / 3600.0 <= max_cycle_hours
            ):
                dfs_cycle(
                    start_node,
                    next_node,
                    path + [next_node],
                    edges_used + [curr_id],
                    first_time,
                    first_amt,
                    curr_time,
                    curr_amt,
                )

    for node in G.nodes():
        for _, dst, k, first_edge in G.out_edges(node, keys=True, data=True):
            if dst != node:
                dfs_cycle(
                    node,
                    dst,
                    [node, dst],
                    [first_edge["id"]],
                    first_edge["datetime"],
                    first_edge["amount"],
                    first_edge["datetime"],
                    first_edge["amount"],
                )

    return flags


def detect_rapid_layering(
    G: nx.MultiDiGraph,
    min_hops: int = 3,
    max_hop_hours: float = 3.0,
    min_pass_through_ratio: float = 0.85,
    min_transfer_amount: float = 1000.0,
) -> list[dict[str, Any]]:
    """
    FATF Typology: Rapid Layering.
    Detects high-velocity pass-through transit chains (>= 3 hops, inter-hop latency < 3h, amount >= $1,000).
    Uses time-constrained DFS for high-performance execution.
    """
    flags: list[dict[str, Any]] = []
    # Defensive bound, same rationale as detect_round_tripping's dfs_cycle:
    # no memoization, so cap total recursive calls per detection run.
    max_recursive_calls = 50_000
    call_count = [0]

    def dfs_layer(
        curr_node: str,
        current_path: list[str],
        current_edges: list[str],
        prev_time: Any,
        prev_amount: float,
    ):
        call_count[0] += 1
        if call_count[0] > max_recursive_calls:
            return
        if len(current_edges) >= min_hops:
            path_key = tuple(current_path)
            already_flagged = any(
                tuple(f["involved_accounts"]) == path_key for f in flags
            )
            if not already_flagged:
                flags.append(
                    {
                        "flag_id": f"FLAG-LAYERING-{current_path[0][-4:]}-{current_path[-1][-4:]}",
                        "typology": "rapid_layering",
                        "fatf_reference": "FATF Money Laundering Typologies - High-Velocity Pass-Through Transfers",
                        "involved_accounts": list(current_path),
                        "evidence_transaction_ids": list(current_edges),
                        "confidence": 0.92,
                    }
                )

        # Max hop depth of 6
        if len(current_edges) >= 6:
            return

        for _, next_node, k, edge_data in G.out_edges(curr_node, keys=True, data=True):
            if next_node in current_path:
                continue  # Avoid loops in layering chain

            curr_time = edge_data["datetime"]
            curr_amount = edge_data["amount"]

            if prev_time is not None:
                inter_hop_hours = (curr_time - prev_time).total_seconds() / 3600.0
                pass_ratio = curr_amount / max(1.0, prev_amount)

                # Check forward time ordering, max latency, and pass-through volume conservation
                if inter_hop_hours < 0 or inter_hop_hours > max_hop_hours:
                    continue
                if pass_ratio < min_pass_through_ratio or pass_ratio > 1.05:
                    continue

            dfs_layer(
                next_node,
                current_path + [next_node],
                current_edges + [edge_data["id"]],
                curr_time,
                curr_amount,
            )

    for start_node in G.nodes():
        for _, first_dst, k, first_edge in G.out_edges(
            start_node, keys=True, data=True
        ):
            if first_edge["amount"] >= min_transfer_amount:
                dfs_layer(
                    first_dst,
                    [start_node, first_dst],
                    [first_edge["id"]],
                    first_edge["datetime"],
                    first_edge["amount"],
                )

    return flags
