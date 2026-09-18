"""
Synthetic FATF Laundering Network Generator.
Person B: Generates realistic multi-party financial graphs seeded with:
1. Structuring / Smurfing clusters (FATF Rec. 10)
2. Round-Tripping circular loops (FATF Beneficial Ownership Guidance)
3. Rapid Layering transit chains (FATF Pass-Through Typologies)
4. Benign commercial & retail background noise

Outputs canonical structure to data/synthetic/synthetic_network.json.
"""

import os
import json
import random
import datetime
from typing import Dict, Any, List


def generate_fatf_network(
    output_file: str = "data/synthetic/synthetic_network.json",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates a deterministic synthetic financial transaction network.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    base_time = datetime.datetime(2026, 9, 10, 8, 0, 0)
    
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    ground_truth_flags: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Define Synthetic Entities (Nodes)
    # -------------------------------------------------------------
    entity_configs = [
        # Structuring Cluster 1
        {"id": "ACC-SMURF-101", "name": "Apex Micro Holdings Ltd", "entity_type": "shell_company", "risk_rating": "high"},
        {"id": "ACC-SMURF-102", "name": "Vortex Trading FZE", "entity_type": "import_export", "risk_rating": "high"},
        {"id": "ACC-SMURF-103", "name": "Zenith General Trading", "entity_type": "retail", "risk_rating": "medium"},
        {"id": "ACC-SMURF-104", "name": "BlueSky Consulting DMCC", "entity_type": "consultancy", "risk_rating": "high"},
        {"id": "ACC-COLLECTOR-100", "name": "Grand Horizon Capital Partners", "entity_type": "holding_co", "risk_rating": "critical"},

        # Round-Tripping Cluster 1
        {"id": "ACC-RT-201", "name": "Silverline Logistics Pte", "entity_type": "logistics", "risk_rating": "high"},
        {"id": "ACC-RT-202", "name": "Offshore Conduit BVI Ltd", "entity_type": "shell_company", "risk_rating": "critical"},
        {"id": "ACC-RT-203", "name": "Meridian Trade Finance Corp", "entity_type": "trade_finance", "risk_rating": "high"},
        {"id": "ACC-RT-204", "name": "Global Equity Advisory Ltd", "entity_type": "investment", "risk_rating": "high"},

        # Rapid Layering Chain 1
        {"id": "ACC-LAYER-301", "name": "Originator Fund Escrow", "entity_type": "escrow", "risk_rating": "medium"},
        {"id": "ACC-LAYER-302", "name": "Hop-1 FastBridge Transit", "entity_type": "msb_payment", "risk_rating": "critical"},
        {"id": "ACC-LAYER-303", "name": "Hop-2 Continental Clearing", "entity_type": "intermediary", "risk_rating": "high"},
        {"id": "ACC-LAYER-304", "name": "Hop-3 Pacific Settle Inc", "entity_type": "intermediary", "risk_rating": "high"},
        {"id": "ACC-LAYER-305", "name": "Beneficiary Apex Assets LLC", "entity_type": "asset_holding", "risk_rating": "critical"},

        # Structuring Cluster 2
        {"id": "ACC-SMURF-401", "name": "Kite Global Suppliers", "entity_type": "retail", "risk_rating": "medium"},
        {"id": "ACC-SMURF-402", "name": "Nova Tech Consulting", "entity_type": "services", "risk_rating": "high"},
        {"id": "ACC-SMURF-403", "name": "Prime Wave Trading", "entity_type": "shell_company", "risk_rating": "high"},
        {"id": "ACC-COLLECTOR-400", "name": "Starlight Asset Management", "entity_type": "wealth_mgmt", "risk_rating": "critical"},

        # Round-Tripping Cluster 2
        {"id": "ACC-RT-501", "name": "Crown Infrastructure Group", "entity_type": "construction", "risk_rating": "high"},
        {"id": "ACC-RT-502", "name": "Seychelles Special Ventures", "entity_type": "offshore_spv", "risk_rating": "critical"},
        {"id": "ACC-RT-503", "name": "Lumina Capital Holdings", "entity_type": "holding_co", "risk_rating": "high"},

        # Benign Commercial & Retail Accounts
        {"id": "ACC-BENIGN-001", "name": "Metro Enterprise Payroll Desk", "entity_type": "corporate", "risk_rating": "low"},
        {"id": "ACC-BENIGN-002", "name": "National Utility Distribution", "entity_type": "utility", "risk_rating": "low"},
        {"id": "ACC-BENIGN-003", "name": "Sunrise Agro Supplies", "entity_type": "agriculture", "risk_rating": "low"},
        {"id": "ACC-BENIGN-004", "name": "TechCore Software Solutions", "entity_type": "it_services", "risk_rating": "low"},
        {"id": "ACC-BENIGN-005", "name": "Summit Medical Supplies", "entity_type": "healthcare", "risk_rating": "low"},
        {"id": "ACC-BENIGN-006", "name": "Reliance Industrial Parts", "entity_type": "manufacturing", "risk_rating": "low"},
        {"id": "ACC-BENIGN-007", "name": "Alpha Retail Superstores", "entity_type": "retail", "risk_rating": "low"},
        {"id": "ACC-BENIGN-008", "name": "Omni Logistics Fleet", "entity_type": "transport", "risk_rating": "low"}
    ]

    for entity in entity_configs:
        nodes.append({
            "account_id": entity["id"],
            "account_name": entity["name"],
            "entity_type": entity["entity_type"],
            "risk_rating": entity["risk_rating"],
            "tier": "synthetic_network"
        })

    tx_counter = 1

    # -------------------------------------------------------------
    # 2. Seed Typology 1: Structuring / Smurfing (Cluster 1)
    # -------------------------------------------------------------
    smurf_tids_1 = []
    smurf_senders_1 = ["ACC-SMURF-101", "ACC-SMURF-102", "ACC-SMURF-103", "ACC-SMURF-104"]
    collector_1 = "ACC-COLLECTOR-100"
    
    structuring_base_time = base_time + datetime.timedelta(days=1, hours=2)
    for i, sender in enumerate(smurf_senders_1):
        # Transactions right below 10,000 threshold ($9,200 - $9,800)
        amt = round(random.uniform(9200.0, 9850.0), 2)
        tx_time = structuring_base_time + datetime.timedelta(hours=i * 2 + random.uniform(0.1, 1.2))
        tid = f"TX-SYNTH-{tx_counter:04d}"
        tx_counter += 1
        smurf_tids_1.append(tid)
        
        edges.append({
            "id": tid,
            "tier": "synthetic_network",
            "from_account": sender,
            "to_account": collector_1,
            "amount": amt,
            "timestamp": tx_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payment_rail": "RTGS",
            "raw_narration": f"SETTLEMENT INVOICE #{random.randint(1000, 9999)}",
            "source_dataset": "synthetic_network.json"
        })
        
    ground_truth_flags.append({
        "flag_id": "FLAG-FATF-SYN-001",
        "typology": "structuring",
        "fatf_reference": "FATF 40 Recommendations - Recommendation 10 & Criteria 10.1 / Sub-threshold Smurfing",
        "involved_accounts": smurf_senders_1 + [collector_1],
        "evidence_transaction_ids": smurf_tids_1,
        "confidence": 0.96
    })

    # -------------------------------------------------------------
    # 3. Seed Typology 2: Round-Tripping (Cluster 1)
    # -------------------------------------------------------------
    rt_tids_1 = []
    rt_cycle_nodes = ["ACC-RT-201", "ACC-RT-202", "ACC-RT-203", "ACC-RT-204", "ACC-RT-201"]
    rt_base_time = base_time + datetime.timedelta(days=2, hours=4)
    current_amt = 85000.0  # Initial outbound capital
    
    for i in range(len(rt_cycle_nodes) - 1):
        src = rt_cycle_nodes[i]
        dst = rt_cycle_nodes[i + 1]
        # Retain 96-98% after minor intermediary fees
        current_amt = round(current_amt * random.uniform(0.97, 0.99), 2)
        tx_time = rt_base_time + datetime.timedelta(hours=i * 8 + random.uniform(0.5, 2.0))
        tid = f"TX-SYNTH-{tx_counter:04d}"
        tx_counter += 1
        rt_tids_1.append(tid)
        
        edges.append({
            "id": tid,
            "tier": "synthetic_network",
            "from_account": src,
            "to_account": dst,
            "amount": current_amt,
            "timestamp": tx_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payment_rail": "CROSS_BORDER_WIRE",
            "raw_narration": f"OVERSEAS CONSULTING SETTLEMENT #{random.randint(4000, 8999)}",
            "source_dataset": "synthetic_network.json"
        })
        
    ground_truth_flags.append({
        "flag_id": "FLAG-FATF-SYN-002",
        "typology": "round_tripping",
        "fatf_reference": "FATF Guidance on Concealment of Beneficial Ownership (Oct 2018) - Circular Capital Flow",
        "involved_accounts": list(dict.fromkeys(rt_cycle_nodes)),
        "evidence_transaction_ids": rt_tids_1,
        "confidence": 0.94
    })

    # -------------------------------------------------------------
    # 4. Seed Typology 3: Rapid Layering (Chain 1)
    # -------------------------------------------------------------
    layer_tids_1 = []
    layer_chain_nodes = ["ACC-LAYER-301", "ACC-LAYER-302", "ACC-LAYER-303", "ACC-LAYER-304", "ACC-LAYER-305"]
    layer_base_time = base_time + datetime.timedelta(days=3, hours=1)
    pass_amt = 120000.0
    
    for i in range(len(layer_chain_nodes) - 1):
        src = layer_chain_nodes[i]
        dst = layer_chain_nodes[i + 1]
        # Fast pass-through: ~98% pass-through within 45-75 minutes per hop
        pass_amt = round(pass_amt * random.uniform(0.985, 0.995), 2)
        tx_time = layer_base_time + datetime.timedelta(minutes=i * 55 + random.randint(5, 20))
        tid = f"TX-SYNTH-{tx_counter:04d}"
        tx_counter += 1
        layer_tids_1.append(tid)
        
        edges.append({
            "id": tid,
            "tier": "synthetic_network",
            "from_account": src,
            "to_account": dst,
            "amount": pass_amt,
            "timestamp": tx_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payment_rail": "HIGH_SPEED_IMPS",
            "raw_narration": f"PASS THROUGH ESCROW RELAY #{random.randint(500, 999)}",
            "source_dataset": "synthetic_network.json"
        })
        
    ground_truth_flags.append({
        "flag_id": "FLAG-FATF-SYN-003",
        "typology": "rapid_layering",
        "fatf_reference": "FATF Money Laundering Typologies - High-Velocity Multi-Hop Pass-Through Accounts",
        "involved_accounts": layer_chain_nodes,
        "evidence_transaction_ids": layer_tids_1,
        "confidence": 0.92
    })

    # -------------------------------------------------------------
    # 5. Seed Background Benign Commercial Traffic (200+ edges)
    # -------------------------------------------------------------
    all_node_ids = [n["account_id"] for n in nodes]
    benign_node_ids = [n["account_id"] for n in nodes if "BENIGN" in n["account_id"]]
    
    for day in range(7):
        current_day = base_time + datetime.timedelta(days=day)
        # Payroll batch on Day 3
        if day == 3:
            for b_acc in benign_node_ids[:5]:
                amt = round(random.uniform(2500.0, 6500.0), 2)
                tid = f"TX-SYNTH-{tx_counter:04d}"
                tx_counter += 1
                edges.append({
                    "id": tid,
                    "tier": "synthetic_network",
                    "from_account": "ACC-BENIGN-001",
                    "to_account": b_acc,
                    "amount": amt,
                    "timestamp": (current_day + datetime.timedelta(hours=9, minutes=random.randint(0, 50))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "payment_rail": "NEFT",
                    "raw_narration": "MONTHLY SALARY / PAYROLL DIRECT DISBURSEMENT",
                    "source_dataset": "synthetic_network.json"
                })
        
        # Vendor invoicing & commercial flows
        for _ in range(25):
            src = random.choice(all_node_ids)
            dst = random.choice([n for n in all_node_ids if n != src])
            amt = round(random.uniform(150.0, 15000.0), 2)
            hour = random.randint(8, 20)
            minute = random.randint(0, 59)
            tid = f"TX-SYNTH-{tx_counter:04d}"
            tx_counter += 1
            
            edges.append({
                "id": tid,
                "tier": "synthetic_network",
                "from_account": src,
                "to_account": dst,
                "amount": amt,
                "timestamp": (current_day + datetime.timedelta(hours=hour, minutes=minute)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "payment_rail": random.choice(["NEFT", "RTGS", "UPI", "CHQ"]),
                "raw_narration": f"COMMERCIAL INVOICE PMT #{random.randint(10000, 99999)}",
                "source_dataset": "synthetic_network.json"
            })

    # Sort edges chronologically
    edges.sort(key=lambda x: x["timestamp"])

    network_output = {
        "metadata": {
            "tier": "synthetic_network",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "typology_count": len(ground_truth_flags)
        },
        "nodes": nodes,
        "edges": edges,
        "ground_truth_flags": ground_truth_flags
    }

    with open(output_file, "w") as f:
        json.dump(network_output, f, indent=2)

    print(f"Generated synthetic FATF network with {len(nodes)} nodes, {len(edges)} transactions, and {len(ground_truth_flags)} seeded typologies -> {output_file}")
    return network_output


if __name__ == "__main__":
    generate_fatf_network()
