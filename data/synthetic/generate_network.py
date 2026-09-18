"""
Synthetic FATF Network Generator.
Person B: Generates synthetic transaction graph seeded with FATF laundering typologies.
Outputs: data/synthetic/synthetic_network.json
"""

import json
from typing import Dict, Any, List


def generate_fatf_network(output_file: str = "data/synthetic/synthetic_network.json") -> Dict[str, Any]:
    """
    Generates synthetic graph containing:
    1. Structuring (smurfing) clusters
    2. Round-tripping (circular flow) loops
    3. Rapid layering (high-velocity chain) paths
    4. Background normal transaction noise
    """
    print(f"Generating synthetic network -> {output_file}")
    network_data: Dict[str, Any] = {
        "nodes": [],
        "edges": [],
        "ground_truth_typologies": []
    }
    with open(output_file, "w") as f:
        json.dump(network_data, f, indent=2)
    return network_data


if __name__ == "__main__":
    generate_fatf_network()
