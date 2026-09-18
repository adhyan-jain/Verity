"""
FATF Typology Detector Service.
Person B: Evaluates graph data and produces TypologyFlag objects.
"""

from typing import List, Dict, Any
from .fatf_rules import detect_structuring, detect_round_tripping, detect_rapid_layering


def detect_all_typologies(network_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Runs all FATF typology detectors on the synthetic network.
    Returns list of TypologyFlag compliant records.
    """
    # TODO: Build networkx graph from network_data and run detection functions
    return []
