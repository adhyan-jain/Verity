"""
Narration Parser for Bank Ledger Data.
Person B: Parses raw narration strings from bank.xlsx to extract counterparty and payment modes.
"""

import pandas as pd


def parse_narrations(file_path: str = "data/raw/bank.xlsx") -> pd.DataFrame:
    """
    Parses transaction narrations from bank.xlsx into structured fields:
    (transaction_type, counterparty, channel, ref_id).
    """
    # TODO: Read excel, extract regex patterns for RTGS/NEFT/UPI/POS/ATM/CHQ
    return pd.DataFrame()
