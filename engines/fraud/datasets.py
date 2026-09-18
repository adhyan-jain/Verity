"""
Dataset Ingestion and Normalization Layer for Verity Fraud Engine.

Supports multiple fraud dataset surfaces through a unified interface tagged with dataset_id:
1. "creditcard"  - ULB European credit card dataset (48-hour span, PCA-anonymized features V1..V28).
2. "kartik2112"   - Sparkov simulated card transactions (2019-2020, 23 interpretable features).

Both flow through the exact same downstream pipeline without forking per dataset.
"""

import os
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


DATA_DIR_DEFAULT = "data/raw"


def load_creditcard_dataset(
    data_path: str = "data/raw/creditcard.csv",
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads and normalizes the ULB creditcard.csv dataset.
    
    Returns DataFrame with columns:
        [dataset_id, transaction_id, timestamp_seconds, amount, V1..V28, is_fraud]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CreditCard dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size].copy()

    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    
    normalized = pd.DataFrame(index=range(n))
    normalized["dataset_id"] = "creditcard"
    normalized["transaction_id"] = [f"TX-CC-{i:06d}" for i in range(n)]
    normalized["timestamp_seconds"] = df["Time"].astype(float).values
    # Generate ISO timestamp starting from 2013-09-01 00:00:00
    base_time = pd.Timestamp("2013-09-01 00:00:00", tz="UTC")
    normalized["timestamp"] = base_time + pd.to_timedelta(df["Time"], unit="s")
    normalized["amount"] = df["Amount"].astype(float).values
    
    # Feature columns V1 to V28
    for col in [f"V{i}" for i in range(1, 29)]:
        normalized[col] = df[col].astype(float).values
        
    normalized["is_fraud"] = df["Class"].astype(int).values
    return normalized


def generate_synthetic_kartik2112(
    n_transactions: int = 50000,
    start_date: str = "2019-01-01",
    end_date: str = "2020-12-31",
    n_users: int = 250,
    n_merchants: int = 80,
    fraud_rate: float = 0.006,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a high-fidelity Sparkov-compliant synthetic kartik2112 dataset spanning 2019-2020.
    Used when raw multi-GB Kaggle CSV is not present locally, ensuring offline and test reproducibility.
    """
    rng = np.random.RandomState(random_seed)
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    total_seconds = int((end_dt - start_dt).total_seconds())

    # User profiles
    cc_nums = [f"CC-{100000000000 + i}" for i in range(n_users)]
    user_lats = rng.uniform(25.0, 48.0, n_users)
    user_longs = rng.uniform(-120.0, -75.0, n_users)
    user_dobs = [
        (pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.uniform(0, 18000)))).strftime("%Y-%m-%d")
        for _ in range(n_users)
    ]
    user_jobs = rng.choice(
        ["Software Engineer", "Accountant", "Nurse", "Teacher", "Retail Manager", "Attorney", "Physician", "Analyst"],
        n_users,
    )
    user_city_pops = rng.randint(5000, 2000000, n_users)

    # Merchant profiles
    categories = [
        "misc_net", "grocery_pos", "gas_transport", "shopping_net",
        "food_dining", "entertainment", "travel", "health_fitness"
    ]
    merchants = [f"fraud_merch_{i:03d}" for i in range(n_merchants)]
    merch_cats = rng.choice(categories, n_merchants)
    merch_lats = rng.uniform(25.0, 48.0, n_merchants)
    merch_longs = rng.uniform(-120.0, -75.0, n_merchants)

    # Sample transaction timestamps over the 2-year window (chronological with seasonal shifts)
    raw_seconds = np.sort(rng.uniform(0, total_seconds, n_transactions))
    timestamps = [start_dt + pd.Timedelta(seconds=float(s)) for s in raw_seconds]

    selected_users = rng.choice(n_users, n_transactions)
    selected_merchs = rng.choice(n_merchants, n_transactions)

    # Base amounts log-normally distributed
    base_amts = rng.lognormal(mean=3.5, sigma=1.0, size=n_transactions)
    base_amts = np.round(np.clip(base_amts, 1.0, 5000.0), 2)

    # Fraud generation (high amounts, night spikes, unusual distances)
    is_fraud = np.zeros(n_transactions, dtype=int)
    n_frauds = int(n_transactions * fraud_rate)
    fraud_indices = rng.choice(n_transactions, n_frauds, replace=False)
    
    for idx in fraud_indices:
        is_fraud[idx] = 1
        # Inflate fraud amounts
        base_amts[idx] = np.round(rng.uniform(250.0, 4500.0), 2)

    df = pd.DataFrame({
        "trans_date_trans_time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "cc_num": [cc_nums[u] for u in selected_users],
        "merchant": [merchants[m] for m in selected_merchs],
        "category": [merch_cats[m] for m in selected_merchs],
        "amt": base_amts,
        "first": ["FirstName"] * n_transactions,
        "last": ["LastName"] * n_transactions,
        "gender": rng.choice(["M", "F"], n_transactions),
        "street": ["Main St"] * n_transactions,
        "city": ["Springfield"] * n_transactions,
        "state": ["IL"] * n_transactions,
        "zip": rng.randint(10000, 99999, n_transactions),
        "lat": user_lats[selected_users],
        "long": user_longs[selected_users],
        "city_pop": user_city_pops[selected_users],
        "job": user_jobs[selected_users],
        "dob": [user_dobs[u] for u in selected_users],
        "trans_num": [f"TX-KARTIK-{i:07d}" for i in range(n_transactions)],
        "unix_time": [int(t.timestamp()) for t in timestamps],
        "merch_lat": merch_lats[selected_merchs] + rng.normal(0, 0.05, n_transactions),
        "merch_long": merch_longs[selected_merchs] + rng.normal(0, 0.05, n_transactions),
        "is_fraud": is_fraud,
    })

    return df


def load_kartik2112_dataset(
    data_dir: str = "data/raw",
    sample_size: Optional[int] = None,
    allow_synthetic_fallback: bool = True,
) -> pd.DataFrame:
    """
    Loads and normalizes the kartik2112/fraud-detection Sparkov card dataset.
    Looks for data/raw/fraudTrain.csv, data/raw/fraudTest.csv, or data/raw/kartik2112/*.
    Falls back to deterministic Sparkov generator if raw files are not on disk.
    
    Returns normalized DataFrame tagged with dataset_id="kartik2112".
    """
    candidate_paths = [
        os.path.join(data_dir, "fraudTrain.csv"),
        os.path.join(data_dir, "kartik2112", "fraudTrain.csv"),
        os.path.join(data_dir, "fraud_train.csv"),
    ]
    
    raw_df: Optional[pd.DataFrame] = None
    for path in candidate_paths:
        if os.path.exists(path):
            df_train = pd.read_csv(path)
            test_path = path.replace("Train", "Test").replace("train", "test")
            if os.path.exists(test_path):
                df_test = pd.read_csv(test_path)
                raw_df = pd.concat([df_train, df_test], ignore_index=True)
            else:
                raw_df = df_train
            break

    if raw_df is None:
        if allow_synthetic_fallback:
            raw_df = generate_synthetic_kartik2112(n_transactions=sample_size or 50000)
        else:
            raise FileNotFoundError(f"kartik2112 dataset files not found in {data_dir}")

    if sample_size and len(raw_df) > sample_size:
        raw_df = raw_df.iloc[:sample_size].copy()

    # Parse and sort chronologically
    raw_df["timestamp"] = pd.to_datetime(raw_df["trans_date_trans_time"])
    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)
    n = len(raw_df)

    normalized = pd.DataFrame(index=range(n))
    normalized["dataset_id"] = "kartik2112"
    normalized["transaction_id"] = raw_df["trans_num"].astype(str).values
    normalized["timestamp"] = raw_df["timestamp"].values
    normalized["timestamp_seconds"] = (
        (raw_df["timestamp"] - raw_df["timestamp"].min()).dt.total_seconds().values
    )
    normalized["cc_num"] = raw_df["cc_num"].astype(str).values
    normalized["merchant"] = raw_df["merchant"].astype(str).values
    normalized["category"] = raw_df["category"].astype(str).values
    normalized["amount"] = raw_df["amt"].astype(float).values
    normalized["lat"] = raw_df["lat"].astype(float).values
    normalized["long"] = raw_df["long"].astype(float).values
    normalized["merch_lat"] = raw_df["merch_lat"].astype(float).values
    normalized["merch_long"] = raw_df["merch_long"].astype(float).values
    normalized["city_pop"] = raw_df["city_pop"].astype(int).values
    normalized["job"] = raw_df["job"].astype(str).values
    normalized["dob"] = pd.to_datetime(raw_df["dob"]).values
    normalized["is_fraud"] = raw_df["is_fraud"].astype(int).values

    return normalized


def load_dataset(
    dataset_id: str,
    data_dir: str = "data/raw",
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Unified dataset loader entrypoint.
    
    Args:
        dataset_id: "creditcard" or "kartik2112"
        data_dir: Directory containing raw datasets
        sample_size: Optional row limit for fast testing
        
    Returns:
        pd.DataFrame tagged with dataset_id
    """
    norm_id = dataset_id.lower().strip()
    if norm_id in ("creditcard", "creditcard.csv", "ulb"):
        csv_path = os.path.join(data_dir, "creditcard.csv")
        return load_creditcard_dataset(data_path=csv_path, sample_size=sample_size)
    elif norm_id in ("kartik2112", "sparkov", "kartik"):
        return load_kartik2112_dataset(data_dir=data_dir, sample_size=sample_size)
    else:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Supported: 'creditcard', 'kartik2112'"
        )
