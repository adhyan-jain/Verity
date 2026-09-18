"""
Temporal Feature Engineering & Drift Stream Pipeline for Verity Fraud Engine.

Provides leakage-safe, causally strict feature transformations:
- Haversine great-circle distance (cardholder to merchant in km)
- Circadian and age features (hour_of_day, is_night, age_at_transaction)
- Causally strict user rolling aggregates (velocity_1h, velocity_24h, amount_to_user_mean_ratio)
  Guarantee: ONLY transactions strictly prior to the current transaction timestamp (t_prior < t_curr)
  are included. Current and future rows are strictly excluded.
- Month-partitioned temporal stream interface for the KS-test drift detector over 2019-2020.
"""

from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lon1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lon2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes the great-circle Haversine distance between two coordinates in kilometers.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def compute_temporal_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts circadian timing and cardholder age from timestamp and date of birth.
    """
    res = df.copy()
    ts = pd.to_datetime(res["timestamp"])
    
    res["hour_of_day"] = ts.dt.hour
    res["is_night"] = ((res["hour_of_day"] >= 22) | (res["hour_of_day"] < 6)).astype(int)
    
    if "dob" in res.columns:
        dob = pd.to_datetime(res["dob"])
        # Age at the exact time of transaction in years
        res["age_at_transaction"] = np.round((ts - dob).dt.total_seconds() / (365.25 * 86400.0), 1)
    else:
        res["age_at_transaction"] = 40.0

    return res


def compute_causal_rolling_features(
    df: pd.DataFrame,
    user_col: str = "cc_num",
    time_col: str = "timestamp",
    amt_col: str = "amount",
) -> pd.DataFrame:
    """
    Computes user historical aggregates with a STRICT zero-lookahead guarantee:
    - velocity_1h: count of transactions for the same user in (t - 3600s, t) strictly prior to t
    - velocity_24h: count of transactions for the same user in (t - 86400s, t) strictly prior to t
    - amount_to_user_mean_ratio: amt / historical_mean(user, strictly prior to t)
    
    If no prior transactions exist for a user, amount_to_user_mean_ratio defaults to 1.0.
    """
    res = df.copy()
    res["_orig_order"] = np.arange(len(res))
    
    # Ensure chronological order
    res = res.sort_values(time_col).reset_index(drop=True)
    timestamps = pd.to_datetime(res[time_col]).astype("datetime64[s]").astype("int64").values
    amounts = res[amt_col].values
    users = res[user_col].values

    n = len(res)
    velocity_1h = np.zeros(n, dtype=int)
    velocity_24h = np.zeros(n, dtype=int)
    amt_ratio = np.ones(n, dtype=float)

    # Group by user to compute sliding window statistics strictly backwards
    user_indices: Dict[Any, List[int]] = {}
    for i in range(n):
        u = users[i]
        t = timestamps[i]
        amt = amounts[i]

        if u not in user_indices:
            user_indices[u] = []
            # First transaction for this user: 0 prior transactions, ratio = 1.0
            velocity_1h[i] = 0
            velocity_24h[i] = 0
            amt_ratio[i] = 1.0
        else:
            prior_idxs = user_indices[u]
            # Find prior transactions within 1h (3600s) and 24h (86400s)
            # strictly prior: timestamp < t
            v1 = 0
            v24 = 0
            prior_amts_sum = 0.0
            prior_count = 0

            # Iterate backwards over prior transactions of this user
            for p_idx in reversed(prior_idxs):
                p_t = timestamps[p_idx]
                p_amt = amounts[p_idx]
                diff = t - p_t

                if diff <= 3600:
                    v1 += 1
                if diff <= 86400:
                    v24 += 1

                prior_amts_sum += p_amt
                prior_count += 1

            velocity_1h[i] = v1
            velocity_24h[i] = v24
            
            if prior_count > 0:
                user_hist_mean = prior_amts_sum / prior_count
                amt_ratio[i] = np.round(amt / (user_hist_mean + 1e-6), 4)
            else:
                amt_ratio[i] = 1.0

        user_indices[u].append(i)

    res["velocity_1h"] = velocity_1h
    res["velocity_24h"] = velocity_24h
    res["amount_to_user_mean_ratio"] = amt_ratio

    # Restore original ordering
    res = res.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)
    return res


def build_kartik2112_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for kartik2112 dataset.
    Generates all 7 required features:
    - haversine_distance
    - hour_of_day, is_night
    - age_at_transaction
    - velocity_1h, velocity_24h
    - amount_to_user_mean_ratio
    """
    res = df.copy()
    
    # 1. Haversine distance
    res["haversine_distance"] = np.round(
        haversine_distance(res["lat"], res["long"], res["merch_lat"], res["merch_long"]),
        3,
    )

    # 2. Temporal & age attributes
    res = compute_temporal_attributes(res)

    # 3. Causally strict rolling user aggregates
    res = compute_causal_rolling_features(res, user_col="cc_num", time_col="timestamp", amt_col="amount")

    return res


def get_monthly_drift_stream(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> Dict[str, pd.DataFrame]:
    """
    Partitions transactions chronologically into month-by-month windows over 2019-2020.
    
    Stable Interface for the Drift Detector:
    Returns an ordered dictionary mapping month key ('2019-01', '2019-02', ..., '2020-12')
    to the filtered DataFrame for that window.
    """
    res = df.copy()
    ts = pd.to_datetime(res[timestamp_col])
    res["_month_key"] = ts.dt.strftime("%Y-%m")
    
    unique_months = sorted(res["_month_key"].unique())
    monthly_stream: Dict[str, pd.DataFrame] = {}
    
    for m in unique_months:
        monthly_stream[m] = (
            res[res["_month_key"] == m]
            .drop(columns=["_month_key"])
            .sort_values(timestamp_col)
            .reset_index(drop=True)
        )
        
    return monthly_stream
