"""
Drift Signal Empirical Verification Script.

Inspects prediction confidence distributions across the 2019-2020 monthly stream
to empirically confirm drift patterns (e.g., seasonal spending shifts, holiday anomalies)
for the KS-test drift detector.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.fraud.datasets import load_dataset
from engines.fraud.feature_engineering import build_kartik2112_features, get_monthly_drift_stream
from eval.evaluate_models import prepare_dataset_splits, train_and_evaluate_model
import lightgbm as lgb


def verify_monthly_drift():
    print("Loading 2019-2020 kartik2112 stream...")
    df = load_dataset("kartik2112", sample_size=60000)
    df_feats = build_kartik2112_features(df)
    
    stream = get_monthly_drift_stream(df_feats)
    print(f"Total partitioned monthly windows: {len(stream)} months")

    # Fit reference model on early 2019 data (first 4 months)
    ref_keys = [k for k in stream.keys() if k.startswith("2019-01") or k.startswith("2019-02") or k.startswith("2019-03") or k.startswith("2019-04")]
    ref_df = pd.concat([stream[k] for k in ref_keys], ignore_index=True)
    
    feature_cols = [
        "amount", "haversine_distance", "hour_of_day", "is_night",
        "age_at_transaction", "velocity_1h", "velocity_24h",
        "amount_to_user_mean_ratio", "city_pop"
    ]
    
    X_ref = ref_df[feature_cols].fillna(0.0)
    y_ref = ref_df["is_fraud"].values
    
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    model.fit(X_ref, y_ref)
    ref_scores = model.predict_proba(X_ref)[:, 1]

    print("\n--- Empirical Monthly Drift Across Windows (KS-test vs Reference Q1-2019) ---")
    print(f"{'Month':<10} {'Records':<10} {'Mean Score':<12} {'Std Score':<12} {'KS Stat':<10} {'p-value':<10} {'Drift Detected?'}")
    print("-" * 80)

    for month_key, month_df in stream.items():
        X_m = month_df[feature_cols].fillna(0.0)
        m_scores = model.predict_proba(X_m)[:, 1]
        
        ks_res = stats.ks_2samp(ref_scores, m_scores)
        drift_detected = "YES (p < 0.01)" if ks_res.pvalue < 0.01 else "NO"
        
        print(f"{month_key:<10} {len(month_df):<10} {np.mean(m_scores):<12.4f} {np.std(m_scores):<12.4f} {ks_res.statistic:<10.4f} {ks_res.pvalue:<10.2e} {drift_detected}")


if __name__ == "__main__":
    verify_monthly_drift()
