"""
Unit tests for Fraud Dataset Ingestion, Temporal Feature Engineering, and Leakage Guarantees.
"""

import numpy as np
import pandas as pd
import pytest

from engines.fraud.datasets import (
    load_dataset,
    load_creditcard_dataset,
    load_kartik2112_dataset,
    generate_synthetic_kartik2112,
)
from engines.fraud.feature_engineering import (
    haversine_distance,
    compute_temporal_attributes,
    compute_causal_rolling_features,
    build_kartik2112_features,
    get_monthly_drift_stream,
)
from eval.evaluate_models import prepare_dataset_splits, train_and_evaluate_model


def test_haversine_distance_accuracy():
    # Known distance: NYC (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    d = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < d < 5650

    # Same point distance should be 0.0
    d_zero = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
    assert abs(d_zero) < 1e-4


def test_zero_lookahead_leakage_guarantee():
    """
    CRITICAL LEAKAGE TEST:
    Verifies that future transactions NEVER alter past historical feature aggregates.
    """
    # 1. Base transactions for User A over 3 hours
    base_data = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2020-01-01 10:00:00"),
            pd.Timestamp("2020-01-01 10:30:00"),
            pd.Timestamp("2020-01-01 12:00:00"),
        ],
        "cc_num": ["USER-A", "USER-A", "USER-A"],
        "amount": [100.0, 200.0, 300.0],
    })

    feats_base = compute_causal_rolling_features(base_data)
    
    # At t1 (10:00): No prior transactions -> velocity_1h=0, velocity_24h=0, ratio=1.0
    assert feats_base.loc[0, "velocity_1h"] == 0
    assert feats_base.loc[0, "velocity_24h"] == 0
    assert feats_base.loc[0, "amount_to_user_mean_ratio"] == 1.0

    # At t2 (10:30): 1 prior tx ($100) within 30 min -> velocity_1h=1, velocity_24h=1, ratio = 200 / 100 = 2.0
    assert feats_base.loc[1, "velocity_1h"] == 1
    assert feats_base.loc[1, "velocity_24h"] == 1
    assert feats_base.loc[1, "amount_to_user_mean_ratio"] == pytest.approx(2.0, rel=1e-3)

    # At t3 (12:00): 2 prior txs ($100, $200). t3 - t2 = 90 min (> 1h), so velocity_1h=0, velocity_24h=2.
    # Prior mean = (100 + 200) / 2 = 150. Ratio = 300 / 150 = 2.0
    assert feats_base.loc[2, "velocity_1h"] == 0
    assert feats_base.loc[2, "velocity_24h"] == 2
    assert feats_base.loc[2, "amount_to_user_mean_ratio"] == pytest.approx(2.0, rel=1e-3)

    # 2. Injected future transaction at t4 (15:00) with a massive $50,000 fraud
    future_data = pd.concat([
        base_data,
        pd.DataFrame({
            "timestamp": [pd.Timestamp("2020-01-01 15:00:00")],
            "cc_num": ["USER-A"],
            "amount": [50000.0],
        })
    ], ignore_index=True)

    feats_future = compute_causal_rolling_features(future_data)

    # Assert that t1, t2, and t3 are EXACTLY identical and unaffected by the future row
    for col in ["velocity_1h", "velocity_24h", "amount_to_user_mean_ratio"]:
        for i in range(3):
            assert feats_base.loc[i, col] == feats_future.loc[i, col], (
                f"LEAKAGE DETECTED in row {i}, column {col}: "
                f"Base={feats_base.loc[i, col]} vs Future={feats_future.loc[i, col]}"
            )


def test_creditcard_dataset_loader():
    df = load_dataset("creditcard", sample_size=1000)
    assert df["dataset_id"].iloc[0] == "creditcard"
    assert "amount" in df.columns
    assert "V1" in df.columns
    assert "V28" in df.columns
    assert "is_fraud" in df.columns
    assert len(df) == 1000


def test_kartik2112_dataset_loader_and_features():
    df = load_dataset("kartik2112", sample_size=500)
    assert df["dataset_id"].iloc[0] == "kartik2112"
    assert "cc_num" in df.columns
    assert "merchant" in df.columns
    assert "is_fraud" in df.columns

    df_feats = build_kartik2112_features(df)
    assert "haversine_distance" in df_feats.columns
    assert "hour_of_day" in df_feats.columns
    assert "is_night" in df_feats.columns
    assert "age_at_transaction" in df_feats.columns
    assert "velocity_1h" in df_feats.columns
    assert "velocity_24h" in df_feats.columns
    assert "amount_to_user_mean_ratio" in df_feats.columns


def test_monthly_drift_stream_partitioning():
    df = generate_synthetic_kartik2112(
        n_transactions=1000,
        start_date="2019-01-01",
        end_date="2020-03-31",
    )
    df["timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
    stream = get_monthly_drift_stream(df)

    assert len(stream) >= 3
    for month_key, month_df in stream.items():
        assert isinstance(month_key, str)
        assert len(month_df) > 0
        # Ensure chronological order within window
        assert month_df["timestamp"].is_monotonic_increasing


def test_eval_harness_single_split_pipeline():
    train_df, test_df, feature_cols = prepare_dataset_splits(
        dataset_id="creditcard", test_frac=0.2, sample_size=2000
    )
    assert len(train_df) == 1600
    assert len(test_df) == 400

    # Train LightGBM
    res = train_and_evaluate_model(
        model_name="LightGBM",
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        dataset_id="creditcard",
    )
    assert "precision" in res
    assert "recall" in res
    assert "f1" in res
    assert "pr_auc" in res
    assert res["dataset_id"] == "creditcard"
