"""
Benchmark Class Imbalance Techniques on Credit Card Fraud Dataset.
Person A: Run SMOTE vs Class Weighting comparison on recall, precision, F1, PR-AUC, and ROC-AUC.
"""

import time
from typing import Any

import numpy as np
import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def run_benchmark(
    data_path: str = "data/raw/creditcard.csv", random_state: int = 42
) -> dict[str, Any]:
    """
    Evaluates SMOTE vs Class-Weighting on creditcard fraud dataset.
    Applies the EXACT SAME 5-fold Stratified CV threshold calibration
    procedure to both techniques to ensure a fair, rigorous comparison.
    """
    print(f"Benchmarking imbalance strategies on {data_path}...")
    df = pd.read_csv(data_path)
    df = df.sort_values("Time").reset_index(drop=True)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Time-based train/test split (80/20) matching train.py
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"Dataset split (time-based): {len(X_train)} train ({y_train.sum()} fraud), "
        f"{len(X_test)} test ({y_test.sum()} fraud)"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # -------------------------------------------------------------------------
    # 1. Class Weighting Strategy (with 5-fold CV threshold calibration)
    # -------------------------------------------------------------------------
    print("\n[1/2] Evaluating Class Weighting (scale_pos_weight)...")
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    oof_probas_cw = np.zeros(len(X_train))
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_va = X_train.iloc[val_idx]

        clf_cw_fold = lgb.LGBMClassifier(
            scale_pos_weight=pos_weight,
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf_cw_fold.fit(X_tr, y_tr)
        oof_probas_cw[val_idx] = clf_cw_fold.predict_proba(X_va)[:, 1]

    prec_cw, rec_cw, th_cw = precision_recall_curve(y_train, oof_probas_cw)
    f1_cw = 2 * (prec_cw * rec_cw) / (prec_cw + rec_cw + 1e-10)
    best_idx_cw = int(np.argmax(f1_cw[: len(th_cw)]))
    calibrated_th_cw = float(th_cw[best_idx_cw])

    clf_cw = lgb.LGBMClassifier(
        scale_pos_weight=pos_weight,
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    clf_cw.fit(X_train, y_train)
    cw_train_time = round(time.time() - t0, 3)

    y_proba_cw = clf_cw.predict_proba(X_test)[:, 1]
    y_pred_cw_default = (y_proba_cw >= 0.5).astype(int)
    y_pred_cw_calibrated = (y_proba_cw >= calibrated_th_cw).astype(int)

    cw_metrics = {
        "uncalibrated_th_0.50": {
            "threshold": 0.50,
            "precision": round(float(precision_score(y_test, y_pred_cw_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_cw_default)), 4),
            "f1": round(float(f1_score(y_test, y_pred_cw_default)), 4),
        },
        "calibrated_oof_th": {
            "calibrated_threshold": round(calibrated_th_cw, 4),
            "precision": round(float(precision_score(y_test, y_pred_cw_calibrated, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_cw_calibrated)), 4),
            "f1": round(float(f1_score(y_test, y_pred_cw_calibrated)), 4),
        },
        "pr_auc": round(float(average_precision_score(y_test, y_proba_cw)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_cw)), 4),
        "training_time_seconds": cw_train_time,
    }

    # -------------------------------------------------------------------------
    # 2. SMOTE Strategy (with 5-fold CV threshold calibration, ratio 0.05)
    # -------------------------------------------------------------------------
    print("\n[2/2] Evaluating SMOTE (sampling_strategy=0.05)...")
    oof_probas_smote = np.zeros(len(X_train))
    t0 = time.time()

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_va = X_train.iloc[val_idx]

        smote_fold = SMOTE(sampling_strategy=0.05, random_state=random_state + fold)
        X_tr_res, y_tr_res = smote_fold.fit_resample(X_tr, y_tr)

        clf_smote_fold = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf_smote_fold.fit(X_tr_res, y_tr_res)
        oof_probas_smote[val_idx] = clf_smote_fold.predict_proba(X_va)[:, 1]

    prec_sm, rec_sm, th_sm = precision_recall_curve(y_train, oof_probas_smote)
    f1_sm = 2 * (prec_sm * rec_sm) / (prec_sm + rec_sm + 1e-10)
    best_idx_sm = int(np.argmax(f1_sm[: len(th_sm)]))
    calibrated_th_smote = float(th_sm[best_idx_sm])

    smote = SMOTE(sampling_strategy=0.05, random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    clf_smote = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    clf_smote.fit(X_train_res, y_train_res)
    smote_train_time = round(time.time() - t0, 3)

    y_proba_smote = clf_smote.predict_proba(X_test)[:, 1]
    y_pred_smote_default = (y_proba_smote >= 0.5).astype(int)
    y_pred_smote_calibrated = (y_proba_smote >= calibrated_th_smote).astype(int)

    smote_metrics = {
        "uncalibrated_th_0.50": {
            "threshold": 0.50,
            "precision": round(float(precision_score(y_test, y_pred_smote_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_smote_default)), 4),
            "f1": round(float(f1_score(y_test, y_pred_smote_default)), 4),
        },
        "calibrated_oof_th": {
            "calibrated_threshold": round(calibrated_th_smote, 4),
            "precision": round(float(precision_score(y_test, y_pred_smote_calibrated, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_smote_calibrated)), 4),
            "f1": round(float(f1_score(y_test, y_pred_smote_calibrated)), 4),
        },
        "pr_auc": round(float(average_precision_score(y_test, y_proba_smote)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_smote)), 4),
        "training_time_seconds": smote_train_time,
    }

    winner = (
        "smote"
        if smote_metrics["calibrated_oof_th"]["f1"] >= cw_metrics["calibrated_oof_th"]["f1"]
        else "class_weighting"
    )

    results = {
        "dataset": data_path,
        "split_strategy": "time_based_80_20",
        "total_records": len(df),
        "train_fraud_cases": int(y_train.sum()),
        "test_fraud_cases": int(y_test.sum()),
        "class_weighting": cw_metrics,
        "smote": smote_metrics,
        "winner": winner,
        "recommendation": (
            f"Selected {winner.upper()} strategy due to superior F1 ({smote_metrics['calibrated_oof_th']['f1']} vs "
            f"{cw_metrics['calibrated_oof_th']['f1']}) and PR-AUC ({smote_metrics['pr_auc']} vs {cw_metrics['pr_auc']}) "
            f"under identical 5-fold CV threshold calibration."
        ),
    }

    return results


if __name__ == "__main__":
    import json
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    benchmark_results = run_benchmark()
    print("\nBenchmark complete:")
    print(json.dumps(benchmark_results, indent=2))

