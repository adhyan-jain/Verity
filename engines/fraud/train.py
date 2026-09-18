"""
Production Model Training Pipeline for Fraud Detection.
Trains authoritative fraud classification model on creditcard.csv using SMOTE oversampling
and XGBoost / LightGBM, calibrated using 5-fold out-of-fold cross-validation on time-based split.
"""

import hashlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    versions = {}
    for pkg in ("xgboost", "lightgbm", "scikit-learn", "shap", "imbalanced-learn", "pandas", "numpy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_fraud_model(
    data_path: str = "data/raw/creditcard.csv",
    output_path: str = "engines/fraud/model.pkl",
    random_state: int = 42,
    model_type: str = "xgboost",
    n_estimators: int = 250,
    smote_ratio: float = 0.05,
    benchmark_path: Optional[str] = "engines/fraud/benchmark_results.json",
) -> Dict[str, Any]:
    """
    Trains the authoritative fraud detection model on creditcard.csv.
    Uses chronological time-based train/test split (80% train, 20% test).
    Calibrates decision threshold via 5-Fold Stratified Cross-Validation on the training set.
    Fits SHAP TreeExplainer on the exact trained model.
    """
    print(f"[Train] Training authoritative fraud detection model ({model_type}) from {data_path}...")
    df = pd.read_csv(data_path)
    dataset_hash = _file_sha256(data_path)
    df = df.sort_values("Time").reset_index(drop=True)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    feature_names = X.columns.tolist()

    # Time-based train/test split (80/20): test is most recent 20% by Time
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"[Train] Chronological split: {len(X_train):,} train ({int(y_train.sum()):,} frauds), "
        f"{len(X_test):,} test ({int(y_test.sum()):,} frauds)"
    )

    # 5-Fold Stratified Cross Validation on Training Data for OOF threshold calibration
    print("[Train] Running 5-Fold Stratified Cross-Validation for threshold calibration...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probas = np.zeros(len(X_train))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_va = X_train.iloc[val_idx]

        smote_fold = SMOTE(
            sampling_strategy=smote_ratio, random_state=random_state + fold
        )
        X_tr_res, y_tr_res = smote_fold.fit_resample(X_tr, y_tr)

        if model_type.lower() == "xgboost":
            fold_clf = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state + fold,
                n_jobs=-1,
                eval_metric="logloss",
            )
        else:
            fold_clf = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state + fold,
                n_jobs=-1,
                verbose=-1,
            )
        fold_clf.fit(X_tr_res, y_tr_res)
        oof_probas[val_idx] = fold_clf.predict_proba(X_va)[:, 1]

    # Calculate optimal threshold on OOF predictions maximizing F1
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probas)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores[: len(thresholds)]))
    calibrated_threshold = float(thresholds[best_idx])
    oof_f1 = float(f1_scores[best_idx])
    oof_pr_auc = float(average_precision_score(y_train, oof_probas))
    print(
        f"[Train] Calibrated OOF Threshold: {calibrated_threshold:.4f} (OOF F1: {oof_f1:.4f}, OOF PR-AUC: {oof_pr_auc:.4f})"
    )

    # Train final authoritative model on full training set
    print(f"[Train] Fitting final production {model_type} model with SMOTE ratio {smote_ratio}...")
    t0 = time.time()
    smote_final = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_res, y_train_res = smote_final.fit_resample(X_train, y_train)

    if model_type.lower() == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    clf.fit(X_train_res, y_train_res)
    train_duration = round(time.time() - t0, 3)

    # Evaluate on strictly held-out test set
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= calibrated_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    # Latency benchmarking
    eval_sample = X_test.iloc[:10000]
    t0 = time.time()
    _ = clf.predict_proba(eval_sample)
    inf_10k_ms = round((time.time() - t0) * 1000, 2)
    inf_per_sample_us = round((inf_10k_ms / 10000.0) * 1000.0, 2)

    metrics = {
        "precision": round(
            float(precision_score(y_test, y_test_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, y_test_pred)), 4),
        "f1": round(float(f1_score(y_test, y_test_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_test_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_test_proba)), 4),
        "confusion_matrix": cm,
        "calibrated_threshold": round(calibrated_threshold, 4),
        "train_time_seconds": train_duration,
        "inference_latency_per_sample_us": inf_per_sample_us,
        "inference_10k_batch_ms": inf_10k_ms,
        "test_size": len(y_test),
        "test_fraud_count": int(y_test.sum()),
    }
    print(f"[Train] Strictly held-out test metrics: {json.dumps(metrics, indent=2)}")

    # Build SHAP TreeExplainer on the exact trained model
    print("[Train] Initializing SHAP TreeExplainer on trained model...")
    explainer = shap.TreeExplainer(clf)
    background_sample = X_train.sample(
        n=min(200, len(X_train)), random_state=random_state
    )

    # Load benchmark summary if present
    benchmark_summary = {}
    if benchmark_path and os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                benchmark_summary = json.load(f)
        except Exception:
            pass

    model_version = f"v2.0-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    artifact = {
        "model": clf,
        "explainer": explainer,
        "feature_names": feature_names,
        "threshold": calibrated_threshold,
        "metrics": metrics,
        "model_version": model_version,
        "model_type": model_type,
        "strategy": f"SMOTE ({smote_ratio}) + {model_type.upper()} + 5-Fold Calibrated Threshold + Time-based Split",
        "background_sample": background_sample,
        "benchmark_summary": benchmark_summary,
        "provenance": {
            "git_sha": _git_sha(),
            "data_path": data_path,
            "data_sha256": dataset_hash,
            "package_versions": _package_versions(),
            "trained_at": model_version,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Versioned immutable copy
    versioned_dir = os.path.join(os.path.dirname(output_path), "versions")
    os.makedirs(versioned_dir, exist_ok=True)
    versioned_path = os.path.join(versioned_dir, f"model_{model_version}.pkl")
    with open(versioned_path, "wb") as f:
        pickle.dump(artifact, f)
    artifact_hash = _file_sha256(versioned_path)

    # Primary pointer file with backup
    if os.path.exists(output_path):
        backup_path = output_path + ".bak"
        os.replace(output_path, backup_path)
        print(f"[Train] Backed up previous artifact to {backup_path}")

    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    with open(output_path + ".sha256", "w", encoding="utf-8") as f:
        f.write(artifact_hash)

    print(f"[Train] Production model artifact successfully saved to {output_path}")
    print(f"[Train] Versioned copy saved to {versioned_path}")
    print(f"[Train] SHA256 checksum: {artifact_hash}")
    return artifact


if __name__ == "__main__":
    train_fraud_model()
