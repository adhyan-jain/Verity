"""
Model Training Module for Fraud Detection.
Person A: Trains production-grade LightGBM model with SMOTE and 5-fold CV threshold calibration.
"""

import os
import pickle
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


def train_fraud_model(
    data_path: str = "data/raw/creditcard.csv",
    output_path: str = "engines/fraud/model.pkl",
    random_state: int = 42,
    n_estimators: int = 350,
    smote_ratio: float = 0.05,
) -> dict[str, Any]:
    """
    Trains fraud classification model on creditcard.csv using SMOTE oversampling
    and LightGBM, calibrated using 5-fold out-of-fold cross-validation.
    """
    print(f"Training production fraud detection model from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    feature_names = X.columns.tolist()

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(
        f"Dataset split: {len(X_train)} train, {len(X_test)} test (Test fraud: {y_test.sum()})"
    )

    # 5-fold Stratified Cross Validation on Training Data to compute out-of-fold threshold
    print("Running 5-Fold Stratified Cross-Validation for threshold calibration...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probas = np.zeros(len(X_train))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_va = X_train.iloc[val_idx]

        smote_fold = SMOTE(
            sampling_strategy=smote_ratio, random_state=random_state + fold
        )
        X_tr_res, y_tr_res = smote_fold.fit_resample(X_tr, y_tr)

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

    # Calculate optimal threshold on OOF predictions.
    # precision_recall_curve returns thresholds with length len(precisions) - 1,
    # so the search is restricted to indices that have a corresponding threshold.
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probas)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores[: len(thresholds)]))
    calibrated_threshold = float(thresholds[best_idx])
    oof_f1 = float(f1_scores[best_idx])
    oof_pr_auc = float(average_precision_score(y_train, oof_probas))
    print(
        f"Calibrated OOF Threshold: {calibrated_threshold:.4f} (OOF F1: {oof_f1:.4f}, OOF PR-AUC: {oof_pr_auc:.4f})"
    )

    # Train final production model on full training set
    print(f"Fitting final production LightGBM model with SMOTE ratio {smote_ratio}...")
    t0 = time.time()
    smote_final = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    X_train_res, y_train_res = smote_final.fit_resample(X_train, y_train)

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

    metrics = {
        "precision": round(
            float(precision_score(y_test, y_test_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_test, y_test_pred)), 4),
        "f1": round(float(f1_score(y_test, y_test_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_test_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_test_proba)), 4),
        "calibrated_threshold": round(calibrated_threshold, 4),
        "train_time_seconds": train_duration,
        "test_size": len(y_test),
        "test_fraud_count": int(y_test.sum()),
    }
    print(f"Evaluation metrics on strictly held-out test set: {metrics}")

    # Build SHAP TreeExplainer
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    background_sample = X_train.sample(
        n=min(200, len(X_train)), random_state=random_state
    )

    artifact = {
        "model": clf,
        "explainer": explainer,
        "feature_names": feature_names,
        "threshold": calibrated_threshold,
        "metrics": metrics,
        "model_version": "v1.1-prod-calibrated",
        "strategy": "SMOTE (0.05) + LightGBM + 5-Fold Calibrated Threshold",
        "background_sample": background_sample,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Production model artifact successfully saved to {output_path}")
    return artifact


if __name__ == "__main__":
    train_fraud_model()
