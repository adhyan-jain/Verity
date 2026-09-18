"""
Model Training Module for Fraud Detection.
Person A: Trains chosen winner model (SMOTE + LightGBM) and saves artifacts to model.pkl.
"""

import os
import pickle
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap


def train_fraud_model(
    data_path: str = "data/raw/creditcard.csv",
    output_path: str = "engines/fraud/model.pkl",
    random_state: int = 42,
    n_estimators: int = 150,
) -> Dict[str, Any]:
    """
    Trains fraud classification model on creditcard.csv using the benchmark winner
    strategy (SMOTE + LightGBM) and serializes model and explainer artifacts to output_path.
    """
    print(f"Training fraud detection model from {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    feature_names = X.columns.tolist()

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(f"Dataset split: {len(X_train)} train, {len(X_test)} test (Positives: {y_test.sum()})")
    print("Applying SMOTE oversampling to training set...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Fitting LightGBM classifier...")
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.08,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train_res, y_train_res)
    train_duration = round(time.time() - t0, 3)

    # Evaluate on held-out test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "train_time_seconds": train_duration,
        "test_size": len(y_test),
        "test_fraud_count": int(y_test.sum()),
    }
    print(f"Evaluation metrics on test set: {metrics}")

    # Build SHAP TreeExplainer
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(clf)

    # Save representative background sample for reference / fast SHAP baseline
    background_sample = X_train.sample(n=min(200, len(X_train)), random_state=random_state)

    artifact = {
        "model": clf,
        "explainer": explainer,
        "feature_names": feature_names,
        "threshold": 0.5,
        "metrics": metrics,
        "model_version": "v1.0-benchmark-winner",
        "strategy": "SMOTE + LightGBM",
        "background_sample": background_sample,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"Model artifact successfully saved to {output_path}")
    return artifact


if __name__ == "__main__":
    train_fraud_model()

