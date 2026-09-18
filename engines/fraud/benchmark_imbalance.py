"""
Benchmark Class Imbalance Techniques on Credit Card Fraud Dataset.
Person A: Run SMOTE vs Class Weighting comparison on recall, precision, F1, PR-AUC, and ROC-AUC.
"""

import time
from typing import Any

import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def run_benchmark(
    data_path: str = "data/raw/creditcard.csv", random_state: int = 42
) -> dict[str, Any]:
    """
    Evaluates SMOTE vs Class-Weighting on creditcard fraud dataset.
    Returns comparison metrics to select the winner model.
    """
    print(f"Benchmarking imbalance strategies on {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 1. Class Weighting Strategy
    t0 = time.time()
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    clf_cw = lgb.LGBMClassifier(
        scale_pos_weight=pos_weight,
        random_state=random_state,
        n_estimators=100,
        verbose=-1,
        n_jobs=-1,
    )
    clf_cw.fit(X_train, y_train)
    cw_train_time = round(time.time() - t0, 3)

    y_pred_cw = clf_cw.predict(X_test)
    y_proba_cw = clf_cw.predict_proba(X_test)[:, 1]

    cw_metrics = {
        "recall": round(float(recall_score(y_test, y_pred_cw)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred_cw, zero_division=0)), 4
        ),
        "f1": round(float(f1_score(y_test, y_pred_cw)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba_cw)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_cw)), 4),
        "training_time_seconds": cw_train_time,
    }

    # 2. SMOTE Strategy
    t0 = time.time()
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    clf_smote = lgb.LGBMClassifier(
        random_state=random_state, n_estimators=100, verbose=-1, n_jobs=-1
    )
    clf_smote.fit(X_train_res, y_train_res)
    smote_train_time = round(time.time() - t0, 3)

    y_pred_smote = clf_smote.predict(X_test)
    y_proba_smote = clf_smote.predict_proba(X_test)[:, 1]

    smote_metrics = {
        "recall": round(float(recall_score(y_test, y_pred_smote)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred_smote, zero_division=0)), 4
        ),
        "f1": round(float(f1_score(y_test, y_pred_smote)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba_smote)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba_smote)), 4),
        "training_time_seconds": smote_train_time,
    }

    winner = (
        "smote"
        if smote_metrics["f1"] > cw_metrics["f1"] and smote_metrics["recall"] >= 0.75
        else "class_weighting"
    )

    results = {
        "dataset": data_path,
        "total_records": len(df),
        "fraud_cases": int(y.sum()),
        "imbalance_ratio": round(float(y.mean()), 6),
        "class_weighting": cw_metrics,
        "smote": smote_metrics,
        "winner": winner,
        "recommendation": (
            f"Selected {winner.upper()} strategy due to superior balance of fraud recall "
            f"and precision (F1: {smote_metrics['f1'] if winner == 'smote' else cw_metrics['f1']} vs "
            f"{cw_metrics['f1'] if winner == 'smote' else smote_metrics['f1']})."
        ),
    }

    return results


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    print("Benchmark complete:")
    import json

    print(json.dumps(benchmark_results, indent=2))
