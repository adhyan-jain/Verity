"""
SHAP Explainability Module for Fraud Detection.
Person A: Splits interpretable features (Time, Amount) vs anonymized signals (V1-V28).
"""

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

_CACHED_ARTIFACT: dict[str, Any] | None = None


def load_fraud_artifact(
    artifact_path: str = "engines/fraud/model.pkl",
) -> dict[str, Any]:
    """
    Loads and caches model artifact containing LightGBM classifier and SHAP explainer.
    """
    global _CACHED_ARTIFACT
    if _CACHED_ARTIFACT is None:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Fraud model artifact not found at {artifact_path}. "
                "Run `engines/fraud/train.py` first."
            )
        with open(artifact_path, "rb") as f:
            _CACHED_ARTIFACT = pickle.load(f)
    return _CACHED_ARTIFACT


def format_time_label(seconds_offset: float) -> str:
    """
    Converts seconds offset into a readable time of day string.
    """
    sec = float(seconds_offset)
    total_seconds = int(sec % 86400)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    ampm = "AM" if hours < 12 else "PM"
    display_hour = hours % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour:02d}:{minutes:02d} {ampm}"


def explain_transaction(
    model: Any = None,
    explainer: Any = None,
    features: dict[str, Any] | None = None,
    transaction_id: str = "unknown",
    artifact: dict[str, Any] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Computes SHAP feature attribution and categorizes features into
    interpretable (Time, Amount) and anonymized (V1-V28).

    Returns a structure adhering to FraudExplanation schema in contracts/schemas.json.
    """
    if features is None:
        features = {}

    if artifact is None:
        if model is None or explainer is None:
            artifact = load_fraud_artifact()
            model = artifact["model"]
            explainer = artifact["explainer"]
            feature_names = artifact["feature_names"]
            threshold = artifact.get("threshold", 0.5)
            model_version = artifact.get("model_version", "v1.1-prod-calibrated")
        else:
            feature_names = [f"V{i}" for i in range(1, 29)]
            feature_names = ["Time"] + feature_names + ["Amount"]
            threshold = 0.5
            model_version = "v1.0"
    else:
        model = artifact["model"]
        explainer = artifact["explainer"]
        feature_names = artifact["feature_names"]
        threshold = artifact.get("threshold", 0.5)
        model_version = artifact.get("model_version", "v1.1-prod-calibrated")

    # Construct clean feature vector matching feature_names
    row_data = {feat: float(features.get(feat, 0.0)) for feat in feature_names}
    input_df = pd.DataFrame([row_data])[feature_names]

    # Model inference
    proba = float(model.predict_proba(input_df)[0, 1])
    risk_score = round(proba, 4)
    verdict = "flagged" if risk_score >= threshold else "clear"

    # Compute SHAP values robustly across versions and output shapes
    shap_output = explainer.shap_values(input_df)
    if isinstance(shap_output, list):
        shap_vals = np.asarray(shap_output[-1]).flatten()
    elif isinstance(shap_output, np.ndarray):
        if shap_output.ndim == 3:
            shap_vals = shap_output[0, :, 1]
        elif shap_output.ndim == 2:
            shap_vals = shap_output[0]
        else:
            shap_vals = shap_output.flatten()
    else:
        shap_vals = np.zeros(len(feature_names))

    # Build top factors
    factors: list[dict[str, Any]] = []
    for feat_name, shap_val in zip(feature_names, shap_vals):
        contrib = round(float(shap_val), 4)
        if feat_name == "Amount":
            amount_val = row_data["Amount"]
            human_label = f"Transaction amount (${amount_val:,.2f})"
            interpretable = True
        elif feat_name == "Time":
            time_str = format_time_label(row_data["Time"])
            human_label = f"Transaction time ({time_str})"
            interpretable = True
        else:
            human_label = f"Anonymized behavioral signal {feat_name}"
            interpretable = False

        factors.append(
            {
                "feature": feat_name,
                "human_label": human_label,
                "contribution": contrib,
                "interpretable": interpretable,
            }
        )

    # Sort factors by absolute magnitude of contribution descending
    factors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    selected_factors = factors[:top_n] if top_n is not None else factors

    return {
        "transaction_id": transaction_id,
        "risk_score": risk_score,
        "verdict": verdict,
        "top_factors": selected_factors,
        "model_version": model_version,
    }


if __name__ == "__main__":
    test_features = {
        "Time": 12120.0,
        "Amount": 4850.00,
        "V14": -5.2,
        "V12": -3.8,
        "V10": -2.1,
    }
    explanation = explain_transaction(
        transaction_id="TX-CARD-TEST", features=test_features
    )
    import json

    print("Sample explanation:")
    print(json.dumps(explanation, indent=2))
