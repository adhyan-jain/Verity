"""
SHAP Explainability Module for Fraud Detection.
Person A: Splits interpretable features (Time, Amount) vs anonymized signals (V1-V28).
"""

import hashlib
import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from engines.fraud.conformal import load_conformal_artifact, predict_risk_interval

logger = logging.getLogger("engines.fraud.explain")

_CACHED_ARTIFACT: dict[str, Any] | None = None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fraud_artifact(
    artifact_path: str = "engines/fraud/model.pkl",
) -> dict[str, Any]:
    """
    Loads and caches model artifact containing LightGBM classifier and SHAP explainer.
    Verifies the artifact's checksum against its companion `.sha256` file (written
    by train.py) when present, so a corrupted or unexpectedly replaced model.pkl
    is rejected instead of silently loaded.
    """
    global _CACHED_ARTIFACT
    if _CACHED_ARTIFACT is None:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Fraud model artifact not found at {artifact_path}. "
                "Run `engines/fraud/train.py` first."
            )
        checksum_path = artifact_path + ".sha256"
        if os.path.exists(checksum_path):
            with open(checksum_path) as f:
                expected_hash = f.read().strip()
            actual_hash = _file_sha256(artifact_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Fraud model artifact at {artifact_path} failed checksum "
                    f"verification (expected {expected_hash}, got {actual_hash}). "
                    "Refusing to load a potentially corrupted or tampered artifact."
                )
        else:
            logger.warning(
                "No checksum file found at %s; loading %s without integrity "
                "verification.",
                checksum_path,
                artifact_path,
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
        artifact = load_fraud_artifact()

    # Feature order/threshold/version always come from the trained artifact so a
    # caller-supplied model/explainer can never be scored against a guessed
    # feature order that silently mismatches how the model was trained.
    feature_names = artifact["feature_names"]
    threshold = artifact.get("threshold", 0.5)
    model_version = artifact.get("model_version", "v1.1-prod-calibrated")
    if model is None:
        model = artifact["model"]
    if explainer is None:
        explainer = artifact["explainer"]

    # Construct clean feature vector matching feature_names
    row_data = {feat: float(features.get(feat, 0.0)) for feat in feature_names}
    input_df = pd.DataFrame([row_data])[feature_names]

    # Model inference
    proba = float(model.predict_proba(input_df)[0, 1])
    risk_score = round(proba, 4)
    verdict = "flagged" if risk_score >= threshold else "clear"

    # Prediction log: lets delayed ground-truth fraud labels be joined back to
    # what the model actually scored, for live precision/recall monitoring.
    logger.info(
        "prediction transaction_id=%s model_version=%s risk_score=%s "
        "threshold=%s verdict=%s",
        transaction_id,
        model_version,
        risk_score,
        threshold,
        verdict,
    )

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

    # Additive: attaches a split-conformal interval around risk_score
    # (engines/fraud/conformal.py) when a calibration artifact is
    # available. Never blocks or changes the SHAP output above - a missing
    # or failing conformal artifact just omits risk_interval.
    risk_interval = None
    conformal_artifact = load_conformal_artifact()
    if conformal_artifact is not None:
        try:
            risk_interval = predict_risk_interval(conformal_artifact, input_df, risk_score)
        except Exception:
            logger.exception(
                "Conformal interval prediction failed for transaction_id=%s; omitting risk_interval.",
                transaction_id,
            )

    return {
        "transaction_id": transaction_id,
        "risk_score": risk_score,
        "verdict": verdict,
        "top_factors": selected_factors,
        "model_version": model_version,
        "risk_interval": risk_interval,
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
