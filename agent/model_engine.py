"""
Model-Backed Scoring & Counterfactual Engine.
Person C: Evaluates transactions and counterfactuals using real model inference
(calibrated logistic regression trained on creditcard.csv) rather than hardcoded step thresholds.
"""

import json
import math
import os
from typing import Any

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_spec.json")

# Default baseline feature vector for TX-CARD-9842 (high risk off-hours transaction)
DEFAULT_TX_FEATURES: dict[str, float] = {
    "Time": 12120.0,  # ~ 03:22 AM
    "Amount": 4850.0,
    "V1": -3.043541,
    "V2": -3.157307,
    "V3": 1.088463,
    "V4": 2.288644,
    "V5": 1.359805,
    "V6": -1.064823,
    "V7": 0.325574,
    "V8": -0.067794,
    "V9": -0.270953,
    "V10": -0.838587,
    "V11": -0.414575,
    "V12": -0.503141,
    "V13": 0.676502,
    "V14": -0.100000,
    "V15": -0.760756,
    "V16": -1.140460,
    "V17": -0.284928,
    "V18": -0.124804,
    "V19": 0.401764,
    "V20": 2.102339,
    "V21": 0.661696,
    "V22": 0.435477,
    "V23": 1.375966,
    "V24": -0.293803,
    "V25": 0.279798,
    "V26": -0.145362,
    "V27": -0.252773,
    "V28": 0.035764,
}


class ModelEngine:
    def __init__(self, spec_path: str | None = None):
        target = spec_path or SPEC_PATH
        if os.path.exists(target):
            with open(target, "r", encoding="utf-8") as f:
                self.spec = json.load(f)
        else:
            # Fallback coefficients
            self.spec = {
                "features": list(DEFAULT_TX_FEATURES.keys()),
                "means": [0.0] * 30,
                "scales": [1.0] * 30,
                "intercept": 0.377,
                "coefficients": [0.0] * 30,
            }

        self.features = self.spec["features"]
        self._feature_lookup = {feat.lower(): feat for feat in self.features}
        self.means = {k: m for k, m in zip(self.features, self.spec["means"])}
        self.scales = {
            k: (s if s != 0 else 1.0)
            for k, s in zip(self.features, self.spec["scales"])
        }
        self.coefficients = {
            k: c for k, c in zip(self.features, self.spec["coefficients"])
        }
        self.intercept = self.spec.get("intercept", 0.0)

    def score_features(
        self, feature_dict: dict[str, float]
    ) -> tuple[float, str, dict[str, float]]:
        """
        Computes model inference using calibrated logistic regression:
        z = intercept + sum(coef_i * (x_i - mean_i) / scale_i)
        probability = 1 / (1 + exp(-z))

        Raises KeyError if feature_dict is missing any of the model's
        required features — callers always pass a fully-populated feature
        vector (seeded from DEFAULT_TX_FEATURES), so a missing key means a
        real bug upstream, not a case to silently paper over with the
        training-set mean.
        """
        missing = [feat for feat in self.features if feat not in feature_dict]
        if missing:
            raise KeyError(f"score_features: missing required feature(s): {missing}")

        z = self.intercept
        contributions: dict[str, float] = {}

        for feat in self.features:
            val = float(feature_dict[feat])
            mean = self.means.get(feat, 0.0)
            scale = self.scales.get(feat, 1.0)
            coef = self.coefficients.get(feat, 0.0)

            scaled_val = (val - mean) / scale
            contrib = coef * scaled_val
            z += contrib
            contributions[feat] = contrib

        # Numerical stability clamp for sigmoid
        z_clamped = max(-20.0, min(20.0, z))
        prob = 1.0 / (1.0 + math.exp(-z_clamped))
        risk_score = round(prob, 2)
        verdict = "flagged" if risk_score >= 0.50 else "clear"

        return risk_score, verdict, contributions

    def evaluate_counterfactual(
        self, base_features: dict[str, float], parameter_overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluates a counterfactual query by re-running the mathematical model
        on the modified feature vector.

        This is a lightweight fallback model used only when the live fraud
        engine's production LightGBM model is unreachable — its coefficients
        are a separate, simpler fit, so results here are directional
        approximations, not the production risk score. Callers should treat
        `model_source` accordingly.
        """
        # 1. Base inference
        orig_score, orig_verdict, orig_contrib = self.score_features(base_features)

        # 2. Modify feature vector, tracking any override that doesn't
        # resolve to a known feature or isn't a valid number, instead of
        # silently dropping it — a caller/analyst needs to know an override
        # they asked for was ignored rather than assuming it was applied.
        modified_features = dict(base_features)
        rejected_overrides: dict[str, str] = {}
        for k, v in parameter_overrides.items():
            norm_k = self._feature_lookup.get(k.lower())
            if norm_k is None:
                rejected_overrides[k] = f"Unknown feature name '{k}'"
                continue
            try:
                modified_features[norm_k] = float(v)
            except (ValueError, TypeError):
                rejected_overrides[k] = f"Invalid numeric value: {v!r}"

        # 3. Model inference on modified features
        recalc_score, recalc_verdict, recalc_contrib = self.score_features(
            modified_features
        )

        # Calculate exact contribution delta for modified features
        deltas = {}
        for k in parameter_overrides:
            norm_k = self._feature_lookup.get(k.lower())
            if norm_k and norm_k in orig_contrib and norm_k in recalc_contrib:
                deltas[norm_k] = round(recalc_contrib[norm_k] - orig_contrib[norm_k], 3)

        explanation = (
            f"Fallback logistic-regression model (not the production fraud engine): applying overrides "
            f"{parameter_overrides} shifted the risk probability from {orig_score:.2f} to {recalc_score:.2f} "
            f"(verdict changed from {orig_verdict} to {recalc_verdict}) based on feature attribution delta {deltas}. "
            f"Treat this as a directional approximation, not the authoritative production score."
        )
        if rejected_overrides:
            explanation += f" Ignored invalid override(s): {rejected_overrides}."

        return {
            "original_risk_score": orig_score,
            "recalculated_risk_score": recalc_score,
            "original_verdict": orig_verdict,
            "recalculated_verdict": recalc_verdict,
            "parameter_overrides": parameter_overrides,
            "rejected_overrides": rejected_overrides,
            "feature_attribution_deltas": deltas,
            "model_source": "fallback_logistic_regression",
            "explanation": explanation,
        }


# Global singleton instance
_ENGINE_INSTANCE: ModelEngine | None = None


def get_model_engine() -> ModelEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ModelEngine()
    return _ENGINE_INSTANCE
