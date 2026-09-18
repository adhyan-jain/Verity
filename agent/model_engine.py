"""
Authoritative Model-Backed Scoring & Counterfactual Engine.
Unifies agent scoring and counterfactual inference directly on the authoritative trained ML model
(engines/fraud/model.pkl), removing all duplicate hardcoded logistic regression formulas.
"""

from typing import Any

from engines.fraud.explain import explain_transaction, load_fraud_artifact

# Baseline high-risk transaction feature vector (derived from creditcard.csv row 197586)
DEFAULT_TX_FEATURES: dict[str, float] = {
    "Time": 132086.0,
    "V1": -0.361428,
    "V2": 1.133472,
    "V3": -2.700000,
    "V4": -0.283073,
    "V5": 0.371452,
    "V6": -0.574680,
    "V7": 4.031513,
    "V8": -0.934398,
    "V9": -0.768255,
    "V10": -2.248115,
    "V11": -0.482409,
    "V12": -0.690550,
    "V13": 0.181275,
    "V14": -2.372552,
    "V15": -0.006868,
    "V16": 0.146399,
    "V17": 1.759314,
    "V18": 1.083040,
    "V19": -0.391048,
    "V20": -0.025862,
    "V21": 0.110815,
    "V22": 0.563861,
    "V23": -0.408436,
    "V24": -0.880079,
    "V25": 1.408392,
    "V26": -0.137402,
    "V27": -0.001250,
    "V28": -0.182751,
    "Amount": 1000.0,
}


class ModelEngine:
    """
    Thin adapter that routes scoring and counterfactual queries directly to the
    authoritative fraud ML model artifact and its SHAP TreeExplainer.
    """

    def __init__(self, artifact_path: str | None = None):
        self.artifact_path = artifact_path or "engines/fraud/model.pkl"
        self._artifact: dict[str, Any] | None = None

    @property
    def artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            self._artifact = load_fraud_artifact(self.artifact_path)
        return self._artifact

    @property
    def features(self) -> list[str]:
        return self.artifact["feature_names"]

    def _resolve_feature_name(self, key: str) -> str | None:
        """Case-insensitive lookup of an override key against the model's real feature names."""
        key_lower = key.lower()
        for feat in self.features:
            if feat.lower() == key_lower:
                return feat
        return None

    def score_features(
        self, feature_dict: dict[str, float]
    ) -> tuple[float, str, dict[str, float]]:
        """
        Computes model inference and SHAP attributions using the authoritative trained model.
        Returns:
            (risk_score, verdict, contributions_dict)
        """
        explanation = explain_transaction(
            features=feature_dict,
            transaction_id="engine-eval",
            artifact=self.artifact,
            top_n=len(self.features),
        )

        risk_score = float(explanation["risk_score"])
        verdict = str(explanation["verdict"])
        contributions = {
            f["feature"]: float(f["contribution"])
            for f in explanation.get("top_factors", [])
        }
        return risk_score, verdict, contributions

    def evaluate_counterfactual(
        self,
        base_features: dict[str, float],
        parameter_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Evaluates a counterfactual query by re-running the authoritative trained model
        on the modified feature vector and computing attribution deltas.
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
            norm_k = self._resolve_feature_name(k)
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
            norm_k = self._resolve_feature_name(k)
            if norm_k is None:
                continue
            if norm_k in orig_contrib and norm_k in recalc_contrib:
                deltas[norm_k] = round(recalc_contrib[norm_k] - orig_contrib[norm_k], 4)
            else:
                deltas[norm_k] = round(recalc_score - orig_score, 4)

        model_ver = self.artifact.get("model_version", "authoritative-ml")
        explanation = (
            f"Authoritative model counterfactual inference: Applying overrides {parameter_overrides} shifted the model "
            f"risk probability from {orig_score:.4f} ({orig_verdict}) to {recalc_score:.4f} ({recalc_verdict}) "
            f"using model version {model_ver} with feature attribution deltas {deltas}."
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
            "model_source": model_ver,
            "explanation": explanation,
        }


# Global singleton instance
_ENGINE_INSTANCE: ModelEngine | None = None


def get_model_engine() -> ModelEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ModelEngine()
    return _ENGINE_INSTANCE
