"""
Authoritative Model-Backed Scoring & Counterfactual Engine.
Unifies agent scoring and counterfactual inference directly on the authoritative trained ML model
(engines/fraud/model.pkl), removing all duplicate hardcoded logistic regression formulas.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from engines.fraud.explain import explain_transaction, load_fraud_artifact

logger = logging.getLogger("verity.agent.model_engine")

# Baseline high-risk transaction feature vector (derived from creditcard.csv row 197586)
DEFAULT_TX_FEATURES: Dict[str, float] = {
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

    def __init__(self, artifact_path: Optional[str] = None):
        self.artifact_path = artifact_path or "engines/fraud/model.pkl"
        self._artifact: Optional[Dict[str, Any]] = None

    @property
    def artifact(self) -> Dict[str, Any]:
        if self._artifact is None:
            self._artifact = load_fraud_artifact(self.artifact_path)
        return self._artifact

    @property
    def features(self) -> list[str]:
        return self.artifact["feature_names"]

    def score_features(
        self, feature_dict: Dict[str, float]
    ) -> Tuple[float, str, Dict[str, float]]:
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
        base_features: Dict[str, float],
        parameter_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluates a counterfactual query by re-running the authoritative trained model
        on the modified feature vector and computing attribution deltas.
        """
        # 1. Base inference
        orig_score, orig_verdict, orig_contrib = self.score_features(base_features)

        # 2. Modify feature vector
        modified_features = dict(base_features)
        for k, v in parameter_overrides.items():
            norm_k = (
                "Amount"
                if k.lower() == "amount"
                else ("Time" if k.lower() == "time" else k)
            )
            if norm_k in modified_features or norm_k in self.features:
                try:
                    modified_features[norm_k] = float(v)
                except (ValueError, TypeError):
                    pass

        # 3. Model inference on modified features
        recalc_score, recalc_verdict, recalc_contrib = self.score_features(
            modified_features
        )

        # Calculate exact contribution delta for modified features
        deltas = {}
        for k in parameter_overrides:
            norm_k = (
                "Amount"
                if k.lower() == "amount"
                else ("Time" if k.lower() == "time" else k)
            )
            if norm_k in orig_contrib and norm_k in recalc_contrib:
                deltas[norm_k] = round(recalc_contrib[norm_k] - orig_contrib[norm_k], 4)
            else:
                deltas[norm_k] = round(recalc_score - orig_score, 4)

        model_ver = self.artifact.get("model_version", "authoritative-ml")
        return {
            "original_risk_score": orig_score,
            "recalculated_risk_score": recalc_score,
            "original_verdict": orig_verdict,
            "recalculated_verdict": recalc_verdict,
            "parameter_overrides": parameter_overrides,
            "feature_attribution_deltas": deltas,
            "explanation": (
                f"Authoritative model counterfactual inference: Applying overrides {parameter_overrides} shifted the model "
                f"risk probability from {orig_score:.4f} ({orig_verdict}) to {recalc_score:.4f} ({recalc_verdict}) "
                f"using model version {model_ver} with feature attribution deltas {deltas}."
            ),
        }


# Global singleton instance
_ENGINE_INSTANCE: Optional[ModelEngine] = None


def get_model_engine() -> ModelEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ModelEngine()
    return _ENGINE_INSTANCE
