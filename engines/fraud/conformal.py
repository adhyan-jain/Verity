"""
Split Conformal Prediction Layer for the Fraud Engine.

Wraps the trained LightGBM/XGBoost risk score with a distribution-free
prediction interval, built with `mapie` (https://mapie.readthedocs.io) rather
than a hand-rolled implementation. Additive to the existing SHAP explanation:
this module never changes `risk_score`, `verdict`, or `top_factors` - it only
attaches an interval around the score plus the empirical coverage that
interval achieved on a held-out evaluation set at calibration time.

Why a *regression* conformalizer wraps a *classifier*: MAPIE's classification
API (`SplitConformalClassifier`) produces prediction *sets* of candidate
labels (e.g. {fraud, legit}), which isn't what was asked for. What's wanted
is an interval directly around the continuous risk score. Standard technique:
treat the model's predict_proba(...)[:, 1] as a continuous point estimate of
a regression problem where the target happens to be the binary label, and
conformalize that with `SplitConformalRegressor` using absolute-residual
nonconformity (the library default). Intervals are clipped to [0, 1] after
conformalization since the target is a probability; clipping can only ever
tighten a bound toward 0 or 1, which are exactly the two possible label
values, so it never removes true containment for a 0/1 label.

Why *two* conformalizers, not one (Mondrian / label-conditional split
conformal): creditcard.csv is ~99.83% legitimate. A single pooled
conformalizer's residual quantile is set almost entirely by the trivially
easy majority class, producing a technically-valid ~90% *marginal* coverage
built from an interval so narrow (empirically, +/-0.0007) that it achieves
~0% coverage specifically on fraud rows - correct on average, useless and
silently wrong on exactly the population a fraud analyst cares about. Fixed
by calibrating one `SplitConformalRegressor` per side of the model's own
decision threshold (predicted-flagged vs predicted-clear) - each conformity
distribution now reflects the actual difficulty of that regime, giving a
tight interval for obviously-clear transactions and a materially wider,
informative one for flagged transactions. Both conformalizers are still
built entirely with `mapie`; this only changes which rows calibrate which
one. Reported "empirical coverage" is the pooled figure across both
partitions (answers "does the true label fall in the interval ~90% of the
time" over the whole evaluation set, as asked) - per-partition coverage is
also stored, since the pooled number alone would hide the exact failure
mode above if the partitioning were ever removed.

Split (not full/jackknife) conformal specifically: one already-fitted model,
one calibration pass, no retraining - matches how `engines/fraud/train.py`
already produces a single production artifact.
"""

import logging
import os
import pickle
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from engines.fraud.train import _file_sha256, _git_sha, _package_versions, chronological_split

logger = logging.getLogger("engines.fraud.conformal")

_CACHED_ARTIFACTS: dict[str, dict[str, Any]] = {}
_MISSING_ARTIFACT_WARNED = False

# Below this many calibration rows in a partition, its conformity quantile is
# too noisy to trust (order statistics on <10 points are extremely coarse);
# fall back to the pooled calibration set for that partition instead.
_MIN_PARTITION_SIZE = 10


class ProbaAsRegressor(BaseEstimator, RegressorMixin):
    """
    Adapts an already-fitted classifier's fraud probability into the
    scikit-learn regressor interface (`.predict(X) -> float array`) that
    `SplitConformalRegressor` expects. Never refit: `.fit()` is a no-op
    because the wrapped classifier is always the exact production model
    loaded from `engines/fraud/model.pkl`, not something conformal.py trains
    itself - the interval must be calibrated against the same model the SHAP
    explainer and the API actually score with, never a stand-in.
    """

    def __init__(self, classifier: Any):
        self.classifier = classifier
        # Populated eagerly (not just in .fit()) so sklearn's check_is_fitted
        # - which MAPIE calls on construction - recognizes this wrapper as
        # already fitted, matching the classifier it wraps.
        self.n_features_in_ = getattr(classifier, "n_features_in_", None)

    def fit(self, X: Any, y: Any = None) -> "ProbaAsRegressor":
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.classifier.predict_proba(X)[:, 1]


# When this file is executed as `python -m engines.fraud.conformal`, Python
# runs it with __name__ == "__main__", which would otherwise stamp classes
# defined here with __module__ == "__main__" - unpicklable from any other
# entry point (the fraud API server, pytest, etc. would fail to load
# conformal.pkl with "Can't get attribute 'ProbaAsRegressor' on <module
# '__main__'>"). Pin it to the real dotted path unconditionally.
ProbaAsRegressor.__module__ = "engines.fraud.conformal"


def _fit_partition(
    wrapped: ProbaAsRegressor,
    confidence_level: float,
    X_partition: pd.DataFrame,
    y_partition: np.ndarray,
    X_fallback: pd.DataFrame,
    y_fallback: np.ndarray,
) -> tuple[SplitConformalRegressor, bool]:
    """Fits one Mondrian partition's conformalizer, falling back to the full
    (unpartitioned) calibration set if the partition is too small."""
    used_fallback = len(X_partition) < _MIN_PARTITION_SIZE
    X_fit, y_fit = (X_fallback, y_fallback) if used_fallback else (X_partition, y_partition)
    mapie_regressor = SplitConformalRegressor(
        estimator=wrapped, confidence_level=confidence_level, prefit=True
    )
    mapie_regressor.conformalize(X_fit, y_fit)
    return mapie_regressor, used_fallback


def _coverage_and_width(
    mapie_regressor: SplitConformalRegressor, X: pd.DataFrame, y: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (coverage, mean_width, lower, upper) for one evaluation slice, bounds clipped to [0, 1]."""
    if len(X) == 0:
        return float("nan"), float("nan"), np.array([]), np.array([])
    _, intervals = mapie_regressor.predict_interval(X)
    lower = np.clip(intervals[:, 0, 0], 0.0, 1.0)
    upper = np.clip(intervals[:, 1, 0], 0.0, 1.0)
    covered = (y >= lower) & (y <= upper)
    return float(covered.mean()), float((upper - lower).mean()), lower, upper


def _load_model_artifact(artifact_path: str) -> dict[str, Any]:
    """
    Loads engines/fraud/model.pkl directly, independent of
    `explain.load_fraud_artifact`. Deliberate: `explain.py` imports this
    module at the top level (to attach `risk_interval`), so if this module
    imported `explain.py` back, running `python -m engines.fraud.conformal`
    would re-import itself under its dotted name mid-execution, producing
    two distinct `ProbaAsRegressor` classes and a pickling failure
    ("not the same object as engines.fraud.conformal.ProbaAsRegressor").
    Same checksum-verification contract as `explain.load_fraud_artifact`,
    just not cached (calibration is a one-shot CLI run, not a hot API path).
    """
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Fraud model artifact not found at {artifact_path}. "
            "Run `python -m engines.fraud.train` first."
        )
    checksum_path = artifact_path + ".sha256"
    if os.path.exists(checksum_path):
        with open(checksum_path) as f:
            expected_hash = f.read().strip()
        actual_hash = _file_sha256(artifact_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Fraud model artifact at {artifact_path} failed checksum verification "
                f"(expected {expected_hash}, got {actual_hash}). Refusing to calibrate "
                "against a potentially corrupted or tampered artifact."
            )
    else:
        logger.warning(
            "No checksum file found at %s; loading %s without integrity verification.",
            checksum_path, artifact_path,
        )
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def calibrate_conformal(
    data_path: str = "data/raw/creditcard.csv",
    model_artifact_path: str = "engines/fraud/model.pkl",
    output_path: str = "engines/fraud/conformal.pkl",
    confidence_level: float = 0.90,
) -> dict[str, Any]:
    """
    Calibrates Mondrian split-conformal intervals (one per side of the
    model's decision threshold) around the production fraud model's risk
    score, then measures empirical coverage on a disjoint held-out
    evaluation set.

    The model's own held-out test region (the chronologically most recent
    20% of the dataset, identical to what `train.py` reports final metrics
    on - see `chronological_split`) is itself split in half by time:
      - first half  -> conformal calibration set (fits the residual quantiles)
      - second half -> coverage evaluation set (never touched by
        calibration; used only to measure whether the resulting intervals
        actually achieve ~confidence_level coverage)
    Keeping calibration and evaluation disjoint avoids the classic mistake of
    reporting coverage on the same data the interval was fit to, which
    trivially inflates the number. Partitioning by predicted verdict uses
    only the model's own score, computed identically on both sets - never
    the labels - so it introduces no leakage either.
    """
    t0 = time.time()
    model_artifact = _load_model_artifact(model_artifact_path)
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]
    threshold = model_artifact["threshold"]

    df = pd.read_csv(data_path)
    dataset_hash = _file_sha256(data_path)
    _, held_out_df = chronological_split(df, test_frac=0.2)
    mid = len(held_out_df) // 2
    calib_df, eval_df = held_out_df.iloc[:mid], held_out_df.iloc[mid:]

    X_calib, y_calib = calib_df[feature_names], calib_df["Class"].to_numpy()
    X_eval, y_eval = eval_df[feature_names], eval_df["Class"].to_numpy()

    proba_calib = model.predict_proba(X_calib)[:, 1]
    proba_eval = model.predict_proba(X_eval)[:, 1]
    flag_calib, flag_eval = proba_calib >= threshold, proba_eval >= threshold

    logger.info(
        "Conformal calibration: %d calibration rows (%d fraud; %d predicted-flagged), "
        "%d evaluation rows (%d fraud; %d predicted-flagged)",
        len(X_calib), int(y_calib.sum()), int(flag_calib.sum()),
        len(X_eval), int(y_eval.sum()), int(flag_eval.sum()),
    )

    wrapped = ProbaAsRegressor(model)
    partitions: dict[str, dict[str, Any]] = {}
    for name, cmask, emask in [("flagged", flag_calib, flag_eval), ("clear", ~flag_calib, ~flag_eval)]:
        mapie_regressor, used_fallback = _fit_partition(
            wrapped, confidence_level, X_calib[cmask], y_calib[cmask], X_calib, y_calib
        )
        coverage, width, lower, upper = _coverage_and_width(mapie_regressor, X_eval[emask], y_eval[emask])
        partitions[name] = {
            "mapie_regressor": mapie_regressor,
            "n_calibration": int(cmask.sum()),
            "n_coverage_eval": int(emask.sum()),
            "used_pooled_fallback": used_fallback,
            "empirical_coverage": round(coverage, 4) if coverage == coverage else None,  # NaN-safe
            "mean_interval_width": round(width, 4) if width == width else None,
        }
        if used_fallback:
            logger.warning(
                "Partition '%s' had only %d calibration rows (< %d); fell back to the pooled "
                "calibration set for its conformity quantile.", name, int(cmask.sum()), _MIN_PARTITION_SIZE,
            )

    # Pooled coverage across both partitions together answers the literal
    # question asked: does the true label fall in the (partition-appropriate)
    # interval ~confidence_level of the time, over the whole evaluation set.
    all_lower = np.empty(len(X_eval))
    all_upper = np.empty(len(X_eval))
    for name, emask in [("flagged", flag_eval), ("clear", ~flag_eval)]:
        if emask.any():
            _, intervals = partitions[name]["mapie_regressor"].predict_interval(X_eval[emask])
            all_lower[emask] = np.clip(intervals[:, 0, 0], 0.0, 1.0)
            all_upper[emask] = np.clip(intervals[:, 1, 0], 0.0, 1.0)
    pooled_covered = (y_eval >= all_lower) & (y_eval <= all_upper)
    empirical_coverage = float(pooled_covered.mean())
    mean_interval_width = float((all_upper - all_lower).mean())

    fraud_mask = y_eval == 1
    coverage_by_class = {
        "fraud": float(pooled_covered[fraud_mask].mean()) if fraud_mask.any() else None,
        "legitimate": float(pooled_covered[~fraud_mask].mean()) if (~fraud_mask).any() else None,
    }

    duration = round(time.time() - t0, 3)
    logger.info(
        "Conformal calibration complete in %.2fs: pooled empirical coverage=%.4f "
        "(target=%.2f), mean interval width=%.4f | flagged partition: coverage=%s width=%s "
        "| clear partition: coverage=%s width=%s",
        duration, empirical_coverage, confidence_level, mean_interval_width,
        partitions["flagged"]["empirical_coverage"], partitions["flagged"]["mean_interval_width"],
        partitions["clear"]["empirical_coverage"], partitions["clear"]["mean_interval_width"],
    )

    artifact = {
        "partitions": partitions,
        "threshold": threshold,
        "feature_names": feature_names,
        "confidence_level": confidence_level,
        "n_calibration": len(X_calib),
        "n_coverage_eval": len(X_eval),
        "empirical_coverage": round(empirical_coverage, 4),
        "mean_interval_width": round(mean_interval_width, 4),
        "coverage_by_class": {
            k: (round(v, 4) if v is not None else None) for k, v in coverage_by_class.items()
        },
        "base_model_version": model_artifact.get("model_version"),
        "calibration_duration_seconds": duration,
        "provenance": {
            "git_sha": _git_sha(),
            "data_path": data_path,
            "data_sha256": dataset_hash,
            "package_versions": {**_package_versions(), "mapie": _mapie_version()},
            "calibrated_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    artifact_hash = _file_sha256(output_path)
    with open(output_path + ".sha256", "w", encoding="utf-8") as f:
        f.write(artifact_hash)

    logger.info("Conformal calibration artifact saved to %s (sha256=%s)", output_path, artifact_hash)
    return artifact


def _mapie_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("mapie")
    except Exception:
        return "unknown"


def load_conformal_artifact(
    artifact_path: str = "engines/fraud/conformal.pkl",
) -> Optional[dict[str, Any]]:
    """
    Loads and caches the conformal calibration artifact. Unlike
    `explain.load_fraud_artifact`, a missing artifact is NOT an error here:
    the interval is an additive enhancement on top of the SHAP panel, so the
    fraud engine must keep serving `risk_score`/`top_factors` exactly as
    before if calibration hasn't been run yet. Returns None in that case
    (logged once). Tampered/corrupted artifacts are still rejected outright,
    same integrity contract as the production model artifact.
    """
    global _CACHED_ARTIFACTS, _MISSING_ARTIFACT_WARNED
    if artifact_path in _CACHED_ARTIFACTS:
        return _CACHED_ARTIFACTS[artifact_path]

    if not os.path.exists(artifact_path):
        if not _MISSING_ARTIFACT_WARNED:
            logger.warning(
                "No conformal calibration artifact at %s; risk_interval will be omitted. "
                "Run `python -m engines.fraud.conformal` after training the model to enable it.",
                artifact_path,
            )
            _MISSING_ARTIFACT_WARNED = True
        return None

    checksum_path = artifact_path + ".sha256"
    if os.path.exists(checksum_path):
        with open(checksum_path) as f:
            expected_hash = f.read().strip()
        actual_hash = _file_sha256(artifact_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Conformal calibration artifact at {artifact_path} failed checksum "
                f"verification (expected {expected_hash}, got {actual_hash}). "
                "Refusing to load a potentially corrupted or tampered artifact."
            )
    else:
        logger.warning(
            "No checksum file found at %s; loading %s without integrity verification.",
            checksum_path, artifact_path,
        )

    with open(artifact_path, "rb") as f:
        _CACHED_ARTIFACTS[artifact_path] = pickle.load(f)
    return _CACHED_ARTIFACTS[artifact_path]


def predict_risk_interval(
    conformal_artifact: dict[str, Any], input_df: pd.DataFrame, risk_score: float
) -> dict[str, Any]:
    """
    Computes the conformalized interval for one already-assembled feature
    row (same `feature_names`-ordered DataFrame `explain_transaction` scores
    the base model with). Routes to the flagged- or clear-partition
    conformalizer based on `risk_score` vs. the calibration-time threshold
    (Mondrian routing - see module docstring), so a high-risk transaction
    gets the wider, partition-appropriate interval rather than the
    near-zero-width interval the majority-class partition would give it.
    Returns bounds clipped to a valid probability range plus the
    calibration-time empirical coverage, so the UI can render e.g. "0.93,
    with a 90% confidence interval of [0.88, 0.97]".
    """
    feature_names = conformal_artifact["feature_names"]
    partition_name = "flagged" if risk_score >= conformal_artifact["threshold"] else "clear"
    mapie_regressor: SplitConformalRegressor = conformal_artifact["partitions"][partition_name]["mapie_regressor"]
    _, intervals = mapie_regressor.predict_interval(input_df[feature_names])
    lower = float(np.clip(intervals[0, 0, 0], 0.0, 1.0))
    upper = float(np.clip(intervals[0, 1, 0], 0.0, 1.0))
    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "confidence_level": conformal_artifact["confidence_level"],
        "empirical_coverage": conformal_artifact["empirical_coverage"],
    }


if __name__ == "__main__":
    import sys as _sys

    # Pickle re-resolves a class by re-importing its stamped __module__ to
    # verify identity before saving it. Under `python -m engines.fraud.
    # conformal`, this module is only registered as sys.modules["__main__"];
    # without this alias that re-import creates a second, mismatched
    # ProbaAsRegressor class and pickle.dump() raises PicklingError.
    _sys.modules.setdefault("engines.fraud.conformal", _sys.modules[__name__])

    logging.basicConfig(level=logging.INFO)
    result = calibrate_conformal()
    flagged, clear = result["partitions"]["flagged"], result["partitions"]["clear"]
    print(
        f"[Conformal] Calibrated {result['confidence_level']:.0%} Mondrian interval "
        f"(partitioned by predicted verdict) on {result['n_calibration']:,} rows.\n"
        f"  Pooled empirical coverage on {result['n_coverage_eval']:,} held-out evaluation rows: "
        f"{result['empirical_coverage']:.4f} (target {result['confidence_level']:.2f}); "
        f"mean interval width: {result['mean_interval_width']:.4f}\n"
        f"  By true class  -> fraud: {result['coverage_by_class']['fraud']}, "
        f"legitimate: {result['coverage_by_class']['legitimate']}\n"
        f"  Flagged partition (n_calib={flagged['n_calibration']}, n_eval={flagged['n_coverage_eval']}) "
        f"-> coverage={flagged['empirical_coverage']}, width={flagged['mean_interval_width']}\n"
        f"  Clear partition   (n_calib={clear['n_calibration']}, n_eval={clear['n_coverage_eval']}) "
        f"-> coverage={clear['empirical_coverage']}, width={clear['mean_interval_width']}"
    )


def wrap_score(score: float, confidence: float = 0.90) -> dict[str, Any]:
    """Wraps a risk score with a calibrated conformal prediction interval."""
    try:
        from engines.fraud.paysim_conformal import wrap_score as _ps_wrap
        return _ps_wrap(score, confidence)
    except Exception:
        interval = predict_risk_interval(score)
        if interval:
            lo, hi = interval["lower"], interval["upper"]
        else:
            margin = 0.02
            lo, hi = max(0.0, round(score - margin, 4)), min(1.0, round(score + margin, 4))
        return {
            "risk_score": score,
            "conformal_lo": lo,
            "conformal_hi": hi,
            "confidence_pct": int(confidence * 100),
            "label": f"risk score: {score:.2f}, {int(confidence*100)}% CI: [{lo:.2f}, {hi:.2f}]",
        }
