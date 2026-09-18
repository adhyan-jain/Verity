"""
Step 4 — Drift Detector.

Applies a Kolmogorov-Smirnov test to the classifier's prediction-confidence
distribution across time windows.  Reference window = first N samples;
current window = subsequent non-overlapping windows of the same size.

No fraud labels are required — we test whether the *score distribution itself*
has shifted, which is a valid proxy for concept drift.

Design:
  - Works on any list of (timestamp, score) pairs (from paysim_score or
    any other scorer that returns probabilities).
  - Reference window: first `window_size` records (chronological).
  - Sliding windows: non-overlapping chunks of `window_size` from remainder.
  - KS stat + p-value logged for each window; drift fired if p < alpha.
  - NO auto-retraining (Step 4 spec: detection + logging only).

Output format per window:
    {
        "window_index": int,
        "window_start": ISO str,
        "window_end":   ISO str,
        "n_samples":    int,
        "ks_statistic": float,
        "p_value":      float,
        "drift_fired":  bool,
        "alpha":        float,
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger("verity.fraud.drift")

_HERE = os.path.dirname(os.path.abspath(__file__))
DRIFT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "logs", "drift_log.jsonl"
)


def run_drift_detection(
    scores: list[float],
    timestamps: list[str] | None = None,
    window_size: int = 500,
    alpha: float = 0.05,
    log_path: str = DRIFT_LOG_PATH,
) -> list[dict[str, Any]]:
    """
    Runs KS-drift detection on classifier confidence scores.

    Parameters
    ----------
    scores      : Ordered list of prediction probabilities (float in [0,1]).
    timestamps  : Optional ISO8601 timestamp strings matching `scores`.
                  If None, synthetic indices are used.
    window_size : Size of each window (reference and current).
    alpha       : Significance level for the KS test (default 0.05).
    log_path    : Where to append JSONL drift records.

    Returns
    -------
    List of window result dicts (one per current window).
    """
    if len(scores) < 2 * window_size:
        logger.warning(
            "Drift detection requires at least 2×window_size=%d samples; "
            "got %d. Skipping.", 2 * window_size, len(scores)
        )
        return []

    arr = np.array(scores, dtype=float)
    reference = arr[:window_size]

    # Build default timestamps if none provided
    if timestamps is None:
        timestamps = [
            datetime.fromtimestamp(i, tz=timezone.utc).isoformat()
            for i in range(len(scores))
        ]

    results: list[dict[str, Any]] = []
    window_idx = 0

    for start in range(window_size, len(arr), window_size):
        end = min(start + window_size, len(arr))
        current = arr[start:end]

        if len(current) < 10:
            break

        stat, pval = ks_2samp(reference, current)
        fired = bool(pval < alpha)

        record: dict[str, Any] = {
            "window_index":  window_idx,
            "window_start":  timestamps[start],
            "window_end":    timestamps[end - 1],
            "n_samples":     len(current),
            "ks_statistic":  round(float(stat), 6),
            "p_value":       round(float(pval), 6),
            "drift_fired":   fired,
            "alpha":         alpha,
            "reference_mean": round(float(reference.mean()), 6),
            "current_mean":   round(float(current.mean()), 6),
        }
        results.append(record)

        level = logging.WARNING if fired else logging.INFO
        logger.log(
            level,
            "[DriftDetector] Window %d: KS=%.4f p=%.4f %s",
            window_idx, stat, pval, "DRIFT FIRED" if fired else "stable",
        )
        window_idx += 1

    # Append to JSONL log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_fired = sum(1 for r in results if r["drift_fired"])
    logger.info(
        "[DriftDetector] %d windows evaluated; drift fired in %d.",
        len(results), n_fired,
    )
    return results


def simulate_drift_on_paysim(
    window_size: int = 1000,
    alpha: float = 0.05,
    n_rows: int = 20_000,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper: loads the PaySim model, scores the first `n_rows`
    PaySim records (using engineer_paysim_features), and runs drift detection
    across simulated time windows.

    This demonstrates the detector end-to-end without requiring bank labels.
    """
    import sys
    _ROOT = os.path.dirname(os.path.dirname(_HERE))
    sys.path.insert(0, _ROOT)
    import pickle

    import pandas as pd
    from engines.fraud.paysim_train import engineer_paysim_features, PAYSIM_PATH

    model_path = os.path.join(_HERE, "paysim_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "paysim_model.pkl not found. Run engines/fraud/paysim_train.py first."
        )

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    lgb_model = artifact["lgb_model"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(PAYSIM_PATH, nrows=n_rows)
    df = df.sort_values("step").reset_index(drop=True)

    X = engineer_paysim_features(df)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]

    scores = lgb_model.predict_proba(X)[:, 1].tolist()
    # Synthetic timestamps from step column
    timestamps = [f"step-{int(s):05d}" for s in df["step"].tolist()]

    return run_drift_detection(
        scores=scores,
        timestamps=timestamps,
        window_size=window_size,
        alpha=alpha,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    print("Running drift detector simulation on PaySim scores ...")
    results = simulate_drift_on_paysim(window_size=1000, alpha=0.05, n_rows=20_000)
    print(f"\nDrift detection complete: {len(results)} windows evaluated")
    print(f"{'Win':>4}  {'KS':>8}  {'p-value':>10}  {'Drift?':>8}  "
          f"{'Ref mean':>10}  {'Cur mean':>10}")
    print("-" * 60)
    for r in results:
        fired_str = "FIRED" if r["drift_fired"] else "stable"
        print(
            f"{r['window_index']:>4}  {r['ks_statistic']:>8.4f}  "
            f"{r['p_value']:>10.4f}  {fired_str:>8}  "
            f"{r['reference_mean']:>10.4f}  {r['current_mean']:>10.4f}"
        )
    print(f"\nLog appended to: {DRIFT_LOG_PATH}")

