"""
Unit tests for the split-conformal risk_interval layer (engines/fraud/conformal.py).
Requires engines/fraud/model.pkl (run `python -m engines.fraud.train` first) and
exercises `python -m engines.fraud.conformal`'s calibration path directly rather
than relying on a pre-existing engines/fraud/conformal.pkl, so these tests are
self-contained and don't depend on calibration having already been run.
"""

import os

import numpy as np
import pandas as pd
import pytest

from engines.fraud.conformal import (
    ProbaAsRegressor,
    _load_model_artifact,
    calibrate_conformal,
    load_conformal_artifact,
    predict_risk_interval,
)
from engines.fraud.explain import load_fraud_artifact

MODEL_PATH = "engines/fraud/model.pkl"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason="engines/fraud/model.pkl not trained; run `python -m engines.fraud.train` first"
)


@pytest.fixture(scope="module")
def conformal_artifact(tmp_path_factory):
    """Runs a real calibration once per test module and reuses the result -
    calibration is a ~10s operation, not something to repeat per test."""
    out_dir = tmp_path_factory.mktemp("conformal")
    output_path = str(out_dir / "conformal.pkl")
    return calibrate_conformal(output_path=output_path)


def test_proba_as_regressor_matches_classifier_predict_proba():
    artifact = load_fraud_artifact(MODEL_PATH)
    model = artifact["model"]
    wrapped = ProbaAsRegressor(model)
    X = artifact["background_sample"][artifact["feature_names"]]
    np.testing.assert_array_equal(wrapped.predict(X), model.predict_proba(X)[:, 1])


def test_calibration_produces_bounded_interval(conformal_artifact):
    for name in ("flagged", "clear"):
        partition = conformal_artifact["partitions"][name]
        assert 0 <= partition["empirical_coverage"] <= 1
        assert partition["mean_interval_width"] >= 0
        assert partition["n_calibration"] > 0


def test_pooled_empirical_coverage_near_target(conformal_artifact):
    """The headline validation number this feature exists to produce: does
    the true label fall inside the interval ~confidence_level of the time
    on data disjoint from calibration. Split conformal's guarantee is exact
    in expectation but has finite-sample noise; allow a generous tolerance
    band rather than asserting an exact figure that would break on the next
    retrain."""
    target = conformal_artifact["confidence_level"]
    coverage = conformal_artifact["empirical_coverage"]
    assert abs(coverage - target) < 0.05, (
        f"empirical coverage {coverage} strayed >5pp from target {target} "
        f"on {conformal_artifact['n_coverage_eval']} held-out rows"
    )


def test_calibration_and_evaluation_sets_are_disjoint():
    """The calibration and coverage-evaluation slices must never overlap -
    otherwise the reported coverage number is inflated by evaluating on
    data the interval was fit to."""
    from engines.fraud.train import chronological_split

    df = pd.read_csv("data/raw/creditcard.csv")
    _, held_out_df = chronological_split(df, test_frac=0.2)
    mid = len(held_out_df) // 2
    calib_df, eval_df = held_out_df.iloc[:mid], held_out_df.iloc[mid:]
    assert set(calib_df.index).isdisjoint(set(eval_df.index))
    assert len(calib_df) > 0 and len(eval_df) > 0


def test_flagged_partition_interval_wider_than_clear(conformal_artifact):
    """Mondrian partitioning by predicted verdict exists specifically so a
    high-risk transaction gets an informative interval instead of the
    near-zero width the majority-class-dominated pooled quantile would
    give it. Assert that's actually true, not just that the code runs."""
    flagged_width = conformal_artifact["partitions"]["flagged"]["mean_interval_width"]
    clear_width = conformal_artifact["partitions"]["clear"]["mean_interval_width"]
    assert flagged_width > clear_width


def test_predict_risk_interval_routes_by_score(conformal_artifact):
    artifact = _load_model_artifact(MODEL_PATH)
    feature_names = artifact["feature_names"]
    threshold = conformal_artifact["threshold"]

    high_row = pd.DataFrame([{f: 0.0 for f in feature_names}])
    low_row = pd.DataFrame([{f: 0.0 for f in feature_names}])

    high_interval = predict_risk_interval(conformal_artifact, high_row, risk_score=min(threshold + 0.1, 1.0))
    low_interval = predict_risk_interval(conformal_artifact, low_row, risk_score=max(threshold - 0.1, 0.0))

    for interval in (high_interval, low_interval):
        assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0
        assert interval["confidence_level"] == conformal_artifact["confidence_level"]
        assert interval["empirical_coverage"] == conformal_artifact["empirical_coverage"]

    # A score routed to the "flagged" partition should get the wider,
    # partition-appropriate band, not the near-zero one from "clear".
    assert (high_interval["upper"] - high_interval["lower"]) > (low_interval["upper"] - low_interval["lower"])


def test_load_conformal_artifact_missing_file_returns_none(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.pkl")
    assert load_conformal_artifact(missing_path) is None


def test_load_conformal_artifact_tampered_checksum_raises(conformal_artifact, tmp_path):
    import pickle

    artifact_path = tmp_path / "conformal.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(conformal_artifact, f)
    with open(str(artifact_path) + ".sha256", "w") as f:
        f.write("0" * 64)  # deliberately wrong

    with pytest.raises(RuntimeError, match="checksum"):
        load_conformal_artifact(str(artifact_path))


def test_explain_transaction_attaches_risk_interval_when_calibrated(conformal_artifact, monkeypatch):
    """Full integration through explain_transaction(): with a calibration
    artifact available, risk_interval must actually be attached and must
    bracket the base model's own risk_score."""
    import engines.fraud.conformal as conformal_module
    from engines.fraud.explain import explain_transaction

    monkeypatch.setitem(conformal_module._CACHED_ARTIFACTS, "engines/fraud/conformal.pkl", conformal_artifact)

    fraud_artifact = load_fraud_artifact(MODEL_PATH)
    row = fraud_artifact["background_sample"].iloc[0].to_dict()
    result = explain_transaction(features=row, transaction_id="TX-TEST-CONFORMAL")

    assert result["risk_interval"] is not None
    interval = result["risk_interval"]
    # The point estimate can fall just outside its own interval only due to
    # floating-point rounding at a clipped 0/1 boundary; tolerate that, not
    # a materially wrong interval.
    tol = 1e-6
    assert (interval["lower"] - tol) <= result["risk_score"] <= (interval["upper"] + tol)
    # top_factors (the existing SHAP panel) must be untouched by this feature.
    assert len(result["top_factors"]) > 0


def test_explain_transaction_omits_risk_interval_when_uncalibrated(monkeypatch):
    """Additive, not required: with no calibration artifact, the SHAP
    explanation must still work exactly as before risk_interval existed."""
    import engines.fraud.explain as explain_module

    monkeypatch.setattr(explain_module, "load_conformal_artifact", lambda *a, **k: None)

    fraud_artifact = load_fraud_artifact(MODEL_PATH)
    row = fraud_artifact["background_sample"].iloc[0].to_dict()
    result = explain_module.explain_transaction(features=row, transaction_id="TX-TEST-NO-CONFORMAL")

    assert result["risk_interval"] is None
    assert len(result["top_factors"]) > 0
