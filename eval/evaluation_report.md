# Verity Model Evaluation Report
Generated: 2026-09-18T20:08:35Z

Every row below was produced by actually executing the corresponding model/detector in this process (see eval/run_all_evaluations.py) - not copied from a doc or a stored artifact.

| model | dataset | accuracy | data_authenticity | f1 | hallucinations_blocked | pr_auc | precision | recall | retention_rate | sentences_retained | sentences_total | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | creditcard |  | real | 0.9302 |  | 0.9697 | 0.9524 | 0.9091 |  |  |  | 0.195 |
| RandomForest | creditcard |  | real | 0.9565 |  | 0.9847 | 0.9167 | 1.0 |  |  |  | 0.3101 |
| LightGBM | kartik2112 (synthetic fallback) |  | synthetic_fallback | 0.8696 |  | 0.9279 | 0.9259 | 0.8197 |  |  |  | 0.919 |
| RandomForest | kartik2112 (synthetic fallback) |  | synthetic_fallback | 0.8618 |  | 0.9257 | 0.8548 | 0.8689 |  |  |  | 0.6189 |
| FATF structuring detector | independent adversarial set (n=11) | 0.9091 | synthetic (labeled adversarial cases) | 0.8889 |  |  | 1.0 | 0.8 |  |  |  |  |
| Grounding filter | hallucination-injection probe |  | synthetic (deterministic probe) |  | 1 |  |  |  | 66.7% | 2 | 3 |  |

## Skipped (honest gaps)

- **PaySim RandomForest** / PaySim CSV: raw dataset not found at /home/adhyan/Desktop/Vertis/data/Pay_sim/PS_20174392719_1491204439457_log.csv
