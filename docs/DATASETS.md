# Dataset Guide & Handling Protocol — Verity

This document outlines the structure, characteristics, and handling rules for all datasets ingested and generated across Verity.

---

## 1. Provided Datasets

### 1.1 Credit Card Fraud Dataset (`data/raw/creditcard.csv`)
* **Source:** Standard European Cardholder Transaction Dataset.
* **Volume:** 284,807 transactions across 2 consecutive days.
* **Class Imbalance:** 492 fraud transactions (0.172% positive rate).
* **Feature Breakdown:**
  * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset (interpretable).
  * `Amount`: Transaction amount in Euro/Dollars (interpretable).
  * `V1` – `V28`: Principal Component Analysis (PCA) transformed numerical features resulting from confidentiality transformations (anonymized signals).
  * `Class`: Binary ground truth target (1 = Fraud, 0 = Legitimate).
* **Handling Rule:** In explainability pipelines (`explain.py`), `Time` and `Amount` are mapped to human-readable explanations; `V1`–`V28` are strictly represented as mathematical behavioral dimensions without speculative human meanings.

---

### 1.2 Bank Ledger Dataset (`data/raw/bank.xlsx`)
* **Source:** Multi-sheet bank ledger exports representing 10 real-world commercial accounts.
* **Characteristics:** Contains transaction dates, values, debit/credit flags, running balances, and unstructured narration strings.
* **Sparse-Data Strategy:** Because the dataset contains 10 independent accounts (not a dense multi-node laundering ring), we do **not** synthesize false cross-account links. Instead, each account is analyzed individually via running baselines and presented via a high-density **horizontal timeline view**.

---

## 2. Generated Synthetic Network (`data/synthetic/synthetic_network.json`)

* **Purpose:** Provides a rich multi-party graph to demonstrate graph-walking capabilities and FATF typology detection algorithms.
* **Topology Generation:**
  * **Nodes:** Synthetic account entities with synthetic attributes (risk ratings, entity types, opening dates).
  * **Edges:** Directed financial transfers with timestamps, amounts, and synthetic payment narrations.
  * **Seeded Clusters:** Structuring smurfs, round-tripping cycles, and rapid layering chains.
  * **Noise:** Random benign transactional traffic mimicking normal commercial patterns.

---

## 3. Data Governance & Git Rules

1. **Large File Exclusion:** All `.csv`, `.xlsx`, `.parquet`, and `.pkl` files in `data/raw/` are excluded via `.gitignore` to keep git operations fast and compliant with GitHub file size limits.
2. **Directory Tracking:** Directory skeletons are tracked using `.gitkeep`.
3. **Mock Data Fixtures:** Lightweight test fixtures representing contract schemas are maintained under `contracts/mock_data/` for rapid component testing.
