# API Specification & Data Contracts — Verity

This document outlines the REST API endpoints and data contracts exposed by Verity detection engines, the AI agent service, and the analyst dashboard.

All endpoints adhere to JSON Schemas in [`contracts/schemas.json`](file:///c:/Users/harsh/Desktop/Web%20Dev/Verity/contracts/schemas.json).

---

## 1. Fraud Engine API (`engines/fraud/api.py`)

**Base URL:** `http://localhost:8001/api/v1/fraud`

### 1.1 Get Transaction Details
* **Endpoint:** `GET /transaction/{transaction_id}`
* **Description:** Retrieves full transactional record for a credit card event.
* **Path Parameters:**
  * `transaction_id` (string, required): Unique identifier of the transaction.
* **Response (200 OK):**
```json
{
  "id": "TX-CARD-9842",
  "tier": "real_card",
  "timestamp": "2026-09-18T03:22:00Z",
  "account_id": null,
  "amount": 4850.00,
  "direction": "debit",
  "raw_narration": null,
  "source_dataset": "creditcard.csv"
}
```

---

### 1.2 Get SHAP Feature Attribution
* **Endpoint:** `GET /explain/{transaction_id}`
* **Description:** Computes SHAP values and segregates interpretable features from anonymized PCA dimensions.
* **Response (200 OK):**
```json
{
  "transaction_id": "TX-CARD-9842",
  "risk_score": 0.89,
  "verdict": "flagged",
  "top_factors": [
    {
      "feature": "Amount",
      "human_label": "Transaction amount ($4,850.00)",
      "contribution": 0.42,
      "interpretable": true
    },
    {
      "feature": "Time",
      "human_label": "Transaction time (03:22 AM)",
      "contribution": 0.19,
      "interpretable": true
    },
    {
      "feature": "V14",
      "human_label": "Anonymized behavioral signal V14",
      "contribution": 0.28,
      "interpretable": false
    }
  ],
  "model_version": "v1.0-benchmark-winner"
}
```

---

## 2. Ledger Engine API (`engines/ledger/api.py`)

**Base URL:** `http://localhost:8002/api/v1/ledger`

### 2.1 Get Reconciliation Anomalies
* **Endpoint:** `GET /anomalies/{account_id}`
* **Description:** Retrieves balance breaks, timing spikes, and reversal outliers detected for a given account.
* **Response (200 OK):**
```json
[
  {
    "account_id": "ACC-1092",
    "anomaly_type": "timing_spike",
    "window_start": "2026-09-15T00:00:00Z",
    "window_end": "2026-09-17T23:59:59Z",
    "severity": 0.88,
    "baseline_value": 2.1,
    "observed_value": 9.4,
    "evidence_transaction_ids": ["TX-LEDGER-3011", "TX-LEDGER-3012", "TX-LEDGER-3015"]
  }
]
```

---

### 2.2 Get Ledger Timeline Data
* **Endpoint:** `GET /timeline/{account_id}`
* **Description:** Retrieves chronologically ordered transactions and overlaid anomaly windows for horizontal timeline visualization.
* **Response (200 OK):**
```json
{
  "account_id": "ACC-1092",
  "account_name": "Acme Retail Trade Account",
  "total_transactions": 48,
  "anomalies": [
    {
      "anomaly_type": "balance_break",
      "severity": 0.92,
      "window_start": "2026-09-16T12:00:00Z",
      "window_end": "2026-09-16T18:00:00Z",
      "baseline_value": 15400.0,
      "observed_value": -3200.0
    }
  ],
  "transactions": [
    {
      "id": "TX-LEDGER-3011",
      "timestamp": "2026-09-16T14:15:00Z",
      "amount": 12500.0,
      "direction": "debit",
      "balance": -800.0,
      "narration": "BULK UNREGISTERED TXFR"
    }
  ]
}
```

---

## 3. Typology Engine API (`engines/typology/api.py`)

**Base URL:** `http://localhost:8003/api/v1/typology`

### 3.1 Get Typology Flags
* **Endpoint:** `GET /flags`
* **Description:** Retrieves all identified FATF laundering flags across the synthetic network.
* **Response (200 OK):**
```json
[
  {
    "flag_id": "FLAG-FATF-001",
    "typology": "round_tripping",
    "fatf_reference": "FATF Guidance on Concealment of Beneficial Ownership (Oct 2018)",
    "involved_accounts": ["ACC-SYN-401", "ACC-SYN-402", "ACC-SYN-403"],
    "evidence_transaction_ids": ["TX-SYNTH-5501", "TX-SYNTH-5502", "TX-SYNTH-5503"],
    "confidence": 0.94
  }
]
```

---

### 3.2 Walk Synthetic Graph
* **Endpoint:** `GET /walk/{account_id}?depth=2`
* **Description:** Traverses connected nodes in the synthetic laundering network.
* **Response (200 OK):**
```json
[
  {
    "step_index": 1,
    "from_account": "ACC-SYN-401",
    "to_account": "ACC-SYN-402",
    "tier": "synthetic_network",
    "amount": 49000.0,
    "timestamp": "2026-09-18T06:00:00Z",
    "narration": "CONSULTING RETAINER FEE",
    "tool_call_id": "TOOL-CALL-8821"
  }
]
```

---

## 4. Agent Tools Contract (`agent/tools.py`)

The agent interacts with the backend engines through four strictly typed Python tool wrappers:

```python
def get_transaction(transaction_id: str) -> TransactionRecord: ...
def get_shap_explanation(transaction_id: str) -> FraudExplanation: ...
def walk_graph(account_id: str, tier: str = "real_ledger", depth: int = 2) -> List[GraphWalkStep]: ...
def counterfactual(transaction_id: str, parameter_overrides: dict) -> CounterfactualResult: ...
```

---

## 5. Counterfactual Interface

* **Endpoint:** `POST /api/v1/agent/counterfactual`
* **Request Body:**
```json
{
  "transaction_id": "TX-CARD-9842",
  "parameter_overrides": {
    "Amount": 150.00,
    "Time": 50000
  }
}
```
* **Response (200 OK):**
```json
{
  "transaction_id": "TX-CARD-9842",
  "original_risk_score": 0.89,
  "recalculated_risk_score": 0.18,
  "original_verdict": "flagged",
  "recalculated_verdict": "clear",
  "modifications": {
    "Amount": 150.00,
    "Time": 50000
  }
}
```
