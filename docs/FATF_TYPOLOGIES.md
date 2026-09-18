# FATF Typology Reference & Detection Rules — Verity

Verity incorporates formal definitions from the **Financial Action Task Force (FATF)** 40 Recommendations and typology guidance to power its synthetic network detection engine.

---

## 1. Structuring (Smurfing)

### 1.1 FATF Definition & Citation
* **FATF Reference:** *FATF 40 Recommendations (Recommendation 10 / Customer Due Diligence Criteria 10.1 & 10.3)*
* **Mechanism:** Dividing large currency transactions into multiple smaller transactions to avoid threshold-based reporting requirements (e.g., just below the $10,000 / ₹10,00,000 regulatory trigger limit).

### 1.2 Mathematical Formulation & Detection Criteria
A subgraph pattern represents structuring if:
1. **Threshold Closeness:** Each transaction $e_i$ satisfies:
   $$\theta_{\text{min}} \le \text{amount}(e_i) < \theta_{\text{threshold}}$$
   *(e.g., $9,000 \le \text{amount} < $10,000)*
2. **Temporal Proximity:** For a set of deposits $D = \{e_1, e_2, \dots, e_k\}$ into target account $A_{\text{collector}}$:
   $$\max(t_i) - \min(t_i) \le \Delta T_{\text{window}} \quad (\text{typically } 24\text{ hours})$$
3. **Cumulative Volume:**
   $$\sum_{i=1}^k \text{amount}(e_i) \ge 2.5 \times \theta_{\text{threshold}}$$

---

## 2. Round-Tripping (Circular Fund Flows)

### 2.1 FATF Definition & Citation
* **FATF Reference:** *FATF Guidance on Concealment of Beneficial Ownership (October 2018) & Trade-Based Money Laundering Typologies.*
* **Mechanism:** Routing domestic funds through offshore accounts or shell companies and returning them to the originator disguised as legitimate foreign investment, loan repayments, or service fees.

### 2.2 Mathematical Formulation & Detection Criteria
A directed cycle $C = (v_1, v_2, \dots, v_n, v_1)$ in graph $G$ represents round-tripping if:
1. **Cycle Preservation:** The initial sender $v_1$ is also the final receiver $v_1$.
2. **Volume Conservation:** The returned amount retains at least $\beta\%$ of the original outflow after minor intermediary friction fees:
   $$\frac{\text{amount}(v_n \to v_1)}{\text{amount}(v_1 \to v_2)} \ge \beta \quad (\text{where } \beta \ge 0.85)$$
3. **Rapid Turnaround:** Intermediary hops occur within a compressed window:
   $$t(v_n \to v_1) - t(v_1 \to v_2) \le \Delta T_{\text{cycle}} \quad (\text{e.g., } \le 72\text{ hours})$$

---

## 3. Rapid Layering (High-Velocity Pass-Through Transfers)

### 3.1 FATF Definition & Citation
* **FATF Reference:** *FATF Money Laundering Typologies — Pass-Through Accounts and Layering Techniques.*
* **Mechanism:** Rapidly moving illicit funds through multiple intermediary accounts to obscure the transaction trail and distance the proceeds from the originating predicate offense.

### 3.2 Mathematical Formulation & Detection Criteria
A directed path $P = (v_1 \to v_2 \to \dots \to v_m)$ represents rapid layering if:
1. **Hop Count:**
   $$m \ge 3 \quad (\ge 3 \text{ distinct hops})$$
2. **Low Dwell Time (High Velocity):** For consecutive hops $e_k = (v_k \to v_{k+1})$ and $e_{k+1} = (v_{k+1} \to v_{k+2})$:
   $$t(e_{k+1}) - t(e_k) \le \Delta T_{\text{hop}} \quad (\text{where } \Delta T_{\text{hop}} \le 2\text{ hours})$$
3. **Pass-Through Ratio:** Inflow approximately equals outflow at each node with minimal balance retention:
   $$\left| \frac{\text{outflow}(v_k) - \text{inflow}(v_k)}{\text{inflow}(v_k)} \right| \le 0.10 \quad \forall v_k \in \text{Intermediaries}$$

---

## 4. Adversarial Hold-Out Validation (`adversarial_set.json`)

To prevent overfitting heuristic detectors:
* An independent test suite of synthetic edge cases (e.g. legitimate high-velocity merchant batch payouts, payroll splits, dividend returns) is maintained in `data/synthetic/adversarial_set.json`.
* Both **True Positive Rate (Typology Recall)** and **False Positive Rate** are tracked and validated.
