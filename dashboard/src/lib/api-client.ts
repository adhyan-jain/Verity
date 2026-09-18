/**
 * Browser-side client for Verity's four backend services. Base URLs and the
 * fraud engine's API key are injected at build time via Vite `VITE_*` env
 * vars (see root `.env.example`); every function fails soft (returns `null`
 * / a fallback payload) rather than throwing into a render path, since the
 * whole point of the dual-engine + agent design is that a down service never
 * blocks the desk — see `docs/VERITY_BUILD_SPEC.md` section 6.
 *
 * NOTE: embedding all four `VITE_*_API_KEY` values in a client bundle is a
 * known prototype-only shortcut (each key is visible to anyone who opens
 * the bundle). Flagged in ARCHITECTURE.md as a follow-up for a real
 * deployment (proxy every engine/agent call through the TanStack Start
 * server instead, keeping keys server-side only).
 */

const AGENT_BASE = import.meta.env.VITE_AGENT_API_URL ?? "http://localhost:8000";
const FRAUD_BASE = import.meta.env.VITE_FRAUD_API_URL ?? "http://localhost:8001/api/v1/fraud";
const LEDGER_BASE = import.meta.env.VITE_LEDGER_API_URL ?? "http://localhost:8002/api/v1/ledger";
const TYPOLOGY_BASE =
  import.meta.env.VITE_TYPOLOGY_API_URL ?? "http://localhost:8003/api/v1/typology";
const FRAUD_API_KEY = import.meta.env.VITE_FRAUD_API_KEY ?? "";
const LEDGER_API_KEY = import.meta.env.VITE_LEDGER_API_KEY ?? "";
const TYPOLOGY_API_KEY = import.meta.env.VITE_TYPOLOGY_API_KEY ?? "";
const AGENT_API_KEY = import.meta.env.VITE_AGENT_API_KEY ?? "";
const HEALTH_TIMEOUT_MS = 6000;
const REQUEST_TIMEOUT_MS = 15000;

export type EngineStatus = {
  agent: boolean;
  fraud: boolean;
  ledger: boolean;
  typology: boolean;
};

export type TopFactor = {
  feature: string;
  human_label: string;
  contribution: number;
  interpretable: boolean;
};

export type FraudExplanation = {
  transaction_id: string;
  risk_score: number;
  verdict: "flagged" | "clear";
  top_factors: TopFactor[];
  model_version: string;
};

export type ReconciliationAnomaly = {
  account_id: string;
  anomaly_type: "balance_break" | "timing_spike" | "reversal_outlier";
  window_start: string;
  window_end: string;
  severity: number;
  baseline_value: number;
  observed_value: number;
  evidence_transaction_ids: string[];
};

export type LedgerTimeline = {
  account_id: string;
  account_name: string;
  total_transactions: number;
  anomalies: ReconciliationAnomaly[];
  transactions: {
    id: string;
    timestamp: string;
    amount: number;
    direction: "debit" | "credit";
    balance: number;
    narration: string | null;
  }[];
};

export type TypologyNetwork = {
  metadata: Record<string, unknown>;
  nodes: {
    account_id: string;
    account_name: string;
    entity_type: string;
    risk_rating: string;
    tier: string;
  }[];
  edges: {
    id: string;
    from_account: string;
    to_account: string;
    amount: number;
    timestamp: string;
    raw_narration: string | null;
  }[];
  ground_truth_flags?: unknown[];
};

export type AgentTraceEvent = {
  event_id: string;
  case_id: string;
  timestamp: string;
  tool_called: string;
  tool_input: Record<string, unknown>;
  tool_output_summary: string;
  narration_sentence: string;
};

export type Case = {
  case_id: string;
  tier_origin: "real_card" | "real_ledger" | "synthetic_network";
  status: "open" | "investigating" | "closed";
  primary_transaction_id: string;
  risk_score: number;
  trace_events: AgentTraceEvent[];
  narrative: string;
};

export type ChatOrCounterfactualResult = {
  response: string;
  is_fallback: boolean;
  fallback_notice: string | null;
  recalculated_risk_score?: number;
};

async function fetchJson<T>(url: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

async function probeHealth(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function checkEnginesHealth(): Promise<EngineStatus> {
  const [agent, fraud, ledger, typology] = await Promise.all([
    probeHealth(`${AGENT_BASE}/health`),
    probeHealth(`${FRAUD_BASE}/health`),
    probeHealth(`${LEDGER_BASE}/health`),
    probeHealth(`${TYPOLOGY_BASE}/health`),
  ]);
  return { agent, fraud, ledger, typology };
}

/** Runs the agent's dynamic investigation loop for a case; returns null on any failure (caller keeps showing fixture data). */
export async function investigateCase(
  caseId: string,
  transactionId: string,
  tierOrigin: Case["tier_origin"],
): Promise<Case | null> {
  try {
    return await fetchJson<Case>(
      `${AGENT_BASE}/api/v1/agent/investigate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": AGENT_API_KEY },
        body: JSON.stringify({
          case_id: caseId,
          transaction_id: transactionId,
          tier_origin: tierOrigin,
        }),
      },
      REQUEST_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
}

/** Direct SHAP factor lookup against the fraud engine (real_card tier only). Null on any failure, including a missing API key. */
export async function fetchShapExplanation(
  transactionId: string,
): Promise<FraudExplanation | null> {
  if (!FRAUD_API_KEY) return null;
  try {
    return await fetchJson<FraudExplanation>(
      `${FRAUD_BASE}/explain/${encodeURIComponent(transactionId)}`,
      { headers: { "X-API-Key": FRAUD_API_KEY } },
      REQUEST_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
}

/** First real account summary from the ledger engine, used to drive live ledger evidence views without hardcoding a bank account number. */
export async function fetchFirstLedgerAccountId(): Promise<string | null> {
  try {
    const accounts = await fetchJson<{ account_id: string }[]>(
      `${LEDGER_BASE}/accounts`,
      { headers: { "X-API-Key": LEDGER_API_KEY } },
      REQUEST_TIMEOUT_MS,
    );
    return accounts[0]?.account_id ?? null;
  } catch {
    return null;
  }
}

export async function fetchLiveTimeline(accountId: string): Promise<LedgerTimeline | null> {
  try {
    return await fetchJson<LedgerTimeline>(
      `${LEDGER_BASE}/timeline/${encodeURIComponent(accountId)}`,
      { headers: { "X-API-Key": LEDGER_API_KEY } },
      REQUEST_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
}

export async function fetchLiveTypologyNetwork(): Promise<TypologyNetwork | null> {
  try {
    return await fetchJson<TypologyNetwork>(
      `${TYPOLOGY_BASE}/network`,
      { headers: { "X-API-Key": TYPOLOGY_API_KEY } },
      REQUEST_TIMEOUT_MS,
    );
  } catch {
    return null;
  }
}

const AMOUNT_PATTERN = /\$?([\d,]+(?:\.\d+)?)/;

/**
 * Routes a free-form analyst query to the agent's counterfactual endpoint
 * (when the question proposes a different amount) or its conversational
 * chat endpoint otherwise. Both endpoints already carry the 20s latency
 * watchdog and cached-answer fallback server-side (`agent/fallback.py`) —
 * this only decides which one to call and normalizes the two response
 * shapes into one for `QueryPanel`.
 */
export async function askCounterfactualOrChat(
  caseId: string,
  transactionId: string,
  query: string,
): Promise<ChatOrCounterfactualResult> {
  const wantsCounterfactual = /what if|instead of|amount (were|was)/i.test(query);
  const amountMatch = wantsCounterfactual ? query.match(AMOUNT_PATTERN) : null;

  if (wantsCounterfactual && amountMatch) {
    const overrideAmount = Number((amountMatch[1] ?? "0").replace(/,/g, ""));
    const result = await fetchJson<{
      original_risk_score: number;
      recalculated_risk_score: number;
      original_verdict: string;
      recalculated_verdict: string;
      explanation?: string;
    }>(
      `${AGENT_BASE}/api/v1/agent/counterfactual`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": AGENT_API_KEY },
        body: JSON.stringify({
          transaction_id: transactionId,
          parameter_overrides: { Amount: overrideAmount },
        }),
      },
      REQUEST_TIMEOUT_MS,
    );
    return {
      response:
        result.explanation ??
        `Re-scoring with Amount = $${overrideAmount.toLocaleString()}: risk moves from ${result.original_risk_score.toFixed(2)} (${result.original_verdict}) to ${result.recalculated_risk_score.toFixed(2)} (${result.recalculated_verdict}).`,
      is_fallback: false,
      fallback_notice: null,
      recalculated_risk_score: result.recalculated_risk_score,
    };
  }

  const result = await fetchJson<{
    response: string;
    is_fallback: boolean;
    fallback_notice: string | null;
    risk_score?: number;
  }>(
    `${AGENT_BASE}/api/v1/agent/chat`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": AGENT_API_KEY },
      body: JSON.stringify({ case_id: caseId, query }),
    },
    REQUEST_TIMEOUT_MS,
  );
  return {
    response: result.response,
    is_fallback: result.is_fallback,
    fallback_notice: result.fallback_notice,
  };
}

export type SentenceCitation = {
  sentence: string;
  event_id: string | null;
  tool_called: string | null;
  timestamp: string | null;
};

export type ValidatorNote = {
  sentence: string;
  reason: string;
};

export type ReasonSuggestion = {
  code: string;
  label: string;
  basis: string;
};

export type StrExportableData = {
  case_id: string;
  tier_origin: string;
  primary_transaction_id: string;
  risk_score: number;
  account_ids: string[];
  transaction_ids: string[];
  amounts_cited: string[];
  date_range: { start: string | null; end: string | null };
  reasons_for_suspicion: ReasonSuggestion[];
  missing_sections: string[];
  generated_at: string;
};

export type StrDraft = {
  narrative: string;
  sentences_with_citations: SentenceCitation[];
  validator_notes: ValidatorNote[];
  exportable_data: StrExportableData;
};

function strDraftRequestBody(investigation: Case) {
  return JSON.stringify({
    tier_origin: investigation.tier_origin,
    primary_transaction_id: investigation.primary_transaction_id,
    risk_score: investigation.risk_score,
    trace_events: investigation.trace_events,
    narrative: investigation.narrative,
  });
}

/**
 * Drafts an STR from a completed investigation. Stateless on the server —
 * nothing is stored, so this always reflects exactly the `investigation`
 * passed in; re-running the investigation and calling this again yields a
 * fresh draft, never a stale cached one.
 */
export async function generateStrDraft(caseId: string, investigation: Case): Promise<StrDraft> {
  return fetchJson<StrDraft>(
    `${AGENT_BASE}/api/v1/agent/draft_str/${encodeURIComponent(caseId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": AGENT_API_KEY },
      body: strDraftRequestBody(investigation),
    },
    REQUEST_TIMEOUT_MS,
  );
}

/** Fetches the same draft rendered as a .docx and triggers a browser download. Throws on failure — caller shows an inline error rather than a silent no-op. */
export async function downloadStrDraftDocx(caseId: string, investigation: Case): Promise<void> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response: globalThis.Response;
  try {
    response = await fetch(
      `${AGENT_BASE}/api/v1/agent/draft_str/${encodeURIComponent(caseId)}/docx`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": AGENT_API_KEY },
        body: strDraftRequestBody(investigation),
        signal: controller.signal,
      },
    );
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok) throw new Error(`STR docx export -> HTTP ${response.status}`);

  const disposition = response.headers.get("content-disposition") ?? "";
  const filenameMatch = /filename="([^"]+)"/.exec(disposition);
  const filename = filenameMatch?.[1] ?? `STR_${caseId}.docx`;

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
