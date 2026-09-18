import {
  ArrowRight,
  Check,
  ChevronRight,
  CircleDot,
  Clock3,
  FileCheck2,
  Network,
  Search,
  Send,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Cpu,
  RefreshCw,
  AlertTriangle,
  Radio,
} from "lucide-react";

import { useEffect, useMemo, useState } from "react";
import { UnifiedAmlCockpit, PaginationControl } from "./aml-screens";
import { TrustScoreMeter } from "./trust-score-meter";
import { NetworkGraph3D } from "./network-graph-3d";
import { formatTimingLabel } from "../lib/copy";
import {
  checkEnginesHealth,
  askCounterfactualOrChat,
  investigateCase,
  fetchShapExplanation,
  fetchFirstLedgerAccountId,
  fetchLiveTimeline,
  fetchLiveTypologyNetwork,
  type EngineStatus,
  type Case,
  type FraudExplanation,
  type LedgerTimeline,
  type TypologyNetwork,
} from "../lib/api-client";

type Filter = "all" | "real_card" | "real_ledger" | "synthetic_network";
type EvidenceView = "timeline" | "network";

type CaseFile = {
  id: string;
  tier: Exclude<Filter, "all">;
  tierLabel: string;
  status: "Open" | "Investigating";
  title: string;
  summary: string;
  amount: string;
  risk: number;
  opened: string;
  transactionId: string;
  factors: { label: string; detail: string; tone: "signal" | "amber" | "teal" }[];
  trace: { id: string; time: string; tool: string; sentence: string }[];
};

const CASES: [CaseFile, ...CaseFile[]] = [
  {
    id: "CASE-SYNTH-003",
    tier: "synthetic_network",
    tierLabel: "Synthetic network",
    status: "Investigating",
    title: "Round-trip laundering across 3 accounts",
    summary:
      "$49,000 returned to its origin through a closed three-account path in under six hours.",
    amount: "$49,000",
    risk: 0.94,
    opened: "09:30",
    transactionId: "TX-SYNTH-5501",
    factors: [
      { label: "Closed transaction loop", detail: "3 accounts · 3 verified hops", tone: "signal" },
      { label: "Rapid circulation", detail: "$49,000 moved within 6 hours", tone: "signal" },
      { label: "FATF typology", detail: "Round-tripping pattern match", tone: "amber" },
    ],
    trace: [
      {
        id: "EVT-301",
        time: "09:30:00",
        tool: "walk_graph",
        sentence:
          "Graph traversal revealed a closed round-tripping topology circulating $49,000 across 3 intermediate accounts within 6 hours.",
      },
    ],
  },
  {
    id: "CASE-CARD-001",
    tier: "real_card",
    tierLabel: "Real card",
    status: "Open",
    title: "Off-hours card transaction",
    summary: "A $4,850 authorization at 03:22 was flagged by amount and temporal signals.",
    amount: "$4,850",
    risk: 0.89,
    opened: "09:15",
    transactionId: "TX-CARD-9842",
    factors: [
      { label: "Transaction amount", detail: "+0.42 model contribution", tone: "signal" },
      { label: "Anonymized signal V14", detail: "+0.28 · meaning not inferred", tone: "amber" },
      { label: "Off-hours timing", detail: "03:22 · +0.19 contribution", tone: "teal" },
    ],
    trace: [
      {
        id: "EVT-101",
        time: "09:15:00",
        tool: "get_transaction",
        sentence:
          "Retrieved transaction TX-CARD-9842 for $4,850.00 processed at an anomalous off-hours timestamp (03:22 AM).",
      },
      {
        id: "EVT-102",
        time: "09:15:05",
        tool: "get_shap_explanation",
        sentence:
          "SHAP feature attribution indicates elevated risk driven primarily by unusually high amount (+0.42 contribution) and temporal anomaly (+0.19 contribution).",
      },
    ],
  },
  {
    id: "CASE-LEDGER-002",
    tier: "real_ledger",
    tierLabel: "Real ledger",
    status: "Open",
    title: "Ledger velocity anomaly",
    summary: "Account ACC-1092 recorded a severe debit spike over a concentrated 48-hour window.",
    amount: "$12,500",
    risk: 0.74,
    opened: "09:20",
    transactionId: "TX-LEDGER-3011",
    factors: [
      { label: "Timing spike", detail: "Velocity exceeded 4× baseline", tone: "signal" },
      { label: "Debit concentration", detail: "$12,500 primary transfer", tone: "amber" },
      { label: "Account evidence", detail: "ACC-1092 · real ledger", tone: "teal" },
    ],
    trace: [
      {
        id: "EVT-201",
        time: "09:20:00",
        tool: "get_transaction",
        sentence:
          "Retrieved ledger entry TX-LEDGER-3011 for account ACC-1092 representing a $12,500.00 debit transfer.",
      },
      {
        id: "EVT-202",
        time: "09:20:06",
        tool: "walk_graph",
        sentence:
          "Account history analysis detected a severe timing spike anomaly where transaction velocity exceeded 4x baseline within a 48-hour window.",
      },
    ],
  },
];

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "real_card", label: "Card" },
  { id: "real_ledger", label: "Ledger" },
  { id: "synthetic_network", label: "Synthetic" },
];

const fallbackPreparedAnswers: Record<string, string> = {
  "CASE-SYNTH-003":
    "The closed loop remains the decisive signal. If the final transfer did not return to the origin, the round-tripping flag would be removed and the case would require a fresh engine score.",
  "CASE-CARD-001":
    "Reducing the amount would lower its +0.42 contribution, but Verity re-runs the calibrated model before changing the verdict; the 03:22 timing signal remains.",
  "CASE-LEDGER-002":
    "A lower transfer amount alone may not clear the case because velocity exceeded the account baseline by more than 4× during the observed window.",
};

/** Live evidence fetched for the currently selected case; any field staying null means that surface keeps showing its labeled "demo exhibit" fixture. */
type LiveEvidence = {
  loading: boolean;
  investigation: Case | null;
  factors: FraudExplanation | null;
  timeline: LedgerTimeline | null;
  network: TypologyNetwork | null;
};

const EMPTY_LIVE_EVIDENCE: LiveEvidence = {
  loading: false,
  investigation: null,
  factors: null,
  timeline: null,
  network: null,
};

function ActionButton({
  children,
  onClick,
  variant = "primary",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "quiet";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        variant === "primary"
          ? "control-button bg-ink text-paper"
          : "control-button border border-border bg-panel text-ink"
      }
    >
      {children}
    </button>
  );
}

function BrandMark() {
  return (
    <div className="brand-mark" aria-hidden="true">
      <span>V</span>
      <i />
    </div>
  );
}

function Masthead({ engines }: { engines: EngineStatus }) {
  const onlineCount = Object.values(engines).filter(Boolean).length;
  return (
    <header className="sticky top-0 z-40 border-b border-ink/10 bg-panel/95 backdrop-blur-md">
      <div className="mx-auto grid max-w-[1480px] grid-cols-[minmax(0,1fr)_auto] items-center gap-4 px-4 py-3 sm:flex sm:justify-between sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <BrandMark />
          <div className="min-w-0 leading-tight">
            <div className="font-display text-lg font-extrabold">Verity</div>
            <div className="truncate font-mono text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
              Financial crime desk
            </div>
          </div>
        </div>
        <div className="hidden items-center gap-2 font-mono text-[10px] lg:flex">
          <span className="status-chip">
            Queue <b>3</b>
          </span>
          <span className="status-chip">
            Open <b className="text-signal">2</b>
          </span>
          <span
            className="status-chip"
            title="Engines: Agent :8000 · Fraud :8001 · Ledger :8002 · Typology :8003"
          >
            <CircleDot
              className={`size-3 ${onlineCount > 0 ? "text-teal animate-pulse" : "text-muted-foreground"}`}
            />
            {onlineCount > 0 ? `Engines online (${onlineCount}/4)` : "Offline fixtures active"}
          </span>
        </div>
        <div className="hidden md:flex items-center">
          <TrustScoreMeter />
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <div className="hidden text-right leading-tight sm:block">
            <div className="text-xs font-semibold">Chitrita G.</div>
            <div className="font-mono text-[9px] uppercase text-muted-foreground">
              Analyst · Tier 2
            </div>
          </div>
          <div className="grid size-9 place-items-center rounded-full border border-border bg-panel-2 font-mono text-xs font-medium text-ink">
            CG
          </div>
        </div>
      </div>
    </header>
  );
}

function CaseRail({
  selectedId,
  onSelect,
}: {
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [search, setSearch] = useState("");
  const [casePage, setCasePage] = useState(1);
  const casePageSize = 3;

  useEffect(() => {
    setCasePage(1);
  }, [filter, search]);

  const cases = CASES.filter(
    (item) =>
      (filter === "all" || item.tier === filter) &&
      `${item.id} ${item.title}`.toLowerCase().includes(search.toLowerCase()),
  );

  const totalCasePages = Math.max(1, Math.ceil(cases.length / casePageSize));
  const safeCasePage = Math.min(Math.max(1, casePage), totalCasePages);
  const paginatedCases = cases.slice((safeCasePage - 1) * casePageSize, safeCasePage * casePageSize);

  return (
    <aside className="animate-rise space-y-3 lg:sticky lg:top-[78px] lg:self-start">
      <div className="flex items-center justify-between px-1">
        <span className="eyebrow">Case queue</span>
        <span className="font-mono text-[10px]">
          {cases.length} / {CASES.length}
        </span>
      </div>
      <label className="flex h-10 items-center gap-2 rounded-md border border-ink/10 bg-panel px-3 shadow-soft">
        <Search className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="sr-only">Search cases</span>
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Find case or signal"
          className="min-w-0 flex-1 bg-transparent text-xs outline-none placeholder:text-muted-foreground"
        />
      </label>
      <div className="grid grid-cols-4 gap-1.5">
        {FILTERS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setFilter(item.id)}
            className={`filter-button ${filter === item.id ? "filter-button-active" : ""}`}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="grid gap-2.5 sm:grid-cols-3 lg:grid-cols-1">
        {paginatedCases.map((item) => {
          const active = selectedId === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              className={`case-item ${active ? "case-item-active" : ""}`}
            >
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
                <span className="truncate font-mono text-[10px]">{item.id}</span>
                <span
                  className={`font-display text-xl font-extrabold ${active ? "" : "text-signal"}`}
                >
                  {item.risk.toFixed(2)}
                </span>
              </div>
              <div className="mt-1 text-sm font-semibold">{item.title}</div>
              <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] gap-2 font-mono text-[9px] uppercase opacity-60">
                <span className="truncate">{item.tierLabel}</span>
                <span>{item.amount}</span>
              </div>
            </button>
          );
        })}
        {cases.length === 0 && (
          <div className="rounded-md border border-dashed border-ink/20 p-5 text-center text-xs text-muted-foreground">
            No matching cases
          </div>
        )}
      </div>
      {cases.length > casePageSize && (
        <PaginationControl
          currentPage={safeCasePage}
          totalPages={totalCasePages}
          totalItems={cases.length}
          pageSize={casePageSize}
          onPageChange={setCasePage}
          itemLabel="cases"
          compact={true}
        />
      )}
      <div className="hidden border-t border-ink/10 px-1 pt-4 font-mono text-[9px] uppercase leading-relaxed text-muted-foreground lg:block">
        Verity Decision Support
        <br />
        Dual-Engine Grounded Workspace
      </div>
    </aside>
  );
}

function RiskHeader({
  item,
  liveRisk,
  riskSource,
}: {
  item: CaseFile;
  liveRisk: number | null;
  riskSource: "counterfactual" | "investigation" | null;
}) {
  const currentRisk = liveRisk !== null ? liveRisk : item.risk;
  const label =
    riskSource === "counterfactual"
      ? "Re-scored risk"
      : riskSource === "investigation"
        ? "Live agent risk"
        : "Risk score";
  return (
    <section className="animate-rise rounded-md bg-panel p-5 shadow-soft md:p-6">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-4 md:gap-8">
        <div className="min-w-0">
          <div className="eyebrow flex items-center gap-2">
            <span className="size-1.5 rounded-full bg-ink" /> Selected · {item.tierLabel}
          </div>
          <h1 className="mt-3 max-w-3xl text-balance font-display text-3xl font-extrabold leading-[1.02] md:text-5xl">
            {item.title}
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {item.summary}
          </p>
          <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[9px] uppercase text-muted-foreground">
            <span>{item.id}</span>
            <span>{item.transactionId}</span>
            <span>Opened {item.opened} UTC</span>
          </div>
        </div>
        <div className="risk-stamp min-w-[94px] shrink-0 px-3 py-3 text-center md:min-w-[138px] md:px-5">
          <div className="eyebrow">{label}</div>
          <div className="mt-1 font-display text-4xl font-extrabold leading-none text-signal md:text-6xl">
            {currentRisk.toFixed(2)}
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-ink/10">
            <div
              className="h-full rounded-full bg-signal transition-[width] duration-700"
              style={{ width: `${Math.min(100, currentRisk * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function EvidenceNetwork({ item, network }: { item: CaseFile; network: TypologyNetwork | null }) {
  const [graphMode, setGraphMode] = useState<"3d" | "2d">("3d");
  const isSynthetic = item.tier === "synthetic_network";
  const isLive = isSynthetic && network !== null;
  const flaggedEdge = isLive
    ? (network!.edges.find((e) => network!.nodes.some((n) => n.account_id === e.from_account)) ??
      network!.edges[0])
    : null;

  return (
    <div className="space-y-2">
      {/* 3D vs 2D Sub-switcher */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-1 bg-paper rounded p-0.5 border border-ink/10 text-[10px] font-mono">
          <button
            type="button"
            onClick={() => setGraphMode("3d")}
            className={`px-2 py-0.5 rounded font-bold transition ${
              graphMode === "3d" ? "bg-signal text-signal-foreground shadow-xs" : "text-muted-foreground hover:text-ink"
            }`}
          >
            3D Force Graph
          </button>
          <button
            type="button"
            onClick={() => setGraphMode("2d")}
            className={`px-2 py-0.5 rounded font-bold transition ${
              graphMode === "2d" ? "bg-signal text-signal-foreground shadow-xs" : "text-muted-foreground hover:text-ink"
            }`}
          >
            2D Schematic
          </button>
        </div>
        <span className="font-mono text-[9px] text-muted-foreground uppercase">
          {graphMode === "3d" ? "WebGL Three.js · Ambient Drift" : "Static Vector Exhibit"}
        </span>
      </div>

      {graphMode === "3d" ? (
        <div className="h-[360px] rounded-lg overflow-hidden border border-ink/10 shadow-soft">
          <NetworkGraph3D selectedAccountId={item.transactionId.includes("SYNTH") ? "ACC-SYN-402" : "409000493210"} />
        </div>
      ) : (
        <div className="network-stage" aria-label="Transaction network visualization">
          <svg
            viewBox="0 0 760 270"
            role="img"
            aria-label={
              isSynthetic ? "Closed three-account transaction loop" : "Transaction evidence path"
            }
          >
            <path className="network-link" d="M165 138 C250 18 413 18 516 116" />
            <path className="network-link network-link-delay" d="M516 116 C590 176 463 246 338 220" />
            <path className="network-link network-link-late" d="M338 220 C228 236 91 196 165 138" />
            <circle className="node-halo" cx="165" cy="138" r="41" />
            <circle className="node-main" cx="165" cy="138" r="23" />
            <circle className="node-halo amber" cx="516" cy="116" r="38" />
            <circle className="node-main amber" cx="516" cy="116" r="21" />
            <circle className="node-halo teal" cx="338" cy="220" r="34" />
            <circle className="node-main teal" cx="338" cy="220" r="19" />
            <text x="113" y="88" className="network-label">
              {isLive
                ? (flaggedEdge?.from_account ?? "ACC-SYN-401")
                : isSynthetic
                  ? "ACC-SYN-401"
                  : "SOURCE"}
            </text>
            <text x="491" y="69" className="network-label">
              {isLive
                ? (flaggedEdge?.to_account ?? "ACC-SYN-402")
                : isSynthetic
                  ? "ACC-SYN-402"
                  : "PRIMARY"}
            </text>
            <text x="301" y="266" className="network-label">
              {isSynthetic ? "ACC-SYN-403" : "EVIDENCE"}
            </text>
            <text x="311" y="58" className="network-amount">
              {isLive && flaggedEdge ? `$${flaggedEdge.amount.toLocaleString()}` : item.amount}
            </text>
          </svg>
          <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between gap-3 font-mono text-[9px] uppercase text-muted-foreground">
            <span>
              {isLive
                ? `Live network · ${network!.nodes.length} accounts · ${network!.edges.length} transactions`
                : isSynthetic
                  ? "Closed loop · 3 hops · 6h"
                  : "Evidence path · bounded view"}
            </span>
            <span>
              {isLive
                ? "Live typology engine"
                : isSynthetic
                  ? "Synthetic data · demo exhibit"
                  : "Real evidence"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function EvidenceTimeline({ item, timeline }: { item: CaseFile; timeline: LedgerTimeline | null }) {
  const isLive = item.tier === "real_ledger" && timeline !== null;
  const points = isLive
    ? timeline!.transactions.slice(0, 4).map((tx, index) => ({
        time: formatTimingLabel(item.tier, tx.timestamp),
        label: `$${tx.amount.toLocaleString()} ${tx.direction}`,
        tone: (["teal", "amber", "signal", "ink"] as const)[index % 4],
      }))
    : item.tier === "real_card"
      ? [
          { time: "02:58", label: "Account observed", tone: "teal" as const },
          { time: "03:22", label: "$4,850 authorization", tone: "amber" as const },
          { time: "03:41", label: `${item.amount} primary event`, tone: "signal" as const },
          { time: "04:02", label: "Case created", tone: "ink" as const },
        ]
      : [
          { time: "Day 1", label: item.tier === "real_ledger" ? "$3,100 debit" : "Account observed", tone: "teal" as const },
          { time: "Day 1", label: "Velocity accelerates", tone: "amber" as const },
          { time: "Day 2", label: `${item.amount} primary event`, tone: "signal" as const },
          { time: "Day 2", label: item.tier === "synthetic_network" ? "Funds return" : "Case created", tone: "ink" as const },
        ];
  return (
    <div className="timeline-stage">
      <div className="timeline-track" />
      {isLive && (
        <div className="mb-2 font-mono text-[9px] uppercase text-teal">
          Live · {timeline!.account_name || timeline!.account_id} · {timeline!.total_transactions}{" "}
          transactions · {timeline!.anomalies.length} anomaly window(s)
        </div>
      )}
      <div className="grid grid-cols-4 gap-2">
        {points.map((point, index) => (
          <div key={`${point.time}-${index}`} className="relative min-w-0 pt-10">
            <span className={`timeline-dot tone-${point.tone}`} />
            <div className="font-mono text-[9px] text-muted-foreground">{point.time}</div>
            <div className="mt-1 text-xs font-medium leading-snug">{point.label}</div>
            <div className="mt-2 hidden font-mono text-[8px] uppercase text-muted-foreground sm:block">
              EX-{String(index + 14).padStart(3, "0")}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidencePanel({
  item,
  timeline,
  network,
}: {
  item: CaseFile;
  timeline: LedgerTimeline | null;
  network: TypologyNetwork | null;
}) {
  const [view, setView] = useState<EvidenceView>(
    item.tier === "synthetic_network" ? "network" : "timeline",
  );
  const isLive = view === "timeline" ? timeline !== null : network !== null;

  return (
    <section className="animate-rise-delay rounded-md bg-panel p-4 shadow-soft md:p-5">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="eyebrow">Evidence surface</span>
            {isLive ? (
              <span className="inline-flex items-center gap-1 rounded bg-teal/15 px-2 py-0.5 font-mono text-[9px] font-semibold uppercase text-teal border border-teal/30">
                <Radio className="size-3 animate-pulse" /> Live Surface
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded border border-border bg-paper px-2 py-0.5 font-mono text-[9px] font-semibold uppercase text-muted-foreground">
                <AlertTriangle className="size-3" /> Cached Exhibit
              </span>
            )}
          </div>
          <h2 className="mt-1 font-display text-xl font-bold">Transaction reconstruction</h2>
        </div>
        <div className="flex shrink-0 gap-1 rounded-md bg-paper p-1 shadow-inset">
          <button
            type="button"
            title="Timeline view"
            aria-label="Timeline view"
            onClick={() => setView("timeline")}
            className={`icon-tab ${view === "timeline" ? "icon-tab-active" : ""}`}
          >
            <Clock3 className="size-4" />
          </button>
          <button
            type="button"
            title="Network view"
            aria-label="Network view"
            onClick={() => setView("network")}
            className={`icon-tab ${view === "network" ? "icon-tab-active" : ""}`}
          >
            <Network className="size-4" />
          </button>
        </div>
      </div>
      <div className="mt-4">
        {view === "network" ? (
          <EvidenceNetwork item={item} network={network} />
        ) : (
          <EvidenceTimeline item={item} timeline={timeline} />
        )}
      </div>
    </section>
  );
}

function Factors({ item, liveFactors }: { item: CaseFile; liveFactors: FraudExplanation | null }) {
  const factors = liveFactors
    ? liveFactors.top_factors.map((f) => ({
        label: f.human_label,
        detail: `${f.contribution >= 0 ? "+" : ""}${f.contribution.toFixed(2)} model contribution${f.interpretable ? "" : " · meaning not inferred"}`,
        tone: f.interpretable ? ("signal" as const) : ("amber" as const),
      }))
    : item.factors;
  return (
    <section className="rounded-md bg-panel p-4 shadow-soft md:p-5">
      <div className="flex items-center justify-between">
        <div className="eyebrow">Risk factors</div>
        {liveFactors ? (
          <span className="inline-flex items-center gap-1 rounded bg-teal/15 px-2 py-0.5 font-mono text-[9px] font-semibold uppercase text-teal border border-teal/30">
            <Radio className="size-3 animate-pulse" /> Live SHAP ({liveFactors.model_version})
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 rounded border border-border bg-paper px-2 py-0.5 font-mono text-[9px] font-semibold uppercase text-muted-foreground">
            <AlertTriangle className="size-3" /> Cached Exhibit
          </span>
        )}
      </div>
      {liveFactors?.risk_interval && (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Risk score <span className="font-mono font-semibold text-ink">{liveFactors.risk_score.toFixed(2)}</span>
          , with a {Math.round(liveFactors.risk_interval.confidence_level * 100)}% confidence interval of{" "}
          <span className="font-mono font-semibold text-ink">
            [{liveFactors.risk_interval.lower.toFixed(2)}, {liveFactors.risk_interval.upper.toFixed(2)}]
          </span>
          <span className="ml-1 font-mono text-[10px] uppercase tracking-wide text-muted-foreground/80">
            (split-conformal · validated at {(liveFactors.risk_interval.empirical_coverage * 100).toFixed(1)}% empirical coverage)
          </span>
        </p>
      )}
      <div className="mt-4 divide-y divide-ink/10">
        {factors.map((factor) => (
          <div
            key={factor.label}
            className="grid grid-cols-[auto_minmax(0,1fr)] gap-3 py-3 first:pt-0 last:pb-0"
          >
            <span className={`mt-1 size-2 rounded-full bg-${factor.tone}`} />
            <div className="min-w-0">
              <div className="text-sm font-semibold">{factor.label}</div>
              <div className="mt-0.5 font-mono text-[9px] uppercase leading-relaxed text-muted-foreground">
                {factor.detail}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ReasoningTrace({
  item,
  investigation,
  loading,
}: {
  item: CaseFile;
  investigation: Case | null;
  loading: boolean;
}) {
  const traceRows = investigation
    ? investigation.trace_events.map((evt) => ({
        id: evt.event_id,
        time: evt.timestamp.slice(11, 19),
        tool: evt.tool_called,
        sentence: evt.narration_sentence,
      }))
    : item.trace;
  return (
    <section className="rounded-md bg-panel p-4 shadow-soft md:p-5">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3">
        <div>
          <div className="eyebrow">Reasoning trace</div>
          <h2 className="mt-1 font-display text-xl font-bold">Every sentence has a source</h2>
        </div>
        <span className="flex shrink-0 items-center gap-1.5 font-mono text-[9px] uppercase">
          {loading ? (
            <span className="inline-flex items-center gap-1 rounded bg-amber/15 px-2 py-0.5 font-semibold text-amber border border-amber/30">
              <RefreshCw className="size-3 animate-spin" /> Investigating…
            </span>
          ) : investigation ? (
            <span className="inline-flex items-center gap-1 rounded bg-teal/15 px-2 py-0.5 font-semibold text-teal border border-teal/30">
              <Radio className="size-3 animate-pulse" /> Live Grounded Trace
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 rounded border border-border bg-paper px-2 py-0.5 font-semibold text-muted-foreground">
              <AlertTriangle className="size-3" /> Cached Demo Trace (Agent Offline)
            </span>
          )}
        </span>
      </div>
      <div className="mt-4 space-y-0">
        {traceRows.map((event, index) => (
          <article key={event.id} className="trace-row">
            <div className="trace-index">{index + 1}</div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[9px] uppercase text-muted-foreground">
                <span className="text-signal">{event.id}</span>
                <span>{event.time}</span>
                <span>{event.tool}</span>
              </div>
              <p className="mt-2 text-sm leading-relaxed">{event.sentence}</p>
            </div>
          </article>
        ))}
      </div>
      {investigation?.narrative && (
        <p className="mt-4 border-t border-ink/10 pt-4 text-sm leading-relaxed text-muted-foreground">
          {investigation.narrative}
        </p>
      )}
    </section>
  );
}

function QueryPanel({

  item,
  onScoreChange,
}: {
  item: CaseFile;
  onScoreChange: (score: number | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const run = async () => {
    if (!query.trim()) return;
    setRunning(true);
    setAnswer("");
    setNotice(null);

    try {
      const res = await askCounterfactualOrChat(item.id, item.transactionId, query);
      setAnswer(res.response);
      setNotice(
        res.fallback_notice ?? (res.is_fallback ? "Using a prepared benchmark answer." : null),
      );
      if (res.recalculated_risk_score !== undefined) {
        onScoreChange(res.recalculated_risk_score);
      }
    } catch {
      setAnswer(
        fallbackPreparedAnswers[item.id] ??
          "No prepared benchmark answer is available for this case.",
      );
      setNotice("Using a prepared benchmark answer · model not rerun");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="rounded-md bg-ink p-4 text-paper shadow-lift md:p-5">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-3">
        <div>
          <div className="eyebrow text-paper/50">Counterfactual desk</div>
          <h2 className="mt-1 font-display text-xl font-bold">Interrogate the evidence</h2>
        </div>
        <Sparkles className={running ? "size-4 text-amber animate-pulse" : "size-4 text-muted-foreground"} />
      </div>
      {answer && (
        <div className="mt-4 border-l-2 border-teal bg-paper/5 px-4 py-3 text-sm leading-relaxed">
          {notice && (
            <div className="mb-1 font-mono text-[9px] uppercase text-paper/45">{notice}</div>
          )}
          {answer}
        </div>
      )}
      <form
        className="mt-4 flex flex-col gap-2 sm:flex-row"
        onSubmit={(event) => {
          event.preventDefault();
          run();
        }}
      >
        <label className="flex min-h-11 flex-1 items-center gap-2 rounded-md border border-paper/15 bg-paper/5 px-3">
          <ChevronRight className="size-4 shrink-0 text-amber" />
          <span className="sr-only">Ask a counterfactual question</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={
              item.tier === "real_card"
                ? "What if the amount were $1,200?"
                : "What evidence would change this score?"
            }
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-paper/40"
          />
        </label>
        <button
          type="submit"
          disabled={running || !query.trim()}
          className={`control-button min-h-11 disabled:cursor-not-allowed disabled:opacity-50 ${
            running ? "bg-amber text-amber-foreground" : "bg-paper text-ink"
          }`}
        >
          {running ? (
            <>
              <span className="size-2 animate-pulse rounded-full bg-amber-foreground" />{" "}
              Investigating
            </>
          ) : (
            <>
              <Send className="size-3.5" /> Run analysis
            </>
          )}
        </button>
      </form>
    </section>
  );
}

export function VerityWorkspace() {
  const [selectedId, setSelectedId] = useState(CASES[0].id);
  const [decision, setDecision] = useState<string | null>(null);
  const [counterfactualRisk, setCounterfactualRisk] = useState<number | null>(null);
  const [engines, setEngines] = useState<EngineStatus>({
    agent: false,
    fraud: false,
    ledger: false,
    typology: false,
  });
  const [live, setLive] = useState<LiveEvidence>(EMPTY_LIVE_EVIDENCE);

  const item = useMemo(
    () => CASES.find((entry) => entry.id === selectedId) ?? CASES[0],
    [selectedId],
  );

  useEffect(() => {
    let mounted = true;
    const updateHealth = () => {
      checkEnginesHealth().then((status) => {
        if (!mounted) return;
        const count = Object.values(status).filter(Boolean).length;
        // If all 4 failed (common when laptop or tab wakes from sleep), retry once before dropping to offline
        if (count === 0) {
          setTimeout(() => {
            if (!mounted) return;
            checkEnginesHealth().then((retryStatus) => {
              if (mounted) setEngines(retryStatus);
            });
          }, 1200);
        } else {
          setEngines(status);
        }
      });
    };

    updateHealth();
    const interval = setInterval(updateHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // Replace fixture evidence with live engine/agent data for the selected case.
  // Every fetch fails soft (see lib/api-client.ts); panels fall back to the
  // labeled fixture below rather than showing a broken/empty state.
  useEffect(() => {
    let cancelled = false;
    setLive({ ...EMPTY_LIVE_EVIDENCE, loading: true });

    (async () => {
      const [investigation, factors, network] = await Promise.all([
        investigateCase(item.id, item.transactionId, item.tier),
        item.tier === "real_card"
          ? fetchShapExplanation(item.transactionId)
          : Promise.resolve(null),
        item.tier === "synthetic_network" ? fetchLiveTypologyNetwork() : Promise.resolve(null),
      ]);

      let timeline: LedgerTimeline | null = null;
      if (item.tier === "real_ledger") {
        const accountId = await fetchFirstLedgerAccountId();
        timeline = accountId ? await fetchLiveTimeline(accountId) : null;
      }

      if (!cancelled) {
        setLive({ loading: false, investigation, factors, timeline, network });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [item.id, item.tier, item.transactionId]);

  const [activeDesk, setActiveDesk] = useState<"aml" | "multi_tier">("aml");

  const liveRisk = counterfactualRisk ?? live.investigation?.risk_score ?? null;
  const riskSource: "counterfactual" | "investigation" | null =
    counterfactualRisk !== null ? "counterfactual" : live.investigation ? "investigation" : null;

  return (
    <div className="min-h-screen bg-paper text-ink">
      <Masthead engines={engines} />

      {/* Top Level Desk Switcher */}
      <div className="border-b border-ink/10 bg-panel/85 backdrop-blur px-4 sm:px-6 py-2.5 sticky top-0 z-30 shadow-xs">
        <div className="mx-auto flex max-w-[1520px] items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 p-1 bg-paper rounded-lg border border-ink/10 text-xs font-mono">
            <button
              type="button"
              onClick={() => setActiveDesk("aml")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md font-bold transition ${
                activeDesk === "aml"
                  ? "bg-signal text-signal-foreground shadow-sm"
                  : "text-muted-foreground hover:text-ink hover:bg-ink/5"
              }`}
            >
              <ShieldAlert className="size-3.5" />
              <span>AML Operations Cockpit (Screens 1–5)</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveDesk("multi_tier")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md font-bold transition ${
                activeDesk === "multi_tier"
                  ? "bg-signal text-signal-foreground shadow-sm"
                  : "text-muted-foreground hover:text-ink hover:bg-ink/5"
              }`}
            >
              <Cpu className="size-3.5" />
              <span>Multi-Tier Architecture Showcase</span>
            </button>
          </div>

          <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-teal animate-pulse" />
              <span className="text-ink font-semibold">Live Operational Gate Active</span>
            </div>
            <span>·</span>
            <span>Gate 1: Date-Only bank.xlsx</span>
          </div>
        </div>
      </div>

      {activeDesk === "aml" ? (
        <div className="mx-auto max-w-[1520px] px-4 py-4 sm:px-6">
          <UnifiedAmlCockpit />
        </div>
      ) : (
        <div className="mx-auto grid max-w-[1480px] grid-cols-1 gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[290px_minmax(0,1fr)] lg:gap-6 lg:py-6">
          <CaseRail
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id);
              setDecision(null);
              setCounterfactualRisk(null);
            }}
          />
          <main className="min-w-0 space-y-4">
            {!live.loading && !live.investigation && (
              <div className="flex items-start gap-3 rounded-md border border-border bg-panel p-4 text-xs text-muted-foreground font-mono">
                <AlertTriangle className="size-4 shrink-0 mt-0.5" />
                <div className="leading-relaxed">
                  <span className="font-bold uppercase tracking-wider text-ink">Showing Cached Data — Live Service Unavailable:</span>{" "}
                  The workspace is rendering pre-computed benchmark fixtures because backend microservices are offline. Run{" "}
                  <code className="rounded bg-paper px-1.5 py-0.5 font-bold text-ink">python scripts/dev_up.py</code> to connect live LightGBM/SHAP and FATF traversal engines (Ports 8000–8003).
                </div>
              </div>
            )}
            <RiskHeader item={item} liveRisk={liveRisk} riskSource={riskSource} />

            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(260px,0.7fr)]">
              <EvidencePanel
                key={`${item.id}-evidence`}
                item={item}
                timeline={live.timeline}
                network={live.network}
              />
              <Factors item={item} liveFactors={live.factors} />
            </div>
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.85fr)]">
              <ReasoningTrace item={item} investigation={live.investigation} loading={live.loading} />
              <div className="space-y-4">
                <section className="rounded-md bg-panel p-4 shadow-soft md:p-5">
                  <div className="eyebrow">Analyst disposition</div>
                  {decision ? (
                    <div className="mt-4 flex items-center gap-3 rounded-md bg-cleared/10 p-3 text-sm text-cleared">
                      <Check className="size-4 shrink-0" /> Marked "{decision}" in this session
                    </div>
                  ) : (
                    <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
                      <ActionButton onClick={() => setDecision("Escalate")}>
                        Escalate <ArrowRight className="size-3.5" />
                      </ActionButton>
                      <ActionButton variant="quiet" onClick={() => setDecision("Needs review")}>
                        <FileCheck2 className="size-3.5" /> Needs review
                      </ActionButton>
                    </div>
                  )}
                  <p className="mt-3 font-mono text-[9px] uppercase leading-relaxed text-muted-foreground">
                    Decision Support Prototype · Auditable Trace
                  </p>
                </section>
                <QueryPanel
                  key={`${item.id}-query`}
                  item={item}
                  onScoreChange={(newScore) => setCounterfactualRisk(newScore)}
                />
              </div>
            </div>
            <footer className="py-5 text-center font-mono text-[9px] uppercase leading-relaxed text-muted-foreground">
              Verity prototype · synthetic exhibits are labeled · grounded reasoning engine
            </footer>
          </main>
        </div>
      )}
    </div>
  );
}
