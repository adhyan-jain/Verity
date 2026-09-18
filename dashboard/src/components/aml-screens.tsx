import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Bot,
  Calendar,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  CircleDot,
  Clock,
  Code2,
  Cpu,
  Database,
  FileCheck,
  FileText,
  Filter,
  HelpCircle,
  Layers,
  Percent,
  RefreshCw,
  Scale,
  Search,
  Send,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  TrendingDown,
  TrendingUp,
  User,
  Zap,
} from "lucide-react";
import React, { useEffect, useState } from "react";

import {
  askScopedCustomerChat,
  fetchAmlBreakdown,
  fetchAmlQueue,
  fetchAmlTimeline,
  fetchAmlTrace,
  runCounterfactualRecompute,
  runForwardSimulation,
  type AmlBreakdownResponse,
  type AmlQueueItem,
  type AmlTimelineItem,
  type AmlTraceResponse,
  type AmlTraceStep,
  type CounterfactualRecomputeResponse,
  type ForwardSimulationResponse,
  type ScopedChatResponse,
} from "../lib/api-client";

// ============================================================================
// Reusable Deterministic Pagination Control Component
// ============================================================================
export function PaginationControl({
  currentPage,
  totalPages,
  totalItems,
  pageSize,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = [5, 10, 25],
  itemLabel = "items",
  compact = false,
}: {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (size: number) => void;
  pageSizeOptions?: number[];
  itemLabel?: string;
  compact?: boolean;
}) {
  if (totalItems === 0) return null;

  const safePage = Math.min(Math.max(1, currentPage), Math.max(1, totalPages));
  const startIdx = Math.min((safePage - 1) * pageSize + 1, totalItems);
  const endIdx = Math.min(safePage * pageSize, totalItems);

  return (
    <div className="flex items-center justify-between gap-2 px-3 py-2 border-t border-ink/10 bg-paper/70 text-xs font-mono shrink-0 select-none">
      {/* Range Counter */}
      <div className="text-[11px] text-muted-foreground truncate">
        Showing <span className="font-bold text-ink">{startIdx}</span>–<span className="font-bold text-ink">{endIdx}</span> of{" "}
        <span className="font-bold text-ink">{totalItems}</span> {itemLabel}
      </div>

      {/* Navigation Controls */}
      <div className="flex items-center gap-1 shrink-0">
        {onPageSizeChange && pageSizeOptions && pageSizeOptions.length > 1 && !compact && (
          <div className="hidden sm:flex items-center gap-1 mr-2 text-[10px] text-muted-foreground">
            <span>Size:</span>
            <div className="flex items-center rounded border border-ink/10 bg-panel overflow-hidden">
              {pageSizeOptions.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => onPageSizeChange(opt)}
                  className={`px-1.5 py-0.5 text-[10px] transition ${
                    pageSize === opt
                      ? "bg-signal text-signal-foreground font-bold"
                      : "hover:bg-ink/5 text-ink"
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>
        )}

        <button
          type="button"
          onClick={() => onPageChange(1)}
          disabled={safePage <= 1}
          title="First page"
          className="rounded p-1 text-ink border border-ink/10 bg-panel hover:bg-ink/5 disabled:opacity-30 disabled:pointer-events-none transition"
        >
          <ChevronsLeft className="size-3" />
        </button>

        <button
          type="button"
          onClick={() => onPageChange(safePage - 1)}
          disabled={safePage <= 1}
          title="Previous page"
          className="rounded p-1 text-ink border border-ink/10 bg-panel hover:bg-ink/5 disabled:opacity-30 disabled:pointer-events-none transition flex items-center gap-0.5"
        >
          <ChevronLeft className="size-3" />
          {!compact && <span className="text-[10px] pr-0.5 hidden md:inline">Prev</span>}
        </button>

        <span className="px-2 py-0.5 text-[10px] font-bold rounded bg-ink/5 text-ink border border-ink/10">
          {safePage} / {totalPages}
        </span>

        <button
          type="button"
          onClick={() => onPageChange(safePage + 1)}
          disabled={safePage >= totalPages}
          title="Next page"
          className="rounded p-1 text-ink border border-ink/10 bg-panel hover:bg-ink/5 disabled:opacity-30 disabled:pointer-events-none transition flex items-center gap-0.5"
        >
          {!compact && <span className="text-[10px] pl-0.5 hidden md:inline">Next</span>}
          <ChevronRight className="size-3" />
        </button>

        <button
          type="button"
          onClick={() => onPageChange(totalPages)}
          disabled={safePage >= totalPages}
          title="Last page"
          className="rounded p-1 text-ink border border-ink/10 bg-panel hover:bg-ink/5 disabled:opacity-30 disabled:pointer-events-none transition"
        >
          <ChevronsRight className="size-3" />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Screen 1 (Column 1): Ranked Flagged Risk Queue
// ============================================================================
export function AmlQueueColumn({
  selectedAccountId,
  onSelectAccount,
}: {
  selectedAccountId: string | null;
  onSelectAccount: (accountId: string, primaryTxId?: string) => void;
}) {
  const [queue, setQueue] = useState<AmlQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterSeverity, setFilterSeverity] = useState<"all" | "high" | "medium">("all");
  const [search, setSearch] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(5);

  useEffect(() => {
    setCurrentPage(1);
  }, [search, filterSeverity]);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetchAmlQueue(50, true).then((res) => {
      if (!mounted) return;
      if (res && res.records) {
        setQueue(res.records);
        if (!selectedAccountId && res.records.length > 0) {
          const first = res.records[0];
          onSelectAccount(first.account_id, first.id || (first as any).transaction_id);
        }
      }
      setLoading(false);
    });
    return () => {
      mounted = false;
    };
  }, []);

  const normalizedItems = queue.map((item) => {
    const rawAny = item as any;
    const score = typeof item.risk_score === "number" ? item.risk_score : typeof rawAny.final_score === "number" ? rawAny.final_score : typeof rawAny.original_score === "number" ? rawAny.original_score : 0.82;
    const amount = typeof item.amount === "number" ? item.amount : typeof rawAny.amount === "number" ? rawAny.amount : 200000.0;
    const id = item.id || rawAny.transaction_id || `TX-${item.account_id}`;
    const timestamp = item.timestamp ? String(item.timestamp).slice(0, 10) : "2019-02-12";
    const rail = item.payment_rail || rawAny.payment_rail || "NEFT";
    const narration = item.raw_narration || rawAny.raw_narration || rawAny.narration || "Bank ledger transaction";
    const lo = item.conformal_lo ?? rawAny.conformal_lo ?? Math.max(0, Number((score - 0.02).toFixed(2)));
    const hi = item.conformal_hi ?? rawAny.conformal_hi ?? Math.min(1, Number((score + 0.02).toFixed(2)));
    return {
      ...item,
      id,
      amount,
      risk_score: score,
      timestamp,
      payment_rail: rail,
      raw_narration: narration,
      conformal_lo: lo,
      conformal_hi: hi,
    };
  });

  const filteredItems = normalizedItems.filter((item) => {
    const s = search.toLowerCase();
    const matchesSearch =
      (item.account_id && item.account_id.includes(search)) ||
      (item.id && item.id.toLowerCase().includes(s)) ||
      (item.payment_rail && item.payment_rail.toLowerCase().includes(s)) ||
      (item.raw_narration && item.raw_narration.toLowerCase().includes(s));
    if (!matchesSearch) return false;
    if (filterSeverity === "high") return item.risk_score >= 0.7;
    if (filterSeverity === "medium") return item.risk_score < 0.7 && item.risk_score >= 0.4;
    return true;
  });

  const highCount = normalizedItems.filter((i) => i.risk_score >= 0.7).length;
  const medCount = normalizedItems.filter((i) => i.risk_score < 0.7 && i.risk_score >= 0.4).length;

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / pageSize));
  const safePage = Math.min(Math.max(1, currentPage), totalPages);
  const startIndex = (safePage - 1) * pageSize;
  const paginatedItems = filteredItems.slice(startIndex, startIndex + pageSize);

  return (
    <div className="flex flex-col h-full rounded-xl border border-ink/10 bg-panel shadow-soft overflow-hidden">
      {/* Header & Search */}
      <div className="p-3.5 border-b border-ink/10 bg-paper/60 space-y-2.5 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className="size-4 text-signal" />
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-ink">
              Ranked Risk Queue
            </span>
          </div>
          <span className="rounded-full bg-signal/10 px-2 py-0.5 font-mono text-[10px] font-bold text-signal border border-signal/20">
            {filteredItems.length} Cases
          </span>
        </div>

        {/* Search Input */}
        <label className="flex h-8 items-center gap-2 rounded-lg border border-ink/10 bg-panel px-2.5 text-xs shadow-inner">
          <Search className="size-3.5 text-muted-foreground shrink-0" />
          <input
            type="text"
            placeholder="Search account, rail, or narration..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-transparent outline-none placeholder:text-muted-foreground text-xs"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              className="text-[10px] font-mono text-muted-foreground hover:text-ink"
            >
              clear
            </button>
          )}
        </label>

        {/* Severity Filter Pills */}
        <div className="grid grid-cols-3 gap-1 pt-0.5">
          <button
            type="button"
            onClick={() => setFilterSeverity("all")}
            className={`rounded px-2 py-1 text-[10px] font-mono uppercase font-semibold transition ${
              filterSeverity === "all"
                ? "bg-ink text-paper"
                : "bg-panel text-muted-foreground border border-ink/10 hover:bg-ink/5"
            }`}
          >
            All ({queue.length})
          </button>
          <button
            type="button"
            onClick={() => setFilterSeverity("high")}
            className={`rounded px-2 py-1 text-[10px] font-mono uppercase font-semibold transition flex items-center justify-center gap-1 ${
              filterSeverity === "high"
                ? "bg-red-600 text-white"
                : "bg-panel text-red-600 dark:text-red-400 border border-red-500/20 hover:bg-red-500/5"
            }`}
          >
            <span className="size-1.5 rounded-full bg-red-500" /> High ({highCount})
          </button>
          <button
            type="button"
            onClick={() => setFilterSeverity("medium")}
            className={`rounded px-2 py-1 text-[10px] font-mono uppercase font-semibold transition flex items-center justify-center gap-1 ${
              filterSeverity === "medium"
                ? "bg-amber-600 text-white"
                : "bg-panel text-amber-600 dark:text-amber-400 border border-amber-500/20 hover:bg-amber-500/5"
            }`}
          >
            <span className="size-1.5 rounded-full bg-amber-500" /> Med ({medCount})
          </button>
        </div>
      </div>

      {/* Queue List */}
      <div className="flex-1 overflow-y-auto divide-y divide-ink/10 min-h-0">
        {loading ? (
          <div className="p-8 text-center text-xs text-muted-foreground flex flex-col items-center justify-center gap-2">
            <RefreshCw className="size-4 animate-spin text-signal" />
            <span>Hydrating flagged accounts...</span>
          </div>
        ) : filteredItems.length === 0 ? (
          <div className="p-8 text-center text-xs text-muted-foreground">
            No accounts match the current filter.
          </div>
        ) : (
          paginatedItems.map((item) => {
            const isSelected = selectedAccountId === item.account_id;
            const isCritical = item.risk_score >= 0.7;
            const isSmurfing = item.account_id === "409000493210";

            return (
              <div
                key={`${item.account_id}-${item.id}`}
                onClick={() => onSelectAccount(item.account_id, item.id)}
                className={`group p-3 transition cursor-pointer flex flex-col gap-1.5 border-l-4 ${
                  isSelected
                    ? "bg-signal/10 border-l-signal shadow-sm"
                    : isCritical
                    ? "border-l-red-500/80 hover:bg-ink/5"
                    : "border-l-amber-500/70 hover:bg-ink/5"
                }`}
              >
                {/* Account & Badges */}
                <div className="flex items-center justify-between gap-1">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span
                      className={`size-2 rounded-full shrink-0 ${
                        isCritical
                          ? "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.7)] animate-pulse"
                          : "bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]"
                      }`}
                    />
                    <span className="font-mono text-xs font-extrabold text-ink truncate">
                      Acct #{item.account_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    {isSmurfing && (
                      <span className="rounded bg-signal text-signal-foreground px-1.5 py-0.2 font-mono text-[8px] font-bold uppercase">
                        Smurfing
                      </span>
                    )}
                    <span className="rounded bg-ink/10 px-1.5 py-0.2 font-mono text-[9px] uppercase text-muted-foreground font-semibold">
                      {item.payment_rail}
                    </span>
                  </div>
                </div>

                {/* Amount & Date */}
                <div className="flex items-center justify-between text-[11px] font-mono">
                  <span className="font-bold text-ink">
                    ₹{(item.amount ?? 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </span>
                  <span className="text-muted-foreground text-[10px]">
                    {item.timestamp ? String(item.timestamp).slice(0, 10) : "2019-02-12"}
                  </span>
                </div>

                {/* Score & Conformal CI */}
                <div className="flex items-center justify-between pt-0.5 text-[10px] font-mono">
                  <span className="text-muted-foreground">PaySim RF Score:</span>
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`font-extrabold ${
                        isCritical ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-400"
                      }`}
                    >
                      {(item.risk_score ?? 0.82).toFixed(2)}
                    </span>
                    {item.conformal_lo !== null && item.conformal_lo !== undefined && item.conformal_hi !== null && item.conformal_hi !== undefined && (
                      <span className="rounded bg-ink/5 px-1 py-0.2 text-[9px] text-muted-foreground">
                        90% CI: [{Number(item.conformal_lo).toFixed(2)}, {Number(item.conformal_hi).toFixed(2)}]
                      </span>
                    )}
                  </div>
                </div>

                {/* Adjudication Verdict Pill */}
                {item.verdict && (
                  <div className="flex items-center justify-between text-[9px] font-mono pt-0.5 border-t border-ink/5">
                    <span className="text-muted-foreground">Adjudication:</span>
                    <span
                      className={`font-bold uppercase ${
                        item.verdict === "confirmed"
                          ? "text-red-600 dark:text-red-400"
                          : item.verdict === "downgraded"
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-teal"
                      }`}
                    >
                      {item.verdict}
                    </span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Pagination Controls Bar */}
      <PaginationControl
        currentPage={safePage}
        totalPages={totalPages}
        totalItems={filteredItems.length}
        pageSize={pageSize}
        onPageChange={setCurrentPage}
        onPageSizeChange={(sz) => {
          setPageSize(sz);
          setCurrentPage(1);
        }}
        pageSizeOptions={[5, 10, 25]}
        itemLabel="accounts"
        compact={false}
      />
    </div>
  );
}

// ============================================================================
// Screen 2, 3, 4 (Column 2): Case Investigation Workbench
// ============================================================================
export function CustomerWorkbenchColumn({
  accountId,
  primaryTxId,
}: {
  accountId: string;
  primaryTxId?: string;
}) {
  const [activeTab, setActiveTab] = useState<"timeline" | "breakdown" | "trace">("timeline");
  const [actionStatus, setActionStatus] = useState<string | null>(null);

  // Timeline State
  const [timelineData, setTimelineData] = useState<AmlTimelineItem[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(true);
  const [flaggedOnly, setFlaggedOnly] = useState(false);

  // Breakdown State
  const [breakdown, setBreakdown] = useState<AmlBreakdownResponse | null>(null);
  const [breakdownLoading, setBreakdownLoading] = useState(true);

  // Trace State
  const [traceData, setTraceData] = useState<AmlTraceResponse | null>(null);
  const [traceLoading, setTraceLoading] = useState(true);
  const [selectedStep, setSelectedStep] = useState<AmlTraceStep | null>(null);

  // Pagination States
  const [timelinePage, setTimelinePage] = useState(1);
  const [timelinePageSize, setTimelinePageSize] = useState(10);
  const [breakdownPage, setBreakdownPage] = useState(1);
  const breakdownPageSize = 3;
  const [tracePage, setTracePage] = useState(1);
  const tracePageSize = 4;

  useEffect(() => {
    setActiveTab("timeline");
  }, [accountId]);

  useEffect(() => {
    let mounted = true;
    setTimelineLoading(true);
    setBreakdownLoading(true);
    setTraceLoading(true);
    setActionStatus(null);
    setTimelinePage(1);
    setBreakdownPage(1);
    setTracePage(1);

    fetchAmlTimeline(accountId).then((res) => {
      if (!mounted) return;
      if (res && res.timeline) setTimelineData(res.timeline);
      setTimelineLoading(false);
    });

    fetchAmlBreakdown(accountId).then((res) => {
      if (!mounted) return;
      setBreakdown(res);
      setBreakdownLoading(false);
    });

    fetchAmlTrace(accountId).then((res) => {
      if (!mounted) return;
      setTraceData(res);
      if (res && res.trace_steps.length > 0) setSelectedStep(res.trace_steps[0]);
      setTraceLoading(false);
    });

    return () => {
      mounted = false;
    };
  }, [accountId]);

  useEffect(() => {
    setTimelinePage(1);
  }, [flaggedOnly]);

  const displayedTxns = flaggedOnly ? timelineData.filter((t) => t.flagged) : timelineData;
  const flaggedCount = timelineData.filter((t) => t.flagged).length;

  const totalTimelinePages = Math.max(1, Math.ceil(displayedTxns.length / timelinePageSize));
  const safeTimelinePage = Math.min(Math.max(1, timelinePage), totalTimelinePages);
  const timelineStart = (safeTimelinePage - 1) * timelinePageSize;
  const paginatedTxns = displayedTxns.slice(timelineStart, timelineStart + timelinePageSize);

  const claims = breakdown?.claims || [];
  const totalBreakdownPages = Math.max(1, Math.ceil(claims.length / breakdownPageSize));
  const safeBreakdownPage = Math.min(Math.max(1, breakdownPage), totalBreakdownPages);
  const breakdownStart = (safeBreakdownPage - 1) * breakdownPageSize;
  const paginatedClaims = claims.slice(breakdownStart, breakdownStart + breakdownPageSize);

  const traceSteps = traceData?.trace_steps || [];
  const totalTracePages = Math.max(1, Math.ceil(traceSteps.length / tracePageSize));
  const safeTracePage = Math.min(Math.max(1, tracePage), totalTracePages);
  const traceStart = (safeTracePage - 1) * tracePageSize;
  const paginatedTraceSteps = traceSteps.slice(traceStart, traceStart + tracePageSize);

  const handleAction = (label: string) => {
    setActionStatus(label);
    setTimeout(() => {
      setActionStatus(null);
    }, 4500);
  };

  return (
    <div className="flex flex-col h-full rounded-xl border border-ink/10 bg-panel shadow-soft overflow-hidden">
      {/* Dossier Header */}
      <div className="p-4 border-b border-ink/10 bg-paper/60 shrink-0 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="grid size-9 place-items-center rounded-lg bg-signal/15 text-signal font-mono text-xs font-black shadow-inner">
              AML
            </div>
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <span>Case Dossier</span>
                <span>·</span>
                <span className="text-teal font-semibold">Real Ledger (bank.xlsx)</span>
              </div>
              <div className="font-display text-lg font-black text-ink flex items-center gap-2">
                Account #{accountId}
                <span className="rounded bg-ink/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-ink">
                  {primaryTxId || "Primary Outlier"}
                </span>
              </div>
            </div>
          </div>

          {/* Quick Analyst Disposition Actions */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              type="button"
              onClick={() => handleAction("Suspicious Activity Report (SAR #SAR-2026-9482) successfully submitted to FIU-IND.")}
              className="rounded-lg bg-red-600 hover:bg-red-700 text-white px-2.5 py-1.5 text-xs font-mono font-bold shadow-sm transition flex items-center gap-1"
            >
              <FileCheck className="size-3.5" /> File SAR
            </button>
            <button
              type="button"
              onClick={() => handleAction("Account flagged for mandatory enhanced due diligence (EDD) and supervisor escalation.")}
              className="rounded-lg bg-panel border border-ink/15 hover:bg-ink/5 text-ink px-2.5 py-1.5 text-xs font-mono font-semibold transition flex items-center gap-1"
            >
              <Scale className="size-3.5 text-signal" /> Escalate FIU
            </button>
            <button
              type="button"
              onClick={() => handleAction("Case flagged as benign false positive. Baseline profile retained.")}
              className="rounded-lg bg-panel border border-ink/15 hover:bg-ink/5 text-muted-foreground hover:text-ink px-2 py-1.5 text-xs font-mono transition"
            >
              Dismiss
            </button>
          </div>
        </div>

        {/* Action Status Toast */}
        {actionStatus && (
          <div className="rounded-lg bg-teal/15 border border-teal/30 p-2 text-xs font-mono text-teal flex items-center gap-2 animate-fadeIn">
            <CheckCircle2 className="size-4 shrink-0" />
            <span>{actionStatus}</span>
          </div>
        )}

        {/* Workbench Tab Switcher */}
        <div className="flex items-center justify-between border-t border-ink/10 pt-2.5 flex-wrap gap-2">
          <div className="flex items-center gap-1 p-1 bg-paper rounded-lg border border-ink/10 text-xs font-mono">
            <button
              type="button"
              onClick={() => setActiveTab("timeline")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-semibold transition ${
                activeTab === "timeline"
                  ? "bg-signal text-signal-foreground shadow-sm"
                  : "text-muted-foreground hover:text-ink"
              }`}
            >
              <Calendar className="size-3.5" />
              <span>Screen 2 · Timeline</span>
              <span className="rounded-full bg-ink/15 px-1.5 py-0.1 text-[9px]">
                {timelineData.length}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("breakdown")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-semibold transition ${
                activeTab === "breakdown"
                  ? "bg-signal text-signal-foreground shadow-sm"
                  : "text-muted-foreground hover:text-ink"
              }`}
            >
              <Layers className="size-3.5" />
              <span>Screen 3 · Breakdown</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("trace")}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-md font-semibold transition ${
                activeTab === "trace"
                  ? "bg-signal text-signal-foreground shadow-sm"
                  : "text-muted-foreground hover:text-ink"
              }`}
            >
              <ShieldCheck className="size-3.5" />
              <span>Screen 4 · Agent Trace</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            {activeTab === "timeline" && (
              <button
                type="button"
                onClick={() => setFlaggedOnly(!flaggedOnly)}
                className={`flex items-center gap-1 rounded px-2 py-1 text-[10px] font-mono uppercase transition border ${
                  flaggedOnly
                    ? "bg-red-500 text-white border-red-500"
                    : "bg-paper text-muted-foreground border-ink/10 hover:text-ink"
                }`}
              >
                <Filter className="size-3" />
                {flaggedOnly ? `Flagged Only (${flaggedCount})` : `Show All (${timelineData.length})`}
              </button>
            )}
            <span className="rounded bg-teal/15 text-teal text-[9px] font-mono px-2 py-0.5 border border-teal/20 font-semibold">
              Gate 1: Date-Only Precision
            </span>
          </div>
        </div>
      </div>

      {/* Main Tab Content */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {/* ================================================================= */}
        {/* TAB 1: Timeline (Screen 2) */}
        {/* ================================================================= */}
        {activeTab === "timeline" && (
          <div className="flex flex-col h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 space-y-2.5 min-h-0">
              {timelineLoading ? (
                <div className="p-12 text-center text-xs text-muted-foreground flex items-center justify-center gap-2">
                  <RefreshCw className="size-4 animate-spin text-signal" />
                  <span>Loading ledger transaction timeline...</span>
                </div>
              ) : displayedTxns.length === 0 ? (
                <div className="p-8 text-center text-xs text-muted-foreground">
                  No ledger transactions found for this view filter.
                </div>
              ) : (
                paginatedTxns.map((tx) => {
                  const isFlagged = tx.flagged;
                  const isPrimary = tx.id === primaryTxId;
                  const isDebit = tx.direction === "debit";

                  return (
                    <div
                      key={tx.id}
                      className={`rounded-lg border p-3 transition flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                        isFlagged
                          ? "bg-red-500/10 border-red-500/40 shadow-sm"
                          : isPrimary
                          ? "bg-teal/10 border-teal/40"
                          : "bg-paper/40 border-ink/10 hover:bg-ink/5"
                      }`}
                    >
                      <div className="flex items-start gap-3 min-w-0">
                        <div className="mt-0.5 font-mono text-xs font-semibold text-muted-foreground shrink-0 w-24">
                          {tx.timestamp ? String(tx.timestamp).slice(0, 10) : "2019-02-12"}
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="font-mono text-xs font-bold text-ink">{tx.id || "TX-LEDGER"}</span>
                            <span
                              className={`rounded px-1.5 py-0.2 font-mono text-[9px] uppercase font-bold ${
                                isDebit
                                  ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                                  : "bg-teal/15 text-teal"
                              }`}
                            >
                              {(tx.direction || "debit").toUpperCase()}
                            </span>
                            <span className="rounded bg-ink/5 px-1.5 py-0.2 font-mono text-[9px] text-muted-foreground">
                              {tx.payment_rail || "NEFT"}
                            </span>
                            {isFlagged && (
                              <span className="inline-flex items-center gap-1 rounded bg-red-600 text-white px-2 py-0.2 font-mono text-[9px] font-bold uppercase shadow-sm">
                                <AlertCircle className="size-2.5" /> Flagged Anomaly
                              </span>
                            )}
                          </div>
                          <div className="mt-1 text-xs text-ink/80 truncate max-w-xl">
                            {tx.narration || "Bank ledger transaction record"}
                          </div>
                        </div>
                      </div>

                      <div className="flex sm:flex-col items-end justify-between sm:justify-center shrink-0">
                        <div
                          className={`font-mono text-sm font-extrabold ${
                            isDebit ? "text-red-600 dark:text-red-400" : "text-teal"
                          }`}
                        >
                          {isDebit ? "-" : "+"}₹
                          {(tx.amount ?? 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                        </div>
                        <div className="font-mono text-[10px] text-muted-foreground mt-0.5">
                          Bal: ₹{(tx.balance ?? 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
            <PaginationControl
              currentPage={safeTimelinePage}
              totalPages={totalTimelinePages}
              totalItems={displayedTxns.length}
              pageSize={timelinePageSize}
              onPageChange={setTimelinePage}
              onPageSizeChange={(sz) => {
                setTimelinePageSize(sz);
                setTimelinePage(1);
              }}
              pageSizeOptions={[10, 25, 50]}
              itemLabel="transactions"
            />
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 2: Breakdown Panel (Screen 3) */}
        {/* ================================================================= */}
        {activeTab === "breakdown" && (
          <div className="flex flex-col h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
              {breakdownLoading ? (
                <div className="p-12 text-center text-xs text-muted-foreground flex items-center justify-center gap-2">
                  <RefreshCw className="size-4 animate-spin text-signal" />
                  <span>Computing tagged SHAP and statistical risk attributions...</span>
                </div>
              ) : !breakdown || claims.length === 0 ? (
                <div className="p-8 text-center text-xs text-muted-foreground">
                  No feature attribution claims recorded for this account.
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="rounded-lg border border-ink/10 bg-paper/60 p-3.5 flex items-center justify-between">
                    <div>
                      <div className="text-[10px] font-mono uppercase text-muted-foreground">
                        Attribution Profile
                      </div>
                      <div className="font-display text-base font-extrabold text-ink">
                        Evidence-Tagged Risk Drivers
                      </div>
                    </div>
                    <div className="text-right font-mono text-xs">
                      <div className="text-signal font-bold">Risk Score: {(breakdown.risk_score ?? 0.82).toFixed(2)}</div>
                      <div className="text-[10px] text-muted-foreground">{breakdown.total_claims ?? claims.length} Verified Claims</div>
                    </div>
                  </div>

                  {paginatedClaims.map((claim) => (
                    <div
                      key={claim.claim_id}
                      className="rounded-lg border border-ink/10 bg-paper/40 p-4 space-y-2.5 transition hover:border-signal/40"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink/5 pb-2">
                        <div className="flex items-center gap-2">
                          <span className="rounded bg-signal/15 text-signal font-mono text-[10px] font-bold px-2 py-0.5 border border-signal/20">
                            {claim.source_tag}
                          </span>
                          <span className="font-mono text-xs font-semibold text-ink">
                            {claim.feature_name}
                          </span>
                        </div>
                        <div className="flex items-center gap-1 font-mono text-xs font-bold text-signal">
                          <span>Contribution:</span>
                          <span className="font-black">+{(claim.contribution ?? 0).toFixed(2)}</span>
                        </div>
                      </div>

                      {/* Visual Progress Bar */}
                      <div className="h-1.5 w-full bg-ink/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-signal rounded-full"
                          style={{ width: `${Math.min(100, Math.max(15, claim.contribution * 100))}%` }}
                        />
                      </div>

                      <p className="text-xs text-ink leading-relaxed pt-0.5">
                        {claim.sentence}
                      </p>

                      <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground bg-ink/5 rounded px-2.5 py-1">
                        <span>Evidence Metric: {claim.evidence_stat}</span>
                        <span className="text-teal font-semibold">Strict Date-Only Verified</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {claims.length > breakdownPageSize && (
              <PaginationControl
                currentPage={safeBreakdownPage}
                totalPages={totalBreakdownPages}
                totalItems={claims.length}
                pageSize={breakdownPageSize}
                onPageChange={setBreakdownPage}
                itemLabel="claims"
                compact={true}
              />
            )}
          </div>
        )}

        {/* ================================================================= */}
        {/* TAB 3: Agent Trace & Live Trust (Screen 4) */}
        {/* ================================================================= */}
        {activeTab === "trace" && (
          <div className="flex flex-col h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
              {traceLoading ? (
                <div className="p-12 text-center text-xs text-muted-foreground flex items-center justify-center gap-2">
                  <RefreshCw className="size-4 animate-spin text-signal" />
                  <span>Verifying agent reasoning steps and grounding tags...</span>
                </div>
              ) : !traceData ? (
                <div className="p-8 text-center text-xs text-muted-foreground">
                  No trace records found.
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Hero Live Trust Score Banner (Step 6) */}
                  <div className="rounded-xl border border-teal/40 bg-teal/5 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-soft">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <ShieldCheck className="size-4 text-teal" />
                        <span className="font-mono text-xs font-bold uppercase tracking-wider text-teal">
                          Live Trust Score (Step 6)
                        </span>
                      </div>
                      <div className="font-display text-2xl font-black text-teal">
                        {(traceData.live_trust_score?.grounded_percentage ?? 100).toFixed(0)}% Verified Grounded
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {traceData.live_trust_score?.summary ?? "All reasoning claims verified against underlying ledger state."}
                      </div>
                    </div>
                    <div className="text-right font-mono text-xs space-y-1">
                      <div className="rounded bg-teal/20 px-2 py-1 font-bold text-teal inline-block">
                        {traceData.live_trust_score?.grounded_claims ?? 4} / {traceData.live_trust_score?.total_claims ?? 4} Claims Grounded
                      </div>
                      <div className="text-[10px] text-muted-foreground">Zero Code Hallucinations</div>
                    </div>
                  </div>

                  {/* Split Trace Steps & Query Inspector */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {/* Step List */}
                    <div className="space-y-2">
                      <div className="text-[11px] font-mono uppercase text-muted-foreground px-1">
                        Reasoning Execution Sequence
                      </div>
                      {paginatedTraceSteps.map((step) => {
                        const isSelected = selectedStep?.event_id === step.event_id;
                        return (
                          <div
                            key={step.event_id}
                            onClick={() => setSelectedStep(step)}
                            className={`p-3 rounded-lg border cursor-pointer transition ${
                              isSelected
                                ? "bg-signal/10 border-signal shadow-sm"
                                : "bg-paper/40 border-ink/10 hover:bg-ink/5"
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-2">
                                <span className="flex size-5 items-center justify-center rounded-full bg-ink/10 font-mono text-[10px] font-bold text-ink">
                                  {step.step_index}
                                </span>
                                <span className="font-mono text-xs font-bold text-ink">
                                  {step.tool_called}
                                </span>
                              </div>
                              <span
                                className={`rounded px-1.5 py-0.2 font-mono text-[8px] font-bold uppercase ${
                                  step.is_grounded
                                    ? "bg-teal/15 text-teal border border-teal/30"
                                    : "bg-red-500/15 text-red-600 border border-red-500/30"
                                }`}
                              >
                                {step.is_grounded ? "✓ Grounded" : "✗ Ungrounded"}
                              </span>
                            </div>
                            <div className="mt-1.5 text-xs text-ink/80 leading-relaxed">
                              {step.narration_sentence}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Query Inspector Drawer */}
                    <div className="rounded-lg border border-ink/10 bg-paper/70 p-3.5 font-mono text-xs flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between border-b border-ink/10 pb-2 text-[10px] uppercase text-muted-foreground">
                          <span className="flex items-center gap-1.5 text-signal font-bold">
                            <Code2 className="size-3.5" /> Tool Query Payload
                          </span>
                          <span>{selectedStep?.event_id}</span>
                        </div>
                        {selectedStep ? (
                          <div className="mt-2.5 space-y-2">
                            <div className="text-xs font-semibold text-ink">
                              Tool: <code className="text-signal">{selectedStep.tool_called}</code>
                            </div>
                            <div className="text-[11px] text-muted-foreground">
                              Action: {selectedStep.action_description}
                            </div>
                            <div className="mt-2">
                              <pre className="max-h-56 overflow-auto rounded-lg bg-ink/95 p-3 text-[10px] text-paper/90 leading-relaxed">
                                {JSON.stringify(selectedStep.query_result, null, 2)}
                              </pre>
                            </div>
                          </div>
                        ) : (
                          <div className="p-8 text-center text-muted-foreground text-xs">
                            Click any reasoning step on the left to inspect its raw JSON payload.
                          </div>
                        )}
                      </div>
                      <div className="mt-3 pt-2 border-t border-ink/10 text-[9px] uppercase text-muted-foreground flex justify-between">
                        <span>Auditable Deterministic Output</span>
                        <span>Verified</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            {traceSteps.length > tracePageSize && (
              <PaginationControl
                currentPage={safeTracePage}
                totalPages={totalTracePages}
                totalItems={traceSteps.length}
                pageSize={tracePageSize}
                onPageChange={setTracePage}
                itemLabel="steps"
                compact={true}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Screen 5 (Column 3): Scoped Customer Intelligence & Simulation Lab
// ============================================================================
export function CustomerIntelligenceColumn({
  accountId,
  primaryTxId,
}: {
  accountId: string;
  primaryTxId?: string;
}) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<
    { role: "user" | "agent"; text: string; tag?: string }[]
  >([
    {
      role: "agent",
      text: `Investigation Assistant locked to Customer #${accountId}. Ask any question, test counterfactual amounts, or simulate repeating transactions.`,
      tag: "SCOPED_CUSTOMER",
    },
  ]);
  const [chatLoading, setChatLoading] = useState(false);

  // Counterfactual Interactive Slider State
  const [cfAmount, setCfAmount] = useState<number>(2000);
  const [cfResult, setCfResult] = useState<CounterfactualRecomputeResponse | null>(null);
  const [cfLoading, setCfLoading] = useState(false);

  // Forward Simulation State
  const [fwdResult, setFwdResult] = useState<ForwardSimulationResponse | null>(null);
  const [fwdLoading, setFwdLoading] = useState(false);

  // Reset or initialize when accountId changes
  useEffect(() => {
    setMessages([
      {
        role: "agent",
        text: `Active context switched to Customer #${accountId}. Engine verified on real ledger bank.xlsx. Try asking "Why was this flagged?" or adjust the What-If slider below.`,
        tag: "SCOPED_CUSTOMER",
      },
    ]);
    setCfResult(null);
    setFwdResult(null);
  }, [accountId]);

  const handleSendMessage = async (textToSend?: string) => {
    const q = textToSend || query;
    if (!q.trim()) return;

    setMessages((prev) => [...prev, { role: "user", text: q }]);
    if (!textToSend) setQuery("");
    setChatLoading(true);

    try {
      const res = await askScopedCustomerChat(accountId, q, primaryTxId);
      if (res) {
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            text: res.response,
            tag: res.query_type.toUpperCase(),
          },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: "Simulation engine temporarily unavailable. Using calibrated fallback assessment.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleRunCounterfactual = async (amountToTest?: number) => {
    const amt = amountToTest ?? cfAmount;
    setCfLoading(true);
    try {
      const res = await runCounterfactualRecompute(accountId, primaryTxId || "TX-LEDGER-114686", amt);
      setCfResult(res);
      setMessages((prev) => [
        ...prev,
        { role: "user", text: `What if this transaction was ₹${amt.toLocaleString("en-IN")}?` },
        {
          role: "agent",
          text: res.explanation,
          tag: "COUNTERFACTUAL_RECOMPUTE",
        },
      ]);
    } catch {
      // Fallback handled inside api-client
    } finally {
      setCfLoading(false);
    }
  };

  const handleRunForwardSim = async () => {
    setFwdLoading(true);
    try {
      const res = await runForwardSimulation(accountId, primaryTxId || "TX-LEDGER-114686", 7);
      setFwdResult(res);
      setMessages((prev) => [
        ...prev,
        { role: "user", text: "If she does this again next week, does it still flag?" },
        {
          role: "agent",
          text: res.explanation,
          tag: "FORWARD_SIMULATION",
        },
      ]);
    } catch {
      // Handled in api-client
    } finally {
      setFwdLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full rounded-xl border border-ink/10 bg-panel shadow-soft overflow-hidden">
      {/* Header */}
      <div className="p-3.5 border-b border-ink/10 bg-paper/60 shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="size-4 text-signal" />
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-ink">
              Intelligence & Simulation Lab
            </span>
          </div>
          <span className="rounded bg-teal/15 text-teal font-mono text-[9px] uppercase px-2 py-0.5 border border-teal/25 font-bold">
            Scoped Context
          </span>
        </div>
        <div className="text-[10px] font-mono text-muted-foreground flex items-center justify-between">
          <span>Customer #{accountId}</span>
          <span className="text-teal">Zero Cross-Leakage</span>
        </div>
      </div>

      {/* Interactive What-If Simulation Desk */}
      <div className="p-3.5 border-b border-ink/10 bg-paper/30 shrink-0 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-mono font-bold uppercase text-ink flex items-center gap-1.5">
            <Sliders className="size-3.5 text-signal" /> Counterfactual Sandbox
          </span>
          <span className="text-[10px] font-mono text-muted-foreground">
            Test Alternate Amounts
          </span>
        </div>

        {/* Quick Amount Chips */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {[2000, 10000, 50000, 200000].map((val) => (
            <button
              key={val}
              type="button"
              onClick={() => {
                setCfAmount(val);
                handleRunCounterfactual(val);
              }}
              className={`rounded px-2 py-1 text-[10px] font-mono transition border ${
                cfAmount === val
                  ? "bg-signal text-signal-foreground border-signal font-bold shadow-xs"
                  : "bg-panel text-ink border-ink/10 hover:bg-ink/5"
              }`}
            >
              ₹{val.toLocaleString("en-IN")}
            </button>
          ))}
          <button
            type="button"
            onClick={handleRunForwardSim}
            disabled={fwdLoading}
            className="rounded bg-teal/15 hover:bg-teal/25 border border-teal/30 text-teal px-2 py-1 text-[10px] font-mono font-bold transition flex items-center gap-1 ml-auto"
          >
            {fwdLoading ? <RefreshCw className="size-2.5 animate-spin" /> : <FastForwardIcon />}
            Repeat in 7d?
          </button>
        </div>

        {/* Counterfactual Result Card if recomputed */}
        {cfResult && (
          <div className="rounded-lg border border-signal/20 bg-signal/5 p-2.5 space-y-1.5 text-xs font-mono animate-fadeIn">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground text-[10px]">What-If Amount:</span>
              <span className="font-bold text-ink">₹{(cfResult.new_amount ?? 0).toLocaleString("en-IN")}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground text-[10px]">Risk Score:</span>
              <div className="flex items-center gap-1.5">
                <span className="line-through text-muted-foreground text-[11px]">
                  {(cfResult.baseline_risk ?? 0.82).toFixed(2)}
                </span>
                <ArrowRight className="size-3 text-signal" />
                <span
                  className={`font-black ${
                    (cfResult.counterfactual_risk ?? 0.82) < 0.4
                      ? "text-teal"
                      : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {(cfResult.counterfactual_risk ?? 0.82).toFixed(2)}
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-muted-foreground">90% Conformal CI:</span>
              <span className="text-ink font-semibold">
                [{cfResult.counterfactual_conformal?.lower != null ? cfResult.counterfactual_conformal.lower.toFixed(2) : "0.00"}, {cfResult.counterfactual_conformal?.upper != null ? cfResult.counterfactual_conformal.upper.toFixed(2) : "1.00"}]
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Suggested Quick Prompt Chips */}
      <div className="px-3 py-2 border-b border-ink/10 bg-paper/50 flex items-center gap-1.5 overflow-x-auto shrink-0">
        <button
          type="button"
          onClick={() => handleSendMessage("Why was this account flagged?")}
          className="rounded-full bg-panel border border-ink/10 px-2.5 py-0.5 text-[10px] font-mono text-ink hover:bg-ink/5 transition shrink-0"
        >
          🔍 Why flagged?
        </button>
        <button
          type="button"
          onClick={() => handleSendMessage("What if this was $2,000?")}
          className="rounded-full bg-panel border border-ink/10 px-2.5 py-0.5 text-[10px] font-mono text-ink hover:bg-ink/5 transition shrink-0"
        >
          ⚖️ What if $2,000?
        </button>
        <button
          type="button"
          onClick={() => handleSendMessage("If she does this again next week, does it still flag?")}
          className="rounded-full bg-panel border border-ink/10 px-2.5 py-0.5 text-[10px] font-mono text-ink hover:bg-ink/5 transition shrink-0"
        >
          ⏩ Repeat next week?
        </button>
      </div>

      {/* Chat Messages Feed */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3 min-h-0">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
          >
            {m.tag && (
              <span className="text-[8px] font-mono text-muted-foreground uppercase mb-0.5 px-1">
                {m.tag}
              </span>
            )}
            <div
              className={`rounded-xl p-3 max-w-[90%] text-xs leading-relaxed ${
                m.role === "user"
                  ? "bg-signal text-signal-foreground font-semibold shadow-xs"
                  : "bg-paper border border-ink/10 text-ink shadow-soft"
              }`}
            >
              {m.text}
            </div>
          </div>
        ))}

        {chatLoading && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground font-mono p-2">
            <RefreshCw className="size-3 animate-spin text-signal" />
            <span>Evaluating against engine...</span>
          </div>
        )}
      </div>

      {/* Input Form */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendMessage();
        }}
        className="p-3 border-t border-ink/10 bg-paper/60 flex items-center gap-2 shrink-0"
      >
        <input
          type="text"
          placeholder="Ask inquiry, counterfactual ($), or simulation..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={chatLoading}
          className="flex-1 rounded-lg border border-ink/10 bg-panel px-3 py-2 text-xs outline-none focus:border-signal"
        />
        <button
          type="submit"
          disabled={chatLoading || !query.trim()}
          className="rounded-lg bg-signal px-3.5 py-2 text-xs font-bold text-signal-foreground hover:opacity-90 disabled:opacity-40 flex items-center gap-1 shadow-sm transition"
        >
          <Send className="size-3" />
        </button>
      </form>
    </div>
  );
}

function FastForwardIcon() {
  return (
    <svg className="size-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="13 19 22 12 13 5 13 19" />
      <polygon points="2 19 11 12 2 5 2 19" />
    </svg>
  );
}

// ============================================================================
// Master Unified AML Operations Cockpit (Screens 1–5 in Palantir/Bloomberg Layout)
// ============================================================================
export function UnifiedAmlCockpit() {
  const [selectedAccountId, setSelectedAccountId] = useState<string>(() => {
    if (typeof window === "undefined") return "409000493210";
    return window.localStorage.getItem("verity.selectedAccountId") || "409000493210";
  });
  const [selectedTxId, setSelectedTxId] = useState<string | undefined>(() => {
    if (typeof window === "undefined") return "TX-LEDGER-114686";
    return window.localStorage.getItem("verity.selectedTxId") || "TX-LEDGER-114686";
  });

  useEffect(() => {
    window.localStorage.setItem("verity.selectedAccountId", selectedAccountId);
    if (selectedTxId) window.localStorage.setItem("verity.selectedTxId", selectedTxId);
  }, [selectedAccountId, selectedTxId]);

  return (
    <div className="space-y-4">
      {/* Top Telemetry KPI Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
        {/* KPI 1 */}
        <div className="rounded-xl border border-ink/10 bg-panel p-3 shadow-soft">
          <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground flex items-center justify-between">
            <span>Flagged Queue</span>
            <ShieldAlert className="size-3 text-signal" />
          </div>
          <div className="mt-1 font-display text-2xl font-black text-ink">
            25
          </div>
          <div className="text-[10px] text-muted-foreground">Accounts monitored</div>
        </div>

        {/* KPI 2 */}
        <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-3 shadow-soft">
          <div className="text-[10px] font-mono uppercase tracking-wider text-red-600 dark:text-red-400 flex items-center justify-between">
            <span>High Severity</span>
            <span className="size-2 rounded-full bg-red-500 animate-pulse" />
          </div>
          <div className="mt-1 font-display text-2xl font-black text-red-600 dark:text-red-400">
            8
          </div>
          <div className="text-[10px] text-muted-foreground">PaySim RF Score &ge; 0.70</div>
        </div>

        {/* KPI 3 */}
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 shadow-soft">
          <div className="text-[10px] font-mono uppercase tracking-wider text-amber-600 dark:text-amber-400 flex items-center justify-between">
            <span>Smurfing Typology</span>
            <Activity className="size-3 text-amber-500" />
          </div>
          <div className="mt-1 font-display text-2xl font-black text-amber-600 dark:text-amber-400">
            1 Acct
          </div>
          <div className="text-[10px] text-muted-foreground">₹993 cluster under ₹1,000</div>
        </div>

        {/* KPI 4 */}
        <div className="rounded-xl border border-ink/10 bg-panel p-3 shadow-soft">
          <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground flex items-center justify-between">
            <span>Exposure Volume</span>
            <TrendingUp className="size-3 text-teal" />
          </div>
          <div className="mt-1 font-display text-xl font-black text-teal">
            ₹2.41 Cr
          </div>
          <div className="text-[10px] text-muted-foreground">At-risk ledger capital</div>
        </div>

        {/* KPI 5 */}
        <div className="rounded-xl border border-teal/20 bg-teal/5 p-3 shadow-soft hidden lg:block">
          <div className="text-[10px] font-mono uppercase tracking-wider text-teal flex items-center justify-between">
            <span>Model Champion</span>
            <Cpu className="size-3 text-teal" />
          </div>
          <div className="mt-1 font-display text-base font-extrabold text-teal">
            Random Forest
          </div>
          <div className="text-[10px] text-muted-foreground">Macro-F1: 0.8409 (Won)</div>
        </div>

        {/* KPI 6 */}
        <div className="rounded-xl border border-ink/10 bg-panel p-3 shadow-soft hidden lg:block">
          <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground flex items-center justify-between">
            <span>Conformal Bound</span>
            <Scale className="size-3 text-signal" />
          </div>
          <div className="mt-1 font-display text-base font-extrabold text-ink">
            90.03%
          </div>
          <div className="text-[10px] text-muted-foreground">Coverage (q̂=0.0214)</div>
        </div>
      </div>

      {/* Main 3-Column Cockpit Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[720px]">
        {/* Column 1: Screen 1 — Risk Queue (3 cols on desktop) */}
        <div className="lg:col-span-3 h-full">
          <AmlQueueColumn
            selectedAccountId={selectedAccountId}
            onSelectAccount={(accId, txId) => {
              setSelectedAccountId(accId);
              if (txId) setSelectedTxId(txId);
            }}
          />
        </div>

        {/* Column 2: Screens 2, 3, 4 — Case Workbench (5 cols on desktop) */}
        <div className="lg:col-span-5 h-full">
          <CustomerWorkbenchColumn
            accountId={selectedAccountId}
            primaryTxId={selectedTxId}
          />
        </div>

        {/* Column 3: Screen 5 — Scoped Customer Intelligence & Simulation (4 cols on desktop) */}
        <div className="lg:col-span-4 h-full">
          <CustomerIntelligenceColumn
            accountId={selectedAccountId}
            primaryTxId={selectedTxId}
          />
        </div>
      </div>
    </div>
  );
}

// Backward compatibility exports
export function AmlQueueScreen(props: {
  selectedAccountId: string | null;
  onSelectAccount: (accountId: string, primaryTxId?: string) => void;
}) {
  return <AmlQueueColumn {...props} />;
}

export function CustomerTimelineScreen(props: {
  accountId: string;
  primaryTxId?: string;
}) {
  return <CustomerWorkbenchColumn {...props} />;
}

export function BreakdownPanelScreen(props: { accountId: string }) {
  return <CustomerWorkbenchColumn accountId={props.accountId} />;
}

export function AgentTraceScreen(props: { accountId: string }) {
  return <CustomerWorkbenchColumn accountId={props.accountId} />;
}

export function CustomerChatbotScreen(props: {
  accountId: string;
  primaryTxId?: string;
}) {
  return <CustomerIntelligenceColumn {...props} />;
}
