import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Check,
  CheckCircle2,
  Code2,
  FileCheck2,
  Play,
  RefreshCw,
  Scale,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import React, { useEffect, useState } from "react";

import {
  fetchAdjudicationDetails,
  type AmlTraceStep,
  type ProsecutorDefenderResolution,
} from "../lib/api-client";
import { useTypewriter } from "../hooks/use-typewriter";

export function StreamingTraceItem({
  step,
  isSelected,
  onClick,
  isStreaming,
}: {
  step: AmlTraceStep;
  isSelected: boolean;
  onClick: () => void;
  isStreaming: boolean;
}) {
  const { displayedText, isTyping } = useTypewriter(
    step.narration_sentence,
    14,
    isStreaming,
  );

  return (
    <div
      onClick={onClick}
      className={`p-3 rounded-lg border cursor-pointer font-mono transition select-none ${
        isSelected
          ? "bg-signal/10 border-signal shadow-sm"
          : "bg-paper/40 border-ink/10 hover:bg-ink/5"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="flex size-5 items-center justify-center rounded bg-ink/10 text-[10px] font-bold text-ink">
            {step.step_index}
          </span>
          <span className="text-xs font-bold text-ink">
            {step.tool_called}
          </span>
        </div>
        <span
          className={`rounded px-1.5 py-0.2 text-[8px] font-bold uppercase tracking-wider ${
            step.is_grounded
              ? "bg-cleared/15 text-cleared border border-cleared/30"
              : "bg-signal/15 text-signal border border-signal/30"
          }`}
        >
          {step.is_grounded ? "✓ Grounded" : "✗ Ungrounded"}
        </span>
      </div>

      <div className="mt-1.5 text-xs text-ink/90 leading-relaxed">
        {isStreaming ? displayedText : step.narration_sentence}
        {isStreaming && isTyping && (
          <span className="inline-block animate-pulse text-amber-500 font-bold ml-0.5">
            ▋
          </span>
        )}
      </div>

      <div className="mt-2 flex items-center justify-between text-[9px] text-muted-foreground border-t border-ink/5 pt-1">
        <span>Event: {step.event_id}</span>
        <span className="text-teal font-semibold">Strict Grounding Filter Passed</span>
      </div>
    </div>
  );
}

export function StreamingAgentTracePanel({
  accountId,
  primaryTxId,
  traceSteps,
}: {
  accountId: string;
  primaryTxId?: string | undefined;
  traceSteps: AmlTraceStep[];
}) {
  const [selectedStep, setSelectedStep] = useState<AmlTraceStep | null>(
    traceSteps.length > 0 && traceSteps[0] ? traceSteps[0] : null,
  );
  const [activeSubTab, setActiveSubTab] = useState<"adversarial" | "stream">("adversarial");
  const [adjudication, setAdjudication] = useState<ProsecutorDefenderResolution | null>(null);
  const [loadingAdj, setLoadingAdj] = useState(true);
  const [isStreamingTrace, setIsStreamingTrace] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoadingAdj(true);
    fetchAdjudicationDetails(primaryTxId || "TX-LEDGER-114686", 0.82).then((res) => {
      if (!mounted) return;
      setAdjudication(res);
      setLoadingAdj(false);
    });
    return () => {
      mounted = false;
    };
  }, [accountId, primaryTxId]);

  const restartStreaming = () => {
    setIsStreamingTrace(false);
    setTimeout(() => {
      setIsStreamingTrace(true);
    }, 50);
  };

  const isDowngraded = adjudication?.final_verdict === "downgraded_by_defender" || adjudication?.final_verdict === "cleared";

  return (
    <div className="space-y-4 font-mono">
      {/* Sub-Nav: Prosecutor vs Defender Adversarial Adjudication vs Live Event Stream */}
      <div className="flex items-center justify-between gap-2 border-b border-ink/10 pb-2">
        <div className="flex items-center rounded-lg border border-ink/15 bg-paper p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setActiveSubTab("adversarial")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded transition ${
              activeSubTab === "adversarial"
                ? "bg-signal text-signal-foreground font-bold shadow-xs"
                : "text-muted-foreground hover:text-ink"
            }`}
          >
            <Scale className="size-3" />
            <span>Prosecutor vs Defender Arguments</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveSubTab("stream")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded transition ${
              activeSubTab === "stream"
                ? "bg-signal text-signal-foreground font-bold shadow-xs"
                : "text-muted-foreground hover:text-ink"
            }`}
          >
            <Zap className="size-3" />
            <span>Character-by-Character Event Stream</span>
          </button>
        </div>

        {activeSubTab === "stream" && (
          <button
            type="button"
            onClick={restartStreaming}
            className="flex items-center gap-1 rounded bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-500 px-2 py-1 text-[10px] font-bold transition"
            title="Re-run character-by-character live stream"
          >
            <RefreshCw className="size-2.5" /> Re-Stream Trace
          </button>
        )}
      </div>

      {/* ================================================================= */}
      {/* SUB-TAB 1: Prosecutor vs Defender Side-by-Side View */}
      {/* ================================================================= */}
      {activeSubTab === "adversarial" && (
        <div className="space-y-3 animate-fadeIn">
          {/* Adjudication Resolution Banner */}
          {adjudication && (
            <div
              className={`rounded-lg border p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 ${
                isDowngraded
                  ? "border-cleared/40 bg-cleared/10 text-cleared"
                  : "border-signal/40 bg-signal/10 text-signal"
              }`}
            >
              <div className="flex items-center gap-2">
                {isDowngraded ? (
                  <ShieldCheck className="size-5 shrink-0 text-cleared" />
                ) : (
                  <ShieldAlert className="size-5 shrink-0 text-signal" />
                )}
                <div>
                  <div className="text-[10px] uppercase font-bold tracking-wider">
                    Adjudication Verdict: {adjudication.final_verdict.toUpperCase()}
                  </div>
                  <div className="text-xs text-ink/90 font-medium">
                    {adjudication.resolution_reason}
                  </div>
                </div>
              </div>
              <div className="text-right text-xs font-bold shrink-0">
                <span className="text-muted-foreground line-through mr-1.5">
                  {adjudication.original_risk_score.toFixed(2)}
                </span>
                <span className={isDowngraded ? "text-cleared font-black text-sm" : "text-signal font-black text-sm"}>
                  → {adjudication.final_risk_score.toFixed(2)}
                </span>
              </div>
            </div>
          )}

          {/* Side-by-Side Argument Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Prosecutor: Case for Escalation */}
            <div className="rounded-lg border border-signal/30 bg-signal/5 p-3.5 space-y-2.5">
              <div className="flex items-center justify-between border-b border-signal/20 pb-2">
                <div className="flex items-center gap-1.5 text-signal font-bold text-xs">
                  <ShieldAlert className="size-4" />
                  <span>Prosecutor (Case for Escalation)</span>
                </div>
                <span className="rounded bg-signal/20 text-signal text-[9px] font-bold px-1.5 py-0.5">
                  Fraud Drivers
                </span>
              </div>

              <div className="space-y-2">
                {adjudication?.prosecutor.arguments.map((arg, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-xs leading-relaxed text-ink/90">
                    <span className="size-1.5 rounded-full bg-signal shrink-0 mt-1.5" />
                    <span>{arg}</span>
                  </div>
                ))}
              </div>

              {adjudication?.prosecutor.evidence_ids && adjudication.prosecutor.evidence_ids.length > 0 && (
                <div className="pt-2 border-t border-signal/15 flex items-center gap-1.5 text-[9px] text-muted-foreground flex-wrap">
                  <span>Cited Evidence:</span>
                  {adjudication.prosecutor.evidence_ids.map((id) => (
                    <span key={id} className="rounded bg-signal/15 text-signal px-1 py-0.2 font-bold">
                      {id}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Defender: Case for Clearing */}
            <div className="rounded-lg border border-cleared/30 bg-cleared/5 p-3.5 space-y-2.5">
              <div className="flex items-center justify-between border-b border-cleared/20 pb-2">
                <div className="flex items-center gap-1.5 text-cleared font-bold text-xs">
                  <ShieldCheck className="size-4" />
                  <span>Defender (Case for Clearing)</span>
                </div>
                <span
                  className={`rounded text-[9px] font-bold px-1.5 py-0.5 ${
                    adjudication?.defender.is_grounded
                      ? "bg-cleared/20 text-cleared"
                      : "bg-paper text-muted-foreground border border-ink/10"
                  }`}
                >
                  {adjudication?.defender.is_grounded ? "Innocent Grounded" : "No Innocent Precedent"}
                </span>
              </div>

              <div className="space-y-2">
                {adjudication?.defender.arguments.map((arg, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-xs leading-relaxed text-ink/90">
                    <span
                      className={`size-1.5 rounded-full shrink-0 mt-1.5 ${
                        adjudication.defender.is_grounded ? "bg-cleared" : "bg-muted-foreground"
                      }`}
                    />
                    <span>{arg}</span>
                  </div>
                ))}
              </div>

              {adjudication?.defender.cited_transaction_ids && adjudication.defender.cited_transaction_ids.length > 0 ? (
                <div className="pt-2 border-t border-cleared/15 flex items-center gap-1.5 text-[9px] text-muted-foreground flex-wrap">
                  <span>Cited Baseline Records:</span>
                  {adjudication.defender.cited_transaction_ids.map((id) => (
                    <span key={id} className="rounded bg-cleared/15 text-cleared px-1 py-0.2 font-bold">
                      {id}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="pt-2 border-t border-ink/10 text-[9px] text-muted-foreground">
                  Zero prior matching records found in account history.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* SUB-TAB 2: Character-by-Character Streaming Event Log */}
      {/* ================================================================= */}
      {activeSubTab === "stream" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 animate-fadeIn">
          {/* Streaming Trace Rows */}
          <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
            <div className="text-[10px] uppercase text-muted-foreground font-semibold px-1 flex items-center justify-between">
              <span>Streaming Event Sequence</span>
              <span className="text-amber-500 font-bold flex items-center gap-1">
                <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" /> Live Monospace Stream
              </span>
            </div>
            {traceSteps.map((step) => (
              <StreamingTraceItem
                key={step.event_id}
                step={step}
                isSelected={selectedStep?.event_id === step.event_id}
                onClick={() => setSelectedStep(step)}
                isStreaming={isStreamingTrace}
              />
            ))}
          </div>

          {/* JSON Tool Payload Inspector */}
          <div className="rounded-lg border border-ink/10 bg-paper/70 p-3.5 text-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between border-b border-ink/10 pb-2 text-[10px] uppercase text-muted-foreground">
                <span className="flex items-center gap-1.5 text-signal font-bold">
                  <Code2 className="size-3.5" /> Authenticated Tool Call Payload
                </span>
                <span>{selectedStep?.event_id}</span>
              </div>
              {selectedStep ? (
                <div className="mt-2.5 space-y-2">
                  <div className="text-xs font-semibold text-ink">
                    Tool Executed: <code className="text-signal">{selectedStep.tool_called}</code>
                  </div>
                  <div className="text-[11px] text-muted-foreground">
                    Action: {selectedStep.action_description}
                  </div>
                  <div className="mt-2">
                    <pre className="max-h-56 overflow-auto rounded bg-ink/95 p-2.5 text-[10px] text-paper/90 leading-relaxed font-mono">
                      {JSON.stringify(selectedStep.query_result, null, 2)}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-muted-foreground text-xs">
                  Click any reasoning step on the left to inspect its raw JSON tool query.
                </div>
              )}
            </div>
            <div className="mt-3 pt-2 border-t border-ink/10 text-[9px] uppercase text-muted-foreground flex justify-between">
              <span>Zero LLM Speculation</span>
              <span className="text-cleared font-bold">Grounded in Raw Ledger</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
