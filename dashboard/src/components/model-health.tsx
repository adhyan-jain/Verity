import {
  Activity,
  AlertTriangle,
  Award,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  Info,
  Layers,
  LineChart as LineChartIcon,
  RefreshCw,
  Scale,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";
import React, { useState } from "react";

interface DriftWindowRecord {
  window_index: number;
  window_start: string;
  window_end: string;
  n_samples: number;
  ks_statistic: number;
  p_value: number;
  drift_fired: boolean;
  alpha: number;
}

const MOCK_DRIFT_WINDOWS: DriftWindowRecord[] = [
  { window_index: 1, window_start: "2026-08-01", window_end: "2026-08-07", n_samples: 1000, ks_statistic: 0.021, p_value: 0.482, drift_fired: false, alpha: 0.05 },
  { window_index: 2, window_start: "2026-08-08", window_end: "2026-08-14", n_samples: 1000, ks_statistic: 0.026, p_value: 0.315, drift_fired: false, alpha: 0.05 },
  { window_index: 3, window_start: "2026-08-15", window_end: "2026-08-21", n_samples: 1000, ks_statistic: 0.031, p_value: 0.220, drift_fired: false, alpha: 0.05 },
  { window_index: 4, window_start: "2026-08-22", window_end: "2026-08-28", n_samples: 1000, ks_statistic: 0.029, p_value: 0.264, drift_fired: false, alpha: 0.05 },
  { window_index: 5, window_start: "2026-08-29", window_end: "2026-09-04", n_samples: 1000, ks_statistic: 0.038, p_value: 0.112, drift_fired: false, alpha: 0.05 },
  { window_index: 6, window_start: "2026-09-05", window_end: "2026-09-11", n_samples: 1000, ks_statistic: 0.041, p_value: 0.082, drift_fired: false, alpha: 0.05 },
  { window_index: 7, window_start: "2026-09-12", window_end: "2026-09-18", n_samples: 1000, ks_statistic: 0.034, p_value: 0.176, drift_fired: false, alpha: 0.05 },
];

export function ModelHealthView() {
  const [activeTab, setActiveTab] = useState<"drift" | "benchmark">("drift");
  const latestWindow = MOCK_DRIFT_WINDOWS[MOCK_DRIFT_WINDOWS.length - 1] ?? {
    window_index: 7,
    window_start: "2026-09-12",
    window_end: "2026-09-18",
    n_samples: 1000,
    ks_statistic: 0.034,
    p_value: 0.176,
    drift_fired: false,
    alpha: 0.05,
  };

  return (
    <div className="rounded-xl border border-ink/15 bg-panel p-5 font-mono shadow-soft space-y-5 animate-fadeIn">
      {/* Header & Sub-Nav */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-ink/10 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="size-4 text-signal" />
            <span className="text-xs font-bold uppercase tracking-wider text-ink">
              Model Health & Statistical Integrity
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground mt-0.5">
            Continuous drift validation (KS-test on score distribution) & PaySim engine benchmark
          </p>
        </div>

        <div className="flex items-center rounded-lg border border-ink/15 bg-paper p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab("drift")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded transition ${
              activeTab === "drift"
                ? "bg-signal text-signal-foreground font-bold shadow-xs"
                : "text-muted-foreground hover:text-ink"
            }`}
          >
            <Activity className="size-3" />
            <span>Concept Drift (KS Stat)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("benchmark")}
            className={`flex items-center gap-1.5 px-3 py-1 rounded transition ${
              activeTab === "benchmark"
                ? "bg-signal text-signal-foreground font-bold shadow-xs"
                : "text-muted-foreground hover:text-ink"
            }`}
          >
            <Award className="size-3" />
            <span>RandomForest vs LightGBM</span>
          </button>
        </div>
      </div>

      {/* ================================================================= */}
      {/* TAB 1: Concept Drift Monitor (Kolmogorov-Smirnov Test) */}
      {/* ================================================================= */}
      {activeTab === "drift" && (
        <div className="space-y-4">
          {/* Top KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="rounded-lg border border-ink/10 bg-paper/60 p-3">
              <div className="text-[10px] text-muted-foreground uppercase">Current KS Statistic</div>
              <div className="text-xl font-black text-ink mt-1">
                {latestWindow.ks_statistic.toFixed(3)}
              </div>
              <div className="text-[9px] text-muted-foreground">Max CDF deviation</div>
            </div>

            <div className="rounded-lg border border-ink/10 bg-paper/60 p-3">
              <div className="text-[10px] text-muted-foreground uppercase">KS p-value</div>
              <div className="text-xl font-black text-ink mt-1">
                {latestWindow.p_value.toFixed(3)}
              </div>
              <div className="text-[9px] text-muted-foreground">Significance threshold α=0.05</div>
            </div>

            <div className="rounded-lg border border-ink/10 bg-paper/60 p-3">
              <div className="text-[10px] text-muted-foreground uppercase">Drift Status</div>
              <div className="flex items-center gap-1.5 mt-1">
                <span className="size-2 rounded-full bg-cleared animate-pulse" />
                <span className="text-sm font-black text-cleared uppercase">
                  No Drift (Stationary)
                </span>
              </div>
              <div className="text-[9px] text-muted-foreground">p &gt; α (0.082 &gt; 0.05)</div>
            </div>

            <div className="rounded-lg border border-ink/10 bg-paper/60 p-3">
              <div className="text-[10px] text-muted-foreground uppercase">Window Size</div>
              <div className="text-xl font-black text-ink mt-1">
                1,000 samples
              </div>
              <div className="text-[9px] text-muted-foreground">Chronological sliding chunks</div>
            </div>
          </div>

          {/* Timeline of Windows */}
          <div className="rounded-lg border border-ink/10 bg-paper/50 p-4 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-ink">Score Distribution KS Trajectory</span>
              <span className="text-[10px] text-muted-foreground">7 evaluation windows monitored</span>
            </div>

            {/* Visual Micro-Bars for KS Stats across windows */}
            <div className="grid grid-cols-7 gap-2 pt-2">
              {MOCK_DRIFT_WINDOWS.map((win) => {
                const barHeightPct = Math.min(100, Math.max(10, (win.ks_statistic / 0.06) * 100));
                return (
                  <div key={win.window_index} className="flex flex-col items-center gap-1.5">
                    <span className="text-[10px] font-bold text-ink">{win.ks_statistic.toFixed(3)}</span>
                    <div className="h-20 w-full rounded bg-ink/10 relative flex items-end p-1">
                      <div
                        className="w-full rounded bg-teal/70 hover:bg-teal transition-all"
                        style={{ height: `${barHeightPct}%` }}
                      />
                    </div>
                    <span className="text-[9px] text-muted-foreground">W-{win.window_index}</span>
                  </div>
                );
              })}
            </div>

            <div className="pt-2 border-t border-ink/10 flex items-center justify-between text-[10px] text-muted-foreground">
              <span>Test: Two-sample Kolmogorov-Smirnov test on classifier confidence outputs</span>
              <span className="text-cleared font-semibold">0 False Positive Trigger Fires</span>
            </div>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* TAB 2: Model Champion Comparison (RandomForest vs LightGBM) */}
      {/* ================================================================= */}
      {activeTab === "benchmark" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Champion: Random Forest */}
            <div className="rounded-lg border-2 border-cleared/60 bg-cleared/5 p-4 space-y-3 relative">
              <div className="absolute top-3 right-3 flex items-center gap-1 rounded bg-cleared text-paper text-[9px] font-black uppercase px-2 py-0.5">
                <Award className="size-3" /> Production Champion
              </div>

              <div>
                <div className="text-xs font-bold text-cleared uppercase tracking-wider">
                  Random Forest (50 Trees)
                </div>
                <div className="text-xl font-black text-ink mt-0.5">
                  Macro-F1: 0.8409
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">Recall@K (12,487 alerts)</div>
                  <div className="text-sm font-black text-cleared">99.39%</div>
                </div>
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">Precision@K</div>
                  <div className="text-sm font-black text-ink">33.83%</div>
                </div>
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">ROC-AUC</div>
                  <div className="text-sm font-black text-ink">0.9994</div>
                </div>
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">Train Time</div>
                  <div className="text-sm font-bold text-muted-foreground">66.67s</div>
                </div>
              </div>

              <p className="text-[10.5px] text-ink/80 leading-relaxed border-t border-ink/10 pt-2">
                <strong>Why Random Forest Won:</strong> In severe AML imbalance (0.13% positive rate), catching laundered funds is paramount. Random Forest captured <strong>99.39%</strong> of all fraudulent funds in the top alert budget versus LightGBM's 28.45%.
              </p>
            </div>

            {/* Challenger: LightGBM */}
            <div className="rounded-lg border border-ink/15 bg-paper/40 p-4 space-y-3 relative opacity-85">
              <div className="absolute top-3 right-3 flex items-center gap-1 rounded bg-ink/10 text-muted-foreground text-[9px] font-semibold uppercase px-2 py-0.5">
                Challenger
              </div>

              <div>
                <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider">
                  LightGBM (Gradient Boosted)
                </div>
                <div className="text-xl font-black text-ink mt-0.5">
                  Macro-F1: 0.5253
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">Recall@K (12,487 alerts)</div>
                  <div className="text-sm font-black text-signal">28.45%</div>
                </div>
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">Precision@K</div>
                  <div className="text-sm font-black text-ink">9.68%</div>
                </div>
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">ROC-AUC</div>
                  <div className="text-sm font-black text-ink">0.8659</div>
                </div>
                <div className="bg-paper/80 p-2 rounded border border-ink/10">
                  <div className="text-[10px] text-muted-foreground">Train Time</div>
                  <div className="text-sm font-bold text-ink">19.61s (3.4x faster)</div>
                </div>
              </div>

              <p className="text-[10.5px] text-muted-foreground leading-relaxed border-t border-ink/10 pt-2">
                While LightGBM trains 3.4× faster, its decision tree depth was overwhelmed by extreme class skew, missing over 70% of high-severity laundering cases in the test horizon.
              </p>
            </div>
          </div>

          <div className="rounded bg-paper/60 p-2.5 border border-ink/10 text-[10px] text-muted-foreground flex items-center justify-between">
            <span>PaySim Benchmark: 5,113,884 train rows · 1,248,736 test rows · Split step: 355</span>
            <span className="text-teal font-semibold">Offline Artifact: paysim_benchmark.json</span>
          </div>
        </div>
      )}
    </div>
  );
}
