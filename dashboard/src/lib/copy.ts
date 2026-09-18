/**
 * Shared copy-formatting helpers, centralizing wording decisions that must
 * stay consistent across every screen rather than being hand-typed per
 * component.
 *
 * BLOCKING DEPENDENCY (v2.1 §6): whether bank.xlsx carries real time-of-day
 * (vs. date-only) is unresolved. Until a teammate audits and confirms it,
 * no real_ledger-tier copy may claim a specific clock time ("flagged at
 * 3am") — it must degrade to day-level language ("flagged on an unusual
 * day for this account").
 *
 * Card tier is NOT affected: creditcard.csv's `Time` column is a real,
 * already-verified seconds-since-day-start field (engines/fraud/explain.py
 * already converts it to HH:MM for the SHAP narrative).
 *
 * Synthetic-network tier is NOT affected either: its timestamps come from
 * `data/synthetic/generate_network.py`, a generator we control end to end
 * (verified: real minute-level precision, e.g. "2026-09-10T16:14:00Z") —
 * not derived from bank.xlsx, so none of the §6 ambiguity applies to it.
 */

export type Tier = "real_card" | "real_ledger" | "synthetic_network";

/**
 * Flip to true only once a teammate confirms bank.xlsx has real time-of-day
 * granularity (not just a date). That's the ONLY change needed to upgrade
 * every real_ledger timing claim in the app to real clock times — every
 * call site below already routes through this flag.
 */
export const LEDGER_TIME_OF_DAY_CONFIRMED = false;

function hasConfirmedTimeOfDay(tier: Tier): boolean {
  return tier !== "real_ledger" || LEDGER_TIME_OF_DAY_CONFIRMED;
}

function formatClockTime(isoTimestamp: string): string {
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) return "an unusual time";
  let hours = date.getUTCHours();
  const minutes = date.getUTCMinutes();
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12 || 12;
  return `${hours}:${String(minutes).padStart(2, "0")} ${ampm}`;
}

/**
 * Renders a timing claim for a single event. Card and synthetic tiers (or
 * real_ledger once confirmed) get a real clock time; real_ledger degrades
 * to day-level language rather than fabricating precision the data may
 * not support.
 */
export function formatTimingClaim(tier: Tier, isoTimestamp: string): string {
  if (hasConfirmedTimeOfDay(tier)) {
    return `at ${formatClockTime(isoTimestamp)}`;
  }
  return "on an unusual day for this account";
}

/**
 * Same idea for a short inline label (timeline points, trace meta rows)
 * rather than a full sentence fragment.
 */
export function formatTimingLabel(tier: Tier, isoTimestamp: string): string {
  if (hasConfirmedTimeOfDay(tier)) {
    return formatClockTime(isoTimestamp);
  }
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) return "unusual day";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * Renders a claim about a *range* of events (structuring flags spanning
 * multiple transactions) — never a single clock time, since the claim is
 * inherently about a pattern over a window, not one moment.
 */
export function formatRangeTimingClaim(tier: Tier, count: number): string {
  if (tier === "real_card" || tier === "synthetic_network") {
    return `across ${count} transactions in quick succession`;
  }
  return `across ${count} transactions over an unusual window for this account`;
}
