/**
 * Client-side error reporting hook for the root route's error boundary.
 * The original Lovable.dev platform posts these to a hosted dashboard;
 * outside that platform there is no such endpoint, so this logs locally
 * (loudly, not silently) instead of failing or no-op'ing invisibly.
 */
export function reportLovableError(error: Error, context?: Record<string, unknown>): void {
  console.error("[verity] unhandled UI error", error, context);
}
