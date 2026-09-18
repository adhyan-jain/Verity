/**
 * Minimal server-side error capture used by `server.ts` to recover the real
 * error object when TanStack Start/h3 swallows an in-handler throw into a
 * generic `{"unhandled":true,"message":"HTTPError"}` 500 response. Without
 * this, that swallowed-error path would only ever log a synthetic message.
 */
let lastCapturedError: unknown = null;

export function captureError(error: unknown): void {
  lastCapturedError = error;
}

export function consumeLastCapturedError(): unknown {
  const error = lastCapturedError;
  lastCapturedError = null;
  return error;
}

if (typeof process !== "undefined" && typeof process.on === "function") {
  process.on("uncaughtException", captureError);
  process.on("unhandledRejection", captureError);
}
