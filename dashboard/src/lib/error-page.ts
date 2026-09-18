/**
 * Renders a minimal, dependency-free HTML error page for catastrophic SSR
 * failures (before React has a chance to mount an error boundary).
 * Used by `server.ts` and `start.ts`.
 */
export function renderErrorPage(): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Verity — Something went wrong</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font-family: system-ui, sans-serif; background: #f4f2ee; color: #14110f; display: grid; place-items: center; min-height: 100vh; margin: 0; }
      main { max-width: 32rem; padding: 2rem; text-align: center; }
      h1 { font-size: 1.25rem; margin-bottom: 0.5rem; }
      p { color: #6b6560; font-size: 0.875rem; }
    </style>
  </head>
  <body>
    <main>
      <h1>Verity hit an unexpected server error</h1>
      <p>The request failed before the analyst workspace could render. Reload the page, or check the server logs for details.</p>
    </main>
  </body>
</html>`;
}
