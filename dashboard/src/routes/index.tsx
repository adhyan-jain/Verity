import { createFileRoute } from "@tanstack/react-router";
import { VerityWorkspace } from "../components/verity-workspace";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Verity — Financial Crime Analyst Workspace" },
      { name: "description", content: "Investigate card fraud, ledger anomalies, and transaction networks in one trace-grounded analyst workspace." },
      { property: "og:title", content: "Verity — Financial Crime Analyst Workspace" },
      { property: "og:description", content: "A unified, trace-grounded workspace for financial-crime investigation." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return <VerityWorkspace />;
}
