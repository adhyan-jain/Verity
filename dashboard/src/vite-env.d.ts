/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_AGENT_API_URL?: string;
  readonly VITE_FRAUD_API_URL?: string;
  readonly VITE_LEDGER_API_URL?: string;
  readonly VITE_TYPOLOGY_API_URL?: string;
  readonly VITE_FRAUD_API_KEY?: string;
  readonly VITE_LEDGER_API_KEY?: string;
  readonly VITE_TYPOLOGY_API_KEY?: string;
  readonly VITE_AGENT_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
