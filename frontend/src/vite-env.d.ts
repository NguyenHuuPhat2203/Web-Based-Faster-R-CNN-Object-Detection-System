/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  /** `v1` | `v2`. Default v2 when unset. Set `v1` to hit bare `/predict` endpoints. */
  readonly VITE_API_VERSION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
