import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { docsCatalogue } from './plugins/docs-catalogue.ts'
import { hubIndex } from './plugins/hub-index.ts'
import { recipeMeta } from './plugins/recipe-meta.ts'

/** The mdml-js site: one app with three sections (gallery, hub, docs) on a shared shell. */
export default defineConfig({
  root: import.meta.dirname,
  // The path the site is served under: `/` locally, a sub-path when deployed (set by the deploy workflow).
  base: process.env['MDML_SITE_BASE'] ?? '/',
  plugins: [recipeMeta(), react(), tailwindcss(), docsCatalogue(), hubIndex()],
  // Workspace packages resolve to their source, as in the tests.
  resolve: { conditions: ['source'] },
  // A single-page app with path URLs (/gallery/..., /hub/..., /docs/...).
  appType: 'spa',
  // Module workers, so the hub's worker loads only the entry it runs (its entry modules are separate chunks).
  worker: { format: 'es' },
  server: { port: 5300 },
  preview: { port: 5300 },
})
