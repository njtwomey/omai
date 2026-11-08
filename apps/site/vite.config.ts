import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { docsCatalogue } from './plugins/docs-catalogue.ts'
import { recipeMeta } from './plugins/recipe-meta.ts'

/** The omai site: one app with two sections (gallery, docs) on a shared shell. */
export default defineConfig({
  root: import.meta.dirname,
  // The path the site is served under: `/` locally, a sub-path when deployed (set by the deploy workflow).
  base: process.env['OMAI_SITE_BASE'] ?? '/',
  plugins: [recipeMeta(), react(), tailwindcss(), docsCatalogue()],
  // Workspace packages resolve to their source, as in the tests.
  // The front page runs programs from `examples/`, which sit outside the site: one copy of React for all of them.
  resolve: { conditions: ['source'], dedupe: ['react', 'react-dom'] },
  // A single-page app with path URLs (/gallery/..., /docs/...).
  appType: 'spa',
  // Module workers, so a worker loads only the modules it runs (as separate chunks).
  worker: { format: 'es' },
  server: { port: 5300 },
  preview: { port: 5300 },
})
