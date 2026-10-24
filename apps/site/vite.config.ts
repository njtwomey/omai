import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { docsCatalogue } from './plugins/docs-catalogue.ts'

/** The whitebox-js site: one app with three sections (gallery, hub, docs) on a shared shell. */
export default defineConfig({
  root: import.meta.dirname,
  // The path the site is served under: `/` locally, a sub-path when deployed (set by the deploy workflow).
  base: process.env['WHITEBOX_SITE_BASE'] ?? '/',
  plugins: [react(), tailwindcss(), docsCatalogue()],
  // Workspace packages resolve to their source, as in the tests.
  resolve: { conditions: ['source'] },
  // A single-page app with path URLs (/gallery/..., /hub/..., /docs/...).
  appType: 'spa',
  server: { port: 5300 },
  preview: { port: 5300 },
})
