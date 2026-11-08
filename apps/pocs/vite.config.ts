import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/** The proofs of concept app: its own pages on the site's shell (imported from apps/site/src/shell). */
export default defineConfig({
  root: import.meta.dirname,
  base: process.env['OMAI_SITE_BASE'] ?? '/',
  plugins: [react(), tailwindcss()],
  // Workspace packages resolve to their source, as in the tests.
  resolve: { conditions: ['source'] },
  appType: 'spa',
  worker: { format: 'es' },
  // The shell lives in the site's folder, one level up.
  server: { port: 5310, fs: { allow: ['../..'] } },
  preview: { port: 5310 },
})
