import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/** The showcase app: its own pages, on the colours and code block of the site's shell (imported from apps/site/src/shell). */
export default defineConfig({
  root: import.meta.dirname,
  base: process.env['OMAI_SITE_BASE'] ?? '/',
  plugins: [react(), tailwindcss()],
  // Workspace packages resolve to their source, as in the tests.
  resolve: { conditions: ['source'] },
  appType: 'spa',
  // A story's worker is bundled apart from the page, so the engine it uses never loads in the page.
  worker: { format: 'es' },
  // The shell lives in the site's folder, one level up.
  server: { port: Number(process.env['OMAI_SHOWCASE_PORT'] ?? 5320), fs: { allow: ['../..'] } },
  preview: { port: 5320 },
})
