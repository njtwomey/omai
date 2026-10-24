import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/** The whitebox-hub site: every hub entry as a page, generated from the catalogue. */
export default defineConfig({
  root: import.meta.dirname,
  // The path the app is served under: \`/\` locally, a sub-path when deployed (set by the deploy workflow).
  base: process.env['WHITEBOX_APP_BASE'] ?? '/',
  plugins: [react(), tailwindcss()],
  // Workspace packages resolve to their source, as in the tests.
  resolve: { conditions: ['source'] },
  appType: 'spa',
  server: { port: 5302 },
  preview: { port: 5302 },
})
