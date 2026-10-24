import { defineConfig } from 'vitest/config'

// Test workers: half the cores by default, so several checkouts (agents' worktrees) can test at once without
// saturating the machine; set WHITEBOX_TEST_WORKERS to a count or a percentage to override.
const workers = process.env['WHITEBOX_TEST_WORKERS']
const maxWorkers = workers === undefined ? '50%' : workers.endsWith('%') ? workers : Number(workers)

// One Vitest run for the workspace. Packages resolve each other's source through the `source` export condition, so
// tests never need a build.
export default defineConfig({
  resolve: { conditions: ['source'] },
  ssr: { resolve: { conditions: ['source'], externalConditions: ['source'] } },
  test: {
    maxWorkers,
    projects: [
      {
        extends: true,
        test: {
          name: 'engine',
          root: 'packages/engine',
          include: ['src/**/*.test.ts'],
          environment: 'node',
          typecheck: { enabled: true, include: ['src/**/*.test-d.ts'], tsconfig: 'tsconfig.json' },
        },
      },
      {
        extends: true,
        test: {
          name: 'render',
          root: 'packages/render',
          include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
          environment: 'node',
        },
      },
      {
        extends: true,
        test: {
          name: 'meta',
          root: 'packages/meta',
          include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
          environment: 'node',
        },
      },
      {
        extends: true,
        test: {
          name: 'hub',
          root: 'packages/hub',
          include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
          environment: 'node',
        },
      },
    ],
    coverage: {
      provider: 'v8',
      include: ['packages/*/src/**/*.ts'],
      exclude: ['**/*.test.ts', '**/*.test-d.ts', '**/index.ts', '**/*.data.ts', '**/testing/**'],
      thresholds: { lines: 85, functions: 85, branches: 75, statements: 85 },
    },
  },
})
