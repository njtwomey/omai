import { defineConfig } from 'vitest/config'

const onCi = process.env['CI'] !== undefined

// Test workers: locally half the cores, so several checkouts (agents' worktrees) can test at once without saturating
// the machine; on CI, where the runner is ours alone, all of them. Set MDML_TEST_WORKERS to a count or a percentage to
// override either.
const workers = process.env['MDML_TEST_WORKERS']
const maxWorkers = workers === undefined ? (onCi ? '100%' : '50%') : workers.endsWith('%') ? workers : Number(workers)

// The CI runner is several times slower than a developer machine and runs the whole suite at once: a test that takes
// a second or two locally can pass 5 s there. Locally the strict default stays, so a slow test still shows up.
const testTimeout = onCi ? 30_000 : 5_000

// Each test file normally gets a fresh module graph, which re-imports and re-transforms the engine every time: about
// half the worker time of a full run (measured: the suite took 716 s with isolation and 407 s without, at two workers
// and CI's timeout, and every test passed both ways). CI shares the graph between the files of a worker; locally files
// stay isolated, so a test that leaks state between files still shows up when running a few of them.
const isolate = !onCi

// Only the engine's tests gain from that (a very large module graph imported by 1,500 files). The tests of render, meta
// and the apps are few, drive React with stand-in globals (window, requestAnimationFrame, performance.now) and keep
// module-level state (the stored session preferences, the last player to play), so they keep a fresh module graph per
// file on CI as well: sharing one made files fail depending on which ran first in a worker.
const keepIsolation = { isolate: true }

// One Vitest run for the workspace. Packages resolve each other's source through the `source` export condition, so
// tests never need a build.
export default defineConfig({
  resolve: { conditions: ['source'] },
  ssr: { resolve: { conditions: ['source'], externalConditions: ['source'] } },
  test: {
    maxWorkers,
    testTimeout,
    isolate,
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
          name: 'scripts',
          root: 'scripts',
          include: ['*.test.ts'],
          environment: 'node',
        },
      },
      {
        extends: true,
        test: {
          name: 'render',
          ...keepIsolation,
          root: 'packages/render',
          include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
          environment: 'node',
        },
      },
      {
        extends: true,
        test: {
          name: 'meta',
          ...keepIsolation,
          root: 'packages/meta',
          include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
          environment: 'node',
        },
      },
      {
        extends: true,
        test: {
          name: 'pocs',
          ...keepIsolation,
          root: 'apps/pocs',
          include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
          environment: 'node',
        },
      },
      {
        extends: true,
        test: {
          name: 'site',
          ...keepIsolation,
          root: 'apps/site',
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
