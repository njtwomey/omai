import { defineConfig } from 'vitest/config'

// One Vitest run for the workspace. Packages resolve each other's source through the `source` export condition, so
// tests never need a build.
export default defineConfig({
  resolve: { conditions: ['source'] },
  ssr: { resolve: { conditions: ['source'], externalConditions: ['source'] } },
  test: {
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
    ],
    coverage: {
      provider: 'v8',
      include: ['packages/*/src/**/*.ts'],
      exclude: ['**/*.test.ts', '**/*.test-d.ts', '**/index.ts', '**/*.data.ts', '**/testing/**'],
      thresholds: { lines: 85, functions: 85, branches: 75, statements: 85 },
    },
  },
})
