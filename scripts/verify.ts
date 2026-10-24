/**
 * Verify a change: the checks a change actually needs, scoped to what it touches (docs/structure.md section 10).
 *
 *   node scripts/verify.ts [--base <ref>] [--dry-run]
 *
 * The change is everything that differs from the merge base with `--base` (default `main`), plus staged, unstaged and
 * untracked files. Then:
 * - Always, because they take seconds over the whole repo: barrels, layers, Biome, banned globals, ruff and tsc.
 * - Tests: `vitest related` on the changed TypeScript, which runs every test that imports a changed file directly or
 *   transitively (so a change to a shared module tests its dependents), plus the golden tests that load a changed
 *   fixture.
 * - Python: pyright on the changed oracle files, and a fixture check of just the changed case modules.
 * - Everything, as `make check` and `make fixtures-check` do, when shared infrastructure changes: dependencies, tool
 *   configuration, scripts, or the oracle's shared helpers.
 */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, relative } from 'node:path'

const ROOT = join(import.meta.dirname, '..')
const FIXTURES = 'packages/engine/fixtures/'
const CASES = 'oracle/cases/'
const ORACLE_GROUPS = [
  '--group',
  'oracle-core',
  '--group',
  'oracle-torch',
  '--group',
  'oracle-prob',
  '--group',
  'oracle-rl',
]

// Files whose change can affect every test or fixture.
const GLOBAL_TS = [
  /^package\.json$/,
  /^pnpm-lock\.yaml$/,
  /^pnpm-workspace\.yaml$/,
  /^vitest\.config\.ts$/,
  /(^|\/)tsconfig[^/]*\.json$/,
  /^biome\.jsonc?$/,
  /^packages\/[^/]+\/package\.json$/,
  /^scripts\//,
  /^Makefile$/,
]
const GLOBAL_PY = [/^pyproject\.toml$/, /^uv\.lock$/, /^oracle\/whitebox_oracle\//]

function git(...args: string[]): string[] {
  const r = spawnSync('git', args, { cwd: ROOT, encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`git ${args.join(' ')}: ${r.stderr}`)
  return r.stdout.split('\n').filter((l) => l !== '')
}

function changedFiles(base: string): string[] {
  const mergeBase = git('merge-base', base, 'HEAD')[0] ?? base
  const all = new Set([
    ...git('diff', '--name-only', `${mergeBase}...HEAD`),
    ...git('diff', '--name-only', 'HEAD'),
    ...git('ls-files', '--others', '--exclude-standard'),
  ])
  return [...all].filter((f) => !/(^|\/)node_modules(\/|$)/.test(f)).sort()
}

function walkTests(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name.startsWith('.')) continue
    const path = join(dir, name)
    if (statSync(path).isDirectory()) walkTests(path, out)
    else if (/\.test\.tsx?$/.test(name)) out.push(path)
  }
  return out
}

// Golden tests name their fixtures as strings (`loadFixture('stats/tests/means')`), so the import graph cannot see them.
function testsLoadingFixtures(fixtures: readonly string[]): string[] {
  if (fixtures.length === 0) return []
  const keys = fixtures.map((f) => f.slice(FIXTURES.length).replace(/\.json$/, ''))
  const out: string[] = []
  for (const file of walkTests(join(ROOT, 'packages'))) {
    const text = readFileSync(file, 'utf8')
    if (keys.some((k) => text.includes(`'${k}'`) || text.includes(`"${k}"`) || text.includes(`\`${k}`)))
      out.push(relative(ROOT, file))
  }
  return out
}

interface Step {
  readonly label: string
  readonly command: readonly string[]
}

function plan(files: readonly string[]): Step[] {
  const has = (patterns: readonly RegExp[]): boolean => files.some((f) => patterns.some((p) => p.test(f)))
  const globalTs = has(GLOBAL_TS)
  const globalPy = has(GLOBAL_PY)
  const steps: Step[] = [
    { label: 'barrels', command: ['node', 'scripts/barrels.ts', '--check'] },
    { label: 'layers', command: ['node', 'scripts/layers.ts'] },
    { label: 'biome', command: ['pnpm', 'exec', 'biome', 'check', '.'] },
    { label: 'banned globals', command: ['node', 'scripts/banned.ts'] },
    { label: 'ruff', command: ['uv', 'run', '--group', 'dev', 'ruff', 'check', 'oracle', 'scripts'] },
    {
      label: 'ruff format',
      command: ['uv', 'run', '--group', 'dev', 'ruff', 'format', '--check', 'oracle', 'scripts'],
    },
  ]
  for (const p of ['engine', 'render', 'meta', 'hub'])
    steps.push({
      label: `tsc ${p}`,
      command: ['pnpm', '--dir', `packages/${p}`, 'exec', 'tsc', '-p', 'tsconfig.json', '--noEmit'],
    })
  steps.push({
    label: 'tsc apps/site',
    command: ['pnpm', '--dir', 'apps/site', 'exec', 'tsc', '-p', 'tsconfig.json', '--noEmit'],
  })
  steps.push({ label: 'tsc scripts and config', command: ['pnpm', 'exec', 'tsc', '-p', 'tsconfig.json'] })

  const python = files.filter((f) => f.startsWith('oracle/') && f.endsWith('.py') && existsSync(join(ROOT, f)))
  if (globalPy)
    steps.push({
      label: 'pyright (all)',
      command: ['uv', 'run', '--group', 'dev', ...ORACLE_GROUPS, 'pyright', 'oracle'],
    })
  else if (python.length > 0)
    steps.push({
      label: 'pyright (changed)',
      command: ['uv', 'run', '--group', 'dev', ...ORACLE_GROUPS, 'pyright', ...python],
    })

  if (globalTs) steps.push({ label: 'tests (all)', command: ['pnpm', 'exec', 'vitest', 'run'] })
  else {
    const sources = files.filter((f) => /^(packages|apps)\/.*\.tsx?$/.test(f) && existsSync(join(ROOT, f)))
    const cases = files.filter((f) => f.startsWith(CASES) && f.endsWith('.py'))
    const fixtures = [
      ...files.filter((f) => f.startsWith(FIXTURES) && f.endsWith('.json') && !f.endsWith('manifest.json')),
      ...cases.map((c) => `${FIXTURES}${c.slice(CASES.length).replace(/\.py$/, '.json')}`),
    ]
    const targets = [...new Set([...sources, ...testsLoadingFixtures(fixtures)])]
    if (targets.length > 0)
      steps.push({
        label: `tests related to ${targets.length} changed files`,
        command: ['pnpm', 'exec', 'vitest', 'related', '--run', '--passWithNoTests', ...targets],
      })
  }

  if (globalPy)
    steps.push({
      label: 'fixtures (all)',
      command: ['uv', 'run', ...ORACLE_GROUPS, 'python', '-m', 'whitebox_oracle', 'check'],
    })
  else {
    const only = [
      ...new Set(
        files
          .filter(
            (f) =>
              (f.startsWith(CASES) && f.endsWith('.py')) ||
              (f.startsWith(FIXTURES) && f.endsWith('.json') && !f.endsWith('manifest.json')),
          )
          .map((f) =>
            f.startsWith(CASES)
              ? f.slice(CASES.length).replace(/\.py$/, '')
              : f.slice(FIXTURES.length).replace(/\.json$/, ''),
          ),
      ),
    ]
    if (only.length > 0)
      steps.push({
        label: `fixtures (${only.length} case modules)`,
        command: ['uv', 'run', ...ORACLE_GROUPS, 'python', '-m', 'whitebox_oracle', 'check', '--only', only.join(',')],
      })
  }
  return steps
}

// Several checkouts (agents' worktrees) verify at once; each test run already uses half the cores, so at most
// WHITEBOX_VERIFY_SLOTS (default 2) run their tests together and the rest wait for a slot.
const SLOTS_DIR = join(tmpdir(), 'whitebox-verify-slots')

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

function acquireSlot(): () => void {
  const slots = Number(process.env['WHITEBOX_VERIFY_SLOTS'] ?? 2)
  mkdirSync(SLOTS_DIR, { recursive: true })
  let announced = false
  for (;;) {
    for (let i = 0; i < slots; i++) {
      const dir = join(SLOTS_DIR, String(i))
      const owner = join(dir, 'pid')
      if (existsSync(dir)) {
        const pid = existsSync(owner) ? Number(readFileSync(owner, 'utf8')) : Number.NaN
        if (Number.isFinite(pid) && alive(pid)) continue
        rmSync(dir, { recursive: true, force: true })
      }
      try {
        mkdirSync(dir)
      } catch {
        continue
      }
      writeFileSync(owner, String(process.pid))
      const release = (): void => rmSync(dir, { recursive: true, force: true })
      process.on('exit', release)
      for (const signal of ['SIGINT', 'SIGTERM'] as const)
        process.on(signal, () => {
          release()
          process.exit(130)
        })
      return release
    }
    if (!announced) process.stdout.write(`verify: waiting for a test slot (${slots} in use)\n`)
    announced = true
    spawnSync('sleep', ['5'])
  }
}

function main(): void {
  const args = process.argv.slice(2)
  const at = args.indexOf('--base')
  const base = at >= 0 ? (args[at + 1] ?? 'main') : (process.env['VERIFY_BASE'] ?? 'main')
  const files = changedFiles(base)
  process.stdout.write(`verify: ${files.length} changed files against ${base}\n`)
  const steps = plan(files)
  let release: (() => void) | undefined
  for (const step of steps) {
    if (release === undefined && step.label.startsWith('tests') && !args.includes('--dry-run')) release = acquireSlot()
    process.stdout.write(`\n== ${step.label}\n`)
    if (args.includes('--dry-run')) {
      process.stdout.write(`   ${step.command.join(' ').slice(0, 300)}\n`)
      continue
    }
    const [cmd, ...rest] = ['mise', 'exec', '--', ...step.command]
    const r = spawnSync(cmd as string, rest, { cwd: ROOT, stdio: 'inherit' })
    if (r.status !== 0) {
      process.stdout.write(`\nverify: ${step.label} failed\n`)
      process.exit(r.status ?? 1)
    }
  }
  process.stdout.write('\nverify: all checks passed\n')
}

main()
