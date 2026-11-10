/**
 * The periodic validation of the probabilistic-programming code: simulation-based calibration, agreement with the grid
 * posterior, the funnel under non-centring and Geweke tests (the campaigns of
 * `packages/engine/src/lib/probprog/validation/campaigns.ts`), then the cross-engine agreement of every benchmark model
 * with its truth (`benchmark/<model>`, from `packages/engine/src/lib/probprog/models/campaigns.ts`). It is NOT part of `make verify` or `make check`: run it
 * when the sampling, probprog or message-passing code changed, and commit the tables it writes.
 *
 *   make probprog-validate [ARGS="--only sbc,geweke --scale 2 --seed 1 --out docs/probprog/results"]
 *   make probprog-validate ARGS="--check"
 *
 * Each campaign is written to `docs/probprog/results/<name>.md` (a `/` in the name becomes `-`) with the commit, the date,
 * the seed, the budgets and the time; `README.md` there lists them. `--check` prints which campaigns are older than the
 * last commit that touched the code they test (it never fails). The library has no clock and no file system, so the date,
 * the commit, the clock and the writing live here (scripts are exempt from `scripts/banned.ts`).
 */
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import {
  benchmarkCampaigns,
  evidenceCampaigns,
  structureCampaigns,
} from '../packages/engine/src/lib/probprog/models/index.ts'
import {
  campaignMarkdown,
  runCampaigns,
  validationCampaigns,
} from '../packages/engine/src/lib/probprog/validation/index.ts'

const ROOT = join(import.meta.dirname, '..')
const WATCHED = [
  'packages/engine/src/lib/probprog',
  'packages/engine/src/lib/sampling',
  'packages/engine/src/lib/message-passing',
]

function option(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : undefined
}

const git = (...args: string[]): string => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim()
const fileOf = (name: string): string => `${name.replaceAll('/', '-')}.md`
const out = resolve(ROOT, option('out') ?? 'docs/probprog/results')
// The benchmark suite's cross-engine agreement comes after the kernel's own campaigns, the evidence and particle campaigns after those and the structure reports last, so adding them changes no other campaign's key.
const campaigns = [...validationCampaigns(), ...benchmarkCampaigns(), ...evidenceCampaigns(), ...structureCampaigns()]

if (process.argv.includes('--check')) {
  const last = Number(git('log', '-1', '--format=%ct', '--', ...WATCHED)) * 1000
  for (const c of campaigns) {
    const path = join(out, fileOf(c.name))
    const state = !existsSync(path) ? 'missing' : statSync(path).mtimeMs < last ? 'stale' : 'fresh'
    process.stdout.write(`${state.padEnd(8)}${c.name}\n`)
  }
  process.exit(0)
}

const only = option('only')?.split(',')
const scale = Number(option('scale') ?? 1)
const seed = Number(option('seed') ?? 0)
const commit = git('rev-parse', '--short', 'HEAD')
const date = new Date().toISOString().slice(0, 10)
const runs = runCampaigns(campaigns, {
  seed,
  scale,
  ...(only === undefined ? {} : { only }),
  clock: () => performance.now() / 1000,
})
mkdirSync(out, { recursive: true })
for (const run of runs) {
  writeFileSync(join(out, fileOf(run.name)), campaignMarkdown(run, { date, commit }))
  process.stdout.write(
    `${run.passed ? 'ok    ' : 'FAILED'} ${run.name}  ${run.seconds?.toFixed(1) ?? '?'} s  ${run.verdict}\n`,
  )
}
if (only === undefined) {
  const lines = [
    '# Validation results',
    '',
    `Written by \`make probprog-validate\` on ${date} at commit ${commit} (seed ${seed}, budget scale ${scale}). Each file is one campaign; a diff of these tables is a regression.`,
    '',
    '| campaign | verdict | seconds |',
    '| --- | --- | --- |',
    ...runs.map(
      (r) =>
        `| [${r.name}](${fileOf(r.name)}) | ${r.passed ? 'passed' : '**FAILED**'}: ${r.verdict} | ${r.seconds?.toFixed(1) ?? ''} |`,
    ),
    '',
  ]
  writeFileSync(join(out, 'README.md'), lines.join('\n'))
}
process.exit(runs.every((r) => r.passed) ? 0 : 1)
