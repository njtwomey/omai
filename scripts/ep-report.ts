/**
 * The EP status report: what expectation propagation covers, measured from the code, so the page cannot go stale. It writes
 * `docs/probprog/ep-status.md` from the library's own measurements (the operator registry, `epCoverage`, `vmpCoverage`, the
 * atoms, the gaps with the refusals they derive, the directions) and reads the atlas `docs/probprog/ep-priors.md` against the code
 * (`scripts/ep-atlas.ts`): every atlas row says in its `ran by` column which atom or test runs it, its `status` word is the one the
 * code gives, and an atom it names must exist and run.
 *
 *   make ep-report         write docs/probprog/ep-status.md and the status cells of the atlas
 *   make ep-report-check   fail (exit 1) when the status file is out of date, a status word or a reference of the atlas has drifted
 *                          from the code (an atom that does not run, a test that is gone, a row that names neither), or the atlas
 *                          names an atom in prose that does not run
 *
 * The report holds counts and tables only, no clocks and no random source, so two runs of the same code give the same text.
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { defaultOperators } from '../packages/engine/src/lib/message-passing/index.ts'
import {
  type CoverageRow,
  epAtoms,
  epCoverage,
  epDirections,
  epGapsRefusing,
  unclaimedRefusals,
  vmpCoverage,
} from '../packages/engine/src/lib/probprog/ep/index.ts'
import { type AtlasReading, readAtlas } from './ep-atlas.ts'

const ROOT = join(import.meta.dirname, '..')
const STATUS = join(ROOT, 'docs/probprog/ep-status.md')
const PRIORS = join(ROOT, 'docs/probprog/ep-priors.md')

/** A markdown table from a header and rows. */
function table(header: readonly string[], rows: readonly (readonly (string | number)[])[]): string {
  const line = (cells: readonly (string | number)[]): string =>
    `| ${cells.map((c) => String(c).replaceAll('|', '\\|')).join(' | ')} |`
  return [line(header), line(header.map(() => '---')), ...rows.map(line)].join('\n')
}

const count = <T>(xs: readonly T[], f: (x: T) => boolean): number => xs.filter(f).length

function operatorSection(): string {
  const ops = Object.values(defaultOperators())
  const modes = (['ep', 'vmp', 'bp'] as const).map((m) => {
    const has = ops.filter((o) => o.support[m] !== undefined)
    const by = (s: string): number => count(has, (o) => o.support[m] === s)
    return [m, has.length, by('exact'), by('bound'), by('numeric')]
  })
  const powered = count(ops, (o) => o.power !== undefined && o.power !== false)
  return [
    `The registry (\`defaultOperators()\`) holds **${ops.length}** factor operators. Rules by message mode:`,
    '',
    table(['mode', 'operators with a rule', 'exact', 'bound', 'numeric'], modes),
    '',
    `${powered} operators take a power (fractional EP).`,
  ].join('\n')
}

function atomSection(c: ReturnType<typeof epCoverage>, v: ReturnType<typeof vmpCoverage>): string {
  const basic = count(epAtoms, (a) => a.extends === undefined)
  const families = [...new Set(epAtoms.map((a) => a.family))].sort()
  const vmpByName = new Map(v.atoms.map((r) => [r.name, r]))
  const rows = families.map((f) => {
    const inFamily = epAtoms.filter((a) => a.family === f)
    const ep = count(c.atoms, (r) => r.group === f && r.status === 'runs')
    const vmp = count(inFamily, (a) => vmpByName.get(a.name)?.status === 'runs')
    return [f, inFamily.length, count(inFamily, (a) => a.extends !== undefined), ep, vmp]
  })
  return [
    `**${epAtoms.length}** atoms (${basic} basic, ${epAtoms.length - basic} with unknown parameters learned from simulated data). Expectation propagation runs **${c.atomsRun}** of them. Variational message passing runs **${v.atomsRun}** (${v.atomsConverged} to a fixed point); **${v.behindEp}** run under EP and not under VMP.`,
    '',
    table(['family', 'atoms', 'learned twins', 'EP runs', 'VMP runs'], rows),
  ].join('\n')
}

function benchmarkSection(c: ReturnType<typeof epCoverage>): string {
  const via = (v: string): number => count(c.benchmarks, (r) => r.via === v)
  const rows = c.benchmarks.map((r: CoverageRow) => [
    r.name,
    r.status === 'runs' ? `runs (${r.via})` : `refused: ${r.reason}`,
  ])
  return [
    `**${c.benchmarks.length}** benchmark models; EP runs **${c.benchmarksRun}** (${via('ep')} as they are, ${via('ep-grid')} inside a grid over hyperparameters).`,
    '',
    table(['model', 'EP'], rows),
  ].join('\n')
}

function gapSection(c: ReturnType<typeof epCoverage>): string {
  const refused = [...c.atoms, ...c.benchmarks].filter((r) => r.status !== 'runs')
  const gaps = epGapsRefusing(refused)
  const orphans = unclaimedRefusals(refused)
  return [
    refused.length === 0
      ? 'EP refuses nothing in the suite today: every atom and every benchmark model runs. The gaps below are shortfalls of quality or reach, not refusals.'
      : `EP refuses ${refused.length} item(s): ${refused.map((r) => `\`${r.name}\``).join(', ')}.${orphans.length === 0 ? '' : ` **Unclaimed by any gap:** ${orphans.join(', ')}.`}`,
    '',
    table(
      ['gap', 'kind', 'status', 'size', 'refuses'],
      gaps.map((g) => [
        `${g.id}: ${g.title}`,
        g.kind,
        g.status,
        g.size,
        g.refuses.length === 0 ? '-' : g.refuses.join(', '),
      ]),
    ),
    '',
    `**${epDirections.length}** directions for the suite to grow into:`,
    '',
    table(
      ['direction', 'area', 'status', 'size'],
      epDirections.map((d) => [d.title, d.area, d.status, d.size]),
    ),
  ].join('\n')
}

/** The atoms the atlas names (as backticked words) that exist, and those among them that do not run. */
function atlasCheck(c: ReturnType<typeof epCoverage>): {
  readonly named: readonly string[]
  readonly notRunning: readonly string[]
} {
  const names = new Set(epAtoms.map((a) => a.name))
  const text = existsSync(PRIORS) ? readFileSync(PRIORS, 'utf8') : ''
  const named = [
    ...new Set([...text.matchAll(/`([a-z][a-z0-9-]*)`/g)].map((m) => m[1] as string).filter((w) => names.has(w))),
  ].sort()
  const runs = new Set(c.atoms.filter((r) => r.status === 'runs').map((r) => r.name))
  return { named, notRunning: named.filter((n) => !runs.has(n)) }
}

/** The atlas read against the code: the atoms that run, and the test files with the texts they hold. */
function atlasReading(c: ReturnType<typeof epCoverage>): AtlasReading {
  const atoms = new Map(epAtoms.map((a) => [a.name, c.atoms.some((r) => r.name === a.name && r.status === 'runs')]))
  const fileHas = (path: string, needle: string | undefined): boolean => {
    const file = join(ROOT, path)
    return existsSync(file) && (needle === undefined || readFileSync(file, 'utf8').includes(needle))
  }
  return readAtlas(existsSync(PRIORS) ? readFileSync(PRIORS, 'utf8') : '', { atoms, fileHas })
}

function report(): {
  readonly text: string
  readonly atlasText: string
  readonly problems: readonly string[]
  readonly unfixable: readonly string[]
} {
  const c = epCoverage()
  const v = vmpCoverage()
  const atlas = atlasCheck(c)
  const rows = atlasReading(c)
  const byStatus = (s: string): number => count(rows.rows, (r) => r.status === s)
  const refAtoms = new Set(rows.rows.flatMap((r) => r.atoms))
  const refTests = new Set(rows.rows.flatMap((r) => r.tests))
  const text = [
    '# EP status',
    '',
    'Generated by `make ep-report` from the code (`scripts/ep-report.ts`): the operator registry, `epCoverage`, `vmpCoverage`, the atoms and',
    'the gaps with the refusals they derive. Do not edit by hand; `make ep-report-check` fails when this file is out of date. The design is',
    'in [`ep.md`](ep.md), what happened when in [`ep-changelog.md`](ep-changelog.md).',
    '',
    '## Message-passing operators',
    '',
    operatorSection(),
    '',
    '## Atoms',
    '',
    atomSection(c, v),
    '',
    '## Benchmark models',
    '',
    benchmarkSection(c),
    '',
    '## Refusals, gaps and directions',
    '',
    gapSection(c),
    '',
    '## The prior atlas',
    '',
    `[\`ep-priors.md\`](ep-priors.md) has **${rows.rows.length}** rows, each with the atom or test that runs it (\`ran by\`) and a status word set from the code: **${byStatus('runs')}** run, **${byStatus('planned')}** marked \`not yet\`, **${byStatus('broken')}** have a reference that does not hold. The rows name **${refAtoms.size}** atoms and **${refTests.size}** tests.`,
    '',
    `Atoms named in its prose: ${atlas.named.length}; ${atlas.notRunning.length === 0 ? 'every one of them runs under EP' : `these do not run: ${atlas.notRunning.join(', ')}`}.`,
    '',
  ].join('\n')
  const prose = atlas.notRunning.map((n) => `ep-priors.md names the atom '${n}', which does not run under EP`)
  return {
    text,
    atlasText: rows.text,
    problems: [...rows.problems, ...prose],
    unfixable: [...rows.unfixable, ...prose],
  }
}

const { text, problems, unfixable, atlasText } = report()
if (process.argv.includes('--check')) {
  const current = existsSync(STATUS) ? readFileSync(STATUS, 'utf8') : ''
  const stale = current !== text
  for (const p of problems) process.stdout.write(`ep-report-check: ${p}\n`)
  if (stale) process.stdout.write('ep-report-check: docs/probprog/ep-status.md is out of date (run `make ep-report`)\n')
  if (stale || problems.length > 0) process.exit(1)
  process.stdout.write(
    'ep-report-check: docs/probprog/ep-status.md is current and every row of the atlas agrees with the code\n',
  )
} else {
  writeFileSync(STATUS, text)
  if (existsSync(PRIORS) && readFileSync(PRIORS, 'utf8') !== atlasText) writeFileSync(PRIORS, atlasText)
  // A stale status word is what writing settles; a reference that does not hold needs a person.
  for (const p of unfixable) process.stdout.write(`ep-report: ${p}\n`)
  process.stdout.write(`ep-report: wrote ${STATUS} and the status cells of ${PRIORS}\n`)
  if (unfixable.length > 0) process.exit(1)
}
