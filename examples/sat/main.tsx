/**
 * Example "A SAT solver at work": the pigeonhole puzzle (six pigeons, five holes, one pigeon to a hole) written as a
 * Boolean formula, which has no solution, and two solvers proving it: DPLL by backtracking and CDCL by learning a clause
 * from each conflict. Both are traced runs; the figure is their conflicts so far and the depth of the search, read back from the tapes.
 */
import { type Cnf, cdcl, cnf, dpll } from 'mdml-engine/logic'
import { type Run, recorder, tracePolicy } from 'mdml-engine/trace'
import { runsChart } from 'mdml-render/trace'
import { INK } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'Two SAT solvers on a hard puzzle',
  summary:
    'Six pigeons cannot each have their own hole in five holes. Two SAT solvers, DPLL and CDCL, both prove it, and their traces differ.',
  readme: false,
  alt: 'Conflicts and decision level against steps for the DPLL and CDCL solvers on the pigeonhole formula with six pigeons and five holes',
}

const pigeons = 6
const holes = 5

/** Variable (i, j) is "pigeon i sits in hole j"; each pigeon sits somewhere, and no hole holds two. */
function pigeonhole(): Cnf {
  const v = (i: number, j: number): number => i * holes + j + 1
  const clauses: number[][] = []
  for (let i = 0; i < pigeons; i++) clauses.push(Array.from({ length: holes }, (_, j) => v(i, j)))
  for (let j = 0; j < holes; j++)
    for (let i = 0; i < pigeons; i++) for (let k = i + 1; k < pigeons; k++) clauses.push([-v(i, j), -v(k, j)])
  return cnf(clauses, pigeons * holes)
}

interface Result {
  readonly runs: Readonly<Record<'dpll' | 'cdcl', Run>>
}

export function run(): Outcome<Result> {
  const formula = pigeonhole()
  const policy = tracePolicy({ metrics: ['conflicts', 'level'], schedule: { every: 1 } })
  const tapes = { dpll: recorder(), cdcl: recorder() }
  const a = dpll(formula, { tracer: tapes.dpll, policy })
  const b = cdcl(formula, { tracer: tapes.cdcl, policy })
  const runs = { dpll: tapes.dpll.run(), cdcl: tapes.cdcl.run() }
  const notes = [
    `${formula.numVars} variables, ${formula.clauses.length} clauses`,
    `dpll ${a.status}: ${a.stats.decisions} decisions, ${a.stats.conflicts} conflicts`,
    `cdcl ${b.status}: ${b.stats.decisions} decisions, ${b.stats.conflicts} conflicts, ${b.stats.learned} clauses learned`,
  ]
  return { result: { runs }, tapes: runs, notes }
}

export function View({ result }: { result: Result }) {
  const both = [
    { name: 'DPLL', run: result.runs.dpll },
    { name: 'CDCL', run: result.runs.cdcl },
  ]
  return (
    <Figure width={900} height={420} title={meta.alt}>
      <text x={12} y={22} fontFamily="system-ui, sans-serif" fontSize={14} fontWeight={600} fill={INK}>
        Can 6 pigeons each have their own hole among 5 holes? No: two SAT solvers prove it, one step at a time
      </text>
      <g transform="translate(0, 30)">
        <Panel x={0} width={450} height={320} title="conflicts found so far" still={runsChart(both, 'conflicts')} />
        <Panel
          x={450}
          width={450}
          height={320}
          title="decision level (how deep the search is)"
          still={runsChart(both, 'level')}
        />
      </g>
      <text x={450} y={410} textAnchor="middle" fontFamily="system-ui, sans-serif" fontSize={12} fill={INK}>
        A conflict is a dead end: the choices made so far cannot all be right. DPLL backs up and tries again; CDCL also
        learns a rule from each dead end.
      </text>
    </Figure>
  )
}
