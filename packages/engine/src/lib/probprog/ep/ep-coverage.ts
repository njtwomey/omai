/**
 * The measurement of what expectation propagation covers today: every EP atom (run, not only compiled: the atoms are
 * milliseconds) and every benchmark model of the suite (compiled to a factor graph and asked for the EP verdict, the audit
 * of `docs/probprog/ep-plan.md` section 1), each as `runs` or `planned` with the library's reason. It is a measurement for
 * the developer: the coverage page lists it, and its test fixes the figure so that coverage only goes up.
 */

import { epGridMethod, epSpecMethod } from '../inference/index.ts'
import { analysisOf } from '../inference/model-input.ts'
import { benchmarkModels } from '../models/index.ts'
import { epAtoms } from './ep-atom-list.ts'
import { atomRun } from './ep-atoms.ts'

/** One row of the coverage: a setup or a model, whether EP runs on it, and why not. */
export interface CoverageRow {
  readonly name: string
  readonly title: string
  /** `atom` (a bare setup of one factor) or `benchmark` (a model of the benchmark suite). */
  readonly kind: 'atom' | 'benchmark'
  /** The family of an atom; the tags of a benchmark model. */
  readonly group: string
  readonly status: 'runs' | 'planned'
  /** For an atom, what it declares (its expectation); a mismatch with `status` is a rule that landed or a regression. */
  readonly declared?: 'runs' | 'planned'
  /** The library's reason when EP does not run (the verdict, or the error of the run). */
  readonly reason: string
  /** What the library says is missing. */
  readonly missing: readonly string[]
  /** How EP runs when it does: `ep` (the model as it is) or `ep-grid` (EP inside a grid over the hyperparameters a custom function reads). Absent for an atom. */
  readonly via?: 'ep' | 'ep-grid'
  /** The factors an atom exercises. */
  readonly factors: readonly string[]
}

/** The coverage: the rows and the counts. */
export interface EpCoverage {
  readonly atoms: readonly CoverageRow[]
  readonly benchmarks: readonly CoverageRow[]
  readonly atomsRun: number
  readonly benchmarksRun: number
}

/**
 * What EP covers today: each atom run once at its default knobs, and each benchmark model compiled to a factor graph and
 * asked for the EP verdict (and, when that refuses, the verdict of EP inside a grid over the hyperparameters). A model the compiler throws on counts as planned with the error as its reason (a gap to report).
 *
 * @returns The rows and the counts.
 * @concept expectation-propagation
 * @index function What EP covers today: atoms and benchmark models as runs or planned, with reasons.
 * @example
 * ```ts
 * import { epCoverage } from 'omai-engine/probprog/ep'
 * const c = epCoverage()
 * c.atomsRun > 0 && c.benchmarks.length > 20 // => true
 * ```
 */
export function epCoverage(): EpCoverage {
  const atoms = epAtoms.map((a): CoverageRow => {
    const run = atomRun(a.problem(a.defaults), a.ep ?? {})
    return {
      name: a.name,
      title: a.title,
      kind: 'atom',
      group: a.family,
      status: run.status === 'ran' ? 'runs' : 'planned',
      declared: a.expected,
      reason: run.status === 'ran' ? '' : run.reason,
      missing: run.status === 'refused' ? run.missing : [],
      factors: a.factors,
    }
  })
  const benchmarks = benchmarkModels().map((m): CoverageRow => {
    const base = { name: m.name, title: m.title, kind: 'benchmark' as const, group: m.tags.join(', '), factors: [] }
    if (m.definition === undefined)
      return {
        ...base,
        status: 'planned',
        reason: 'the model is a closure program, which has no factor graph',
        missing: ['a model spec'],
      }
    try {
      const bound = m.bound()
      const verdict = epSpecMethod(bound).check?.(analysisOf(bound))
      if (verdict?.applies === true) return { ...base, status: 'runs', via: 'ep', reason: '', missing: [] }
      // A model whose graph exists once some scalar latents are known runs inside a grid over them.
      const grid = epGridMethod(bound).check?.(analysisOf(bound))
      if (grid?.applies === true) return { ...base, status: 'runs', via: 'ep-grid', reason: '', missing: [] }
      return { ...base, status: 'planned', reason: verdict?.reason ?? 'no verdict', missing: verdict?.missing ?? [] }
    } catch (e) {
      return {
        ...base,
        status: 'planned',
        reason: `the compiler throws: ${e instanceof Error ? e.message : String(e)}`,
        missing: ['a refusal instead of an error'],
      }
    }
  })
  return {
    atoms,
    benchmarks,
    atomsRun: atoms.filter((r) => r.status === 'runs').length,
    benchmarksRun: benchmarks.filter((r) => r.status === 'runs').length,
  }
}
