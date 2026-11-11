/**
 * The measurement of what expectation propagation covers today: every EP atom (run, not only compiled: the atoms are
 * milliseconds) and every benchmark model of the suite (compiled to a factor graph and asked for the EP verdict, the audit
 * of `docs/probprog/ep-plan.md` section 1), each as `runs` or `planned` with the library's reason. It is a measurement for
 * the developer: the coverage page lists it, and its test fixes the figure so that coverage only goes up.
 */

import { type Config, session } from '../../trace/index.ts'
import { epGridMethod, epSpecMethod, vmpSpecMethod } from '../inference/index.ts'
import { analysisOf } from '../inference/model-input.ts'
import { benchmarkModels } from '../models/index.ts'
import { epAtoms } from './ep-atom-list.ts'
import { atomBeliefs, atomBound, atomCheck, atomReference, atomRun } from './ep-atoms.ts'

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

/** One row of the variational coverage: an atom, whether variational message passing runs it, how well, and why not. */
export interface VmpCoverageRow {
  readonly name: string
  readonly title: string
  /** The family of the atom. */
  readonly group: string
  /** The factors the atom exercises. */
  readonly factors: readonly string[]
  /** `runs` when every operator of the atom's graph has a variational message and the run gave proper beliefs; `planned` otherwise. */
  readonly status: 'runs' | 'planned'
  /** The library's reason when variational message passing does not run: the operators with no rule, with their own reason. */
  readonly reason: string
  readonly missing: readonly string[]
  /** Whether expectation propagation runs the atom (the variational method is compared where both do). */
  readonly epRuns: boolean
  /** For a run: whether it converged and in how many sweeps. */
  readonly converged?: boolean
  readonly sweeps?: number
  /** For a run with the accuracy measured: the largest error of a mean in reference standard deviations, and the range of the ratios of the standard deviations to the reference's (the mean field under-estimates the spread). */
  readonly meanZ?: number
  readonly sdRatio?: readonly [number, number]
}

/** The variational coverage: a row per atom, and the counts. */
export interface VmpCoverage {
  readonly atoms: readonly VmpCoverageRow[]
  readonly atomsRun: number
  readonly atomsConverged: number
  /** Atoms expectation propagation runs and variational message passing does not: the rule gaps. */
  readonly behindEp: number
}

/** The knobs variational message passing runs the atoms with: the tolerance, the sweep cap and a damping that steadies the non-conjugate messages. */
const VMP_KNOBS: Config = { tol: 1e-9, maxSweeps: 300, damping: 0.3, improper: 'damp' }

/** The heavier knobs a run that did not settle is tried again with. */
const VMP_STEADY_KNOBS: Config = { tol: 1e-9, maxSweeps: 1000, damping: 0.7, improper: 'damp' }

/**
 * What variational message passing covers today: each atom compiled to a factor graph and asked for the variational verdict
 * (every operator of the graph needs a mean-field message; the ones that have none carry their reason), and, when it applies,
 * run to its fixed point; with `accuracy` the beliefs are checked against the atom's reference (closed form or grid), as the mean's
 * error in reference standard deviations and the ratio of the standard deviations (below one for a mean field).
 *
 * @param options - `accuracy` also computes each atom's reference (a grid for most: about a minute in all); `names` restricts the atoms.
 * @returns The rows and the counts.
 * @concept variational-message-passing
 * @index function What variational message passing covers today: atoms as runs or planned, with the operators lacking a rule.
 * @example
 * ```ts
 * import { vmpCoverage } from 'omai-engine/probprog/ep'
 * const c = vmpCoverage()
 * c.atomsRun > 40 && c.atoms.length > 80 // => true
 * ```
 */
export function vmpCoverage(
  options: { readonly accuracy?: boolean; readonly names?: readonly string[] } = {},
): VmpCoverage {
  const chosen = options.names === undefined ? epAtoms : epAtoms.filter((a) => options.names?.includes(a.name))
  const atoms = chosen.map((a): VmpCoverageRow => {
    const base = {
      name: a.name,
      title: a.title,
      group: a.family,
      factors: a.factors,
      epRuns: a.expected === 'runs',
    }
    const problem = a.problem(a.defaults)
    try {
      const bound = atomBound(problem)
      const method = vmpSpecMethod(bound)
      const verdict = method.check?.(analysisOf(bound))
      if (verdict?.applies !== true)
        return { ...base, status: 'planned', reason: verdict?.reason ?? 'no verdict', missing: verdict?.missing ?? [] }
      let s = session(method, { seed: 1, steps: 302, mode: 'eager', config: VMP_KNOBS, logs: 'off' })
      // A run that did not settle is tried again with a heavier damping and more sweeps (the non-conjugate messages overshoot).
      if (s.status !== 'converged')
        s = session(method, { seed: 1, steps: 1002, mode: 'eager', config: VMP_STEADY_KNOBS, logs: 'off' })
      const posterior = method.posterior?.(s.state, s.config)
      const beliefs = posterior === undefined ? {} : atomBeliefs(problem, posterior as never)
      // A derived quantity (the value a gate selects, from EP's branch beliefs) is a reading of EP's, not of the mean field's.
      const direct = problem.quantities.filter((q) => q.derive === undefined)
      const proper = direct.every((q) => beliefs[q.name]?.proper === true)
      const row = { ...base, converged: s.status === 'converged', sweeps: s.t }
      if (!proper)
        return {
          ...row,
          status: 'planned',
          reason: 'the run ended with a belief that has no density',
          missing: ['a proper start'],
        }
      if (options.accuracy !== true) return { ...row, status: 'runs', reason: '', missing: [] }
      const names = new Set(direct.map((q) => q.name))
      const checked = atomCheck(problem, beliefs, atomReference(a, problem), a.tolerance).rows.filter((r) =>
        names.has(r.name),
      )
      const errors = checked.map((r) => r.meanZ).filter((v) => Number.isFinite(v))
      const ratios = checked.map((r) => r.sdRatio).filter((v) => Number.isFinite(v))
      return {
        ...row,
        status: 'runs',
        reason: '',
        missing: [],
        ...(errors.length === 0 ? {} : { meanZ: Math.max(...errors) }),
        ...(ratios.length === 0 ? {} : { sdRatio: [Math.min(...ratios), Math.max(...ratios)] as const }),
      }
    } catch (e) {
      return {
        ...base,
        status: 'planned',
        reason: `the run throws: ${e instanceof Error ? e.message : String(e)}`,
        missing: ['a refusal instead of an error'],
      }
    }
  })
  return {
    atoms,
    atomsRun: atoms.filter((r) => r.status === 'runs').length,
    atomsConverged: atoms.filter((r) => r.status === 'runs' && r.converged === true).length,
    behindEp: atoms.filter((r) => r.epRuns && r.status !== 'runs').length,
  }
}
