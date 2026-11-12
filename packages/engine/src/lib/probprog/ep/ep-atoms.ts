/**
 * The functions of the expectation propagation atoms: the reference answer of an atom (closed-form marginals, or the grid
 * posterior of the model written as a program, as curves with their moments and the joint of its pair), the compile-time verdict
 * alone (all a coverage table needs), the check of a run's beliefs against the reference under the atom's tolerance, and the
 * atom run from start to finish. `atomRun` is `runEp` (`run.ts`) on the atom's model. A page of the proofs of concept is a
 * composition of these, and a test generated from the registry asserts every atom passes its check.
 *
 * @index structure EP atom functions: the reference of an atom, its verdict, the check of a run against it.
 */

import { OmaiError } from '../../base/index.ts'
import type { Config } from '../../trace/index.ts'
import type { Verdict } from '../inference/index.ts'
import { epRoute } from '../inference/index.ts'
import { type BoundModel, toProgram } from '../model/index.ts'
import { type ExactMarginal, gridPosterior, lg } from '../validation/index.ts'
import type {
  AtomBuild,
  AtomCheck,
  AtomProblem,
  AtomQuantity,
  AtomReference,
  AtomSupport,
  AtomTolerance,
  Belief,
  EpAtom,
  EpRun,
  QuantityCheck,
  RefMarginal,
} from './atom-types.ts'
import { runEp } from './run.ts'

/**
 * The density of a belief on the support of its quantity, matched to its mean and standard deviation: a normal on the line,
 * a gamma on the positive axis, a beta on the unit interval (the families EP carries for those).
 *
 * @param support - Where the quantity lives.
 * @param mean - The mean.
 * @param sd - The standard deviation.
 * @returns The density.
 * @concept expectation-propagation
 * @index function The density of a belief from its mean and sd on a support.
 * @example
 * ```ts
 * import { beliefDensity } from 'omai-engine/probprog/ep'
 * beliefDensity('real', 0, 1)(0).toFixed(4) // => '0.3989'
 * ```
 */
export function beliefDensity(support: AtomSupport, mean: number, sd: number): (x: number) => number {
  if (support === 'positive') {
    const shape = (mean / sd) ** 2
    const rate = mean / sd ** 2
    const logNorm = shape * Math.log(rate) - lg(shape)
    return (x) => (x <= 0 ? 0 : Math.exp(logNorm + (shape - 1) * Math.log(x) - rate * x))
  }
  if (support === 'unit') {
    const common = (mean * (1 - mean)) / sd ** 2 - 1
    const a = mean * common
    const b = (1 - mean) * common
    const logNorm = -(lg(a) + lg(b) - lg(a + b))
    return (x) => (x <= 0 || x >= 1 ? 0 : Math.exp(logNorm + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)))
  }
  return (x) => Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
}
const CURVE_POINTS = 400
/** A marginal from a closed form: the density sampled on its interval. */
function exactCurve(name: string, support: AtomSupport, m: ExactMarginal): RefMarginal {
  if (support === 'probability') return { name, mean: m.mean, sd: m.sd, x: [], p: [] }
  const lo0 = m.range?.[0] ?? m.mean - 8 * m.sd
  const hi0 = m.range?.[1] ?? m.mean + 8 * m.sd
  const lo = support === 'real' ? lo0 : Math.max(lo0, support === 'unit' ? 0 : 0)
  const hi = support === 'unit' ? Math.min(hi0, 1) : hi0
  const x = Array.from({ length: CURVE_POINTS }, (_, i) => lo + ((hi - lo) * i) / (CURVE_POINTS - 1))
  return { name, mean: m.mean, sd: m.sd, x, p: x.map((v) => m.pdf(v)) }
}
/**
 * The bound model of an atom's problem.
 *
 * @param problem - The problem.
 * @returns The model with its data bound.
 * @example
 * ```ts
 * import { atomBound, epAtom } from 'omai-engine/probprog/ep'
 * const gaussianAtom = epAtom('gaussian')
 * atomBound(gaussianAtom.problem(gaussianAtom.defaults)).sizes // => { i: 3 }
 * ```
 */
export function atomBound(problem: AtomBuild): BoundModel {
  return problem.definition.bind(problem.data, problem.bindOptions)
}
/**
 * The reference answer of an atom: closed-form marginals, or the grid posterior of the model written as a program (which
 * runs whether or not expectation propagation applies), as curves with their moments, and the joint of the atom's pair.
 *
 * @param atom - The atom.
 * @param problem - The problem a configuration made.
 * @returns The reference.
 * @throws OmaiError with code `value` when an `exact` atom gives no closed form for a quantity.
 * @concept posterior-validation
 * @index function The reference of an EP atom: exact or grid marginals as curves, and a joint.
 * @example
 * ```ts
 * import { atomReference, epAtom } from 'omai-engine/probprog/ep'
 * const gaussianAtom = epAtom('gaussian')
 * const r = atomReference(gaussianAtom, gaussianAtom.problem(gaussianAtom.defaults))
 * [r.kind, Object.keys(r.marginals)] // => ['exact', ['mu']]
 * ```
 */
export function atomReference<K extends Config>(atom: EpAtom<K>, problem: AtomProblem): AtomReference {
  if (atom.reference === 'truth')
    return {
      kind: 'truth',
      source: 'the true values the data were simulated from (no reference posterior)',
      marginals: {},
    }
  if (atom.reference === 'exact') {
    const marginals: Record<string, RefMarginal> = {}
    for (const q of problem.quantities) {
      const m = problem.exact?.[q.name]
      if (m === undefined)
        throw new OmaiError('value', `atomReference: '${atom.name}' has no closed form for '${q.name}'`)
      marginals[q.name] = exactCurve(q.name, q.support, m)
    }
    return {
      kind: 'exact',
      source: problem.referenceSource ?? 'the closed-form posterior',
      marginals,
      ...(problem.exactJoint === undefined ? {} : { joint: problem.exactJoint }),
      ...(problem.exactLogEvidence === undefined ? {} : { logEvidence: problem.exactLogEvidence }),
    }
  }
  const grid = gridPosterior(toProgram(atomBound(problem)), undefined, { maxRefinements: 0, ...problem.grid })
  const marginals: Record<string, RefMarginal> = {}
  for (const q of problem.quantities) {
    if (q.support === 'probability') {
      const m = problem.exact?.[q.name]
      if (m === undefined)
        throw new OmaiError('value', `atomReference: '${atom.name}' has no closed form for '${q.name}'`)
      marginals[q.name] = exactCurve(q.name, q.support, m)
      continue
    }
    const o = grid.marginal(q.name)
    const d = grid.density(q.name)
    marginals[q.name] = { name: q.name, mean: o.mean, sd: o.sd, x: d.x, p: d.p }
  }
  const pair = problem.pair
  const joint = pair === undefined ? undefined : { a: pair[0], b: pair[1], ...grid.joint(pair[0], pair[1]) }
  return {
    kind: 'grid',
    source: `quadrature of the posterior density on a ${grid.points}-point grid over ${grid.dim} coordinate(s)`,
    marginals,
    ...(joint === undefined ? {} : { joint }),
    logEvidence: grid.logEvidence,
  }
}
/**
 * Whether the library's expectation propagation applies to an atom's problem, and if not why not, without running
 * anything: the compile-time verdict (the factor graph, and a message rule for every operator of it).
 *
 * @param problem - The problem.
 * @returns The verdict.
 * @concept expectation-propagation
 * @index function Whether EP applies to an atom's model, and the library's reason if not.
 * @example
 * ```ts
 * import { atomVerdict, epAtom } from 'omai-engine/probprog/ep'
 * const gaussianAtom = epAtom('gaussian')
 * atomVerdict(gaussianAtom.problem(gaussianAtom.defaults)).applies // => true
 * ```
 */
export function atomVerdict(problem: AtomBuild): Verdict {
  return epRoute(atomBound(problem), problem.hyperGrid === undefined ? {} : { gridOver: problem.hyperGrid.over })
    .verdict
}
/**
 * The means of the reference's quantities, by name (what a data figure draws the exact answer from).
 *
 * @param reference - The reference.
 * @returns The means.
 * @example
 * ```ts
 * import { atomReference, epAtom, marginalMean } from 'omai-engine/probprog/ep'
 * const gaussianAtom = epAtom('gaussian')
 * const r = atomReference(gaussianAtom, gaussianAtom.problem(gaussianAtom.defaults))
 * Object.keys(marginalMean(r)) // => ['mu']
 * ```
 */
export function marginalMean(reference: AtomReference): Record<string, number> {
  return Object.fromEntries(Object.entries(reference.marginals).map(([k, m]) => [k, m.mean]))
}
/** The Kolmogorov-Smirnov distance between a density and a reference curve, on the reference's points. */
function ksOf(pdf: (x: number) => number, ref: RefMarginal): number {
  const cum = (p: readonly number[]): number[] => {
    const out = [0]
    for (let i = 1; i < ref.x.length; i++)
      out.push((out[i - 1] ?? 0) + (((p[i] ?? 0) + (p[i - 1] ?? 0)) / 2) * ((ref.x[i] ?? 0) - (ref.x[i - 1] ?? 0)))
    const total = out[out.length - 1] || 1
    return out.map((v) => v / total)
  }
  const a = cum(ref.x.map(pdf))
  const b = cum(ref.p)
  return a.reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)
}
/**
 * The check of a run's beliefs against the reference: for each quantity the mean's error in reference standard deviations,
 * the ratio of the standard deviations and the Kolmogorov-Smirnov distance between EP's belief and the reference, each
 * against the atom's tolerance.
 *
 * @param problem - The problem (the quantities and their supports).
 * @param beliefs - EP's beliefs.
 * @param reference - The reference.
 * @param tolerance - The tolerance.
 * @returns The check: a row per quantity and whether all pass.
 * @concept posterior-validation
 * @index function EP's beliefs against the reference: mean error, sd ratio, KS against a tolerance.
 * @example
 * ```ts
 * import { atomCheck, atomReference, atomRun, epAtom } from 'omai-engine/probprog/ep'
 * const gaussianAtom = epAtom('gaussian')
 * const p = gaussianAtom.problem(gaussianAtom.defaults)
 * const r = atomRun(p)
 * r.status === 'ran' && atomCheck(p, r.beliefs, atomReference(gaussianAtom, p), gaussianAtom.tolerance).pass // => true
 * ```
 */
export function atomCheck(
  problem: Pick<AtomBuild, 'quantities'>,
  beliefs: Readonly<Record<string, Belief>>,
  reference: AtomReference,
  tolerance: AtomTolerance,
): AtomCheck {
  const rows = problem.quantities.flatMap((q): QuantityCheck[] => {
    const ref = reference.marginals[q.name]
    if (ref === undefined) return []
    const ep = beliefs[q.name] ?? { mean: Number.NaN, sd: Number.NaN, proper: false }
    const meanZ = Math.abs(ep.mean - ref.mean) / ref.sd
    // A probability's spread is that of a coin: where it is nearly certain both are about zero and their ratio means nothing.
    const sdRatio = q.support === 'probability' ? (ep.sd + 1e-2) / (ref.sd + 1e-2) : ep.sd / ref.sd
    const ks =
      q.support === 'probability' ? 0 : ep.proper ? ksOf(beliefDensity(q.support, ep.mean, ep.sd), ref) : Number.NaN
    const pass =
      ep.proper &&
      meanZ <= tolerance.meanZ &&
      sdRatio >= tolerance.sdRatio[0] &&
      sdRatio <= tolerance.sdRatio[1] &&
      ks <= tolerance.ks
    return [{ name: q.name, label: q.label, ep, refMean: ref.mean, refSd: ref.sd, meanZ, sdRatio, ks, pass }]
  })
  return { pass: rows.every((r) => r.pass), rows, tolerance }
}
/**
 * Run an atom at its default knobs (or the given ones) from start to finish: the problem, the verdict, the reference, the
 * EP run and the check. What the registry test and the coverage table do per atom.
 *
 * @param atom - The atom.
 * @param knobs - Knob values over the defaults.
 * @returns The problem, the reference, the run and (when EP ran) the check.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { epAtom, verifyAtom } from 'omai-engine/probprog/ep'
 * const gaussianAtom = epAtom('gaussian')
 * verifyAtom(gaussianAtom).check?.pass // => true
 * ```
 */
export function verifyAtom<K extends Config>(
  atom: EpAtom<K>,
  knobs: Partial<K> = {},
): {
  readonly problem: AtomProblem
  readonly reference: AtomReference
  readonly run: EpRun
  readonly check?: AtomCheck
} {
  const problem = atom.problem({ ...atom.defaults, ...knobs })
  const reference = atomReference(atom, problem)
  const run = atomRun(problem, atom.ep ?? {})
  return {
    problem,
    reference,
    run,
    ...(run.status === 'ran' ? { check: atomCheck(problem, run.beliefs, reference, atom.tolerance) } : {}),
  }
}

/**
 * Run expectation propagation on an atom's problem, sweep by sweep, and keep the beliefs at each sweep and the messages as a
 * model-run record (`runEp` on the atom's model, reading the atom's quantities). A problem the library refuses is not run: the
 * result carries the library's verdict.
 *
 * @param problem - The problem.
 * @param ep - The knobs of EP over its defaults (damping, schedule, tolerance, the sweep cap, the power).
 * @param options - `light` keeps no tape and no per-sweep beliefs (only the final ones): what a calibration over many seeds needs.
 * @returns The run, or the refusal.
 * @throws OmaiError with code `value` for a key of `ep` that is not a knob of the method (a misspelled knob would run on the defaults).
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { atomRun, epAtom } from 'omai-engine/probprog/ep'
 * const gaussian = epAtom('gaussian')
 * atomRun(gaussian.problem(gaussian.defaults)).status // => 'ran'
 * ```
 */
export function atomRun(problem: AtomBuild, ep: Config = {}, options: { readonly light?: boolean } = {}): EpRun {
  return runEp(atomBound(problem), {
    quantities: problem.quantities,
    ep,
    ...(problem.hyperGrid === undefined ? {} : { hyperGrid: problem.hyperGrid }),
    ...(options.light === undefined ? {} : { light: options.light }),
  })
}
