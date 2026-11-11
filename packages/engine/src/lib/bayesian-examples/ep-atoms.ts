/**
 * The scaffold of the expectation propagation atoms: very small, concrete problems (six noisy measurements of a length,
 * a handful of labelled points, three players and their matches) each of which exercises one factor of the message
 * passing suite and has a reference answer that can be checked by eye. An atom declares a problem from a few knobs (the
 * model as a `ModelDefinition` with its data, the quantities it asks about with their human names, the picture of the
 * data), the reference (the exact posterior in closed form, or the posterior on a grid of the same model written as a
 * program, so it exists whether or not expectation propagation applies), a tolerance with its reason, and what is
 * expected today: `runs` (the library's EP applies) or `planned` (it is refused, and the library's verdict says why).
 * `runAtom` compiles the model, asks the EP method for its verdict, runs it sweep by sweep and returns the beliefs at every
 * sweep, the messages as a model-run record and the check against the reference; `atomReference` is the reference alone
 * (marginal curves, and the joint of a pair for contours); `atomVerdict` is the compile-time verdict alone, which is all a
 * coverage table needs. A page of the proofs of concept is a composition of these, and a test generated from the registry
 * asserts every `runs` atom passes its check and every `planned` atom is refused with a reason, so coverage only goes up.
 *
 * @index structure EP atom scaffold: a tiny concrete problem, its reference, a tolerance and the EP verdict and run.
 */

import { OmaiError } from '../base/index.ts'
import { type ModelRun, modelRunFromTape } from '../exposition/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { epSpecMethod, type Verdict } from '../probprog/inference/index.ts'
import { analysisOf } from '../probprog/inference/model-input.ts'
import {
  type BindOptions,
  type BoundModel,
  type DataInput,
  type ModelDefinition,
  toProgram,
  toStructure,
} from '../probprog/model/index.ts'
import { type GridOptions, gridPosterior } from '../probprog/validation/index.ts'
import type { DictSpace } from '../space/index.ts'
import { defaults } from '../space/index.ts'
import { item, toFlat } from '../tensor/index.ts'
import { type Config, session } from '../trace/index.ts'

/** Where a quantity lives, which fixes the family its belief is drawn in. */
export type AtomSupport = 'real' | 'positive' | 'unit'

/** One number the atom asks about: its name in the model, what a person calls it, and the value that made the data. */
export interface AtomQuantity {
  /** The name of the coordinate in the posterior and the grid (`mu`, `skill_ana`, `w[0]`). */
  readonly name: string
  /** What it is, in words (`length`, `noise sd`, `skill of Ana`). */
  readonly label: string
  readonly support: AtomSupport
  /** The value the data were generated with, when there is one (drawn as a mark). */
  readonly truth?: number
}

/** A set of points of the data figure. */
export interface FigurePoints {
  readonly name: string
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** A group per point (a class, an outlier flag), with the name of each group. */
  readonly group?: readonly number[]
  readonly groupNames?: Readonly<Record<number, string>>
}

/** A curve of the data figure. */
export interface FigureCurve {
  readonly name: string
  readonly x: readonly number[]
  readonly y: readonly number[]
}

/** The picture of the data of an atom, and of the answer in the data's own coordinates. */
export interface AtomFigure {
  /** What the picture shows, in a sentence. */
  readonly caption: string
  readonly xLabel: string
  readonly yLabel: string
  readonly xLimits?: readonly [number, number]
  readonly yLimits?: readonly [number, number]
  readonly points: readonly FigurePoints[]
  /** Curves made from the data themselves (a running share, a smoothed line), drawn under the answer. */
  readonly curves?: readonly FigureCurve[]
  /** The curves a set of means stands for (a fitted line, a decision boundary), in the figure's axes; drawn for the exact means and for EP's. */
  readonly answer?: (means: Readonly<Record<string, number>>) => readonly FigureCurve[]
  /** Lines of text that say the data in words (the matches played, the labels given), for data a plot does not carry. */
  readonly notes?: readonly string[]
}

/** The closed-form marginal of a quantity: its moments and its density. */
export interface ExactMarginal {
  readonly mean: number
  readonly sd: number
  readonly pdf: (x: number) => number
  /** The interval the curve is drawn on (default the mean plus or minus eight standard deviations, kept inside the support). */
  readonly range?: readonly [number, number]
}

/** What an atom's knobs make: the model, its data and everything the page and the checks need around them. */
export interface AtomProblem {
  readonly definition: ModelDefinition
  readonly data: DataInput
  readonly bindOptions?: BindOptions
  /** The quantities checked and drawn, in order (at most three coordinates when the reference is a grid). */
  readonly quantities: readonly AtomQuantity[]
  /** Two quantities whose joint posterior is drawn as contours, when it is worth seeing. */
  readonly pair?: readonly [string, string]
  readonly figure: AtomFigure
  /** The closed-form marginals, by quantity name (when the atom's reference is `exact`). */
  readonly exact?: Readonly<Record<string, ExactMarginal>>
  /** Options of the grid (points per axis, an explicit box) when the reference is a grid. */
  readonly grid?: GridOptions
}

/** How close EP's beliefs must be to the reference for the atom to pass, with the reason EP is that far off (it is approximate). */
export interface AtomTolerance {
  /** The most the mean may be off, in reference standard deviations. */
  readonly meanZ: number
  /** The range of the ratio of EP's standard deviation to the reference's. */
  readonly sdRatio: readonly [number, number]
  /** The most the Kolmogorov-Smirnov distance between EP's belief and the reference may be. */
  readonly ks: number
  /** Why this much (an exact method: rounding; a probit: the Gaussian projection of a truncation). */
  readonly reason: string
}

/** What an atom declares. */
export interface AtomSpec<K extends Config> {
  /** The slug (`gaussian-mean`), the name of the page `ep-<name>`. */
  readonly name: string
  readonly title: string
  /** One sentence of what the problem is, in everyday words. */
  readonly story: string
  /** The factors or rules it exercises, by the library's names (`gaussian-gamma-precision`, `probit`). */
  readonly factors: readonly string[]
  /** The family of the atom, for the coverage table's grouping. */
  readonly family: string
  /** `runs` when EP applies today, `planned` when the library refuses (the verdict is asked at run time either way). */
  readonly expected: 'runs' | 'planned'
  /** For a `planned` atom, where in `docs/probprog/ep-plan.md` the missing rule is, and what it is. */
  readonly plan?: string
  readonly reference: 'exact' | 'grid'
  readonly tolerance: AtomTolerance
  /** The knobs of the problem (sizes, values, the seed). */
  readonly knobs: DictSpace
  /** The problem a configuration of the knobs makes. */
  readonly problem: (knobs: K) => AtomProblem
  /** Knobs of EP the atom opens on, over the method's defaults (`{ damping: 0.3 }` for an atom that needs it). */
  readonly ep?: Readonly<Record<string, number | string | boolean>>
}

/** A declared atom. */
export interface EpAtom<K extends Config = Config> extends AtomSpec<K> {
  /** The default knobs. */
  readonly defaults: K
}

/**
 * Declare an atom.
 *
 * @param spec - What the atom declares.
 * @returns The atom, with its default knobs.
 * @throws OmaiError with code `value` for an atom with no name, no factors, or a `planned` atom with no plan.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * [gaussianMeanAtom.name, gaussianMeanAtom.expected] // => ['gaussian-mean', 'runs']
 * ```
 */
export function defineAtom<K extends Config>(spec: AtomSpec<K>): EpAtom<K> {
  if (spec.name === '' || spec.factors.length === 0)
    throw new OmaiError('value', 'defineAtom: an atom has a name and the factors it exercises')
  if (spec.expected === 'planned' && (spec.plan === undefined || spec.plan === ''))
    throw new OmaiError('value', `defineAtom: the planned atom '${spec.name}' says where its rule is planned`)
  return { ...spec, defaults: defaults(spec.knobs) as K }
}

/** The belief of a coordinate: its mean and standard deviation, and whether it is a proper belief (not an unset message). */
export interface Belief {
  readonly mean: number
  readonly sd: number
  readonly proper: boolean
}

/**
 * The density of a belief on the support of its quantity, matched to its mean and standard deviation: a normal on the line,
 * a gamma on the positive axis, a beta on the unit interval (the families EP carries for those).
 *
 * @param support - Where the quantity lives.
 * @param mean - The mean.
 * @param sd - The standard deviation.
 * @returns The density.
 * @concept expectation-propagation
 * @index function The density of a belief from its mean and sd on a support (normal, gamma or beta).
 * @example
 * ```ts
 * import { beliefDensity } from 'omai-engine/bayesian-examples'
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

/** `log Gamma(x)` of a number. */
const lg = (x: number): number => item(lgamma(x))

/** A marginal as a curve with its moments. */
export interface RefMarginal {
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly x: readonly number[]
  readonly p: readonly number[]
}

/** The joint of two quantities on a grid, for contours. */
export interface RefJoint {
  readonly a: string
  readonly b: string
  readonly x: readonly number[]
  readonly y: readonly number[]
  readonly p: readonly (readonly number[])[]
}

/** The reference answer of an atom. */
export interface AtomReference {
  /** `exact` (closed form) or `grid` (quadrature of the posterior density of the model written as a program). */
  readonly kind: 'exact' | 'grid'
  readonly source: string
  readonly marginals: Readonly<Record<string, RefMarginal>>
  readonly joint?: RefJoint
  /** log p(data), when the grid knows it. */
  readonly logEvidence?: number
}

const CURVE_POINTS = 400

/** A marginal from a closed form: the density sampled on its interval. */
function exactCurve(name: string, support: AtomSupport, m: ExactMarginal): RefMarginal {
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
 * import { atomBound, gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * atomBound(gaussianMeanAtom.problem(gaussianMeanAtom.defaults)).sizes // => { measurement: 6 }
 * ```
 */
export function atomBound(problem: AtomProblem): BoundModel {
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
 * @index function The reference of an EP atom: closed-form or grid marginals as curves, and a joint for contours.
 * @example
 * ```ts
 * import { atomReference, gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * const r = atomReference(gaussianMeanAtom, gaussianMeanAtom.problem(gaussianMeanAtom.defaults))
 * [r.kind, Object.keys(r.marginals)] // => ['exact', ['mu']]
 * ```
 */
export function atomReference<K extends Config>(atom: EpAtom<K>, problem: AtomProblem): AtomReference {
  if (atom.reference === 'exact') {
    const marginals: Record<string, RefMarginal> = {}
    for (const q of problem.quantities) {
      const m = problem.exact?.[q.name]
      if (m === undefined)
        throw new OmaiError('value', `atomReference: '${atom.name}' has no closed form for '${q.name}'`)
      marginals[q.name] = exactCurve(q.name, q.support, m)
    }
    return { kind: 'exact', source: 'the closed-form posterior', marginals }
  }
  const grid = gridPosterior(toProgram(atomBound(problem)), undefined, { maxRefinements: 0, ...problem.grid })
  const marginals: Record<string, RefMarginal> = {}
  for (const q of problem.quantities) {
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
 * @index function Whether EP applies to an atom's model and the library's reason when it does not.
 * @example
 * ```ts
 * import { atomVerdict, gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * atomVerdict(gaussianMeanAtom.problem(gaussianMeanAtom.defaults)).applies // => true
 * ```
 */
export function atomVerdict(problem: AtomProblem): Verdict {
  const bound = atomBound(problem)
  const method = epSpecMethod(bound)
  return (
    method.check?.(analysisOf(bound)) ?? { applies: false, reason: 'the method has no check', missing: [] as string[] }
  )
}

/** The beliefs at one sweep. */
export interface SweepBeliefs {
  readonly sweep: number
  readonly beliefs: Readonly<Record<string, Belief>>
}

/** One quantity's agreement with the reference. */
export interface QuantityCheck {
  readonly name: string
  readonly label: string
  readonly ep: Belief
  readonly refMean: number
  readonly refSd: number
  /** The mean's error in reference standard deviations. */
  readonly meanZ: number
  readonly sdRatio: number
  /** The Kolmogorov-Smirnov distance between EP's belief (as its family's density) and the reference. */
  readonly ks: number
  readonly pass: boolean
}

/** The check of a run against the reference under a tolerance. */
export interface AtomCheck {
  readonly pass: boolean
  readonly rows: readonly QuantityCheck[]
  readonly tolerance: AtomTolerance
}

/** The result of running EP on an atom. */
export type AtomRun =
  | {
      readonly status: 'refused'
      /** The library's verdict: why EP does not apply. */
      readonly reason: string
      readonly missing: readonly string[]
    }
  | {
      readonly status: 'ran'
      readonly converged: boolean
      readonly sweeps: number
      /** The beliefs at the end. */
      readonly beliefs: Readonly<Record<string, Belief>>
      /** The beliefs at each sweep kept (the first and last always; thinned beyond 60). */
      readonly history: readonly SweepBeliefs[]
      /** The messages and beliefs on the model graph, step by step (factor granularity), for the model-run player. */
      readonly record: () => ModelRun
      readonly evidence?: number
    }

/**
 * The means of the reference's quantities, by name (what a data figure draws the exact answer from).
 *
 * @param reference - The reference.
 * @returns The means.
 * @example
 * ```ts
 * import { atomReference, gaussianMeanAtom, marginalMean } from 'omai-engine/bayesian-examples'
 * const r = atomReference(gaussianMeanAtom, gaussianMeanAtom.problem(gaussianMeanAtom.defaults))
 * Object.keys(marginalMean(r)) // => ['mu']
 * ```
 */
export function marginalMean(reference: AtomReference): Record<string, number> {
  return Object.fromEntries(Object.entries(reference.marginals).map(([k, m]) => [k, m.mean]))
}

/** Moments of a quantity (`name` or `site[i]`) in a posterior. */
function momentsOf(
  posterior: { mean: (s: string) => never; stddev: (s: string) => never; sites: readonly string[] },
  name: string,
): Belief {
  const m = /^(.*)\[(\d+)\]$/.exec(name)
  const site = m?.[1] ?? name
  const i = Number(m?.[2] ?? 0)
  if (!posterior.sites.includes(site)) return { mean: Number.NaN, sd: Number.NaN, proper: false }
  const mean = toFlat(posterior.mean(site))[i] ?? Number.NaN
  const sd = toFlat(posterior.stddev(site))[i] ?? Number.NaN
  return { mean, sd, proper: Number.isFinite(mean) && Number.isFinite(sd) && sd < 1e5 && sd > 0 }
}

/**
 * Run expectation propagation on an atom's problem, sweep by sweep, and keep the beliefs at each sweep and the messages as
 * a model-run record. A problem the library refuses is not run: the result carries the library's verdict.
 *
 * @param problem - The problem.
 * @param ep - The knobs of EP over its defaults (damping, schedule, tolerance, the sweep cap).
 * @returns The run, or the refusal.
 * @concept expectation-propagation
 * @index function Run EP on an atom's model: beliefs at each sweep, messages as a record, or the library's refusal.
 * @example
 * ```ts
 * import { atomRun, gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * const r = atomRun(gaussianMeanAtom.problem(gaussianMeanAtom.defaults))
 * r.status // => 'ran'
 * ```
 */
export function atomRun(problem: AtomProblem, ep: Config = {}): AtomRun {
  const bound = atomBound(problem)
  const method = epSpecMethod(bound)
  const verdict = atomVerdict(problem)
  if (!verdict.applies) return { status: 'refused', reason: verdict.reason, missing: verdict.missing }
  const config: Config = { tol: 1e-9, ...ep }
  const maxSweeps = typeof config['maxSweeps'] === 'number' ? config['maxSweeps'] : 200
  const s = session(method, { seed: 1, steps: maxSweeps + 2, mode: 'eager', config, logs: 'all' })
  const read = (state: never): Record<string, Belief> => {
    const posterior = method.posterior?.(state, s.config as never)
    if (posterior === undefined) return {}
    return Object.fromEntries(problem.quantities.map((q) => [q.name, momentsOf(posterior as never, q.name)] as const))
  }
  const last = s.t
  const keep = new Set<number>([0, last])
  const stride = Math.max(1, Math.ceil(last / 60))
  for (let t = 0; t <= last; t += stride) keep.add(t)
  const history = [...keep]
    .filter((t) => t >= 0 && t <= last)
    .sort((a, b) => a - b)
    .map((t): SweepBeliefs => ({ sweep: t, beliefs: read(s.stateAt(t) as never) }))
  const final = read(s.state as never)
  const posterior = method.posterior?.(s.state, s.config as never)
  const evidence = posterior?.logEvidence?.value
  return {
    status: 'ran',
    converged: s.status === 'converged',
    sweeps: last,
    beliefs: final,
    history,
    record: () =>
      modelRunFromTape(toStructure(bound, { view: 'factor' }), { run: s.run() }, { maxSteps: 120, maxMessages: 6 }),
    ...(evidence === undefined ? {} : { evidence }),
  }
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
 * @index function EP's beliefs against the reference: mean error in sds, sd ratio and KS, each against a tolerance.
 * @example
 * ```ts
 * import { atomCheck, atomReference, atomRun, gaussianMeanAtom } from 'omai-engine/bayesian-examples'
 * const p = gaussianMeanAtom.problem(gaussianMeanAtom.defaults)
 * const r = atomRun(p)
 * r.status === 'ran' && atomCheck(p, r.beliefs, atomReference(gaussianMeanAtom, p), gaussianMeanAtom.tolerance).pass // => true
 * ```
 */
export function atomCheck(
  problem: AtomProblem,
  beliefs: Readonly<Record<string, Belief>>,
  reference: AtomReference,
  tolerance: AtomTolerance,
): AtomCheck {
  const rows = problem.quantities.map((q): QuantityCheck => {
    const ep = beliefs[q.name] ?? { mean: Number.NaN, sd: Number.NaN, proper: false }
    const ref = reference.marginals[q.name] as RefMarginal
    const meanZ = Math.abs(ep.mean - ref.mean) / ref.sd
    const sdRatio = ep.sd / ref.sd
    const ks = ep.proper ? ksOf(beliefDensity(q.support, ep.mean, ep.sd), ref) : Number.NaN
    const pass =
      ep.proper &&
      meanZ <= tolerance.meanZ &&
      sdRatio >= tolerance.sdRatio[0] &&
      sdRatio <= tolerance.sdRatio[1] &&
      ks <= tolerance.ks
    return { name: q.name, label: q.label, ep, refMean: ref.mean, refSd: ref.sd, meanZ, sdRatio, ks, pass }
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
 * import { gaussianMeanAtom, verifyAtom } from 'omai-engine/bayesian-examples'
 * verifyAtom(gaussianMeanAtom).check?.pass // => true
 * ```
 */
export function verifyAtom<K extends Config>(
  atom: EpAtom<K>,
  knobs: Partial<K> = {},
): {
  readonly problem: AtomProblem
  readonly reference: AtomReference
  readonly run: AtomRun
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
