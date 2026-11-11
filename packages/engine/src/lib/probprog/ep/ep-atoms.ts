/**
 * The scaffold of the expectation propagation atoms: very small, concrete problems (six noisy measurements of a length,
 * a handful of labelled points, three players and their matches) each of which exercises one factor of the message
 * passing suite and has a reference answer that can be checked by eye. An atom declares a problem from a few knobs (the
 * model as a `ModelDefinition` with its data, the quantities it asks about with their human names, the picture of the
 * data), the reference (the exact posterior in closed form, or the posterior on a grid of the same model written as a
 * program, so it exists whether or not expectation propagation applies), a tolerance with its reason, and what is
 * expected (`runs`, the default, or `planned` for an atom the library refuses on purpose, with the verdict's reason).
 * `atomRun` compiles the model, asks the EP method for its verdict, runs it sweep by sweep and returns the beliefs at every
 * sweep, the messages as a model-run record and the check against the reference; `atomReference` is the reference alone
 * (marginal curves, and the joint of a pair for contours); `atomVerdict` is the compile-time verdict alone, which is all a
 * coverage table needs. A page of the proofs of concept is a composition of these, and a test generated from the registry
 * asserts every atom passes its check (and every `planned` one, none today, is refused with a reason).
 *
 * @index structure EP atom scaffold: a bare setup, its reference, a tolerance, the EP verdict and run.
 */

import { OmaiError } from '../../base/index.ts'
import { type ModelRun, modelRunFromTape } from '../../exposition/index.ts'
import type { DictSpace } from '../../space/index.ts'
import { defaults } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Config, type Session, session } from '../../trace/index.ts'
import {
  checkKnobKeys,
  type EpGridState,
  type EpKnobs,
  epRoute,
  epRunKnobs,
  epSpecMethod,
  type InferenceMethod,
  type ModelAnalysis,
  type Verdict,
} from '../inference/index.ts'
import {
  type BindOptions,
  type BoundModel,
  bind,
  type DataInput,
  doSpec,
  type ModelDefinition,
  toProgram,
  toStructure,
} from '../model/index.ts'
import type { Posterior } from '../posterior.ts'
import { type ExactMarginal, type GridOptions, gridPosterior, lg, type RefJoint } from '../validation/index.ts'

/** Where a quantity lives, which fixes the family its belief is drawn in. */
export type AtomSupport = 'real' | 'positive' | 'unit' | 'probability'

/** One number the atom asks about: its name in the model, what a person calls it, and the value that made the data. */
export interface AtomQuantity {
  /** The name of the coordinate in the posterior and the grid (`mu`, `skill_ana`, `w[0]`). */
  readonly name: string
  /** What it is, in words (`length`, `noise sd`, `skill of Ana`). */
  readonly label: string
  readonly support: AtomSupport
  /** The value the data were generated with, when there is one (drawn as a mark). */
  readonly truth?: number
  /**
   * A quantity that is not a site of the model but a function of EP's beliefs of others (the value a gate selects, from the
   * beliefs of its components and its switch): computed from the beliefs at each sweep, after the quantities before it.
   */
  readonly derive?: (beliefs: Readonly<Record<string, Belief>>) => Belief
  /** Checked but not drawn on its own (a component of a gate, which is drawn through the gate's branches). */
  readonly hidden?: boolean
  /**
   * A label shared by the members of a group (the skills of the players, the rates of the groups): the calibration table
   * pools them in one row, so a population of values is one line instead of one per member.
   */
  readonly pool?: string
}

/** One component of a mixture: what it weighs, and its Gaussian. */
export interface MixtureBranch {
  readonly label: string
  readonly weight: number
  readonly mean: number
  readonly sd: number
}

/**
 * A mixture drawn as the gate it is: the exact branches (each its Gaussian scaled by its posterior weight), their sum, the
 * branches as EP's beliefs imply them (dashed) and the single Gaussian EP's belief is, the moment-matched projection of
 * the mixture.
 */
export interface MixtureView {
  readonly title: string
  /** The quantity whose belief this replaces on the page (its single Gaussian is drawn here, as the projection of the mixture). */
  readonly quantity: string
  /** The axis label (the quantity the mixture is of). */
  readonly label: string
  /** The exact branches, with their posterior weights. */
  readonly exact: readonly MixtureBranch[]
  /** The branches EP's beliefs imply at a sweep (its weights are EP's switch probabilities), or undefined when they cannot be told apart. */
  readonly ep: (beliefs: Readonly<Record<string, Belief>>) => readonly MixtureBranch[] | undefined
}

/** A set of points of the data figure. */
export interface FigurePoints {
  readonly name: string
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** A group per point (a class, an outlier flag), with the name of each group. */
  readonly group?: readonly number[]
  readonly groupNames?: Readonly<Record<number, string>>
  /** The groups from the means of the quantities (a gate's assignment from its probabilities), instead of `group`, when there are means to draw. */
  readonly groupFrom?: (means: Readonly<Record<string, number>>) => readonly number[]
}

/** A curve of the data figure. */
export interface FigureCurve {
  readonly name: string
  /** A colour token (`tab1`) for a curve that is not the answer; default a muted line. */
  readonly colour?: string
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

/** What an atom builds from its knobs: the model, its data and the quantities it asks about. All the run needs. */
export interface AtomBuild {
  readonly definition: ModelDefinition
  readonly data: DataInput
  readonly bindOptions?: BindOptions
  /** The quantities checked and drawn, in order (at most three coordinates when the reference is a grid). */
  readonly quantities: readonly AtomQuantity[]
}

/** The reference answer an atom brings (for `exact` and `grid` references), written by hand and independent of the rule under test. */
export interface AtomReferenceData {
  /** The closed-form marginals, by quantity name (when the atom's reference is `exact`). */
  readonly exact?: Readonly<Record<string, ExactMarginal>>
  /** The closed-form joint of the atom's pair (for contours), when the reference is `exact`. */
  readonly exactJoint?: RefJoint
  /** What the `exact` marginals are, when they are not a closed form (`quadrature of the four unknowns`, `a Gibbs sampler`). */
  readonly referenceSource?: string
  /** Options of the grid (points per axis, an explicit box) when the reference is a grid. */
  readonly grid?: GridOptions
}

/** What a page shows of an atom besides its answer: the model in words, the picture of the data, the pair drawn as contours and the mixtures. */
export interface AtomPresentation {
  /** Two quantities whose joint posterior is drawn as contours, when it is worth seeing. */
  readonly pair?: readonly [string, string]
  /** The model written out, one line each, with its numbers (`x ~ N(0, 1)`, `g = exp(x)`, `y ~ N(g, 0.3²)`, `observe y = 2.5`). */
  readonly equations: readonly string[]
  /** A picture of the data, when there is data worth a picture (points in the plane); most setups have none. */
  readonly figure?: AtomFigure
  /** Mixtures to draw as their branches (a gate's selected value, a component that may or may not be selected). */
  readonly mixtures?: readonly MixtureView[]
}

/** What an atom's knobs make, in its three parts: what is built and run, the reference it is checked against, and what a page shows. */
export interface AtomParts {
  readonly build: AtomBuild
  /** Absent for a `truth` atom (no reference posterior) and for a `grid` atom that needs no option. */
  readonly reference?: AtomReferenceData
  readonly present: AtomPresentation
}

/**
 * What an atom's knobs make, composed: the three parts in one record, which is what the runs, the references and the pages
 * read. A part's lazy fields (a reference grid that a calibration over many seeds never reads) stay lazy.
 */
export type AtomProblem = AtomBuild & AtomReferenceData & AtomPresentation

/**
 * The three parts of an atom as one problem record, field by field (a getter stays a getter, so a reference that is computed
 * when first read is not computed here).
 *
 * @param parts - The parts.
 * @returns The composed problem.
 * @example
 * ```ts
 * import { composeProblem, gaussianAtom } from 'omai-engine/probprog/ep'
 * Object.keys(composeProblem(gaussianAtom.parts(gaussianAtom.defaults))).includes('equations') // => true
 * ```
 */
export function composeProblem(parts: AtomParts): AtomProblem {
  const out = {}
  for (const part of [parts.build, parts.reference ?? {}, parts.present])
    Object.defineProperties(out, Object.getOwnPropertyDescriptors(part))
  return out as AtomProblem
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

/**
 * What an extended atom promises about recovering its ground truth: over `seeds` simulated data sets, the share of runs in
 * which the true value falls inside EP's 90% interval is at least `coverage` for every parameter (it should be near 0.9: the
 * floor leaves room for the finite number of seeds and for EP's approximation), and the error of the mean is at most `zrms`
 * posterior sds (root mean square; near 1 for a calibrated posterior).
 */
export interface AtomRecoverySpec {
  readonly seeds: number
  readonly coverage: number
  readonly zrms: number
  /** Why these floors (the number of seeds, what EP approximates, what the prior brings). */
  readonly reason: string
}

/** The families of the suite (the files of this folder, one each, and the rows of the coverage table). */
export type AtomFamily =
  | 'conjugate'
  | 'priors'
  | 'scales'
  | 'likelihoods'
  | 'gates'
  | 'gate weights'
  | 'functions'
  | 'kernels'
  | 'covariance'
  | 'nonparametric'
  | 'counts'

/** What an atom declares. */
export interface AtomSpec<K extends Config> {
  /** The slug (`gaussian-mean`), the name of the page `ep-<name>`. */
  readonly name: string
  readonly title: string
  /** One sentence of what the setup is, in the factor's own terms (no story). */
  readonly what: string
  /** The factors or rules it exercises, by the library's names (`gaussian-gamma-precision`, `probit`). */
  readonly factors: readonly string[]
  /** The family of the atom, for the coverage table's grouping. */
  readonly family: AtomFamily
  /**
   * `runs` (the default: every atom of the suite runs under EP, and the registry test fails when one does not) or `planned`, the
   * escape hatch for an atom the library refuses on purpose (the verdict is asked at run time either way). Nothing is planned today.
   */
  readonly expected?: 'runs' | 'planned'
  /** For a `planned` atom, what the missing rule is and where it is described (`epGaps`). */
  readonly plan?: string
  /**
   * `exact` (closed form), `grid` (quadrature of the model), or `truth`: no reference posterior exists to compare with (a model with
   * many latents), and the check is recovery of the true values alone (an extended atom's calibration).
   */
  readonly reference: 'exact' | 'grid' | 'truth'
  readonly tolerance: AtomTolerance
  /** The knobs of the problem (sizes, values, the seed). */
  readonly knobs: DictSpace
  /** What a configuration of the knobs makes: the build, the reference and the presentation. */
  readonly parts: (knobs: K) => AtomParts
  /** Knobs of EP the atom opens on, over the method's defaults (`{ damping: 0.3 }` for an atom that needs it); a key that is not a knob is refused. */
  readonly ep?: EpKnobs
  /**
   * The name of the basic atom this one extends: the same setup in which the numbers the basic atom fixes are unknown,
   * with priors, and the data are simulated from a stated truth (knobs: the true values and a `seed`). Set together with
   * `recovery`; every quantity of the problem carries its `truth`.
   */
  readonly extends?: string
  /** What recovering the truth means for an extended atom (see `atomCalibration`). */
  readonly recovery?: AtomRecoverySpec
}

/** A declared atom. */
export interface EpAtom<K extends Config = Config> extends Omit<AtomSpec<K>, 'expected'> {
  /** What the atom promises: `runs` unless it was declared `planned`. */
  readonly expected: 'runs' | 'planned'
  /** The problem a configuration of the knobs makes, the three parts composed. */
  readonly problem: (knobs: K) => AtomProblem
  /** The default knobs. */
  readonly defaults: K
}

/**
 * Declare an atom.
 *
 * @param spec - What the atom declares.
 * @returns The atom, with its default knobs.
 * @throws OmaiError with code `value` for an atom with no name, no factors, a `planned` atom with no plan, one that
 * gives `extends` without `recovery` (or the reverse), or an `ep` with a key that is not an EP knob.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { gaussianAtom } from 'omai-engine/probprog/ep'
 * [gaussianAtom.name, gaussianAtom.expected] // => ['gaussian', 'runs']
 * ```
 */
export function defineAtom<K extends Config>(spec: AtomSpec<K>): EpAtom<K> {
  if (spec.name === '' || spec.factors.length === 0)
    throw new OmaiError('value', 'defineAtom: an atom has a name and the factors it exercises')
  if (spec.expected === 'planned' && (spec.plan === undefined || spec.plan === ''))
    throw new OmaiError('value', `defineAtom: the planned atom '${spec.name}' says where its rule is planned`)
  if (spec.ep !== undefined) checkKnobKeys(`defineAtom: '${spec.name}'`, spec.ep, epRunKnobs('ep'))
  if ((spec.extends === undefined) !== (spec.recovery === undefined))
    throw new OmaiError('value', `defineAtom: '${spec.name}' gives both 'extends' and 'recovery' or neither`)
  return {
    ...spec,
    expected: spec.expected ?? 'runs',
    problem: (knobs) => composeProblem(spec.parts(knobs)),
    defaults: defaults(spec.knobs) as K,
  }
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

/** A marginal as a curve with its moments. */
export interface RefMarginal {
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly x: readonly number[]
  readonly p: readonly number[]
}

/** The reference answer of an atom. */
export interface AtomReference {
  /** `exact` (closed form), `grid` (quadrature of the posterior density of the model written as a program) or `truth` (no marginals). */
  readonly kind: 'exact' | 'grid' | 'truth'
  readonly source: string
  readonly marginals: Readonly<Record<string, RefMarginal>>
  readonly joint?: RefJoint
  /** log p(data), when the grid knows it. */
  readonly logEvidence?: number
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
 * import { atomBound, gaussianAtom } from 'omai-engine/probprog/ep'
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
 * import { atomReference, gaussianAtom } from 'omai-engine/probprog/ep'
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
 * import { atomVerdict, gaussianAtom } from 'omai-engine/probprog/ep'
 * atomVerdict(gaussianAtom.problem(gaussianAtom.defaults)).applies // => true
 * ```
 */
export function atomVerdict(problem: AtomBuild): Verdict {
  return epRoute(atomBound(problem)).verdict
}

/** The messages of EP at the most probable node of a grid run: the model conditioned on the hyperparameters there, run as a plain EP session. */
function gridRecord(bound: BoundModel, state: EpGridState): ModelRun {
  let best = state.nodes[0]
  for (const n of state.nodes)
    if (best === undefined || n.logEvidence + n.logPrior > best.logEvidence + best.logPrior) best = n
  const values = Object.fromEntries(state.axes.map((a, k) => [a.name, best?.x[k] ?? 0]))
  const cond = bind({ spec: doSpec(bound.spec, values), customs: bound.customs }, bound.data, bound.options)
  const s = session(epSpecMethod(cond), { seed: 1, steps: 202, mode: 'eager', config: { tol: 1e-9 } })
  return modelRunFromTape(toStructure(cond, { view: 'factor' }), { run: s.run() }, { maxSteps: 120, maxMessages: 6 })
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
      /** EP applies (the compiler accepts the model) but the run broke: the error, which is a gap of the library to report. */
      readonly status: 'failed'
      readonly reason: string
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
 * import { atomReference, gaussianAtom, marginalMean } from 'omai-engine/probprog/ep'
 * const r = atomReference(gaussianAtom, gaussianAtom.problem(gaussianAtom.defaults))
 * Object.keys(marginalMean(r)) // => ['mu']
 * ```
 */
export function marginalMean(reference: AtomReference): Record<string, number> {
  return Object.fromEntries(Object.entries(reference.marginals).map(([k, m]) => [k, m.mean]))
}

/** Moments of a quantity (`name` or `site[i]`) in a posterior. */
function momentsOf(
  posterior: Pick<Posterior, 'mean' | 'stddev' | 'sites'>,
  name: string,
  support: AtomSupport,
): Belief {
  const m = /^(.*)\[(\d+)\]$/.exec(name)
  const site = m?.[1] ?? name
  const i = Number(m?.[2] ?? 0)
  if (!posterior.sites.includes(site)) return { mean: Number.NaN, sd: Number.NaN, proper: false }
  const mean = toFlat(posterior.mean(site))[i] ?? Number.NaN
  const sd = toFlat(posterior.stddev(site))[i] ?? Number.NaN
  return {
    mean,
    sd,
    proper: Number.isFinite(mean) && Number.isFinite(sd) && sd < 1e5 && (sd > 0 || support === 'probability'),
  }
}

/**
 * The beliefs of an atom's quantities in a posterior: the mean and standard deviation of each (a derived quantity from the
 * beliefs before it), and whether each is a proper belief. What a run of any message-passing method reads at a sweep.
 *
 * @param problem - The problem (the quantities and their supports).
 * @param posterior - The posterior of a message-passing state.
 * @returns The beliefs by quantity name.
 * @example
 * ```ts
 * import { atomBeliefs, atomBound, gaussianAtom } from 'omai-engine/probprog/ep'
 * import { epSpecMethod } from 'omai-engine/probprog/inference'
 * import { session } from 'omai-engine/trace'
 * const problem = gaussianAtom.problem(gaussianAtom.defaults)
 * const method = epSpecMethod(atomBound(problem))
 * const s = session(method, { steps: 50, mode: 'eager' })
 * const posterior = method.posterior?.(s.state, s.config)
 * posterior === undefined ? false : atomBeliefs(problem, posterior).mu?.proper // => true
 * ```
 */
export function atomBeliefs(
  problem: Pick<AtomBuild, 'quantities'>,
  posterior: Pick<Posterior, 'mean' | 'stddev' | 'sites'>,
): Record<string, Belief> {
  const out: Record<string, Belief> = {}
  for (const q of problem.quantities) {
    const b = q.derive === undefined ? momentsOf(posterior, q.name, q.support) : q.derive(out)
    const spread = q.support === 'probability' ? b.sd >= 0 : b.sd > 0
    out[q.name] = { ...b, proper: b.proper && Number.isFinite(b.mean) && Number.isFinite(b.sd) && spread }
  }
  return out
}

/**
 * Run expectation propagation on an atom's problem, sweep by sweep, and keep the beliefs at each sweep and the messages as
 * a model-run record. A problem the library refuses is not run: the result carries the library's verdict.
 *
 * @param problem - The problem.
 * @param ep - The knobs of EP over its defaults (damping, schedule, tolerance, the sweep cap, the power).
 * @param options - `light` keeps no tape and no per-sweep beliefs (only the final ones): what a calibration over many seeds needs.
 * @returns The run, or the refusal.
 * @throws OmaiError with code `value` for a key of `ep` that is not a knob of the method (a misspelled knob would run on the defaults).
 * @concept expectation-propagation
 * @index function Run EP on an atom's model: beliefs per sweep, a message record, or the refusal.
 * @example
 * ```ts
 * import { atomRun, gaussianAtom } from 'omai-engine/probprog/ep'
 * const r = atomRun(gaussianAtom.problem(gaussianAtom.defaults))
 * r.status // => 'ran'
 * ```
 */
export function atomRun(problem: AtomBuild, ep: Config = {}, options: { readonly light?: boolean } = {}): AtomRun {
  // region inference
  const bound = problem.definition.bind(problem.data, problem.bindOptions)
  // compile to a factor graph and run EP on it: a rule for every factor
  // (a covariance that is a custom function of a random value makes it a grid over that value)
  const route = epRoute(bound)
  // endregion
  // a misspelled knob is an error, not a run on the defaults
  checkKnobKeys('atomRun', ep, route.method?.config ?? epRunKnobs('ep'))
  if (route.via === 'none') return { status: 'refused', reason: route.verdict.reason, missing: route.verdict.missing }
  const config: Config = { tol: 1e-9, ...ep }
  const maxSweeps = typeof config['maxSweeps'] === 'number' ? config['maxSweeps'] : 200
  const logs = options.light === true ? 'off' : 'all'
  const run = { problem, config, logs, light: options.light === true } as const
  return route.via === 'ep'
    ? runRoute(run, route.method, maxSweeps + 2, (s) => ({
        converged: s.status === 'converged',
        record: () =>
          modelRunFromTape(toStructure(bound, { view: 'factor' }), { run: s.run() }, { maxSteps: 120, maxMessages: 6 }),
      }))
    : runRoute<EpGridState>(run, route.method, 1000, (s, posterior) => ({
        converged: posterior?.status === 'converged',
        record: () => gridRecord(bound, s.state),
      }))
}

/** How a run of one route is read: whether it converged and the messages as a model-run record. */
interface RouteReading {
  readonly converged: boolean
  readonly record: () => ModelRun
}

/** Run the method of a route sweep by sweep (one step of the grid method is one EP run) and read the beliefs at the sweeps kept. */
function runRoute<S>(
  run: {
    readonly problem: AtomBuild
    readonly config: Config
    readonly logs: 'off' | 'all'
    readonly light: boolean
  },
  method: InferenceMethod<S, Config, string, ModelAnalysis>,
  steps: number,
  reading: (s: Session<S, Config, string>, posterior: Posterior | undefined) => RouteReading,
): AtomRun {
  const { problem, config, logs, light } = run
  let s: Session<S, Config, string>
  try {
    // region inference
    // one step is one sweep of EP over the factors, recorded
    s = session(method, { seed: 1, steps, mode: 'eager', config, logs })
    // endregion
  } catch (e) {
    return { status: 'failed', reason: e instanceof Error ? e.message : String(e) }
  }
  const read = (state: S): Record<string, Belief> => {
    // region inference
    // the beliefs by site
    const posterior = method.posterior?.(state, s.config)
    // endregion
    return posterior === undefined ? {} : atomBeliefs(problem, posterior)
  }
  const last = s.t
  const keep = new Set<number>(light ? [last] : [0, last])
  const stride = Math.max(1, Math.ceil(last / 60))
  if (!light) for (let t = 0; t <= last; t += stride) keep.add(t)
  const history = [...keep]
    .filter((t) => t >= 0 && t <= last)
    .sort((a, b) => a - b)
    .map((t): SweepBeliefs => ({ sweep: t, beliefs: read(s.stateAt(t)) }))
  const final = read(s.state)
  const posterior = method.posterior?.(s.state, s.config)
  const evidence = posterior?.logEvidence?.value
  const { converged, record } = reading(s, posterior)
  return {
    status: 'ran',
    converged,
    sweeps: last,
    beliefs: final,
    history,
    record,
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
 * @index function EP's beliefs against the reference: mean error, sd ratio, KS against a tolerance.
 * @example
 * ```ts
 * import { atomCheck, atomReference, atomRun, gaussianAtom } from 'omai-engine/probprog/ep'
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
 * import { gaussianAtom, verifyAtom } from 'omai-engine/probprog/ep'
 * verifyAtom(gaussianAtom).check?.pass // => true
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
