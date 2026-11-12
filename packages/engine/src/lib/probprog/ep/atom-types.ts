/**
 * The records of the expectation propagation atoms: a declared atom (`EpAtom`), what its knobs make (`AtomProblem`, composed of
 * the build, the reference and the presentation), the reference answer and the check of a run against it, and the result of a
 * run (`EpRun`). Types only; `defineAtom` (authoring) is internal to the folder and the functions are in `ep-atoms.ts` and `run.ts`.
 */

import type { ModelRun } from '../../exposition/index.ts'
import type { DictSpace } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import type { EpKnobs } from '../inference/index.ts'
import type { BindOptions, DataInput, ModelDefinition } from '../model/index.ts'
import type { ExactMarginal, GridOptions, RefJoint } from '../validation/index.ts'

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
  /**
   * Run EP inside a grid over these scalar latents (`over`, with the grid method's knobs) instead of the route plain EP would take: a
   * model that plain EP accepts but whose answer for a parameter is a single narrow Gaussian (a persistence that multiplies a chain's
   * state) is run on a grid of that parameter, each node weighted by its EP evidence.
   */
  readonly hyperGrid?: { readonly over: readonly string[]; readonly knobs?: Config }
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
  /** log p(data) of the exact reference with every constant of the model in it, when the reference knows it (a lazy field: read only by what compares evidence). */
  readonly exactLogEvidence?: number
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
  | 'composed'
/** The belief of a coordinate: its mean and standard deviation, and whether it is a proper belief (not an unset message). */
export interface Belief {
  readonly mean: number
  readonly sd: number
  readonly proper: boolean
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
export type EpRun =
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
/** A declared atom: what it declares, its default knobs and the problem its knobs make. */
export interface EpAtom<K extends Config = Config> {
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
   * `runs` (every atom of the suite runs under EP, and the registry test fails when one does not) or `planned`, the
   * escape hatch for an atom the library refuses on purpose (the verdict is asked at run time either way). Nothing is planned today.
   */
  readonly expected: 'runs' | 'planned'
  /** For a `planned` atom, what the missing rule is and where it is described (`epGaps`). */
  readonly plan?: string
  /**
   * `exact` (closed form), `grid` (quadrature of the model), or `truth`: no reference posterior exists to compare with (a model with
   * many latents), and the check is recovery of the true values alone (an extended atom's calibration).
   */
  readonly reference: 'exact' | 'grid' | 'truth'
  readonly tolerance: AtomTolerance
  /**
   * How close EP's log evidence is to the grid evidence of the same model, in nats, where the grid is fast enough to test it, and why
   * it is not tighter (the tests of `ep-evidence.test.ts` read it).
   */
  readonly evidence?: { readonly nats: number; readonly reason: string }
  /** The knobs of the problem (sizes, values, the seed). */
  readonly knobs: DictSpace
  /** What a configuration of the knobs makes: the build, the reference and the presentation. */
  readonly parts: (knobs: K) => AtomParts
  /** The problem a configuration of the knobs makes, the three parts composed. */
  readonly problem: (knobs: K) => AtomProblem
  /** The default knobs. */
  readonly defaults: K
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
