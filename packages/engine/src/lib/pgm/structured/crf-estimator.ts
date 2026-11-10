/**
 * The linear-chain CRF as an estimator, in the shape of sklearn-crfsuite's `CRF`: hyperparameters named as CRFsuite's
 * (`algorithm`, `c1`, `c2`, `minFreq`, `allPossibleStates`, `allPossibleTransitions`, `epsilon`, `period`, `delta`),
 * `fit(X, y)` over sequences of feature dictionaries (or lists of attribute names, or a bare name) or over raw inputs
 * with a feature extractor of `features.ts` (CRF++, Wapiti and CRFsuite templates, a function of your own, hashing),
 * and a fitted value with `predict`, `predictMarginals`, `score` (flat token accuracy, as sklearn-crfsuite),
 * `stateFeatures` and `transitionFeatures` (the weights as dictionaries, as `state_features_` and
 * `transition_features_`), the sequence score and log-likelihood, and the top features. Underneath it is the iterative
 * base of `learn/bases`: the fit is a traced run of the solver slot (`crfEstimatorSolvers`: L-BFGS, OWL-QN for L1, every
 * smooth solver, the perceptron family and the structured SVMs), so `method` and `session` give the training tape
 * (objective, gradient, sequence error, nonzero weights) and `fittedAt(t)` the model at any step. CRFsuite's feature
 * generation is reproduced by a mask of the weights that stay at zero (a state or transition feature that never
 * occurs, or occurs less than `minFreq` times), so the weights equal CRFsuite's for the same penalties. The estimator
 * protocol of `learn/estimator.ts` is tensors in and tensors out; sequences of dictionaries are not tensors, so this
 * estimator has the protocol's shape (`name`, `hyper`, `fit`, `revive`, `with`, `method`, `session`) over its own data
 * type (`crfData`) and its own verbs, and the `CRF` class wraps it for `new CRF(...).fit(X, y).predict(X)` code.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import type { Exposition } from '../../exposition/index.ts'
import {
  type FitOptions,
  fittedSession,
  type IterativeEstimator,
  type IterativePath,
  type IterativeSession,
  type IterativeSessionOptions,
  iterative,
  type LearnData,
  type SerialisedFitted,
  type Verbs,
} from '../../learn/index.ts'
import {
  familySolvers,
  owlqnMode,
  type SmoothProblem,
  type SolverReached,
  slotAlgorithm,
  type VectorParams,
} from '../../optim/index.ts'
import type { Key } from '../../random/index.ts'
import { type DictSpace, dict, real, type SpaceValue } from '../../space/index.ts'
import { asTensor, mul, tensor, type Value } from '../../tensor/index.ts'
import { featureRecord, type ItemAttributes } from '../../text/index.ts'
import {
  type Algorithm,
  type Config,
  Log,
  type Method,
  type ModeTable,
  type Run,
  type Session,
} from '../../trace/index.ts'
import type { ChainInput, ChainLabels } from './chain-oracle.ts'
import {
  type CrfInput,
  type CrfObservation,
  crfDecode,
  crfLogLikelihood,
  crfMarginals,
  crfScore,
  type LinearChainCrf,
  linearChainCrf,
} from './crf.ts'
import { type CrfTrainState, crfStructuredProblem } from './crf-training.ts'
import {
  type CrfFeatureSummary,
  crfModelFromInputs,
  crfTemplateWeights,
  crfTopFeatures,
  type FeatureExtractor,
  type TemplateWeight,
  type TopFeaturesOptions,
} from './features.ts'
import { structuredSmoothProblem } from './objective.ts'
import type { StructuredOracle, StructuredProblem } from './oracle.ts'
import { overStructuredProblem, structuredModes, structuredTrainMetrics, structuredTrainView } from './solvers.ts'
import { structuredSvmModes } from './structured-svm.ts'

// ---------------------------------------------------------------------------------------------------------------------
// Inputs

/**
 * One item of a sequence in the forms sklearn-crfsuite (python-crfsuite) accepts: a dictionary from attribute names to
 * values (a number, a boolean, or a string, which makes the attribute `name:value` with value 1), a list of attribute
 * names (each with value 1), or one attribute name.
 */
export type CrfItem = string | readonly string[] | Readonly<Record<string, number | boolean | string>>

/** A sequence for the estimator without an extractor: its items, or a ready observation (item and edge attributes). */
export type CrfSequence = readonly CrfItem[] | CrfObservation

/**
 * One item as the attribute record a CRF scores: a name is one attribute, a list counts its names, a dictionary keeps
 * numbers, reads `true` as 1 and `false` as 0, and turns a string value into the attribute `name:value`. Attributes of
 * value 0 are dropped (they take no part in a model).
 *
 * @param item - The item in any of the accepted forms.
 * @returns The attribute record.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfItem } from 'omai-engine/pgm'
 * crfItem({ w: 'cat', n: 2.5, up: true, low: false }) // => { 'w:cat': 1, n: 2.5, up: 1 }
 * ```
 */
export function crfItem(item: CrfItem): ItemAttributes {
  if (typeof item === 'string') return { [item]: 1 }
  if (Array.isArray(item)) return featureRecord(item as readonly string[])
  const out: Record<string, number> = {}
  for (const [name, value] of Object.entries(item as Readonly<Record<string, number | boolean | string>>)) {
    if (typeof value === 'string') out[`${name}:${value}`] = 1
    else {
      const v = typeof value === 'boolean' ? (value ? 1 : 0) : value
      if (v !== 0) out[name] = v
    }
  }
  return out
}

/**
 * The observation of a sequence of items in any accepted form (a ready `CrfObservation` passes through).
 *
 * @param sequence - The items.
 * @returns The CRF input.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfSequence } from 'omai-engine/pgm'
 * crfSequence([['a', 'b'], 'c', { d: 2 }]) // => [{ a: 1, b: 1 }, { c: 1 }, { d: 2 }]
 * ```
 */
export function crfSequence(sequence: CrfSequence): CrfInput {
  return 'unigram' in sequence ? sequence : (sequence as readonly CrfItem[]).map(crfItem)
}

/**
 * What the CRF estimator fits: sequences (feature dictionaries, or raw inputs when the estimator has a `features`
 * extractor) and their label sequences. It is a `LearnData` whose `x` is the sequence numbers, so the data goes through
 * the same `fit` of the estimator protocol; the sequences and labels ride beside it.
 */
export interface CrfData<I = CrfSequence> extends LearnData {
  /** The sequences. */
  readonly sequences: readonly I[]
  /** The labels of each sequence (omitted for data to predict on). */
  readonly labels?: readonly (readonly string[])[]
}

/**
 * Data for the CRF estimator.
 *
 * @param sequences - The sequences: lists of feature dictionaries, or raw inputs for an estimator with `features`.
 * @param labels - The label sequences, one label per item.
 * @returns The data.
 * @throws OmaiError with code `shape` for a different number of sequences and label lists.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { crfData } from 'omai-engine/pgm'
 * crfData([[['a'], ['b']]], [['X', 'Y']]).labels // => [['X', 'Y']]
 * ```
 */
export function crfData<I = CrfSequence>(sequences: readonly I[], labels?: readonly (readonly string[])[]): CrfData<I> {
  if (labels !== undefined && labels.length !== sequences.length)
    throw new OmaiError('shape', `crfData: ${sequences.length} sequences and ${labels.length} label lists`)
  return { x: asTensor(sequences.map((_, i) => i)), sequences, ...(labels === undefined ? {} : { labels }) }
}

// ---------------------------------------------------------------------------------------------------------------------
// Hyperparameters

/** The training algorithms by their sklearn-crfsuite names, and the library's own solver names. */
export type CrfAlgorithm =
  | 'lbfgs'
  | 'l2sgd'
  | 'ap'
  | 'pa'
  | 'arow'
  | 'owlqn'
  | 'perceptron'
  | 'averaged'
  | 'passive-aggressive'
  | 'subgradient'
  | 'bcfw'
  | 'one-slack'
  | 'n-slack'
  | 'gradientDescent'
  | 'conjugateGradient'
  | 'newton'
  | 'bfgs'
  | 'coordinateDescent'
  | 'gradient'

/** How a fitted CRF decodes a sequence (default Viterbi): the model and the observation to a label sequence. */
export type CrfDecoder = (crf: LinearChainCrf, x: CrfInput) => readonly string[]

/**
 * The hyperparameters of {@link crf}: sklearn-crfsuite's, under the same names in camelCase, and what the library adds.
 */
export interface CrfHyper<I = CrfSequence> {
  /**
   * A feature extractor of `features.ts` (CRF++, Wapiti, CRFsuite templates, a function, hashing): the sequences are
   * then raw inputs of the extractor (token tables) rather than feature dictionaries. The extractor also says whether
   * the model has label-pair transitions (CRFsuite templates) or gets them from a bigram line (CRF++, Wapiti).
   */
  readonly features?: FeatureExtractor<I>
  /**
   * The training algorithm: `lbfgs` (default; OWL-QN when `c1` > 0, as CRFsuite), `l2sgd` (stochastic gradient on the L2
   * objective), `ap` (averaged perceptron), `pa` (passive-aggressive), `arow`, or any mode of `crfEstimatorSolvers`
   * (`owlqn`, `perceptron`, `averaged`, `passive-aggressive`, `bfgs`, `newton`, the structured SVMs `bcfw`,
   * `subgradient`, `one-slack`, `n-slack`, ...).
   */
  readonly algorithm: CrfAlgorithm
  /** A full value of `crfEstimatorSolvers.space` (`{ kind: 'lbfgs', memory: 6 }`), which overrides `algorithm`. */
  readonly solver?: SpaceValue
  /** The L1 coefficient: the objective adds c1 ||w||_1 (default 0). */
  readonly c1: number
  /** The L2 coefficient: the objective adds c2 ||w||^2 (default 1, as CRFsuite). */
  readonly c2: number
  /** The most iterations (epochs for the online algorithms); default 10000 for the batch solvers, 1000 for `l2sgd`, 100 for the online ones. */
  readonly maxIterations?: number
  /**
   * CRFsuite's feature cut-off: a state, transition or bigram feature is kept when its (value-weighted) number of
   * occurrences in the training data is at least this (default 0: every feature that occurs; with
   * `allPossibleStates` and a cut-off above 0, features that never occur are dropped too, as in CRFsuite).
   */
  readonly minFreq: number
  /** The CRF++ cut-off on attributes (`-f`): attributes seen fewer times than this are not in the vocabulary (default 1). */
  readonly minAttributeFrequency?: number
  /** Generate state features that never occur in the training data (negative state features; default false). */
  readonly allPossibleStates: boolean
  /** Generate transition features for every label pair, seen or not (default false). */
  readonly allPossibleTransitions: boolean
  /** Whether the model has plain label-pair transitions (default: what the extractor asks for, or true). */
  readonly transitions?: boolean
  /** The convergence tolerance on the gradient, the solver's `gradTol` unless the solver is given one (default 1e-5). */
  readonly epsilon: number
  /** The stopping threshold: stop when the relative improvement of the objective over the last `period` iterations is below it (default 1e-5; 0 turns it off). */
  readonly delta: number
  /** The iterations over which the improvement is tested (default 10). */
  readonly period: number
  /** The aggressiveness of `pa` (default 1). */
  readonly c: number
  /** The initial variance of every weight and the loss trade-off of `arow` (default 1 and 1). */
  readonly variance: number
  readonly gamma: number
  /** Fix the labels and their order (default: the labels of the training data in order of first appearance, as CRFsuite). */
  readonly labels?: readonly string[]
  /**
   * How `predict` decodes (default Viterbi). Constrained or n-best decoding plugs in here as a function of the model
   * and the observation.
   */
  readonly decoder?: CrfDecoder
  /**
   * Keep every channel of the training tape (the weights at each step included) instead of the scalar curves that are
   * kept anyway (objective, gradient, loss, sequence error, nonzero weights, mistakes). Default false.
   */
  readonly verbose?: boolean
}

const defaults: CrfHyper<never> = {
  algorithm: 'lbfgs',
  c1: 0,
  c2: 1,
  minFreq: 0,
  allPossibleStates: false,
  allPossibleTransitions: false,
  epsilon: 1e-5,
  delta: 1e-5,
  period: 10,
  c: 1,
  variance: 1,
  gamma: 1,
}

// ---------------------------------------------------------------------------------------------------------------------
// The solver slot

/**
 * What the estimator trains: the CRF training problem over the full parameter vector, the same with the oracle masked
 * to the features CRFsuite would generate (for the modes that work from the oracle), and the 0/1 mask itself.
 */
export interface CrfEstimatorProblem {
  readonly full: StructuredProblem<ChainInput, ChainLabels>
  readonly masked: StructuredProblem<ChainInput, ChainLabels>
  /** 1 for a feature that exists, 0 for one that stays at zero; absent when every feature exists. */
  readonly mask?: Float64Array
}

/** The oracle with the weights outside the mask fixed at zero: their feature counts and expected counts are zeroed. */
function maskedOracle<X, Y>(oracle: StructuredOracle<X, Y>, mask: Float64Array): StructuredOracle<X, Y> {
  const keep = (v: Float64Array): Float64Array => Float64Array.from(v, (value, q) => value * (mask[q] ?? 0))
  return {
    size: oracle.size,
    features: (x, y) => keep(oracle.features(x, y)),
    expectations: (w, x) => {
      const free = oracle.expectations(w, x)
      return { counts: keep(free.counts), logPartition: free.logPartition }
    },
    decode: oracle.decode,
    loss: oracle.loss,
  }
}

/**
 * The smooth problem with the masked weights held at zero: the objective is composed with the mask, so the gradient of
 * a masked weight is exactly zero and a solver that starts at zero never moves it (L-BFGS and OWL-QN included).
 */
function maskedSmooth(problem: SmoothProblem, mask: Float64Array): SmoothProblem {
  const m = tensor(Array.from(mask))
  return {
    ...problem,
    objective: ((x: Value, batch?: readonly number[]) =>
      problem.objective(mul(x, m), batch)) as SmoothProblem['objective'],
  }
}

const smoothOf = (p: CrfEstimatorProblem): SmoothProblem => {
  const smooth = structuredSmoothProblem(p.full)
  return p.mask === undefined ? smooth : maskedSmooth(smooth, p.mask)
}

const onlineModes = Object.fromEntries(Object.entries(structuredModes).filter(([name]) => name !== 'owlqn'))

/**
 * The ways to train the CRF estimator: L-BFGS (the default) and every smooth solver on the penalised negative
 * log-likelihood, `owlqn` for the L1 penalty, the online algorithms (perceptron, averaged, passive-aggressive, AROW) and
 * the structured SVMs (`bcfw`, `subgradient`, `one-slack`, `n-slack`, which report the structured hinge as their
 * objective), all over the weights CRFsuite's feature generation leaves free. It is `structuredSolvers` plus the SVM
 * modes with the mask of {@link CrfEstimatorProblem}.
 *
 * @index table CRF estimator solvers: L-BFGS, OWL-QN, SGD, perceptron family and structured SVMs over free features.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { crfEstimatorSolvers } from 'omai-engine/pgm'
 * [crfEstimatorSolvers.modeOf({}).name, crfEstimatorSolvers.names.includes('bcfw'), crfEstimatorSolvers.names.includes('owlqn')] // => ['lbfgs', true, true]
 * ```
 */
export const crfEstimatorSolvers: ModeTable<CrfEstimatorProblem, VectorParams> = familySolvers<CrfEstimatorProblem>(
  {
    ...overStructuredProblem<CrfEstimatorProblem>({ ...onlineModes, ...structuredSvmModes }, (p) => p.masked),
    owlqn: owlqnMode<CrfEstimatorProblem>({
      toSmooth: (p) => smoothOf({ ...p, full: { ...p.full, penalty: { c2: p.full.penalty?.c2 ?? 0 } } }),
      l1: (p) => p.full.penalty?.c1 ?? 0,
    }),
  },
  smoothOf,
  {
    label: 'solver',
    description: 'how the penalised conditional log-likelihood is maximised or the structured loss reduced',
    default: 'lbfgs',
  },
)

const BATCH_FREE = new Set([
  'perceptron',
  'averaged',
  'passive-aggressive',
  'arow',
  'subgradient',
  'bcfw',
  'one-slack',
  'n-slack',
])

// ---------------------------------------------------------------------------------------------------------------------
// Preparing the data

/** The data of one fit, prepared: the model's layout, the problem over it and the label order. */
interface Prepared {
  readonly crf: LinearChainCrf
  readonly problem: CrfEstimatorProblem
  readonly sequences: number
}

function extract<I>(h: CrfHyper<I>, x: I): CrfInput {
  return h.features === undefined ? crfSequence(x as unknown as CrfSequence) : h.features.extract(x)
}

function sequencesOf<I>(data: LearnData, who: string): CrfData<I> {
  const d = data as Partial<CrfData<I>>
  if (d.sequences === undefined) throw new OmaiError('value', `${who}: needs sequences (use crfData)`)
  return d as CrfData<I>
}

/** The features CRFsuite generates: observed ones that reach `minFreq`, and all of them with `allPossible*` and no cut-off. */
function featureMask(
  problem: StructuredProblem<ChainInput, ChainLabels>,
  crf: LinearChainCrf,
  h: CrfHyper<never>,
): Float64Array | undefined {
  if (h.allPossibleStates && h.allPossibleTransitions && h.minFreq <= 0) return undefined
  const size = problem.oracle.size
  const counts = new Float64Array(size)
  for (const { x, y } of problem.data) {
    const f = problem.oracle.features(x, y)
    for (let q = 0; q < size; q++) counts[q] = (counts[q] ?? 0) + (f[q] ?? 0)
  }
  const states = crf.stateWeights.length
  const mask = new Float64Array(size)
  for (let q = 0; q < size; q++) {
    const count = counts[q] ?? 0
    const everything = q < states ? h.allPossibleStates : h.allPossibleTransitions
    mask[q] = (count > 0 ? count >= h.minFreq : everything && h.minFreq <= 0) ? 1 : 0
  }
  return mask
}

function prepare<I>(data: LearnData, h: CrfHyper<I>): Prepared {
  const { sequences, labels } = sequencesOf<I>(data, 'crf')
  if (labels === undefined) throw new OmaiError('value', 'crf: fit needs the label sequences')
  const inputs: CrfInput[] = []
  const targets: (readonly string[])[] = []
  for (const [i, s] of sequences.entries()) {
    const x = extract(h, s)
    const y = labels[i] ?? []
    const length = 'unigram' in x ? x.unigram.length : x.length
    if (length !== y.length)
      throw new OmaiError('shape', `crf: sequence ${i} has ${length} items and ${y.length} labels`)
    if (length === 0) continue
    inputs.push(x)
    targets.push(y)
  }
  if (inputs.length === 0) throw new OmaiError('value', 'crf: no non-empty training sequences')
  const names = h.labels === undefined ? [...new Set(targets.flat())] : [...h.labels]
  const known = new Set(names)
  for (const label of targets.flat())
    if (!known.has(label)) throw new OmaiError('value', `crf: label ${label} is not among the labels`)
  const crf = crfModelFromInputs(names, inputs, {
    ...(h.minAttributeFrequency === undefined ? {} : { minFrequency: h.minAttributeFrequency }),
    transitions: h.transitions ?? h.features?.labelTransitions ?? true,
  })
  const full = crfStructuredProblem({
    crf,
    data: inputs.map((x, i) => ({ x, y: targets[i] ?? [] })),
    penalty: { c2: h.c2, ...(h.c1 > 0 ? { c1: h.c1 } : {}) },
  })
  const mask = featureMask(full, crf, h as CrfHyper<never>)
  const masked = mask === undefined ? full : { ...full, oracle: maskedOracle(full.oracle, mask) }
  return { crf, problem: { full, masked, ...(mask === undefined ? {} : { mask }) }, sequences: inputs.length }
}

const preparations = new WeakMap<LearnData, Map<string, Prepared>>()
const extractorIds = new WeakMap<object, number>()
let extractorCount = 0

/** The preparation of a dataset under the hyperparameters that shape it (built once per dataset and settings). */
function prepared<I>(data: LearnData, h: CrfHyper<I>): Prepared {
  let id = 0
  if (h.features !== undefined) {
    id = extractorIds.get(h.features) ?? ++extractorCount
    extractorIds.set(h.features, id)
  }
  const label = JSON.stringify([
    h.labels,
    h.minFreq,
    h.minAttributeFrequency,
    h.allPossibleStates,
    h.allPossibleTransitions,
    h.transitions,
    h.c1,
    h.c2,
    id,
  ])
  let byData = preparations.get(data)
  if (byData === undefined) {
    byData = new Map()
    preparations.set(data, byData)
  }
  let made = byData.get(label)
  if (made === undefined) {
    made = prepare(data, h)
    byData.set(label, made)
  }
  return made
}

// ---------------------------------------------------------------------------------------------------------------------
// The algorithm

/** The state of the CRF estimator's fit: the training state of `crfTrainAlgorithm`, and the last objectives for `delta`. */
export type CrfEstimatorState = CrfTrainState & { readonly recent: readonly number[] }

/** The solver of the hyperparameters: `solver`, or the value `algorithm` names. */
function solverOf(h: CrfHyper<never>, sequences: number): SpaceValue {
  const solver =
    h.solver ??
    (() => {
      switch (h.algorithm) {
        case 'lbfgs':
          return { kind: h.c1 > 0 ? 'owlqn' : 'lbfgs' }
        case 'l2sgd':
          return { kind: 'gradient', batch: Math.min(sequences, 16), learningRate: Math.min(1, 0.1 / sequences) }
        case 'ap':
          return { kind: 'averaged' }
        case 'pa':
          return { kind: 'passive-aggressive', aggressiveness: h.c }
        case 'arow':
          return { kind: 'arow', variance: h.variance, gamma: h.gamma }
        default:
          return { kind: h.algorithm }
      }
    })()
  const config = solver as { readonly kind: string; readonly gradTol?: number; readonly ftol?: number }
  const knobs = (crfEstimatorSolvers.space.options[config.kind] as DictSpace | undefined)?.spaces ?? {}
  return {
    ...config,
    // the tolerance on the gradient is epsilon; the reduction test of L-BFGS is left to `delta` and `period` (below)
    ...('gradTol' in knobs && config.gradTol === undefined
      ? { gradTol: Math.min(1e-2, Math.max(1e-12, h.epsilon)) }
      : {}),
    ...('ftol' in knobs && config.ftol === undefined ? { ftol: 1e-16 } : {}),
  } as SpaceValue
}

/** The same algorithm with CRFsuite's stopping test: the relative improvement over `period` iterations below `delta`. */
function withStopping(alg: Algorithm<CrfTrainState>, delta: number, period: number): Algorithm<CrfEstimatorState> {
  const remember = (before: readonly number[], s: CrfTrainState): CrfEstimatorState => ({
    ...s,
    recent: [...before, s.objective].slice(-(period + 1)),
  })
  const { deserialise: _deserialise, snapshot: _snapshot, ...rest } = alg
  return {
    ...rest,
    deserialise: (json) => alg.deserialise?.(json) as CrfEstimatorState,
    init: (k) => remember([], alg.init(k)),
    step: (s, ctx) => remember(s.recent, alg.step(s, ctx)),
    done: (s, t) => {
      const inner = alg.done?.(s, t)
      if (inner !== undefined || delta <= 0 || s.recent.length <= period) return inner
      const past = s.recent[0] ?? 0
      const now = s.recent[s.recent.length - 1] ?? 0
      return now > 0 && (past - now) / now < delta ? 'converged' : undefined
    },
  }
}

function estimatorAlgorithm<I>(data: LearnData, h: CrfHyper<I>, key: Key): Algorithm<CrfEstimatorState> {
  const prep = prepared(data, h)
  const solver = solverOf(h as CrfHyper<never>, prep.sequences)
  const P = prep.problem.full.oracle.size
  const kind = (solver as { readonly kind: string }).kind
  const slot = slotAlgorithm<CrfTrainState, CrfEstimatorProblem>({
    name: 'crf-estimator',
    config: {
      solver,
      sequences: prep.sequences,
      c1: h.c1,
      c2: h.c2,
      nonMasked: prep.problem.mask === undefined,
    } as Json,
    schema: {
      metrics: structuredTrainMetrics({
        name: 'sequenceError',
        description: 'the fraction of training sequences not decoded exactly right',
      }),
      tensors: { weights: { role: 'param', description: 'the CRF weights: state, transition, bigram', shape: [P] } },
    },
    problem: prep.problem,
    table: crfEstimatorSolvers,
    solver,
    tol: h.epsilon,
    key,
    view: (r: SolverReached) => {
      const { error, ...rest } = structuredTrainView(prep.problem.full, r)
      return { ...rest, sequenceError: error, evaluations: 0, stalled: false }
    },
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        gradient: s.gradient,
        loss: s.loss,
        sequenceError: s.sequenceError,
        nonzeros: s.nonzeros,
        mistakes: s.mistakes,
      },
      tensors: { weights: { role: 'param', value: s.weights, shape: [P] } },
    }),
  })
  return withStopping(slot, BATCH_FREE.has(kind) ? 0 : h.delta, h.period)
}

// ---------------------------------------------------------------------------------------------------------------------
// The fitted value

/** The fitted parameters of the CRF estimator, as plain JSON: the vocabularies, the weights and the training summary. */
export type CrfParams = {
  /** The classes, in the order of the weights (first appearance in the training data, or the `labels` given). */
  readonly labels: string[]
  readonly attributes: string[]
  /** The attributes of the edges (CRF++ bigram templates). */
  readonly bigramAttributes: string[]
  /** Whether the model has plain label-pair transition weights. */
  readonly transitions: boolean
  /** State weights, row-major [attribute][label]. */
  readonly state: number[]
  /** Transition weights, row-major [from][to] (empty without transitions). */
  readonly transition: number[]
  /** Bigram weights, row-major [attribute][from][to]. */
  readonly bigram: number[]
  /** The penalised objective at the weights, the share of training sequences not decoded right, the weights away from zero. */
  readonly objective: number
  readonly sequenceError: number
  readonly nonzeros: number
}

/** A weight table: `{ [from]: { [to]: weight } }`, only the weights that are not exactly zero. */
export type CrfWeightTable = Record<string, Record<string, number>>

/** The verbs and facts of a fitted CRF estimator. */
export interface CrfFitted<I = CrfSequence> {
  readonly estimator: string
  readonly kind: 'other'
  /** The fitted parameters (plain JSON; see `serialise`). */
  readonly params: CrfParams
  /** The records of the fit: the strongest features per label and the strongest transitions. */
  readonly exposition: readonly Exposition[]
  /** The tape of the fit (empty for a revived value). */
  readonly run: Run
  /** The class labels, in the order of the weights (sklearn's `classes_`). */
  readonly classes: readonly string[]
  /** The attributes with a nonzero state weight (sklearn's `attributes_`). */
  readonly attributes: readonly string[]
  /** The fitted model as a `LinearChainCrf`, for the functions of `crf.ts`, `chain-oracle.ts` and other decoders. */
  readonly model: LinearChainCrf
  /** The most probable labelling of each sequence. */
  predict(X: readonly I[]): string[][]
  predictSingle(x: I): string[]
  /** For each position of each sequence, the probability of each label (sklearn's `predict_marginals`). */
  predictMarginals(X: readonly I[]): Record<string, number>[][]
  predictMarginalsSingle(x: I): Record<string, number>[]
  /** The flat accuracy over all items of all sequences (sklearn-crfsuite's `score`). */
  score(X: readonly I[], y: readonly (readonly string[])[]): number
  /** The unnormalised score and the conditional log-likelihood of a labelling of a sequence. */
  scoreSequence(x: I, y: readonly string[]): number
  logLikelihood(x: I, y: readonly string[]): number
  /** The state weights `{ [attribute]: { [label]: weight } }` (sklearn's `state_features_`), the nonzero ones. */
  stateFeatures(): CrfWeightTable
  /** The transition weights `{ [from]: { [to]: weight } }` (sklearn's `transition_features_`), the nonzero ones. */
  transitionFeatures(): CrfWeightTable
  /** The weights of edge features `{ [attribute]: { [from->to]: weight } }` (CRF++ bigram templates), the nonzero ones. */
  bigramFeatures(): CrfWeightTable
  /** The strongest features per label and the strongest transitions. */
  topFeatures(options?: TopFeaturesOptions): CrfFeatureSummary
  /** The weight carried by each template of the extractor (needs `features`). */
  templateWeights(inputs: readonly I[]): TemplateWeight[]
  /** The fitted value as JSON: the estimator's name and its parameters (`crf.revive(params)` rebuilds it). */
  serialise(): SerialisedFitted
}

/** The paths of a CRF estimator's session: those of the iterative base (the solver's channels, `model/params`). */
export type CrfPath = IterativePath<never>

/** A session of a CRF fit: the tape to advance, seek, probe and fork, and the fitted value at any reached step. */
export type CrfSession<I = CrfSequence> = Omit<
  IterativeSession<CrfEstimatorState, CrfParams, Verbs, never>,
  'fittedAt' | 'fork'
> & {
  fittedAt(t: number): CrfFitted<I>
  fork(t: number, options?: { readonly seed?: number; readonly config?: Partial<Config> }): CrfSession<I>
}

/**
 * The CRF estimator: the shape of the estimator protocol (name, hyperparameters, `fit`, `revive`, `with`, plus the
 * `method` and `session` of the log contract) over {@link CrfData} and {@link CrfFitted}.
 */
export interface CrfEstimator<I = CrfSequence> {
  readonly name: string
  readonly kind: 'other'
  readonly hyper: CrfHyper<I>
  readonly hyperSpace?: DictSpace
  /** Fit on sequences and their labels. */
  fit(data: CrfData<I>, options?: FitOptions): CrfFitted<I>
  /** `fit(crfData(X, y))`, as sklearn's `fit(X, y)`. */
  fitSequences(X: readonly I[], y: readonly (readonly string[])[], options?: FitOptions): CrfFitted<I>
  /** Rebuild a fitted value from its parameters. */
  revive(params: Json): CrfFitted<I>
  /** A copy with some hyperparameters changed (sklearn's `clone` and `set_params`). */
  with(changes: Partial<CrfHyper<I>>): CrfEstimator<I>
  /** The method of the fit on a dataset, to drive as a session or to read its contract. */
  method(data: CrfData<I>): Method<CrfEstimatorState, Config, CrfPath>
  /** A session of the fit: live by default, with the fitted value at any step. */
  session(data: CrfData<I>, options?: IterativeSessionOptions<CrfPath>): CrfSession<I>
}

const modelOf = (p: CrfParams): LinearChainCrf =>
  linearChainCrf(p.labels, p.attributes, {
    state: p.state,
    transitions: p.transitions,
    ...(p.transitions ? { transition: p.transition } : {}),
    bigramAttributes: p.bigramAttributes,
    bigram: p.bigram,
  })

const labelDictionary = (labels: readonly string[], row: ArrayLike<number>, at: number): Record<string, number> =>
  Object.fromEntries(labels.map((label, k) => [label, row[at + k] ?? 0]))

function table(
  rows: readonly string[],
  columns: readonly string[],
  w: ArrayLike<number>,
  width = columns.length,
): CrfWeightTable {
  const out: CrfWeightTable = {}
  for (const [i, row] of rows.entries()) {
    const cells: Record<string, number> = {}
    for (const [k, column] of columns.entries()) {
      const v = w[i * width + k] ?? 0
      if (v !== 0) cells[column] = v
    }
    if (Object.keys(cells).length > 0) out[row] = cells
  }
  return out
}

function verbsOf<I>(
  params: CrfParams,
  h: CrfHyper<I>,
  wrapped: { readonly exposition: readonly Exposition[]; readonly run: Run },
): CrfFitted<I> {
  const crf = modelOf(params)
  const K = params.labels.length
  const input = (x: I): CrfInput => extract(h, x)
  const decode = (x: I): string[] => {
    const observation = input(x)
    const length = 'unigram' in observation ? observation.unigram.length : observation.length
    if (length === 0) return []
    return [...(h.decoder === undefined ? crfDecode(crf, observation).labels : h.decoder(crf, observation))]
  }
  const marginals = (x: I): Record<string, number>[] =>
    crfMarginals(crf, input(x)).map((row) => labelDictionary(params.labels, row, 0))
  const attributes = params.attributes.filter((_, i) =>
    params.labels.some((_l, k) => (params.state[i * K + k] ?? 0) !== 0),
  )
  return {
    estimator: 'pgm/crf',
    kind: 'other',
    params,
    exposition: wrapped.exposition,
    run: wrapped.run,
    classes: params.labels,
    attributes,
    model: crf,
    predict: (X) => X.map(decode),
    predictSingle: decode,
    predictMarginals: (X) => X.map(marginals),
    predictMarginalsSingle: marginals,
    score: (X, y) => {
      if (X.length !== y.length)
        throw new OmaiError('shape', `crf score: ${X.length} sequences and ${y.length} label lists`)
      let right = 0
      let total = 0
      for (const [i, x] of X.entries()) {
        const predicted = decode(x)
        const gold = y[i] ?? []
        if (predicted.length !== gold.length)
          throw new OmaiError(
            'shape',
            `crf score: sequence ${i} has ${predicted.length} items and ${gold.length} labels`,
          )
        for (const [t, label] of gold.entries()) if (predicted[t] === label) right++
        total += gold.length
      }
      return total === 0 ? Number.NaN : right / total
    },
    scoreSequence: (x, y) => crfScore(crf, input(x), y),
    logLikelihood: (x, y) => crfLogLikelihood(crf, input(x), y),
    stateFeatures: () => table(params.attributes, params.labels, params.state),
    transitionFeatures: () => (params.transitions ? table(params.labels, params.labels, params.transition) : {}),
    bigramFeatures: () => {
      const out: CrfWeightTable = {}
      for (const [a, attribute] of params.bigramAttributes.entries()) {
        const cells: Record<string, number> = {}
        for (const [i, from] of params.labels.entries())
          for (const [j, to] of params.labels.entries()) {
            const v = params.bigram[(a * K + i) * K + j] ?? 0
            if (v !== 0) cells[`${from}->${to}`] = v
          }
        if (Object.keys(cells).length > 0) out[attribute] = cells
      }
      return out
    },
    topFeatures: (options) => crfTopFeatures(crf, options),
    templateWeights: (inputs) => {
      if (h.features === undefined)
        throw new OmaiError('value', 'crf templateWeights: the estimator has no `features` extractor')
      return crfTemplateWeights(crf, h.features, inputs)
    },
    serialise: () => ({ estimator: 'pgm/crf', params: params as unknown as Json }),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The estimator

/**
 * The linear-chain CRF as an estimator, in sklearn-crfsuite's shape. `fit` takes {@link crfData}: sequences of feature
 * dictionaries (or attribute-name lists, or one name per item) and their label sequences, or raw inputs (token tables)
 * with a `features` extractor (`crfppExtractor`, `wapitiExtractor`, `attributeExtractor`, `functionExtractor`,
 * `hashedExtractor`), the model's vocabulary built from the training data. The solver is a slot (`algorithm`: `lbfgs`,
 * `owlqn` for L1, `l2sgd`, the perceptron family `ap`, `pa`, `arow`, the structured SVMs, every smooth solver), the
 * penalties are `c1` and `c2`, and CRFsuite's feature generation (`minFreq`, `allPossibleStates`,
 * `allPossibleTransitions`) is reproduced, so the weights equal CRFsuite's. The fitted value predicts labels and
 * marginals, scores accuracy, exposes the weights as dictionaries, and carries the training tape (`run`), whose objective,
 * gradient, sequence error and nonzero weights are the paths of `session`.
 *
 * @param hyper - Hyperparameters, as {@link CrfHyper} (defaults as sklearn-crfsuite).
 * @returns The estimator.
 * @concept conditional-random-fields
 * @concept log-contract
 * @index estimator Linear-chain CRF estimator over feature dictionaries or templates, sklearn-crfsuite shape.
 * @example
 * ```ts
 * import { crf, crfData } from 'omai-engine/pgm'
 * const X = [[['w=a'], ['w=b']], [['w=a'], ['w=b']], [['w=b'], ['w=a']]]
 * const y = [['X', 'Y'], ['X', 'Y'], ['Y', 'X']]
 * const fitted = crf({ c2: 0.1 }).fit(crfData(X, y))
 * [fitted.predict([[['w=a'], ['w=b']]]), fitted.classes] // => [[['X', 'Y']], ['X', 'Y']]
 * ```
 */
export function crf<I = CrfSequence>(hyper: Partial<CrfHyper<I>> = {}): CrfEstimator<I> {
  const full: CrfHyper<I> = { ...(defaults as CrfHyper<I>), ...hyper }
  const base: IterativeEstimator<CrfHyper<I>, CrfEstimatorState, CrfParams, Verbs, never> = iterative<
    CrfHyper<I>,
    CrfEstimatorState,
    CrfParams,
    Verbs,
    never
  >(
    {
      name: 'pgm/crf',
      kind: 'other',
      defaults: full,
      hyperSpace: (h) =>
        dict({
          c1: real(0, 1000, { default: h.c1, label: 'L1 coefficient c1' }),
          c2: real(0, 1000, { default: h.c2, label: 'L2 coefficient c2' }),
        }),
      algorithm: (data, h, key) => estimatorAlgorithm(data, h, key),
      maxSteps: (h, data) => {
        if (h.maxIterations !== undefined) return h.maxIterations
        const kind = (solverOf(h as CrfHyper<never>, prepared(data, h).sequences) as { readonly kind: string }).kind
        return kind === 'gradient' ? 1000 : BATCH_FREE.has(kind) ? 100 : 10_000
      },
      params: (s, data, h) => crfParamsOf(s, prepared(data, h).crf),
      verbs: () => ({}),
      logs: {
        'solver/objective': Log(),
        'solver/gradient': Log(),
        'solver/loss': Log(),
        'solver/sequenceError': Log(),
        'solver/nonzeros': Log(),
        'solver/mistakes': Log(),
      },
      records: (s, data, h) => {
        const top = crfTopFeatures(modelOf(crfParamsOf(s, prepared(data, h).crf)), { top: 10 })
        return [top.transitions, ...top.labels.slice(0, 8).map((l) => l.features)]
      },
    },
    full,
  )
  const wrap = (
    fitted: { readonly params: CrfParams; readonly exposition?: readonly Exposition[]; readonly run?: Run },
    h: CrfHyper<I>,
  ): CrfFitted<I> =>
    verbsOf(fitted.params, h, {
      exposition: fitted.exposition ?? [],
      run: fitted.run ?? { schema: 'omai.run/1', events: [] },
    })
  const loud = <O extends { readonly logs?: unknown }>(options: O | undefined): O | undefined =>
    full.verbose === true && options?.logs === undefined ? ({ ...options, logs: 'all' } as unknown as O) : options
  const estimator: CrfEstimator<I> = {
    name: 'pgm/crf',
    kind: 'other',
    hyper: full,
    ...(base.hyperSpace === undefined ? {} : { hyperSpace: base.hyperSpace as DictSpace }),
    fit: (data, options) => wrap(base.fit(data, loud(options)), full),
    fitSequences: (X, y, options) => estimator.fit(crfData(X, y), options),
    revive: (params) => wrap({ params: params as unknown as CrfParams }, full),
    with: (changes) => crf<I>({ ...full, ...changes }),
    method: (data) => base.method(data) as Method<CrfEstimatorState, Config, CrfPath>,
    session: (data, options) => {
      const inner = base.session(data, loud(options) as IterativeSessionOptions<IterativePath<never>>)
      const sessionOf = (s: Session<CrfEstimatorState, Config, CrfPath>): CrfSession<I> =>
        fittedSession<CrfEstimatorState, CrfParams, Verbs, CrfPath>(s, (state, config) => {
          const h = { ...full, ...config } as CrfHyper<I>
          const prep = prepared(data, h)
          const params = crfParamsOf(state, prep.crf)
          return wrap({ params }, h) as never
        }) as unknown as CrfSession<I>
      return sessionOf(inner as unknown as Session<CrfEstimatorState, Config, CrfPath>)
    },
  }
  return estimator
}

function crfParamsOf(s: CrfEstimatorState, crf: LinearChainCrf): CrfParams {
  const A = crf.stateWeights.length
  const T = crf.transitionWeights.length
  return {
    labels: [...crf.labels],
    attributes: [...crf.attributes],
    bigramAttributes: [...crf.bigramAttributes],
    transitions: T > 0,
    state: s.weights.slice(0, A),
    transition: s.weights.slice(A, A + T),
    bigram: s.weights.slice(A + T),
    objective: s.objective,
    sequenceError: s.sequenceError,
    nonzeros: s.nonzeros,
  }
}

/**
 * sklearn-crfsuite's `CRF` class over the estimator: `new CRF({ c2: 0.1 }).fit(X, y).predict(X)`. The wrapper holds the
 * latest fitted value (the only mutable state, replaced on every `fit`); the functional estimator underneath is
 * {@link crf}. Its accessors are the fitted value's, named as sklearn-crfsuite's attributes without the underscore.
 *
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { CRF } from 'omai-engine/pgm'
 * const model = new CRF({ c2: 0.1 })
 * model.fit([[['w=a'], ['w=b']], [['w=b'], ['w=a']]], [['X', 'Y'], ['Y', 'X']])
 * [model.predict([[['w=a'], ['w=b']]]), model.score([[['w=b'], ['w=a']]], [['Y', 'X']])] // => [[['X', 'Y']], 1]
 * ```
 */
export class CRF<I = CrfSequence> {
  /** The functional estimator. */
  estimator: CrfEstimator<I>
  /** The latest fitted value, if any. */
  fitted: CrfFitted<I> | undefined

  /**
   * Wrap a CRF estimator made from hyperparameters.
   *
   * @param hyper - The hyperparameters of {@link crf}.
   */
  constructor(hyper: Partial<CrfHyper<I>> = {}) {
    this.estimator = crf<I>(hyper)
    this.fitted = undefined
  }

  /** The fitted value, or an error when `fit` has not been called. */
  get model(): CrfFitted<I> {
    if (this.fitted === undefined) throw new OmaiError('value', 'CRF: call fit first')
    return this.fitted
  }

  /** The class labels (sklearn's `classes_`). */
  get classes(): readonly string[] {
    return this.model.classes
  }

  /** The state weights as dictionaries (sklearn's `state_features_`). */
  get stateFeatures(): CrfWeightTable {
    return this.model.stateFeatures()
  }

  /** The transition weights as dictionaries (sklearn's `transition_features_`). */
  get transitionFeatures(): CrfWeightTable {
    return this.model.transitionFeatures()
  }

  /**
   * Fit on sequences and their labels.
   *
   * @param X - The sequences.
   * @param y - The label sequences.
   * @param options - Randomness and tracing.
   * @returns This wrapper.
   */
  fit(X: readonly I[], y: readonly (readonly string[])[], options: FitOptions = {}): this {
    this.fitted = this.estimator.fitSequences(X, y, options)
    return this
  }

  /**
   * The most probable labelling of each sequence.
   *
   * @param X - The sequences.
   * @returns The labels.
   */
  predict(X: readonly I[]): string[][] {
    return this.model.predict(X)
  }

  /**
   * The most probable labelling of one sequence.
   *
   * @param x - The sequence.
   * @returns The labels.
   */
  predictSingle(x: I): string[] {
    return this.model.predictSingle(x)
  }

  /**
   * The marginal probability of each label at each position of each sequence.
   *
   * @param X - The sequences.
   * @returns For each sequence and position, `{ label: probability }`.
   */
  predictMarginals(X: readonly I[]): Record<string, number>[][] {
    return this.model.predictMarginals(X)
  }

  /**
   * The marginals of one sequence.
   *
   * @param x - The sequence.
   * @returns For each position, `{ label: probability }`.
   */
  predictMarginalsSingle(x: I): Record<string, number>[] {
    return this.model.predictMarginalsSingle(x)
  }

  /**
   * The flat accuracy over the items of all sequences.
   *
   * @param X - The sequences.
   * @param y - Their labels.
   * @returns The accuracy.
   */
  score(X: readonly I[], y: readonly (readonly string[])[]): number {
    return this.model.score(X, y)
  }

  /** The hyperparameters. */
  getParams(): CrfHyper<I> {
    return this.estimator.hyper
  }

  /**
   * Change hyperparameters (dropping the fitted model).
   *
   * @param changes - Hyperparameters to change.
   * @returns This wrapper.
   */
  setParams(changes: Partial<CrfHyper<I>>): this {
    this.estimator = this.estimator.with(changes)
    this.fitted = undefined
    return this
  }
}
