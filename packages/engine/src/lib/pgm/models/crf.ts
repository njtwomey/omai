import { type Json, WhiteboxError } from '../../base/index.ts'
import type { GradientTransformation } from '../../optim/index.ts'
import { adam } from '../../optim/index.ts'
import {
  type CrfppTemplate,
  crfppFeatures,
  featureIndex,
  type ItemAttributes,
  type TokenTable,
} from '../../text/index.ts'
import { type Algorithm, type RunStatus, runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
import { type DescentState, gradientDescent } from '../_descent.ts'
import { type DiscreteVariable, type Factor, factorMarginal, logFactor } from '../factor.ts'
import { type BeliefPropagationOptions, beliefPropagation } from '../inference/belief-propagation.ts'
import { type ChainModel, forwardBackward, viterbi } from '../inference/chain.ts'
import { calibrate, junctionTreeQuery } from '../inference/junction-tree.ts'
import { type FactorGraph, factorGraph } from './factor-graph.ts'

/**
 * The observation of one item of a sequence: attribute names with values (1 for an indicator, as CRFsuite). Feature
 * templates that produce them live in `text/templates.ts` and `text/attribute-templates.ts`.
 */
export type ItemFeatures = ItemAttributes

/**
 * The observation of a sequence with observation-dependent transitions (CRF++'s bigram templates): the unigram
 * attributes of each item and the bigram attributes of each edge, `bigram[t]` belonging to the edge from item t - 1 to
 * item t (`bigram[0]` is unused).
 */
export interface CrfObservation {
  readonly unigram: readonly ItemFeatures[]
  readonly bigram: readonly ItemFeatures[]
}

/** The input of a linear-chain CRF: item attributes, or item and edge attributes. */
export type CrfInput = readonly ItemFeatures[] | CrfObservation

function unigramOf(x: CrfInput): readonly ItemFeatures[] {
  return 'unigram' in x ? x.unigram : (x as readonly ItemFeatures[])
}

function bigramOf(x: CrfInput): readonly ItemFeatures[] | undefined {
  return 'bigram' in x ? x.bigram : undefined
}

/**
 * A linear-chain conditional random field (Lafferty, McCallum and Pereira 2001). State features pair each attribute
 * with each label (weight w(a, y)), transition features pair consecutive labels (weight w(y', y)), and bigram features
 * pair an edge attribute with a label pair (weight w(b, y', y), CRF++'s bigram templates):
 * log p(y | x) = sum_t [sum_a x_t(a) w(a, y_t) + w(y_{t-1}, y_t) + sum_b e_t(b) w(b, y_{t-1}, y_t)] - log Z(x).
 *
 * @concept conditional-random-fields
 */
export interface LinearChainCrf {
  readonly kind: 'linearChainCrf'
  readonly labels: readonly string[]
  readonly attributes: readonly string[]
  /** State weights, row-major [attribute][label]. */
  readonly stateWeights: Float64Array
  /** Transition weights, row-major [from][to] (empty for a model without label-pair transitions). */
  readonly transitionWeights: Float64Array
  /** Edge attributes of the bigram features. */
  readonly bigramAttributes: readonly string[]
  /** Bigram weights, row-major [attribute][from][to]. */
  readonly bigramWeights: Float64Array
}

type Weights = ArrayLike<number> | Readonly<Record<string, Readonly<Record<string, number>>>>

/**
 * Build a linear-chain CRF from its labels and attribute vocabulary, with zero weights or the given ones (dense
 * arrays, or nested records by attribute and label and by label pair). Bigram attributes add observation-dependent
 * transitions (dense weights [attribute][from][to]); `transitions: false` leaves out the plain label-pair weights, as
 * a CRF++ template without a bare `B` line does.
 *
 * @param labels - The labels.
 * @param attributes - The attributes.
 * @param weights - State, transition and bigram weights, the bigram attributes and whether to have transitions.
 * @returns The model.
 * @throws WhiteboxError with code `shape` for dense weights of the wrong length, `value` for unknown names.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf } from 'whitebox-engine/pgm'
 * Array.from(linearChainCrf(['B', 'I'], ['w=a'], { transition: { B: { I: 2 } } }).transitionWeights) // => [0, 2, 0, 0]
 * ```
 */
export function linearChainCrf(
  labels: readonly string[],
  attributes: readonly string[],
  weights: {
    readonly state?: Weights
    readonly transition?: Weights
    readonly transitions?: boolean
    readonly bigramAttributes?: readonly string[]
    readonly bigram?: ArrayLike<number>
  } = {},
): LinearChainCrf {
  const K = labels.length
  const dense = (w: Weights | undefined, rows: readonly string[], name: string): Float64Array => {
    const out = new Float64Array(rows.length * K)
    if (w === undefined) return out
    if ('length' in w && typeof w.length === 'number') {
      if (w.length !== out.length)
        throw new WhiteboxError('shape', `linearChainCrf: ${name} weights need ${out.length} entries`)
      out.set(Array.from(w as ArrayLike<number>))
      return out
    }
    for (const [r, cols] of Object.entries(w as Record<string, Record<string, number>>)) {
      const i = rows.indexOf(r)
      if (i < 0) throw new WhiteboxError('value', `linearChainCrf: unknown ${name} row ${r}`)
      for (const [label, v] of Object.entries(cols)) {
        const j = labels.indexOf(label)
        if (j < 0) throw new WhiteboxError('value', `linearChainCrf: unknown label ${label}`)
        out[i * K + j] = v
      }
    }
    return out
  }
  if (new Set(labels).size !== K) throw new WhiteboxError('value', 'linearChainCrf: labels must be distinct')
  const bigramAttributes = [...(weights.bigramAttributes ?? [])]
  const bigramWeights = new Float64Array(bigramAttributes.length * K * K)
  if (weights.bigram !== undefined) {
    if (weights.bigram.length !== bigramWeights.length)
      throw new WhiteboxError('shape', `linearChainCrf: bigram weights need ${bigramWeights.length} entries`)
    bigramWeights.set(Array.from(weights.bigram))
  }
  if (weights.transitions === false && weights.transition !== undefined)
    throw new WhiteboxError('value', 'linearChainCrf: transition weights given for a model without transitions')
  return {
    kind: 'linearChainCrf',
    labels: [...labels],
    attributes: [...attributes],
    stateWeights: dense(weights.state, attributes, 'state'),
    transitionWeights:
      weights.transitions === false ? new Float64Array(0) : dense(weights.transition, labels, 'transition'),
    bigramAttributes,
    bigramWeights,
  }
}

/** Attribute indices of a model (attributes outside the vocabulary are ignored, as at test time in CRFsuite). */
function lookup(names: readonly string[]): Map<string, number> {
  return new Map(names.map((a, i) => [a, i]))
}

/** The log transition matrix into item t (t >= 1): plain transitions plus the bigram attributes of the edge. */
function edgeMatrix(crf: LinearChainCrf, edge: ItemFeatures | undefined, index: Map<string, number>): number[][] {
  const K = crf.labels.length
  const m = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => crf.transitionWeights[i * K + j] ?? 0),
  )
  if (edge === undefined) return m
  for (const [b, v] of Object.entries(edge)) {
    const a = index.get(b)
    if (a === undefined) continue
    for (let i = 0; i < K; i++) {
      const row = m[i] as number[]
      for (let j = 0; j < K; j++) row[j] = (row[j] ?? 0) + v * (crf.bigramWeights[(a * K + i) * K + j] ?? 0)
    }
  }
  return m
}

/**
 * The potentials of a linear-chain CRF on one sequence, as a chain for forward-backward and Viterbi: unary log
 * potentials sum_a x_t(a) w(a, y), and the transition weights plus each edge's bigram attributes (no initial or final
 * transition, as CRFsuite and CRF++).
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes of the sequence.
 * @returns The chain.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfChain } from 'whitebox-engine/pgm'
 * crfChain(linearChainCrf(['A', 'B'], ['f'], { state: [1, -1] }), [{ f: 2 }]).logEmission // => [[2, -2]]
 * ```
 */
export function crfChain(crf: LinearChainCrf, x: CrfInput): ChainModel {
  const K = crf.labels.length
  const items = unigramOf(x)
  const index = lookup(crf.attributes)
  const logEmission = items.map((item) => {
    const row = new Array<number>(K).fill(0)
    for (const [a, v] of Object.entries(item)) {
      const i = index.get(a)
      if (i === undefined) continue
      for (let y = 0; y < K; y++) row[y] = (row[y] ?? 0) + v * (crf.stateWeights[i * K + y] ?? 0)
    }
    return row
  })
  const edges = bigramOf(x)
  const logInitial = new Array<number>(K).fill(0)
  if (edges === undefined || crf.bigramAttributes.length === 0 || items.length < 2)
    return { logInitial, logTransition: edgeMatrix(crf, undefined, index), logEmission }
  const bindex = lookup(crf.bigramAttributes)
  const logTransition = items.slice(1).map((_, t) => edgeMatrix(crf, edges[t + 1], bindex))
  return { logInitial, logTransition, logEmission }
}

function labelIndices(crf: LinearChainCrf, y: readonly (string | number)[]): number[] {
  return y.map((l) => {
    const i = typeof l === 'number' ? l : crf.labels.indexOf(l)
    if (!(i >= 0 && i < crf.labels.length)) throw new WhiteboxError('value', `crf: unknown label ${String(l)}`)
    return i
  })
}

/** The transition log potential of the edge into item t (t >= 1) of a chain. */
function transitionAt(chain: ChainModel, t: number, from: number, to: number): number {
  const tr = chain.logTransition
  const m = (Array.isArray(tr[0]?.[0]) ? (tr as readonly (readonly (readonly number[])[])[])[t - 1] : tr) as
    | readonly (readonly number[])[]
    | undefined
  return m?.[from]?.[to] ?? 0
}

/**
 * The unnormalised log score of a labelling: the sum of its state, transition and bigram weights.
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @param y - The labels.
 * @returns The score.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfScore } from 'whitebox-engine/pgm'
 * crfScore(linearChainCrf(['A', 'B'], ['f'], { state: [1, -1], transition: [0, 3, 0, 0] }), [{ f: 1 }, { f: 1 }], ['A', 'B']) // => 3
 * ```
 */
export function crfScore(crf: LinearChainCrf, x: CrfInput, y: readonly (string | number)[]): number {
  const chain = crfChain(crf, x)
  const idx = labelIndices(crf, y)
  let s = 0
  for (const [t, k] of idx.entries()) {
    s += chain.logEmission[t]?.[k] ?? 0
    if (t > 0) s += transitionAt(chain, t, idx[t - 1] ?? 0, k)
  }
  return s
}

/**
 * The conditional log-likelihood log p(y | x) of a labelling.
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @param y - The labels.
 * @returns log p(y | x).
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfLogLikelihood } from 'whitebox-engine/pgm'
 * crfLogLikelihood(linearChainCrf(['A', 'B'], ['f']), [{ f: 1 }, { f: 1 }], ['A', 'B']) === -Math.log(4) // => true
 * ```
 */
export function crfLogLikelihood(crf: LinearChainCrf, x: CrfInput, y: readonly (string | number)[]): number {
  return crfScore(crf, x, y) - forwardBackward(crfChain(crf, x), { method: 'log' }).logLikelihood
}

/**
 * The marginal p(y_t | x) of every label at every position, by forward-backward.
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @returns For each position, the probability of each label (in label order).
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfMarginals } from 'whitebox-engine/pgm'
 * crfMarginals(linearChainCrf(['A', 'B'], ['f'], { state: [Math.log(3), 0] }), [{ f: 1 }]) // => [[0.75, 0.25]]
 * ```
 */
export function crfMarginals(crf: LinearChainCrf, x: CrfInput): number[][] {
  if (unigramOf(x).length === 0) return []
  return forwardBackward(crfChain(crf, x)).posteriors
}

/**
 * The most probable labelling (Viterbi decoding) and its unnormalised score.
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @returns The labels and their score.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfDecode } from 'whitebox-engine/pgm'
 * crfDecode(linearChainCrf(['A', 'B'], ['f'], { state: [1, 0], transition: [0, 5, 0, 0] }), [{ f: 1 }, { f: 1 }]).labels // => ['A', 'B']
 * ```
 */
export function crfDecode(crf: LinearChainCrf, x: CrfInput): { readonly labels: string[]; readonly score: number } {
  if (unigramOf(x).length === 0) return { labels: [], score: 0 }
  const v = viterbi(crfChain(crf, x))
  return { labels: v.path.map((k) => crf.labels[k] ?? ''), score: v.logProbability }
}

/** Count each feature string once per firing, as attribute records with value = number of firings. */
function counted(features: readonly string[]): Record<string, number> {
  const out: Record<string, number> = {}
  for (const f of features) out[f] = (out[f] ?? 0) + 1
  return out
}

/**
 * The CRF input of a sequence under a CRF++ template (`text.parseCrfppTemplate`): the expanded unigram features of
 * each item and bigram features of each edge, as attributes that count their firings.
 *
 * @param template - The parsed CRF++ template.
 * @param table - The token-by-attribute table of the sequence.
 * @returns The observation.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfppObservation } from 'whitebox-engine/pgm'
 * import { parseCrfppTemplate } from 'whitebox-engine/text'
 * crfppObservation(parseCrfppTemplate('U00:%x[0,0]\nB'), [['a'], ['b']]) // => { unigram: [{ 'U00:a': 1 }, { 'U00:b': 1 }], bigram: [{}, { B: 1 }] }
 * ```
 */
export function crfppObservation(template: CrfppTemplate, table: TokenTable): CrfObservation {
  const f = crfppFeatures(template, table)
  return { unigram: f.unigram.map(counted), bigram: f.bigram.map(counted) }
}

/**
 * A linear-chain CRF in CRF++'s form for a template and training tables: a state weight per unigram feature and label
 * and a weight per bigram feature and label pair (the bare `B` line is the label-pair transition, as a bigram feature
 * that fires on every edge), over the features seen at least `minFrequency` times (CRF++'s `-f`).
 *
 * @param template - The parsed CRF++ template.
 * @param labels - The labels.
 * @param tables - The token-by-attribute tables of the training sequences.
 * @param options - The minimum feature frequency (default 1).
 * @returns The model with zero weights, ready for {@link fitCrf} on {@link crfppObservation} inputs.
 * @concept crf-features
 * @example
 * ```ts
 * import { crfppCrf } from 'whitebox-engine/pgm'
 * import { parseCrfppTemplate } from 'whitebox-engine/text'
 * const crf = crfppCrf(parseCrfppTemplate('U00:%x[0,0]\nB'), ['D', 'N'], [[['the'], ['dog']]])
 * [crf.attributes, crf.bigramAttributes, crf.transitionWeights.length] // => [['U00:the', 'U00:dog'], ['B'], 0]
 * ```
 */
export function crfppCrf(
  template: CrfppTemplate,
  labels: readonly string[],
  tables: readonly TokenTable[],
  options: { readonly minFrequency?: number } = {},
): LinearChainCrf {
  const expanded = tables.map((t) => crfppFeatures(template, t))
  const min = options.minFrequency === undefined ? {} : { minFrequency: options.minFrequency }
  const unigram = featureIndex(
    expanded.map((e) => e.unigram),
    min,
  )
  const bigram = featureIndex(
    expanded.map((e) => e.bigram.slice(1)),
    min,
  )
  return linearChainCrf(labels, unigram.features, { transitions: false, bigramAttributes: bigram.features })
}

/** A labelled training sequence. */
export interface LabelledSequence {
  readonly x: CrfInput
  readonly y: readonly (string | number)[]
}

/** Regularisation of the training objective. */
export interface CrfPenalty {
  /** L2 coefficient: the objective adds c2 * ||w||^2 (CRFsuite's `c2`). */
  readonly c2?: number
  /** L1 coefficient: the objective adds c1 * ||w||_1 (a subgradient is used at zero). */
  readonly c1?: number
}

/**
 * The parameters of a linear-chain CRF as one vector: state weights, then transition weights, then bigram weights.
 *
 * @param crf - The model.
 * @returns The parameter vector.
 * @example
 * ```ts
 * import { linearChainCrf, crfParameters } from 'whitebox-engine/pgm'
 * crfParameters(linearChainCrf(['A', 'B'], ['f'])).length // => 6
 * ```
 */
export function crfParameters(crf: LinearChainCrf): Float64Array {
  const out = new Float64Array(crf.stateWeights.length + crf.transitionWeights.length + crf.bigramWeights.length)
  out.set(crf.stateWeights)
  out.set(crf.transitionWeights, crf.stateWeights.length)
  out.set(crf.bigramWeights, crf.stateWeights.length + crf.transitionWeights.length)
  return out
}

/**
 * A copy of a linear-chain CRF with new parameters (state, transition, then bigram weights).
 *
 * @param crf - The model.
 * @param parameters - The parameter vector.
 * @returns The new model.
 * @throws WhiteboxError with code `shape` for a vector of the wrong length.
 * @example
 * ```ts
 * import { linearChainCrf, withCrfParameters } from 'whitebox-engine/pgm'
 * Array.from(withCrfParameters(linearChainCrf(['A', 'B'], ['f']), [1, 2, 3, 4, 5, 6]).transitionWeights) // => [3, 4, 5, 6]
 * ```
 */
export function withCrfParameters(crf: LinearChainCrf, parameters: ArrayLike<number>): LinearChainCrf {
  const n = crf.stateWeights.length
  const m = crf.transitionWeights.length
  if (parameters.length !== n + m + crf.bigramWeights.length)
    throw new WhiteboxError('shape', 'withCrfParameters: wrong number of parameters')
  const p = Float64Array.from(parameters)
  return { ...crf, stateWeights: p.slice(0, n), transitionWeights: p.slice(n, n + m), bigramWeights: p.slice(n + m) }
}

function penalise(w: Float64Array, value: number, gradient: Float64Array, penalty: CrfPenalty): number {
  const c2 = penalty.c2 ?? 0
  const c1 = penalty.c1 ?? 0
  let v = value
  for (let i = 0; i < w.length; i++) {
    const wi = w[i] ?? 0
    v += c2 * wi * wi + c1 * Math.abs(wi)
    gradient[i] = (gradient[i] ?? 0) + 2 * c2 * wi + c1 * Math.sign(wi)
  }
  return v
}

/**
 * The training objective of a linear-chain CRF and its gradient: the negative conditional log-likelihood of the data
 * plus the penalties, -sum log p(y | x) + c2 ||w||^2 + c1 ||w||_1 (CRFsuite's L-BFGS objective), whose gradient is the
 * expected minus the empirical feature counts plus the penalty terms. Expectations come from forward-backward.
 *
 * @param crf - The model (its weights are the point of evaluation).
 * @param data - Labelled sequences.
 * @param penalty - L2 and L1 coefficients.
 * @returns The objective and its gradient (in {@link crfParameters} order).
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfObjective } from 'whitebox-engine/pgm'
 * const r = crfObjective(linearChainCrf(['A', 'B'], ['f']), [{ x: [{ f: 1 }], y: ['A'] }])
 * [r.value === Math.LN2, Array.from(r.gradient)] // => [true, [-0.5, 0.5, 0, 0, 0, 0]]
 * ```
 */
export function crfObjective(
  crf: LinearChainCrf,
  data: readonly LabelledSequence[],
  penalty: CrfPenalty = {},
): { readonly value: number; readonly gradient: Float64Array } {
  const K = crf.labels.length
  const nState = crf.stateWeights.length
  const nTrans = crf.transitionWeights.length
  const bigramAt = nState + nTrans
  const gradient = new Float64Array(bigramAt + crf.bigramWeights.length)
  const index = lookup(crf.attributes)
  const bindex = lookup(crf.bigramAttributes)
  const add = (i: number, v: number): void => {
    gradient[i] = (gradient[i] ?? 0) + v
  }
  let value = 0
  for (const { x, y } of data) {
    const items = unigramOf(x)
    if (items.length === 0) continue
    const edges = bigramOf(x)
    const idx = labelIndices(crf, y)
    if (idx.length !== items.length) throw new WhiteboxError('shape', 'crfObjective: one label per item')
    const chain = crfChain(crf, x)
    const fb = forwardBackward(chain)
    let score = 0
    for (const [t, item] of items.entries()) {
      const yt = idx[t] ?? 0
      score += chain.logEmission[t]?.[yt] ?? 0
      const post = fb.posteriors[t] ?? []
      for (const [a, v] of Object.entries(item)) {
        const i = index.get(a)
        if (i === undefined) continue
        add(i * K + yt, -v)
        for (let k = 0; k < K; k++) add(i * K + k, v * (post[k] ?? 0))
      }
      if (t === 0) continue
      const prev = idx[t - 1] ?? 0
      score += transitionAt(chain, t, prev, yt)
      const xi = fb.pairwise[t - 1] ?? []
      if (nTrans > 0) {
        add(nState + prev * K + yt, -1)
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) add(nState + i * K + j, xi[i]?.[j] ?? 0)
      }
      for (const [b, v] of Object.entries(edges?.[t] ?? {})) {
        const a = bindex.get(b)
        if (a === undefined) continue
        const base = bigramAt + a * K * K
        add(base + prev * K + yt, -v)
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) add(base + i * K + j, v * (xi[i]?.[j] ?? 0))
      }
    }
    value -= score - fb.logLikelihood
  }
  return { value: penalise(crfParameters(crf), value, gradient, penalty), gradient }
}

/** The state of a CRF training run: the parameters, the optimiser state, the objective and its gradient. */
export type CrfTrainingState = DescentState

/** Options of {@link crfTrainingAlgorithm} and {@link fitCrf}. */
export interface CrfTrainingOptions extends CrfPenalty {
  /** The optimiser (default Adam with learning rate 0.1). */
  readonly method?: GradientTransformation
  /** Stop when the gradient norm falls below this (default 1e-6). */
  readonly gradTol?: number
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/**
 * Training a linear-chain CRF as an {@link Algorithm}: minimise {@link crfObjective} with a gradient transformation
 * from `optim` (any first-order method), recording the objective and the gradient norm each step.
 *
 * @param crf - The starting model (labels, attributes and initial weights).
 * @param data - Labelled sequences.
 * @param options - Penalties, optimiser and stopping tolerance.
 * @returns The algorithm.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfTrainingAlgorithm } from 'whitebox-engine/pgm'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const alg = crfTrainingAlgorithm(linearChainCrf(['A', 'B'], ['f']), [{ x: [{ f: 1 }], y: ['A'] }], { c2: 1 })
 * runAlgorithm(alg, { maxSteps: 2000 }).status // => 'converged'
 * ```
 */
export function crfTrainingAlgorithm(
  crf: LinearChainCrf,
  data: readonly LabelledSequence[],
  options: CrfTrainingOptions = {},
): Algorithm<CrfTrainingState> {
  const method = options.method ?? adam({ learningRate: 0.1 })
  return gradientDescent(
    'crfTraining',
    {
      c1: options.c1 ?? 0,
      c2: options.c2 ?? 0,
      method: method.name,
      labels: [...crf.labels],
      attributes: crf.attributes.length,
      bigramAttributes: crf.bigramAttributes.length,
    } as Json,
    crfParameters(crf),
    (w) => crfObjective(withCrfParameters(crf, w), data, options),
    method,
    options.gradTol ?? 1e-6,
  )
}

/**
 * Fit a linear-chain CRF by penalised conditional maximum likelihood ({@link crfTrainingAlgorithm}). The labels and the
 * attribute vocabularies come from the starting model, or from the data when only options are given.
 *
 * @param data - Labelled sequences.
 * @param options - Penalties, optimiser, stopping rule, tracer and an optional starting model.
 * @returns The fitted model, the final objective and convergence information.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { fitCrf, crfDecode } from 'whitebox-engine/pgm'
 * const data = [{ x: [{ 'w=the': 1 }, { 'w=dog': 1 }], y: ['D', 'N'] }, { x: [{ 'w=a': 1 }, { 'w=cat': 1 }], y: ['D', 'N'] }]
 * const { crf } = fitCrf(data, { c2: 0.1, maxIter: 300 })
 * crfDecode(crf, [{ 'w=the': 1 }, { 'w=cat': 1 }]).labels // => ['D', 'N']
 * ```
 */
export function fitCrf(
  data: readonly LabelledSequence[],
  options: CrfTrainingOptions & { readonly start?: LinearChainCrf } = {},
): {
  readonly crf: LinearChainCrf
  readonly objective: number
  readonly converged: boolean
  readonly status: RunStatus
  readonly steps: number
} {
  const start =
    options.start ??
    linearChainCrf(
      [...new Set(data.flatMap((d) => d.y.map(String)))],
      featureIndex(data.map((d) => unigramOf(d.x))).features,
      { bigramAttributes: featureIndex(data.map((d) => (bigramOf(d.x) ?? []).slice(1))).features },
    )
  const result = runAlgorithm(crfTrainingAlgorithm(start, data, options), {
    maxSteps: options.maxIter ?? 1000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return {
    crf: withCrfParameters(start, result.state.parameters),
    objective: result.state.value,
    converged: result.status === 'converged',
    status: result.status,
    steps: result.steps,
  }
}

/**
 * A clique of a general CRF instance: its output variables and, for each joint assignment of them (row-major), the
 * active features as (feature index, value) pairs computed from the input x.
 */
export interface CrfClique {
  readonly variables: readonly string[]
  readonly features: readonly (readonly (readonly [number, number])[])[]
}

/**
 * A general (arbitrary-graph) conditional random field: output variables and a weight per feature; each input x is
 * given as an instance, its cliques with their feature values, so potentials are psi_c(y_c) = exp(sum_k w_k f_k(y_c,
 * x)). Linear chains, skip chains, grids and relational templates are all instances of it.
 *
 * @concept conditional-random-fields
 */
export interface GeneralCrf {
  readonly kind: 'crf'
  readonly variables: readonly DiscreteVariable[]
  readonly features: readonly string[]
  readonly weights: Float64Array
}

/** One input of a general CRF. */
export interface CrfInstance {
  readonly cliques: readonly CrfClique[]
}

/**
 * Build a general CRF.
 *
 * @param variables - The output variables.
 * @param features - Feature names.
 * @param weights - Initial weights (default zero).
 * @returns The model.
 * @throws WhiteboxError with code `shape` for weights of the wrong length.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { generalCrf } from 'whitebox-engine/pgm'
 * Array.from(generalCrf([{ name: 'y', states: ['0', '1'] }], ['bias']).weights) // => [0]
 * ```
 */
export function generalCrf(
  variables: readonly DiscreteVariable[],
  features: readonly string[],
  weights?: ArrayLike<number>,
): GeneralCrf {
  if (weights !== undefined && weights.length !== features.length)
    throw new WhiteboxError('shape', 'generalCrf: one weight per feature')
  return {
    kind: 'crf',
    variables: variables.map((v) => ({ ...v })),
    features: [...features],
    weights: weights === undefined ? new Float64Array(features.length) : Float64Array.from(weights),
  }
}

/**
 * The factor graph of a general CRF on one input: one factor per clique with log potential sum_k w_k f_k.
 *
 * @param crf - The model.
 * @param instance - The input's cliques and features.
 * @returns The conditional factor graph, ready for any inference routine.
 * @throws WhiteboxError with code `shape` for a clique with the wrong number of assignments.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { generalCrf, crfFactorGraph, factorValues } from 'whitebox-engine/pgm'
 * const crf = generalCrf([{ name: 'y', states: ['0', '1'] }], ['f'], [Math.log(2)])
 * crfFactorGraph(crf, { cliques: [{ variables: ['y'], features: [[], [[0, 1]]] }] }).factors.map(factorValues) // => [[1, 2]]
 * ```
 */
export function crfFactorGraph(crf: GeneralCrf, instance: CrfInstance): FactorGraph {
  const card = new Map(crf.variables.map((v) => [v.name, v.states.length]))
  const factors: Factor[] = instance.cliques.map((c) => {
    const cards = c.variables.map((v) => card.get(v) ?? 0)
    const size = cards.reduce((a, b) => a * b, 1)
    if (c.features.length !== size)
      throw new WhiteboxError('shape', `crfFactorGraph: clique [${c.variables.join(', ')}] needs ${size} assignments`)
    return logFactor(
      c.variables,
      cards,
      c.features.map((fs) => fs.reduce((s, [k, v]) => s + v * (crf.weights[k] ?? 0), 0)),
    )
  })
  return factorGraph(crf.variables, factors)
}

/** Clique marginals of an instance under the current weights: exactly (junction tree) or by loopy BP. */
function cliqueMarginals(
  fg: FactorGraph,
  instance: CrfInstance,
  inference: 'exact' | 'bp',
  bpOptions: Omit<BeliefPropagationOptions, 'mode' | 'evidence'>,
): { marginals: number[][]; logZ: number } {
  if (inference === 'exact') {
    const tree = calibrate(fg)
    return {
      marginals: instance.cliques.map((c) => Array.from(junctionTreeQuery(tree, c.variables).logValues, Math.exp)),
      logZ: tree.logPartition,
    }
  }
  const bp = beliefPropagation(fg, bpOptions)
  return {
    marginals: instance.cliques.map((c, i) => {
      const b = bp.factorBeliefs[i] as Factor
      return Array.from(factorMarginal(b, c.variables).logValues, Math.exp)
    }),
    logZ: bp.logPartition,
  }
}

/**
 * The training objective of a general CRF and its gradient: -sum log p(y | x) + c2 ||w||^2 + c1 ||w||_1, with clique
 * marginals from exact inference (a junction tree) or loopy belief propagation (the Bethe approximation).
 *
 * @param crf - The model.
 * @param data - Instances with their labellings (state names or indices by variable).
 * @param options - Penalties, the inference method and BP options.
 * @returns The objective and its gradient.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { generalCrf, generalCrfObjective } from 'whitebox-engine/pgm'
 * const crf = generalCrf([{ name: 'y', states: ['0', '1'] }], ['f'])
 * const r = generalCrfObjective(crf, [{ instance: { cliques: [{ variables: ['y'], features: [[], [[0, 1]]] }] }, y: { y: '1' } }])
 * [r.value === Math.LN2, Array.from(r.gradient)] // => [true, [-0.5]]
 * ```
 */
export function generalCrfObjective(
  crf: GeneralCrf,
  data: readonly { readonly instance: CrfInstance; readonly y: Readonly<Record<string, string | number>> }[],
  options: CrfPenalty & {
    readonly inference?: 'exact' | 'bp'
    readonly bp?: Omit<BeliefPropagationOptions, 'mode' | 'evidence'>
  } = {},
): { readonly value: number; readonly gradient: Float64Array } {
  const gradient = new Float64Array(crf.features.length)
  let value = 0
  for (const { instance, y } of data) {
    const fg = crfFactorGraph(crf, instance)
    const states = new Map(crf.variables.map((v) => [v.name, v.states]))
    const label = (v: string): number => {
      const raw = y[v]
      const i = typeof raw === 'number' ? raw : (states.get(v) ?? []).indexOf(String(raw))
      if (!(i >= 0)) throw new WhiteboxError('value', `generalCrfObjective: no label for ${v}`)
      return i
    }
    const { marginals, logZ } = cliqueMarginals(fg, instance, options.inference ?? 'exact', options.bp ?? {})
    let score = 0
    for (const [ci, c] of instance.cliques.entries()) {
      let idx = 0
      for (const v of c.variables) idx = idx * (states.get(v)?.length ?? 1) + label(v)
      for (const [k, v] of c.features[idx] ?? []) {
        score += v * (crf.weights[k] ?? 0)
        gradient[k] = (gradient[k] ?? 0) - v
      }
      const m = marginals[ci] ?? []
      for (const [e, fs] of c.features.entries())
        for (const [k, v] of fs) gradient[k] = (gradient[k] ?? 0) + v * (m[e] ?? 0)
    }
    value -= score - logZ
  }
  return { value: penalise(crf.weights, value, gradient, options), gradient }
}

/**
 * Fit a general CRF by penalised conditional maximum likelihood with a gradient transformation, using exact or loopy-BP
 * clique marginals, as a traced run.
 *
 * @param crf - The starting model.
 * @param data - Instances with their labellings.
 * @param options - Penalties, inference, optimiser and stopping rule.
 * @returns The fitted model, the final objective and convergence information.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { generalCrf, fitGeneralCrf } from 'whitebox-engine/pgm'
 * const crf = generalCrf([{ name: 'y', states: ['0', '1'] }], ['f'])
 * const inst = { cliques: [{ variables: ['y'], features: [[], [[0, 1]]] }] }
 * const r = fitGeneralCrf(crf, [{ instance: inst, y: { y: '1' } }, { instance: inst, y: { y: '1' } }, { instance: inst, y: { y: '0' } }], { maxIter: 3000, gradTol: 1e-8 })
 * Math.round((r.crf.weights[0] ?? 0) * 1e6) / 1e6 // => 0.693147
 * ```
 */
export function fitGeneralCrf(
  crf: GeneralCrf,
  data: readonly { readonly instance: CrfInstance; readonly y: Readonly<Record<string, string | number>> }[],
  options: CrfTrainingOptions & {
    readonly inference?: 'exact' | 'bp'
    readonly bp?: Omit<BeliefPropagationOptions, 'mode' | 'evidence'>
  } = {},
): {
  readonly crf: GeneralCrf
  readonly objective: number
  readonly converged: boolean
  readonly status: RunStatus
  readonly steps: number
} {
  const method = options.method ?? adam({ learningRate: 0.1 })
  const alg = gradientDescent(
    'generalCrfTraining',
    { c1: options.c1 ?? 0, c2: options.c2 ?? 0, method: method.name, inference: options.inference ?? 'exact' } as Json,
    Float64Array.from(crf.weights),
    (w) => generalCrfObjective({ ...crf, weights: w }, data, options),
    method,
    options.gradTol ?? 1e-6,
  )
  const result = runAlgorithm(alg, {
    maxSteps: options.maxIter ?? 1000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return {
    crf: { ...crf, weights: result.state.parameters },
    objective: result.state.value,
    converged: result.status === 'converged',
    status: result.status,
    steps: result.steps,
  }
}
