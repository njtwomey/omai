import { type Json, WhiteboxError } from '../../base/index.ts'
import type { GradientTransformation } from '../../optim/index.ts'
import { adam } from '../../optim/index.ts'
import { type Algorithm, type RunStatus, runAlgorithm, type TracePolicy, type Tracer } from '../../trace/index.ts'
import { type DescentState, gradientDescent } from '../_descent.ts'
import { type DiscreteVariable, type Factor, factorMarginal, logFactor } from '../factor.ts'
import { type BeliefPropagationOptions, beliefPropagation } from '../inference/belief-propagation.ts'
import { type ChainModel, forwardBackward, viterbi } from '../inference/chain.ts'
import { calibrate, junctionTreeQuery } from '../inference/junction-tree.ts'
import { type FactorGraph, factorGraph } from './factor-graph.ts'

/** The observation of one item of a sequence: attribute names with values (1 for an indicator, as CRFsuite). */
export type ItemFeatures = Readonly<Record<string, number>>

/** One token of a sequence as named columns (word, part of speech, ...), the input of attribute templates. */
export type Token = Readonly<Record<string, string>>

/** An attribute template: (column, offset) pairs joined into one attribute, as CRFsuite's `w[-1]|w[0]`. */
export type AttributeTemplate = readonly (readonly [string, number])[]

/**
 * Expand attribute templates over a sequence of tokens (CRFsuite-style feature templates): each template joins the
 * values of its (column, offset) cells into one attribute named like `w[-1]|w[0]=the|cat`. Cells outside the sequence
 * are skipped (`boundary: 'skip'`, CRFsuite) or padded with `_B-k` / `_B+k` (`'pad'`, CRF++'s convention). A `bias`
 * attribute fires at every position when asked for.
 *
 * @param tokens - The tokens.
 * @param templates - The templates.
 * @param options - Boundary handling and the bias attribute.
 * @returns The attributes of each position (value 1).
 * @concept crf-features
 * @example
 * ```ts
 * import { expandTemplates } from 'whitebox-engine/pgm'
 * const tokens = [{ w: 'the' }, { w: 'cat' }]
 * expandTemplates(tokens, [[['w', 0]], [['w', -1], ['w', 0]]]) // => [{ 'w[0]=the': 1 }, { 'w[0]=cat': 1, 'w[-1]|w[0]=the|cat': 1 }]
 * ```
 */
export function expandTemplates(
  tokens: readonly Token[],
  templates: readonly AttributeTemplate[],
  options: { readonly boundary?: 'skip' | 'pad'; readonly bias?: boolean } = {},
): Record<string, number>[] {
  const pad = options.boundary === 'pad'
  return tokens.map((_, t) => {
    const out: Record<string, number> = {}
    if (options.bias === true) out['bias'] = 1
    for (const template of templates) {
      const values: string[] = []
      let ok = true
      for (const [column, offset] of template) {
        const i = t + offset
        const v =
          i < 0
            ? pad
              ? `_B${i}`
              : undefined
            : i >= tokens.length
              ? pad
                ? `_B+${i - tokens.length + 1}`
                : undefined
              : tokens[i]?.[column]
        if (v === undefined) {
          ok = false
          break
        }
        values.push(v)
      }
      if (!ok) continue
      out[`${template.map(([c, o]) => `${c}[${o}]`).join('|')}=${values.join('|')}`] = 1
    }
    return out
  })
}

/**
 * The attribute vocabulary of a training set with each attribute's frequency, dropping those seen fewer than
 * `minFrequency` times (CRF++'s `-f`).
 *
 * @param sequences - Item features of each sequence.
 * @param options - The minimum frequency (default 1).
 * @returns The attributes (in first-seen order) and their counts.
 * @concept crf-features
 * @example
 * ```ts
 * import { attributeIndex } from 'whitebox-engine/pgm'
 * attributeIndex([[{ a: 1 }, { a: 1, b: 1 }]], { minFrequency: 2 }) // => { attributes: ['a'], counts: [2] }
 * ```
 */
export function attributeIndex(
  sequences: readonly (readonly ItemFeatures[])[],
  options: { readonly minFrequency?: number } = {},
): { readonly attributes: string[]; readonly counts: number[] } {
  const counts = new Map<string, number>()
  for (const s of sequences)
    for (const item of s) for (const a of Object.keys(item)) counts.set(a, (counts.get(a) ?? 0) + 1)
  const min = options.minFrequency ?? 1
  const kept = [...counts].filter(([, c]) => c >= min)
  return { attributes: kept.map(([a]) => a), counts: kept.map(([, c]) => c) }
}

/**
 * A linear-chain conditional random field (Lafferty, McCallum and Pereira 2001) in CRFsuite's form: state features
 * pair each attribute with each label (weight w(a, y)) and transition features pair consecutive labels (weight
 * w(y', y)); log p(y | x) = sum_t [sum_a x_t(a) w(a, y_t) + w(y_{t-1}, y_t)] - log Z(x).
 *
 * @concept conditional-random-fields
 */
export interface LinearChainCrf {
  readonly kind: 'linearChainCrf'
  readonly labels: readonly string[]
  readonly attributes: readonly string[]
  /** State weights, row-major [attribute][label]. */
  readonly stateWeights: Float64Array
  /** Transition weights, row-major [from][to]. */
  readonly transitionWeights: Float64Array
}

/**
 * Build a linear-chain CRF from its labels and attribute vocabulary, with zero weights or the given ones (dense
 * arrays, or nested records by attribute and label and by label pair).
 *
 * @param labels - The labels.
 * @param attributes - The attributes.
 * @param weights - State and transition weights.
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
    readonly state?: ArrayLike<number> | Readonly<Record<string, Readonly<Record<string, number>>>>
    readonly transition?: ArrayLike<number> | Readonly<Record<string, Readonly<Record<string, number>>>>
  } = {},
): LinearChainCrf {
  const K = labels.length
  const dense = (
    w: ArrayLike<number> | Readonly<Record<string, Readonly<Record<string, number>>>> | undefined,
    rows: readonly string[],
    name: string,
  ): Float64Array => {
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
  return {
    kind: 'linearChainCrf',
    labels: [...labels],
    attributes: [...attributes],
    stateWeights: dense(weights.state, attributes, 'state'),
    transitionWeights: dense(weights.transition, labels, 'transition'),
  }
}

/** Attribute indices of a model (attributes outside the vocabulary are ignored, as at test time in CRFsuite). */
function attributeLookup(crf: LinearChainCrf): Map<string, number> {
  return new Map(crf.attributes.map((a, i) => [a, i]))
}

/**
 * The potentials of a linear-chain CRF on one sequence, as a chain for forward-backward and Viterbi: unary log
 * potentials sum_a x_t(a) w(a, y) and the transition weights (no initial or final transition, as CRFsuite).
 *
 * @param crf - The model.
 * @param x - The item features of the sequence.
 * @returns The chain.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfChain } from 'whitebox-engine/pgm'
 * crfChain(linearChainCrf(['A', 'B'], ['f'], { state: [1, -1] }), [{ f: 2 }]).logEmission // => [[2, -2]]
 * ```
 */
export function crfChain(crf: LinearChainCrf, x: readonly ItemFeatures[]): ChainModel {
  const K = crf.labels.length
  const index = attributeLookup(crf)
  const logEmission = x.map((item) => {
    const row = new Array<number>(K).fill(0)
    for (const [a, v] of Object.entries(item)) {
      const i = index.get(a)
      if (i === undefined) continue
      for (let y = 0; y < K; y++) row[y] = (row[y] ?? 0) + v * (crf.stateWeights[i * K + y] ?? 0)
    }
    return row
  })
  const logTransition = Array.from({ length: K }, (_, i) =>
    Array.from(crf.transitionWeights.subarray(i * K, (i + 1) * K)),
  )
  return { logInitial: new Array<number>(K).fill(0), logTransition, logEmission }
}

function labelIndices(crf: LinearChainCrf, y: readonly (string | number)[]): number[] {
  return y.map((l) => {
    const i = typeof l === 'number' ? l : crf.labels.indexOf(l)
    if (!(i >= 0 && i < crf.labels.length)) throw new WhiteboxError('value', `crf: unknown label ${String(l)}`)
    return i
  })
}

/**
 * The unnormalised log score of a labelling: the sum of its state and transition weights.
 *
 * @param crf - The model.
 * @param x - The item features.
 * @param y - The labels.
 * @returns The score.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfScore } from 'whitebox-engine/pgm'
 * crfScore(linearChainCrf(['A', 'B'], ['f'], { state: [1, -1], transition: [0, 3, 0, 0] }), [{ f: 1 }, { f: 1 }], ['A', 'B']) // => 3
 * ```
 */
export function crfScore(crf: LinearChainCrf, x: readonly ItemFeatures[], y: readonly (string | number)[]): number {
  const chain = crfChain(crf, x)
  const idx = labelIndices(crf, y)
  let s = 0
  for (const [t, k] of idx.entries()) {
    s += chain.logEmission[t]?.[k] ?? 0
    if (t > 0) s += crf.transitionWeights[(idx[t - 1] ?? 0) * crf.labels.length + k] ?? 0
  }
  return s
}

/**
 * The conditional log-likelihood log p(y | x) of a labelling.
 *
 * @param crf - The model.
 * @param x - The item features.
 * @param y - The labels.
 * @returns log p(y | x).
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfLogLikelihood } from 'whitebox-engine/pgm'
 * crfLogLikelihood(linearChainCrf(['A', 'B'], ['f']), [{ f: 1 }, { f: 1 }], ['A', 'B']) === -Math.log(4) // => true
 * ```
 */
export function crfLogLikelihood(
  crf: LinearChainCrf,
  x: readonly ItemFeatures[],
  y: readonly (string | number)[],
): number {
  return crfScore(crf, x, y) - forwardBackward(crfChain(crf, x), { method: 'log' }).logLikelihood
}

/**
 * The marginal p(y_t | x) of every label at every position, by forward-backward.
 *
 * @param crf - The model.
 * @param x - The item features.
 * @returns For each position, the probability of each label (in label order).
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfMarginals } from 'whitebox-engine/pgm'
 * crfMarginals(linearChainCrf(['A', 'B'], ['f'], { state: [Math.log(3), 0] }), [{ f: 1 }]) // => [[0.75, 0.25]]
 * ```
 */
export function crfMarginals(crf: LinearChainCrf, x: readonly ItemFeatures[]): number[][] {
  if (x.length === 0) return []
  return forwardBackward(crfChain(crf, x)).posteriors
}

/**
 * The most probable labelling (Viterbi decoding) and its unnormalised score.
 *
 * @param crf - The model.
 * @param x - The item features.
 * @returns The labels and their score.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, crfDecode } from 'whitebox-engine/pgm'
 * crfDecode(linearChainCrf(['A', 'B'], ['f'], { state: [1, 0], transition: [0, 5, 0, 0] }), [{ f: 1 }, { f: 1 }]).labels // => ['A', 'B']
 * ```
 */
export function crfDecode(
  crf: LinearChainCrf,
  x: readonly ItemFeatures[],
): { readonly labels: string[]; readonly score: number } {
  if (x.length === 0) return { labels: [], score: 0 }
  const v = viterbi(crfChain(crf, x))
  return { labels: v.path.map((k) => crf.labels[k] ?? ''), score: v.logProbability }
}

/** A labelled training sequence. */
export interface LabelledSequence {
  readonly x: readonly ItemFeatures[]
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
 * The parameters of a linear-chain CRF as one vector: state weights then transition weights.
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
  const out = new Float64Array(crf.stateWeights.length + crf.transitionWeights.length)
  out.set(crf.stateWeights)
  out.set(crf.transitionWeights, crf.stateWeights.length)
  return out
}

/**
 * A copy of a linear-chain CRF with new parameters (state weights then transition weights).
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
  if (parameters.length !== n + crf.transitionWeights.length)
    throw new WhiteboxError('shape', 'withCrfParameters: wrong number of parameters')
  const p = Float64Array.from(parameters)
  return { ...crf, stateWeights: p.slice(0, n), transitionWeights: p.slice(n) }
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
 * @returns The objective and its gradient (state weights then transition weights).
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
  const gradient = new Float64Array(nState + crf.transitionWeights.length)
  const index = attributeLookup(crf)
  let value = 0
  for (const { x, y } of data) {
    if (x.length === 0) continue
    const idx = labelIndices(crf, y)
    if (idx.length !== x.length) throw new WhiteboxError('shape', 'crfObjective: one label per item')
    const chain = crfChain(crf, x)
    const fb = forwardBackward(chain)
    let score = 0
    for (const [t, item] of x.entries()) {
      const yt = idx[t] ?? 0
      score += chain.logEmission[t]?.[yt] ?? 0
      for (const [a, v] of Object.entries(item)) {
        const i = index.get(a)
        if (i === undefined) continue
        gradient[i * K + yt] = (gradient[i * K + yt] ?? 0) - v
        const post = fb.posteriors[t] ?? []
        for (let k = 0; k < K; k++) gradient[i * K + k] = (gradient[i * K + k] ?? 0) + v * (post[k] ?? 0)
      }
      if (t > 0) {
        const prev = idx[t - 1] ?? 0
        score += crf.transitionWeights[prev * K + yt] ?? 0
        gradient[nState + prev * K + yt] = (gradient[nState + prev * K + yt] ?? 0) - 1
        const xi = fb.pairwise[t - 1] ?? []
        for (let i = 0; i < K; i++)
          for (let j = 0; j < K; j++)
            gradient[nState + i * K + j] = (gradient[nState + i * K + j] ?? 0) + (xi[i]?.[j] ?? 0)
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
    } as Json,
    crfParameters(crf),
    (w) => crfObjective(withCrfParameters(crf, w), data, options),
    method,
    options.gradTol ?? 1e-6,
  )
}

/**
 * Fit a linear-chain CRF by penalised conditional maximum likelihood ({@link crfTrainingAlgorithm}). The labels and the
 * attribute vocabulary come from the starting model, or from the data when only options are given.
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
    linearChainCrf([...new Set(data.flatMap((d) => d.y.map(String)))], attributeIndex(data.map((d) => d.x)).attributes)
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
