/**
 * The CRF layer of a neural tagger, pytorch-crf's `torchcrf.CRF` (and AllenNLP's `ConditionalRandomField`): emission
 * scores `[B, T, K]` come from an encoder, the label-to-label `transitions` `[K, K]` and the `startTransitions` and
 * `endTransitions` `[K]` are the layer's parameters, and the loss is the negative log-likelihood of the gold tags. The
 * forward algorithm runs in log space over tensors, so `grad` of the loss reaches the emissions (and through them the
 * encoder) and the three parameter tensors; padding is a boolean mask of right-padded sequences. Prediction is exact
 * Viterbi (the chain code of `pgm/inference`, with the start and end scores folded into the first and last step),
 * optionally restricted to allowed label transitions, and the marginals are the gradient of log Z with respect to the
 * emissions. The functions take the parameters as a record; `CRF` is the same layer as a module with torchcrf's
 * parameter names and initialisation.
 */

import { grad } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { type ApplyContext, initKey, Module, type Params, uniformInit } from '../../nn/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  div,
  expandDims,
  logsumexp,
  mean,
  mul,
  reshape,
  sliceAxis,
  sub,
  sum,
  type TensorLike,
  take,
  tensor,
  toFlat,
  transpose,
  type Value,
  where,
} from '../../tensor/index.ts'
import { viterbi } from '../inference/chain.ts'

/**
 * How a batch of log-likelihoods is reduced, as in torchcrf: `none` keeps one per sequence, `sum` and `mean` reduce
 * over the sequences, `token_mean` divides the sum by the number of unmasked tokens.
 */
export type NeuralCrfReduction = 'none' | 'sum' | 'mean' | 'token_mean'

/** The three parameter tensors of a CRF layer over `K` tags. */
export interface NeuralCrfParameters {
  /** `[K, K]`: the score of moving from tag i to tag j. */
  readonly transitions: Value
  /** `[K]`: the score of starting a sequence in each tag. */
  readonly startTransitions: Value
  /** `[K]`: the score of ending a sequence in each tag. */
  readonly endTransitions: Value
}

/** A mask of a batch: true or 1 for a real token. */
export type NeuralCrfMask = TensorLike | readonly (readonly boolean[])[]

/** Layout and masking of a batch: emissions, tags and mask are `[B, T]`-first when `batchFirst`, else `[T, B]`-first. */
export interface NeuralCrfOptions {
  /** Whether the first axis is the batch (default false, as torchcrf: `[T, B, K]`). */
  readonly batchFirst?: boolean
  /**
   * Which positions are real tokens, `[T, B]` (`[B, T]` with `batchFirst`). Sequences are right-padded: the mask of
   * each starts with a run of ones, and the first position of every sequence must be on. Default: every position.
   */
  readonly mask?: NeuralCrfMask
}

/** A restriction of the tag sequences Viterbi may return: false entries are forbidden (scored minus infinity). */
export interface NeuralCrfAllowed {
  /** `[K][K]`: whether tag j may follow tag i. */
  readonly transitions?: readonly (readonly boolean[])[]
  /** `[K]`: tags a sequence may start with. */
  readonly start?: readonly boolean[]
  /** `[K]`: tags a sequence may end with. */
  readonly end?: readonly boolean[]
}

/** The best tag paths of a batch and their scores. */
export interface NeuralCrfDecoding {
  /** One path per sequence, as long as its unmasked length. */
  readonly paths: number[][]
  /** The score of each path (the unnormalised log-probability: emissions, transitions, start and end). */
  readonly scores: number[]
}

const UNIFORM_BOUND = 0.1

/**
 * Fresh parameters of a CRF layer, initialised as torchcrf does: all three uniform on (-0.1, 0.1).
 *
 * @param numTags - The number of tags K.
 * @param key - The random key (default: the next key of the default context).
 * @returns The parameters.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { neuralCrfParameters } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * neuralCrfParameters(3, key(0)).transitions.shape // => [3, 3]
 * ```
 */
export function neuralCrfParameters(numTags: number, key?: Key): NeuralCrfParameters {
  if (!(Number.isInteger(numTags) && numTags > 0)) throw new OmaiError('value', 'neuralCrf: numTags must be positive')
  const k = initKey('CRF', key)
  const u = (i: number, shape: number[]): Value => uniformInit(foldIn(k, i), shape, -UNIFORM_BOUND, UNIFORM_BOUND)
  return { startTransitions: u(0, [numTags]), endTransitions: u(1, [numTags]), transitions: u(2, [numTags, numTags]) }
}

/** A rectangular array of numbers as rows (booleans become 0 and 1). */
function grid(x: NeuralCrfMask | TensorLike, name: string): number[][] {
  const numeric = Array.isArray(x)
    ? x.map((r) => (Array.isArray(r) ? r.map((v: unknown) => (typeof v === 'boolean' ? Number(v) : v)) : r))
    : x
  const t = asTensor(numeric as TensorLike)
  if (t.shape.length !== 2) throw new OmaiError('shape', `${name}: expected a 2-D array, got [${t.shape.join(', ')}]`)
  const [rows = 0, n = 0] = t.shape
  const flat = toFlat(t)
  return Array.from({ length: rows }, (_, i) => flat.slice(i * n, (i + 1) * n))
}

const transposeGrid = (g: number[][]): number[][] => (g[0] ?? []).map((_, t) => g.map((row) => row[t] ?? 0))

/** A batch in batch-first layout, validated. */
interface Prepared {
  /** Emissions `[B, T, K]`. */
  readonly e: Value
  readonly B: number
  readonly T: number
  readonly K: number
  readonly batchFirst: boolean
  /** `[B][T]`, 1 for a token. */
  readonly mask: number[][]
  readonly lengths: number[]
}

function prepare(emissions: TensorLike, numTags: number, options: NeuralCrfOptions, name: string): Prepared {
  const raw = asTensor(emissions)
  if (raw.shape.length !== 3)
    throw new OmaiError('shape', `${name}: emissions must be 3-D, got [${raw.shape.join(', ')}]`)
  const batchFirst = options.batchFirst ?? false
  const e = batchFirst ? raw : transpose(raw, [1, 0, 2])
  const [B = 0, T = 0, K = 0] = e.shape
  if (K !== numTags) throw new OmaiError('shape', `${name}: expected ${numTags} tags in the last axis, got ${K}`)
  if (T === 0 || B === 0) throw new OmaiError('shape', `${name}: empty batch or sequences`)
  let mask: number[][]
  if (options.mask === undefined) mask = Array.from({ length: B }, () => new Array<number>(T).fill(1))
  else {
    const g = grid(options.mask, `${name}: mask`)
    mask = batchFirst ? g : transposeGrid(g)
  }
  if (mask.length !== B || mask.some((r) => r.length !== T))
    throw new OmaiError('shape', `${name}: mask does not match the emissions' batch and length`)
  const lengths = mask.map((r, b) => {
    const n = r.reduce((a, v) => a + (v === 0 ? 0 : 1), 0)
    if (n === 0 || r.some((v, t) => (v !== 0) !== t < n))
      throw new OmaiError('value', `${name}: sequence ${b} must start at the first position and be right-padded`)
    return n
  })
  return { e, B, T, K, batchFirst, mask, lengths }
}

function tagGrid(tags: TensorLike, p: Prepared, name: string): number[][] {
  const g = grid(tags, `${name}: tags`)
  const t = p.batchFirst ? g : transposeGrid(g)
  if (t.length !== p.B || t.some((r) => r.length !== p.T))
    throw new OmaiError('shape', `${name}: tags do not match the emissions' batch and length`)
  t.forEach((row, b) => {
    for (let s = 0; s < (p.lengths[b] ?? 0); s++) {
      const k = row[s] ?? 0
      if (!(Number.isInteger(k) && k >= 0 && k < p.K))
        throw new OmaiError('value', `${name}: tag ${k} of sequence ${b} is outside [0, ${p.K})`)
    }
  })
  // Masked positions may hold anything; point them at a valid tag.
  return t.map((row, b) => row.map((k, s) => (s < (p.lengths[b] ?? 0) ? k : 0)))
}

const maskColumn = (rows: number[][], t: number): Value =>
  tensor(
    rows.map((r) => [(r[t] ?? 0) !== 0]),
    { dtype: 'bool' },
  )

/** log Z of a prepared batch: the forward algorithm, the alpha of a finished sequence is carried through padding. */
function partition(p: Prepared, params: NeuralCrfParameters): Value {
  const { e, B, T, K } = p
  const at = (t: number): Value => reshape(sliceAxis(e, 1, t, t + 1), [B, K])
  let alpha = add(at(0), expandDims(params.startTransitions, 0))
  const moves = expandDims(params.transitions, 0)
  for (let t = 1; t < T; t++) {
    // next[b, j] = logsumexp over i of alpha[b, i] + transitions[i, j], plus emissions[b, t, j]
    const next = add(logsumexp(add(expandDims(alpha, 2), moves), 1), at(t))
    alpha = where(maskColumn(p.mask, t), next, alpha)
  }
  return logsumexp(add(alpha, expandDims(params.endTransitions, 0)), 1)
}

/** The score of the gold tags of a prepared batch, vectorised over positions with gathers. */
function goldScore(p: Prepared, tags: number[][], params: NeuralCrfParameters): Value {
  const { e, B, T, K } = p
  const flat: number[] = []
  const on: number[] = []
  for (let b = 0; b < B; b++)
    for (let t = 0; t < T; t++) {
      flat.push((b * T + t) * K + (tags[b]?.[t] ?? 0))
      on.push(p.mask[b]?.[t] === 0 ? 0 : 1)
    }
  let total = sum(reshape(mul(take(reshape(e, [-1]), flat), tensor(on)), [B, T]), 1)
  if (T > 1) {
    const pair: number[] = []
    const live: number[] = []
    for (let b = 0; b < B; b++)
      for (let t = 1; t < T; t++) {
        pair.push((tags[b]?.[t - 1] ?? 0) * K + (tags[b]?.[t] ?? 0))
        live.push(p.mask[b]?.[t] === 0 ? 0 : 1)
      }
    const moves = mul(take(reshape(params.transitions, [-1]), pair), tensor(live))
    total = add(total, sum(reshape(moves, [B, T - 1]), 1))
  }
  const first = tags.map((r) => r[0] ?? 0)
  const last = tags.map((r, b) => r[(p.lengths[b] ?? 1) - 1] ?? 0)
  return add(add(total, take(params.startTransitions, first)), take(params.endTransitions, last))
}

const sizeOfTags = (params: NeuralCrfParameters): number => params.startTransitions.shape[0] ?? 0

/**
 * The unnormalised log-score of given tag sequences: the emission of each tagged position, the transition between
 * consecutive tags, the start score of the first tag and the end score of the last.
 *
 * @param emissions - Emission scores, `[T, B, K]` (`[B, T, K]` with `batchFirst`); may be traced.
 * @param tags - The tags, `[T, B]` (`[B, T]` with `batchFirst`); entries at masked positions are ignored.
 * @param params - The CRF parameters.
 * @param options - Layout and mask.
 * @returns The scores, `[B]`.
 * @throws OmaiError with code `shape` or `value` for inconsistent shapes, a tag out of range or a mask that is not
 *   right-padded.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { neuralCrfScore } from 'omai-engine/pgm'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const params = {
 *   transitions: tensor([[0, 1], [2, 3]]),
 *   startTransitions: tensor([10, 20]),
 *   endTransitions: tensor([100, 200]),
 * }
 * toArray(neuralCrfScore([[[1, 2], [3, 4]]], [[1, 0]], params, { batchFirst: true })) // => [127]
 * ```
 */
export function neuralCrfScore(
  emissions: TensorLike,
  tags: TensorLike,
  params: NeuralCrfParameters,
  options: NeuralCrfOptions = {},
): Value {
  const p = prepare(emissions, sizeOfTags(params), options, 'neuralCrfScore')
  return goldScore(p, tagGrid(tags, p, 'neuralCrfScore'), params)
}

/**
 * The log partition function log Z of each sequence (the log-sum of the scores of every tag path), by the forward
 * algorithm in log space; differentiable in the emissions and the parameters.
 *
 * @param emissions - Emission scores, `[T, B, K]` (`[B, T, K]` with `batchFirst`); may be traced.
 * @param params - The CRF parameters.
 * @param options - Layout and mask.
 * @returns log Z, `[B]`.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { neuralCrfPartition } from 'omai-engine/pgm'
 * import { toArray, zeros } from 'omai-engine/tensor'
 * const params = { transitions: zeros([2, 2]), startTransitions: zeros([2]), endTransitions: zeros([2]) }
 * toArray(neuralCrfPartition(zeros([1, 3, 2]), params, { batchFirst: true })).map((z) => Math.round(Number(z) * 1000)) // => [2079]
 * ```
 */
export function neuralCrfPartition(
  emissions: TensorLike,
  params: NeuralCrfParameters,
  options: NeuralCrfOptions = {},
): Value {
  return partition(prepare(emissions, sizeOfTags(params), options, 'neuralCrfPartition'), params)
}

/**
 * The log-likelihood of tag sequences under the layer (torchcrf's `CRF.forward`): score minus log Z, reduced as asked.
 * Differentiable in the emissions and the parameters, so `grad` of it trains an encoder and the transitions together.
 *
 * @param emissions - Emission scores, `[T, B, K]` (`[B, T, K]` with `batchFirst`); may be traced.
 * @param tags - The gold tags, `[T, B]` (`[B, T]` with `batchFirst`).
 * @param params - The CRF parameters.
 * @param options - Layout, mask and the reduction (default `sum`).
 * @returns The log-likelihood: `[B]` for `none`, a scalar otherwise.
 * @concept conditional-random-fields
 * @index function CRF layer: log-likelihood of tag sequences from emissions, differentiable in the encoder.
 * @example
 * ```ts
 * import { neuralCrfLogLikelihood } from 'omai-engine/pgm'
 * import { item, zeros } from 'omai-engine/tensor'
 * const params = { transitions: zeros([2, 2]), startTransitions: zeros([2]), endTransitions: zeros([2]) }
 * Math.round(item(neuralCrfLogLikelihood(zeros([1, 3, 2]), [[0, 1, 0]], params, { batchFirst: true })) * 1000) // => -2079
 * ```
 */
export function neuralCrfLogLikelihood(
  emissions: TensorLike,
  tags: TensorLike,
  params: NeuralCrfParameters,
  options: NeuralCrfOptions & { readonly reduction?: NeuralCrfReduction } = {},
): Value {
  const p = prepare(emissions, sizeOfTags(params), options, 'neuralCrfLogLikelihood')
  const llh = sub(goldScore(p, tagGrid(tags, p, 'neuralCrfLogLikelihood'), params), partition(p, params))
  switch (options.reduction ?? 'sum') {
    case 'none':
      return llh
    case 'sum':
      return sum(llh)
    case 'mean':
      return mean(llh)
    case 'token_mean':
      return div(
        sum(llh),
        p.lengths.reduce((a, n) => a + n, 0),
      )
  }
}

/**
 * The negative log-likelihood, the loss of a CRF tagger: {@link neuralCrfLogLikelihood} negated, by default averaged
 * over the sequences.
 *
 * @param emissions - Emission scores, `[T, B, K]` (`[B, T, K]` with `batchFirst`); may be traced.
 * @param tags - The gold tags, laid out like the emissions' first two axes.
 * @param params - The CRF parameters.
 * @param options - Layout, mask and the reduction (default `mean`).
 * @returns The loss: `[B]` for `none`, a scalar otherwise.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { neuralCrfNegLogLikelihood } from 'omai-engine/pgm'
 * import { item, zeros } from 'omai-engine/tensor'
 * const params = { transitions: zeros([2, 2]), startTransitions: zeros([2]), endTransitions: zeros([2]) }
 * Math.round(item(neuralCrfNegLogLikelihood(zeros([1, 3, 2]), [[0, 1, 0]], params, { batchFirst: true })) * 1000) // => 2079
 * ```
 */
export function neuralCrfNegLogLikelihood(
  emissions: TensorLike,
  tags: TensorLike,
  params: NeuralCrfParameters,
  options: NeuralCrfOptions & { readonly reduction?: NeuralCrfReduction } = {},
): Value {
  return mul(neuralCrfLogLikelihood(emissions, tags, params, { reduction: 'mean', ...options }), -1)
}

/**
 * The exact best tag path of each sequence (torchcrf's `CRF.decode`, with scores): Viterbi over the masked length of
 * each sequence, the start score added to the first emission and the end score to the last. With `allowed`, paths
 * that use a forbidden start, transition or end are excluded, so every decoded sequence is valid (no I-X after O); it
 * throws when no allowed path exists. Works on concrete values (it reads the emissions' numbers).
 *
 * @param emissions - Emission scores, `[T, B, K]` (`[B, T, K]` with `batchFirst`), concrete.
 * @param params - The CRF parameters.
 * @param options - Layout, mask and the allowed transitions.
 * @returns The paths and their scores.
 * @throws OmaiError with code `value` when every path of a sequence is forbidden.
 * @concept conditional-random-fields
 * @concept viterbi
 * @index function Exact batched, masked Viterbi decoding of a CRF layer, optionally with allowed transitions.
 * @example
 * ```ts
 * import { neuralCrfDecode } from 'omai-engine/pgm'
 * import { tensor, zeros } from 'omai-engine/tensor'
 * const params = { transitions: zeros([2, 2]), startTransitions: zeros([2]), endTransitions: zeros([2]) }
 * neuralCrfDecode(tensor([[[1, 0], [0, 2], [3, 0]]]), params, { batchFirst: true }).paths // => [[0, 1, 0]]
 * ```
 */
export function neuralCrfDecode(
  emissions: TensorLike,
  params: NeuralCrfParameters,
  options: NeuralCrfOptions & { readonly allowed?: NeuralCrfAllowed } = {},
): NeuralCrfDecoding {
  const p = prepare(emissions, sizeOfTags(params), options, 'neuralCrfDecode')
  const { K, T } = p
  const E = toFlat(p.e)
  const start = toFlat(params.startTransitions)
  const end = toFlat(params.endTransitions)
  const A = toFlat(params.transitions)
  const allowed = options.allowed
  const ban = (ok: boolean | undefined): number => (ok === false ? Number.NEGATIVE_INFINITY : 0)
  const logTransition = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) => (A[i * K + j] ?? 0) + ban(allowed?.transitions?.[i]?.[j])),
  )
  const logInitial = start.map((s, k) => s + ban(allowed?.start?.[k]))
  const paths: number[][] = []
  const scores: number[] = []
  for (let b = 0; b < p.B; b++) {
    const length = p.lengths[b] ?? 1
    const logEmission = Array.from({ length }, (_, t) =>
      Array.from({ length: K }, (_, k) => {
        const v = E[(b * T + t) * K + k] ?? 0
        return t === length - 1 ? v + (end[k] ?? 0) + ban(allowed?.end?.[k]) : v
      }),
    )
    const r = viterbi({ logInitial, logTransition, logEmission })
    paths.push(r.path)
    scores.push(r.logProbability)
  }
  return { paths, scores }
}

/**
 * The posterior marginals of every tag at every position, `p(y_t = k | emissions)`: the gradient of log Z with respect
 * to the emissions (log Z is the log-sum-exp of the path scores, so its gradient is the expected one-hot tag), taken
 * with `grad` through the forward algorithm. Masked positions are zero.
 *
 * @param emissions - Emission scores, `[T, B, K]` (`[B, T, K]` with `batchFirst`).
 * @param params - The CRF parameters.
 * @param options - Layout and mask.
 * @returns The marginals, laid out like the emissions.
 * @concept conditional-random-fields
 * @concept forward-backward
 * @example
 * ```ts
 * import { neuralCrfMarginals } from 'omai-engine/pgm'
 * import { toArray, zeros } from 'omai-engine/tensor'
 * const params = { transitions: zeros([2, 2]), startTransitions: zeros([2]), endTransitions: zeros([2]) }
 * toArray(neuralCrfMarginals(zeros([1, 2, 2]), params, { batchFirst: true })) // => [[[0.5, 0.5], [0.5, 0.5]]]
 * ```
 */
export function neuralCrfMarginals(
  emissions: TensorLike,
  params: NeuralCrfParameters,
  options: NeuralCrfOptions = {},
): Value {
  const raw = asTensor(emissions)
  prepare(raw, sizeOfTags(params), options, 'neuralCrfMarginals')
  return grad((e: Value) => sum(neuralCrfPartition(e, params, options)))(raw)
}

/** The input of {@link CRF}: emissions and tags with an optional mask and reduction, as torchcrf's `forward` takes. */
export interface NeuralCrfInput {
  readonly emissions: TensorLike
  readonly tags: TensorLike
  readonly mask?: NeuralCrfMask
  /** Default `sum` (torchcrf's default is `sum`). */
  readonly reduction?: NeuralCrfReduction
}

/** Options of {@link CRF}. */
export interface CrfModuleOptions {
  /** Whether emissions, tags and masks are batch-first `[B, T, ...]` (default false: `[T, B, ...]`, as torchcrf). */
  readonly batchFirst?: boolean
  readonly key?: Key
}

const LAYER_KEYS = { start: 'start_transitions', end: 'end_transitions', transitions: 'transitions' } as const

/**
 * A conditional random field layer over a neural encoder (`torchcrf.CRF`): parameters `start_transitions`,
 * `end_transitions` and `transitions` (uniform on (-0.1, 0.1) as in torchcrf, same state-dict names); `forward` takes
 * `{ emissions, tags, mask, reduction }` and returns the log-likelihood of the tags, `decode` the Viterbi paths. The
 * functional form `apply(params, input)` is what `grad` differentiates, in the emissions and the parameters.
 *
 * @concept conditional-random-fields
 * @index module CRF layer over emission scores: transition, start and end parameters, log-likelihood, Viterbi.
 * @example
 * ```ts
 * import { CRF } from 'omai-engine/pgm'
 * import { zeros } from 'omai-engine/tensor'
 * const crf = new CRF(3, { batchFirst: true })
 * Object.keys(crf.stateDict()) // => ['end_transitions', 'start_transitions', 'transitions']
 * crf.decode(zeros([2, 4, 3])).paths.length // => 2
 * ```
 */
export class CRF extends Module<NeuralCrfInput, Value> {
  readonly kind = 'CRF'
  readonly numTags: number
  readonly batchFirst: boolean

  constructor(numTags: number, options: CrfModuleOptions = {}) {
    super()
    this.numTags = numTags
    this.batchFirst = options.batchFirst ?? false
    this.params = this.initParams(initKey('CRF', options.key))
  }

  override initParams(key: Key): Params {
    const p = neuralCrfParameters(this.numTags, key)
    return {
      [LAYER_KEYS.start]: p.startTransitions,
      [LAYER_KEYS.end]: p.endTransitions,
      [LAYER_KEYS.transitions]: p.transitions,
    }
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  /** The parameters as the record the functions take. */
  layerParameters(params: Params = this.params): NeuralCrfParameters {
    return {
      transitions: params[LAYER_KEYS.transitions] as Value,
      startTransitions: params[LAYER_KEYS.start] as Value,
      endTransitions: params[LAYER_KEYS.end] as Value,
    }
  }

  private layout(mask: NeuralCrfMask | undefined): NeuralCrfOptions {
    return { batchFirst: this.batchFirst, ...(mask === undefined ? {} : { mask }) }
  }

  apply(params: Params, x: NeuralCrfInput, _ctx: ApplyContext = {}): Value {
    return neuralCrfLogLikelihood(x.emissions, x.tags, this.layerParameters(params), {
      ...this.layout(x.mask),
      ...(x.reduction === undefined ? {} : { reduction: x.reduction }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [inputShape[this.batchFirst ? 0 : 1] ?? 0]
  }

  /** The negative log-likelihood, averaged over the sequences by default. */
  negLogLikelihood(input: NeuralCrfInput, params: Params = this.params): Value {
    return neuralCrfNegLogLikelihood(input.emissions, input.tags, this.layerParameters(params), {
      ...this.layout(input.mask),
      ...(input.reduction === undefined ? {} : { reduction: input.reduction }),
    })
  }

  /** log Z of each sequence. */
  partition(emissions: TensorLike, mask?: NeuralCrfMask, params: Params = this.params): Value {
    return neuralCrfPartition(emissions, this.layerParameters(params), this.layout(mask))
  }

  /** The best paths (torchcrf's `decode`), optionally restricted to allowed transitions. */
  decode(
    emissions: TensorLike,
    options: { readonly mask?: NeuralCrfMask; readonly allowed?: NeuralCrfAllowed } = {},
    params: Params = this.params,
  ): NeuralCrfDecoding {
    return neuralCrfDecode(emissions, this.layerParameters(params), {
      ...this.layout(options.mask),
      ...(options.allowed === undefined ? {} : { allowed: options.allowed }),
    })
  }

  /** The posterior marginals of every tag at every position. */
  marginals(emissions: TensorLike, mask?: NeuralCrfMask, params: Params = this.params): Value {
    return neuralCrfMarginals(emissions, this.layerParameters(params), this.layout(mask))
  }

  override extraRepr(): string {
    return `num_tags=${this.numTags}${this.batchFirst ? ', batch_first=True' : ''}`
  }
}
