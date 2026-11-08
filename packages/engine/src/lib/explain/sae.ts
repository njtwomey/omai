/**
 * Sparse autoencoders for the activations of a network (Bricken et al. 2023, Cunningham et al. 2023): a wide hidden
 * layer that must reconstruct an activation vector from a sparse code over a dictionary of unit-norm directions, so
 * that the directions come out as the features the network packs into its (narrower) space. Two ways of keeping the
 * code sparse: an L1 penalty on the code, or keeping only the k largest pre-activations (Gao et al. 2024). Training is
 * an `Algorithm` (one Adam step on a batch per step) that logs the loss, the reconstruction error, the mean number of
 * active features and the count of dead features (those that fire on nothing), and can resample the dead ones from
 * the worst-reconstructed inputs. `dictionaryRecovery` scores a learned dictionary against a planted one.
 */
import { valueAndGrad } from '../autodiff/index.ts'
import type { Json } from '../base/index.ts'
import { OmaiError } from '../base/index.ts'
import { matrixView } from '../learn/validation.ts'
import { adam, applyUpdates, type LearningRate } from '../optim/index.ts'
import { type Key, normal, permutation } from '../random/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  matmul,
  mean,
  mul,
  primalOf,
  relu,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'

/** The parameters of a sparse autoencoder: an encoder `[d, h]` with bias `[h]`, a decoder (the dictionary) `[h, d]` and `[d]`. */
export interface SaeParams {
  readonly encoder: Value
  readonly encoderBias: Value
  readonly decoder: Value
  readonly decoderBias: Value
}

/** Options of {@link saeAlgorithm}. */
export interface SaeOptions {
  /** The dictionary size (the hidden width). */
  readonly features: number
  /** How the code is kept sparse: an L1 penalty (default) or the `k` largest pre-activations. */
  readonly sparsity?: 'l1' | 'topk'
  /** The L1 weight (default 0.05; used by `l1`). */
  readonly l1Coefficient?: number
  /** The number of active features kept (default 2; used by `topk`). */
  readonly k?: number
  /** Adam's learning rate (default 0.01). */
  readonly learningRate?: LearningRate
  /** Rows per step (default all of them). */
  readonly batchSize?: number
  /** Resample the dead features from the worst-reconstructed inputs every this many steps (default never). */
  readonly resampleEvery?: number
  /** The number of steps before the algorithm reports completion (default 500). */
  readonly steps?: number
}

/** The state of {@link saeAlgorithm}. */
export interface SaeState {
  readonly params: SaeParams
  readonly optimiser: unknown
  readonly step: number
  /** The objective on the last batch. */
  readonly loss: number
  /** The mean squared reconstruction error per vector over all the data, after the update. */
  readonly reconstruction: number
  /** The mean number of active features per vector. */
  readonly l0: number
  /** The features that fire on no input. */
  readonly dead: number
}

/** The paths of {@link saeAlgorithm}. */
export type SaePath = 'sae/loss' | 'sae/reconstruction' | 'sae/l0' | 'sae/dead' | 'sae/dictionary'

/** What a sparse-autoencoder run logs. */
export const saeContract: Contract<SaeState, SaePath> = defineContract({
  'sae/loss': scalar({
    description: 'the objective on the batch: reconstruction plus the L1 penalty',
    direction: 'lower',
  }),
  'sae/reconstruction': scalar({
    description: 'the mean squared reconstruction error over the data',
    direction: 'lower',
  }),
  'sae/l0': scalar({ description: 'the mean number of active features per vector', direction: 'lower' }),
  'sae/dead': scalar({ description: 'the number of features that fire on no input', direction: 'lower' }),
  'sae/dictionary': tensorSource({
    role: 'param',
    shape: ['features', 'dim'],
    description: 'the dictionary directions, one per row',
  }),
})

const rowsNormalised = (m: Float64Array, rows: number, cols: number): Float64Array => {
  const out = new Float64Array(m.length)
  for (let i = 0; i < rows; i++) {
    let s = 0
    for (let j = 0; j < cols; j++) s += (m[i * cols + j] as number) ** 2
    const norm = Math.sqrt(s) || 1
    for (let j = 0; j < cols; j++) out[i * cols + j] = (m[i * cols + j] as number) / norm
  }
  return out
}

/** The mask keeping each row's `k` largest entries. */
function topkMask(pre: Value, k: number): Tensor {
  const flat = toFlat(primalOf(pre))
  const [n = 1, h = 1] = pre.shape
  const mask = new Float64Array(n * h)
  for (let i = 0; i < n; i++) {
    const idx = Array.from({ length: h }, (_, j) => j).sort(
      (a, b) => (flat[i * h + b] as number) - (flat[i * h + a] as number) || a - b,
    )
    for (const j of idx.slice(0, k)) mask[i * h + j] = 1
  }
  return fromStorage(mask, [n, h], 'float64')
}

function code(p: SaeParams, x: Value, mode: 'l1' | 'topk', k: number): Value {
  const pre = add(matmul(sub(x, p.decoderBias), p.encoder), p.encoderBias)
  return mode === 'topk' ? mul(relu(pre), topkMask(pre, k)) : relu(pre)
}

/**
 * The sparse code of activations under a trained autoencoder.
 *
 * @param params - The parameters ({@link SaeState.params}).
 * @param x - Activations `[n, d]`.
 * @param options - The sparsity mode and `k` it was trained with.
 * @returns The codes `[n, features]`.
 * @example
 * ```ts
 * import { saeEncode } from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * const p = { encoder: tensor([[1, 0], [0, 1]]), encoderBias: tensor([0, 0]), decoder: tensor([[1, 0], [0, 1]]), decoderBias: tensor([0, 0]) }
 * saeEncode(p, [[2, -1]]).shape // => [1, 2]
 * ```
 */
export function saeEncode(
  params: SaeParams,
  x: TensorLike,
  options: { readonly sparsity?: 'l1' | 'topk'; readonly k?: number } = {},
): Tensor {
  return primalOf(code(params, asTensor(x), options.sparsity ?? 'l1', options.k ?? 2))
}

/**
 * The reconstruction of codes under a trained autoencoder.
 *
 * @param params - The parameters.
 * @param codes - Codes `[n, features]`.
 * @returns The reconstructed activations `[n, d]`.
 * @example
 * ```ts
 * import { saeDecode } from 'omai-engine/explain'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * const p = { encoder: tensor([[1, 0], [0, 1]]), encoderBias: tensor([0, 0]), decoder: tensor([[1, 0], [0, 1]]), decoderBias: tensor([1, 1]) }
 * toFlat(saeDecode(p, [[2, 0]])) // => [3, 1]
 * ```
 */
export function saeDecode(params: SaeParams, codes: TensorLike): Tensor {
  return primalOf(add(matmul(asTensor(codes), params.decoder), params.decoderBias))
}

/** The measures of an autoencoder on data: reconstruction error, active features, dead features. */
function measure(
  p: SaeParams,
  x: Tensor,
  mode: 'l1' | 'topk',
  k: number,
): { reconstruction: number; l0: number; dead: number } {
  const codes = code(p, x, mode, k)
  const f = toFlat(primalOf(codes))
  const xhat = toFlat(primalOf(add(matmul(codes, p.decoder), p.decoderBias)))
  const xf = toFlat(x)
  const [n = 1, d = 1] = x.shape
  const h = p.decoder.shape[0] ?? 1
  let err = 0
  for (let i = 0; i < n * d; i++) err += ((xf[i] as number) - (xhat[i] as number)) ** 2
  let active = 0
  const fires = new Uint8Array(h)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < h; j++)
      if ((f[i * h + j] as number) > 0) {
        active++
        fires[j] = 1
      }
  return { reconstruction: err / n, l0: active / n, dead: fires.reduce((a, b) => a + (b === 0 ? 1 : 0), 0) }
}

/**
 * Train a sparse autoencoder on activations as an algorithm. The dictionary starts as random unit directions with the
 * encoder as its transpose, the decoder bias at the data mean, and the dictionary's rows are renormalised to unit norm
 * after every update. Each step is one Adam step on a batch of the objective `mean ||x - x_hat||^2 + lambda mean
 * ||f||_1` (`l1`) or on the reconstruction alone with the code cut to its `k` largest entries (`topk`); the run logs
 * the loss, the reconstruction error and the active and dead feature counts over the whole data. With
 * `resampleEvery`, features that fire on nothing are re-seeded at the worst-reconstructed inputs.
 *
 * @param activations - Activations `[n, d]`.
 * @param options - Dictionary size, sparsity mode and weights, optimiser, batching and resampling.
 * @returns The algorithm (its key seeds the dictionary and the batches).
 * @concept sparse-autoencoders
 * @example
 * ```ts
 * import { saeAlgorithm } from 'omai-engine/explain'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 40 }, (_, i) => (i % 2 === 0 ? [1, 0, 0] : [0, 1, 0]))
 * const r = runAlgorithm(saeAlgorithm(x, { features: 2, steps: 150, sparsity: 'topk', k: 1 }), { maxSteps: 150 })
 * r.state.reconstruction < 0.05 // => true
 * ```
 */
export function saeAlgorithm(activations: TensorLike, options: SaeOptions): Algorithm<SaeState, SaePath> {
  const m = matrixView(activations, 'saeAlgorithm')
  const { n, d } = m
  const h = options.features
  if (!Number.isInteger(h) || h < 1) throw new OmaiError('value', 'saeAlgorithm: features must be a positive integer')
  const mode = options.sparsity ?? 'l1'
  const lambda = options.l1Coefficient ?? 0.05
  const k = options.k ?? 2
  const steps = options.steps ?? 500
  const batch = Math.min(options.batchSize ?? n, n)
  const resampleEvery = options.resampleEvery ?? 0
  const x = fromStorage(Float64Array.from(m.data), [n, d], 'float64')
  const optimiser = adam({ learningRate: options.learningRate ?? 0.01 })
  const meanRow = Float64Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += m.data[i * d + j] as number
    return s / n
  })
  const fresh = (dictionary: Float64Array): SaeParams => ({
    encoder: fromStorage(
      Float64Array.from({ length: d * h }, (_, i) => dictionary[(i % h) * d + Math.floor(i / h)] as number),
      [d, h],
      'float64',
    ),
    encoderBias: fromStorage(new Float64Array(h), [h], 'float64'),
    decoder: fromStorage(dictionary, [h, d], 'float64'),
    decoderBias: fromStorage(Float64Array.from(meanRow), [d], 'float64'),
  })
  const objective = (params: SaeParams, rows: Value): Value => {
    const f = code(params, rows, mode, k)
    const xhat = add(matmul(f, params.decoder), params.decoderBias)
    const rec = mean(sum(square(sub(rows, xhat)), -1))
    return mode === 'l1' ? add(rec, mul(mean(sum(f, -1)), lambda)) : rec
  }
  const rowsOf = (indices: ArrayLike<number>): Tensor => {
    const out = new Float64Array(indices.length * d)
    for (let r = 0; r < indices.length; r++)
      for (let j = 0; j < d; j++) out[r * d + j] = m.data[(indices[r] as number) * d + j] as number
    return fromStorage(out, [indices.length, d], 'float64')
  }
  return {
    name: 'sparseAutoencoder',
    config: { features: h, sparsity: mode, l1Coefficient: lambda, k, batch, resampleEvery, steps } as Json,
    init: (key: Key) => {
      const raw = toFlat(normal(key, [h, d]))
      const dictionary = rowsNormalised(Float64Array.from(raw), h, d)
      const params = fresh(dictionary)
      return { params, optimiser: optimiser.init(params), step: 0, loss: Number.NaN, ...measure(params, x, mode, k) }
    },
    step: (s, ctx) => {
      const indices = batch === n ? undefined : Array.from(permutation(ctx.key, n).data).slice(0, batch)
      const rows = indices === undefined ? x : rowsOf(indices)
      const vg = valueAndGrad<SaeParams>((p) => objective(p, rows))
      const { value, grad } = vg(s.params)
      const { updates, state } = optimiser.update(grad, s.optimiser as unknown[], s.params)
      const moved = applyUpdates(s.params, updates) as SaeParams
      let params: SaeParams = {
        ...moved,
        decoder: fromStorage(
          rowsNormalised(Float64Array.from(toFlat(asTensor(moved.decoder))), h, d),
          [h, d],
          'float64',
        ),
      }
      let opt: unknown = state
      const t = s.step + 1
      let stats = measure(params, x, mode, k)
      if (resampleEvery > 0 && t % resampleEvery === 0 && stats.dead > 0) {
        // Re-seed each dead feature at the input the autoencoder reconstructs worst.
        const f = code(params, x, mode, k)
        const xhat = toFlat(primalOf(add(matmul(f, params.decoder), params.decoderBias)))
        const errors = Array.from({ length: n }, (_, i) => {
          let e = 0
          for (let j = 0; j < d; j++) e += ((m.data[i * d + j] as number) - (xhat[i * d + j] as number)) ** 2
          return [e, i] as const
        }).sort((a, b) => b[0] - a[0] || a[1] - b[1])
        const fl = toFlat(primalOf(f))
        const dictionary = Float64Array.from(toFlat(asTensor(params.decoder)))
        const encoder = Float64Array.from(toFlat(asTensor(params.encoder)))
        const encoderBias = Float64Array.from(toFlat(asTensor(params.encoderBias)))
        let next = 0
        for (let j = 0; j < h; j++) {
          let fires = false
          for (let i = 0; i < n && !fires; i++) fires = (fl[i * h + j] as number) > 0
          if (fires) continue
          const row = errors[next++ % n]?.[1] ?? 0
          const dir = Float64Array.from(
            { length: d },
            (_, q) => (m.data[row * d + q] as number) - (meanRow[q] as number),
          )
          const norm = Math.hypot(...dir) || 1
          for (let q = 0; q < d; q++) {
            dictionary[j * d + q] = (dir[q] as number) / norm
            encoder[q * h + j] = (dir[q] as number) / norm
          }
          encoderBias[j] = 0
        }
        params = {
          encoder: fromStorage(encoder, [d, h], 'float64'),
          encoderBias: fromStorage(encoderBias, [h], 'float64'),
          decoder: fromStorage(dictionary, [h, d], 'float64'),
          decoderBias: params.decoderBias,
        }
        opt = optimiser.init(params)
        stats = measure(params, x, mode, k)
      }
      const loss = toFlat(asTensor(value))[0] as number
      ctx.log('sae/loss', loss)
      ctx.log('sae/reconstruction', stats.reconstruction)
      ctx.log('sae/l0', stats.l0)
      ctx.log('sae/dead', stats.dead)
      if (ctx.wants('sae/dictionary')) ctx.log('sae/dictionary', asTensor(params.decoder) as Tensor)
      return { params, optimiser: opt, step: t, loss, ...stats }
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { loss: s.loss, reconstruction: s.reconstruction, l0: s.l0, dead: s.dead },
      tensors: { dictionary: { role: 'param', value: s.params.decoder } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SaeState,
  }
}

/** How well a learned dictionary contains a planted one. */
export interface DictionaryRecovery {
  /** For each planted direction, the largest absolute cosine with any learned direction. */
  readonly similarity: readonly number[]
  /** The mean of {@link similarity}. */
  readonly mean: number
  /** For each planted direction, the learned direction that matched it. */
  readonly match: readonly number[]
}

/**
 * Score a learned dictionary against a planted one: for each true direction, the best absolute cosine similarity with
 * a learned direction (the directions are recovered up to sign and order). A mean near 1 is full recovery.
 *
 * @param truth - The planted directions, one per row.
 * @param learned - The learned directions, one per row `[features, d]`.
 * @returns The similarity of each planted direction, their mean and which learned direction matched.
 * @example
 * ```ts
 * import { dictionaryRecovery } from 'omai-engine/explain'
 * dictionaryRecovery([[1, 0], [0, 1]], [[0, -2], [3, 0.01]]).match // => [1, 0]
 * ```
 */
export function dictionaryRecovery(truth: readonly (readonly number[])[], learned: TensorLike): DictionaryRecovery {
  const l = matrixView(learned, 'dictionaryRecovery')
  const norm = (v: ArrayLike<number>): number => Math.hypot(...Array.from(v)) || 1
  const similarity: number[] = []
  const match: number[] = []
  for (const t of truth) {
    if (t.length !== l.d) throw new OmaiError('shape', 'dictionaryRecovery: directions must share one dimension')
    let best = -1
    let arg = 0
    for (let j = 0; j < l.n; j++) {
      const row = l.data.subarray(j * l.d, (j + 1) * l.d)
      let dot = 0
      for (let q = 0; q < l.d; q++) dot += (t[q] as number) * (row[q] as number)
      const cos = Math.abs(dot) / (norm(t) * norm(row))
      if (cos > best) {
        best = cos
        arg = j
      }
    }
    similarity.push(best)
    match.push(arg)
  }
  return { similarity, mean: similarity.reduce((a, b) => a + b, 0) / similarity.length, match }
}
