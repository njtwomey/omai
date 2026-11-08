/**
 * Normalisation layers as functions, torch.nn.functional style: batch, layer, RMS, group, instance and local response
 * normalisation, Lp normalisation along an axis, and the weight reparametrisations: weight normalisation (`w = g v /
 * ||v||`) and spectral normalisation, which divides a weight by its largest singular value estimated by power
 * iteration.
 */

import { OmaiError } from '../../base/index.ts'
import {
  add,
  concat,
  div,
  matmul,
  maximum,
  mean,
  moveaxis,
  mul,
  norm,
  pow,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  square,
  stopGradient,
  sub,
  sum,
  swapaxes,
  type TensorLike,
  type Value,
  variance,
  zeros,
} from '../../tensor/index.ts'

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

/** `[1, C, 1, ...]` for a per-channel vector broadcast against `[B, C, *S]`. */
const channelShape = (v: Value, rank: number): Value => reshape(v, [1, -1, ...new Array<number>(rank - 2).fill(1)])

/** Options of {@link batchNorm}. */
export interface BatchNormOptions {
  /** Running mean `[C]` (absent: batch statistics are always used and none are returned). */
  readonly runningMean?: TensorLike
  /** Running variance `[C]`. */
  readonly runningVar?: TensorLike
  /** Scale `[C]`. */
  readonly weight?: TensorLike
  /** Shift `[C]`. */
  readonly bias?: TensorLike
  /** Normalise by batch statistics and update the running ones (default true). */
  readonly training?: boolean
  /** Weight of the new batch in the running averages (default 0.1); `null` is a cumulative average. */
  readonly momentum?: number | null
  /** Batches seen before this one (for the cumulative average). */
  readonly numBatchesTracked?: number
  readonly eps?: number
}

/** Result of {@link batchNorm}. */
export interface BatchNormResult {
  readonly output: Value
  /** The updated running mean (unchanged in evaluation; absent without running statistics). */
  readonly runningMean?: Value
  /** The updated running variance (updated with the unbiased batch variance, as PyTorch). */
  readonly runningVar?: Value
}

/**
 * Batch normalisation (torch.nn.functional.batch_norm, Ioffe and Szegedy 2015): normalise each channel (axis 1) by
 * its mean and variance over the batch and spatial axes, then scale and shift. In training the batch statistics are
 * used and the running averages are updated and returned (running_mean <- (1 - m) running_mean + m mean, the variance
 * term unbiased); in evaluation the running statistics are used.
 *
 * @param input - Input `[B, C, *S]`.
 * @param options - Running statistics, affine weight and bias, training flag, momentum and eps (1e-5).
 * @returns The output and the updated running statistics.
 * @concept batch-normalisation
 * @reference Ioffe and Szegedy (2015), Batch normalization: accelerating deep network training by reducing internal
 *   covariate shift.
 * @example
 * ```ts
 * import { batchNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(batchNorm([[1], [3]], { eps: 0 }).output) // => [[-1], [1]]
 * ```
 */
export function batchNorm(input: TensorLike, options: BatchNormOptions = {}): BatchNormResult {
  const x = fl(input)
  const rank = x.shape.length
  if (rank < 2) throw new OmaiError('shape', 'batchNorm: input must be [B, C, ...]')
  const eps = options.eps ?? 1e-5
  const axes = [0, ...x.shape.slice(2).map((_, i) => i + 2)]
  const hasRunning = options.runningMean !== undefined && options.runningVar !== undefined
  const training = options.training ?? true
  let m: Value
  let v: Value
  let runningMean: Value | undefined
  let runningVar: Value | undefined
  if (training || !hasRunning) {
    m = mean(x, axes)
    v = variance(x, axes)
    if (hasRunning && training) {
      const n = axes.reduce((a, ax) => a * (x.shape[ax] ?? 1), 1)
      const f = options.momentum === null ? 1 / ((options.numBatchesTracked ?? 0) + 1) : (options.momentum ?? 0.1)
      const unbiased = mul(stopGradient(v), n / Math.max(n - 1, 1))
      runningMean = add(mul(fl(options.runningMean as TensorLike), 1 - f), mul(stopGradient(m), f))
      runningVar = add(mul(fl(options.runningVar as TensorLike), 1 - f), mul(unbiased, f))
    } else if (hasRunning) {
      runningMean = fl(options.runningMean as TensorLike)
      runningVar = fl(options.runningVar as TensorLike)
    }
  } else {
    m = fl(options.runningMean as TensorLike)
    v = fl(options.runningVar as TensorLike)
    runningMean = m
    runningVar = v
  }
  let y = div(sub(x, channelShape(m, rank)), sqrt(add(channelShape(v, rank), eps)))
  if (options.weight !== undefined) y = mul(y, channelShape(fl(options.weight), rank))
  if (options.bias !== undefined) y = add(y, channelShape(fl(options.bias), rank))
  return {
    output: y,
    ...(runningMean === undefined ? {} : { runningMean }),
    ...(runningVar === undefined ? {} : { runningVar }),
  }
}

/** Options of the affine normalisations. */
export interface AffineNormOptions {
  readonly weight?: TensorLike
  readonly bias?: TensorLike
  readonly eps?: number
}

const lastAxes = (x: Value, normalizedShape: number | readonly number[], name: string): number[] => {
  const shape = typeof normalizedShape === 'number' ? [normalizedShape] : normalizedShape
  const rank = x.shape.length
  if (shape.some((n, i) => x.shape[rank - shape.length + i] !== n))
    throw new OmaiError('shape', `${name}: input [${x.shape.join(', ')}] does not end in [${shape.join(', ')}]`)
  return shape.map((_, i) => rank - shape.length + i)
}

/**
 * Layer normalisation (torch.nn.functional.layer_norm, Ba et al. 2016): normalise over the trailing `normalizedShape`
 * axes of each example, then scale and shift elementwise.
 *
 * @param input - Input `[..., *normalizedShape]`.
 * @param normalizedShape - The trailing shape to normalise over.
 * @param options - Elementwise weight and bias of that shape, and eps (1e-5).
 * @returns The normalised input.
 * @concept layer-normalisation
 * @example
 * ```ts
 * import { layerNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(layerNorm([[1, 3]], 2, { eps: 0 })) // => [[-1, 1]]
 * ```
 */
export function layerNorm(
  input: TensorLike,
  normalizedShape: number | readonly number[],
  options: AffineNormOptions = {},
): Value {
  const x = fl(input)
  const axes = lastAxes(x, normalizedShape, 'layerNorm')
  const m = mean(x, axes, true)
  const v = variance(x, axes, { keepDims: true })
  let y = div(sub(x, m), sqrt(add(v, options.eps ?? 1e-5)))
  if (options.weight !== undefined) y = mul(y, fl(options.weight))
  if (options.bias !== undefined) y = add(y, fl(options.bias))
  return y
}

/**
 * Root-mean-square normalisation (torch.nn.functional.rms_norm, Zhang and Sennrich 2019): divide by the root mean
 * square over the trailing axes (no centring), then scale. LLaMA's and T5's normalisation.
 *
 * @param input - Input `[..., *normalizedShape]`.
 * @param normalizedShape - The trailing shape to normalise over.
 * @param options - Elementwise weight and eps (default the float64 machine epsilon, as PyTorch).
 * @returns The normalised input.
 * @reference Zhang and Sennrich (2019), Root mean square layer normalization.
 * @example
 * ```ts
 * import { rmsNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rmsNorm([[3, 4]], 2, { eps: 0 })).flat().map((v) => Math.round(v * 1e6) / 1e6) // => [0.848528, 1.131371]
 * ```
 */
export function rmsNorm(
  input: TensorLike,
  normalizedShape: number | readonly number[],
  options: Omit<AffineNormOptions, 'bias'> = {},
): Value {
  const x = fl(input)
  const axes = lastAxes(x, normalizedShape, 'rmsNorm')
  const ms = mean(square(x), axes, true)
  let y = div(x, sqrt(add(ms, options.eps ?? Number.EPSILON)))
  if (options.weight !== undefined) y = mul(y, fl(options.weight))
  return y
}

/**
 * Group normalisation (torch.nn.functional.group_norm, Wu and He 2018): split the channels into groups and normalise
 * each group of each example over its channels and spatial axes, then scale and shift per channel.
 *
 * @param input - Input `[B, C, *S]`.
 * @param numGroups - The number of groups (dividing C).
 * @param options - Per-channel weight and bias, and eps (1e-5).
 * @returns The normalised input.
 * @reference Wu and He (2018), Group normalization.
 * @example
 * ```ts
 * import { groupNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(groupNorm([[[1], [3]]], 1, { eps: 0 })) // => [[[-1], [1]]]
 * ```
 */
export function groupNorm(input: TensorLike, numGroups: number, options: AffineNormOptions = {}): Value {
  const x = fl(input)
  const [b = 0, c = 0, ...spatial] = x.shape
  if (c % numGroups !== 0) throw new OmaiError('shape', `groupNorm: ${c} channels do not divide into ${numGroups}`)
  const g = reshape(x, [b, numGroups, -1])
  const m = mean(g, 2, true)
  const v = variance(g, 2, { keepDims: true })
  let y = reshape(div(sub(g, m), sqrt(add(v, options.eps ?? 1e-5))), [b, c, ...spatial])
  if (options.weight !== undefined) y = mul(y, channelShape(fl(options.weight), x.shape.length))
  if (options.bias !== undefined) y = add(y, channelShape(fl(options.bias), x.shape.length))
  return y
}

/** Options of {@link instanceNorm}. */
export interface InstanceNormOptions extends AffineNormOptions {
  readonly runningMean?: TensorLike
  readonly runningVar?: TensorLike
  /** Normalise by each instance's own statistics (default true; false uses the running statistics). */
  readonly useInputStats?: boolean
  readonly momentum?: number
}

/**
 * Instance normalisation (torch.nn.functional.instance_norm, Ulyanov et al. 2016): normalise each channel of each
 * example over its spatial axes (style transfer). With running statistics, their update (averaged over the batch) is
 * returned as for batch normalisation.
 *
 * @param input - Input `[B, C, *S]`.
 * @param options - Per-channel weight and bias, eps (1e-5), and optional running statistics with momentum (0.1).
 * @returns The output and, when running statistics are given, their update.
 * @example
 * ```ts
 * import { instanceNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(instanceNorm([[[1, 3]]], { eps: 0 }).output) // => [[[-1, 1]]]
 * ```
 */
export function instanceNorm(input: TensorLike, options: InstanceNormOptions = {}): BatchNormResult {
  const x = fl(input)
  const [b = 0, c = 0, ...spatial] = x.shape
  const useInput = options.useInputStats ?? true
  const hasRunning = options.runningMean !== undefined && options.runningVar !== undefined
  if (!useInput) {
    if (!hasRunning) throw new OmaiError('value', 'instanceNorm: useInputStats false needs running statistics')
    return batchNorm(x, { ...options, training: false })
  }
  // An instance is a channel of an example: fold the batch into the channels and use batch normalisation's arithmetic.
  const folded = reshape(x, [1, b * c, ...spatial])
  const tile = (v: TensorLike | undefined): Value | undefined =>
    v === undefined
      ? undefined
      : reshape(
          concat(
            Array.from({ length: b }, () => fl(v)),
            0,
          ),
          [b * c],
        )
  const r = batchNorm(folded, {
    training: true,
    eps: options.eps ?? 1e-5,
    momentum: options.momentum ?? 0.1,
    ...(hasRunning
      ? { runningMean: tile(options.runningMean) as Value, runningVar: tile(options.runningVar) as Value }
      : {}),
  })
  let y = reshape(r.output, x.shape)
  if (options.weight !== undefined) y = mul(y, channelShape(fl(options.weight), x.shape.length))
  if (options.bias !== undefined) y = add(y, channelShape(fl(options.bias), x.shape.length))
  if (!hasRunning || r.runningMean === undefined || r.runningVar === undefined) return { output: y }
  return {
    output: y,
    runningMean: mean(reshape(r.runningMean, [b, c]), 0),
    runningVar: mean(reshape(r.runningVar, [b, c]), 0),
  }
}

/**
 * Local response normalisation across channels (torch.nn.functional.local_response_norm, AlexNet): divide by
 * (k + alpha / size * sum of squares over `size` neighbouring channels)^beta.
 *
 * @param input - Input `[B, C, *S]`.
 * @param size - The number of neighbouring channels.
 * @param alpha - Scale (default 1e-4).
 * @param beta - Exponent (default 0.75).
 * @param k - Offset (default 1).
 * @returns The normalised input.
 * @example
 * ```ts
 * import { localResponseNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(localResponseNorm([[[0], [0]]], 2)) // => [[[0], [0]]]
 * ```
 */
export function localResponseNorm(input: TensorLike, size: number, alpha = 1e-4, beta = 0.75, k = 1): Value {
  const x = fl(input)
  const c = x.shape[1] ?? 0
  const sq = square(x)
  // Zero-pad the channel axis by floor(size / 2) before and (size - 1) / 2 after, then sum sliding windows.
  const before = Math.floor(size / 2)
  const after = Math.floor((size - 1) / 2)
  const zerosLike = (n: number): Value => zeros([x.shape[0] ?? 0, n, ...x.shape.slice(2)])
  const padded = concat(
    [zerosLike(before), sq, zerosLike(after)].filter((v) => (v.shape[1] ?? 0) > 0),
    1,
  )
  let total: Value = sliceAxis(padded, 1, 0, c)
  for (let j = 1; j < size; j++) total = add(total, sliceAxis(padded, 1, j, j + c))
  return div(x, pow(add(mul(total, alpha / size), k), beta))
}

/**
 * Lp-normalise along an axis (torch.nn.functional.normalize): x / max(||x||_p, eps).
 *
 * @param input - Input.
 * @param p - The norm's order (default 2).
 * @param axis - The axis (default 1).
 * @param eps - Floor of the norm (default 1e-12).
 * @returns The normalised input.
 * @example
 * ```ts
 * import { normalize } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(normalize([[3, 4]])) // => [[0.6, 0.8]]
 * ```
 */
export function normalize(input: TensorLike, p = 2, axis = 1, eps = 1e-12): Value {
  const x = fl(input)
  return div(x, maximum(norm(x, p, axis, true), eps))
}

/** All axes of a rank except one. */
const otherAxes = (rank: number, axis: number): number[] =>
  Array.from({ length: rank }, (_, i) => i).filter((i) => i !== (axis < 0 ? rank + axis : axis))

/**
 * Weight normalisation (torch.nn.utils.parametrizations.weight_norm, Salimans and Kingma 2016): w = g v / ||v||, the
 * norm taken over every axis except `axis` (one norm per output unit by default), so direction and length are learnt
 * separately.
 *
 * @param v - The direction parameter (the weight's shape).
 * @param g - The magnitude, with size 1 on every axis but `axis` (`[out, 1, ...]` by default).
 * @param axis - The axis kept (default 0; `null` for a single norm of the whole weight).
 * @returns The weight.
 * @reference Salimans and Kingma (2016), Weight normalization.
 * @example
 * ```ts
 * import { weightNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(weightNorm([[3, 4]], [[10]])) // => [[6, 8]]
 * ```
 */
export function weightNorm(v: TensorLike, g: TensorLike, axis: number | null = 0): Value {
  const vv = fl(v)
  const n = axis === null ? norm(reshape(vv, [-1]), 2) : sqrt(sum(square(vv), otherAxes(vv.shape.length, axis), true))
  return mul(fl(g), div(vv, n))
}

/** Options of {@link spectralNorm}. */
export interface SpectralNormOptions {
  /** Power iterations per call in training (default 1). */
  readonly nPowerIterations?: number
  readonly eps?: number
  /** The output axis of the weight (default 0; 1 for transposed convolutions). */
  readonly axis?: number
  /** Run the power iterations (default true); in evaluation the stored vectors are used as they are. */
  readonly training?: boolean
}

/** Result of {@link spectralNorm}. */
export interface SpectralNormResult {
  /** The weight divided by its estimated largest singular value. */
  readonly weight: Value
  /** The updated left singular vector estimate (a buffer). */
  readonly u: Value
  /** The updated right singular vector estimate (a buffer). */
  readonly v: Value
  /** The estimated spectral norm. */
  readonly sigma: Value
}

/**
 * Spectral normalisation (torch.nn.utils.parametrizations.spectral_norm, Miyato et al. 2018): divide the weight by an
 * estimate of its largest singular value from power iteration, with the vectors u and v kept as buffers between calls.
 * The weight is viewed as a matrix `[shape[axis], rest]`; the iterations do not carry gradients, and sigma = u^T W v
 * does.
 *
 * @param weight - The weight.
 * @param u - Left vector `[shape[axis]]`.
 * @param v - Right vector `[rest]`.
 * @param options - Iterations, eps (1e-12), axis and training flag.
 * @returns The normalised weight, the updated vectors and sigma.
 * @reference Miyato, Kataoka, Koyama and Yoshida (2018), Spectral normalization for generative adversarial networks.
 * @example
 * ```ts
 * import { spectralNorm } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(spectralNorm([[2, 0], [0, 1]], [1, 0], [1, 0]).sigma) // => 2
 * ```
 */
export function spectralNorm(
  weight: TensorLike,
  u: TensorLike,
  v: TensorLike,
  options: SpectralNormOptions = {},
): SpectralNormResult {
  const w = fl(weight)
  const axis = options.axis ?? 0
  const eps = options.eps ?? 1e-12
  const moved = axis === 0 ? w : moveaxis(w, axis, 0)
  const mat = reshape(moved, [moved.shape[0] ?? 0, -1])
  const matConst = stopGradient(mat)
  const unit = (x: Value): Value => div(x, maximum(norm(x, 2), eps))
  let uu = fl(u)
  let vv = fl(v)
  if (options.training ?? true) {
    for (let i = 0; i < (options.nPowerIterations ?? 1); i++) {
      vv = unit(reshape(matmul(swapaxes(matConst, 0, 1), reshape(uu, [-1, 1])), [-1]))
      uu = unit(reshape(matmul(matConst, reshape(vv, [-1, 1])), [-1]))
    }
    uu = stopGradient(uu)
    vv = stopGradient(vv)
  }
  const sigma = sum(mul(uu, reshape(matmul(mat, reshape(vv, [-1, 1])), [-1])))
  return { weight: div(w, sigma), u: uu, v: vv, sigma }
}
