/**
 * Weight initialisers, torch.nn.init style: fan computation and the recommended gain per activation, Xavier (Glorot),
 * Kaiming (He) and LeCun schemes in uniform and normal forms, PyTorch's default, constant, uniform, normal and
 * truncated normal draws, orthogonal and identity matrices, the Dirac convolution weight and sparse initialisation.
 * Each draws from an explicit key.
 *
 * @index function Weight initialisers: Xavier, Kaiming, LeCun, orthogonal.
 */

import { OmaiError } from '../base/index.ts'
import { qr } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal, permutation, truncatedNormal, uniform } from '../random/index.ts'
import {
  diagonal,
  fromStorage,
  full,
  mul,
  sign,
  sizeOf,
  type Tensor,
  toFlat,
  transpose,
  zeros,
} from '../tensor/index.ts'

/**
 * Fan-in and fan-out of a weight shape (PyTorch's convention: `[out, in, ...kernel]`).
 *
 * @param shape - The weight shape (at least two axes).
 * @returns `fanIn` (inputs times receptive field) and `fanOut` (outputs times receptive field).
 * @example
 * ```ts
 * import { fans } from 'omai-engine/nn'
 * fans([8, 4, 3, 3]) // => { fanIn: 36, fanOut: 72 }
 * ```
 */
export function fans(shape: readonly number[]): { readonly fanIn: number; readonly fanOut: number } {
  if (shape.length < 2) throw new OmaiError('shape', 'fans: need at least two dimensions')
  const receptive = shape.slice(2).reduce((a, b) => a * b, 1)
  return { fanIn: (shape[1] ?? 0) * receptive, fanOut: (shape[0] ?? 0) * receptive }
}

/** The nonlinearities {@link calculateGain} knows. */
export type GainNonlinearity =
  | 'linear'
  | 'identity'
  | 'conv1d'
  | 'conv2d'
  | 'conv3d'
  | 'convTranspose1d'
  | 'convTranspose2d'
  | 'convTranspose3d'
  | 'sigmoid'
  | 'tanh'
  | 'relu'
  | 'leakyRelu'
  | 'selu'

/**
 * The recommended gain for an activation (torch.nn.init.calculate_gain): 1 for linear maps, convolutions and sigmoid,
 * 5/3 for tanh, sqrt 2 for ReLU, sqrt(2 / (1 + slope^2)) for leaky ReLU, 3/4 for SELU.
 *
 * @param nonlinearity - The activation.
 * @param param - The leaky ReLU slope (default 0.01).
 * @returns The gain.
 * @example
 * ```ts
 * import { calculateGain } from 'omai-engine/nn'
 * calculateGain('tanh') // => 1.6666666666666667
 * ```
 */
export function calculateGain(nonlinearity: GainNonlinearity, param = 0.01): number {
  switch (nonlinearity) {
    case 'tanh':
      return 5 / 3
    case 'relu':
      return Math.SQRT2
    case 'leakyRelu':
      return Math.sqrt(2 / (1 + param * param))
    case 'selu':
      return 3 / 4
    default:
      return 1
  }
}

/**
 * Xavier (Glorot) uniform initialisation: U(-a, a) with a = gain sqrt(6 / (fanIn + fanOut)).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @param gain - Scale factor (default 1).
 * @returns The weights.
 * @concept initialisation
 * @example
 * ```ts
 * import { xavierUniform } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * xavierUniform(key(0), [3, 2]).shape // => [3, 2]
 * ```
 */
export function xavierUniform(key: Key, shape: readonly number[], gain = 1): Tensor {
  const { fanIn, fanOut } = fans(shape)
  const a = gain * Math.sqrt(6 / (fanIn + fanOut))
  return uniform(key, shape, { low: -a, high: a })
}

/**
 * Xavier (Glorot) normal initialisation: N(0, gain^2 2 / (fanIn + fanOut)).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @param gain - Scale factor.
 * @returns The weights.
 * @example
 * ```ts
 * import { xavierNormal } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * xavierNormal(key(0), [3, 2]).shape // => [3, 2]
 * ```
 */
export function xavierNormal(key: Key, shape: readonly number[], gain = 1): Tensor {
  const { fanIn, fanOut } = fans(shape)
  return normal(key, shape, { scale: gain * Math.sqrt(2 / (fanIn + fanOut)) })
}

/** Options of the Kaiming initialisers. */
export interface KaimingOptions {
  /** The negative slope of the rectifier that follows (default 0, ReLU). */
  readonly a?: number
  /** Preserve the variance of activations (`fanIn`, default) or of gradients (`fanOut`). */
  readonly mode?: 'fanIn' | 'fanOut'
  /** The nonlinearity whose gain to use (default leaky ReLU with slope `a`). */
  readonly nonlinearity?: GainNonlinearity
}

const kaimingStd = (shape: readonly number[], options: KaimingOptions): number => {
  const { fanIn, fanOut } = fans(shape)
  const fan = options.mode === 'fanOut' ? fanOut : fanIn
  return calculateGain(options.nonlinearity ?? 'leakyRelu', options.a ?? 0) / Math.sqrt(fan)
}

/**
 * Kaiming (He) uniform initialisation: U(-b, b) with b = gain sqrt(3 / fan) (torch.nn.init.kaiming_uniform_).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @param options - `a` (negative slope, default 0), `mode` ('fanIn' default or 'fanOut') and `nonlinearity`.
 * @returns The weights.
 * @example
 * ```ts
 * import { kaimingUniform } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * kaimingUniform(key(0), [4, 3]).shape // => [4, 3]
 * ```
 */
export function kaimingUniform(key: Key, shape: readonly number[], options: KaimingOptions = {}): Tensor {
  const bound = Math.sqrt(3) * kaimingStd(shape, options)
  return uniform(key, shape, { low: -bound, high: bound })
}

/**
 * Kaiming (He) normal initialisation: N(0, gain^2 / fan) (torch.nn.init.kaiming_normal_).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @param options - `a` (negative slope), `mode` and `nonlinearity`.
 * @returns The weights.
 * @example
 * ```ts
 * import { kaimingNormal } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * kaimingNormal(key(0), [4, 3]).shape // => [4, 3]
 * ```
 */
export function kaimingNormal(key: Key, shape: readonly number[], options: KaimingOptions = {}): Tensor {
  return normal(key, shape, { scale: kaimingStd(shape, options) })
}

/**
 * Uniform U(-1/sqrt(fanIn), 1/sqrt(fanIn)): PyTorch's default for linear and convolution weights and biases
 * (kaiming_uniform with a = sqrt 5).
 *
 * @param key - The random key.
 * @param shape - The shape to fill.
 * @param fanIn - The layer's fan-in.
 * @returns The values.
 * @example
 * ```ts
 * import { torchDefaultUniform } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * torchDefaultUniform(key(0), [3], 4).data.every((v) => Math.abs(v) <= 0.5) // => true
 * ```
 */
export function torchDefaultUniform(key: Key, shape: readonly number[], fanIn: number): Tensor {
  const bound = fanIn > 0 ? 1 / Math.sqrt(fanIn) : 0
  return uniform(key, shape, { low: -bound, high: bound })
}

/**
 * Zeros of a shape (for biases).
 *
 * @param shape - The shape.
 * @returns The zeros.
 * @example
 * ```ts
 * import { zerosInit } from 'omai-engine/nn'
 * zerosInit([2]).data // => Float64Array [0, 0]
 * ```
 */
export function zerosInit(shape: readonly number[]): Tensor {
  return zeros(shape)
}

/**
 * Ones of a shape (torch.nn.init.ones_; normalisation scales).
 *
 * @param shape - The shape.
 * @returns The ones.
 * @example
 * ```ts
 * import { onesInit } from 'omai-engine/nn'
 * onesInit([2]).data // => Float64Array [1, 1]
 * ```
 */
export function onesInit(shape: readonly number[]): Tensor {
  return full(shape, 1)
}

/**
 * A constant (torch.nn.init.constant_).
 *
 * @param shape - The shape.
 * @param value - The value.
 * @returns The filled tensor.
 * @example
 * ```ts
 * import { constantInit } from 'omai-engine/nn'
 * constantInit([2], 0.5).data // => Float64Array [0.5, 0.5]
 * ```
 */
export function constantInit(shape: readonly number[], value: number): Tensor {
  return full(shape, value)
}

/**
 * Uniform U(low, high) (torch.nn.init.uniform_).
 *
 * @param key - The random key.
 * @param shape - The shape.
 * @param low - Lower bound (default 0).
 * @param high - Upper bound (default 1).
 * @returns The values.
 * @example
 * ```ts
 * import { uniformInit } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * uniformInit(key(0), [5], -1, 1).data.every((v) => v >= -1 && v < 1) // => true
 * ```
 */
export function uniformInit(key: Key, shape: readonly number[], low = 0, high = 1): Tensor {
  return uniform(key, shape, { low, high })
}

/**
 * Normal N(mean, std^2) (torch.nn.init.normal_).
 *
 * @param key - The random key.
 * @param shape - The shape.
 * @param mean - Mean (default 0).
 * @param std - Standard deviation (default 1).
 * @returns The values.
 * @example
 * ```ts
 * import { normalInit } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * normalInit(key(0), [2, 3], 0, 0.02).shape // => [2, 3]
 * ```
 */
export function normalInit(key: Key, shape: readonly number[], mean = 0, std = 1): Tensor {
  return normal(key, shape, { loc: mean, scale: std })
}

/** Options of {@link truncatedNormalInit}. */
export interface TruncatedNormalInitOptions {
  readonly mean?: number
  readonly std?: number
  /** Lower cut-off in absolute units (default -2). */
  readonly a?: number
  /** Upper cut-off in absolute units (default 2). */
  readonly b?: number
}

/**
 * Normal N(mean, std^2) truncated to [a, b] (torch.nn.init.trunc_normal_; ViT and BERT weights use std 0.02).
 *
 * @param key - The random key.
 * @param shape - The shape.
 * @param options - Mean (0), standard deviation (1) and cut-offs a (-2) and b (2), in absolute units as in PyTorch.
 * @returns The values.
 * @example
 * ```ts
 * import { truncatedNormalInit } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * truncatedNormalInit(key(0), [100], { std: 0.02, a: -0.04, b: 0.04 }).data.every((v) => Math.abs(v) <= 0.04) // => true
 * ```
 */
export function truncatedNormalInit(
  key: Key,
  shape: readonly number[],
  options: TruncatedNormalInitOptions = {},
): Tensor {
  const mean = options.mean ?? 0
  const std = options.std ?? 1
  const a = options.a ?? -2
  const b = options.b ?? 2
  const z = truncatedNormal(key, shape, (a - mean) / std, (b - mean) / std)
  return fromStorage(
    Float64Array.from(z.data, (v) => Math.min(b, Math.max(a, mean + std * v))),
    [...shape],
    'float64',
  )
}

/**
 * LeCun uniform initialisation: U(-a, a) with a = sqrt(3 / fanIn), variance 1 / fanIn (JAX's lecun_uniform).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @returns The weights.
 * @example
 * ```ts
 * import { lecunUniform } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * lecunUniform(key(0), [4, 3]).data.every((v) => Math.abs(v) <= 1) // => true
 * ```
 */
export function lecunUniform(key: Key, shape: readonly number[]): Tensor {
  const a = Math.sqrt(3 / fans(shape).fanIn)
  return uniform(key, shape, { low: -a, high: a })
}

/**
 * LeCun normal initialisation: variance 1 / fanIn (SELU networks), drawn from a normal truncated at two standard
 * deviations and rescaled to that variance, as JAX's lecun_normal does.
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @returns The weights.
 * @example
 * ```ts
 * import { lecunNormal } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * lecunNormal(key(0), [4, 3]).shape // => [4, 3]
 * ```
 */
export function lecunNormal(key: Key, shape: readonly number[]): Tensor {
  // The standard deviation of a standard normal truncated to [-2, 2].
  const truncatedStd = 0.8796256610342398
  const scale = 1 / Math.sqrt(fans(shape).fanIn) / truncatedStd
  const z = truncatedNormal(key, shape, -2, 2)
  return fromStorage(
    Float64Array.from(z.data, (v) => scale * v),
    [...shape],
    'float64',
  )
}

/**
 * A (semi-)orthogonal matrix (torch.nn.init.orthogonal_, Saxe et al. 2014): the Q factor of a Gaussian matrix with
 * the signs of R's diagonal folded in, reshaped to `shape` (rows `shape[0]`, columns the rest), times `gain`.
 *
 * @param key - The random key.
 * @param shape - Weight shape (at least two axes).
 * @param gain - Scale factor (default 1).
 * @returns The weights; rows (or columns, whichever are fewer) are orthonormal times `gain`.
 * @reference Saxe, McClelland and Ganguli (2014), Exact solutions to the nonlinear dynamics of learning in deep linear
 *   neural networks.
 * @example
 * ```ts
 * import { orthogonal } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * import { matmul, transpose, toArray } from 'omai-engine/tensor'
 * const q = orthogonal(key(0), [2, 2])
 * toArray(matmul(q, transpose(q))).flat().map((v) => Math.round(v * 1e9) / 1e9) // => [1, 0, 0, 1]
 * ```
 */
export function orthogonal(key: Key, shape: readonly number[], gain = 1): Tensor {
  if (shape.length < 2) throw new OmaiError('shape', 'orthogonal: need at least two dimensions')
  const rows = shape[0] ?? 0
  const cols = sizeOf(shape) / rows
  const wide = rows < cols
  const a = normal(key, wide ? [cols, rows] : [rows, cols])
  const { q, r } = qr(a)
  let m = mul(q, sign(diagonal(r)))
  if (wide) m = transpose(m)
  return fromStorage(
    Float64Array.from(toFlat(m), (v) => gain * v),
    [...shape],
    'float64',
  )
}

/**
 * The identity matrix as a weight (torch.nn.init.eye_; for 2-D weights, also non-square).
 *
 * @param shape - `[out, in]`.
 * @returns Ones on the diagonal.
 * @example
 * ```ts
 * import { eyeInit } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(eyeInit([2, 3])) // => [[1, 0, 0], [0, 1, 0]]
 * ```
 */
export function eyeInit(shape: readonly [number, number]): Tensor {
  const [m, n] = shape
  const out = new Float64Array(m * n)
  for (let i = 0; i < Math.min(m, n); i++) out[i * n + i] = 1
  return fromStorage(out, [m, n], 'float64')
}

/**
 * The Dirac delta as a convolution weight (torch.nn.init.dirac_): each output channel copies one input channel, so the
 * convolution is the identity on the first min(out, in) channels; with groups, per group.
 *
 * @param shape - `[out, in, ...kernel]` (3 to 5 axes).
 * @param groups - Number of groups (default 1).
 * @returns The weights.
 * @example
 * ```ts
 * import { diracInit } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(diracInit([1, 1, 3])) // => [[[0, 1, 0]]]
 * ```
 */
export function diracInit(shape: readonly number[], groups = 1): Tensor {
  if (shape.length < 3 || shape.length > 5) throw new OmaiError('shape', 'diracInit: need a 3-, 4- or 5-D weight')
  const [out = 0, cin = 0, ...kernel] = shape
  if (out % groups !== 0) throw new OmaiError('value', 'diracInit: out channels must divide by groups')
  const perGroup = out / groups
  const ksize = sizeOf(kernel)
  const centre = kernel.reduce((acc, k, i) => acc + Math.floor(k / 2) * sizeOf(kernel.slice(i + 1)), 0)
  const data = new Float64Array(sizeOf(shape))
  for (let g = 0; g < groups; g++) {
    for (let d = 0; d < Math.min(perGroup, cin); d++) data[((g * perGroup + d) * cin + d) * ksize + centre] = 1
  }
  return fromStorage(data, [...shape], 'float64')
}

/**
 * A sparse weight (torch.nn.init.sparse_, Martens 2010): N(0, std^2) entries with ceil(sparsity rows) entries of each
 * column set to zero at random.
 *
 * @param key - The random key.
 * @param shape - `[rows, cols]`.
 * @param sparsity - The fraction of zeros in each column.
 * @param std - Standard deviation of the non-zero entries (default 0.01).
 * @returns The weights.
 * @example
 * ```ts
 * import { sparseInit } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * sparseInit(key(0), [10, 4], 0.5).data.filter((v) => v === 0).length // => 20
 * ```
 */
export function sparseInit(key: Key, shape: readonly [number, number], sparsity: number, std = 0.01): Tensor {
  const [rows, cols] = shape
  const zerosPerColumn = Math.ceil(sparsity * rows)
  const data = Float64Array.from(normal(foldIn(key, 0), shape, { scale: std }).data)
  for (let c = 0; c < cols; c++) {
    const order = permutation(foldIn(key, c + 1), rows).data
    for (let i = 0; i < zerosPerColumn; i++) data[(order[i] ?? 0) * cols + c] = 0
  }
  return fromStorage(data, [rows, cols], 'float64')
}
