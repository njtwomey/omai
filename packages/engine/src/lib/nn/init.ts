import { WhiteboxError } from '../base/index.ts'
import { type Key, normal, uniform } from '../random/index.ts'
import { type Tensor, zeros } from '../tensor/index.ts'

/** Fan-in and fan-out of a weight shape (PyTorch's convention: `[out, in, ...kernel]`). */
export function fans(shape: readonly number[]): { readonly fanIn: number; readonly fanOut: number } {
  if (shape.length < 2) throw new WhiteboxError('shape', 'fans: need at least two dimensions')
  const receptive = shape.slice(2).reduce((a, b) => a * b, 1)
  return { fanIn: (shape[1] ?? 0) * receptive, fanOut: (shape[0] ?? 0) * receptive }
}

/** The recommended gain for an activation (torch.nn.init.calculate_gain). */
export function calculateGain(
  nonlinearity: 'linear' | 'sigmoid' | 'tanh' | 'relu' | 'leakyRelu' | 'selu',
  param = 0.01,
): number {
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
 * import { xavierUniform } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
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
 * import { xavierNormal } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * xavierNormal(key(0), [3, 2]).shape // => [3, 2]
 * ```
 */
export function xavierNormal(key: Key, shape: readonly number[], gain = 1): Tensor {
  const { fanIn, fanOut } = fans(shape)
  return normal(key, shape, { scale: gain * Math.sqrt(2 / (fanIn + fanOut)) })
}

/**
 * Kaiming (He) uniform initialisation: U(-b, b) with b = gain sqrt(3 / fan) and gain for leaky ReLU with slope `a`
 * (torch.nn.init.kaiming_uniform_).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @param options - `a` (negative slope, default 0), `mode` ('fanIn' default or 'fanOut').
 * @returns The weights.
 * @example
 * ```ts
 * import { kaimingUniform } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * kaimingUniform(key(0), [4, 3]).shape // => [4, 3]
 * ```
 */
export function kaimingUniform(
  key: Key,
  shape: readonly number[],
  options: { readonly a?: number; readonly mode?: 'fanIn' | 'fanOut' } = {},
): Tensor {
  const { fanIn, fanOut } = fans(shape)
  const fan = options.mode === 'fanOut' ? fanOut : fanIn
  const gain = calculateGain('leakyRelu', options.a ?? 0)
  const bound = gain * Math.sqrt(3 / fan)
  return uniform(key, shape, { low: -bound, high: bound })
}

/**
 * Kaiming (He) normal initialisation: N(0, gain^2 / fan).
 *
 * @param key - The random key.
 * @param shape - Weight shape.
 * @param options - `a` (negative slope) and `mode`.
 * @returns The weights.
 * @example
 * ```ts
 * import { kaimingNormal } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * kaimingNormal(key(0), [4, 3]).shape // => [4, 3]
 * ```
 */
export function kaimingNormal(
  key: Key,
  shape: readonly number[],
  options: { readonly a?: number; readonly mode?: 'fanIn' | 'fanOut' } = {},
): Tensor {
  const { fanIn, fanOut } = fans(shape)
  const fan = options.mode === 'fanOut' ? fanOut : fanIn
  return normal(key, shape, { scale: calculateGain('leakyRelu', options.a ?? 0) / Math.sqrt(fan) })
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
 * import { torchDefaultUniform } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
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
 * import { zerosInit } from 'whitebox-engine/nn'
 * zerosInit([2]).data // => Float64Array [0, 0]
 * ```
 */
export function zerosInit(shape: readonly number[]): Tensor {
  return zeros(shape)
}
