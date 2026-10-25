/**
 * Stochastic regularisers as functions of a key, active only in training: dropout and its channel and alpha variants,
 * drop path (stochastic depth, which skips whole residual branches), Gaussian noise injection and Gaussian dropout, and
 * cutout, which blanks a random square of each image.
 */

import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, normal, randint, uniform } from '../../random/index.ts'
import {
  add,
  cast,
  div,
  fromStorage,
  greater,
  mul,
  promoteFloat,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'

/** Options of the stochastic regularisers. */
export interface StochasticOptions {
  /** The key for the random mask (needed in training). */
  readonly key?: Key
  /** Apply the regulariser (default true); in evaluation each is the identity. */
  readonly training?: boolean
}

const fl = (x: TensorLike): Value => promoteFloat(x)[0] as Value

function checkP(p: number, name: string): void {
  if (!(p >= 0 && p <= 1)) throw new WhiteboxError('value', `${name}: p must be in [0, 1], got ${p}`)
}

function needKey(options: StochasticOptions, name: string): Key {
  if (options.key === undefined) throw new WhiteboxError('value', `${name}: training mode needs a key`)
  return options.key
}

/** A 0/1 keep mask of a shape (1 with probability 1 - p). */
const keepMask = (key: Key, shape: readonly number[], p: number): Value =>
  cast(greater(uniform(key, shape), p), 'float64')

/**
 * Dropout (torch.nn.functional.dropout, Srivastava et al. 2014): in training, zero each element with probability p and
 * scale the rest by 1 / (1 - p) so the expectation is unchanged; the identity in evaluation.
 *
 * @param input - Input.
 * @param p - Drop probability (default 0.5).
 * @param options - The key and the training flag.
 * @returns The input with dropped elements.
 * @concept dropout
 * @example
 * ```ts
 * import { dropout } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(dropout([1, 1, 1, 1], 0.5, { key: key(0) })).every((v) => v === 0 || v === 2) // => true
 * ```
 */
export function dropout(input: TensorLike, p = 0.5, options: StochasticOptions = {}): Value {
  checkP(p, 'dropout')
  const x = fl(input)
  if (options.training === false || p === 0) return x
  if (p === 1) return mul(x, 0)
  return div(mul(x, keepMask(needKey(options, 'dropout'), x.shape, p)), 1 - p)
}

/**
 * Channel dropout (torch.nn.functional.dropout1d/2d/3d, Tompson et al. 2015): zero whole channels of each example
 * (axis 1 of `[B, C, *S]`) with probability p, scaling the rest; neighbouring activations within a feature map are
 * correlated, so dropping single elements does little.
 *
 * @param input - Input `[B, C, *S]`.
 * @param p - Drop probability (default 0.5).
 * @param options - The key and the training flag.
 * @returns The input with dropped channels.
 * @example
 * ```ts
 * import { channelDropout } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * channelDropout([[[1, 1], [1, 1]]], 0.5, { key: key(0) }).shape // => [1, 2, 2]
 * ```
 */
export function channelDropout(input: TensorLike, p = 0.5, options: StochasticOptions = {}): Value {
  checkP(p, 'channelDropout')
  const x = fl(input)
  if (options.training === false || p === 0) return x
  if (x.shape.length < 2) throw new WhiteboxError('shape', 'channelDropout: input must be [B, C, ...]')
  const shape = [x.shape[0] ?? 1, x.shape[1] ?? 1, ...new Array<number>(x.shape.length - 2).fill(1)]
  if (p === 1) return mul(x, 0)
  return div(mul(x, keepMask(needKey(options, 'channelDropout'), shape, p)), 1 - p)
}

// SELU's negative saturation value, -scale alpha.
const ALPHA_PRIME = -1.7580993408473766

function alphaMask(x: Value, mask: Value, p: number): Value {
  const a = 1 / Math.sqrt((1 - p) * (1 + p * ALPHA_PRIME * ALPHA_PRIME))
  const b = -a * ALPHA_PRIME * p
  // a (x mask + alpha' (1 - mask)) + b
  return add(mul(add(mul(x, mask), mul(add(mul(mask, -1), 1), ALPHA_PRIME)), a), b)
}

/**
 * Alpha dropout (torch.nn.functional.alpha_dropout, Klambauer et al. 2017): dropped units are set to SELU's
 * saturation value and the result is affinely corrected, so a self-normalising network keeps zero mean and unit
 * variance.
 *
 * @param input - Input.
 * @param p - Drop probability (default 0.5).
 * @param options - The key and the training flag.
 * @returns The input with dropped elements.
 * @reference Klambauer, Unterthiner, Mayr and Hochreiter (2017), Self-normalizing neural networks.
 * @example
 * ```ts
 * import { alphaDropout } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(alphaDropout([1, 2], 0.5, { training: false })) // => [1, 2]
 * ```
 */
export function alphaDropout(input: TensorLike, p = 0.5, options: StochasticOptions = {}): Value {
  checkP(p, 'alphaDropout')
  const x = fl(input)
  if (options.training === false || p === 0) return x
  if (p === 1) return mul(x, 0)
  return alphaMask(x, keepMask(needKey(options, 'alphaDropout'), x.shape, p), p)
}

/**
 * Feature alpha dropout (torch.nn.functional.feature_alpha_dropout): alpha dropout of whole channels.
 *
 * @param input - Input `[B, C, *S]`.
 * @param p - Drop probability (default 0.5).
 * @param options - The key and the training flag.
 * @returns The input with dropped channels.
 * @example
 * ```ts
 * import { featureAlphaDropout } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * featureAlphaDropout([[[1, 2]]], 0.5, { key: key(0) }).shape // => [1, 1, 2]
 * ```
 */
export function featureAlphaDropout(input: TensorLike, p = 0.5, options: StochasticOptions = {}): Value {
  checkP(p, 'featureAlphaDropout')
  const x = fl(input)
  if (options.training === false || p === 0) return x
  if (p === 1) return mul(x, 0)
  const shape = [x.shape[0] ?? 1, x.shape[1] ?? 1, ...new Array<number>(Math.max(0, x.shape.length - 2)).fill(1)]
  return alphaMask(x, keepMask(needKey(options, 'featureAlphaDropout'), shape, p), p)
}

/**
 * Drop path, or stochastic depth (Huang et al. 2016; timm's drop_path): drop the whole residual branch of an example
 * with probability p, scaling kept examples by 1 / (1 - p). Applied to a branch before it is added to the shortcut.
 *
 * @param input - The branch output `[B, ...]`.
 * @param p - Drop probability (default 0.1).
 * @param options - The key, the training flag, and `scaleByKeep` (default true).
 * @returns The branch with dropped examples.
 * @reference Huang, Sun, Liu, Sedra and Weinberger (2016), Deep networks with stochastic depth.
 * @example
 * ```ts
 * import { dropPath } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(dropPath([[1, 1], [1, 1]], 0, { key: key(0) })) // => [[1, 1], [1, 1]]
 * ```
 */
export function dropPath(
  input: TensorLike,
  p = 0.1,
  options: StochasticOptions & { readonly scaleByKeep?: boolean } = {},
): Value {
  checkP(p, 'dropPath')
  const x = fl(input)
  if (options.training === false || p === 0) return x
  const shape = [x.shape[0] ?? 1, ...new Array<number>(Math.max(0, x.shape.length - 1)).fill(1)]
  const kept = mul(x, keepMask(needKey(options, 'dropPath'), shape, p))
  return (options.scaleByKeep ?? true) && p < 1 ? div(kept, 1 - p) : kept
}

/**
 * Gaussian noise injection (Keras's GaussianNoise): add N(0, std^2) noise in training, or multiply by N(1, std^2)
 * noise (Gaussian dropout, Srivastava et al. 2014) with `multiplicative`.
 *
 * @param input - Input.
 * @param std - The noise's standard deviation.
 * @param options - The key, the training flag and `multiplicative`.
 * @returns The noisy input.
 * @example
 * ```ts
 * import { gaussianNoise } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(gaussianNoise([1, 2], 0.1, { training: false })) // => [1, 2]
 * ```
 */
export function gaussianNoise(
  input: TensorLike,
  std: number,
  options: StochasticOptions & { readonly multiplicative?: boolean } = {},
): Value {
  const x = fl(input)
  if (options.training === false || std === 0) return x
  const noise = normal(needKey(options, 'gaussianNoise'), x.shape, {
    loc: options.multiplicative === true ? 1 : 0,
    scale: std,
  })
  return options.multiplicative === true ? mul(x, noise) : add(x, noise)
}

/**
 * Cutout (DeVries and Taylor 2017): zero one random square patch of side `size` in each image of `[B, C, H, W]` (the
 * same patch across channels; patches may overhang the border, as in the paper).
 *
 * @param input - Images `[B, C, H, W]`.
 * @param size - The patch side.
 * @param options - The key and the training flag.
 * @returns The images with patches cut out.
 * @reference DeVries and Taylor (2017), Improved regularization of convolutional neural networks with cutout.
 * @example
 * ```ts
 * import { cutout } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(cutout([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], 1, { key: key(0) })).filter((v) => v === 0).length // => 1
 * ```
 */
export function cutout(input: TensorLike, size: number, options: StochasticOptions = {}): Value {
  const x = fl(input)
  if (options.training === false || size <= 0) return x
  if (x.shape.length !== 4) throw new WhiteboxError('shape', 'cutout: input must be [B, C, H, W]')
  const [b = 0, , h = 0, w = 0] = x.shape
  const k = needKey(options, 'cutout')
  const cy = randint(foldIn(k, 0), [b], 0, h).data
  const cx = randint(foldIn(k, 1), [b], 0, w).data
  const mask = new Float64Array(b * h * w).fill(1)
  const half = Math.floor(size / 2)
  for (let n = 0; n < b; n++) {
    const y0 = Math.max(0, (cy[n] ?? 0) - half)
    const y1 = Math.min(h, (cy[n] ?? 0) - half + size)
    const x0 = Math.max(0, (cx[n] ?? 0) - half)
    const x1 = Math.min(w, (cx[n] ?? 0) - half + size)
    for (let i = y0; i < y1; i++) for (let j = x0; j < x1; j++) mask[n * h * w + i * w + j] = 0
  }
  return mul(x, fromStorage(mask, [b, 1, h, w], 'float64'))
}
