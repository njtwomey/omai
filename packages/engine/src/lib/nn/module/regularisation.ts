/**
 * Stochastic regularisation layers as modules, active only in training mode and drawing from the apply context's key:
 * dropout and its channel and alpha variants, `DropPath` (stochastic depth, as in timm), Gaussian noise and Gaussian
 * dropout, and `Cutout`.
 */

import { OmaiError } from '../../base/index.ts'
import type { Value } from '../../tensor/index.ts'
import {
  alphaDropout,
  channelDropout,
  cutout,
  dropout,
  dropPath,
  featureAlphaDropout,
  gaussianNoise,
  type StochasticOptions,
} from '../functional/regularisation.ts'
import { type ApplyContext, isTraining, Module, type Params } from './module.ts'

/** A parameter-free stochastic module: a regulariser in training, the identity in evaluation. */
class Stochastic extends Module {
  readonly kind: string
  readonly p: number
  private readonly fn: (x: Value, options: StochasticOptions) => Value

  constructor(kind: string, p: number, fn: (x: Value, options: StochasticOptions) => Value) {
    super()
    if (!(p >= 0 && p <= 1)) throw new OmaiError('value', `${kind}: p must be in [0, 1], got ${p}`)
    this.kind = kind
    this.p = p
    this.fn = fn
  }

  apply(_params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const training = isTraining(ctx)
    if (training && ctx.key === undefined && this.p > 0)
      throw new OmaiError('value', `${this.kind}: training mode needs a key in the apply context`)
    return this.fn(x, { training, ...(ctx.key === undefined ? {} : { key: ctx.key }) })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `p=${this.p}`
  }
}

/**
 * Dropout (torch.nn.Dropout): in training, zero each element with probability p and scale the rest by 1 / (1 - p),
 * using the key in the apply context; the identity in evaluation mode.
 *
 * @index module Dropout, active only in training mode.
 * @example
 * ```ts
 * import { Dropout } from 'omai-engine/nn'
 * new Dropout(0.5).eval().forward([1, 2]).shape // => [2]
 * ```
 */
export class Dropout extends Stochastic {
  constructor(p = 0.5) {
    super('Dropout', p, (x, o) => dropout(x, p, o))
  }
}

/**
 * Channel dropout for `[B, C, L]` inputs (torch.nn.Dropout1d): whole channels are zeroed.
 *
 * @example
 * ```ts
 * import { Dropout1d } from 'omai-engine/nn'
 * new Dropout1d(0.2).kind // => 'Dropout1d'
 * ```
 */
export class Dropout1d extends Stochastic {
  constructor(p = 0.5) {
    super('Dropout1d', p, (x, o) => channelDropout(x, p, o))
  }
}

/**
 * Channel dropout for `[B, C, H, W]` inputs (torch.nn.Dropout2d).
 *
 * @example
 * ```ts
 * import { Dropout2d } from 'omai-engine/nn'
 * new Dropout2d(0.2).kind // => 'Dropout2d'
 * ```
 */
export class Dropout2d extends Stochastic {
  constructor(p = 0.5) {
    super('Dropout2d', p, (x, o) => channelDropout(x, p, o))
  }
}

/**
 * Channel dropout for `[B, C, D, H, W]` inputs (torch.nn.Dropout3d).
 *
 * @example
 * ```ts
 * import { Dropout3d } from 'omai-engine/nn'
 * new Dropout3d(0.2).kind // => 'Dropout3d'
 * ```
 */
export class Dropout3d extends Stochastic {
  constructor(p = 0.5) {
    super('Dropout3d', p, (x, o) => channelDropout(x, p, o))
  }
}

/**
 * Alpha dropout for self-normalising networks (torch.nn.AlphaDropout).
 *
 * @example
 * ```ts
 * import { AlphaDropout } from 'omai-engine/nn'
 * new AlphaDropout(0.1).kind // => 'AlphaDropout'
 * ```
 */
export class AlphaDropout extends Stochastic {
  constructor(p = 0.5) {
    super('AlphaDropout', p, (x, o) => alphaDropout(x, p, o))
  }
}

/**
 * Alpha dropout of whole channels (torch.nn.FeatureAlphaDropout).
 *
 * @example
 * ```ts
 * import { FeatureAlphaDropout } from 'omai-engine/nn'
 * new FeatureAlphaDropout(0.1).kind // => 'FeatureAlphaDropout'
 * ```
 */
export class FeatureAlphaDropout extends Stochastic {
  constructor(p = 0.5) {
    super('FeatureAlphaDropout', p, (x, o) => featureAlphaDropout(x, p, o))
  }
}

/**
 * Drop path (stochastic depth) of a residual branch (timm's DropPath).
 *
 * @example
 * ```ts
 * import { DropPath } from 'omai-engine/nn'
 * new DropPath(0.1).kind // => 'DropPath'
 * ```
 */
export class DropPath extends Stochastic {
  constructor(p = 0.1, scaleByKeep = true) {
    super('DropPath', p, (x, o) => dropPath(x, p, { ...o, scaleByKeep }))
  }
}

/**
 * Additive (or multiplicative) Gaussian noise in training (Keras's GaussianNoise and GaussianDropout).
 *
 * @example
 * ```ts
 * import { GaussianNoise } from 'omai-engine/nn'
 * new GaussianNoise(0.1).kind // => 'GaussianNoise'
 * ```
 */
export class GaussianNoise extends Module {
  readonly kind = 'GaussianNoise'
  readonly std: number
  readonly multiplicative: boolean

  constructor(std: number, multiplicative = false) {
    super()
    this.std = std
    this.multiplicative = multiplicative
  }

  apply(_params: Params, x: Value, ctx: ApplyContext = {}): Value {
    return gaussianNoise(x, this.std, {
      training: isTraining(ctx),
      multiplicative: this.multiplicative,
      ...(ctx.key === undefined ? {} : { key: ctx.key }),
    })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `std=${this.std}${this.multiplicative ? ', multiplicative' : ''}`
  }
}

/**
 * Cutout of one random square patch per image in training (DeVries and Taylor 2017).
 *
 * @example
 * ```ts
 * import { Cutout } from 'omai-engine/nn'
 * new Cutout(8).kind // => 'Cutout'
 * ```
 */
export class Cutout extends Module {
  readonly kind = 'Cutout'
  readonly size: number

  constructor(size: number) {
    super()
    this.size = size
  }

  apply(_params: Params, x: Value, ctx: ApplyContext = {}): Value {
    return cutout(x, this.size, { training: isTraining(ctx), ...(ctx.key === undefined ? {} : { key: ctx.key }) })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `size=${this.size}`
  }
}
