/**
 * SIREN (Sitzmann et al. 2020), sinusoidal representation networks: layers `sin(omega0 (W x + b))` with the paper's
 * initialisation, which keeps the distribution of activations stable through depth and makes the derivatives of the
 * network (also sinusoids) as well behaved as the signal, so a SIREN fits images, audio and signed distance fields
 * with their gradients. `SineLayer` is one layer (or the final linear layer of the network) and `siren` the stack.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import { mul, sin, type Value } from '../../tensor/index.ts'
import { linear } from '../functional/linear.ts'
import { torchDefaultUniform } from '../init.ts'
import { Sequential } from './containers.ts'
import { initKey, Module, type Params } from './module.ts'

/** Options of {@link SineLayer}. */
export interface SineLayerOptions {
  /** The frequency factor `omega0` (default 30). */
  readonly omega0?: number
  /** The first layer of the network: weights uniform on +-1/n instead of the hidden layers' bound (default false). */
  readonly first?: boolean
  /** A plain affine layer without the sine, for the output (default false). */
  readonly linear?: boolean
  readonly key?: Key
}

/**
 * One SIREN layer `sin(omega0 (W x + b))`. Weights are uniform on `+-1/n` in the first layer and on
 * `+-sqrt(6/n)/omega0` after it (`n` the input width), so the pre-activations `omega0 W x` of a hidden layer are
 * standard-arcsine-distributed whatever the depth; biases are PyTorch's default `+-1/sqrt(n)`. With `linear` it is the
 * affine output layer with the hidden layers' initialisation.
 *
 * @concept siren
 * @reference Sitzmann, Martel, Bergman, Lindell and Wetzstein (2020), Implicit neural representations with periodic
 *   activation functions.
 * @example
 * ```ts
 * import { SineLayer } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const layer = new SineLayer(2, 8, { first: true, key: key(0) })
 * [layer.parameterCount(), layer.forward([[0.1, 0.2]]).shape] // => [24, [1, 8]]
 * ```
 */
export class SineLayer extends Module {
  readonly kind = 'SineLayer'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly omega0: number
  readonly first: boolean
  readonly affine: boolean

  constructor(inFeatures: number, outFeatures: number, options: SineLayerOptions = {}) {
    super()
    this.inFeatures = inFeatures
    this.outFeatures = outFeatures
    this.omega0 = options.omega0 ?? 30
    this.first = options.first ?? false
    this.affine = options.linear ?? false
    this.params = this.initParams(initKey('SineLayer', options.key))
  }

  /** The bound of the weights' uniform initialisation. */
  get weightBound(): number {
    return this.first ? 1 / this.inFeatures : Math.sqrt(6 / this.inFeatures) / this.omega0
  }

  override initParams(key: Key): Params {
    const b = this.weightBound
    return {
      weight: uniform(foldIn(key, 0), [this.outFeatures, this.inFeatures], { low: -b, high: b }),
      bias: torchDefaultUniform(foldIn(key, 1), [this.outFeatures], this.inFeatures),
    }
  }

  apply(params: Params, x: Value): Value {
    const z = linear(x, params['weight'] as Value, params['bias'] as Value)
    return this.affine ? z : sin(mul(z, this.omega0))
  }

  outputShape(inputShape: readonly number[]): number[] {
    if (inputShape[inputShape.length - 1] !== this.inFeatures)
      throw new OmaiError(
        'shape',
        `SineLayer: expected ${this.inFeatures} input features, got [${inputShape.join(', ')}]`,
      )
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in_features=${this.inFeatures}, out_features=${this.outFeatures}, omega0=${this.omega0}, first=${this.first}`
  }
}

/**
 * A SIREN: sine layers of the given widths (the first with its own initialisation) and an affine output layer.
 *
 * @param widths - Widths from input to output (at least two); every layer but the last is a sine layer.
 * @param options - `omega0` (default 30) and a key.
 * @returns The network.
 * @concept siren
 * @index module SIREN: a sine-activated network for coordinate inputs, from layer widths.
 * @example
 * ```ts
 * import { siren } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * siren([2, 16, 16, 3], { key: key(0) }).forward([[0.5, -0.5]]).shape // => [1, 3]
 * ```
 */
export function siren(
  widths: readonly number[],
  options: { readonly omega0?: number; readonly key?: Key } = {},
): Sequential {
  if (widths.length < 2) throw new OmaiError('value', 'siren: need at least an input and an output width')
  const layers: Module[] = []
  for (let i = 0; i + 1 < widths.length; i++) {
    layers.push(
      new SineLayer(widths[i] as number, widths[i + 1] as number, {
        ...(options.omega0 === undefined ? {} : { omega0: options.omega0 }),
        first: i === 0,
        linear: i + 2 === widths.length,
        key: foldIn(options.key ?? initKey('Siren'), i),
      }),
    )
  }
  return new Sequential(...layers)
}
