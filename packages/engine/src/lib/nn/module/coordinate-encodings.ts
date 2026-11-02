/**
 * The coordinate encodings of neural fields as modules: `IdentityEncoding` (the raw coordinates), `FrequencyEncoding` (the NeRF positional encoding),
 * `FourierFeatureEncoding` (Gaussian random Fourier features, the frequency matrix fixed by a key) and
 * `HashGridEncoding` (the Instant-NGP multiresolution hash grid, whose table is the module's one parameter). Each has
 * an `outputDim`, so a backbone can be sized from it; the functional forms are in `functional/coordinate-encodings`.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import type { Tensor, Value } from '../../tensor/index.ts'
import {
  fourierFeatures,
  gaussianFrequencies,
  type HashGridConfig,
  hashGridEncode,
  hashGridTableShape,
  positionalEncoding,
  positionalEncodingWidth,
  type ResolvedHashGrid,
  resolveHashGrid,
} from '../functional/coordinate-encodings.ts'
import { initKey, Module, type Params } from './module.ts'

/**
 * The identity encoding: the raw coordinates, so a field with no encoding has the same interface as one with. No
 * parameters.
 *
 * @concept coordinate-encoding
 * @example
 * ```ts
 * import { IdentityEncoding } from 'mdml-engine/nn'
 * new IdentityEncoding(3).forward([[1, 2, 3]]).shape // => [1, 3]
 * ```
 */
export class IdentityEncoding extends Module {
  readonly kind = 'IdentityEncoding'
  readonly inputDim: number
  readonly outputDim: number

  constructor(inputDim: number) {
    super()
    this.inputDim = inputDim
    this.outputDim = inputDim
  }

  apply(_params: Params, x: Value): Value {
    return x
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

/**
 * The NeRF positional encoding as a module: `[..., d]` coordinates to `[..., d * 2 L (+ d)]` sines and cosines of
 * `2^l pi` times each coordinate. No parameters.
 *
 * @concept coordinate-encoding
 * @example
 * ```ts
 * import { FrequencyEncoding } from 'mdml-engine/nn'
 * const enc = new FrequencyEncoding(3, { octaves: 4, includeInput: true })
 * [enc.outputDim, enc.forward([[0, 0, 0]]).shape] // => [27, [1, 27]]
 * ```
 */
export class FrequencyEncoding extends Module {
  readonly kind = 'FrequencyEncoding'
  readonly inputDim: number
  readonly octaves: number
  readonly includeInput: boolean
  readonly outputDim: number

  constructor(inputDim: number, options: { readonly octaves: number; readonly includeInput?: boolean }) {
    super()
    this.inputDim = inputDim
    this.octaves = options.octaves
    this.includeInput = options.includeInput ?? false
    this.outputDim = positionalEncodingWidth(inputDim, { octaves: this.octaves, includeInput: this.includeInput })
  }

  apply(_params: Params, x: Value): Value {
    return positionalEncoding(x, { octaves: this.octaves, includeInput: this.includeInput })
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outputDim]
  }

  override extraRepr(): string {
    return `in=${this.inputDim}, octaves=${this.octaves}, include_input=${this.includeInput}`
  }
}

/**
 * Gaussian random Fourier features as a module: `(sin(2 pi v B), cos(2 pi v B))` with `B` of shape `[d, m]` drawn
 * `N(0, sigma^2)` from the key and then fixed (it is a constant of the module, not a parameter, so it is the same for
 * the same key however the module is rebuilt; read it as `matrix`).
 *
 * @concept coordinate-encoding
 * @example
 * ```ts
 * import { FourierFeatureEncoding } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const enc = new FourierFeatureEncoding(2, { features: 16, sigma: 5, key: key(0) })
 * [enc.outputDim, enc.parameterCount()] // => [32, 0]
 * ```
 */
export class FourierFeatureEncoding extends Module {
  readonly kind = 'FourierFeatureEncoding'
  readonly inputDim: number
  readonly features: number
  readonly sigma: number
  readonly outputDim: number
  /** The frequency matrix `B` `[d, m]`. */
  readonly matrix: Tensor

  constructor(inputDim: number, options: { readonly features: number; readonly sigma?: number; readonly key?: Key }) {
    super()
    this.inputDim = inputDim
    this.features = options.features
    this.sigma = options.sigma ?? 10
    this.outputDim = 2 * options.features
    this.matrix = gaussianFrequencies(
      initKey('FourierFeatureEncoding', options.key),
      inputDim,
      options.features,
      this.sigma,
    )
  }

  apply(_params: Params, x: Value): Value {
    return fourierFeatures(x, this.matrix)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape.slice(0, -1), this.outputDim]
  }

  override extraRepr(): string {
    return `in=${this.inputDim}, features=${this.features}, sigma=${this.sigma}`
  }
}

/**
 * The multiresolution hash grid encoding as a module. Its one parameter is the `table` `[L * 2^log2TableSize, F]`,
 * initialised uniform on +-1e-4 as in Instant-NGP, so training starts from a nearly constant field; the output is the
 * `L F` interpolated features. Coordinates are read within `bounds` (default `[-1, 1]`).
 *
 * @concept coordinate-encoding
 * @index module Instant-NGP multiresolution hash grid encoding of coordinates.
 * @example
 * ```ts
 * import { HashGridEncoding } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const enc = new HashGridEncoding({ dim: 2, levels: 4, log2TableSize: 8 }, { key: key(0) })
 * [enc.outputDim, enc.parameterCount(), enc.forward([[0.1, -0.3]]).shape] // => [8, 2048, [1, 8]]
 * ```
 */
export class HashGridEncoding extends Module {
  readonly kind = 'HashGridEncoding'
  readonly config: ResolvedHashGrid
  readonly outputDim: number

  constructor(config: HashGridConfig, options: { readonly key?: Key } = {}) {
    super()
    this.config = resolveHashGrid(config)
    this.outputDim = this.config.levels * this.config.featuresPerLevel
    this.params = this.initParams(initKey('HashGridEncoding', options.key))
  }

  override initParams(key: Key): Params {
    return { table: uniform(foldIn(key, 0), hashGridTableShape(this.config), { low: -1e-4, high: 1e-4 }) }
  }

  apply(params: Params, x: Value): Value {
    const table = params['table'] as Value | undefined
    if (table === undefined) throw new MdmlError('value', 'HashGridEncoding: the parameters hold no table')
    return hashGridEncode(x, table, this.config)
  }

  outputShape(inputShape: readonly number[]): number[] {
    if (inputShape[inputShape.length - 1] !== this.config.dim)
      throw new MdmlError(
        'shape',
        `HashGridEncoding: expected ${this.config.dim} coordinates, got [${inputShape.join(', ')}]`,
      )
    return [...inputShape.slice(0, -1), this.outputDim]
  }

  override extraRepr(): string {
    const c = this.config
    return `dim=${c.dim}, levels=${c.levels}, log2_table_size=${c.log2TableSize}, features=${c.featuresPerLevel}`
  }
}
