/**
 * The slots of a neural field as option tables: `fieldEncodings` (how a coordinate is lifted to features: none, the
 * NeRF positional encoding, Gaussian random Fourier features, a multiresolution hash grid) and `fieldBackbones` (the
 * network on the features: a ReLU-family MLP or a SIREN), each option with its own knobs and a `make` that builds
 * the module. `neuralField` fills the two slots from them; a page or a tuner varies the slot without touching the model.
 */

import {
  FourierFeatureEncoding,
  FrequencyEncoding,
  GELU,
  HashGridEncoding,
  IdentityEncoding,
  type Module,
  mlp,
  ReLU,
  SiLU,
  siren,
  Tanh,
} from '../nn/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { bool, choice, dict, int, type OptionTable, optionTable, real } from '../space/index.ts'

/** An encoding built for a coordinate dimension: the module and the width of what it outputs. */
export interface BuiltEncoding {
  readonly module: Module
  readonly outputDim: number
}

/** An encoding option's shape: given the coordinate dimension and a key, the encoding. */
export type EncodingShape = (inputDim: number, key: Key) => BuiltEncoding

/** What an encoding does, for selectors and validity rules. */
export interface EncodingProperties {
  /** Periodic in the coordinate (sines and cosines), so it cannot extrapolate but represents fine detail. */
  readonly periodic: boolean
  /** Carries trainable parameters of its own (a hash grid's table). */
  readonly trainable: boolean
}

/**
 * How coordinates are encoded before the backbone: `none` (the raw coordinates), `frequency` (the NeRF positional
 * encoding with `octaves` octaves, optionally with the input), `fourier` (Gaussian random Fourier features: `features`
 * frequencies drawn `N(0, sigma^2)` from the key) and `hash` (the Instant-NGP multiresolution hash grid, 1 to 3
 * dimensions).
 *
 * @example
 * ```ts
 * import { fieldEncodings } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * fieldEncodings.make({ kind: 'frequency', octaves: 4, includeInput: true })(2, key(0)).outputDim // => 18
 * fieldEncodings.make({ kind: 'fourier', features: 16, sigma: 10 })(2, key(0)).outputDim // => 32
 * fieldEncodings.properties({ kind: 'hash' }).trainable // => true
 * ```
 */
export const fieldEncodings: OptionTable<EncodingShape, EncodingProperties> = optionTable<
  EncodingShape,
  EncodingProperties
>(
  {
    none: {
      description: 'the raw coordinates',
      space: dict({}),
      properties: { periodic: false, trainable: false },
      make: () => (inputDim) => ({ module: new IdentityEncoding(inputDim), outputDim: inputDim }),
    },
    frequency: {
      description: 'the NeRF positional encoding: sines and cosines of 2^l pi times each coordinate',
      space: dict({
        octaves: int(1, 16, { default: 6, label: 'octaves L' }),
        includeInput: bool({ default: true, label: 'include the input' }),
      }),
      properties: { periodic: true, trainable: false },
      make: (k) => (inputDim) => {
        const module = new FrequencyEncoding(inputDim, {
          octaves: (k['octaves'] as number | undefined) ?? 6,
          includeInput: (k['includeInput'] as boolean | undefined) ?? true,
        })
        return { module, outputDim: module.outputDim }
      },
    },
    fourier: {
      description: 'Gaussian random Fourier features of the coordinates (sin and cos of 2 pi v B, B ~ N(0, sigma^2))',
      space: dict({
        features: int(1, 512, { default: 64, label: 'frequencies m' }),
        sigma: real(0.1, 100, { log: true, default: 10, label: 'frequency scale sigma' }),
      }),
      properties: { periodic: true, trainable: false },
      make: (k) => (inputDim, key) => {
        const module = new FourierFeatureEncoding(inputDim, {
          features: (k['features'] as number | undefined) ?? 64,
          sigma: (k['sigma'] as number | undefined) ?? 10,
          key: foldIn(key, 101),
        })
        return { module, outputDim: module.outputDim }
      },
    },
    hash: {
      description: 'a multiresolution hash grid of trainable features, read by multilinear interpolation (Instant-NGP)',
      space: dict({
        levels: int(1, 24, { default: 8, label: 'levels L' }),
        log2TableSize: int(4, 22, { default: 12, label: 'log2 table size' }),
        featuresPerLevel: int(1, 8, { default: 2, label: 'features per level F' }),
        baseResolution: int(1, 256, { default: 4, label: 'base resolution' }),
        finestResolution: int(2, 4096, { default: 64, label: 'finest resolution' }),
        bound: real(0.1, 100, { log: true, default: 1, label: 'half-width of the box covered' }),
      }),
      properties: { periodic: false, trainable: true },
      make: (k) => (inputDim, key) => {
        const module = new HashGridEncoding(
          {
            dim: inputDim,
            levels: (k['levels'] as number | undefined) ?? 8,
            log2TableSize: (k['log2TableSize'] as number | undefined) ?? 12,
            featuresPerLevel: (k['featuresPerLevel'] as number | undefined) ?? 2,
            baseResolution: (k['baseResolution'] as number | undefined) ?? 4,
            finestResolution: (k['finestResolution'] as number | undefined) ?? 64,
            bounds: [-((k['bound'] as number | undefined) ?? 1), (k['bound'] as number | undefined) ?? 1],
          },
          { key: foldIn(key, 102) },
        )
        return { module, outputDim: module.outputDim }
      },
    },
  },
  { default: 'none', label: 'coordinate encoding' },
)

/** A backbone option's shape: from the input width, the output width, the hidden width and depth and a key, the network. */
export type BackboneShape = (inputDim: number, outputDim: number, width: number, depth: number, key: Key) => Module

/** What a backbone is, for selectors. */
export interface BackboneProperties {
  /** Periodic activations (SIREN): derivatives are as smooth as the function. */
  readonly periodic: boolean
}

/**
 * The network on the encoded coordinates: `mlp` (Linear layers with a `relu`, `gelu`, `silu` or `tanh` activation) and
 * `siren` (sine layers with frequency factor `omega0` and the paper's initialisation, then an affine output). `depth`
 * is the number of hidden layers.
 *
 * @example
 * ```ts
 * import { fieldBackbones } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * fieldBackbones.make({ kind: 'siren', omega0: 30 })(2, 3, 16, 2, key(0)).outputShape([5, 2]) // => [5, 3]
 * ```
 */
export const fieldBackbones: OptionTable<BackboneShape, BackboneProperties> = optionTable<
  BackboneShape,
  BackboneProperties
>(
  {
    mlp: {
      description: 'a multilayer perceptron',
      space: dict({ activation: choice(['relu', 'gelu', 'silu', 'tanh'], { default: 'relu', label: 'activation' }) }),
      properties: { periodic: false },
      make: (k) => (inputDim, outputDim, width, depth, key) => {
        const act = (k['activation'] as string | undefined) ?? 'relu'
        const make = (): Module =>
          act === 'gelu' ? new GELU() : act === 'silu' ? new SiLU() : act === 'tanh' ? new Tanh() : new ReLU()
        return mlp([inputDim, ...Array.from({ length: depth }, () => width), outputDim], { activation: make, key })
      },
    },
    siren: {
      description: 'a sine-activated network (SIREN) with the paper’s initialisation',
      space: dict({ omega0: real(1, 100, { default: 30, label: 'frequency factor omega0' }) }),
      properties: { periodic: true },
      make: (k) => (inputDim, outputDim, width, depth, key) =>
        siren([inputDim, ...Array.from({ length: depth }, () => width), outputDim], {
          omega0: (k['omega0'] as number | undefined) ?? 30,
          key,
        }),
    },
  },
  { default: 'mlp', label: 'backbone' },
)
