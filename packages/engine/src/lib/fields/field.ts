/**
 * The neural field model: a network from coordinates to a signal's value (an image's colour, a distance, a density),
 * built from two slots, a coordinate encoding and a backbone, and an output head. `neuralField` fills the slots from the
 * option tables (`fieldEncodings`, `fieldBackbones`), so the same model is a plain ReLU MLP, an MLP over the NeRF
 * positional encoding, over Gaussian Fourier features or a hash grid, or a SIREN, by changing configuration only.
 */

import { initKey, type Module, Sequential, Sigmoid } from '../nn/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fieldBackbones, fieldEncodings } from './tables.ts'

/** Options of {@link neuralField}. */
export interface NeuralFieldOptions {
  /** The coordinate dimension (1 for a signal, 2 for an image or planar SDF, 3 for a volume or SDF). */
  readonly inputDim: number
  /** The width of the signal's value (3 for colour, 1 for a distance or density). */
  readonly outputDim: number
  /** A `fieldEncodings` configuration (default `{ kind: 'none' }`). */
  readonly encoding?: SpaceValue
  /** A `fieldBackbones` configuration (default `{ kind: 'mlp' }`). */
  readonly backbone?: SpaceValue
  /** The hidden width (default 64). */
  readonly width?: number
  /** The number of hidden layers (default 3). */
  readonly depth?: number
  /** The output head's squashing: `linear` (default) or `sigmoid` (values in [0, 1], for colours). */
  readonly output?: 'linear' | 'sigmoid'
  readonly key?: Key
}

/**
 * A neural field: `encoding -> backbone [-> sigmoid]`, a `Sequential` whose layers are the encoding and the backbone's
 * layers (the last of which is the affine output head), so its parameters are keyed by position as `mlp`'s are.
 *
 * @concept neural-fields
 */
export class NeuralField extends Sequential {
  override readonly kind: string = 'NeuralField'
  readonly inputDim: number
  readonly outputDim: number
  /** The encoding module (the first layer) and the width of its output. */
  readonly encoding: Module
  readonly encodedDim: number
  readonly options: Required<Omit<NeuralFieldOptions, 'key' | 'encoding' | 'backbone'>> & {
    readonly encoding: SpaceValue
    readonly backbone: SpaceValue
  }

  constructor(options: NeuralFieldOptions) {
    const key = initKey('NeuralField', options.key)
    const encodingConfig = options.encoding ?? { kind: 'none' }
    const backboneConfig = options.backbone ?? { kind: 'mlp' }
    const width = options.width ?? 64
    const depth = options.depth ?? 3
    const output = options.output ?? 'linear'
    const encoding = fieldEncodings.make(encodingConfig)(options.inputDim, foldIn(key, 0))
    const backbone = fieldBackbones.make(backboneConfig)(
      encoding.outputDim,
      options.outputDim,
      width,
      depth,
      foldIn(key, 1),
    )
    super(encoding.module, backbone, ...(output === 'sigmoid' ? [new Sigmoid()] : []))
    this.inputDim = options.inputDim
    this.outputDim = options.outputDim
    this.encoding = encoding.module
    this.encodedDim = encoding.outputDim
    this.options = {
      inputDim: options.inputDim,
      outputDim: options.outputDim,
      width,
      depth,
      output,
      encoding: encodingConfig,
      backbone: backboneConfig,
    }
  }

  override extraRepr(): string {
    return `in=${this.inputDim}, out=${this.outputDim}, width=${this.options.width}, depth=${this.options.depth}`
  }
}

/**
 * Build a neural field from an encoding and a backbone slot.
 *
 * @param options - The dimensions, the two slots, the width and depth, the output head and a key.
 * @returns The field.
 * @concept neural-fields
 * @example
 * ```ts
 * import { neuralField } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const f = neuralField({ inputDim: 2, outputDim: 3, encoding: { kind: 'frequency', octaves: 4 }, width: 16, depth: 2, key: key(0) })
 * [f.encodedDim, f.forward([[0.1, 0.2]]).shape] // => [18, [1, 3]]
 * ```
 */
export function neuralField(options: NeuralFieldOptions): NeuralField {
  return new NeuralField(options)
}
