/**
 * The field fit as a method a session can drive: `fitFieldMethod` composes a signal from the dataset generators
 * (`fieldImage`, `signedDistance`), a field from the encoding and backbone slots and an optimiser from the `optimisers`
 * table, and runs `fitFieldAlgorithm` with the contract of `fieldContract`. Everything a viewer may choose (the signal,
 * the encoding, the backbone, the optimiser, its learning rate) is a knob of the method's configuration.
 */

import { generate } from '../data/index.ts'
import { fieldImage, signedDistance } from '../datasets/index.ts'
import { type OptimiserShape, optimisers } from '../optim/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { defineMethod, type Method } from '../trace/index.ts'
import { type NeuralField, neuralField } from './field.ts'
import {
  type FieldData,
  type FieldPath,
  type FieldState,
  type FitFieldOptions,
  fieldContract,
  fitFieldAlgorithm,
} from './fit-field.ts'
import { fieldBackbones, fieldEncodings } from './tables.ts'

/** The configuration of {@link fitFieldMethod}. */
export interface FieldMethodConfig {
  readonly [knob: string]: SpaceValue
  /** The signal: the procedural `image`, or the signed distance of a `sphere`, `box`, `torus` or `union`. */
  readonly signal: string
  /** The image's side in pixels, or the square root of the number of distance samples. */
  readonly resolution: number
  /** The dimension of a distance field (2 or 3). */
  readonly dim: number
  readonly encoding: SpaceValue
  readonly backbone: SpaceValue
  readonly width: number
  readonly depth: number
  readonly optimiser: SpaceValue
  readonly learningRate: number
  /** Rows per step; 0 takes all of them. */
  readonly batchSize: number
  /** The weight of the eikonal term of a distance field (0 off). */
  readonly eikonalWeight: number
  /** Cells per side of the decoding probe. */
  readonly probeResolution: number
}

/** The signal, field and fit options a configuration and a key make. */
export interface FieldProblem {
  readonly data: FieldData
  readonly model: NeuralField
  readonly options: FitFieldOptions
}

/**
 * The problem a configuration describes: the signal sampled by its generator, the field of its two slots and the fit's
 * options with the optimiser filled from the table. The same key makes the same problem.
 *
 * @param config - The method's configuration.
 * @param key - The key.
 * @returns The data, the model and the options.
 * @example
 * ```ts
 * import { fieldProblem } from 'omai-engine/fields'
 * import { key } from 'omai-engine/random'
 * fieldProblem({ signal: 'image', resolution: 8, dim: 2, encoding: { kind: 'none' }, backbone: { kind: 'mlp' }, width: 8, depth: 1, optimiser: { kind: 'adam' }, learningRate: 0.01, batchSize: 0, eikonalWeight: 0, probeResolution: 8 }, key(0)).data.x.shape // => [64, 2]
 * ```
 */
export function fieldProblem(config: FieldMethodConfig, key: Key): FieldProblem {
  const image = config.signal === 'image'
  const generated = image
    ? generate(fieldImage, { key: foldIn(key, 0), knobs: { side: config.resolution } })
    : generate(signedDistance, {
        key: foldIn(key, 0),
        knobs: { shape: config.signal, dim: config.dim, n: config.resolution * config.resolution },
      })
  const model = neuralField({
    inputDim: image ? 2 : config.dim,
    outputDim: image ? 3 : 1,
    encoding: config.encoding,
    backbone: config.backbone,
    width: config.width,
    depth: config.depth,
    key: foldIn(key, 1),
  })
  const optimiser: OptimiserShape = optimisers.make(config.optimiser)
  return {
    data: { x: generated.data.x, y: generated.data.y },
    model,
    options: {
      optimizer: optimiser(config.learningRate),
      ...(config.batchSize > 0 ? { batchSize: config.batchSize } : {}),
      ...(!image && config.eikonalWeight > 0 ? { eikonal: { weight: config.eikonalWeight, extraPoints: 64 } } : {}),
      decode: { resolution: config.probeResolution },
    },
  }
}

/**
 * Fitting a neural field as a method: choose the signal, the encoding, the backbone and the optimiser, and a session
 * runs the fit, logging the loss and the peak signal-to-noise ratio and offering the field decoded on a grid at any
 * step (`field/render`) and the error on the whole data (`field/fullMse`).
 *
 * @concept neural-fields
 * @index estimator Fit an image or SDF with a neural field as a session.
 * @example
 * ```ts
 * import { fitFieldMethod } from 'omai-engine/fields'
 * import { metricSeries, Log, session } from 'omai-engine/trace'
 * const s = session(fitFieldMethod, {
 *   config: { resolution: 8, width: 8, depth: 1, encoding: { kind: 'frequency', octaves: 3, includeInput: true } },
 *   logs: { 'field/loss': Log() },
 *   mode: 'eager',
 *   steps: 20,
 * })
 * metricSeries(s.run(), 'field/loss').values.length // => 20
 * ```
 */
export const fitFieldMethod: Method<FieldState, FieldMethodConfig, FieldPath> = defineMethod<
  FieldState,
  FieldMethodConfig,
  FieldPath
>({
  name: 'fit-field',
  config: dict({
    signal: choice(['image', 'sphere', 'box', 'torus', 'union'], { default: 'image', label: 'signal' }),
    resolution: int(4, 128, { default: 24, label: 'resolution' }),
    dim: choice([2, 3], { default: 2, label: 'distance field dimension' }),
    encoding: fieldEncodings.space,
    backbone: fieldBackbones.space,
    width: int(2, 256, { default: 32, label: 'hidden width' }),
    depth: int(1, 8, { default: 2, label: 'hidden layers' }),
    optimiser: { ...optimisers.space, default: 'adam' },
    learningRate: real(1e-5, 1, { log: true, default: 3e-3, label: 'learning rate' }),
    batchSize: int(0, 100_000, { default: 0, label: 'batch size (0: all rows)' }),
    eikonalWeight: real(0, 10, { default: 0, label: 'eikonal weight' }),
    probeResolution: int(4, 256, { default: 24, label: 'probe resolution' }),
  }),
  make: (config, key) => {
    const p = fieldProblem(config, key)
    return fitFieldAlgorithm(p.model, p.data, p.options)
  },
  contract: (config, key) => {
    const p = fieldProblem(config, key)
    return fieldContract(p.model, p.data, p.options)
  },
})
