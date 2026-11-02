/**
 * The NeRF fit as a method a session can drive: `fitNerfMethod` composes a scene (`multiViewImages`), a radiance field
 * from the encoding slots, an optional fine network for hierarchical sampling, and an optimiser from the `optimisers`
 * table, and runs `fitNerfAlgorithm` with the contract of `nerfContract`: the loss and peak signal-to-noise ratio per
 * step, a view rendered at any step, the weights along a ray and the held-out peak signal-to-noise ratio.
 */

import { generate } from '../data/index.ts'
import { multiViewImages } from '../datasets/index.ts'
import { optimisers } from '../optim/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { bool, choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { defineMethod, type Method } from '../trace/index.ts'
import {
  type FitNerfOptions,
  fitNerfAlgorithm,
  type NerfData,
  type NerfPath,
  type NerfState,
  nerfContract,
  nerfData,
} from './fit-nerf.ts'
import { type NerfModel, nerfModel } from './nerf.ts'
import { fieldEncodings } from './tables.ts'

/** The configuration of {@link fitNerfMethod}. */
export interface NerfMethodConfig {
  readonly [knob: string]: SpaceValue
  /** Training views, held-out views, image side in pixels and objects in the scene. */
  readonly views: number
  readonly heldOut: number
  readonly resolution: number
  readonly objects: number
  /** The cameras' vertical field of view in degrees (default 40) and their distance from the scene's centre (default 4). */
  readonly fov?: number
  readonly radius?: number
  readonly positionEncoding: SpaceValue
  readonly directionEncoding: SpaceValue
  readonly width: number
  readonly depth: number
  readonly densityActivation: string
  /** Add a fine network and resample the coarse weights (hierarchical sampling). */
  readonly hierarchical: boolean
  readonly samples: number
  readonly fineSamples: number
  readonly batchRays: number
  readonly optimiser: SpaceValue
  readonly learningRate: number
}

/** The data, model and fit options a configuration and a key make. */
export interface NerfProblem {
  readonly data: NerfData
  readonly model: NerfModel
  readonly options: FitNerfOptions
}

/**
 * The problem a configuration describes: the scene's views, the model of its encoding slots and the fit's options. The
 * same key makes the same problem.
 *
 * @param config - The method's configuration.
 * @param key - The key.
 * @returns The data, the model and the options.
 * @example
 * ```ts
 * import { nerfProblem } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const config = { views: 3, heldOut: 1, resolution: 4, objects: 1, positionEncoding: { kind: 'frequency', octaves: 2, includeInput: true }, directionEncoding: { kind: 'frequency', octaves: 1, includeInput: true }, width: 8, depth: 1, densityActivation: 'softplus', hierarchical: false, samples: 4, fineSamples: 4, batchRays: 8, optimiser: { kind: 'adam' }, learningRate: 0.01 }
 * nerfProblem(config, key(0)).data.images.shape // => [3, 4, 4, 3]
 * ```
 */
export function nerfProblem(config: NerfMethodConfig, key: Key): NerfProblem {
  const data = nerfData(
    generate(multiViewImages, {
      key: foldIn(key, 0),
      knobs: {
        views: config.views,
        heldOut: config.heldOut,
        resolution: config.resolution,
        objects: config.objects,
        ...(config.fov === undefined ? {} : { fov: config.fov }),
        ...(config.radius === undefined ? {} : { radius: config.radius }),
      },
    }),
  )
  const model = nerfModel({
    positionEncoding: config.positionEncoding,
    directionEncoding: config.directionEncoding,
    width: config.width,
    depth: config.depth,
    densityActivation: config.densityActivation === 'relu' ? 'relu' : 'softplus',
    fine: config.hierarchical,
    key: foldIn(key, 1),
  })
  return {
    data,
    model,
    options: {
      optimizer: optimisers.make(config.optimiser)(config.learningRate),
      batchRays: config.batchRays,
      samples: config.samples,
      fineSamples: config.fineSamples,
    },
  }
}

/**
 * Fitting a radiance field as a method: choose the scene, the encodings, the network, whether to sample hierarchically
 * and the optimiser, and a session runs the fit and offers a view rendered at any step (`nerf/render`), the weights along
 * a ray (`nerf/rayWeights`) and the peak signal-to-noise ratio on the held-out views (`nerf/heldOutPsnr`).
 *
 * @concept neural-radiance-fields
 * @example
 * ```ts
 * import { fitNerfMethod } from 'mdml-engine/fields'
 * import { Log, metricSeries, session } from 'mdml-engine/trace'
 * const s = session(fitNerfMethod, {
 *   config: { views: 3, heldOut: 1, resolution: 4, objects: 1, width: 8, depth: 1, samples: 4, batchRays: 8 },
 *   logs: { 'nerf/loss': Log() },
 *   mode: 'eager',
 *   steps: 3,
 * })
 * metricSeries(s.run(), 'nerf/loss').values.length // => 3
 * ```
 */
export const fitNerfMethod: Method<NerfState, NerfMethodConfig, NerfPath> = defineMethod<
  NerfState,
  NerfMethodConfig,
  NerfPath
>({
  name: 'fit-nerf',
  config: dict({
    views: int(2, 100, { default: 8, label: 'training views' }),
    heldOut: int(0, 20, { default: 3, label: 'held-out views' }),
    resolution: int(4, 64, { default: 8, label: 'image side' }),
    objects: int(1, 4, { default: 1, label: 'objects in the scene' }),
    fov: real(10, 120, { default: 40, label: 'camera field of view (degrees)' }),
    radius: real(2.5, 20, { default: 4, label: 'camera distance from the scene' }),
    positionEncoding: { ...fieldEncodings.space, default: 'frequency' },
    directionEncoding: { ...fieldEncodings.space, default: 'frequency' },
    width: int(4, 256, { default: 24, label: 'width' }),
    depth: int(1, 8, { default: 2, label: 'trunk layers' }),
    densityActivation: choice(['softplus', 'relu'], { default: 'softplus', label: 'density activation' }),
    hierarchical: bool({ default: false, label: 'coarse and fine networks' }),
    samples: int(2, 128, { default: 8, label: 'samples per ray' }),
    fineSamples: int(2, 128, { default: 8, label: 'fine samples per ray' }),
    batchRays: int(1, 4096, { default: 64, label: 'rays per step' }),
    optimiser: { ...optimisers.space, default: 'adam' },
    learningRate: real(1e-5, 1, { log: true, default: 5e-3, label: 'learning rate' }),
  }),
  make: (config, key) => {
    const p = nerfProblem(config, key)
    return fitNerfAlgorithm(p.model, p.data, p.options)
  },
  contract: (config, key) => {
    const p = nerfProblem(config, key)
    return nerfContract(p.model, p.data, p.options)
  },
})
