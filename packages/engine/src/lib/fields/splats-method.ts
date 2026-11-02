/**
 * The Gaussian-splatting fit as a method a session can drive: `fitSplatsMethod` composes a scene (`multiViewImages`, the
 * same views a NeRF fits, so the two can be compared), a start (random Gaussians in a box), an optimiser from the
 * `optimisers` table with a learning rate per parameter group, and optional adaptive density control, and runs
 * `fitSplatsAlgorithm` with the contract of `splatContract`.
 */

import { generate } from '../data/index.ts'
import { multiViewImages } from '../datasets/index.ts'
import { optimisers } from '../optim/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { bool, choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { defineMethod, type Method } from '../trace/index.ts'
import { type NerfData, nerfData } from './fit-nerf.ts'
import {
  type FitSplatsOptions,
  fitSplatsAlgorithm,
  type SplatPath,
  type SplatState,
  splatContract,
} from './fit-splats.ts'
import { type GaussianSet, randomGaussians } from './gaussians.ts'

/** The configuration of {@link fitSplatsMethod}. */
export interface SplatsMethodConfig {
  readonly [knob: string]: SpaceValue
  readonly views: number
  readonly heldOut: number
  readonly resolution: number
  readonly objects: number
  /** The number of Gaussians to start from, scattered in a box of this half-width, with this scale and opacity. */
  readonly gaussians: number
  readonly boxHalfWidth: number
  readonly initialScale: number
  readonly initialOpacity: number
  /** Degree of the spherical harmonics (0: one colour per Gaussian; 1: it depends on the view). */
  readonly shDegree: number
  readonly optimiser: SpaceValue
  readonly meansRate: number
  readonly scalesRate: number
  readonly rotationsRate: number
  readonly opacitiesRate: number
  readonly coloursRate: number
  /** The weight of the squared-error term beside the L1 term. */
  readonly l2Weight: number
  /** Adaptive density control: on or off, its interval and gradient threshold, the cap on the count. */
  readonly densify: boolean
  readonly densifyFrom: number
  readonly densifyUntil: number
  readonly densifyEvery: number
  readonly gradientThreshold: number
  readonly maxGaussians: number
}

/** The start, data and options a configuration and a key make. */
export interface SplatsProblem {
  readonly initial: GaussianSet
  readonly data: NerfData
  readonly options: FitSplatsOptions
}

/**
 * The problem a configuration describes. The same key makes the same problem.
 *
 * @param config - The method's configuration.
 * @param key - The key.
 * @returns The starting Gaussians, the data and the fit's options.
 * @example
 * ```ts
 * import { splatsProblem } from 'mdml-engine/fields'
 * import { key } from 'mdml-engine/random'
 * const config = { views: 3, heldOut: 1, resolution: 4, objects: 1, gaussians: 6, boxHalfWidth: 0.8, initialScale: 0.25, initialOpacity: 0.3, shDegree: 0, optimiser: { kind: 'adam' }, meansRate: 0.005, scalesRate: 0.005, rotationsRate: 0.001, opacitiesRate: 0.05, coloursRate: 0.02, l2Weight: 0, densify: false, densifyFrom: 20, densifyUntil: 400, densifyEvery: 20, gradientThreshold: 0.0005, maxGaussians: 100 }
 * splatsProblem(config, key(0)).initial.means.shape // => [6, 3]
 * ```
 */
export function splatsProblem(config: SplatsMethodConfig, key: Key): SplatsProblem {
  const data = nerfData(
    generate(multiViewImages, {
      key: foldIn(key, 0),
      knobs: { views: config.views, heldOut: config.heldOut, resolution: config.resolution, objects: config.objects },
    }),
  )
  const initial = randomGaussians(foldIn(key, 1), config.gaussians, {
    boxHalfWidth: config.boxHalfWidth,
    scale: config.initialScale,
    opacity: config.initialOpacity,
    shDegree: config.shDegree === 1 ? 1 : 0,
  })
  return {
    initial,
    data,
    options: {
      optimiser: optimisers.make(config.optimiser),
      learningRates: {
        means: config.meansRate,
        scales: config.scalesRate,
        rotations: config.rotationsRate,
        opacities: config.opacitiesRate,
        colours: config.coloursRate,
      },
      l2Weight: config.l2Weight,
      ...(config.densify
        ? {
            densify: {
              from: config.densifyFrom,
              until: config.densifyUntil,
              every: config.densifyEvery,
              gradientThreshold: config.gradientThreshold,
              maxGaussians: config.maxGaussians,
            },
          }
        : {}),
    },
  }
}

/**
 * Fitting 3-D Gaussians as a method: choose the scene, the number of Gaussians to start from, the optimiser and each
 * group's learning rate, and whether to adapt the set, and a session runs the fit and offers a view rendered at any step
 * (`splats/render`), the Gaussians of a view as ellipses (`splats/ellipses`) and the held-out peak signal-to-noise ratio.
 *
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { fitSplatsMethod } from 'mdml-engine/fields'
 * import { Log, metricSeries, session } from 'mdml-engine/trace'
 * const s = session(fitSplatsMethod, {
 *   config: { views: 3, heldOut: 1, resolution: 4, objects: 1, gaussians: 8 },
 *   logs: { 'splats/loss': Log(), 'splats/count': Log() },
 *   mode: 'eager',
 *   steps: 4,
 * })
 * metricSeries(s.run(), 'splats/count').values // => [8, 8, 8, 8]
 * ```
 */
export const fitSplatsMethod: Method<SplatState, SplatsMethodConfig, SplatPath> = defineMethod<
  SplatState,
  SplatsMethodConfig,
  SplatPath
>({
  name: 'fit-splats',
  config: dict({
    views: int(2, 100, { default: 8, label: 'training views' }),
    heldOut: int(0, 20, { default: 3, label: 'held-out views' }),
    resolution: int(4, 64, { default: 8, label: 'image side' }),
    objects: int(1, 4, { default: 1, label: 'objects in the scene' }),
    gaussians: int(1, 5000, { default: 30, label: 'Gaussians at the start' }),
    boxHalfWidth: real(0.1, 5, { default: 0.8, label: 'start box half-width' }),
    initialScale: real(0.01, 2, { log: true, default: 0.25, label: 'initial scale' }),
    initialOpacity: real(0.01, 0.99, { default: 0.3, label: 'initial opacity' }),
    shDegree: choice([0, 1], { default: 0, label: 'spherical-harmonic degree' }),
    optimiser: { ...optimisers.space, default: 'adam' },
    meansRate: real(1e-5, 1, { log: true, default: 5e-3, label: 'learning rate of the means' }),
    scalesRate: real(1e-5, 1, { log: true, default: 5e-3, label: 'learning rate of the scales' }),
    rotationsRate: real(1e-5, 1, { log: true, default: 1e-3, label: 'learning rate of the rotations' }),
    opacitiesRate: real(1e-5, 1, { log: true, default: 5e-2, label: 'learning rate of the opacities' }),
    coloursRate: real(1e-5, 1, { log: true, default: 2e-2, label: 'learning rate of the colours' }),
    l2Weight: real(0, 1, { default: 0, label: 'squared-error weight' }),
    densify: bool({ default: false, label: 'adaptive density control' }),
    densifyFrom: int(1, 100_000, { default: 20, label: 'first densification step' }),
    densifyUntil: int(1, 100_000, { default: 400, label: 'last densification step' }),
    densifyEvery: int(1, 1000, { default: 20, label: 'densification interval' }),
    gradientThreshold: real(1e-6, 1, { log: true, default: 5e-4, label: 'densification gradient threshold' }),
    maxGaussians: int(1, 20_000, { default: 2000, label: 'most Gaussians' }),
  }),
  make: (config, key) => {
    const p = splatsProblem(config, key)
    return fitSplatsAlgorithm(p.initial, p.data, p.options)
  },
  contract: (config, key) => splatContract(splatsProblem(config, key).data),
})
