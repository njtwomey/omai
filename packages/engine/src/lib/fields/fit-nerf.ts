/**
 * Fitting a NeRF to posed images as an `Algorithm`: each step draws a batch of pixels from all the training views, casts
 * their rays through the model (stratified samples, then hierarchical ones with a fine network), composites them by
 * volume rendering and takes the squared error against the pixels' colours, coarse plus fine as in the paper. The
 * optimiser is a slot. The contract logs the loss and the peak signal-to-noise ratio of the batch, and holds three
 * probes a player can draw: a view rendered (colour, depth and opacity) from the training or held-out cameras, the
 * weights along one ray, and the peak signal-to-noise ratio over the held-out views.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import type { Json } from '../base/index.ts'
import type { Generated, ImageData } from '../data/index.ts'
import { cameraRays, type PinholeCamera } from '../graphics/index.ts'
import { mse } from '../losses/index.ts'
import { applyUpdates, type GradientTransformation, globalNorm } from '../optim/index.ts'
import { foldIn, randint } from '../random/index.ts'
import { add, fromStorage, item, primalOf, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { type NerfModel, type NerfParams, nerfParams, renderRays } from './nerf.ts'
import { psnrFromMse } from './quality.ts'

/** Posed images to fit a radiance field to. */
export interface NerfData {
  /** The images `[views, height, width, 3]`, values in [0, 1]. */
  readonly images: Tensor
  /** One camera per view. */
  readonly cameras: readonly PinholeCamera[]
  /** The interval along each ray that holds the scene. */
  readonly near: number
  readonly far: number
  /** Whether the images are on a white background (rays that hit nothing render white). */
  readonly whiteBackground: boolean
  /** Views the fit never sees, for judging novel-view quality. */
  readonly heldOut?: { readonly images: Tensor; readonly cameras: readonly PinholeCamera[] }
}

/**
 * The fitting data of a generated `multiViewImages` dataset: its training images and cameras, the sampling interval,
 * the background and the held-out views from its truth.
 *
 * @param generated - The dataset and its truth.
 * @returns The data.
 * @concept neural-radiance-fields
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { nerfData } from 'mdml-engine/fields'
 * const d = nerfData(generate(multiViewImages, { knobs: { views: 3, resolution: 4, heldOut: 1 } }))
 * [d.images.shape, d.heldOut?.cameras.length] // => [[3, 4, 4, 3], 1]
 * ```
 */
export function nerfData(generated: Generated<ImageData>): NerfData {
  const extra = generated.truth.extra ?? {}
  const held = extra['heldOut'] as { cameras: PinholeCamera[]; images: Tensor } | undefined
  const background = (extra['background'] as readonly number[] | undefined) ?? [1, 1, 1]
  return {
    images: generated.data.images,
    cameras: extra['cameras'] as PinholeCamera[],
    near: extra['near'] as number,
    far: extra['far'] as number,
    whiteBackground: background.every((v) => v === 1),
    ...(held === undefined || held.cameras.length === 0 ? {} : { heldOut: held }),
  }
}

/** The paths of {@link fitNerfAlgorithm}. */
export type NerfPath =
  | 'nerf/loss'
  | 'nerf/mse'
  | 'nerf/psnr'
  | 'nerf/gradNorm'
  | 'nerf/render'
  | 'nerf/rayWeights'
  | 'nerf/heldOutPsnr'

/** The state of a NeRF fit. */
export interface NerfState {
  readonly params: NerfParams
  readonly optState: unknown
  readonly step: number
  /** The last batch's loss (coarse plus fine), the mean squared error of the model's answer and the gradient norm. */
  readonly loss: number
  readonly mse: number
  readonly gradNorm: number
}

/** Options of {@link fitNerfAlgorithm}. */
export interface FitNerfOptions {
  /** The optimiser (the slot). */
  readonly optimizer: GradientTransformation
  /** Rays per step (default 128). */
  readonly batchRays?: number
  /** Coarse samples per ray (default 24) and fine ones (default as the coarse; used when the model has a fine network). */
  readonly samples?: number
  readonly fineSamples?: number
}

/** A view rendered by a NeRF: row-major, row 0 at the top. */
export interface NerfImage {
  readonly kind: 'nerfImage'
  readonly width: number
  readonly height: number
  /** Interleaved red, green, blue `[height * width * 3]`. */
  readonly rgb: readonly number[]
  /** The expected depth and accumulated opacity per pixel. */
  readonly depth: readonly number[]
  readonly alpha: readonly number[]
  /** The peak signal-to-noise ratio against the view's image, when it has one. */
  readonly psnr?: number
}

/** The weights a NeRF gives the samples along one ray. */
export interface NerfRayWeights {
  readonly kind: 'nerfRay'
  /** The distances of the samples (the coarse ones, then with the fine ones merged in, when there is a fine network). */
  readonly t: readonly number[]
  /** The weights `w_i` of the final pass, and the densities of the final pass. */
  readonly weights: readonly number[]
  readonly density: readonly number[]
  /** The coarse pass's distances and weights. */
  readonly coarse?: { readonly t: readonly number[]; readonly weights: readonly number[] }
  readonly alpha: number
}

const flatRays = (
  cameras: readonly PinholeCamera[],
  width: number,
  height: number,
): { origins: Float64Array; directions: Float64Array } => {
  const per = width * height * 3
  const origins = new Float64Array(cameras.length * per)
  const directions = new Float64Array(cameras.length * per)
  cameras.forEach((c, v) => {
    const r = cameraRays(c, width, height)
    origins.set(r.origins, v * per)
    directions.set(r.directions, v * per)
  })
  return { origins, directions }
}

/**
 * Render a view from a camera with a NeRF (one ray per pixel centre, no randomness) and, when the view's image is
 * given, the peak signal-to-noise ratio against it.
 *
 * @param model - The model.
 * @param params - Its parameters.
 * @param camera - The camera.
 * @param size - Width and height in pixels.
 * @param options - The sampling interval and counts, the background and the view's image.
 * @returns The rendered image.
 * @concept neural-radiance-fields
 * @example
 * ```ts
 * import { nerfModel, nerfParams, renderView } from 'mdml-engine/fields'
 * import { pinholeCamera } from 'mdml-engine/graphics'
 * import { key } from 'mdml-engine/random'
 * const model = nerfModel({ width: 8, depth: 1, positionEncoding: { kind: 'frequency', octaves: 2 }, key: key(0) })
 * renderView(model, nerfParams(model), pinholeCamera({ eye: [0, 0, 4] }), [4, 3], { near: 2, far: 6, samples: 6 }).rgb.length // => 36
 * ```
 */
export function renderView(
  model: NerfModel,
  params: NerfParams,
  camera: PinholeCamera,
  size: readonly [number, number],
  options: {
    readonly near: number
    readonly far: number
    readonly samples?: number
    readonly fineSamples?: number
    readonly whiteBackground?: boolean
    readonly image?: Tensor
  },
): NerfImage {
  const [width, height] = size
  const rays = cameraRays(camera, width, height)
  const out = renderRays(model, params, rays, {
    near: options.near,
    far: options.far,
    ...(options.samples === undefined ? {} : { samples: options.samples }),
    ...(options.fineSamples === undefined ? {} : { fineSamples: options.fineSamples }),
    ...(options.whiteBackground === undefined ? {} : { whiteBackground: options.whiteBackground }),
  })
  const answer = out.fine ?? out.coarse
  const rgb = Array.from(toFlat(answer.colour))
  let psnr: number | undefined
  if (options.image !== undefined) {
    const target = toFlat(options.image)
    let s = 0
    for (let i = 0; i < rgb.length; i++) s += ((rgb[i] as number) - (target[i] as number)) ** 2
    psnr = psnrFromMse(s / rgb.length)
  }
  return {
    kind: 'nerfImage',
    width,
    height,
    rgb,
    depth: Array.from(toFlat(answer.depth)),
    alpha: Array.from(toFlat(answer.alpha)),
    ...(psnr === undefined ? {} : { psnr }),
  }
}

/**
 * The contract of a NeRF fit: the batch's loss, error, peak signal-to-noise ratio and gradient norm per step, and the
 * derived probes `nerf/render` (a view as a `NerfImage`; arguments `{ view }` for a training view, `{ heldOut }` for a
 * held-out one), `nerf/rayWeights` (the weights along one ray, arguments `{ view, x, y }`) and `nerf/heldOutPsnr`.
 *
 * @param model - The model.
 * @param data - The posed images.
 * @param options - The fit's options.
 * @returns The contract.
 * @concept log-contract
 * @example
 * ```ts
 * import { fitNerfAlgorithm, nerfContract, nerfModel } from 'mdml-engine/fields'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { generate } from 'mdml-engine/data'
 * import { nerfData } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * const data = nerfData(generate(multiViewImages, { knobs: { views: 2, resolution: 4, heldOut: 0 } }))
 * Object.keys(nerfContract(nerfModel({ width: 4, depth: 1 }), data, { optimizer: adam({ learningRate: 0.01 }) })).length // => 7
 * ```
 */
export function nerfContract(model: NerfModel, data: NerfData, options: FitNerfOptions): Contract<NerfState, NerfPath> {
  const [, height = 0, width = 0] = data.images.shape
  const sampling = {
    near: data.near,
    far: data.far,
    samples: options.samples ?? 24,
    fineSamples: options.fineSamples ?? options.samples ?? 24,
    whiteBackground: data.whiteBackground,
  }
  const viewImage = (images: Tensor, v: number): Tensor =>
    fromStorage(
      images.data.slice(v * width * height * 3, (v + 1) * width * height * 3) as Float64Array,
      [height, width, 3],
      'float64',
    )
  const pick = (args: Json | undefined): { camera: PinholeCamera; image: Tensor } => {
    const a = (args ?? {}) as { view?: number; heldOut?: number }
    if (a.heldOut !== undefined && data.heldOut !== undefined) {
      const v = Math.max(0, Math.min(data.heldOut.cameras.length - 1, a.heldOut))
      return { camera: data.heldOut.cameras[v] as PinholeCamera, image: viewImage(data.heldOut.images, v) }
    }
    const v = Math.max(0, Math.min(data.cameras.length - 1, a.view ?? 0))
    return { camera: data.cameras[v] as PinholeCamera, image: viewImage(data.images, v) }
  }
  return defineContract({
    'nerf/loss': scalar({
      description: 'the batch loss: squared error of the coarse render plus that of the fine one',
    }),
    'nerf/mse': scalar({
      description: 'the batch’s mean squared error of the model’s answer (the fine render when there is one)',
    }),
    'nerf/psnr': scalar({ description: 'the peak signal-to-noise ratio of the batch, in decibels' }),
    'nerf/gradNorm': scalar({ description: 'the global norm of the gradient' }),
    'nerf/render': derived<NerfState>({
      description: 'a view rendered by the model: colour, depth and opacity per pixel (args: view or heldOut)',
      cost: 'dear',
      compute: (s, args) => {
        const { camera, image } = pick(args)
        return renderView(model, s.params, camera, [width, height], { ...sampling, image })
      },
    }),
    'nerf/rayWeights': derived<NerfState>({
      description: 'the weights, densities and distances along one ray (args: view, x, y in pixels)',
      cost: 'dear',
      compute: (s, args) => {
        const a = (args ?? {}) as { view?: number; x?: number; y?: number }
        const camera = data.cameras[Math.max(0, Math.min(data.cameras.length - 1, a.view ?? 0))] as PinholeCamera
        const rays = cameraRays(camera, width, height)
        const px = Math.max(0, Math.min(width - 1, a.x ?? Math.floor(width / 2)))
        const py = Math.max(0, Math.min(height - 1, a.y ?? Math.floor(height / 2)))
        const k = py * width + px
        const one = {
          origins: rays.origins.slice(3 * k, 3 * k + 3),
          directions: rays.directions.slice(3 * k, 3 * k + 3),
        }
        const out = renderRays(model, s.params, one, sampling)
        const last = out.fine ?? out.coarse
        return {
          kind: 'nerfRay',
          t: Array.from(toFlat(last.t)),
          weights: Array.from(toFlat(last.weights)),
          density: Array.from(toFlat(last.density)),
          ...(out.fine === undefined
            ? {}
            : { coarse: { t: Array.from(toFlat(out.coarse.t)), weights: Array.from(toFlat(out.coarse.weights)) } }),
          alpha: item(last.alpha),
        } satisfies NerfRayWeights
      },
    }),
    'nerf/heldOutPsnr': derived<NerfState>({
      description:
        'the mean peak signal-to-noise ratio over the held-out views (the training views when there are none)',
      cost: 'dear',
      compute: (s) => {
        const set = data.heldOut ?? { images: data.images, cameras: data.cameras }
        const values = set.cameras.map(
          (camera, v) =>
            renderView(model, s.params, camera, [width, height], { ...sampling, image: viewImage(set.images, v) })
              .psnr ?? Number.NaN,
        )
        return values.reduce((a, b) => a + b, 0) / Math.max(1, values.length)
      },
    }),
  })
}

/**
 * Fit a NeRF to posed images as an {@link Algorithm}: a step draws `batchRays` pixels from all the views (from the
 * step's key), renders their rays with stratified samples drawn from the same key, and differentiates the squared error
 * of the coarse render plus that of the fine render. Pausing and resuming from a serialised state gives the run that
 * never stopped.
 *
 * @param model - The model (its parameters are the starting point).
 * @param data - The posed images.
 * @param options - The optimiser slot, rays per step and samples per ray.
 * @returns The algorithm.
 * @concept neural-radiance-fields
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { fitNerfAlgorithm, nerfData, nerfModel } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = nerfData(generate(multiViewImages, { knobs: { views: 3, resolution: 4, heldOut: 0, objects: 1 } }))
 * const model = nerfModel({ width: 8, depth: 1, positionEncoding: { kind: 'frequency', octaves: 3 } })
 * const alg = fitNerfAlgorithm(model, data, { optimizer: adam({ learningRate: 0.01 }), batchRays: 16, samples: 6 })
 * runAlgorithm(alg, { maxSteps: 3 }).state.step // => 3
 * ```
 */
export function fitNerfAlgorithm(
  model: NerfModel,
  data: NerfData,
  options: FitNerfOptions,
): Algorithm<NerfState, NerfPath> {
  const [views = 0, height = 0, width = 0] = data.images.shape
  const total = views * width * height
  const batch = Math.min(options.batchRays ?? 128, total)
  const samples = options.samples ?? 24
  const rays = flatRays(data.cameras, width, height)
  const targets = primalOf(data.images).data
  const optimizer = options.optimizer
  return {
    name: 'fit-nerf',
    config: {
      optimizer: optimizer.name,
      batchRays: batch,
      samples,
      fineSamples: model.fine === undefined ? null : (options.fineSamples ?? samples),
      views,
      resolution: [width, height],
    } as Json,
    init: () => ({
      params: nerfParams(model),
      optState: optimizer.init(nerfParams(model)),
      step: 0,
      loss: 0,
      mse: 0,
      gradNorm: 0,
    }),
    step: (s, ctx) => {
      const ids = randint(foldIn(ctx.key, 0), [batch], 0, total).data
      const origins = new Float64Array(3 * batch)
      const directions = new Float64Array(3 * batch)
      const target = new Float64Array(3 * batch)
      for (let i = 0; i < batch; i++) {
        const k = ids[i] as number
        for (let c = 0; c < 3; c++) {
          origins[3 * i + c] = rays.origins[3 * k + c] as number
          directions[3 * i + c] = rays.directions[3 * k + c] as number
          target[3 * i + c] = targets[3 * k + c] as number
        }
      }
      const goal = fromStorage(target, [batch, 3], 'float64')
      const objective = (p: NerfParams): readonly [Value, Value] => {
        const out = renderRays(
          model,
          p,
          { origins, directions },
          {
            near: data.near,
            far: data.far,
            samples,
            ...(options.fineSamples === undefined ? {} : { fineSamples: options.fineSamples }),
            key: foldIn(ctx.key, 1),
            whiteBackground: data.whiteBackground,
          },
        )
        const coarse = mse(out.coarse.colour, goal)
        if (out.fine === undefined) return [coarse, coarse]
        const fine = mse(out.fine.colour, goal)
        return [add(coarse, fine), fine]
      }
      const { value, grad, aux } = valueAndGrad(objective, { hasAux: true })(s.params)
      const r = optimizer.update(grad, s.optState, s.params)
      const answer = item(primalOf(aux as Value))
      const gradNorm = item(globalNorm(grad))
      const loss = item(primalOf(value))
      ctx.log('nerf/loss', loss)
      ctx.log('nerf/mse', answer)
      ctx.log('nerf/psnr', psnrFromMse(answer))
      ctx.log('nerf/gradNorm', gradNorm)
      return {
        params: applyUpdates(s.params, r.updates) as NerfParams,
        optState: r.state,
        step: s.step + 1,
        loss,
        mse: answer,
        gradNorm,
      }
    },
    done: (s) => (Number.isFinite(s.loss) ? undefined : 'diverged'),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NerfState,
  }
}

/** The outcome of {@link fitNerf}. */
export interface NerfFit {
  readonly params: NerfParams
  readonly losses: readonly number[]
  readonly steps: number
  readonly status: string
}

/**
 * Fit a NeRF for a number of steps and load the trained parameters into the model's networks.
 *
 * @param model - The model (updated in place).
 * @param data - The posed images.
 * @param options - As {@link fitNerfAlgorithm}, the step count and the run's key, tracer and policy.
 * @returns The trained parameters, per-step losses and the status.
 * @concept neural-radiance-fields
 * @index function Fit a NeRF to posed images for a number of steps.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { fitNerf, nerfData, nerfModel } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * const data = nerfData(generate(multiViewImages, { knobs: { views: 3, resolution: 4, heldOut: 0, objects: 1 } }))
 * const model = nerfModel({ width: 8, depth: 1, positionEncoding: { kind: 'frequency', octaves: 3 } })
 * fitNerf(model, data, { optimizer: adam({ learningRate: 0.01 }), batchRays: 16, samples: 6, steps: 2 }).losses.length // => 2
 * ```
 */
export function fitNerf(
  model: NerfModel,
  data: NerfData,
  options: FitNerfOptions & { readonly steps: number } & Partial<Omit<RunOptions, 'maxSteps'>>,
): NerfFit {
  const { steps, optimizer, batchRays, samples, fineSamples, ...run } = options
  const algorithm = fitNerfAlgorithm(model, data, {
    optimizer,
    ...(batchRays === undefined ? {} : { batchRays }),
    ...(samples === undefined ? {} : { samples }),
    ...(fineSamples === undefined ? {} : { fineSamples }),
  })
  const losses: number[] = []
  const tracked: Algorithm<NerfState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      losses.push(next.loss)
      return next
    },
  }
  const result = runAlgorithm(tracked, { ...run, maxSteps: steps })
  model.coarse.setParams(result.state.params.coarse)
  if (model.fine !== undefined && result.state.params.fine !== undefined) model.fine.setParams(result.state.params.fine)
  return { params: result.state.params, losses, steps: result.steps, status: result.status }
}
