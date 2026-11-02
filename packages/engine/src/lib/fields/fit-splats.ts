/**
 * Fitting 3-D Gaussians to posed images as an `Algorithm`, 3D Gaussian splatting's training loop on the library's pieces:
 * a step renders one training view (or a batch of its pixels) from the current Gaussians, takes the photometric loss
 * (L1, optionally with a squared-error term; the structural-similarity term of the paper is not offered because the
 * library's SSIM is not differentiable), differentiates through projection and compositing by autodiff, and updates
 * the parameters with one optimiser per parameter group (the optimiser is a slot, each group has its own learning rate:
 * means, scales, rotations, opacities, colours). Optionally the set is adapted as the paper does: the view-space
 * gradient of each Gaussian's centre is accumulated, and at the schedule's steps the ones with a large average
 * gradient are cloned (if small) or split in two (if large), and the nearly transparent or oversized ones are pruned.
 * The optimiser state is reset whenever the set changes size. The contract logs the loss, the peak signal-to-noise
 * ratio, the number of Gaussians and the densification events, with probes that render a view and draw the Gaussians
 * as ellipses.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import type { Json } from '../base/index.ts'
import type { PinholeCamera } from '../graphics/index.ts'
import { mae, mse } from '../losses/index.ts'
import {
  applyUpdates,
  type GradientTransformation,
  globalNorm,
  multiTransform,
  type OptimiserShape,
} from '../optim/index.ts'
import { foldIn, randint } from '../random/index.ts'
import { add, fromStorage, item, mul, norm, primalOf, type Tensor, toFlat, type Value, zeros } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  event,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import type { NerfData } from './fit-nerf.ts'
import { concatGaussians, type GaussianSet, gaussianCount, sampleGaussians, selectGaussians } from './gaussians.ts'
import { psnrFromMse } from './quality.ts'
import { projectGaussians } from './splat-project.ts'
import { rasterise, renderSplats } from './splat-render.ts'

/** The paths of {@link fitSplatsAlgorithm}. */
export type SplatPath =
  | 'splats/loss'
  | 'splats/mse'
  | 'splats/psnr'
  | 'splats/count'
  | 'splats/gradNorm'
  | 'splats/densify'
  | 'splats/render'
  | 'splats/ellipses'
  | 'splats/heldOutPsnr'

/** The state of a splat fit. The set can change size, so the statistics for densifying are per current Gaussian. */
export interface SplatState {
  readonly params: GaussianSet
  readonly optState: unknown
  readonly step: number
  readonly loss: number
  readonly mse: number
  readonly gradNorm: number
  /** The sum, over the steps since the last densification, of each Gaussian's view-space gradient norm `[n]`. */
  readonly gradSum: Tensor
  /** How many of those steps the Gaussian was in view `[n]`. */
  readonly seen: Tensor
}

/** The learning rate of each parameter group. */
export interface SplatRates {
  readonly means: number
  readonly scales: number
  readonly rotations: number
  readonly opacities: number
  readonly colours: number
}

/** The schedule and thresholds of adaptive density control. */
export interface DensifyOptions {
  /** The first step at which the set may change, the last, and the interval (defaults 20, 400, 20). */
  readonly from?: number
  readonly until?: number
  readonly every?: number
  /** The average view-space gradient (in pixels, of the loss) above which a Gaussian is cloned or split (default 5e-4). */
  readonly gradientThreshold?: number
  /** Gaussians whose largest scale is at most this fraction of `sceneExtent` are cloned, larger ones split (default 0.05). */
  readonly denseFraction?: number
  /** The radius of the scene in world units (default 1.5). */
  readonly sceneExtent?: number
  /** A split's children have the parent's scales divided by this (default 1.6). */
  readonly splitFactor?: number
  /** Gaussians with less opacity are pruned (default 0.005). */
  readonly minOpacity?: number
  /** Gaussians with a larger standard deviation than this fraction of `sceneExtent` are pruned (default 0.5). */
  readonly maxScaleFraction?: number
  /** Never grow beyond this many Gaussians (default 2000). */
  readonly maxGaussians?: number
  /** Every this many steps (within the window) cap the opacities at 0.01, so floaters must earn their keep again. */
  readonly opacityResetEvery?: number
}

/** Options of {@link fitSplatsAlgorithm}. */
export interface FitSplatsOptions {
  /** The optimiser (the slot), made once per parameter group with that group's learning rate. */
  readonly optimiser: OptimiserShape
  /** Learning rates per group (defaults 5e-3 for the means and scales, 1e-3 rotations, 5e-2 opacities, 2e-2 colours). */
  readonly learningRates?: Partial<SplatRates>
  /** Pixels rendered per step: all of the view's by default. */
  readonly batchPixels?: number
  /** The weights of the L1 and squared-error terms (default 1 and 0). */
  readonly l1Weight?: number
  readonly l2Weight?: number
  /** Adaptive density control; omit to keep the set the size it starts. */
  readonly densify?: DensifyOptions
}

/** A view rendered from Gaussians. */
export interface SplatRender {
  readonly kind: 'splatImage'
  readonly width: number
  readonly height: number
  /** Interleaved red, green, blue `[height * width * 3]`. */
  readonly rgb: readonly number[]
  readonly depth: readonly number[]
  readonly alpha: readonly number[]
  readonly psnr?: number
}

/** One Gaussian projected to the image as an ellipse of one standard deviation. */
export interface SplatEllipse {
  /** The Gaussian's index in the set. */
  readonly index: number
  /** Centre in pixels. */
  readonly x: number
  readonly y: number
  /** Semi-axes in pixels (longest first) and the angle of the longest from the `x` axis, towards `y`, in radians. */
  readonly major: number
  readonly minor: number
  readonly angle: number
  readonly opacity: number
  readonly rgb: readonly number[]
  readonly depth: number
}

/** The Gaussians of a view as ellipses, far to near (the order to draw them in). */
export interface SplatEllipses {
  readonly kind: 'splatEllipses'
  readonly width: number
  readonly height: number
  readonly ellipses: readonly SplatEllipse[]
}

const RATES: SplatRates = { means: 5e-3, scales: 5e-3, rotations: 1e-3, opacities: 5e-2, colours: 2e-2 }
const DENSIFY: Required<Omit<DensifyOptions, 'opacityResetEvery'>> = {
  from: 20,
  until: 400,
  every: 20,
  gradientThreshold: 5e-4,
  denseFraction: 0.05,
  sceneExtent: 1.5,
  splitFactor: 1.6,
  minOpacity: 0.005,
  maxScaleFraction: 0.5,
  maxGaussians: 2000,
}

/**
 * The optimiser of a splat fit: one transformation per parameter group (`means`, `logScales`, `rotations`,
 * `opacityLogits`, `colours`, `sh`), each made from the slot with the group's learning rate, joined by `multiTransform`.
 *
 * @param slot - The optimiser shape (for example `optimisers.make({ kind: 'adam' })`).
 * @param rates - The learning rates (defaults for the missing).
 * @returns The transformation.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { splatOptimiser } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * splatOptimiser((lr) => adam({ learningRate: lr }), { means: 0.01 }).name.startsWith('multiTransform') // => true
 * ```
 */
export function splatOptimiser(slot: OptimiserShape, rates: Partial<SplatRates> = {}): GradientTransformation {
  const r = { ...RATES, ...rates }
  return multiTransform(
    {
      means: slot(r.means),
      logScales: slot(r.scales),
      rotations: slot(r.rotations),
      opacityLogits: slot(r.opacities),
      colours: slot(r.colours),
      sh: slot(r.colours / 20),
    },
    (p: unknown) => Object.fromEntries(Object.keys(p as object).map((k) => [k, k])),
  ) as GradientTransformation
}

const view = (images: Tensor, v: number, size: number): Float64Array =>
  images.data.slice(v * size, (v + 1) * size) as Float64Array

/** Pixel-centre coordinates of an image, row by row. */
function pixelGrid(width: number, height: number): { xs: Float64Array; ys: Float64Array } {
  const xs = new Float64Array(width * height)
  const ys = new Float64Array(width * height)
  for (let j = 0; j < height; j++)
    for (let i = 0; i < width; i++) {
      xs[j * width + i] = i + 0.5
      ys[j * width + i] = j + 0.5
    }
  return { xs, ys }
}

/**
 * Render a view from Gaussians for a probe: colour, depth and opacity per pixel, and the peak signal-to-noise ratio
 * against the view's image when given.
 *
 * @param set - The Gaussians.
 * @param camera - The camera.
 * @param size - Width and height.
 * @param options - The background and the image to compare with.
 * @returns The render.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { randomGaussians, renderSplatView } from 'mdml-engine/fields'
 * import { pinholeCamera } from 'mdml-engine/graphics'
 * import { key } from 'mdml-engine/random'
 * renderSplatView(randomGaussians(key(0), 5), pinholeCamera({ eye: [0, 0, 4] }), [4, 3]).rgb.length // => 36
 * ```
 */
export function renderSplatView(
  set: GaussianSet,
  camera: PinholeCamera,
  size: readonly [number, number],
  options: { readonly background?: readonly [number, number, number]; readonly image?: Tensor } = {},
): SplatRender {
  const [width, height] = size
  const img = renderSplats(
    set,
    camera,
    width,
    height,
    options.background === undefined ? {} : { background: options.background },
  )
  const rgb = Array.from(toFlat(img.colour))
  let psnr: number | undefined
  if (options.image !== undefined) {
    const target = toFlat(options.image)
    let s = 0
    for (let i = 0; i < rgb.length; i++) s += ((rgb[i] as number) - (target[i] as number)) ** 2
    psnr = psnrFromMse(s / rgb.length)
  }
  return {
    kind: 'splatImage',
    width,
    height,
    rgb,
    depth: Array.from(toFlat(img.depth)),
    alpha: Array.from(toFlat(img.alpha)),
    ...(psnr === undefined ? {} : { psnr }),
  }
}

/**
 * The Gaussians of a view as ellipses (one standard deviation, from the projected covariance's eigenvectors), ordered far
 * to near so drawing them in order is the painter's algorithm.
 *
 * @param set - The Gaussians.
 * @param camera - The camera.
 * @param size - Width and height.
 * @returns The ellipses of the Gaussians in view.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussiansFromPoints, splatEllipses } from 'mdml-engine/fields'
 * import { pinholeCamera } from 'mdml-engine/graphics'
 * const e = splatEllipses(gaussiansFromPoints([[0, 0, 0]], undefined, { scale: 0.5 }), pinholeCamera({ eye: [0, 0, 4], fov: 90 }), [64, 64])
 * [e.ellipses.length, e.ellipses[0]?.x, e.ellipses[0]?.y] // => [1, 32, 32]
 * ```
 */
export function splatEllipses(set: GaussianSet, camera: PinholeCamera, size: readonly [number, number]): SplatEllipses {
  const [width, height] = size
  const p = projectGaussians(set, camera, width, height)
  const mean = toFlat(p.mean2d)
  const cov = toFlat(p.cov2d)
  const depth = toFlat(p.depth)
  const opacity = toFlat(p.opacity)
  const colour = toFlat(p.colour)
  const ellipses: SplatEllipse[] = Array.from(p.index, (index, i) => {
    const [a, b, c] = [cov[3 * i] as number, cov[3 * i + 1] as number, cov[3 * i + 2] as number]
    const mid = 0.5 * (a + c)
    const gap = Math.sqrt(Math.max(0, mid * mid - (a * c - b * b)))
    return {
      index,
      x: mean[2 * i] as number,
      y: mean[2 * i + 1] as number,
      major: Math.sqrt(mid + gap),
      minor: Math.sqrt(Math.max(0, mid - gap)),
      angle: 0.5 * Math.atan2(2 * b, a - c),
      opacity: opacity[i] as number,
      rgb: [colour[3 * i] as number, colour[3 * i + 1] as number, colour[3 * i + 2] as number],
      depth: depth[i] as number,
    }
  })
  ellipses.sort((u, v) => v.depth - u.depth)
  return { kind: 'splatEllipses', width, height, ellipses }
}

/**
 * The contract of a splat fit: loss, error, peak signal-to-noise ratio, Gaussian count and gradient norm per step, the
 * densification event (with its counts as context), and the derived probes `splats/render` (a view, arguments `{ view }`
 * or `{ heldOut }`), `splats/ellipses` (the Gaussians of a view, arguments `{ view }`) and `splats/heldOutPsnr`.
 *
 * @param data - The posed images.
 * @returns The contract.
 * @concept log-contract
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { nerfData, splatContract } from 'mdml-engine/fields'
 * Object.keys(splatContract(nerfData(generate(multiViewImages, { knobs: { views: 2, resolution: 4, heldOut: 0 } })))).length // => 9
 * ```
 */
export function splatContract(data: NerfData): Contract<SplatState, SplatPath> {
  const [, height = 0, width = 0] = data.images.shape
  const background: readonly [number, number, number] = data.whiteBackground ? [1, 1, 1] : [0, 0, 0]
  const size = width * height * 3
  const pick = (args: Json | undefined): { camera: PinholeCamera; image: Tensor } => {
    const a = (args ?? {}) as { view?: number; heldOut?: number }
    const image = (images: Tensor, v: number): Tensor =>
      fromStorage(view(images, v, size), [height, width, 3], 'float64')
    if (a.heldOut !== undefined && data.heldOut !== undefined) {
      const v = Math.max(0, Math.min(data.heldOut.cameras.length - 1, a.heldOut))
      return { camera: data.heldOut.cameras[v] as PinholeCamera, image: image(data.heldOut.images, v) }
    }
    const v = Math.max(0, Math.min(data.cameras.length - 1, a.view ?? 0))
    return { camera: data.cameras[v] as PinholeCamera, image: image(data.images, v) }
  }
  return defineContract({
    'splats/loss': scalar({ description: 'the photometric loss of the step’s view (or pixels)' }),
    'splats/mse': scalar({ description: 'the mean squared error of the step’s render against its target' }),
    'splats/psnr': scalar({ description: 'the peak signal-to-noise ratio of the step’s render, in decibels' }),
    'splats/count': scalar({ description: 'the number of Gaussians' }),
    'splats/gradNorm': scalar({ description: 'the global norm of the gradient' }),
    'splats/densify': event({
      description: 'the set was adapted: Gaussians cloned, split and pruned (counts in the context)',
    }),
    'splats/render': derived<SplatState>({
      description: 'a view rendered from the Gaussians: colour, depth and opacity per pixel (args: view or heldOut)',
      cost: 'dear',
      compute: (s, args) => {
        const { camera, image } = pick(args)
        return renderSplatView(s.params, camera, [width, height], { background, image })
      },
    }),
    'splats/ellipses': derived<SplatState>({
      description: 'the Gaussians of a view as ellipses of one standard deviation, far to near (args: view)',
      compute: (s, args) => splatEllipses(s.params, pick(args).camera, [width, height]),
    }),
    'splats/heldOutPsnr': derived<SplatState>({
      description:
        'the mean peak signal-to-noise ratio over the held-out views (the training views when there are none)',
      cost: 'dear',
      compute: (s) => {
        const set = data.heldOut ?? { images: data.images, cameras: data.cameras }
        const values = set.cameras.map(
          (camera, v) =>
            renderSplatView(s.params, camera, [width, height], {
              background,
              image: fromStorage(view(set.images, v, size), [height, width, 3], 'float64'),
            }).psnr ?? Number.NaN,
        )
        return values.reduce((a, b) => a + b, 0) / Math.max(1, values.length)
      },
    }),
  })
}

const logit = (p: number): number => Math.log(p / (1 - p))
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x))

/** What one densification did. */
interface Densified {
  readonly params: GaussianSet
  readonly cloned: number
  readonly split: number
  readonly pruned: number
}

/** Clone, split and prune a set from the accumulated view-space gradients. */
function densified(
  set: GaussianSet,
  gradSum: Tensor,
  seen: Tensor,
  o: Required<Omit<DensifyOptions, 'opacityResetEvery'>>,
  key: Parameters<typeof sampleGaussians>[0],
): Densified {
  const n = gaussianCount(set)
  const g = gradSum.data
  const c = seen.data
  const scale = toFlat(primalOf(set.logScales))
  const biggest = Array.from({ length: n }, (_, i) =>
    Math.exp(Math.max(scale[3 * i] as number, scale[3 * i + 1] as number, scale[3 * i + 2] as number)),
  )
  const hot: number[] = []
  for (let i = 0; i < n; i++)
    if ((c[i] as number) > 0 && (g[i] as number) / (c[i] as number) >= o.gradientThreshold) hot.push(i)
  hot.sort((a, b) => (g[b] as number) / (c[b] as number) - (g[a] as number) / (c[a] as number) || a - b)
  // The room left, given to the Gaussians with the largest gradient first (a clone adds one Gaussian, a split adds two
  // children and removes the parent: one more either way).
  const cloneSet: number[] = []
  const splitSet: number[] = []
  let room = o.maxGaussians - n
  for (const i of hot) {
    if (room < 1) break
    ;((biggest[i] as number) <= o.denseFraction * o.sceneExtent ? cloneSet : splitSet).push(i)
    room -= 1
  }
  const splitting = new Set(splitSet)
  const kept = Array.from({ length: n }, (_, i) => i).filter((i) => !splitting.has(i))
  let next = selectGaussians(set, kept)
  if (cloneSet.length > 0) next = concatGaussians(next, selectGaussians(set, cloneSet))
  if (splitSet.length > 0) {
    const parents = selectGaussians(set, splitSet)
    const shrink = Math.log(o.splitFactor)
    const child = (k: number): GaussianSet => ({
      ...parents,
      means: sampleGaussians(foldIn(key, k), parents),
      logScales: add(parents.logScales, -shrink),
    })
    next = concatGaussians(concatGaussians(next, child(0)), child(1))
  }
  // Prune the nearly transparent and the oversized.
  const m = gaussianCount(next)
  const logits = toFlat(primalOf(next.opacityLogits))
  const logScales = toFlat(primalOf(next.logScales))
  const survivors: number[] = []
  for (let i = 0; i < m; i++) {
    const largest = Math.exp(
      Math.max(logScales[3 * i] as number, logScales[3 * i + 1] as number, logScales[3 * i + 2] as number),
    )
    if (sigmoid(logits[i] as number) >= o.minOpacity && largest <= o.maxScaleFraction * o.sceneExtent) survivors.push(i)
  }
  const pruned = m - survivors.length
  return {
    params: survivors.length === m ? next : selectGaussians(next, survivors),
    cloned: cloneSet.length,
    split: splitSet.length,
    pruned,
  }
}

/**
 * Fit Gaussians to posed images as an {@link Algorithm}: each step renders a training view from the current set,
 * differentiates the photometric loss through projection and compositing, and applies one optimiser per parameter
 * group; with `densify`, the set is cloned, split and pruned on the schedule. A step draws its view and pixels from the
 * step's key, and splits their children from it, so pausing and resuming from a serialised state gives the run that
 * never stopped.
 *
 * @param initial - The Gaussians to start from (`randomGaussians`, or `gaussiansFromPoints` for a point cloud).
 * @param data - The posed images (the same `NerfData` a NeRF fits, so the two can be compared on the same views).
 * @param options - The optimiser slot, learning rates, pixels per step, loss weights and density control.
 * @returns The algorithm.
 * @concept gaussian-splatting
 * @reference Kerbl, Kopanas, Leimkuehler and Drettakis (2023), 3D Gaussian splatting for real-time radiance field rendering.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { fitSplatsAlgorithm, nerfData, randomGaussians } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = nerfData(generate(multiViewImages, { knobs: { views: 3, resolution: 6, heldOut: 0, objects: 1 } }))
 * const alg = fitSplatsAlgorithm(randomGaussians(key(0), 10), data, { optimiser: (lr) => adam({ learningRate: lr }) })
 * runAlgorithm(alg, { maxSteps: 3 }).state.step // => 3
 * ```
 */
export function fitSplatsAlgorithm(
  initial: GaussianSet,
  data: NerfData,
  options: FitSplatsOptions,
): Algorithm<SplatState, SplatPath> {
  const [views = 0, height = 0, width = 0] = data.images.shape
  const pixels = width * height
  const batch = Math.min(options.batchPixels ?? pixels, pixels)
  const background: readonly [number, number, number] = data.whiteBackground ? [1, 1, 1] : [0, 0, 0]
  const targets = primalOf(data.images).data
  const grid = pixelGrid(width, height)
  const optimiser = (): GradientTransformation => splatOptimiser(options.optimiser, options.learningRates)
  const transform = optimiser()
  const l1 = options.l1Weight ?? 1
  const l2 = options.l2Weight ?? 0
  const d = options.densify === undefined ? undefined : { ...DENSIFY, ...options.densify }
  const resetEvery = options.densify?.opacityResetEvery
  const fresh = (params: GaussianSet): { gradSum: Tensor; seen: Tensor } => ({
    gradSum: zeros([gaussianCount(params)]) as Tensor,
    seen: zeros([gaussianCount(params)]) as Tensor,
  })
  return {
    name: 'fit-splats',
    config: {
      gaussians: gaussianCount(initial),
      batchPixels: batch,
      views,
      resolution: [width, height],
      learningRates: { ...RATES, ...options.learningRates },
      l1Weight: l1,
      l2Weight: l2,
      densify: d === undefined ? null : ({ ...d, opacityResetEvery: resetEvery ?? null } as Json),
    } as Json,
    init: () => ({
      params: initial,
      optState: transform.init(initial),
      step: 0,
      loss: 0,
      mse: 0,
      gradNorm: 0,
      ...fresh(initial),
    }),
    step: (s, ctx) => {
      const v = (randint(foldIn(ctx.key, 0), [1], 0, views).data[0] as number) | 0
      const camera = data.cameras[v] as PinholeCamera
      let xs = grid.xs
      let ys = grid.ys
      let pick: Int32Array | undefined
      if (batch < pixels) {
        pick = randint(foldIn(ctx.key, 1), [batch], 0, pixels).data
        xs = Float64Array.from(pick, (k) => grid.xs[k] as number)
        ys = Float64Array.from(pick, (k) => grid.ys[k] as number)
      }
      const count = xs.length
      const target = new Float64Array(count * 3)
      for (let i = 0; i < count; i++) {
        const k = pick === undefined ? i : (pick[i] as number)
        for (let c = 0; c < 3; c++) target[3 * i + c] = targets[(v * pixels + k) * 3 + c] as number
      }
      const goal = fromStorage(target, [count, 3], 'float64')
      const n = gaussianCount(s.params)
      let visible: Int32Array = new Int32Array(0)
      const objective = (p: { params: GaussianSet; offsets: Value }): readonly [Value, Value] => {
        const projected = projectGaussians(p.params, camera, width, height, { screenOffsets: p.offsets })
        visible = projected.index
        const r = rasterise(projected, xs, ys, { background })
        const error = mse(r.colour, goal)
        const loss = add(mul(mae(r.colour, goal), l1), mul(error, l2))
        return [loss, error]
      }
      const { value, grad, aux } = valueAndGrad(objective, { hasAux: true })({
        params: s.params,
        offsets: zeros([n, 2]),
      })
      const g = (grad as { params: GaussianSet; offsets: Value }).params
      const screen = toFlat(primalOf(norm((grad as { offsets: Value }).offsets, 2, 1)))
      const r = transform.update(g, s.optState, s.params)
      const loss = item(primalOf(value))
      const error = item(primalOf(aux as Value))
      const gradNorm = item(globalNorm(g))
      let params = applyUpdates(s.params, r.updates) as GaussianSet
      let optState = r.state
      let gradSum = s.gradSum
      let seen = s.seen
      // This step's view-space gradient norms of the Gaussians in view.
      const gs = Float64Array.from(gradSum.data)
      const sn = Float64Array.from(seen.data)
      for (const i of visible) {
        gs[i] = (gs[i] as number) + (screen[i] as number)
        sn[i] = (sn[i] as number) + 1
      }
      gradSum = fromStorage(gs, [n], 'float64')
      seen = fromStorage(sn, [n], 'float64')
      const step = s.step + 1
      ctx.log('splats/loss', loss)
      ctx.log('splats/mse', error)
      ctx.log('splats/psnr', psnrFromMse(error))
      ctx.log('splats/gradNorm', gradNorm)
      if (d !== undefined && step >= d.from && step <= d.until) {
        if (step % d.every === 0) {
          const out = densified(params, gradSum, seen, d, foldIn(ctx.key, 2))
          params = out.params
          optState = transform.init(params)
          ;({ gradSum, seen } = fresh(params))
          ctx.emit('splats/densify', {
            step,
            cloned: out.cloned,
            split: out.split,
            pruned: out.pruned,
            count: gaussianCount(params),
          })
        }
        if (resetEvery !== undefined && step % resetEvery === 0) {
          const cap = logit(0.01)
          params = {
            ...params,
            opacityLogits: fromStorage(
              Float64Array.from(toFlat(primalOf(params.opacityLogits)), (x) => Math.min(x, cap)),
              [gaussianCount(params)],
              'float64',
            ),
          }
        }
      }
      ctx.log('splats/count', gaussianCount(params))
      return { params, optState, step, loss, mse: error, gradNorm, gradSum, seen }
    },
    done: (s) => (Number.isFinite(s.loss) ? undefined : 'diverged'),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SplatState,
  }
}

/** The outcome of {@link fitSplats}. */
export interface SplatFit {
  readonly params: GaussianSet
  readonly losses: readonly number[]
  readonly counts: readonly number[]
  readonly steps: number
  readonly status: string
}

/**
 * Fit Gaussians for a number of steps.
 *
 * @param initial - The starting Gaussians.
 * @param data - The posed images.
 * @param options - As {@link fitSplatsAlgorithm}, the step count and the run's key, tracer and policy.
 * @returns The fitted Gaussians, the per-step losses and counts, and the status.
 * @concept gaussian-splatting
 * @index function Fit 3-D Gaussians to posed images, with adaptive density control.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { multiViewImages } from 'mdml-engine/datasets'
 * import { fitSplats, nerfData, randomGaussians } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * const data = nerfData(generate(multiViewImages, { knobs: { views: 3, resolution: 6, heldOut: 0, objects: 1 } }))
 * fitSplats(randomGaussians(key(0), 10), data, { optimiser: (lr) => adam({ learningRate: lr }), steps: 2 }).losses.length // => 2
 * ```
 */
export function fitSplats(
  initial: GaussianSet,
  data: NerfData,
  options: FitSplatsOptions & { readonly steps: number } & Partial<Omit<RunOptions, 'maxSteps'>>,
): SplatFit {
  const { steps, optimiser, learningRates, batchPixels, l1Weight, l2Weight, densify, ...run } = options
  const algorithm = fitSplatsAlgorithm(initial, data, {
    optimiser,
    ...(learningRates === undefined ? {} : { learningRates }),
    ...(batchPixels === undefined ? {} : { batchPixels }),
    ...(l1Weight === undefined ? {} : { l1Weight }),
    ...(l2Weight === undefined ? {} : { l2Weight }),
    ...(densify === undefined ? {} : { densify }),
  })
  const losses: number[] = []
  const counts: number[] = []
  const tracked: Algorithm<SplatState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      losses.push(next.loss)
      counts.push(gaussianCount(next.params))
      return next
    },
  }
  const result = runAlgorithm(tracked, { ...run, maxSteps: steps })
  return { params: result.state.params, losses, counts, steps: result.steps, status: result.status }
}
