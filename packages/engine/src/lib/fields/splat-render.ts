/**
 * Rasterising projected Gaussians by front-to-back alpha compositing, differentiably. The Gaussians are sorted by depth
 * once; for each pixel, the opacity of Gaussian `i` is `alpha_i = min(0.99, o_i exp(-d^T Sigma'^-1 d / 2))` for the
 * offset `d` from its centre, and the pixel's colour is `sum_i T_i alpha_i c_i` with `T_i = prod_{j<i} (1 - alpha_j)`
 * (plus `T_n` times the background), the depth and opacity composited the same way. As in the reference
 * implementation, a Gaussian contributes only inside its 3-standard-deviation radius and when `alpha_i >= 1/255`, and
 * compositing stops before the Gaussian that would take the transmittance below 1e-4 (early termination). The sums are
 * tensor operations over a dense pixels-by-Gaussians table (the masks above are decided on the values), so the
 * gradient with respect to centres, covariances, opacities, colours and depths is the library's autodiff: there is no
 * hand-written backward pass. Pixels are processed in chunks, and any list of pixels can be asked for, so training can
 * render a batch of pixels and a full image is just all of them.
 */

import type { PinholeCamera } from '../graphics/index.ts'
import {
  add,
  concat,
  cumsum,
  div,
  exp,
  expandDims,
  fromStorage,
  log1p,
  matmul,
  minimum,
  mul,
  neg,
  primalOf,
  reshape,
  sliceAxis,
  squeeze,
  sub,
  sum,
  take,
  type Value,
  zeros,
} from '../tensor/index.ts'
import type { GaussianSet } from './gaussians.ts'
import { type Projected, projectGaussians } from './splat-project.ts'

/** Opacities below this are skipped (1/255), as in the reference rasteriser. */
export const MIN_ALPHA: number = 1 / 255
/** Opacities are capped here so a pixel never saturates exactly. */
export const MAX_ALPHA = 0.99
/** Compositing stops before the Gaussian that would take the transmittance below this. */
export const MIN_TRANSMITTANCE = 1e-4

/** What rasterising gives for a list of pixels. */
export interface Splatted {
  /** The colour of each pixel `[p, 3]` (with the background composited). */
  readonly colour: Value
  /** The expected depth `sum w_i z_i` of each pixel `[p]` (0 where nothing was hit). */
  readonly depth: Value
  /** The accumulated opacity `1 - T` of each pixel `[p]`. */
  readonly alpha: Value
}

/** Options of {@link rasterise}. */
export interface RasteriseSplatOptions {
  /** The colour behind everything (default black). */
  readonly background?: readonly [number, number, number]
  /** Pixels processed at a time (default 2048; the table is chunk by number of Gaussians). */
  readonly chunk?: number
}

/**
 * Composite projected Gaussians at a list of pixels (their centres, in pixel units).
 *
 * @param projected - The projected Gaussians.
 * @param xs - The pixels' `x` coordinates (centre of pixel `i` is `i + 0.5`).
 * @param ys - The pixels' `y` coordinates.
 * @param options - The background and the chunk size.
 * @returns Colour, depth and accumulated opacity per pixel.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { gaussiansFromPoints, projectGaussians, rasterise } from 'mdml-engine/fields'
 * import { pinholeCamera } from 'mdml-engine/graphics'
 * import { toArray } from 'mdml-engine/tensor'
 * // The pixel at the centre of a Gaussian of opacity 0.5 (logit 0) is half covered: alpha = 0.5.
 * const g = { ...gaussiansFromPoints([[0, 0, 0]], undefined, { scale: 0.5, opacity: 0.5 }) }
 * const p = projectGaussians(g, pinholeCamera({ eye: [0, 0, 4], fov: 90 }), 64, 64)
 * toArray(rasterise(p, [32], [32]).alpha) // => [0.5]
 * ```
 */
export function rasterise(
  projected: Projected,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  options: RasteriseSplatOptions = {},
): Splatted {
  const count = xs.length
  const background = options.background ?? [0, 0, 0]
  const m = projected.index.length
  const backdrop = (n: number): Splatted => ({
    colour: mul(
      fromStorage(new Float64Array(n * 3).fill(1), [n, 3], 'float64'),
      fromStorage(Float64Array.from(background), [1, 3], 'float64'),
    ),
    depth: zeros([n]),
    alpha: zeros([n]),
  })
  if (m === 0) return backdrop(count)
  // Sort the Gaussians front to back (on the depths' values).
  const order = fromStorage(Int32Array.from(argsortOf(primalOf(projected.depth).data)), [m], 'int32')
  const pick = (v: Value): Value => take(v, order, 0)
  const mean = pick(projected.mean2d)
  const cov = pick(projected.cov2d)
  const opacity = pick(projected.opacity)
  const colour = pick(projected.colour)
  const depth = pick(projected.depth)
  const radius = Float64Array.from(Array.from(order.data), (i) => projected.radius[i] as number)
  const col = (v: Value, k: number): Value => squeeze(sliceAxis(v, -1, k, k + 1), -1)
  const [mx, my] = [col(mean, 0), col(mean, 1)] as [Value, Value]
  const [ca, cb, cc] = [col(cov, 0), col(cov, 1), col(cov, 2)] as [Value, Value, Value]
  const det = sub(mul(ca, cc), mul(cb, cb))
  const size = options.chunk ?? 2048
  const colours: Value[] = []
  const depths: Value[] = []
  const alphas: Value[] = []
  for (let from = 0; from < count; from += size) {
    const to = Math.min(count, from + size)
    const p = to - from
    const px = fromStorage(
      Float64Array.from({ length: p }, (_, i) => xs[from + i] as number),
      [p, 1],
      'float64',
    )
    const py = fromStorage(
      Float64Array.from({ length: p }, (_, i) => ys[from + i] as number),
      [p, 1],
      'float64',
    )
    const dx = sub(px, expandDims(mx, 0))
    const dy = sub(py, expandDims(my, 0))
    // power = -1/2 d^T Sigma'^-1 d, with Sigma'^-1 = [[c, -b], [-b, a]] / det.
    const quad = add(
      sub(mul(expandDims(cc, 0), mul(dx, dx)), mul(mul(expandDims(cb, 0), mul(dx, dy)), 2)),
      mul(expandDims(ca, 0), mul(dy, dy)),
    )
    const power = mul(div(quad, expandDims(det, 0)), -0.5)
    const raw = mul(expandDims(opacity, 0), exp(power))
    const alphaFull = minimum(raw, MAX_ALPHA)
    // The masks, decided on the values: inside the radius, and opaque enough to matter.
    const dxv = primalOf(dx).data
    const dyv = primalOf(dy).data
    const av = primalOf(alphaFull).data
    const live = new Float64Array(p * m)
    for (let i = 0; i < p; i++)
      for (let j = 0; j < m; j++) {
        const k = i * m + j
        const r = radius[j] as number
        live[k] = (dxv[k] as number) ** 2 + (dyv[k] as number) ** 2 <= r * r && (av[k] as number) >= MIN_ALPHA ? 1 : 0
      }
    const alpha = mul(alphaFull, fromStorage(live, [p, m], 'float64'))
    // T_i, the transmittance in front of Gaussian i: exp of the cumulative log(1 - alpha) of those before it.
    const logKeep = log1p(neg(alpha))
    const before = m === 1 ? zeros([p, 1]) : concat([zeros([p, 1]), sliceAxis(logKeep, 1, 0, m - 1)], 1)
    const transmittance = exp(cumsum(before, 1))
    // Early termination: stop before the Gaussian that would take the transmittance under the floor.
    const tv = primalOf(transmittance).data
    const alive = new Float64Array(p * m)
    for (let i = 0; i < p; i++)
      for (let j = 0; j < m; j++) {
        const k = i * m + j
        alive[k] = (tv[k] as number) * (1 - (av[k] as number) * (live[k] as number)) >= MIN_TRANSMITTANCE ? 1 : 0
      }
    const weights = mul(mul(alpha, transmittance), fromStorage(alive, [p, m], 'float64'))
    const total = sum(weights, 1)
    const bg = fromStorage(Float64Array.from(background), [1, 3], 'float64')
    colours.push(add(matmul(weights, colour), mul(expandDims(sub(1, total), -1), bg)))
    depths.push(squeeze(matmul(weights, expandDims(depth, -1)), -1))
    alphas.push(total)
  }
  const join = (parts: Value[], axis: number): Value => (parts.length === 1 ? (parts[0] as Value) : concat(parts, axis))
  return { colour: join(colours, 0), depth: join(depths, 0), alpha: join(alphas, 0) }
}

const argsortOf = (values: ArrayLike<number>): number[] =>
  Array.from({ length: values.length }, (_, i) => i).sort(
    (a, b) => (values[a] as number) - (values[b] as number) || a - b,
  )

/** What {@link renderSplats} gives: the image planes with their shapes `[height, width, ...]`. */
export interface SplatImage {
  readonly colour: Value
  readonly depth: Value
  readonly alpha: Value
}

/**
 * Render a view of a set of Gaussians: project, cull, sort, composite at every pixel centre.
 *
 * @param set - The Gaussians.
 * @param camera - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param options - The background and the chunk size.
 * @returns The colour `[height, width, 3]`, the depth and the opacity `[height, width]`.
 * @concept gaussian-splatting
 * @example
 * ```ts
 * import { randomGaussians, renderSplats } from 'mdml-engine/fields'
 * import { pinholeCamera } from 'mdml-engine/graphics'
 * import { key } from 'mdml-engine/random'
 * renderSplats(randomGaussians(key(0), 20), pinholeCamera({ eye: [0, 0, 4] }), 8, 6).colour.shape // => [6, 8, 3]
 * ```
 */
export function renderSplats(
  set: GaussianSet,
  camera: PinholeCamera,
  width: number,
  height: number,
  options: RasteriseSplatOptions = {},
): SplatImage {
  const xs = new Float64Array(width * height)
  const ys = new Float64Array(width * height)
  for (let j = 0; j < height; j++)
    for (let i = 0; i < width; i++) {
      xs[j * width + i] = i + 0.5
      ys[j * width + i] = j + 0.5
    }
  const r = rasterise(projectGaussians(set, camera, width, height), xs, ys, options)
  return {
    colour: reshape(r.colour, [height, width, 3]),
    depth: reshape(r.depth, [height, width]),
    alpha: reshape(r.alpha, [height, width]),
  }
}
