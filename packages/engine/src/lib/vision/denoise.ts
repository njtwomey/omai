/**
 * Image denoising and inpainting: non-local means (Buades et al. 2005), total-variation denoising by Chambolle's
 * projection algorithm (as scikit-image's denoise_tv_chambolle), Perona-Malik anisotropic diffusion (a traced
 * algorithm and a function), the guided filter (He et al. 2010), and inpainting of masked pixels by harmonic diffusion
 * (a traced algorithm). The adaptive Wiener filter is `signal`'s `wiener`.
 */

// TODO: Telea's fast-marching inpainting (cv2.INPAINT_TELEA) and the fast (integral-image) mode of non-local means.

import { type Json, MdmlError } from '../base/index.ts'
import { image } from '../signal/index.ts'
import { primalOf, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { greyOf, imageTensor, type Raster, rasterOf, rasterTensor } from './_image.ts'

/** The channel planes of a raster. */
const planesOf = (r: Raster): Float64Array[] =>
  Array.from({ length: r.c }, (_, k) => Float64Array.from({ length: r.h * r.w }, (_, i) => r.data[i * r.c + k] ?? 0))

const fromPlanes = (r: Raster, planes: readonly Float64Array[]): Tensor => {
  const out = new Float64Array(r.h * r.w * r.c)
  planes.forEach((p, k) => {
    for (let i = 0; i < r.h * r.w; i++) out[i * r.c + k] = p[i] ?? 0
  })
  return rasterTensor({ ...r, data: out })
}

/** Options of {@link nonLocalMeans}. */
export interface NonLocalMeansOptions {
  /** The patch side (default 5). */
  readonly patchSize?: number
  /** The search radius (default 6). */
  readonly searchRadius?: number
  /** The filtering strength h (default 0.1). */
  readonly h?: number
  /** The noise standard deviation, subtracted from patch distances (default 0). */
  readonly sigma?: number
}

/**
 * Non-local means denoising (Buades, Coll and Morel 2005): each pixel becomes the weighted mean of the pixels in its
 * search window, a pixel's weight exp(-max(d^2 - 2 sigma^2, 0) / h^2) falling with the mean squared difference d^2
 * between the patches around the two (the image mirrored at its borders); colour images compare whole colour patches.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - Patch size, search radius, h and the noise sigma.
 * @returns The denoised image.
 * @concept non-local-means
 * @example
 * ```ts
 * import { nonLocalMeans } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(nonLocalMeans([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], { patchSize: 3, searchRadius: 1 })) // => [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
 * ```
 */
export function nonLocalMeans(x: TensorLike, options: NonLocalMeansOptions = {}): Tensor {
  const r = rasterOf(x, 'nonLocalMeans')
  const { h, w, c } = r
  const half = Math.floor((options.patchSize ?? 5) / 2)
  const d = options.searchRadius ?? 6
  const hh = (options.h ?? 0.1) ** 2
  const s2 = 2 * (options.sigma ?? 0) ** 2
  const mirror = (i: number, n: number): number => {
    let k = i
    while (k < 0 || k >= n) k = k < 0 ? -k : 2 * (n - 1) - k
    return n === 1 ? 0 : k
  }
  const px = (y: number, xx: number, k: number): number => r.data[(mirror(y, h) * w + mirror(xx, w)) * c + k] ?? 0
  const npatch = (2 * half + 1) ** 2 * c
  const out = new Float64Array(r.data.length)
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++) {
      const acc = new Float64Array(c)
      let total = 0
      for (let dy = -d; dy <= d; dy++)
        for (let dx = -d; dx <= d; dx++) {
          const y2 = y + dy
          const x2 = xx + dx
          if (y2 < 0 || y2 >= h || x2 < 0 || x2 >= w) continue
          let dist = 0
          for (let py = -half; py <= half; py++)
            for (let pxx = -half; pxx <= half; pxx++)
              for (let k = 0; k < c; k++) dist += (px(y + py, xx + pxx, k) - px(y2 + py, x2 + pxx, k)) ** 2
          const wgt = Math.exp(-Math.max(dist / npatch - s2, 0) / hh)
          total += wgt
          for (let k = 0; k < c; k++) acc[k] = (acc[k] ?? 0) + wgt * (r.data[(y2 * w + x2) * c + k] ?? 0)
        }
      for (let k = 0; k < c; k++) out[(y * w + xx) * c + k] = (acc[k] ?? 0) / total
    }
  return rasterTensor({ ...r, data: out })
}

/** Chambolle's projection algorithm on one plane (scikit-image's _denoise_tv_chambolle_nd for 2-D). */
function tvChambolle(
  img: Float64Array,
  h: number,
  w: number,
  weight: number,
  eps: number,
  maxIter: number,
): Float64Array {
  const n = h * w
  const p0 = new Float64Array(n)
  const p1 = new Float64Array(n)
  const g0 = new Float64Array(n)
  const g1 = new Float64Array(n)
  let d = new Float64Array(n)
  let out = img
  let eInit = 0
  let ePrev = 0
  const tau = 1 / 4
  for (let i = 0; i < maxIter; i++) {
    if (i > 0) {
      d = Float64Array.from({ length: n }, (_, k) => -((p0[k] ?? 0) + (p1[k] ?? 0)))
      for (let y = 1; y < h; y++)
        for (let x = 0; x < w; x++) d[y * w + x] = (d[y * w + x] ?? 0) + (p0[(y - 1) * w + x] ?? 0)
      for (let y = 0; y < h; y++)
        for (let x = 1; x < w; x++) d[y * w + x] = (d[y * w + x] ?? 0) + (p1[y * w + x - 1] ?? 0)
      out = Float64Array.from(img, (v, k) => v + (d[k] ?? 0))
    } else out = img
    let E = 0
    for (const v of d) E += v * v
    g0.fill(0)
    g1.fill(0)
    for (let y = 0; y + 1 < h; y++)
      for (let x = 0; x < w; x++) g0[y * w + x] = (out[(y + 1) * w + x] ?? 0) - (out[y * w + x] ?? 0)
    for (let y = 0; y < h; y++)
      for (let x = 0; x + 1 < w; x++) g1[y * w + x] = (out[y * w + x + 1] ?? 0) - (out[y * w + x] ?? 0)
    let normSum = 0
    for (let k = 0; k < n; k++) {
      const norm = Math.sqrt((g0[k] ?? 0) ** 2 + (g1[k] ?? 0) ** 2)
      normSum += norm
      const scale = 1 + (norm * tau) / weight
      p0[k] = ((p0[k] ?? 0) - tau * (g0[k] ?? 0)) / scale
      p1[k] = ((p1[k] ?? 0) - tau * (g1[k] ?? 0)) / scale
    }
    E = (E + weight * normSum) / n
    if (i === 0) {
      eInit = E
      ePrev = E
    } else if (Math.abs(ePrev - E) < eps * eInit) break
    else ePrev = E
  }
  return out
}

/**
 * Total-variation denoising by Chambolle's projection algorithm (Chambolle 2004; skimage.restoration's
 * denoise_tv_chambolle): the minimiser of |u - x|^2 / 2 + weight TV(u) through the dual field p, iterated until the
 * relative change of the energy falls below eps; colour images are denoised channel by channel.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param options - weight (default 0.1), eps (default 2e-4) and maxIter (default 200).
 * @returns The denoised image.
 * @concept total-variation
 * @example
 * ```ts
 * import { denoiseTvChambolle } from 'mdml-engine/vision'
 * denoiseTvChambolle([[0, 1, 0], [1, 0, 1]], { weight: 0.2 }).shape // => [2, 3]
 * ```
 */
export function denoiseTvChambolle(
  x: TensorLike,
  options: { readonly weight?: number; readonly eps?: number; readonly maxIter?: number } = {},
): Tensor {
  const r = rasterOf(x, 'denoiseTvChambolle')
  const weight = options.weight ?? 0.1
  if (!(weight > 0)) throw new MdmlError('value', 'denoiseTvChambolle: the weight must be positive')
  return fromPlanes(
    r,
    planesOf(r).map((p) => tvChambolle(p, r.h, r.w, weight, options.eps ?? 2e-4, options.maxIter ?? 200)),
  )
}

/** Options of the Perona-Malik diffusion. */
export interface DiffusionOptions {
  /** The edge threshold kappa on the gradient (default 0.1). */
  readonly kappa?: number
  /** The time step (default 0.2; stable up to 0.25). */
  readonly lambda?: number
  /** The conduction: 'exponential' exp(-(g / kappa)^2) (default) or 'quadratic' 1 / (1 + (g / kappa)^2). */
  readonly conduction?: 'exponential' | 'quadratic'
}

/** The state of {@link anisotropicDiffusionAlgorithm}. */
export interface DiffusionState {
  readonly image: Float64Array
  /** The largest change of the last step. */
  readonly change: number
}

/**
 * Perona-Malik anisotropic diffusion (Perona and Malik 1990) as a traced algorithm: each step moves every pixel by
 * lambda times the conduction-weighted differences to its four neighbours, the conduction small across strong edges so
 * regions smooth while edges stay sharp.
 *
 * @param x - Grey image `[H, W]`.
 * @param options - kappa, lambda and the conduction function.
 * @returns The algorithm.
 * @concept anisotropic-diffusion
 * @example
 * ```ts
 * import { anisotropicDiffusionAlgorithm } from 'mdml-engine/vision'
 * anisotropicDiffusionAlgorithm([[0, 1], [1, 0]]).name // => 'perona-malik'
 * ```
 */
export function anisotropicDiffusionAlgorithm(
  x: TensorLike,
  options: DiffusionOptions = {},
): Algorithm<DiffusionState> {
  const r = greyOf(x, 'anisotropicDiffusion')
  const { h, w } = r
  const kappa = options.kappa ?? 0.1
  const lambda = options.lambda ?? 0.2
  if (!(lambda > 0 && lambda <= 0.25)) throw new MdmlError('value', 'anisotropicDiffusion: lambda must be in (0, 0.25]')
  const quad = options.conduction === 'quadratic'
  const g = (d: number): number => (quad ? 1 / (1 + (d / kappa) ** 2) : Math.exp(-((d / kappa) ** 2)))
  return {
    name: 'perona-malik',
    config: { kappa, lambda, conduction: quad ? 'quadratic' : 'exponential' } as Json,
    schema: channels({ metrics: { change: { description: 'largest pixel change of the step', direction: 'lower' } } }),
    init: () => ({ image: Float64Array.from(r.data), change: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const I = s.image
      const out = new Float64Array(h * w)
      let change = 0
      for (let y = 0; y < h; y++)
        for (let xx = 0; xx < w; xx++) {
          const c = I[y * w + xx] ?? 0
          let flux = 0
          for (const [dy, dx] of [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1],
          ] as const) {
            const yy = y + dy
            const x2 = xx + dx
            if (yy < 0 || yy >= h || x2 < 0 || x2 >= w) continue
            const diff = (I[yy * w + x2] ?? 0) - c
            flux += g(Math.abs(diff)) * diff
          }
          const v = c + lambda * flux
          out[y * w + xx] = v
          change = Math.max(change, Math.abs(v - c))
        }
      return { image: out, change }
    },
    observe: (s) => ({ metrics: { change: Number.isFinite(s.change) ? s.change : -1 } }),
  }
}

/**
 * Perona-Malik anisotropic diffusion run for a number of steps.
 *
 * @param x - Grey image `[H, W]`.
 * @param options - Steps (default 10), kappa, lambda, conduction and tracing.
 * @returns The diffused image.
 * @concept anisotropic-diffusion
 * @example
 * ```ts
 * import { anisotropicDiffusion } from 'mdml-engine/vision'
 * anisotropicDiffusion([[0, 0, 1, 1], [0, 0, 1, 1]], { steps: 5 }).shape // => [2, 4]
 * ```
 */
export function anisotropicDiffusion(
  x: TensorLike,
  options: DiffusionOptions & { readonly steps?: number } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent'>
    > = {},
): Tensor {
  const r = greyOf(x, 'anisotropicDiffusion')
  const st = runAlgorithm(anisotropicDiffusionAlgorithm(x, options), {
    maxSteps: options.steps ?? 10,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state
  return rasterTensor({ ...r, data: st.image })
}

/**
 * The guided filter (He, Sun and Tang 2010): a local linear model q = a I + b of the input in every window of the
 * guide, a and b averaged over the windows covering each pixel; edge-preserving smoothing that follows the guide's
 * edges (box means with the border replicated).
 *
 * @param guide - Grey guide image `[H, W]`.
 * @param x - The image to filter, `[H, W]` (or the guide itself).
 * @param options - The window radius (default 2) and the regularisation eps (default 0.01).
 * @returns The filtered image.
 * @concept guided-filter
 * @example
 * ```ts
 * import { guidedFilter } from 'mdml-engine/vision'
 * guidedFilter([[0, 0, 1, 1]], [[0, 0, 1, 1]], { radius: 1 }).shape // => [1, 4]
 * ```
 */
export function guidedFilter(
  guide: TensorLike,
  x: TensorLike,
  options: { readonly radius?: number; readonly eps?: number } = {},
): Tensor {
  const I = greyOf(guide, 'guidedFilter')
  const p = greyOf(x, 'guidedFilter')
  if (I.h !== p.h || I.w !== p.w) throw new MdmlError('shape', 'guidedFilter: the guide and image differ in shape')
  const size = 2 * (options.radius ?? 2) + 1
  const eps = options.eps ?? 0.01
  const mean = (d: Float64Array): Float64Array =>
    Float64Array.from(primalOf(image.uniformFilter(imageTensor(d, I.h, I.w), size, { mode: 'nearest' })).data, Number)
  const mI = mean(I.data)
  const mp = mean(p.data)
  const mIp = mean(Float64Array.from(I.data, (v, i) => v * (p.data[i] ?? 0)))
  const mII = mean(Float64Array.from(I.data, (v) => v * v))
  const a = Float64Array.from(
    mIp,
    (v, i) => (v - (mI[i] ?? 0) * (mp[i] ?? 0)) / ((mII[i] ?? 0) - (mI[i] ?? 0) ** 2 + eps),
  )
  const b = Float64Array.from(mp, (v, i) => v - (a[i] ?? 0) * (mI[i] ?? 0))
  const ma = mean(a)
  const mb = mean(b)
  return rasterTensor({ ...p, data: Float64Array.from(I.data, (v, i) => (ma[i] ?? 0) * v + (mb[i] ?? 0)) })
}

/** The state of {@link inpaintAlgorithm}. */
export interface InpaintState {
  readonly image: Float64Array
  readonly change: number
}

/**
 * Inpainting by harmonic diffusion as a traced algorithm: the masked pixels repeatedly take the mean of their four
 * neighbours (Jacobi iterations of Laplace's equation with the known pixels fixed), filling holes smoothly from their
 * borders; channels alike.
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param mask - Pixels to fill `[H, W]` (nonzero).
 * @returns The algorithm (stopping when the largest change is below 1e-6).
 * @concept inpainting
 * @example
 * ```ts
 * import { inpaintAlgorithm } from 'mdml-engine/vision'
 * inpaintAlgorithm([[0, 9, 2]], [[0, 1, 0]]).name // => 'diffusion-inpaint'
 * ```
 */
export function inpaintAlgorithm(x: TensorLike, mask: TensorLike): Algorithm<InpaintState> {
  const r = rasterOf(x, 'inpaint')
  const m = greyOf(mask, 'inpaint')
  if (m.h !== r.h || m.w !== r.w) throw new MdmlError('shape', 'inpaint: the mask is [H, W]')
  const { h, w, c } = r
  const holes: number[] = []
  for (let i = 0; i < h * w; i++) if ((m.data[i] ?? 0) !== 0) holes.push(i)
  return {
    name: 'diffusion-inpaint',
    config: { holes: holes.length } as Json,
    schema: channels({ metrics: { change: { description: 'largest change of a filled pixel', direction: 'lower' } } }),
    init: () => {
      // Start the holes at the mean of the known pixels.
      const img = Float64Array.from(r.data)
      for (let k = 0; k < c; k++) {
        let s = 0
        let n = 0
        for (let i = 0; i < h * w; i++)
          if ((m.data[i] ?? 0) === 0) {
            s += r.data[i * c + k] ?? 0
            n++
          }
        for (const i of holes) img[i * c + k] = n > 0 ? s / n : 0
      }
      return { image: img, change: Number.POSITIVE_INFINITY }
    },
    step: (s) => {
      const img = Float64Array.from(s.image)
      let change = 0
      for (const i of holes) {
        const y = Math.floor(i / w)
        const xx = i % w
        for (let k = 0; k < c; k++) {
          let sum = 0
          let n = 0
          for (const [yy, x2] of [
            [y - 1, xx],
            [y + 1, xx],
            [y, xx - 1],
            [y, xx + 1],
          ] as const) {
            if (yy < 0 || yy >= h || x2 < 0 || x2 >= w) continue
            sum += s.image[(yy * w + x2) * c + k] ?? 0
            n++
          }
          const v = n > 0 ? sum / n : (s.image[i * c + k] ?? 0)
          change = Math.max(change, Math.abs(v - (s.image[i * c + k] ?? 0)))
          img[i * c + k] = v
        }
      }
      return { image: img, change }
    },
    done: (s) => (s.change < 1e-6 ? 'converged' : undefined),
    observe: (s) => ({ metrics: { change: Number.isFinite(s.change) ? s.change : -1 } }),
  }
}

/**
 * Inpaint masked pixels by harmonic diffusion ({@link inpaintAlgorithm} run to convergence or maxIter steps).
 *
 * @param x - Image `[H, W]` or `[H, W, C]`.
 * @param mask - Pixels to fill `[H, W]`.
 * @param options - maxIter (default 500) and tracing.
 * @returns The inpainted image.
 * @concept inpainting
 * @example
 * ```ts
 * import { inpaint } from 'mdml-engine/vision'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(inpaint([[0, 9, 2]], [[0, 1, 0]])) // => [[0, 1, 2]]
 * ```
 */
export function inpaint(
  x: TensorLike,
  mask: TensorLike,
  options: { readonly maxIter?: number } & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): Tensor {
  const r = rasterOf(x, 'inpaint')
  const st = runAlgorithm(inpaintAlgorithm(x, mask), {
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state
  return rasterTensor({ ...r, data: st.image })
}
