/**
 * Estimating how images move: template matching by normalised cross-correlation, translation by phase correlation,
 * exhaustive block matching (the motion vectors of video coding), and dense optical flow by Lucas-Kanade (local least
 * squares) and Horn-Schunck (a global smoothness prior solved by Jacobi iterations, also as a traced algorithm).
 */

import { type Json, WhiteboxError } from '../../base/index.ts'
import { fft2, ifft2 } from '../../numerics/fourier/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { float } from '../_shared.ts'
import { gaussianFilter, uniformFilter } from './filters.ts'

// Motion and matching: template matching by normalised cross-correlation and translation by phase correlation (as
// scikit-image), exhaustive block matching, and dense optical flow by Lucas-Kanade (windowed least squares) and
// Horn-Schunck (a traced iterative smoothness-regularised solve). TODO: pyramidal iterative Lucas-Kanade and TV-L1
// (skimage.registration.optical_flow_ilk and optical_flow_tvl1).

interface Plane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

function planeOf(x: TensorLike, name: string): Plane {
  const v = primalOf(float(x))
  if (v.shape.length !== 2) throw new WhiteboxError('shape', `${name}: images are [H, W]`)
  return { h: v.shape[0] ?? 0, w: v.shape[1] ?? 0, data: Float64Array.from(v.data, Number) }
}

/**
 * Template matching by normalised cross-correlation (skimage.feature.match_template): the Pearson correlation of the
 * template with every window of the image (0 where the window is flat); with padInput the image is zero-padded so
 * the output has the image's shape and is centred on the template.
 *
 * @param x - Image `[H, W]`.
 * @param template - Template `[h, w]`.
 * @param padInput - Pad so the response has the image shape (default false).
 * @returns The response `[H - h + 1, W - w + 1]` (or `[H, W]`), in [-1, 1].
 * @concept template-matching
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * const x = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
 * Math.round(((toArray(image.matchTemplate(x, [[1, 2], [3, 4]])) as number[][])[1]?.[1] ?? 0) * 1e9) / 1e9 // => 1
 * ```
 */
export function matchTemplate(x: TensorLike, template: TensorLike, padInput = false): Tensor {
  const img0 = planeOf(x, 'matchTemplate')
  const t = planeOf(template, 'matchTemplate')
  if (t.h > img0.h || t.w > img0.w)
    throw new WhiteboxError('shape', 'matchTemplate: the template is larger than the image')
  let img = img0
  let r0 = 0
  let c0 = 0
  if (padInput) {
    const ph = t.h
    const pw = t.w
    const data = new Float64Array((img0.h + 2 * ph) * (img0.w + 2 * pw))
    for (let r = 0; r < img0.h; r++)
      for (let c = 0; c < img0.w; c++) data[(r + ph) * (img0.w + 2 * pw) + c + pw] = img0.data[r * img0.w + c] ?? 0
    img = { h: img0.h + 2 * ph, w: img0.w + 2 * pw, data }
    // scikit-image's crop of the padded response: windows start (template - 1) // 2 + 1 into the padded image.
    r0 = Math.floor((t.h - 1) / 2) + 1
    c0 = Math.floor((t.w - 1) / 2) + 1
  }
  const n = t.h * t.w
  let tmean = 0
  for (const v of t.data) tmean += v / n
  let tssd = 0
  for (const v of t.data) tssd += (v - tmean) ** 2
  const oh = padInput ? img0.h : img0.h - t.h + 1
  const ow = padInput ? img0.w : img0.w - t.w + 1
  const out = new Float64Array(oh * ow)
  for (let r = 0; r < oh; r++)
    for (let c = 0; c < ow; c++) {
      let s = 0
      let s2 = 0
      let x2 = 0
      for (let i = 0; i < t.h; i++)
        for (let j = 0; j < t.w; j++) {
          const v = img.data[(r + r0 + i) * img.w + c + c0 + j] ?? 0
          s += v
          s2 += v * v
          x2 += v * (t.data[i * t.w + j] ?? 0)
        }
      const num = x2 - s * tmean
      const den = Math.sqrt(Math.max(0, (s2 - (s * s) / n) * tssd))
      out[r * ow + c] = den > Number.EPSILON ? num / den : 0
    }
  return fromStorage(out, [oh, ow], 'float64')
}

/**
 * Translation between two images by phase correlation (skimage.registration.phase_cross_correlation at pixel
 * precision): the peak of the inverse FFT of the normalised cross-power spectrum.
 *
 * @param reference - Reference image `[H, W]`.
 * @param moving - Moving image, the same shape.
 * @returns The shift [rows, cols] that registers moving to reference, the RMS error and the global phase difference.
 * @concept image-registration
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * const a = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => Math.sin(i) + Math.cos(2 * j)))
 * const b = a.map((_, i) => a[(i + 6) % 8] as number[])
 * image.phaseCrossCorrelation(a, b).shift // => [-2, 0]
 * ```
 */
export function phaseCrossCorrelation(
  reference: TensorLike,
  moving: TensorLike,
): { readonly shift: [number, number]; readonly error: number; readonly phasediff: number } {
  const a = planeOf(reference, 'phaseCrossCorrelation')
  const b = planeOf(moving, 'phaseCrossCorrelation')
  if (a.h !== b.h || a.w !== b.w) throw new WhiteboxError('shape', 'phaseCrossCorrelation: images differ in shape')
  const fa = fft2(fromStorage(a.data, [a.h, a.w], 'float64'))
  const fb = fft2(fromStorage(b.data, [b.h, b.w], 'float64'))
  const ar = toFlat(fa.re)
  const ai = toFlat(fa.im)
  const br = toFlat(fb.re)
  const bi = toFlat(fb.im)
  const n = a.h * a.w
  const pr = new Float64Array(n)
  const pi = new Float64Array(n)
  let srcAmp = 0
  let tgtAmp = 0
  for (let k = 0; k < n; k++) {
    const re = (ar[k] ?? 0) * (br[k] ?? 0) + (ai[k] ?? 0) * (bi[k] ?? 0)
    const im = (ai[k] ?? 0) * (br[k] ?? 0) - (ar[k] ?? 0) * (bi[k] ?? 0)
    const mag = Math.max(Math.hypot(re, im), 100 * Number.EPSILON)
    pr[k] = re / mag
    pi[k] = im / mag
    srcAmp += (ar[k] ?? 0) ** 2 + (ai[k] ?? 0) ** 2
    tgtAmp += (br[k] ?? 0) ** 2 + (bi[k] ?? 0) ** 2
  }
  srcAmp /= n
  tgtAmp /= n
  const cc = ifft2({ re: fromStorage(pr, [a.h, a.w], 'float64'), im: fromStorage(pi, [a.h, a.w], 'float64') })
  const cr = toFlat(cc.re)
  const ci = toFlat(cc.im)
  let best = 0
  for (let k = 1; k < n; k++)
    if (Math.hypot(cr[k] ?? 0, ci[k] ?? 0) > Math.hypot(cr[best] ?? 0, ci[best] ?? 0)) best = k
  let sr = Math.floor(best / a.w)
  let sc = best % a.w
  if (sr > Math.trunc(a.h / 2)) sr -= a.h
  if (sc > Math.trunc(a.w / 2)) sc -= a.w
  const mr = cr[best] ?? 0
  const mi = ci[best] ?? 0
  const error = Math.sqrt(Math.abs(1 - (mr * mr + mi * mi) / (srcAmp * tgtAmp)))
  return { shift: [sr, sc], error, phasediff: Math.atan2(mi, mr) }
}

/**
 * Exhaustive block matching: for each block of the first frame, the displacement within the search radius that
 * minimises the sum of absolute differences to the second frame (the motion vectors of video coding).
 *
 * @param frame0 - First frame `[H, W]`.
 * @param frame1 - Second frame.
 * @param options - Block size (default 8) and search radius (default 4).
 * @returns Motion vectors `[blocksH, blocksW, 2]` (row and column displacements) and their costs.
 * @concept block-matching
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const a = Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => Math.sin(i / 2) * Math.cos(j / 3)))
 * const b = a.map((_, i) => (a[i - 1] ?? a[0]) as number[])
 * toFlat(image.blockMatching(a, b, { block: 4, radius: 2 }).vectors).slice(10, 12) // => [1, 0]
 * ```
 */
export function blockMatching(
  frame0: TensorLike,
  frame1: TensorLike,
  options: { readonly block?: number; readonly radius?: number } = {},
): { readonly vectors: Tensor; readonly costs: Tensor } {
  const a = planeOf(frame0, 'blockMatching')
  const b = planeOf(frame1, 'blockMatching')
  const bs = options.block ?? 8
  const rad = options.radius ?? 4
  const bh = Math.floor(a.h / bs)
  const bw = Math.floor(a.w / bs)
  const vec = new Float64Array(bh * bw * 2)
  const cost = new Float64Array(bh * bw)
  for (let by = 0; by < bh; by++)
    for (let bx = 0; bx < bw; bx++) {
      let best = Number.POSITIVE_INFINITY
      let vy = 0
      let vx = 0
      // Search in order of increasing displacement so that ties prefer small motion.
      const cands: [number, number][] = []
      for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) cands.push([dy, dx])
      cands.sort((p, q) => Math.abs(p[0]) + Math.abs(p[1]) - (Math.abs(q[0]) + Math.abs(q[1])))
      for (const [dy, dx] of cands) {
        const y0 = by * bs + dy
        const x0 = bx * bs + dx
        if (y0 < 0 || x0 < 0 || y0 + bs > b.h || x0 + bs > b.w) continue
        let s = 0
        for (let i = 0; i < bs; i++)
          for (let j = 0; j < bs; j++)
            s += Math.abs((a.data[(by * bs + i) * a.w + bx * bs + j] ?? 0) - (b.data[(y0 + i) * b.w + x0 + j] ?? 0))
        if (s < best) {
          best = s
          vy = dy
          vx = dx
        }
      }
      vec[(by * bw + bx) * 2] = vy
      vec[(by * bw + bx) * 2 + 1] = vx
      cost[by * bw + bx] = best
    }
  return { vectors: fromStorage(vec, [bh, bw, 2], 'float64'), costs: fromStorage(cost, [bh, bw], 'float64') }
}

/** Spatial and temporal derivatives of a frame pair: central differences of their mean, and the frame difference. */
function derivatives(a: Plane, b: Plane): { ix: Float64Array; iy: Float64Array; it: Float64Array } {
  const { h, w } = a
  const avg = Float64Array.from(a.data, (v, i) => (v + (b.data[i] ?? 0)) / 2)
  const at = (r: number, c: number): number =>
    avg[Math.min(h - 1, Math.max(0, r)) * w + Math.min(w - 1, Math.max(0, c))] ?? 0
  const ix = new Float64Array(h * w)
  const iy = new Float64Array(h * w)
  for (let r = 0; r < h; r++)
    for (let c = 0; c < w; c++) {
      ix[r * w + c] = (at(r, c + 1) - at(r, c - 1)) / 2
      iy[r * w + c] = (at(r + 1, c) - at(r - 1, c)) / 2
    }
  const it = Float64Array.from(b.data, (v, i) => v - (a.data[i] ?? 0))
  return { ix, iy, it }
}

/**
 * Dense Lucas-Kanade optical flow: at every pixel the least-squares flow (u, v) solving Ix u + Iy v + It = 0 over a
 * Gaussian (or uniform) window; flow is 0 where the window's structure tensor is singular.
 *
 * @param frame0 - First frame `[H, W]`.
 * @param frame1 - Second frame.
 * @param options - Window: Gaussian sigma (default 2) or uniform size.
 * @returns The flow `[2, H, W]` (rows: v then u, as scikit-image).
 * @concept optical-flow
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const f = (dx: number) => Array.from({ length: 20 }, (_, i) => Array.from({ length: 20 }, (_, j) => Math.sin((j - dx) / 3) + Math.cos(i / 4)))
 * Math.round((toFlat(image.lucasKanade(f(0), f(0.2)))[400 + 210] ?? 0) * 10) / 10 // => 0.2
 * ```
 */
export function lucasKanade(
  frame0: TensorLike,
  frame1: TensorLike,
  options: { readonly sigma?: number; readonly windowSize?: number } = {},
): Tensor {
  const a = planeOf(frame0, 'lucasKanade')
  const b = planeOf(frame1, 'lucasKanade')
  const { h, w } = a
  const { ix, iy, it } = derivatives(a, b)
  const smooth = (d: Float64Array): Float64Array => {
    const t = fromStorage(d, [h, w], 'float64')
    const s =
      options.windowSize !== undefined
        ? uniformFilter(t, options.windowSize, { mode: 'nearest' })
        : gaussianFilter(t, options.sigma ?? 2, { mode: 'nearest' })
    return Float64Array.from(primalOf(s).data, Number)
  }
  const prod = (p: Float64Array, q: Float64Array): Float64Array => Float64Array.from(p, (v, i) => v * (q[i] ?? 0))
  const sxx = smooth(prod(ix, ix))
  const sxy = smooth(prod(ix, iy))
  const syy = smooth(prod(iy, iy))
  const sxt = smooth(prod(ix, it))
  const syt = smooth(prod(iy, it))
  const out = new Float64Array(2 * h * w)
  for (let i = 0; i < h * w; i++) {
    const a11 = sxx[i] ?? 0
    const a12 = sxy[i] ?? 0
    const a22 = syy[i] ?? 0
    const det = a11 * a22 - a12 * a12
    if (Math.abs(det) < 1e-12 * Math.max(1, a11 * a11 + a22 * a22)) continue
    const bx = -(sxt[i] ?? 0)
    const by = -(syt[i] ?? 0)
    const u = (a22 * bx - a12 * by) / det
    const v = (a11 * by - a12 * bx) / det
    out[i] = v
    out[h * w + i] = u
  }
  return fromStorage(out, [2, h, w], 'float64')
}

/** The state of {@link hornSchunckAlgorithm}. */
export interface HornSchunckState {
  readonly u: Float64Array
  readonly v: Float64Array
  readonly change: number
}

/**
 * Horn-Schunck optical flow as an algorithm (Horn and Schunck 1981): Jacobi iterations of
 * u = u_avg - Ix (Ix u_avg + Iy v_avg + It) / (alpha^2 + Ix^2 + Iy^2) (and likewise v), where the averages are the
 * classic 3 x 3 weighted neighbourhood means; traced by the largest update.
 *
 * @param frame0 - First frame `[H, W]`.
 * @param frame1 - Second frame.
 * @param options - Smoothness alpha (default 1) and stopping tolerance (default 1e-6).
 * @returns The algorithm.
 * @concept optical-flow
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * image.hornSchunckAlgorithm([[0, 1], [1, 0]], [[1, 0], [0, 1]]).name // => 'horn-schunck'
 * ```
 */
export function hornSchunckAlgorithm(
  frame0: TensorLike,
  frame1: TensorLike,
  options: { readonly alpha?: number; readonly tol?: number } = {},
): Algorithm<HornSchunckState> {
  const a = planeOf(frame0, 'hornSchunck')
  const b = planeOf(frame1, 'hornSchunck')
  const { h, w } = a
  const { ix, iy, it } = derivatives(a, b)
  const alpha2 = (options.alpha ?? 1) ** 2
  const kernel: [number, number, number][] = [
    [-1, -1, 1 / 12],
    [-1, 0, 1 / 6],
    [-1, 1, 1 / 12],
    [0, -1, 1 / 6],
    [0, 1, 1 / 6],
    [1, -1, 1 / 12],
    [1, 0, 1 / 6],
    [1, 1, 1 / 12],
  ]
  const avg = (f: Float64Array, r: number, c: number): number => {
    let s = 0
    for (const [dr, dc, k] of kernel) {
      const rr = Math.min(h - 1, Math.max(0, r + dr))
      const cc = Math.min(w - 1, Math.max(0, c + dc))
      s += k * (f[rr * w + cc] ?? 0)
    }
    return s
  }
  return {
    name: 'horn-schunck',
    config: { alpha: options.alpha ?? 1, h, w } as Json,
    init: () => ({ u: new Float64Array(h * w), v: new Float64Array(h * w), change: Number.POSITIVE_INFINITY }),
    step: (st) => {
      const u = new Float64Array(h * w)
      const v = new Float64Array(h * w)
      let change = 0
      for (let r = 0; r < h; r++)
        for (let c = 0; c < w; c++) {
          const i = r * w + c
          const ub = avg(st.u, r, c)
          const vb = avg(st.v, r, c)
          const gx = ix[i] ?? 0
          const gy = iy[i] ?? 0
          const t = (gx * ub + gy * vb + (it[i] ?? 0)) / (alpha2 + gx * gx + gy * gy)
          u[i] = ub - gx * t
          v[i] = vb - gy * t
          change = Math.max(change, Math.abs((u[i] ?? 0) - (st.u[i] ?? 0)), Math.abs((v[i] ?? 0) - (st.v[i] ?? 0)))
        }
      return { u, v, change }
    },
    done: (st) => (st.change < (options.tol ?? 1e-6) ? 'converged' : undefined),
    observe: (st) => ({ metrics: { change: st.change } }),
  }
}

/**
 * Horn-Schunck optical flow, run to convergence (or maxIter iterations).
 *
 * @param frame0 - First frame `[H, W]`.
 * @param frame1 - Second frame.
 * @param options - Alpha, tolerance, iterations (default 500) and tracing.
 * @returns The flow `[2, H, W]` (v then u).
 * @concept optical-flow
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const f = (dx: number) => Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => Math.sin((j - dx) / 3) + Math.cos(i / 4)))
 * (toFlat(image.hornSchunck(f(0), f(0.2), { alpha: 0.5 }))[256 + 136] ?? 0) > 0.1 // => true
 * ```
 */
export function hornSchunck(
  frame0: TensorLike,
  frame1: TensorLike,
  options: { readonly alpha?: number; readonly tol?: number; readonly maxIter?: number } & Partial<
    Pick<RunOptions, 'tracer' | 'policy' | 'parent'>
  > = {},
): Tensor {
  const st = runAlgorithm(hornSchunckAlgorithm(frame0, frame1, options), {
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state
  const out = new Float64Array(2 * st.u.length)
  out.set(st.v)
  out.set(st.u, st.u.length)
  const p = planeOf(frame0, 'hornSchunck')
  return fromStorage(out, [2, p.h, p.w], 'float64')
}
