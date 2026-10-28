/**
 * Scale-invariant keypoints, SIFT style (Lowe 2004): a Gaussian scale space in octaves, its difference-of-Gaussian
 * extrema refined to sub-pixel and sub-scale position by a quadratic fit, low-contrast and edge-like responses
 * rejected, and each keypoint given one or more dominant gradient orientations from a smoothed 36-bin histogram. The
 * construction follows Lowe and OpenCV's SIFT; the exact keypoints are implementation-defined, so this file is tested
 * by its invariances rather than against OpenCV.
 */

import { image } from '../signal/index.ts'
import { fromStorage, primalOf, type TensorLike } from '../tensor/index.ts'
import { greyOf } from './_image.ts'
import type { Keypoint } from './fast.ts'

/** One image of the scale space: rows, columns and row-major samples. */
export interface ScalePlane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

const at = (p: ScalePlane, y: number, x: number): number => p.data[y * p.w + x] ?? 0

function blurred(p: ScalePlane, sigma: number): ScalePlane {
  const t = fromStorage(p.data, [p.h, p.w], 'float64')
  return {
    h: p.h,
    w: p.w,
    data: Float64Array.from(primalOf(image.gaussianFilter(t, sigma, { mode: 'nearest' })).data, Number),
  }
}

function halved(p: ScalePlane): ScalePlane {
  const h = Math.floor(p.h / 2)
  const w = Math.floor(p.w / 2)
  const data = new Float64Array(h * w)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = at(p, 2 * y, 2 * x)
  return { h, w, data }
}

/** The Gaussian scale space: per octave, layers + 3 images with sigma growing by 2^(1/layers). */
export interface ScaleSpace {
  /** Per octave, the Gaussian images (sigma0 2^(i / layers) relative to the octave). */
  readonly gaussians: ScalePlane[][]
  /** Per octave, the differences of consecutive Gaussians. */
  readonly dogs: ScalePlane[][]
  readonly layers: number
  readonly sigma: number
}

/** Options of {@link buildScaleSpace} and {@link detectSift}. */
export interface ScaleSpaceOptions {
  /** Scales per octave (default 3, Lowe's and OpenCV's). */
  readonly layers?: number
  /** The base scale (default 1.6). */
  readonly sigma?: number
  /** Octaves (default until the image is smaller than 8 pixels). */
  readonly octaves?: number
  /** The blur the input already has (default 0.5, a sampled image). */
  readonly inputSigma?: number
}

/**
 * Build a Gaussian and difference-of-Gaussian scale space (Lowe 2004): the image blurred to the base scale, then
 * layers + 3 progressively blurred images per octave (incremental Gaussian blurs), each octave starting from the
 * twice-blurred layer of the previous one, subsampled by two.
 *
 * @param x - Grey image `[H, W]`.
 * @param options - Layers, base sigma, octaves and the input blur.
 * @returns The Gaussian images and their differences per octave.
 * @concept scale-space
 * @example
 * ```ts
 * import { buildScaleSpace } from 'mdml-engine/vision'
 * const img = Array.from({ length: 32 }, (_, i) => Array.from({ length: 32 }, (_, j) => Math.sin(i / 3) * Math.cos(j / 4)))
 * buildScaleSpace(img).dogs.map((o) => o.length) // => [5, 5, 5]
 * ```
 */
export function buildScaleSpace(x: TensorLike, options: ScaleSpaceOptions = {}): ScaleSpace {
  const r = greyOf(x, 'buildScaleSpace')
  const layers = options.layers ?? 3
  const sigma = options.sigma ?? 1.6
  const input = options.inputSigma ?? 0.5
  const octaves = options.octaves ?? Math.max(1, Math.floor(Math.log2(Math.min(r.h, r.w))) - 2)
  const k = 2 ** (1 / layers)
  // The incremental blurs taking layer i - 1 to layer i.
  const steps = [Math.sqrt(Math.max(sigma * sigma - input * input, 0.01))]
  for (let i = 1; i < layers + 3; i++) {
    const prev = sigma * k ** (i - 1)
    const next = prev * k
    steps.push(Math.sqrt(next * next - prev * prev))
  }
  const gaussians: ScalePlane[][] = []
  let base = blurred({ h: r.h, w: r.w, data: Float64Array.from(r.data) }, steps[0] ?? 1)
  for (let o = 0; o < octaves && base.h >= 8 && base.w >= 8; o++) {
    const octave = [base]
    for (let i = 1; i < layers + 3; i++) octave.push(blurred(octave[i - 1] as ScalePlane, steps[i] ?? 1))
    gaussians.push(octave)
    base = halved(octave[layers] as ScalePlane)
  }
  const dogs = gaussians.map((oct) =>
    oct.slice(1).map((g, i) => ({
      h: g.h,
      w: g.w,
      data: Float64Array.from(g.data, (v, j) => v - ((oct[i] as ScalePlane).data[j] ?? 0)),
    })),
  )
  return { gaussians, dogs, layers, sigma }
}

/** Options of {@link detectSift}. */
export interface SiftDetectOptions extends ScaleSpaceOptions {
  /** The smallest |DoG| response kept, before division by the layers (default 0.04, OpenCV's). */
  readonly contrastThreshold?: number
  /** The largest ratio of principal curvatures kept (default 10). */
  readonly edgeThreshold?: number
  /** The most keypoints, strongest first (default all). */
  readonly maxFeatures?: number
}

/** Solve a 3 x 3 symmetric system by Cramer's rule; undefined when singular. */
function solve3(H: readonly number[], g: readonly number[]): number[] | undefined {
  const [a = 0, b = 0, c = 0, d = 0, e = 0, f = 0] = H // [xx, xy, xs, yy, ys, ss]
  const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)
  if (Math.abs(det) < 1e-15) return undefined
  const [g0 = 0, g1 = 0, g2 = 0] = g
  const x = (g0 * (d * f - e * e) - b * (g1 * f - e * g2) + c * (g1 * e - d * g2)) / det
  const y = (a * (g1 * f - e * g2) - g0 * (b * f - e * c) + c * (b * g2 - g1 * c)) / det
  const s = (a * (d * g2 - g1 * e) - b * (b * g2 - g1 * c) + g0 * (b * e - d * c)) / det
  return [x, y, s]
}

/** The dominant gradient orientations (degrees) around a point of a Gaussian image. */
function orientations(g: ScalePlane, y: number, x: number, scale: number): number[] {
  const nbins = 36
  const radius = Math.round(3 * 1.5 * scale)
  const weight = -1 / (2 * (1.5 * scale) ** 2)
  const hist = new Float64Array(nbins)
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++) {
      const yy = y + dy
      const xx = x + dx
      if (yy <= 0 || yy >= g.h - 1 || xx <= 0 || xx >= g.w - 1) continue
      const gx = at(g, yy, xx + 1) - at(g, yy, xx - 1)
      const gy = at(g, yy - 1, xx) - at(g, yy + 1, xx)
      const mag = Math.hypot(gx, gy)
      let ang = (Math.atan2(gy, gx) * 180) / Math.PI
      if (ang < 0) ang += 360
      const bin = Math.round((nbins * ang) / 360) % nbins
      hist[bin] = (hist[bin] ?? 0) + Math.exp(weight * (dx * dx + dy * dy)) * mag
    }
  // Smooth the circular histogram twice with [1, 4, 6, 4, 1] / 16.
  const smooth = Float64Array.from(hist, (_, i) => {
    const v = (k: number): number => hist[(i + k + nbins) % nbins] ?? 0
    return (v(-2) + v(2) + 4 * (v(-1) + v(1)) + 6 * v(0)) / 16
  })
  let top = 0
  for (const v of smooth) top = Math.max(top, v)
  const out: number[] = []
  for (let i = 0; i < nbins; i++) {
    const l = smooth[(i - 1 + nbins) % nbins] ?? 0
    const c = smooth[i] ?? 0
    const r = smooth[(i + 1) % nbins] ?? 0
    if (c > l && c > r && c >= 0.8 * top) {
      let bin = i + (0.5 * (l - r)) / (l - 2 * c + r)
      bin = bin < 0 ? nbins + bin : bin >= nbins ? bin - nbins : bin
      let angle = (360 * bin) / nbins
      if (Math.abs(angle - 360) < 1e-9) angle = 0
      out.push(angle)
    }
  }
  return out
}

/**
 * Detect SIFT keypoints (Lowe 2004; the detection half of cv2.SIFT): extrema of the difference of Gaussians over
 * their 26 neighbours in space and scale, refined by up to five Newton steps of a quadratic fit, kept when the refined
 * response exceeds contrastThreshold / layers and the ratio of principal curvatures is below edgeThreshold, and given
 * every orientation within 80% of the strongest gradient-histogram peak. Coordinates are in the input image.
 *
 * @param x - Grey image `[H, W]` in [0, 1].
 * @param options - Scale-space options, contrast and edge thresholds and the most keypoints.
 * @returns The keypoints (size twice the scale, angle in degrees), strongest first.
 * @concept sift
 * @example
 * ```ts
 * import { detectSift } from 'mdml-engine/vision'
 * const img = Array.from({ length: 32 }, (_, i) => Array.from({ length: 32 }, (_, j) => Math.exp(-((i - 16) ** 2 + (j - 16) ** 2) / 18)))
 * const k = detectSift(img)[0]
 * [Math.round(k.x), Math.round(k.y)] // => [16, 16]
 * ```
 */
export function detectSift(x: TensorLike, options: SiftDetectOptions = {}): Keypoint[] {
  const ss = buildScaleSpace(x, options)
  const { layers, sigma } = ss
  const contrast = options.contrastThreshold ?? 0.04
  const edge = options.edgeThreshold ?? 10
  const prelim = (0.5 * contrast) / layers
  const out: Keypoint[] = []
  ss.dogs.forEach((dogs, o) => {
    for (let s = 1; s <= layers; s++) {
      const cur = dogs[s] as ScalePlane
      const { h, w } = cur
      for (let y = 5; y < h - 5; y++)
        for (let xx = 5; xx < w - 5; xx++) {
          const v = at(cur, y, xx)
          if (Math.abs(v) <= prelim) continue
          let isMax = true
          let isMin = true
          for (let ds = -1; ds <= 1 && (isMax || isMin); ds++)
            for (let dy = -1; dy <= 1; dy++)
              for (let dx = -1; dx <= 1; dx++) {
                if (ds === 0 && dy === 0 && dx === 0) continue
                const u = at(dogs[s + ds] as ScalePlane, y + dy, xx + dx)
                if (u >= v) isMax = false
                if (u <= v) isMin = false
              }
          if (!isMax && !isMin) continue
          // Newton steps on the quadratic fit of D(x, y, s).
          let ly = y
          let lx = xx
          let ls = s
          let ok = false
          for (let it = 0; it < 5; it++) {
            const D = (dsi: number, dyy: number, dxx: number): number =>
              at(dogs[ls + dsi] as ScalePlane, ly + dyy, lx + dxx)
            const g = [(D(0, 0, 1) - D(0, 0, -1)) / 2, (D(0, 1, 0) - D(0, -1, 0)) / 2, (D(1, 0, 0) - D(-1, 0, 0)) / 2]
            const c = D(0, 0, 0)
            const Hxx = D(0, 0, 1) + D(0, 0, -1) - 2 * c
            const Hyy = D(0, 1, 0) + D(0, -1, 0) - 2 * c
            const Hss = D(1, 0, 0) + D(-1, 0, 0) - 2 * c
            const Hxy = (D(0, 1, 1) - D(0, 1, -1) - D(0, -1, 1) + D(0, -1, -1)) / 4
            const Hxs = (D(1, 0, 1) - D(1, 0, -1) - D(-1, 0, 1) + D(-1, 0, -1)) / 4
            const Hys = (D(1, 1, 0) - D(1, -1, 0) - D(-1, 1, 0) + D(-1, -1, 0)) / 4
            const step = solve3(
              [Hxx, Hxy, Hxs, Hyy, Hys, Hss],
              g.map((v0) => -v0),
            )
            if (step === undefined) break
            if (Math.abs(step[0] ?? 0) < 0.5 && Math.abs(step[1] ?? 0) < 0.5 && Math.abs(step[2] ?? 0) < 0.5) {
              ok = true
              const value =
                c + 0.5 * ((g[0] ?? 0) * (step[0] ?? 0) + (g[1] ?? 0) * (step[1] ?? 0) + (g[2] ?? 0) * (step[2] ?? 0))
              const tr = Hxx + Hyy
              const det = Hxx * Hyy - Hxy * Hxy
              if (Math.abs(value) * layers < contrast || det <= 0 || tr * tr * edge >= (edge + 1) ** 2 * det) ok = false
              if (ok) {
                const scale = sigma * 2 ** ((ls + (step[2] ?? 0)) / layers)
                const factor = 2 ** o
                const gy = Math.round(ly)
                const gx = Math.round(lx)
                for (const angle of orientations(ss.gaussians[o]?.[ls] as ScalePlane, gy, gx, scale))
                  out.push({
                    x: (lx + (step[0] ?? 0)) * factor,
                    y: (ly + (step[1] ?? 0)) * factor,
                    size: 2 * scale * factor,
                    angle,
                    response: Math.abs(value),
                    octave: o,
                  })
              }
              break
            }
            lx += Math.round(step[0] ?? 0)
            ly += Math.round(step[1] ?? 0)
            ls += Math.round(step[2] ?? 0)
            if (ls < 1 || ls > layers || ly < 5 || ly >= h - 5 || lx < 5 || lx >= w - 5) break
          }
        }
    }
  })
  out.sort((a, b) => b.response - a.response)
  return options.maxFeatures === undefined ? out : out.slice(0, options.maxFeatures)
}
