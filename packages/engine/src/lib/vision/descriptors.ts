/**
 * Local descriptors of keypoints: a SIFT-like 128-dimensional gradient-histogram descriptor (4 x 4 cells of 8
 * orientations over a window rotated to the keypoint's angle, trilinearly interpolated, normalised, clipped at 0.2 and
 * renormalised), and binary descriptors by intensity comparisons of point pairs, BRIEF (Calonder et al. 2010) and
 * ORB's rotated BRIEF (Rublee et al. 2011) over oriented FAST keypoints. The sampling patterns come from an explicit
 * key, so binary descriptors agree only with themselves, not with OpenCV's fixed pattern.
 */

import { MdmlError } from '../base/index.ts'
import { type Key, key as makeKey, normal } from '../random/index.ts'
import { image } from '../signal/index.ts'
import { fromStorage, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { greyOf } from './_image.ts'
import { fast, type Keypoint } from './fast.ts'

interface Plane {
  readonly h: number
  readonly w: number
  readonly data: Float64Array
}

const sample = (p: Plane, y: number, x: number): number => {
  const yy = Math.min(p.h - 1, Math.max(0, y))
  const xx = Math.min(p.w - 1, Math.max(0, x))
  return p.data[yy * p.w + xx] ?? 0
}

function smoothed(x: TensorLike, sigma: number, name: string): Plane {
  const r = greyOf(x, name)
  const t = fromStorage(r.data, [r.h, r.w], 'float64')
  const data =
    sigma > 0 ? Float64Array.from(primalOf(image.gaussianFilter(t, sigma, { mode: 'nearest' })).data, Number) : r.data
  return { h: r.h, w: r.w, data }
}

/**
 * SIFT-like descriptors (Lowe 2004): the gradients in a window of 4 x 4 cells, each cell 3 sigma wide (sigma = size /
 * 2), rotated to the keypoint's angle and Gaussian-weighted, vote by trilinear interpolation into 8 orientation bins
 * per cell; the 128-vector is normalised, clipped at 0.2 and renormalised.
 *
 * @param x - Grey image `[H, W]`.
 * @param keypoints - Keypoints with size and angle (an angle of -1 is taken as 0).
 * @returns One 128-dimensional descriptor per keypoint.
 * @concept sift
 * @example
 * ```ts
 * import { siftDescriptors } from 'mdml-engine/vision'
 * const img = Array.from({ length: 32 }, (_, i) => Array.from({ length: 32 }, (_, j) => Math.sin(i / 3) + Math.cos(j / 5)))
 * siftDescriptors(img, [{ x: 16, y: 16, size: 4, angle: 30, response: 1, octave: 0 }])[0].length // => 128
 * ```
 */
export function siftDescriptors(x: TensorLike, keypoints: readonly Keypoint[]): Float64Array[] {
  const p = smoothed(x, 0, 'siftDescriptors')
  const d = 4
  const n = 8
  return keypoints.map((kp) => {
    const sigma = kp.size / 2
    const hist = 3 * sigma
    const angle = ((kp.angle < 0 ? 0 : kp.angle) * Math.PI) / 180
    const cos = Math.cos(angle) / hist
    const sin = Math.sin(angle) / hist
    const radius = Math.round((hist * Math.SQRT2 * (d + 1)) / 2)
    const expScale = -1 / (0.5 * d * d)
    const bins = new Float64Array((d + 2) * (d + 2) * (n + 2))
    const cx = Math.round(kp.x)
    const cy = Math.round(kp.y)
    for (let dy = -radius; dy <= radius; dy++)
      for (let dx = -radius; dx <= radius; dx++) {
        // Rotate into the keypoint's frame, in cell units centred on the window.
        const cr = dx * -sin + dy * cos
        const cc = dx * cos + dy * sin
        const rbin = cr + d / 2 - 0.5
        const cbin = cc + d / 2 - 0.5
        if (!(rbin > -1 && rbin < d && cbin > -1 && cbin < d)) continue
        const yy = cy + dy
        const xx = cx + dx
        if (yy <= 0 || yy >= p.h - 1 || xx <= 0 || xx >= p.w - 1) continue
        const gx = sample(p, yy, xx + 1) - sample(p, yy, xx - 1)
        const gy = sample(p, yy - 1, xx) - sample(p, yy + 1, xx)
        let ori = Math.atan2(gy, gx) - angle
        while (ori < 0) ori += 2 * Math.PI
        while (ori >= 2 * Math.PI) ori -= 2 * Math.PI
        const obin = (ori * n) / (2 * Math.PI)
        const mag = Math.hypot(gx, gy) * Math.exp((cr * cr + cc * cc) * expScale)
        const r0 = Math.floor(rbin)
        const c0 = Math.floor(cbin)
        const o0 = Math.floor(obin)
        const fr = rbin - r0
        const fc = cbin - c0
        const fo = obin - o0
        for (let a = 0; a < 2; a++)
          for (let b = 0; b < 2; b++)
            for (let c = 0; c < 2; c++) {
              const wgt = mag * (a ? fr : 1 - fr) * (b ? fc : 1 - fc) * (c ? fo : 1 - fo)
              const idx = ((r0 + 1 + a) * (d + 2) + c0 + 1 + b) * (n + 2) + ((o0 + c) % n)
              bins[idx] = (bins[idx] ?? 0) + wgt
            }
      }
    const out = new Float64Array(d * d * n)
    for (let r = 0; r < d; r++)
      for (let c = 0; c < d; c++)
        for (let o = 0; o < n; o++) out[(r * d + c) * n + o] = bins[((r + 1) * (d + 2) + c + 1) * (n + 2) + o] ?? 0
    const norm = (v: Float64Array): number => Math.hypot(...v)
    const n1 = norm(out)
    for (let i = 0; i < out.length; i++) out[i] = Math.min((out[i] ?? 0) / (n1 || 1), 0.2)
    const n2 = norm(out)
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / (n2 || 1)
    return out
  })
}

/**
 * A BRIEF sampling pattern: point pairs in a square patch drawn from an isotropic Gaussian of standard deviation
 * patchSize / 5 (Calonder et al.'s G II), clipped to the patch.
 *
 * @param bits - The number of pairs (bits of the descriptor, a multiple of 8).
 * @param options - patchSize (default 31) and the key.
 * @returns Pairs [x1, y1, x2, y2] relative to the patch centre.
 * @concept binary-descriptors
 * @example
 * ```ts
 * import { briefPattern } from 'mdml-engine/vision'
 * briefPattern(256).length // => 256
 * ```
 */
export function briefPattern(
  bits: number,
  options: { readonly patchSize?: number; readonly key?: Key } = {},
): [number, number, number, number][] {
  if (!(Number.isInteger(bits) && bits > 0 && bits % 8 === 0))
    throw new MdmlError('value', 'briefPattern: bits % 8 = 0')
  const size = options.patchSize ?? 31
  const half = Math.floor(size / 2)
  const g = toFlat(normal(options.key ?? makeKey(0), [bits * 4]))
  const clip = (v: number): number => Math.max(-half, Math.min(half, Math.round((v * size) / 5)))
  return Array.from({ length: bits }, (_, i) => [
    clip(g[4 * i] ?? 0),
    clip(g[4 * i + 1] ?? 0),
    clip(g[4 * i + 2] ?? 0),
    clip(g[4 * i + 3] ?? 0),
  ])
}

/** The bits of point-pair comparisons around a keypoint, the pattern rotated by angle (radians), packed in bytes. */
function binaryDescriptor(p: Plane, kp: Keypoint, pattern: readonly (readonly number[])[], angle: number): Uint8Array {
  const out = new Uint8Array(pattern.length / 8)
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  pattern.forEach(([x1 = 0, y1 = 0, x2 = 0, y2 = 0], i) => {
    const a = sample(p, Math.round(kp.y + s * x1 + c * y1), Math.round(kp.x + c * x1 - s * y1))
    const b = sample(p, Math.round(kp.y + s * x2 + c * y2), Math.round(kp.x + c * x2 - s * y2))
    if (a < b) out[i >> 3] = (out[i >> 3] ?? 0) | (1 << (i & 7))
  })
  return out
}

/**
 * BRIEF descriptors (Calonder et al. 2010): on the Gaussian-smoothed image, one bit per point pair of the pattern,
 * set when the first point is darker than the second; the pattern is not rotated.
 *
 * @param x - Grey image `[H, W]`.
 * @param keypoints - The keypoints.
 * @param options - bits (default 256), patchSize (default 31), smoothing sigma (default 2) and the pattern's key.
 * @returns Descriptors as bytes (bits / 8 each).
 * @concept binary-descriptors
 * @example
 * ```ts
 * import { briefDescriptors } from 'mdml-engine/vision'
 * const img = Array.from({ length: 40 }, (_, i) => Array.from({ length: 40 }, (_, j) => (i * j) % 7 / 7))
 * briefDescriptors(img, [{ x: 20, y: 20, size: 31, angle: -1, response: 0, octave: 0 }])[0].length // => 32
 * ```
 */
export function briefDescriptors(
  x: TensorLike,
  keypoints: readonly Keypoint[],
  options: { readonly bits?: number; readonly patchSize?: number; readonly sigma?: number; readonly key?: Key } = {},
): Uint8Array[] {
  const p = smoothed(x, options.sigma ?? 2, 'briefDescriptors')
  const pattern = briefPattern(options.bits ?? 256, options)
  return keypoints.map((kp) => binaryDescriptor(p, kp, pattern, 0))
}

/** The orientation of a patch by its intensity centroid (Rosin 1999), in degrees. */
function centroidAngle(p: Plane, x: number, y: number, radius: number): number {
  let m01 = 0
  let m10 = 0
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy > radius * radius) continue
      const v = sample(p, y + dy, x + dx)
      m10 += dx * v
      m01 += dy * v
    }
  let a = (Math.atan2(m01, m10) * 180) / Math.PI
  if (a < 0) a += 360
  return a
}

/** Options of {@link orb}. */
export interface OrbOptions {
  /** The most keypoints, by Harris response (default 500). */
  readonly nFeatures?: number
  /** The pyramid scale factor (default 1.2) and levels (default 4). */
  readonly scaleFactor?: number
  readonly nLevels?: number
  /** The FAST threshold in image units (default 20 / 255, OpenCV's). */
  readonly fastThreshold?: number
  /** The patch side (default 31). */
  readonly patchSize?: number
  /** The key of the BRIEF pattern. */
  readonly key?: Key
}

/**
 * ORB features (Rublee et al. 2011): FAST corners at every level of an image pyramid, ranked by the Harris response,
 * oriented by the intensity centroid of their patch, and described by BRIEF with the pattern rotated to that angle.
 *
 * @param x - Grey image `[H, W]` in [0, 1].
 * @param options - Feature count, pyramid, FAST threshold, patch size and pattern key.
 * @returns Keypoints (in input coordinates) and their 32-byte descriptors.
 * @concept binary-descriptors
 * @example
 * ```ts
 * import { orb } from 'mdml-engine/vision'
 * const img = Array.from({ length: 48 }, (_, i) => Array.from({ length: 48 }, (_, j) => Math.sin(i * i * 0.05 + j * 0.9) * Math.cos(j * j * 0.03 - i * 0.4) * 0.5 + 0.5))
 * orb(img, { nLevels: 1 }).keypoints.length > 0 // => true
 * ```
 */
export function orb(
  x: TensorLike,
  options: OrbOptions = {},
): { readonly keypoints: Keypoint[]; readonly descriptors: Uint8Array[] } {
  const r = greyOf(x, 'orb')
  const levels = options.nLevels ?? 4
  const factor = options.scaleFactor ?? 1.2
  const patch = options.patchSize ?? 31
  const half = Math.floor(patch / 2)
  const pattern = briefPattern(256, { patchSize: patch, ...(options.key === undefined ? {} : { key: options.key }) })
  const found: { kp: Keypoint; level: number; plane: Plane; lx: number; ly: number }[] = []
  for (let l = 0; l < levels; l++) {
    const scale = factor ** l
    const h = Math.round(r.h / scale)
    const w = Math.round(r.w / scale)
    if (h < patch + 2 || w < patch + 2) break
    const lv =
      l === 0
        ? fromStorage(r.data, [r.h, r.w], 'float64')
        : image.resize(fromStorage(r.data, [r.h, r.w], 'float64'), [h, w])
    const flat = Float64Array.from(toFlat(lv))
    const plane: Plane = { h, w, data: flat }
    const harris = toFlat(image.cornerHarris(lv, { k: 0.04 }))
    const corners = fast(lv, { threshold: options.fastThreshold ?? 20 / 255 })
    for (const c of corners) {
      if (c.x < half || c.y < half || c.x >= w - half || c.y >= h - half) continue
      found.push({
        kp: { ...c, response: harris[c.y * w + c.x] ?? 0 },
        level: l,
        plane,
        lx: c.x,
        ly: c.y,
      })
    }
  }
  found.sort((a, b) => b.kp.response - a.kp.response)
  const kept = found.slice(0, options.nFeatures ?? 500)
  const keypoints: Keypoint[] = []
  const descriptors: Uint8Array[] = []
  const smoothedLevels = new Map<Plane, Plane>()
  for (const f of kept) {
    const scale = factor ** f.level
    const angle = centroidAngle(f.plane, f.lx, f.ly, half)
    let sm = smoothedLevels.get(f.plane)
    if (sm === undefined) {
      sm = smoothed(fromStorage(f.plane.data, [f.plane.h, f.plane.w], 'float64'), 2, 'orb')
      smoothedLevels.set(f.plane, sm)
    }
    const local: Keypoint = { ...f.kp, x: f.lx, y: f.ly }
    descriptors.push(binaryDescriptor(sm, local, pattern, (angle * Math.PI) / 180))
    keypoints.push({
      x: f.lx * scale,
      y: f.ly * scale,
      size: patch * scale,
      angle,
      response: f.kp.response,
      octave: f.level,
    })
  }
  return { keypoints, descriptors }
}
