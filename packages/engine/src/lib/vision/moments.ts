/**
 * Moments of shapes, as OpenCV computes them: the spatial, central and normalised moments up to order three of a
 * polygon (by Green's theorem over its edges) or of an image (by summation), the seven Hu invariants, and shape
 * matching by the log-scaled Hu invariants (`cv2.matchShapes`).
 */

import { OmaiError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { greyOf, type PointsLike, pointsOf } from './_image.ts'

/** Spatial (m), central (mu) and normalised central (nu) moments, OpenCV's cv::Moments. */
export interface Moments {
  readonly m00: number
  readonly m10: number
  readonly m01: number
  readonly m20: number
  readonly m11: number
  readonly m02: number
  readonly m30: number
  readonly m21: number
  readonly m12: number
  readonly m03: number
  readonly mu20: number
  readonly mu11: number
  readonly mu02: number
  readonly mu30: number
  readonly mu21: number
  readonly mu12: number
  readonly mu03: number
  readonly nu20: number
  readonly nu11: number
  readonly nu02: number
  readonly nu30: number
  readonly nu21: number
  readonly nu12: number
  readonly nu03: number
}

type Spatial = Pick<Moments, 'm00' | 'm10' | 'm01' | 'm20' | 'm11' | 'm02' | 'm30' | 'm21' | 'm12' | 'm03'>

const ZERO: Moments = {
  m00: 0,
  m10: 0,
  m01: 0,
  m20: 0,
  m11: 0,
  m02: 0,
  m30: 0,
  m21: 0,
  m12: 0,
  m03: 0,
  mu20: 0,
  mu11: 0,
  mu02: 0,
  mu30: 0,
  mu21: 0,
  mu12: 0,
  mu03: 0,
  nu20: 0,
  nu11: 0,
  nu02: 0,
  nu30: 0,
  nu21: 0,
  nu12: 0,
  nu03: 0,
}

/** Central and normalised moments from the spatial ones (OpenCV's completeMomentState). */
function complete(m: Spatial): Moments {
  let cx = 0
  let cy = 0
  let inv = 0
  if (Math.abs(m.m00) > Number.EPSILON) {
    inv = 1 / m.m00
    cx = m.m10 * inv
    cy = m.m01 * inv
  }
  const mu20 = m.m20 - m.m10 * cx
  const mu11 = m.m11 - m.m10 * cy
  const mu02 = m.m02 - m.m01 * cy
  const mu30 = m.m30 - cx * (3 * mu20 + cx * m.m10)
  const mu11x2 = mu11 + mu11
  const mu21 = m.m21 - cx * (mu11x2 + cx * m.m01) - cy * mu20
  const mu12 = m.m12 - cy * (mu11x2 + cy * m.m10) - cx * mu02
  const mu03 = m.m03 - cy * (3 * mu02 + cy * m.m01)
  const s2 = inv * inv
  const s3 = s2 * Math.sqrt(Math.abs(inv))
  return {
    ...m,
    mu20,
    mu11,
    mu02,
    mu30,
    mu21,
    mu12,
    mu03,
    nu20: mu20 * s2,
    nu11: mu11 * s2,
    nu02: mu02 * s2,
    nu30: mu30 * s3,
    nu21: mu21 * s3,
    nu12: mu12 * s3,
    nu03: mu03 * s3,
  }
}

/**
 * The moments of a polygon (cv2.moments of a contour): Green's theorem over its edges gives the spatial moments of
 * its interior exactly, made orientation-independent; the central and normalised moments follow. A polygon of
 * (almost) zero area has all moments 0.
 *
 * @param contour - Vertices `[n, 2]`.
 * @returns The moments.
 * @concept image-moments
 * @example
 * ```ts
 * import { moments } from 'omai-engine/vision'
 * const m = moments([[0, 0], [4, 0], [4, 2], [0, 2]])
 * [m.m00, m.m10 / m.m00, m.m01 / m.m00, m.mu20] // => [8, 2, 1, 10.666666666666666]
 * ```
 */
export function moments(contour: PointsLike): Moments {
  const p = pointsOf(contour, 'moments')
  const n = p.length
  if (n === 0) return ZERO
  let a00 = 0
  let a10 = 0
  let a01 = 0
  let a20 = 0
  let a11 = 0
  let a02 = 0
  let a30 = 0
  let a21 = 0
  let a12 = 0
  let a03 = 0
  let [xp, yp] = p[n - 1] as readonly [number, number]
  let xp2 = xp * xp
  let yp2 = yp * yp
  for (const [x, y] of p) {
    const x2 = x * x
    const y2 = y * y
    const dxy = xp * y - x * yp
    const xs = xp + x
    const ys = yp + y
    a00 += dxy
    a10 += dxy * xs
    a01 += dxy * ys
    a20 += dxy * (xp * xs + x2)
    a11 += dxy * (xp * (ys + yp) + x * (ys + y))
    a02 += dxy * (yp * ys + y2)
    a30 += dxy * xs * (xp2 + x2)
    a03 += dxy * ys * (yp2 + y2)
    a21 += dxy * (xp2 * (3 * yp + y) + 2 * x * xp * ys + x2 * (yp + 3 * y))
    a12 += dxy * (yp2 * (3 * xp + x) + 2 * y * yp * xs + y2 * (xp + 3 * x))
    xp = x
    yp = y
    xp2 = x2
    yp2 = y2
  }
  if (!(Math.abs(a00) > 1.1920928955078125e-7)) return ZERO
  const s = a00 > 0 ? 1 : -1
  return complete({
    m00: a00 * s * 0.5,
    m10: (a10 * s) / 6,
    m01: (a01 * s) / 6,
    m20: (a20 * s) / 12,
    m11: (a11 * s) / 24,
    m02: (a02 * s) / 12,
    m30: (a30 * s) / 20,
    m21: (a21 * s) / 60,
    m12: (a12 * s) / 60,
    m03: (a03 * s) / 20,
  })
}

/**
 * The moments of an image (cv2.moments of an array): m_ij = sum x^i y^j I(x, y) over the pixels (x the column), or
 * of its nonzero mask with `binary`.
 *
 * @param x - Grey image `[H, W]`.
 * @param options - binary (default false): count nonzero pixels as 1.
 * @returns The moments.
 * @concept image-moments
 * @example
 * ```ts
 * import { imageMoments } from 'omai-engine/vision'
 * const m = imageMoments([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
 * [m.m00, m.m10, m.m01] // => [2, 3, 2]
 * ```
 */
export function imageMoments(x: TensorLike, options: { readonly binary?: boolean } = {}): Moments {
  const r = greyOf(x, 'imageMoments')
  const binary = options.binary ?? false
  const m = { m00: 0, m10: 0, m01: 0, m20: 0, m11: 0, m02: 0, m30: 0, m21: 0, m12: 0, m03: 0 }
  for (let y = 0; y < r.h; y++)
    for (let c = 0; c < r.w; c++) {
      const v0 = r.data[y * r.w + c] ?? 0
      const v = binary ? (v0 !== 0 ? 1 : 0) : v0
      if (v === 0) continue
      const xv = c * v
      const x2v = c * xv
      m.m00 += v
      m.m10 += xv
      m.m01 += y * v
      m.m20 += x2v
      m.m11 += y * xv
      m.m02 += y * y * v
      m.m30 += c * x2v
      m.m21 += y * x2v
      m.m12 += y * y * xv
      m.m03 += y * y * y * v
    }
  return complete(m)
}

/**
 * The seven Hu moment invariants (Hu 1962; cv2.HuMoments) from the normalised central moments: invariant to
 * translation, scale and rotation (the seventh changes sign under reflection).
 *
 * @param m - Moments.
 * @returns [h1, ..., h7].
 * @concept image-moments
 * @example
 * ```ts
 * import { huMoments, moments } from 'omai-engine/vision'
 * huMoments(moments([[0, 0], [4, 0], [4, 4], [0, 4]]))[0] // => 0.16666666666666666
 * ```
 */
export function huMoments(m: Moments): number[] {
  let t0 = m.nu30 + m.nu12
  let t1 = m.nu21 + m.nu03
  let q0 = t0 * t0
  let q1 = t1 * t1
  const n4 = 4 * m.nu11
  const s = m.nu20 + m.nu02
  const d = m.nu20 - m.nu02
  const hu = new Array<number>(7).fill(0)
  hu[0] = s
  hu[1] = d * d + n4 * m.nu11
  hu[3] = q0 + q1
  hu[5] = d * (q0 - q1) + n4 * t0 * t1
  t0 *= q0 - 3 * q1
  t1 *= 3 * q0 - q1
  q0 = m.nu30 - 3 * m.nu12
  q1 = 3 * m.nu21 - m.nu03
  hu[2] = q0 * q0 + q1 * q1
  hu[4] = q0 * t0 + q1 * t1
  hu[6] = q1 * t0 - q0 * t1
  return hu
}

/**
 * Compare two shapes by their Hu invariants (cv2.matchShapes): with m = sign(h) log10 |h| per invariant (those above
 * 1e-5 in both), method 1 sums |1 / m_a - 1 / m_b|, method 2 sums |m_a - m_b| and method 3 takes the largest
 * |m_a - m_b| / |m_a|; 0 for identical shapes.
 *
 * @param a - First contour `[n, 2]`.
 * @param b - Second contour.
 * @param method - 1, 2 or 3 (cv2.CONTOURS_MATCH_I1 .. I3; default 1).
 * @returns The dissimilarity.
 * @concept shape-matching
 * @example
 * ```ts
 * import { matchShapes } from 'omai-engine/vision'
 * const tri = [[0, 0], [4, 0], [1, 3]]
 * matchShapes(tri, tri.map(([x, y]) => [2 * x + 5, 2 * y + 1])) < 1e-12 // => true
 * ```
 */
export function matchShapes(a: PointsLike, b: PointsLike, method: 1 | 2 | 3 = 1): number {
  if (method !== 1 && method !== 2 && method !== 3) throw new OmaiError('value', 'matchShapes: method is 1, 2 or 3')
  const ma = huMoments(moments(a))
  const mb = huMoments(moments(b))
  const eps = 1e-5
  let result = 0
  let anyA = false
  let anyB = false
  for (let i = 0; i < 7; i++) {
    const ha = ma[i] ?? 0
    const hb = mb[i] ?? 0
    const aa = Math.abs(ha)
    const ab = Math.abs(hb)
    if (aa > 0) anyA = true
    if (ab > 0) anyB = true
    if (!(aa > eps && ab > eps)) continue
    const la = Math.sign(ha) * Math.log10(aa)
    const lb = Math.sign(hb) * Math.log10(ab)
    if (method === 1) result += Math.abs(-1 / la + 1 / lb)
    else if (method === 2) result += Math.abs(-la + lb)
    else result = Math.max(result, Math.abs((la - lb) / la))
  }
  return anyA !== anyB ? Number.MAX_VALUE : result
}
