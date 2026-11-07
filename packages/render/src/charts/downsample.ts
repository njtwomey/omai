/**
 * Downsampling long series for drawing: largest-triangle-three-buckets (LTTB, which keeps the points that preserve a
 * curve's visual shape) and min-max decimation (which keeps each bucket's extremes so a noisy envelope survives), with
 * `decimate` as the default choice.
 */

import { MdmlError } from 'mdml-engine/base'
import type { Point2 } from 'mdml-engine/numerics/geometry'

/**
 * Largest-triangle-three-buckets downsampling (Steinarsson, 2013, "Downsampling time series for visual
 * representation"): keeps the first and last points and, from each of `threshold - 2` buckets, the point forming the
 * largest triangle with the previous kept point and the next bucket's mean, so peaks and troughs survive.
 *
 * @param points - The points, in increasing x.
 * @param threshold - How many points to keep (at least 3; at or above the input length returns it unchanged).
 * @returns The kept points.
 * @throws MdmlError with code `value` for a threshold below 3.
 * @example
 * ```ts
 * import { lttb } from 'mdml-render/charts'
 * lttb([[0, 0], [1, 5], [2, 0], [3, 1], [4, 0]], 3) // => [[0, 0], [1, 5], [4, 0]]
 * ```
 */
export function lttb(points: readonly Point2[], threshold: number): Point2[] {
  const n = points.length
  if (threshold >= n) return [...points]
  if (!(threshold >= 3)) throw new MdmlError('value', 'lttb: keep at least 3 points')
  const out: Point2[] = [points[0] as Point2]
  const every = (n - 2) / (threshold - 2)
  let a = 0
  for (let i = 0; i < threshold - 2; i++) {
    const nextStart = Math.floor((i + 1) * every) + 1
    const nextEnd = Math.min(Math.floor((i + 2) * every) + 1, n)
    let ax = 0
    let ay = 0
    for (let j = nextStart; j < nextEnd; j++) {
      ax += points[j]?.[0] ?? 0
      ay += points[j]?.[1] ?? 0
    }
    const count = Math.max(1, nextEnd - nextStart)
    ax /= count
    ay /= count
    const start = Math.floor(i * every) + 1
    const end = Math.floor((i + 1) * every) + 1
    const [px = 0, py = 0] = points[a] ?? []
    let best = -1
    let chosen = start
    for (let j = start; j < end; j++) {
      const [x = 0, y = 0] = points[j] ?? []
      const area = Math.abs((px - ax) * (y - py) - (px - x) * (ay - py))
      if (area > best) {
        best = area
        chosen = j
      }
    }
    out.push(points[chosen] as Point2)
    a = chosen
  }
  out.push(points[n - 1] as Point2)
  return out
}

/**
 * Min-max decimation: split the points into buckets and keep each bucket's lowest and highest point (in x order), so a
 * dense noisy series keeps its envelope exactly. Keeps at most `2 * buckets` points plus the ends.
 *
 * @param points - The points, in increasing x.
 * @param buckets - The number of buckets.
 * @returns The kept points.
 * @example
 * ```ts
 * import { minMaxDecimate } from 'mdml-render/charts'
 * minMaxDecimate([[0, 1], [1, 3], [2, 2], [3, 0], [4, 1], [5, 2]], 1) // => [[0, 1], [1, 3], [3, 0], [5, 2]]
 * ```
 */
export function minMaxDecimate(points: readonly Point2[], buckets: number): Point2[] {
  const n = points.length
  if (buckets <= 0 || 2 * buckets + 2 >= n) return [...points]
  const out: Point2[] = []
  const size = n / buckets
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * size)
    const end = Math.min(n, Math.floor((b + 1) * size))
    let lo = start
    let hi = start
    for (let j = start; j < end; j++) {
      if ((points[j]?.[1] ?? 0) < (points[lo]?.[1] ?? 0)) lo = j
      if ((points[j]?.[1] ?? 0) > (points[hi]?.[1] ?? 0)) hi = j
    }
    for (const j of lo === hi ? [lo] : [Math.min(lo, hi), Math.max(lo, hi)]) out.push(points[j] as Point2)
  }
  const first = points[0] as Point2
  const last = points[n - 1] as Point2
  if (out[0] !== first) out.unshift(first)
  if (out[out.length - 1] !== last) out.push(last)
  return out
}

/**
 * Downsample y values against x (default the index) for drawing: LTTB to about `target` points.
 *
 * @param y - The values.
 * @param target - About how many points to keep (default 1000).
 * @param x - The x values (default 0, 1, 2, ...).
 * @returns The kept x and y values.
 * @example
 * ```ts
 * import { decimate } from 'mdml-render/charts'
 * decimate(Array.from({ length: 10000 }, (_, i) => Math.sin(i / 100)), 500).x.length // => 500
 * ```
 */
export function decimate(
  y: readonly number[],
  target = 1000,
  x?: readonly number[],
): { readonly x: number[]; readonly y: number[] } {
  const pts: Point2[] = y.map((v, i) => [x?.[i] ?? i, v])
  const kept = lttb(pts, Math.max(3, target))
  return { x: kept.map((p) => p[0]), y: kept.map((p) => p[1]) }
}
