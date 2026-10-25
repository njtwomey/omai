/**
 * Geometry for selection tools: the data rectangle between two corners however it was dragged, in the shape linked
 * figures share, and the point nearest a pixel position within a radius.
 */

import type { Point2, Projection } from './projection.ts'

/**
 * A rectangle in data coordinates as x and y ranges, low end first: the shape of a brush in `mdml-render/figure`'s
 * link state, so a brushed rectangle goes straight to `brushSelection` or a `LinkStore`.
 */
export interface DataRect {
  readonly x: readonly [number, number]
  readonly y: readonly [number, number]
}

/**
 * The rectangle with corners `a` and `b`, whichever way it was dragged.
 *
 * @param a - One corner.
 * @param b - The opposite corner.
 * @returns The rectangle.
 */
export function rectBetween(a: Point2, b: Point2): DataRect {
  return { x: [Math.min(a[0], b[0]), Math.max(a[0], b[0])], y: [Math.min(a[1], b[1]), Math.max(a[1], b[1])] }
}

/**
 * The point nearest to a pixel position, measured on screen, if any lies within `radius` pixels.
 *
 * @param points - The points, in data coordinates.
 * @param pixel - The position, in pixels.
 * @param projection - The figure's projection.
 * @param radius - The largest distance that counts, in pixels (default 12).
 * @returns The index of the nearest point, or -1.
 */
export function nearestPoint(points: readonly Point2[], pixel: Point2, projection: Projection, radius = 12): number {
  let best = -1
  let bestD = radius * radius
  points.forEach((p, i) => {
    const [px, py] = projection.toPixel(p)
    const d = (px - pixel[0]) ** 2 + (py - pixel[1]) ** 2
    if (d <= bestD) {
      bestD = d
      best = i
    }
  })
  return best
}
