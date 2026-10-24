import type { Point2, Projection } from './projection.ts'

/** A rectangle in data coordinates, `x0 <= x1` and `y0 <= y1`. */
export interface DataRect {
  readonly x0: number
  readonly x1: number
  readonly y0: number
  readonly y1: number
}

/**
 * The rectangle with corners `a` and `b`, whichever way it was dragged.
 *
 * @param a - One corner.
 * @param b - The opposite corner.
 * @returns The rectangle.
 */
export function rectBetween(a: Point2, b: Point2): DataRect {
  return { x0: Math.min(a[0], b[0]), x1: Math.max(a[0], b[0]), y0: Math.min(a[1], b[1]), y1: Math.max(a[1], b[1]) }
}

/**
 * The indices of the points inside a rectangle (edges included).
 *
 * @param points - The points.
 * @param rect - The rectangle.
 * @returns The indices, ascending.
 */
export function pointsInRect(points: readonly Point2[], rect: DataRect): number[] {
  const out: number[] = []
  points.forEach(([x, y], i) => {
    if (x >= rect.x0 && x <= rect.x1 && y >= rect.y0 && y <= rect.y1) out.push(i)
  })
  return out
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
