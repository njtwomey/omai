/**
 * Constraints on where a dragged or added point may go: inside a box, on the graph of a function, on a segment,
 * polyline or circle, snapped to a grid, along one axis, or several chained in turn.
 */

import type { Point2 } from 'mdml-engine/numerics/geometry'

/**
 * A rule for where a dragged or added point may go: the allowed point nearest to (or standing for) the one asked.
 *
 * @index structure Where a dragged point may go; build with `inBox`, `onGraph`, `onCircle`, `onGrid`, `chain`
 */
export type Constraint = (p: Point2) => Point2

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi)

/**
 * Keep points inside a box.
 *
 * @param x - The x range.
 * @param y - The y range.
 * @returns The constraint.
 */
export function inBox(x: Point2, y: Point2): Constraint {
  return ([u, v]) => [clamp(u, x[0], x[1]), clamp(v, y[0], y[1])]
}

/**
 * Keep points on the graph of a function, `y = f(x)`, with x optionally kept in a range (a point dragged along a
 * curve).
 *
 * @param f - The function.
 * @param x - The x range (default unbounded).
 * @returns The constraint.
 */
export function onGraph(f: (x: number) => number, x?: Point2): Constraint {
  return ([u]) => {
    const t = x === undefined ? u : clamp(u, x[0], x[1])
    return [t, f(t)]
  }
}

/**
 * Keep points on the segment from `a` to `b` (the nearest point of it).
 *
 * @param a - One end.
 * @param b - The other end.
 * @returns The constraint.
 */
export function onSegment(a: Point2, b: Point2): Constraint {
  return (p) => nearestOnSegment(p, a, b)
}

function nearestOnSegment(p: Point2, a: Point2, b: Point2): Point2 {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const len2 = dx * dx + dy * dy
  const t = len2 === 0 ? 0 : clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2, 0, 1)
  return [a[0] + t * dx, a[1] + t * dy]
}

/**
 * Keep points on a polyline (a sampled curve or a path): the nearest point of any of its segments.
 *
 * @param points - The polyline's vertices, in order.
 * @returns The constraint.
 */
export function onPolyline(points: readonly Point2[]): Constraint {
  return (p) => {
    if (points.length === 0) return p
    if (points.length === 1) return points[0] as Point2
    let best: Point2 = points[0] as Point2
    let bestD = Number.POSITIVE_INFINITY
    for (let i = 1; i < points.length; i++) {
      const q = nearestOnSegment(p, points[i - 1] as Point2, points[i] as Point2)
      const d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
      if (d < bestD) {
        bestD = d
        best = q
      }
    }
    return best
  }
}

/**
 * Keep points on a circle (the radial projection; the centre itself goes to the rightmost point).
 *
 * @param centre - The centre.
 * @param radius - The radius.
 * @returns The constraint.
 */
export function onCircle(centre: Point2, radius: number): Constraint {
  return ([u, v]) => {
    const dx = u - centre[0]
    const dy = v - centre[1]
    const r = Math.hypot(dx, dy)
    return r === 0 ? [centre[0] + radius, centre[1]] : [centre[0] + (dx / r) * radius, centre[1] + (dy / r) * radius]
  }
}

/**
 * Snap points to a grid of spacing `dx` by `dy` (default `dx`), anchored at the origin.
 *
 * @param dx - The x spacing.
 * @param dy - The y spacing.
 * @returns The constraint.
 */
export function onGrid(dx: number, dy: number = dx): Constraint {
  const snap = (v: number, d: number): number => Number((Math.round(v / d) * d).toPrecision(12))
  return ([u, v]) => [snap(u, dx), snap(v, dy)]
}

/**
 * Let points move only horizontally (at height `y`) or only vertically (at `x`).
 *
 * @param axis - The axis points move along.
 * @param at - The fixed coordinate of the other axis.
 * @returns The constraint.
 */
export function alongAxis(axis: 'x' | 'y', at: number): Constraint {
  return ([u, v]) => (axis === 'x' ? [u, at] : [at, v])
}

/**
 * Apply constraints in turn (snap to a grid, then keep in a box).
 *
 * @param constraints - The constraints, first applied first.
 * @returns The combined constraint.
 */
export function chain(...constraints: readonly Constraint[]): Constraint {
  return (p) => constraints.reduce((q, c) => c(q), p)
}
