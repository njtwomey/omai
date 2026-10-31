/**
 * The exposition record of a vector field on the plane (`vectorField`): a flow, a score field or a phase portrait
 * sampled on a grid, with optional nullclines and classified fixed points, so a view can draw arrows or streamlines
 * and a session panel can redraw it at each step (a score field at a noise level, a system as its parameter moves).
 */

import { MdmlError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'

/** The stability class of a fixed point of a two-dimensional system. */
export type FixedPointClass = 'stable' | 'unstable' | 'saddle' | 'centre'

/** A fixed point of the field: where it is, its class and an optional name. */
export interface FixedPoint {
  readonly at: readonly [number, number]
  readonly stability: FixedPointClass
  readonly label?: string
}

/** A curve on which one component of the field vanishes (`dx/dt = 0`), as polylines of `[x, y]` points. */
export interface Nullcline {
  readonly name: string
  readonly paths: readonly (readonly (readonly [number, number])[])[]
}

/**
 * A vector field on a grid over two coordinates: `u[j][i]` and `v[j][i]` are the components at `(x[i], y[j])`. The
 * field of a flow, a score (the gradient of a log density), a velocity; optionally with the curves on which a component
 * vanishes and the fixed points with their stability.
 */
export interface VectorField {
  readonly kind: 'vectorField'
  /** The coordinates' names (default `x`, `y`). */
  readonly features?: readonly [string, string]
  /** The grid's coordinates, increasing. */
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** u[j][i]: the x component at (x[i], y[j]). */
  readonly u: readonly (readonly number[])[]
  /** v[j][i]: the y component at (x[i], y[j]). */
  readonly v: readonly (readonly number[])[]
  /** What the field is (`velocity`, `score`). */
  readonly field?: { readonly name: string }
  readonly nullclines?: readonly Nullcline[]
  readonly fixedPoints?: readonly FixedPoint[]
}

/**
 * A vector field, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a component grid does not match the coordinates; `value` when a fixed point or a
 *   nullcline point is not finite.
 * @example
 * ```ts
 * import { vectorFieldRecord } from 'mdml-engine/exposition'
 * const r = vectorFieldRecord({
 *   x: [-1, 1],
 *   y: [-1, 1],
 *   u: [[1, 1], [1, 1]],
 *   v: [[0, 0], [0, 0]],
 *   fixedPoints: [{ at: [0, 0], stability: 'saddle' }],
 * })
 * r.fixedPoints?.[0]?.stability // => 'saddle'
 * ```
 */
export function vectorFieldRecord(record: Omit<VectorField, 'kind'>): VectorField {
  for (const name of ['u', 'v'] as const) {
    sameLength('vectorField', name, record[name].length, record.y.length)
    rowsOf('vectorField', name, record[name], record.x.length)
  }
  for (const [k, f] of (record.fixedPoints ?? []).entries())
    if (!f.at.every(Number.isFinite)) throw new MdmlError('value', `vectorField: fixedPoints[${k}] is not finite`)
  for (const [k, n] of (record.nullclines ?? []).entries())
    for (const p of n.paths)
      if (p.some((q) => !(Number.isFinite(q[0]) && Number.isFinite(q[1]))))
        throw new MdmlError('value', `vectorField: nullclines[${k}] holds a point that is not finite`)
  return { kind: 'vectorField', ...record }
}

/**
 * Sample a field `(x, y) => [u, v]` on a grid as a record.
 *
 * @param f - The field.
 * @param x - The grid's x coordinates.
 * @param y - The grid's y coordinates.
 * @param extra - Names, nullclines and fixed points to carry.
 * @returns The record.
 * @example
 * ```ts
 * import { sampleVectorField } from 'mdml-engine/exposition'
 * const r = sampleVectorField((x, y) => [-y, x], [-1, 0, 1], [-1, 0, 1])
 * [r.u[2]?.[0], r.v[2]?.[0]] // => [-1, -1]
 * ```
 */
export function sampleVectorField(
  f: (x: number, y: number) => readonly [number, number],
  x: readonly number[],
  y: readonly number[],
  extra: Pick<VectorField, 'features' | 'field' | 'nullclines' | 'fixedPoints'> = {},
): VectorField {
  const cells = y.map((yj) => x.map((xi) => f(xi, yj)))
  return vectorFieldRecord({
    x,
    y,
    u: cells.map((row) => row.map((c) => c[0])),
    v: cells.map((row) => row.map((c) => c[1])),
    ...extra,
  })
}
