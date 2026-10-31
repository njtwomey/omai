/**
 * Shape constraints on a smooth term as linear inequalities on its coefficients: monotone (increasing or decreasing),
 * convex or concave, and bounded. On a P-spline the first and second differences of the coefficients are sufficient (and
 * for equally spaced knots exactly right) for monotonicity and convexity of the spline, so the constraint is a handful of
 * rows; on every other basis, and for bounds, the constraint is imposed on the fitted effect over a dense grid of the
 * covariate. A fit under such constraints is a quadratic program at each penalised least-squares step, which
 * `solvePenalised` solves with the active-set method of `optim`.
 */

import { dgemm } from '../numerics/linalg/index.ts'

/** The shapes a smooth can be constrained to. */
export type ShapeKind = 'increasing' | 'decreasing' | 'convex' | 'concave'

/** Linear inequality constraints `rows beta >= rhs`, one row per constraint over the columns of a term (or a design). */
export interface ConstraintBlock {
  /** `[count, width]`, row-major. */
  readonly rows: Float64Array
  readonly rhs: Float64Array
  readonly count: number
  readonly width: number
}

/** What {@link shapeRows} needs to know about a term. */
export interface ShapeSpec {
  readonly shapes: readonly ShapeKind[]
  readonly lower: number | undefined
  readonly upper: number | undefined
  /** The basis kind: `ps` (and `bs` for monotone shapes) use the coefficient differences. */
  readonly basis: string
  /** The raw (uncentred) basis dimension. */
  readonly k: number
  /** The sum-to-zero map `[k, width]` of the centred term, or `undefined` when the term is not centred. */
  readonly Z: Float64Array | undefined
  readonly width: number
  /** The centred effect's basis evaluated on an equally spaced grid, `[grid, width]`. */
  readonly grid: Float64Array
  readonly gridSize: number
}

/**
 * The constraint rows of a term's shape specification, over the term's (centred) coefficients.
 *
 * @param spec - The shapes, bounds and the term's basis.
 * @returns The constraints `rows beta >= rhs`, or `undefined` if there are none.
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { shapeRows } from 'mdml-engine/gam'
 * const grid = Float64Array.from([0, 0.5, 1])
 * const block = shapeRows({ shapes: ['increasing'], lower: undefined, upper: undefined, basis: 'cr', k: 1, Z: undefined, width: 1, grid, gridSize: 3 })
 * block?.count // => 2
 * ```
 */
export function shapeRows(spec: ShapeSpec): ConstraintBlock | undefined {
  const rows: number[][] = []
  const rhs: number[] = []
  const { width, gridSize, grid } = spec
  const gridRow = (i: number): Float64Array => grid.subarray(i * width, (i + 1) * width)
  const push = (row: ArrayLike<number>, sign: number, bound: number): void => {
    rows.push(Array.from({ length: width }, (_, j) => sign * (row[j] ?? 0)))
    rhs.push(sign * bound)
  }
  const exact = (shape: ShapeKind): boolean =>
    spec.basis === 'ps' || (spec.basis === 'bs' && (shape === 'increasing' || shape === 'decreasing'))
  const centred = (raw: Float64Array, count: number): Float64Array =>
    spec.Z === undefined ? raw : dgemm(raw, spec.Z, count, spec.k, width)
  for (const shape of spec.shapes) {
    const sign = shape === 'increasing' || shape === 'convex' ? 1 : -1
    const second = shape === 'convex' || shape === 'concave'
    if (exact(shape)) {
      const k = spec.k
      const count = k - (second ? 2 : 1)
      const raw = new Float64Array(Math.max(count, 0) * k)
      for (let i = 0; i < count; i++) {
        if (second) {
          // The spline's second derivative is a nonnegative combination of second differences of the coefficients.
          raw[i * k + i] = 1
          raw[i * k + i + 1] = -2
          raw[i * k + i + 2] = 1
        } else {
          raw[i * k + i] = -1
          raw[i * k + i + 1] = 1
        }
      }
      const block = centred(raw, count)
      for (let i = 0; i < count; i++) push(block.subarray(i * width, (i + 1) * width), sign, 0)
    } else {
      const steps = gridSize - (second ? 2 : 1)
      for (let i = 0; i < steps; i++) {
        const row = new Float64Array(width)
        const a = gridRow(i)
        const b = gridRow(i + 1)
        const c = second ? gridRow(i + 2) : undefined
        for (let j = 0; j < width; j++) {
          row[j] = c === undefined ? (b[j] ?? 0) - (a[j] ?? 0) : (a[j] ?? 0) - 2 * (b[j] ?? 0) + (c[j] ?? 0)
        }
        push(row, sign, 0)
      }
    }
  }
  for (let i = 0; i < gridSize; i++) {
    if (spec.lower !== undefined) push(gridRow(i), 1, spec.lower)
    if (spec.upper !== undefined) push(gridRow(i), -1, spec.upper)
  }
  if (rows.length === 0) return undefined
  return { rows: Float64Array.from(rows.flat()), rhs: Float64Array.from(rhs), count: rows.length, width }
}
