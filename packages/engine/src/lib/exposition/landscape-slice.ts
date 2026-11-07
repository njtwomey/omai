/**
 * The exposition record of a slice through a loss landscape (`landscapeSlice`): the loss along one direction through
 * the current iterate (a 1-D curve, as a line search sees it), or over a plane spanned by two directions (a grid to
 * draw as contours or a surface), with the optimiser's iterates projected onto the slice and, on a curve, the lines
 * that decide a step (the Armijo and Wolfe conditions). A session probes it at each step, centring the slice on the
 * iterate it has reached.
 */

import { MdmlError } from '../base/index.ts'
import { seal } from './_check.ts'

/** The iterates of an optimiser in the coordinates of a slice. */
export interface LandscapePath {
  /** A name for the legend (`gradient descent`). */
  readonly name?: string
  /**
   * coords[k]: iterate k as `[alpha]` on a curve or `[alpha, beta]` on a plane: its coordinates along the slice's
   * directions, relative to the slice's origin.
   */
  readonly coords: readonly (readonly number[])[]
  /** The loss at each iterate (its height above the slice need not be on it; default read off the slice). */
  readonly loss?: readonly number[]
}

/** A straight line to draw on a curve: the line of the Armijo condition, the curvature condition, the tangent. */
export interface LandscapeGuide {
  readonly name: string
  /** The ends as `[alpha, loss]`. */
  readonly from: readonly [number, number]
  readonly to: readonly [number, number]
}

/** A slice through a loss landscape, in one or two directions. */
export interface LandscapeSlice {
  readonly kind: 'landscapeSlice'
  /** One direction (a curve) or two (a plane). */
  readonly dims: 1 | 2
  /** The names of the directions (`descent direction`, `random direction 1`), one per dimension. */
  readonly axes: readonly string[]
  /** The grid along the first direction (alpha), increasing. */
  readonly x: readonly number[]
  /** The grid along the second direction (beta), increasing: a plane only. */
  readonly y?: readonly number[]
  /** curve[j]: the loss at alpha = x[j] (a curve only). */
  readonly curve?: readonly number[]
  /** grid[i][j]: the loss at (x[j], y[i]) (a plane only). */
  readonly grid?: readonly (readonly number[])[]
  /** The iterate the slice is centred on, in slice coordinates (default the origin). */
  readonly at?: readonly number[]
  /** The optimiser's iterates in the slice, in order, up to the viewed step. */
  readonly paths?: readonly LandscapePath[]
  /** Lines drawn on a curve. */
  readonly guides?: readonly LandscapeGuide[]
  /** What the loss is called (`training loss`). */
  readonly lossName?: string
  /** What the slice is (`at step 12, along the negative gradient`), for the caption. */
  readonly caption?: string
}

/**
 * A landscape slice, with its grid and every path checked against the number of directions.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the curve or the grid does not match the axes, a coordinate has the wrong number of
 *   entries or the axes do not match `dims`; `value` when a grid is not increasing or a value is not finite.
 * @example
 * ```ts
 * import { landscapeSlice } from 'mdml-engine/exposition'
 * const r = landscapeSlice({ dims: 1, axes: ['step'], x: [0, 1, 2], curve: [3, 1, 2], paths: [{ coords: [[0], [1]] }] })
 * r.curve // => [3, 1, 2]
 * ```
 */
export function landscapeSlice(record: Omit<LandscapeSlice, 'kind'>): LandscapeSlice {
  const { dims, x, y, curve, grid } = record
  if (record.axes.length !== dims)
    throw new MdmlError('shape', `landscapeSlice: ${record.axes.length} axes for ${dims} directions`)
  const increasing = (name: string, a: readonly number[]): void => {
    if (a.length < 2) throw new MdmlError('shape', `landscapeSlice: ${name} needs at least two points`)
    for (let k = 0; k < a.length; k++) {
      if (!Number.isFinite(a[k])) throw new MdmlError('value', `landscapeSlice: ${name}[${k}] is not finite`)
      if (k > 0 && (a[k] as number) <= (a[k - 1] as number))
        throw new MdmlError('value', `landscapeSlice: ${name} is not increasing at ${k}`)
    }
  }
  increasing('x', x)
  if (dims === 1) {
    if (curve === undefined || curve.length !== x.length)
      throw new MdmlError('shape', `landscapeSlice: curve needs ${x.length} values`)
    if (y !== undefined || grid !== undefined)
      throw new MdmlError('shape', 'landscapeSlice: a curve has neither y nor a grid')
  } else {
    if (y === undefined || grid === undefined)
      throw new MdmlError('shape', 'landscapeSlice: a plane needs y and a grid')
    increasing('y', y)
    if (grid.length !== y.length) throw new MdmlError('shape', `landscapeSlice: grid needs ${y.length} rows`)
    for (const [i, row] of grid.entries())
      if (row.length !== x.length)
        throw new MdmlError('shape', `landscapeSlice: grid[${i}] has ${row.length} entries, expected ${x.length}`)
    if (curve !== undefined) throw new MdmlError('shape', 'landscapeSlice: a plane has no curve')
  }
  if (record.at !== undefined && record.at.length !== dims)
    throw new MdmlError('shape', `landscapeSlice: at has ${record.at.length} entries, expected ${dims}`)
  for (const [k, p] of (record.paths ?? []).entries()) {
    for (const [i, c] of p.coords.entries())
      if (c.length !== dims)
        throw new MdmlError(
          'shape',
          `landscapeSlice: paths[${k}].coords[${i}] has ${c.length} entries, expected ${dims}`,
        )
    if (p.loss !== undefined && p.loss.length !== p.coords.length)
      throw new MdmlError(
        'shape',
        `landscapeSlice: paths[${k}].loss has ${p.loss.length} entries, expected ${p.coords.length}`,
      )
  }
  return seal({ kind: 'landscapeSlice', ...record })
}

/**
 * Sample a loss over a line or a plane of the slice's coordinates into a record: `loss(alpha)` for one direction,
 * `loss(alpha, beta)` for two. The caller closes over the origin and the directions, so the function is of the slice's
 * own coordinates (see {@link sliceCoordinates} for the way back from parameters).
 *
 * @param loss - The loss as a function of the slice coordinates.
 * @param x - The grid along the first direction.
 * @param y - The grid along the second direction; omit for a curve.
 * @param options - The rest of the record: axes, paths, guides, names and caption.
 * @returns The record.
 * @example
 * ```ts
 * import { sampleLandscape } from 'mdml-engine/exposition'
 * sampleLandscape((a) => (a - 1) ** 2, [0, 1, 2]).curve // => [1, 0, 1]
 * sampleLandscape((a, b) => a * b, [0, 1], [0, 1, 2]).grid // => [[0, 0], [0, 1], [0, 2]]
 * ```
 */
export function sampleLandscape(
  loss: (alpha: number, beta: number) => number,
  x: readonly number[],
  y?: readonly number[],
  options: Partial<Omit<LandscapeSlice, 'kind' | 'dims' | 'x' | 'y' | 'curve' | 'grid'>> = {},
): LandscapeSlice {
  const names = options.axes ?? (y === undefined ? ['direction'] : ['direction 1', 'direction 2'])
  return y === undefined
    ? landscapeSlice({ ...options, dims: 1, axes: names, x, curve: x.map((a) => loss(a, 0)) })
    : landscapeSlice({ ...options, dims: 2, axes: names, x, y, grid: y.map((b) => x.map((a) => loss(a, b))) })
}

/**
 * The coordinates of parameter vectors in a slice: for each, the components along the directions relative to the
 * origin, `(theta - origin) . d / (d . d)`, which is the position in the slice when the directions are orthogonal (a
 * descent direction and a random one made orthogonal to it, two filter-normalised directions).
 *
 * @param iterates - iterates[k]: the parameters after k steps.
 * @param origin - The parameters the slice is centred on.
 * @param directions - One direction for a curve, two for a plane.
 * @returns coords[k] as `[alpha]` or `[alpha, beta]`.
 * @throws MdmlError `shape` when a vector's length differs from the origin's.
 * @example
 * ```ts
 * import { sliceCoordinates } from 'mdml-engine/exposition'
 * sliceCoordinates([[1, 1], [3, 2]], [1, 0], [[2, 0], [0, 1]]) // => [[0, 1], [1, 2]]
 * ```
 */
export function sliceCoordinates(
  iterates: readonly (readonly number[])[],
  origin: readonly number[],
  directions: readonly (readonly number[])[],
): number[][] {
  const norms = directions.map((d) => d.reduce((s, v) => s + v * v, 0))
  return iterates.map((theta, k) => {
    if (theta.length !== origin.length)
      throw new MdmlError(
        'shape',
        `sliceCoordinates: iterate ${k} has ${theta.length} entries, expected ${origin.length}`,
      )
    return directions.map((d, j) => {
      const dot = d.reduce((s, v, i) => s + ((theta[i] ?? 0) - (origin[i] ?? 0)) * v, 0)
      return dot / (norms[j] || 1)
    })
  })
}
