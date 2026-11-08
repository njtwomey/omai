/**
 * Shapes and their arithmetic: element counts, row-major strides, numpy broadcasting of two shapes, normalising
 * (possibly negative) axis arguments, and comparing and formatting shapes for messages.
 */

import { OmaiError } from '../base/index.ts'

/** A tensor shape: the length of each axis, outermost first. `[]` is a scalar. */
export type Shape = readonly number[]

/**
 * Number of elements in a shape.
 *
 * @param shape - The shape.
 * @returns The product of the axis lengths (1 for a scalar).
 * @example
 * ```ts
 * import { sizeOf } from 'omai-engine/tensor'
 * sizeOf([2, 3, 4]) // => 24
 * sizeOf([]) // => 1
 * ```
 */
export function sizeOf(shape: Shape): number {
  let size = 1
  for (const n of shape) size *= n
  return size
}

/**
 * Row-major strides of a shape (elements to skip per step along each axis).
 *
 * @param shape - The shape.
 * @returns One stride per axis.
 * @example
 * ```ts
 * import { stridesOf } from 'omai-engine/tensor'
 * stridesOf([2, 3, 4]) // => [12, 4, 1]
 * ```
 */
export function stridesOf(shape: Shape): number[] {
  const strides = new Array<number>(shape.length)
  let stride = 1
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    strides[axis] = stride
    stride *= shape[axis] ?? 1
  }
  return strides
}

/**
 * The shape two shapes broadcast to (numpy rules: align trailing axes; a length-1 axis stretches).
 *
 * @param a - First shape.
 * @param b - Second shape.
 * @returns The broadcast shape.
 * @throws OmaiError with code `shape` when the shapes are incompatible.
 * @concept broadcasting
 * @example
 * ```ts
 * import { broadcastShapes } from 'omai-engine/tensor'
 * broadcastShapes([3, 1], [1, 4]) // => [3, 4]
 * broadcastShapes([2, 3], [3]) // => [2, 3]
 * ```
 */
export function broadcastShapes(a: Shape, b: Shape): number[] {
  const rank = Math.max(a.length, b.length)
  const out = new Array<number>(rank)
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - rank + i] ?? 1
    const db = b[b.length - rank + i] ?? 1
    if (da !== db && da !== 1 && db !== 1) {
      throw new OmaiError('shape', `shapes [${a.join(', ')}] and [${b.join(', ')}] do not broadcast`, { a, b })
    }
    out[i] = da === 1 ? db : da
  }
  return out
}

/**
 * Normalise an axis (negative axes count from the end) and check it is in range.
 *
 * @param axis - The axis, possibly negative.
 * @param rank - The number of axes.
 * @returns The axis in `[0, rank)`.
 * @throws OmaiError with code `shape` when the axis is out of range.
 * @example
 * ```ts
 * import { normaliseAxis } from 'omai-engine/tensor'
 * normaliseAxis(-1, 3) // => 2
 * ```
 */
export function normaliseAxis(axis: number, rank: number): number {
  const a = axis < 0 ? axis + rank : axis
  if (!Number.isInteger(a) || a < 0 || a >= rank) {
    throw new OmaiError('shape', `axis ${axis} is out of range for rank ${rank}`, { axis, rank })
  }
  return a
}

/**
 * Normalise an axis argument (undefined means all axes) to a sorted list of distinct axes.
 *
 * @param axis - An axis, a list of axes, or undefined.
 * @param rank - The number of axes.
 * @returns Sorted distinct axes in `[0, rank)`.
 * @throws OmaiError with code `shape` for out-of-range or repeated axes.
 * @example
 * ```ts
 * import { normaliseAxes } from 'omai-engine/tensor'
 * normaliseAxes([-1, 0], 3) // => [0, 2]
 * normaliseAxes(undefined, 2) // => [0, 1]
 * ```
 */
export function normaliseAxes(axis: number | readonly number[] | undefined, rank: number): number[] {
  if (axis === undefined) return Array.from({ length: rank }, (_, i) => i)
  const list = (typeof axis === 'number' ? [axis] : axis).map((a) => normaliseAxis(a, rank))
  if (new Set(list).size !== list.length) throw new OmaiError('shape', `repeated axis in [${list.join(', ')}]`)
  return list.sort((x, y) => x - y)
}

/**
 * Whether two shapes are identical.
 *
 * @param a - First shape.
 * @param b - Second shape.
 * @returns `true` when every axis length matches.
 * @example
 * ```ts
 * import { shapesEqual } from 'omai-engine/tensor'
 * shapesEqual([2, 3], [2, 3]) // => true
 * ```
 */
export function shapesEqual(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((n, i) => n === b[i])
}

/**
 * Format a shape for messages.
 *
 * @param shape - The shape.
 * @returns For example `[2, 3]`.
 * @example
 * ```ts
 * import { formatShape } from 'omai-engine/tensor'
 * formatShape([2, 3]) // => '[2, 3]'
 * ```
 */
export function formatShape(shape: Shape): string {
  return `[${shape.join(', ')}]`
}
