// Box bounds shared by the derivative-free and global methods (not exported from the barrel).
import { OmaiError } from '../../base/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import type { ParamTree } from '../transform.ts'
import { type Bounds, type BoundVectors, boxOf, clipToBox, type FlatObjective } from './objective.ts'

/** Lower and upper bound vectors (the shared {@link BoundVectors}). */
export type BoundArrays = BoundVectors

/** The bound vectors for `n` parameters, checking the shape and order. */
export function boundArrays(bounds: Bounds, n: number, name: string): BoundArrays {
  try {
    return boxOf(bounds, n)
  } catch (e) {
    throw new OmaiError((e as OmaiError).code ?? 'value', `${name}: ${(e as Error).message}`)
  }
}

/** Finite bound vectors (for global methods, which sample the box). */
export function finiteBounds(bounds: Bounds, n: number, name: string): BoundArrays {
  const b = boundArrays(bounds, n, name)
  for (let i = 0; i < n; i++) {
    if (
      !Number.isFinite(b.lower[i] ?? 0) ||
      !Number.isFinite(b.upper[i] ?? 0) ||
      (b.lower[i] ?? 0) >= (b.upper[i] ?? 0)
    )
      throw new OmaiError('value', `${name}: bounds must be finite with lower < upper (parameter ${i})`)
  }
  return b
}

/** x clipped into the box (a new array): the shared {@link clipToBox}. */
export const clipInto: (x: ArrayLike<number>, b: BoundArrays) => Float64Array = clipToBox

/** The bounds as `(lower, upper)` pairs again. */
export function asPairs(b: BoundArrays): readonly (readonly [number, number])[] {
  return Array.from(b.lower, (lo, i) => [lo, b.upper[i] ?? 0] as const)
}

/** The flat values of a constraint function (one value or a vector) at a flat point. */
export function constraintValues(
  c: (params: ParamTree) => Value,
  objective: FlatObjective,
  x: ArrayLike<number>,
): number[] {
  return toFlat(c(objective.unravel(x)))
}
