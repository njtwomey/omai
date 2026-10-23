// Box bounds shared by the derivative-free and global methods (not exported from the barrel).
import { WhiteboxError } from '../../base/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import type { Tree } from '../transform.ts'
import type { Bounds } from './derivative-free.ts'
import type { FlatObjective } from './objective.ts'

/** Lower and upper bound vectors. */
export interface BoundArrays {
  readonly lower: Float64Array
  readonly upper: Float64Array
}

/** The bound vectors for `n` parameters, checking the shape and order. */
export function boundArrays(bounds: Bounds, n: number, name: string): BoundArrays {
  if (bounds.length !== n)
    throw new WhiteboxError('shape', `${name}: ${bounds.length} bounds given for ${n} parameters`)
  const lower = Float64Array.from(bounds, (b) => b[0])
  const upper = Float64Array.from(bounds, (b) => b[1])
  for (let i = 0; i < n; i++) {
    if (!((lower[i] ?? 0) <= (upper[i] ?? 0)))
      throw new WhiteboxError('value', `${name}: lower bound ${lower[i]} exceeds upper bound ${upper[i]} at ${i}`)
  }
  return { lower, upper }
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
      throw new WhiteboxError('value', `${name}: bounds must be finite with lower < upper (parameter ${i})`)
  }
  return b
}

/** x clipped into the box (a new array). */
export function clipInto(x: ArrayLike<number>, b: BoundArrays): Float64Array {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = Math.min(Math.max(x[i] ?? 0, b.lower[i] ?? 0), b.upper[i] ?? 0)
  return out
}

/** The bounds as `(lower, upper)` pairs again. */
export function asPairs(b: BoundArrays): Bounds {
  return Array.from(b.lower, (lo, i) => [lo, b.upper[i] ?? 0] as const)
}

/** The flat values of a constraint function (one value or a vector) at a flat point. */
export function constraintValues(c: (params: Tree) => Value, objective: FlatObjective, x: ArrayLike<number>): number[] {
  return toFlat(c(objective.unravel(x)))
}
