// Samples as point sets, digamma of counts and neighbour counting, shared by the estimators. Not exported.
import { OmaiError } from '../base/index.ts'
import { nextAfter } from '../numerics/approximation/index.ts'
import type { SpatialTree } from '../numerics/spatial/index.ts'
import { digamma } from '../numerics/special/index.ts'
import { type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { float } from './_shared.ts'

/** Samples as rows: `[n]` (one variable) or `[n, d]`. */
export interface Points {
  readonly data: Float64Array
  readonly n: number
  readonly d: number
}

export function points(x: TensorLike, name: string): Points {
  const v = float(x)
  if (v.shape.length !== 1 && v.shape.length !== 2)
    throw new OmaiError('shape', `${name}: samples must be [n] or [n, d]`)
  const n = v.shape[0] ?? 0
  return { data: Float64Array.from(toFlat(v)), n, d: v.shape.length === 1 ? 1 : (v.shape[1] ?? 0) }
}

export function concat(...parts: readonly Points[]): Points {
  const n = parts[0]?.n ?? 0
  if (parts.some((p) => p.n !== n)) throw new OmaiError('shape', 'estimators: samples differ in number')
  const d = parts.reduce((s, p) => s + p.d, 0)
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    let o = 0
    for (const p of parts) {
      for (let j = 0; j < p.d; j++) data[i * d + o + j] = p.data[i * p.d + j] ?? 0
      o += p.d
    }
  }
  return { data, n, d }
}

export function tensorOf(p: Points): Value {
  return float(
    p.data.length === 0
      ? []
      : Array.from({ length: p.n }, (_, i) => Array.from(p.data.subarray(i * p.d, (i + 1) * p.d))),
  )
}

export function row(p: Points, i: number): number[][] {
  return [Array.from(p.data.subarray(i * p.d, (i + 1) * p.d))]
}

/** Digamma of each number. */
export function psi(xs: ArrayLike<number>): number[] {
  return toFlat(digamma(float(Float64Array.from(xs as ArrayLike<number>))))
}

/** Count, for each point, the others strictly within its radius (the point itself is included in the count). */
export function countWithin(tree: SpatialTree, p: Points, radii: ArrayLike<number>, strict: boolean): number[] {
  return Array.from({ length: p.n }, (_, i) => {
    const r = radii[i] ?? 0
    return tree.countNeighbours(row(p, i), strict ? nextAfter(r, 0) : r)[0] ?? 0
  })
}

export function checkK(k: number, n: number, name: string): void {
  if (!Number.isInteger(k) || k < 1 || k >= n)
    throw new OmaiError('value', `${name}: k must be an integer in [1, n - 1], got ${k} with n = ${n}`)
}
