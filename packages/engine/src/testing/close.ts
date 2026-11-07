import { expect } from 'vitest'
import { type NestedArray, toFlat, type Value } from '../lib/tensor/index.ts'

const flat = (x: Value | ArrayLike<number>): number[] =>
  typeof (x as Value).shape === 'undefined' ? Array.from(x as ArrayLike<number>) : toFlat(x as Value)

/**
 * Assert that two tensors (or number arrays) have the same length and agree entry by entry within `tolerance`
 * (relative to `1 + |want|`).
 */
export function expectClose(got: Value | ArrayLike<number>, want: Value | ArrayLike<number>, tolerance = 1e-10): void {
  const a = flat(got)
  const b = flat(want)
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) {
    const w = b[i] as number
    expect(Math.abs(v - w), `entry ${i}: ${v} against ${w}`).toBeLessThanOrEqual(tolerance * (1 + Math.abs(w)))
  }
}

const flatNested = (x: NestedArray): number[] => (Array.isArray(x) ? x.flatMap(flatNested) : [Number(x)])
const shapeOfNested = (x: NestedArray): number[] =>
  Array.isArray(x) ? [x.length, ...(x.length > 0 ? shapeOfNested(x[0] as NestedArray) : [])] : []

/**
 * Why a value is not a reference answer, or `undefined` when it is: the shape (checked unless the answer is empty, which
 * carries no inner shape), then each entry, equal, both NaN, or within `rtol` relative to `1 + |want|`.
 */
export function mismatch(got: Value, want: NestedArray, rtol = 1e-12): string | undefined {
  const w = flatNested(want)
  if (w.length > 0 && got.shape.join() !== shapeOfNested(want).join()) {
    return `shape [${got.shape}], expected [${shapeOfNested(want)}]`
  }
  const g = toFlat(got)
  if (g.length !== w.length) return `${g.length} values, expected ${w.length}`
  for (const [k, v] of g.entries()) {
    const b = w[k] as number
    if (!((Number.isNaN(v) && Number.isNaN(b)) || v === b || Math.abs(v - b) <= rtol * Math.max(1, Math.abs(b)))) {
      return `[${k}] is ${v}, expected ${b}`
    }
  }
  return undefined
}

/**
 * Assert a table of `[name, thunk, reference]` rows (references from numpy or scipy, pasted): every row is evaluated and
 * every mismatch, or throw, is reported together, so one run shows all that is wrong.
 */
export function expectRows(rows: readonly (readonly [string, () => Value, NestedArray])[], rtol = 1e-12): void {
  const wrong: string[] = []
  for (const [name, thunk, want] of rows) {
    try {
      const why = mismatch(thunk(), want, rtol)
      if (why !== undefined) wrong.push(`${name}: ${why}`)
    } catch (e) {
      wrong.push(`${name}: threw ${e instanceof Error ? e.message : String(e)}`)
    }
  }
  expect(wrong).toEqual([])
}
