import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { cdist } from '../numerics/geometry/index.ts'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import { euclideanDistances, squaredDistances } from './_distance.ts'

/** Deterministic points: `offset + spread * (a smooth pseudo-random coordinate)`, row 1 repeats row 0. */
function points(m: number, d: number, offset: number, spread: number): number[][] {
  const rows = Array.from({ length: m }, (_, i) =>
    Array.from({ length: d }, (_, p) => offset + spread * Math.sin(1.7 * i * (p + 1) + 0.3 * p * p + i * i * 0.01)),
  )
  rows[1] = [...(rows[0] as number[])]
  return rows
}

describe('squaredDistances against the broadcast form of cdist', () => {
  // The offsets are where the expansion |x|^2 + |y|^2 - 2 x.y loses everything; the primitive takes differences.
  const cases = [
    { offset: 0, spread: 1, d: 1, m: 40 },
    { offset: 1e4, spread: 1e-3, d: 3, m: 40 },
    { offset: 1e8, spread: 1, d: 20, m: 60 },
    { offset: 1e8, spread: 1e-3, d: 50, m: 60 },
  ]
  it.each(cases)('offset $offset, spread $spread, d = $d: same values, symmetric, zero diagonal, differentiable', (c) => {
    const x = tensor(points(c.m, c.d, c.offset, c.spread))
    const y = tensor(points(c.m - 7, c.d, c.offset, c.spread * 1.5))
    for (const [a, b] of [
      [x, undefined],
      [x, y],
    ] as const) {
      const got = toFlat(squaredDistances(a, b))
      const want = toFlat(cdist(a, b ?? a, 'sqeuclidean'))
      want.forEach((w, i) => {
        expect(Math.abs((got[i] ?? Number.NaN) - w)).toBeLessThanOrEqual(1e-13 * w)
      })
    }
    const n = c.m
    const g = toFlat(squaredDistances(x))
    for (let i = 0; i < n; i++) {
      expect(g[i * n + i]).toBe(0)
      for (let j = 0; j < n; j++) {
        expect(g[i * n + j]).toBe(g[j * n + i])
        expect(g[i * n + j]).toBeGreaterThanOrEqual(0)
      }
    }
    expect(g[1]).toBe(0) // the repeated row: exactly zero off the diagonal too
    if (c.d <= 3) {
      // Finite differences, jvp, vjp and vmap, with the coincident pair and a Euclidean distance that is not smooth there.
      const small = [[0.1, 0.2, 0.3].slice(0, c.d), [0.1, 0.2, 0.3].slice(0, c.d), [0.9, -0.4, 0.5].slice(0, c.d)]
      const other = [[0.1, 0.2, 0.3].slice(0, c.d), [-0.6, 0.7, 0.2].slice(0, c.d)]
      expectConformant((p: Value) => squaredDistances(p), [small])
      expectConformant((p: Value, q: Value) => euclideanDistances(p, q), [small, other])
      expectConformant((p: Value) => euclideanDistances(p), [small])
    }
  })
})
