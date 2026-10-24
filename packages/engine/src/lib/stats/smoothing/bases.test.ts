import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { type Tensor, toArray, toFlat } from '../../tensor/index.ts'
import { ols } from '../_linear.ts'
import * as B from './index.ts'

const xs = Array.from({ length: 30 }, (_, i) => (i / 29) * 10)
const eigen = (S: Tensor): number[] => {
  const k = S.shape[0] ?? 0
  // Gershgorin-free check through the quadratic form on random vectors.
  const v = toFlat(normal(key(k), [50, k]))
  const s = toFlat(S)
  return Array.from({ length: 50 }, (_, r) => {
    let q = 0
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++) q += (v[r * k + i] ?? 0) * (s[i * k + j] ?? 0) * (v[r * k + j] ?? 0)
    return q
  })
}
const reproduces = (X: Tensor, f: (x: number) => number): number => {
  const n = X.shape[0] ?? 0
  const p = X.shape[1] ?? 0
  return ols(toFlat(X), n, p, xs.map(f)).residuals.reduce((m, r) => Math.max(m, Math.abs(r)), 0)
}

describe('smoothing bases', () => {
  it('every basis has a positive semidefinite penalty with the stated null space', () => {
    const bases = [
      B.bs(xs, { df: 7 }),
      B.ns(xs, { df: 6 }),
      B.ps(xs, { k: 8 }),
      B.cr(xs, { k: 8 }),
      B.cc(xs, { k: 8 }),
      B.tp(xs, { k: 8 }),
      B.gp(xs, { k: 8 }),
      B.duchon(xs, { k: 8, m: 2, s: 0 }),
    ]
    for (const b of bases) {
      for (const q of eigen(b.penalties[0] as Tensor)) expect(q).toBeGreaterThan(-1e-8)
      expect(B.nullSpace(b.penalties[0] as Tensor, { tol: 1e-8 }).dimension).toBe(b.nullSpaceDimension)
    }
    expect(B.nullSpace(B.ts(xs, { k: 8 }).penalties[0] as Tensor).dimension).toBe(0)
  })

  it('unpenalised bases reproduce their null spaces exactly', () => {
    for (const b of [B.cr(xs, { k: 6 }), B.tp(xs, { k: 6 }), B.ps(xs, { k: 6 }), B.gp(xs, { k: 6 })])
      expect(reproduces(b.X, (x) => 2 - 0.5 * x)).toBeLessThan(1e-8)
    expect(reproduces(B.bs(xs, { df: 5, intercept: true }).X, (x) => x ** 3 - x)).toBeLessThan(1e-8)
  })

  it("cr's penalty is the integrated squared second derivative of the interpolating natural spline", () => {
    const b = B.cr(xs, { k: 5 })
    const beta = [1, -2, 0.5, 3, -1]
    const S = toArray(b.penalties[0] as Tensor) as number[][]
    let quad = 0
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) quad += (beta[i] ?? 0) * (S[i]?.[j] ?? 0) * (beta[j] ?? 0)
    // Integrate f''^2 numerically from the basis evaluated on a fine grid.
    const grid = Array.from({ length: 4001 }, (_, i) => (i / 4000) * 10)
    const f = toFlat(b.evaluate(grid)).reduce<number[]>((acc, v, idx) => {
      const r = Math.floor(idx / 5)
      acc[r] = (acc[r] ?? 0) + v * (beta[idx % 5] ?? 0)
      return acc
    }, [])
    const h = 10 / 4000
    let integral = 0
    for (let i = 1; i < 4000; i++)
      integral += (((f[i + 1] ?? 0) - 2 * (f[i] ?? 0) + (f[i - 1] ?? 0)) / (h * h)) ** 2 * h
    expect(integral / quad).toBeCloseTo(1, 2)
  })

  it('cyclic splines join smoothly at the ends and natural splines extrapolate linearly', () => {
    const b = B.cc(xs, { k: 6 })
    const ends = toFlat(b.evaluate([0, 10]))
    for (let j = 0; j < 5; j++) expect(ends[j]).toBeCloseTo(ends[5 + j] ?? 0, 10)
    const n = B.ns(xs, { df: 4 })
    const out = toFlat(n.evaluate([12, 14, 16]))
    for (let j = 0; j < 4; j++) expect((out[8 + j] ?? 0) - 2 * (out[4 + j] ?? 0) + (out[j] ?? 0)).toBeCloseTo(0, 8)
  })

  it('random-effect, factor-smooth and tensor-product bases have the expected shapes and penalties', () => {
    const r = B.re(['b', 'a', 'b', 'c'])
    expect(r.X.shape).toEqual([4, 3])
    const f = B.fs([0, 1, 2, 3, 0, 1, 2, 3], ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], { k: 4 })
    expect(f.penalties.length).toBe(2)
    expect(toFlat(f.evaluate([[1, 1]])).slice(0, 4)).toEqual([0, 0, 0, 0])
    const z = xs.map((v) => Math.sin(v))
    const t = B.te([B.cr(xs, { k: 4 }), B.cr(z, { k: 3 })])
    expect(t.X.shape).toEqual([30, 12])
    expect(t.penalties.length).toBe(2)
    expect(B.ti([B.cr(xs, { k: 4 }), B.cr(z, { k: 3 })]).X.shape).toEqual([30, 6])
    // On a full grid the centred margins make every ti column sum to zero over the data (main effects excluded).
    const gx = [0, 1, 2, 3, 4, 5].flatMap((a) => [0, 1, 2, 3, 4].map(() => a))
    const gz = [0, 1, 2, 3, 4, 5].flatMap(() => [0, 0.3, 0.5, 0.8, 1])
    const i = B.ti([B.cr(gx, { k: 4 }), B.cr(gz, { k: 3 })])
    const X = toFlat(i.X)
    for (let c = 0; c < 6; c++) {
      let s = 0
      for (let rr = 0; rr < 30; rr++) s += X[rr * 6 + c] ?? 0
      expect(Math.abs(s)).toBeLessThan(1e-9)
    }
    expect(toFlat(t.evaluate([[1, 0.5]])).length).toBe(12)
  })

  it('places knots and validates inputs', () => {
    expect(toFlat(B.placeKnots([1, 1, 2, 3, 3], 3))).toEqual([1, 2, 3])
    expect(() => B.tp(xs, { k: 1 })).toThrow()
    expect(() => B.bs(xs)).toThrow()
  })
})

describe('penalties and constraints', () => {
  it('shape constraints give monotone and convex coefficients for any gamma', () => {
    const g = normal(key(3), [10])
    const inc = toFlat(B.shapeConstraint(10, 'increasing').transform(g))
    for (let i = 1; i < 10; i++) expect(inc[i] ?? 0).toBeGreaterThan(inc[i - 1] ?? 0)
    const dec = toFlat(B.shapeConstraint(10, 'decreasing').transform(g))
    for (let i = 1; i < 10; i++) expect(dec[i] ?? 0).toBeLessThan(dec[i - 1] ?? 0)
    const cvx = toFlat(B.shapeConstraint(10, 'convex').transform(g))
    for (let i = 2; i < 10; i++) expect((cvx[i] ?? 0) - 2 * (cvx[i - 1] ?? 0) + (cvx[i - 2] ?? 0)).toBeGreaterThan(0)
    const ccv = toFlat(B.shapeConstraint(10, 'concave').transform(g))
    for (let i = 2; i < 10; i++) expect((ccv[i] ?? 0) - 2 * (ccv[i - 1] ?? 0) + (ccv[i - 2] ?? 0)).toBeLessThan(0)
  })

  it('the sum-to-zero constraint annihilates the column means', () => {
    const X = toArray(normal(key(4), [12, 4])) as number[][]
    const Z = toArray(B.sumToZeroConstraint(X)) as number[][]
    const means = [0, 1, 2, 3].map((j) => X.reduce((s, r) => s + (r[j] ?? 0), 0) / 12)
    for (let c = 0; c < 3; c++) expect(means.reduce((s, m, j) => s + m * (Z[j]?.[c] ?? 0), 0)).toBeCloseTo(0, 12)
    expect(() => B.differencePenalty(3, 3)).toThrow()
  })
})
