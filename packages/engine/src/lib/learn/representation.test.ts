import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { alignment, cka, effectiveRank, uniformity } from './representation.ts'

const gaussian = (seed: number, n: number, d: number): number[][] => {
  const z = toFlat(normal(key(seed), [n * d]))
  return Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => z[i * d + j] ?? 0))
}

describe('alignment and uniformity', () => {
  it('alignment is the mean power of the distance between normalised positive pairs', () => {
    expect(
      alignment(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toBe(0)
    expect(alignment([[1, 0]], [[0, 1]])).toBeCloseTo(2, 12)
    expect(alignment([[2, 0]], [[0, 5]], 1)).toBeCloseTo(Math.SQRT2, 12)
    expect(alignment([[2, 0]], [[0, 5]], 2, { normalise: false })).toBeCloseTo(29, 12)
    expect(() =>
      alignment(
        [[1, 0]],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow()
  })

  it('uniformity is 0 at a collapse, falls as points spread, and is lowest for the uniform law on the circle', () => {
    expect(
      uniformity([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
    ).toBeCloseTo(0, 12)
    // Two antipodal points: exp(-2 * 4) = exp(-8).
    expect(
      uniformity([
        [1, 0],
        [-1, 0],
      ]),
    ).toBeCloseTo(-8, 12)
    const circle = (n: number, spread: number): number[][] =>
      Array.from({ length: n }, (_, i) => [
        Math.cos((spread * 2 * Math.PI * i) / n),
        Math.sin((spread * 2 * Math.PI * i) / n),
      ])
    const even = uniformity(circle(60, 1))
    const clumped = uniformity(circle(60, 0.2))
    expect(even).toBeLessThan(clumped)
    expect(clumped).toBeLessThan(0)
    // For the uniform law on S^1, E exp(-2 |x - y|^2) = e^-4 I_0(4) = 0.2071 (I_0(4) = 11.30192). Equally spaced points are
    // exactly uniform in this average, and dropping the n self-pairs from the n^2 gives (n E - 1) / (n - 1).
    const expected = Math.log((60 * Math.exp(-4) * 11.301921 - 1) / 59)
    expect(even).toBeCloseTo(expected, 4)
    expect(() => uniformity([[1, 0]])).toThrow()
  })

  it('contrastive-style alignment and uniformity separate a collapsed encoder from a spread one', () => {
    const x = gaussian(1, 80, 6)
    const noisy = x.map((r, i) => r.map((v, j) => v + 0.05 * (gaussian(2, 80, 6)[i]?.[j] ?? 0)))
    const spread = { align: alignment(x, noisy), uniform: uniformity(x) }
    const collapsed = x.map((r) => r.map((v, j) => (j === 0 ? 1 : 0.01 * v)))
    expect(spread.align).toBeLessThan(0.05)
    expect(spread.uniform).toBeLessThan(uniformity(collapsed) - 0.5)
  })
})

describe('effective rank', () => {
  it('counts the used dimensions: identity gives its size, a rank-one matrix gives one', () => {
    expect(
      effectiveRank([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]),
    ).toBeCloseTo(3, 10)
    expect(
      effectiveRank([
        [1, 2],
        [2, 4],
        [3, 6],
      ]),
    ).toBeCloseTo(1, 8)
    const m = gaussian(3, 200, 8)
    expect(effectiveRank(m)).toBeGreaterThan(7)
    // Singular values (2, 1): p = (2/3, 1/3), exp(entropy) = exp(0.6365).
    expect(
      effectiveRank([
        [2, 0],
        [0, 1],
      ]),
    ).toBeCloseTo(Math.exp(-(2 / 3) * Math.log(2 / 3) - (1 / 3) * Math.log(1 / 3)), 10)
    expect(() =>
      effectiveRank([
        [0, 0],
        [0, 0],
      ]),
    ).toThrow()
  })
})

describe('centred kernel alignment', () => {
  const x = gaussian(5, 60, 4)
  it('is 1 for a representation and a rotated, rescaled copy of it', () => {
    const c = Math.cos(0.7)
    const s = Math.sin(0.7)
    const rotated = x.map(([a = 0, b = 0, c2 = 0, d = 0]) => [3 * (c * a - s * b), 3 * (s * a + c * b), 3 * c2, 3 * d])
    expect(cka(x, rotated)).toBeCloseTo(1, 10)
    expect(cka(x, x)).toBeCloseTo(1, 12)
    expect(cka(x, rotated, { kernel: 'rbf' })).toBeGreaterThan(0.5)
  })

  it('is small for independent representations and symmetric', () => {
    const y = gaussian(6, 60, 4)
    expect(cka(x, y)).toBeLessThan(0.35)
    expect(cka(x, y)).toBeCloseTo(cka(y, x), 12)
    expect(cka(x, y, { kernel: 'rbf' })).toBeLessThan(0.5)
    // A non-orthogonal mixing is not invariant: a representation and its squashed copy score below one but well above independent.
    const squashed = x.map((r) => [r[0] ?? 0, 0.1 * (r[1] ?? 0), r[2] ?? 0, 0.1 * (r[3] ?? 0)])
    expect(cka(x, squashed)).toBeGreaterThan(cka(x, y))
    expect(cka(x, squashed)).toBeLessThan(1)
    expect(() => cka(x, y.slice(0, 10))).toThrow()
  })
})
