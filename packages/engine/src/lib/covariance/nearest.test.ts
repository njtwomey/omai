import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blockCovarianceData } from '../datasets/index.ts'
import { key, normal } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { frobeniusDistance, symmetricEigen, toCorrelation } from './_matrix.ts'
import {
  clipCorrelation,
  isCorrelationMatrix,
  nearestCorrelation,
  nearestCorrelationAlgorithm,
  nearestCovariance,
  nearestSpd,
} from './nearest.ts'

const HIGHAM = [
  [1, 1, 0],
  [1, 1, 1],
  [0, 1, 1],
]
const CHAIN = [
  [1, 0.9, 0.4, 0],
  [0.9, 1, 0.9, 0.4],
  [0.4, 0.9, 1, 0.9],
  [0, 0.4, 0.9, 1],
]

/** A random valid correlation matrix: the sample correlation of random data. */
function validCorrelation(seed: number, p: number): number[][] {
  return toCorrelation(toArray(empiricalCovariance(normal(key(seed), [p + 5, p]))) as number[][]).correlation
}

const inner = (a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number =>
  a.reduce((s, r, i) => s + r.reduce((t, v, j) => t + v * (b[i]?.[j] ?? 0), 0), 0)
const minus = (a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] =>
  a.map((r, i) => r.map((v, j) => v - (b[i]?.[j] ?? 0)))

describe('nearestCorrelation', () => {
  it('returns a valid correlation matrix for indefinite input', () => {
    for (const a of [HIGHAM, CHAIN]) {
      expect(isCorrelationMatrix(a)).toBe(false)
      const r = nearestCorrelation(a)
      expect(isCorrelationMatrix(r.matrix, 1e-12)).toBe(true)
      expect(r.converged).toBe(true)
      expect(r.minEigenvalue).toBeGreaterThan(-1e-12)
      expect(r.distance).toBeGreaterThan(0)
    }
  })

  it('has the known nearest correlation matrix of the classic 3 x 3 example (Higham 2002)', () => {
    const r = nearestCorrelation(HIGHAM, { tol: 1e-15 }).matrix
    const expected = [
      [1, 0.7607, 0.1573],
      [0.7607, 1, 0.7607],
      [0.1573, 0.7607, 1],
    ]
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(r[i]?.[j]).toBeCloseTo(expected[i]?.[j] ?? 0, 4)
  })

  it('equals the input when it is already a correlation matrix', () => {
    for (const p of [3, 6, 10]) {
      const c = validCorrelation(p, p)
      const r = nearestCorrelation(c)
      expect(frobeniusDistance(r.matrix, c)).toBeLessThan(1e-12)
      expect(r.iterations).toBeLessThanOrEqual(3)
    }
  })

  it('is no farther from the input than any other valid correlation matrix, and is the projection', () => {
    for (const a of [HIGHAM, CHAIN]) {
      const nearest = nearestCorrelation(a, { tol: 1e-15 })
      const clipped = clipCorrelation(a, 1e-9)
      expect(nearest.distance).toBeLessThanOrEqual(clipped.distance + 1e-12)
      for (let s = 0; s < 40; s++) {
        const v = validCorrelation(s, a.length)
        expect(nearest.distance).toBeLessThanOrEqual(frobeniusDistance(a, v) + 1e-12)
        // The variational inequality of a projection onto a convex set: <a - r, v - r> <= 0 for every member v.
        expect(inner(minus(a, nearest.matrix), minus(v, nearest.matrix))).toBeLessThan(1e-8)
      }
    }
  })

  it('keeps a symmetric result for an asymmetric input by using its symmetric part', () => {
    const a = [
      [1, 0.9, 0.1],
      [0.5, 1, 0.9],
      [0.5, 0.2, 1],
    ]
    const r = nearestCorrelation(a)
    expect(isCorrelationMatrix(r.matrix, 1e-12)).toBe(true)
  })

  it('traces the distance and the negative eigenvalues sweep by sweep until none remain', () => {
    const out = runAlgorithm(nearestCorrelationAlgorithm(CHAIN), { maxSteps: 500 })
    expect(out.status).toBe('converged')
    expect(out.state.negativeEigenvalues).toBe(0)
    expect(out.state.minEigenvalue).toBeGreaterThan(-1e-10)
    expect(out.steps).toBeGreaterThan(3)
    expect(out.state.distance).toBeCloseTo(nearestCorrelation(CHAIN).distance, 8)
    const first = runAlgorithm(nearestCorrelationAlgorithm(CHAIN), { maxSteps: 1 })
    expect(first.state.negativeEigenvalues).toBeGreaterThan(0)
  })

  it('stops at the sweep budget and still returns a valid matrix', () => {
    const r = nearestCorrelation(CHAIN, { maxIter: 2 })
    expect(r.converged).toBe(false)
    expect(isCorrelationMatrix(r.matrix, 1e-10)).toBe(true)
  })

  it('rejects a non-square matrix', () => {
    expect(() => nearestCorrelation([[1, 2, 3]])).toThrow(/square/)
  })
})

describe('clipCorrelation', () => {
  it('raises the small eigenvalues to the floor and rescales to a unit diagonal', () => {
    const r = clipCorrelation(HIGHAM, 1e-3)
    expect(isCorrelationMatrix(r.matrix, 1e-12)).toBe(true)
    expect(r.minEigenvalue).toBeGreaterThan(5e-4)
  })

  it('returns a valid input as it is', () => {
    const c = validCorrelation(1, 5)
    expect(clipCorrelation(c).matrix).toEqual(c)
    expect(clipCorrelation(c).distance).toBe(0)
  })
})

describe('nearestCovariance', () => {
  const cov = [
    [4, 3.6, 1.2],
    [3.6, 4, 3.6],
    [1.2, 3.6, 4],
  ].map((r, i) => r.map((v, j) => (i === j ? 4 : v)))

  it('keeps the variances and gives a positive semidefinite matrix by either method', () => {
    for (const method of ['clipped', 'nearest'] as const) {
      const r = nearestCovariance(cov, { method, threshold: 1e-8 })
      for (let i = 0; i < 3; i++) expect(r.covariance[i]?.[i]).toBeCloseTo(4, 12)
      expect(symmetricEigen(r.covariance).values[2]).toBeGreaterThan(-1e-9)
      expect(r.std).toEqual([2, 2, 2])
    }
  })

  it('is nearer to the input correlation by Higham than by clipping', () => {
    const near = nearestCovariance(cov, { method: 'nearest' })
    const clip = nearestCovariance(cov, { method: 'clipped' })
    expect(near.distance).toBeLessThanOrEqual(clip.distance + 1e-12)
  })

  it('returns a valid covariance unchanged and refuses a non-positive variance', () => {
    expect(
      nearestCovariance([
        [4, 0],
        [0, 9],
      ]).covariance,
    ).toEqual([
      [4, 0],
      [0, 9],
    ])
    expect(() =>
      nearestCovariance([
        [0, 0],
        [0, 1],
      ]),
    ).toThrow(/diagonal/)
  })

  it('repairs the estimate of a long-run block structure that thresholding broke', () => {
    const { data } = generate(blockCovarianceData, { key: key(3), knobs: { n: 60, p: 12, blocks: 3 } })
    const s = toArray(empiricalCovariance(data.x)) as number[][]
    // Hard-threshold the small covariances, which is how an estimate stops being positive semidefinite.
    const thresholded = s.map((r, i) => r.map((v, j) => (i === j || Math.abs(v) > 0.35 ? v : 0)))
    const r = nearestCovariance(thresholded, { method: 'nearest' })
    expect(symmetricEigen(r.covariance).values[11]).toBeGreaterThan(-1e-9)
  })
})

describe('nearestSpd', () => {
  it('clips the negative eigenvalues of the symmetric part to zero', () => {
    const r = nearestSpd([
      [1, 0],
      [0, -1],
    ])
    expect(r.matrix[0]?.[0]).toBeCloseTo(1, 14)
    expect(r.matrix[1]?.[1]).toBeCloseTo(0, 14)
    expect(r.distance).toBeCloseTo(1, 14)
  })

  it('returns a positive semidefinite matrix no farther than any other, and a PSD input as it is', () => {
    const a = [
      [2, -1.5, 0.3],
      [-1.5, 1, 1.2],
      [0.3, 1.2, -0.5],
    ]
    const r = nearestSpd(a)
    expect(symmetricEigen(r.matrix).values[2]).toBeGreaterThan(-1e-12)
    for (let s = 0; s < 30; s++) {
      const v = validCorrelation(s, 3)
      expect(r.distance).toBeLessThanOrEqual(frobeniusDistance(a, v) + 1e-12)
    }
    const c = validCorrelation(1, 4)
    expect(frobeniusDistance(nearestSpd(c).matrix, c)).toBeLessThan(1e-12)
  })

  it('raises the eigenvalues to a floor when asked for a definite matrix', () => {
    const r = nearestSpd(
      [
        [1, 1],
        [1, 1],
      ],
      { floor: 0.1 },
    )
    expect(r.minEigenvalue).toBe(0.1)
    expect(symmetricEigen(r.matrix).values[1]).toBeCloseTo(0.1, 12)
  })
})

describe('isCorrelationMatrix', () => {
  it('checks the diagonal, the symmetry and the eigenvalues', () => {
    expect(
      isCorrelationMatrix([
        [1, 0.5],
        [0.5, 1],
      ]),
    ).toBe(true)
    expect(
      isCorrelationMatrix([
        [1, 2],
        [2, 1],
      ]),
    ).toBe(false)
    expect(
      isCorrelationMatrix([
        [2, 0],
        [0, 1],
      ]),
    ).toBe(false)
    expect(
      isCorrelationMatrix([
        [1, 0.3],
        [0.5, 1],
      ]),
    ).toBe(false)
  })
})
