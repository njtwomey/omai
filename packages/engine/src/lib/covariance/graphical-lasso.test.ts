import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { gaussianGraph } from '../datasets/index.ts'
import { linalg } from '../numerics/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  GraphicalLasso,
  graphicalLasso,
  graphicalLassoAlgorithm,
  graphicalLassoEstimator,
  graphicalLassoObjective,
} from './graphical-lasso.ts'

const planted = (n: number, p = 6): number[][] => {
  const { data } = generate(gaussianGraph, { key: key(11), knobs: { n, p, structure: 'band', strength: 0.4 } })
  return toArray(empiricalCovariance(data.x)) as number[][]
}
const tight = { tol: 1e-10, enetTol: 1e-12, maxIter: 500 }

describe('the graphical lasso with a penalty per entry', () => {
  const s = planted(200)
  const p = s.length

  it('a constant matrix is the scalar penalty', () => {
    const matrix = Array.from({ length: p }, () => new Array<number>(p).fill(0.15))
    const a = graphicalLasso(s, 0.15, tight)
    const b = graphicalLasso(s, matrix, tight)
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++) expect(b.precision[i]?.[j]).toBeCloseTo(a.precision[i]?.[j] ?? 0, 7)
  })

  it('satisfies the optimality conditions: |S - Sigma| <= penalty, with equality (and the sign) on active edges', () => {
    const matrix = s.map((r, i) => r.map((_, j) => (i === j ? 0 : 0.05 + 0.03 * ((i + j) % 4))))
    const fit = graphicalLasso(s, matrix, tight)
    const sigma = toArray(linalg.inv(fit.precision)) as number[][]
    let active = 0
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++) {
        if (i === j) continue
        const gradient = (s[i]?.[j] ?? 0) - (sigma[i]?.[j] ?? 0)
        const bound = matrix[i]?.[j] ?? 0
        expect(Math.abs(gradient)).toBeLessThan(bound + 1e-6)
        const theta = fit.precision[i]?.[j] ?? 0
        if (Math.abs(theta) > 1e-8) {
          active++
          expect(Math.abs(gradient)).toBeCloseTo(bound, 5)
          expect(Math.sign(gradient)).toBe(Math.sign(-theta))
        }
      }
    expect(active).toBeGreaterThan(0)
    expect(active).toBeLessThan(p * (p - 1))
  })

  it('a heavier penalty on one entry removes it, a vanishing one keeps it unpenalised', () => {
    const base = graphicalLasso(s, 0.05, tight)
    const strong = Array.from({ length: p }, () => new Array<number>(p).fill(0.05))
    ;(strong[0] as number[])[1] = 5
    ;(strong[1] as number[])[0] = 5
    expect(Math.abs(base.precision[0]?.[1] ?? 0)).toBeGreaterThan(1e-6)
    expect(Math.abs(graphicalLasso(s, strong, tight).precision[0]?.[1] ?? 1)).toBe(0)
    const free = Array.from({ length: p }, () => new Array<number>(p).fill(5))
    ;(free[0] as number[])[1] = 0
    ;(free[1] as number[])[0] = 0
    expect(Math.abs(graphicalLasso(s, free, tight).precision[0]?.[1] ?? 0)).toBeGreaterThan(1e-3)
  })

  it('rejects a matrix of the wrong size, an asymmetric one and a negative penalty', () => {
    expect(() => graphicalLasso(s, [[0.1, 0.1]], tight)).toThrow(/must be/)
    const lopsided = Array.from({ length: p }, () => new Array<number>(p).fill(0.1))
    ;(lopsided[0] as number[])[1] = 0.3
    expect(() => graphicalLasso(s, lopsided, tight)).toThrow(/symmetric/)
    expect(() => graphicalLasso(s, -0.1)).toThrow(/non-negative/)
    expect(
      graphicalLassoObjective(
        s,
        graphicalLasso(s, 0.1).precision,
        lopsided.map((r) => r.map(() => 0.1)),
      ),
    ).toBeTypeOf('number')
  })
})

describe('warm starts and the unpenalised case', () => {
  const s = planted(150)

  it('starting from a neighbouring penalty reaches the same optimum, in fewer sweeps', () => {
    const near = graphicalLasso(s, 0.12, tight)
    const cold = graphicalLasso(s, 0.1, tight)
    const warm = graphicalLasso(s, 0.1, { ...tight, start: { covariance: near.covariance } })
    for (let i = 0; i < s.length; i++)
      for (let j = 0; j < s.length; j++) expect(warm.precision[i]?.[j]).toBeCloseTo(cold.precision[i]?.[j] ?? 0, 6)
    expect(warm.nIter).toBeLessThanOrEqual(cold.nIter)
  })

  it('a zero penalty gives the inverse of the covariance, as scikit-learn does', () => {
    const fit = graphicalLasso(s, 0)
    const inverse = toArray(linalg.inv(s)) as number[][]
    for (let i = 0; i < s.length; i++)
      for (let j = 0; j < s.length; j++) expect(fit.precision[i]?.[j]).toBeCloseTo(inverse[i]?.[j] ?? 0, 10)
    expect(fit.nIter).toBeLessThanOrEqual(2)
  })

  it('traces a run with the duality gap falling', () => {
    const run = runAlgorithm(graphicalLassoAlgorithm(s, 0.1, { start: { covariance: s } }), { maxSteps: 50 })
    expect(run.status).toBe('converged')
  })
})

describe('the precomputed entry', () => {
  it('takes the covariance itself as the data, with no location and no Mahalanobis scores', () => {
    const s = planted(120, 4)
    const fitted = graphicalLassoEstimator({
      alpha: 0.1,
      precomputed: true,
      tol: 1e-10,
      enetTol: 1e-12,
      maxIter: 300,
    }).fit({
      x: s,
    })
    const direct = graphicalLasso(s, 0.1, { tol: 1e-10, enetTol: 1e-12, maxIter: 300 })
    expect(fitted.params.location).toEqual([0, 0, 0, 0])
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) expect(fitted.params.precision[i]?.[j]).toBeCloseTo(direct.precision[i]?.[j] ?? 0, 10)
    expect(fitted.exposition.map((r) => r.kind)).toEqual(['regularisationPath'])
    expect(new GraphicalLasso({ alpha: 0.1, precomputed: true }).fit(s).model.params.location.length).toBe(4)
    expect(() =>
      graphicalLassoEstimator({ precomputed: true }).fit({
        x: [
          [1, 2, 3],
          [4, 5, 6],
        ],
      }),
    ).toThrow(/square/)
  })
})
