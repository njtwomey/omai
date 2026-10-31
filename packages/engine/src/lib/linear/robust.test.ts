import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { designOf } from './design.ts'
import { huber, huberAlgorithm, quantileRegression, quantileRegressor, theilSen, theilSenAlgorithm } from './robust.ts'
import { leastSquaresSolve } from './solvers.ts'

const n = 30
const x = Array.from({ length: n }, (_, i) => [Math.sin(i * 1.9) * 2, Math.cos(i * 0.8)])
const clean = x.map((r, i) => 1 + 2 * (r[0] ?? 0) - (r[1] ?? 0) + 0.1 * Math.sin(i * 13.7))
const y = clean.map((v, i) => (i < 3 ? v + ([18, -15, 22][i] ?? 0) : v))

describe('Theil-Sen', () => {
  it('gives the median slope of a line with outliers', () => {
    const f = theilSen().fit({ x: [[0], [1], [2], [3], [4], [5]], y: [1, 3, 5, 7, 9, 100] })
    expect(f.params.coef[0]).toBeCloseTo(2, 2)
    expect(f.params.intercept).toBeCloseTo(1, 2)
    expect(f.params.converged).toBe(true)
  })

  it('resists outliers where least squares does not', () => {
    const f = theilSen().fit({ x, y })
    const ols = leastSquaresSolve(designOf({ x, y }, 'test'), 0, true, 'svd')
    const err = (c: ArrayLike<number>): number => Math.hypot((c[0] ?? 0) - 2, (c[1] ?? 0) + 1)
    expect(err(f.params.coef)).toBeLessThan(0.2)
    expect(err(ols.coef)).toBeGreaterThan(err(f.params.coef))
  })

  it('subsamples the subsets reproducibly from a key', () => {
    const design = designOf({ x, y }, 'test')
    const run = (k: number): number[] =>
      runAlgorithm(theilSenAlgorithm(design, { maxSubpopulation: 60 }, key(k)), { maxSteps: 300 }).state.median
    expect(run(1)).toEqual(run(1))
    const a = run(1)
    expect(Math.abs((a[1] ?? 0) - 2)).toBeLessThan(0.5)
  })

  it('refuses a subset size outside what the data allows and resumes from a serialised state', () => {
    const design = designOf({ x, y }, 'test')
    expect(() => theilSenAlgorithm(design, { nSubsamples: 1 })).toThrow(/nSubsamples/)
    const alg = theilSenAlgorithm(design)
    const full = runAlgorithm(alg, { maxSteps: 3 })
    const first = runAlgorithm(alg, { maxSteps: 1 })
    const rest = runAlgorithm(alg, {
      maxSteps: 2,
      start: { state: alg.serialise?.(first.state), step: first.steps, serialised: true },
    })
    expect(rest.state.median).toEqual(full.state.median)
  })
})

describe('Huber regression', () => {
  it('is least squares when the threshold is huge', () => {
    const f = huber({ epsilon: 1e6, alpha: 0, tol: 1e-12, maxIter: 5000 }).fit({ x, y })
    const ols = leastSquaresSolve(designOf({ x, y }, 'test'), 0, true, 'svd')
    f.params.coef.forEach((v, j) => {
      expect(v).toBeCloseTo(ols.coef[j] ?? 0, 6)
    })
  })

  it('resists outliers, flags them, and decreases its objective at every step', () => {
    const f = huber({ alpha: 0 }).fit({ x, y })
    expect(f.params.coef[0]).toBeCloseTo(2, 0)
    expect(f.params.outliers.slice(0, 3)).toEqual([true, true, true])
    expect(f.params.outliers.filter(Boolean).length).toBeLessThan(n / 2)
    const alg = huberAlgorithm(designOf({ x, y }, 'test'), { alpha: 0 })
    let s = alg.init(key(0))
    for (let t = 0; t < 15; t++) {
      const next = alg.step(s, { t } as never)
      expect(next.objective).toBeLessThanOrEqual(s.objective + 1e-10)
      s = next
    }
  })
})

describe('quantile regression', () => {
  const pinball = (c: { coef: number[]; intercept: number }, tau: number): number =>
    y.reduce((a, v, i) => {
      const r = v - c.intercept - (c.coef[0] ?? 0) * (x[i]?.[0] ?? 0) - (c.coef[1] ?? 0) * (x[i]?.[1] ?? 0)
      return a + (r >= 0 ? tau * r : (tau - 1) * r)
    }, 0)

  it('has the right share of residuals on each side and a loss no perturbation lowers', () => {
    for (const tau of [0.2, 0.5, 0.8]) {
      const f = quantileRegression({ x, y }, { quantile: tau })
      const res = y.map(
        (v, i) => v - f.intercept - (f.coef[0] ?? 0) * (x[i]?.[0] ?? 0) - (f.coef[1] ?? 0) * (x[i]?.[1] ?? 0),
      )
      const below = res.filter((r) => r < -1e-9).length / n
      const atOrBelow = res.filter((r) => r <= 1e-9).length / n
      expect(below).toBeLessThanOrEqual(tau + 1e-9)
      expect(atOrBelow).toBeGreaterThanOrEqual(tau - 1e-9)
      const best = pinball(f, tau)
      for (const d of [-1e-3, 1e-3])
        for (let j = 0; j < 3; j++) {
          const c = { coef: [...f.coef], intercept: f.intercept }
          if (j < 2) c.coef[j] = (c.coef[j] ?? 0) + d
          else c.intercept += d
          expect(pinball(c, tau)).toBeGreaterThanOrEqual(best - 1e-9)
        }
      expect(f.objective).toBeCloseTo(best / n, 9)
    }
  })

  it('orders the quantiles, shrinks with an L1 penalty and predicts', () => {
    const lo = quantileRegressor({ quantile: 0.1 }).fit({ x, y })
    const hi = quantileRegressor({ quantile: 0.9 }).fit({ x, y })
    expect(toFlat(lo.predict([[0, 0]]))[0]).toBeLessThan(toFlat(hi.predict([[0, 0]]))[0] ?? 0)
    const penalised = quantileRegression({ x, y }, { alpha: 0.5 })
    const plain = quantileRegression({ x, y })
    expect(Math.abs(penalised.coef[0] ?? 0)).toBeLessThan(Math.abs(plain.coef[0] ?? 0))
    expect(() => quantileRegression({ x, y }, { quantile: 1 })).toThrow(/quantile/)
  })
})
