/**
 * Tests for kernel expectile and quantile regression: the expectile at one half equal to kernel ridge, monotone in the
 * level, the solvers of the slot agreeing, stationarity of the objective; and the quantile fit's coverage, ordering and
 * sparsity on a planted heteroscedastic signal.
 */

import { describe, expect, it } from 'vitest'
import { pairwiseKernels } from '../kernels/index.ts'
import { flatObjective, smoothSolverConfig } from '../optim/index.ts'
import { toFlat } from '../tensor/index.ts'
import { normals, uniformRows } from './_testdata.ts'
import { kernelExpectileProblem, kernelExpectileRegression, kernelQuantileRegression, kernelRidge } from './index.ts'

const n = 60
const x = uniformRows(n, 1, 121, 0, 3)
const e = normals(n, 122)
// A planted signal with noise that grows with x, so the upper and lower levels fan out.
const y = x.map((r, i) => Math.sin(2 * (r[0] ?? 0)) + (0.1 + 0.25 * (r[0] ?? 0)) * (e[i] ?? 0))
const test = uniformRows(30, 1, 123, 0, 3)
const rbf = { kernel: 'rbf', gamma: 1.5 } as const
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)))
  return m
}

describe('kernel expectile regression', () => {
  it('is kernel ridge regression at one half', () => {
    const alpha = 0.3
    const fit = kernelExpectileRegression({ ...rbf, tau: 0.5, alpha }).fit({ x, y })
    const ridge = kernelRidge.with({ ...rbf, alpha }).fit({ x, y })
    expect(maxDiff(toFlat(fit.predict(test)), toFlat(ridge.predict(test)))).toBeLessThan(1e-8)
  })

  it('is monotone in the level: higher expectiles sit above lower ones', () => {
    const at = (tau: number): Float64Array =>
      Float64Array.from(
        toFlat(
          kernelExpectileRegression({ ...rbf, tau, alpha: 0.3 })
            .fit({ x, y })
            .predict(test),
        ),
      )
    const lo = at(0.1)
    const mid = at(0.5)
    const hi = at(0.9)
    for (let i = 0; i < 30; i++) {
      expect(mid[i] ?? 0).toBeGreaterThan((lo[i] ?? 0) - 1e-9)
      expect(hi[i] ?? 0).toBeGreaterThan((mid[i] ?? 0) - 1e-9)
    }
    expect(hi.reduce((a, b) => a + b, 0)).toBeGreaterThan(lo.reduce((a, b) => a + b, 0) + 1)
  })

  it('stops where the gradient of the objective vanishes', () => {
    const fit = kernelExpectileRegression({ ...rbf, tau: 0.8, alpha: 0.3 }).fit({ x, y })
    const k = pairwiseKernels(x, undefined, 'rbf', { gamma: 1.5 })
    const problem = kernelExpectileProblem(k, y, 0.8, 0.3)
    const g = flatObjective(problem.objective, problem.x0).valueAndGrad(fit.params.coef).grad
    expect(Math.max(...Array.from(g).map(Math.abs))).toBeLessThan(1e-7)
  })

  for (const solver of [{ kind: 'newton' }, { kind: 'lbfgs', ftol: 1e-16 }, { kind: 'bfgs' }] as const) {
    it(`reaches the IRLS optimum with ${solver.kind}`, () => {
      const small = { x: x.slice(0, 25), y: y.slice(0, 25) }
      const base = { ...rbf, tau: 0.8, alpha: 0.3 }
      const irls = kernelExpectileRegression(base).fit(small)
      const other = kernelExpectileRegression({
        ...base,
        solver: smoothSolverConfig({ ...solver, gradTol: 1e-8 }),
        maxIter: 300,
      }).fit(small)
      expect(maxDiff(toFlat(other.predict(test)), toFlat(irls.predict(test)))).toBeLessThan(1e-3)
    })
  }

  it('refuses a level outside (0, 1) and a non-positive penalty', () => {
    expect(() => kernelExpectileRegression({ tau: 1 }).fit({ x, y })).toThrow()
    expect(() => kernelExpectileRegression({ alpha: 0 }).fit({ x, y })).toThrow()
  })
})

describe('sparse kernel quantile regression', () => {
  const fitAt = (quantile: number, alpha = 0.002) =>
    kernelQuantileRegression(x, y, { kernel: { ...rbf, degree: 3, coef0: 1 }, quantile, alpha })

  it('has in-sample coverage near the level', () => {
    for (const tau of [0.1, 0.5, 0.9]) {
      const f = toFlat(fitAt(tau).predict(x))
      const below = y.filter((v, i) => v <= (f[i] ?? 0) + 1e-9).length / n
      expect(Math.abs(below - tau)).toBeLessThan(0.15)
    }
  })

  it('orders the quantiles at the test points and fans out with the noise', () => {
    const lo = toFlat(fitAt(0.1).predict(test))
    const mid = toFlat(fitAt(0.5).predict(test))
    const hi = toFlat(fitAt(0.9).predict(test))
    let ordered = 0
    for (let i = 0; i < 30; i++) ordered += (lo[i] ?? 0) <= (mid[i] ?? 0) && (mid[i] ?? 0) <= (hi[i] ?? 0) ? 1 : 0
    expect(ordered).toBeGreaterThanOrEqual(27)
    const width = (lo2: ArrayLike<number>, hi2: ArrayLike<number>, from: number, to: number): number => {
      let s = 0
      let c = 0
      test.forEach((r, i) => {
        if ((r[0] ?? 0) >= from && (r[0] ?? 0) < to) {
          s += (hi2[i] ?? 0) - (lo2[i] ?? 0)
          c++
        }
      })
      return s / c
    }
    expect(width(lo, hi, 2, 3)).toBeGreaterThan(width(lo, hi, 0, 1))
  })

  it('is sparser with a larger penalty, and the median resists an outlier', () => {
    // The support of an exact LP vertex is not monotone in the penalty (37, 32, 37, 17, 14, 24 at 0.05 to 0.0005); the
    // old count of 60 at 0.0005 was solver noise above 1e-9 (the objective was 1e-10 worse). Large against small is robust.
    expect(fitAt(0.5, 0.1).supportCount).toBeLessThan(fitAt(0.5, 0.002).supportCount)
    const line = Array.from({ length: 9 }, (_, i) => [i])
    const values = [0, 1, 2, 3, 4, 5, 6, 7, 400]
    const median = kernelQuantileRegression(line, values, {
      kernel: { kernel: 'linear', gamma: 1, degree: 1, coef0: 0 },
      alpha: 0,
    })
    expect(toFlat(median.predict([[4]]))[0]).toBeCloseTo(4, 6)
  })

  it('refuses a quantile outside (0, 1)', () => {
    expect(() => kernelQuantileRegression(x, y, { quantile: 1 })).toThrow()
  })
})
