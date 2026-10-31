import { describe, expect, it } from 'vitest'
import { inv } from '../numerics/linalg/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key, normal as randomNormal } from '../random/index.ts'
import { item, primalOf, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { matrixTensor } from './design.ts'
import { ar1Covariance, gls, glsar, glsarAlgorithm, glsFit, heteroscedasticRegressor } from './gls.ts'
import { olsSummary } from './inference.ts'

const frac = (v: number): number => v - Math.floor(v)
const normal = (i: number, a = 0.5698402909980532): number => item(ndtri(frac((i + 1) * a) * 0.998 + 0.001))

describe('gls', () => {
  it('builds the AR(1) covariance', () => {
    expect(ar1Covariance(3, 0.5)).toEqual([
      [1, 0.5, 0.25],
      [0.5, 1, 0.5],
      [0.25, 0.5, 1],
    ])
  })

  const n = 25
  const x = Array.from({ length: n }, (_, i) => [frac((i + 1) * 0.7548776662) * 3, Math.sin(i)])
  const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) - 0.5 * (r[1] ?? 0) + 0.4 * normal(i))

  it('is least squares for a scalar covariance and weighted least squares for a vector', () => {
    const ols = olsSummary({ x, y })
    const a = glsFit({ x, y }, { sigma: 3 })
    expect([...a.coef, a.intercept].map((v) => +v.toFixed(9))).toEqual(ols.table.estimates.map((v) => +v.toFixed(9)))
    const variances = x.map((r) => 0.5 + (r[0] ?? 0))
    const wls = olsSummary({ x, y, weights: variances.map((v) => 1 / v) })
    const b = glsFit({ x, y }, { sigma: variances })
    for (let j = 0; j < 3; j++) {
      expect([...b.coef, b.intercept][j]).toBeCloseTo(wls.table.estimates[j] ?? Number.NaN, 9)
      expect(Math.sqrt(b.covariance[j]?.[j] ?? Number.NaN)).toBeCloseTo(wls.table.standardErrors?.[j] ?? Number.NaN, 9)
    }
  })

  it('solves the generalised normal equations (X^T S^-1 X) b = X^T S^-1 y for a full covariance', () => {
    const sigma = ar1Covariance(n, 0.6)
    const f = glsFit({ x, y }, { sigma })
    const s = Float64Array.from(toFlat(primalOf(inv(matrixTensor(sigma.flat(), n, n)))))
    const p = 3
    const a = x.map((r) => [...r, 1])
    // gradient of the generalised sum of squares at the solution must vanish
    const beta = [...f.coef, f.intercept]
    const resid = y.map((v, i) => v - (a[i] ?? []).reduce((acc, aj, j) => acc + aj * (beta[j] ?? 0), 0))
    for (let j = 0; j < p; j++) {
      let g = 0
      for (let i = 0; i < n; i++)
        for (let k = 0; k < n; k++) g += (a[i]?.[j] ?? 0) * (s[i * n + k] ?? 0) * (resid[k] ?? 0)
      expect(Math.abs(g)).toBeLessThan(1e-9)
    }
  })

  it('rejects a covariance of the wrong size or one that is not positive definite', () => {
    expect(() => glsFit({ x, y }, { sigma: [1, 2] })).toThrow(/rows/)
    expect(() => glsFit({ x, y }, { sigma: Array.from({ length: n }, () => new Array<number>(n).fill(1)) })).toThrow(
      /positive definite/,
    )
  })

  it('predicts the mean and reports a coefficient table', () => {
    const f = gls({ sigma: 1 }).fit({ x, y })
    expect(toFlat(f.predict(x)).length).toBe(n)
    expect(f.exposition[0]?.kind).toBe('coefficientTable')
  })
})

describe('glsar', () => {
  const n = 300
  const xs = Array.from({ length: n }, (_, i) => i / 30)
  const noise = toFlat(randomNormal(key(7), [n]))
  const e: number[] = []
  const y = xs.map((v, i) => {
    e[i] = 0.8 * (e[i - 1] ?? 0) + 0.5 * (noise[i] ?? 0)
    return 1 + 2 * v + (e[i] ?? 0)
  })
  const x = xs.map((v) => [v])

  it('with one round and given coefficients is least squares on the whitened data', () => {
    const rho = 0.6
    const f = glsar({ order: 1, rho: [rho], maxIter: 1 }).fit({ x, y })
    // The same fit by hand: rows t >= 1 of (z_t - rho z_{t-1}), the intercept column whitened too.
    const wx = xs.slice(1).map((v, t) => [v - rho * (xs[t] ?? 0), 1 - rho])
    const wy = y.slice(1).map((v, t) => v - rho * (y[t] ?? 0))
    const by = olsSummary({ x: wx.map((r) => [r[0] ?? 0, r[1] ?? 0]), y: wy }, { fitIntercept: false })
    expect(f.params.coef[0]).toBeCloseTo(by.table.estimates[0] ?? Number.NaN, 9)
    expect(f.params.intercept).toBeCloseTo(by.table.estimates[1] ?? Number.NaN, 9)
    expect(f.params.rho).toEqual([rho])
  })

  it('with maxIter 1 and no start is least squares without the first row (statsmodels drops p rows)', () => {
    const f = glsar({ maxIter: 1 }).fit({ x, y })
    const ols = olsSummary({ x: x.slice(1), y: y.slice(1) })
    expect(f.params.coef[0]).toBeCloseTo(ols.table.estimates[0] ?? Number.NaN, 9)
  })

  it('recovers the planted autocorrelation and the line', () => {
    const f = glsar({ order: 1, maxIter: 12 }).fit({ x, y })
    // The Yule-Walker estimate has a standard error near 0.035 at this length.
    expect(Math.abs((f.params.rho[0] ?? 0) - 0.8)).toBeLessThan(0.1)
    expect(f.params.coef[0]).toBeCloseTo(2, 0)
  })

  it('stops early when the coefficients settle, and serialises its state', () => {
    const algorithm = glsarAlgorithm({ x, y }, { maxIter: 30, rtol: 1e-6 })
    const straight = runAlgorithm(algorithm, { key: key(0), maxSteps: 29 })
    expect(straight.steps).toBeLessThan(29)
    const head = runAlgorithm(algorithm, { key: key(0), maxSteps: 2 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      key: key(0),
      maxSteps: 29 - head.steps,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(resumed.state.rho).toEqual(straight.state.rho)
  })
})

describe('heteroscedasticRegressor', () => {
  const n = 600
  const xs = Array.from({ length: n }, (_, i) => frac((i + 1) * 0.7548776662) * 4)
  // The noise sd is 0.2 exp(0.5 x): the log-variance is 2 ln 0.2 + x.
  const y = xs.map((v, i) => 1 + 2 * v + 0.2 * Math.exp(0.5 * v) * normal(i))
  const x = xs.map((v) => [v])

  it('recovers the line and the log-variance slope by feasible GLS and by maximum likelihood', () => {
    for (const method of ['fgls', 'ml'] as const) {
      const f = heteroscedasticRegressor({ method }).fit({ x, y })
      expect(f.params.coef[0], method).toBeCloseTo(2, 1)
      expect(f.params.gamma[0], method).toBeCloseTo(1, 0)
      expect(f.params.converged, method).toBe(true)
    }
  })

  it('estimates the slope more precisely than ordinary least squares', () => {
    const f = heteroscedasticRegressor({ method: 'ml' }).fit({ x, y })
    const gen = Math.sqrt(f.params.covariance[0]?.[0] ?? Number.NaN)
    const ols = olsSummary({ x, y }, { covType: 'HC0' }).table.standardErrors?.[0] ?? Number.NaN
    expect(gen).toBeLessThan(ols)
  })

  it('maximum likelihood reaches one optimum with Newton, BFGS and L-BFGS', () => {
    const ref = heteroscedasticRegressor({ method: 'ml' }).fit({ x, y }).params
    for (const solver of [
      smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-10 }),
      smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-10, ftol: 1e-16 }),
    ]) {
      const p = heteroscedasticRegressor({ method: 'ml', solver, tol: 1e-8, maxIter: 3000 }).fit({ x, y }).params
      expect(p.coef[0]).toBeCloseTo(ref.coef[0] ?? Number.NaN, 4)
      expect(p.gamma[0]).toBeCloseTo(ref.gamma[0] ?? Number.NaN, 4)
      expect(p.logLikelihood).toBeCloseTo(ref.logLikelihood, 6)
    }
  })

  it('with a constant variance reduces to least squares', () => {
    const flatY = xs.map((v, i) => 1 + 2 * v + 0.3 * normal(i))
    const f = heteroscedasticRegressor({ method: 'ml', varianceFeatures: [] }).fit({ x, y: flatY })
    const ols = olsSummary({ x, y: flatY })
    expect(f.params.coef[0]).toBeCloseTo(ols.table.estimates[0] ?? Number.NaN, 7)
    expect(Math.exp(f.params.gamma[0] ?? 0)).toBeCloseTo(ols.residuals.reduce((a, r) => a + r * r, 0) / n, 7)
  })
})
