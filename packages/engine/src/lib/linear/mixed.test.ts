import { describe, expect, it } from 'vitest'
import { slogdet, solve } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, item, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  covarianceFactor,
  type MixedLayout,
  mixedAlgorithm,
  mixedModel,
  mixedProblemOf,
  mixedProfile,
} from './mixed.ts'

// Eight groups of five rows with one feature, a random intercept and a group-specific slope on it.
const groups = 8
const per = 5
const x: number[][] = []
const y: number[] = []
for (let g = 0; g < groups; g++)
  for (let t = 0; t < per; t++) {
    const f = Math.sin(g * 3.1 + t * 1.7) * 1.5
    x.push([g + 10, f])
    y.push(1 + 2 * f + 1.3 * Math.sin(g * 5.3) + 0.6 * Math.cos(g * 2.1) * f + 0.4 * Math.sin(g * 7.7 + t * 4.1))
  }
const layout: MixedLayout = { groupColumn: 0, fitIntercept: true, randomIntercept: true, randomSlopes: [0] }

/** The restricted log-likelihood by dense matrices: V = I + Z L L' Z' per group, no sufficient statistics. */
function denseDeviance(theta: number[], reml: boolean): number {
  const N = x.length
  const p = 2
  const L = covarianceFactor(theta, 2, 'unstructured')
  const xmx = new Float64Array(p * p)
  const xmy = new Float64Array(p)
  let ymy = 0
  let logDet = 0
  for (let g = 0; g < groups; g++) {
    const rows = x.slice(g * per, (g + 1) * per)
    const ys = y.slice(g * per, (g + 1) * per)
    const Z = rows.map((r) => [1, r[1] ?? 0])
    const X = rows.map((r) => [r[1] ?? 0, 1])
    const V = Array.from({ length: per }, (_, i) =>
      Array.from({ length: per }, (__, j) => {
        let s = i === j ? 1 : 0
        for (let a = 0; a < 2; a++)
          for (let b = 0; b < 2; b++) {
            let psi = 0
            for (let c = 0; c < 2; c++) psi += (L[a * 2 + c] ?? 0) * (L[b * 2 + c] ?? 0)
            s += (Z[i]?.[a] ?? 0) * psi * (Z[j]?.[b] ?? 0)
          }
        return s
      }),
    )
    const vt = fromStorage(Float64Array.from(V.flat()), [per, per], 'float64')
    logDet += item(slogdet(vt).logabsdet)
    const vinvX = toFlat(solve(vt, fromStorage(Float64Array.from(X.flat()), [per, 2], 'float64')))
    const vinvY = toFlat(solve(vt, fromStorage(Float64Array.from(ys), [per], 'float64')))
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++)
        for (let i = 0; i < per; i++)
          xmx[a * p + b] = (xmx[a * p + b] ?? 0) + (X[i]?.[a] ?? 0) * (vinvX[i * p + b] ?? 0)
      for (let i = 0; i < per; i++) xmy[a] = (xmy[a] ?? 0) + (X[i]?.[a] ?? 0) * (vinvY[i] ?? 0)
    }
    for (let i = 0; i < per; i++) ymy += (ys[i] ?? 0) * (vinvY[i] ?? 0)
  }
  const beta = toFlat(solve(fromStorage(xmx, [p, p], 'float64'), fromStorage(xmy, [p], 'float64')))
  const rss = ymy - (beta[0] ?? 0) * (xmy[0] ?? 0) - (beta[1] ?? 0) * (xmy[1] ?? 0)
  const dof = reml ? N - p : N
  const ld = reml ? item(slogdet(fromStorage(xmx, [p, p], 'float64')).logabsdet) : 0
  return dof * (Math.log(2 * Math.PI) + Math.log(rss / dof) + 1) + logDet + ld
}

describe('mixed model likelihood', () => {
  const problem = mixedProblemOf({ x, y }, layout, 'test')

  it('equals the dense marginal likelihood from the full covariance matrices', () => {
    for (const theta of [
      [1, 0.3, 0.8],
      [0.2, -0.5, 0.4],
      [2, 1, 0.1],
    ])
      for (const reml of [true, false]) {
        const prof = mixedProfile(problem, covarianceFactor(theta, 2, 'unstructured'), reml)
        expect(prof?.deviance).toBeCloseTo(denseDeviance(theta, reml), 8)
      }
  })

  it('finds a minimum of the profiled deviance', () => {
    const run = runAlgorithm(mixedAlgorithm(problem), { maxSteps: 100, key: key(0) })
    expect(run.status).toBe('converged')
    const best = run.state.deviance
    for (let j = 0; j < 3; j++)
      for (const d of [-1e-3, 1e-3]) {
        const t = [...run.state.theta]
        t[j] = (t[j] ?? 0) + d
        const prof = mixedProfile(problem, covarianceFactor(t, 2, 'unstructured'))
        expect(prof?.deviance ?? 0).toBeGreaterThan(best - 1e-9)
      }
  })

  it('has a diagonal covariance that fits no better than an unstructured one', () => {
    const full = runAlgorithm(mixedAlgorithm(problem), { maxSteps: 100 }).state.deviance
    const diag = runAlgorithm(mixedAlgorithm(problem, { covariance: 'diagonal' }), { maxSteps: 100 }).state.deviance
    expect(diag).toBeGreaterThanOrEqual(full - 1e-9)
  })

  it('resumes from a serialised state', () => {
    const alg = mixedAlgorithm(problem)
    const full = runAlgorithm(alg, { maxSteps: 4, key: key(0) })
    const first = runAlgorithm(alg, { maxSteps: 2, key: key(0) })
    const rest = runAlgorithm(alg, {
      maxSteps: 2,
      key: key(0),
      start: { state: alg.serialise?.(first.state), step: first.steps, serialised: true },
    })
    expect(rest.state.theta).toEqual(full.state.theta)
  })
})

describe('mixed model estimator', () => {
  it('recovers the balanced one-way ANOVA variance components in closed form', () => {
    // y_gt = mu + a_g + e_gt with 6 groups of 4: REML gives sigma2 = MSW and tau2 = (MSB - MSW) / 4.
    const k = 6
    const m = 4
    const xs: number[][] = []
    const ys: number[] = []
    for (let g = 0; g < k; g++)
      for (let t = 0; t < m; t++) {
        xs.push([g])
        ys.push(5 + 2 * Math.sin(g * 4.4) + 0.7 * Math.cos(g * 9.1 + t * 2.3))
      }
    const f = mixedModel().fit({ x: xs, y: ys })
    const means = Array.from({ length: k }, (_, g) => ys.slice(g * m, (g + 1) * m).reduce((a, b) => a + b, 0) / m)
    const grand = means.reduce((a, b) => a + b, 0) / k
    const msb = (m * means.reduce((a, v) => a + (v - grand) ** 2, 0)) / (k - 1)
    const msw = ys.reduce((a, v, i) => a + (v - (means[Math.floor(i / m)] ?? 0)) ** 2, 0) / (k * (m - 1))
    expect(f.params.sigma2).toBeCloseTo(msw, 6)
    expect(f.params.randomCovariance[0]?.[0]).toBeCloseTo((msb - msw) / m, 6)
    expect(f.params.intercept).toBeCloseTo(grand, 6)
  })

  it('shrinks the group effects and predicts groups it has seen', () => {
    const f = mixedModel({ randomSlopes: [0] }).fit({ x, y })
    expect(f.params.converged).toBe(true)
    expect(f.params.groupLabels).toEqual([10, 11, 12, 13, 14, 15, 16, 17])
    const fixed = toFlat(f.predictFixed(x))
    const cond = toFlat(f.predict(x))
    // Conditional predictions fit the data better than the population ones.
    const sse = (pred: ArrayLike<number>): number => y.reduce((a, v, i) => a + (v - (pred[i] ?? 0)) ** 2, 0)
    expect(sse(cond)).toBeLessThan(sse(fixed))
    // A group the model has not seen gets the population prediction.
    expect(toFlat(f.predict([[99, 0.5]]))[0]).toBeCloseTo(toFlat(f.predictFixed([[99, 0.5]]))[0] ?? 0, 12)
    // The random intercepts average to about zero.
    const mean = f.params.randomEffects.reduce((a, r) => a + (r[0] ?? 0), 0) / groups
    expect(Math.abs(mean)).toBeLessThan(0.5)
    expect(f.score({ x, y })).toBeGreaterThan(0.9)
    expect(f.params.aic).toBeCloseTo(-2 * f.params.logLikelihood + 2 * (2 + 3 + 1), 10)
  })

  it('gives a smaller residual variance by ML than by REML and refuses weights', () => {
    const reml = mixedModel({ randomSlopes: [0] }).fit({ x, y }).params
    const ml = mixedModel({ randomSlopes: [0], reml: false }).fit({ x, y }).params
    expect(ml.sigma2).toBeLessThan(reml.sigma2)
    expect(() => mixedModel().fit({ x, y, weights: y.map(() => 1) })).toThrow(/weights/)
  })

  it('reports a Wald table of the fixed effects', () => {
    const f = mixedModel({ randomSlopes: [0] }).fit({ x, y })
    const table = f.exposition[0]
    expect(table?.kind).toBe('coefficientTable')
    if (table?.kind === 'coefficientTable') expect(table.names).toEqual(['x1', 'intercept'])
  })
})
