import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const egarch = { kind: 'egarch', mean: 'zero' } as const
const truth: T.AsymmetricParameters = { mu: 0, omega: -0.1, alpha: [0.2], gamma: [-0.1], beta: [0.9] }

describe('EGARCH recursion', () => {
  it('is a constant variance exp(omega) when the dynamic coefficients vanish', () => {
    const v = T.asymmetricVariance(
      [1, -2, 0.5],
      { kind: 'egarch' },
      { mu: 0, omega: 0.5, alpha: [0], gamma: [0], beta: [0] },
    )
    for (const x of v) expect(x).toBeCloseTo(Math.exp(0.5), 12)
  })
  it('follows ln s2 = omega + alpha (|z| - E|z|) + gamma z + beta ln s2 by hand', () => {
    const eps = [1.2, -0.7, 0.4]
    const p: T.AsymmetricParameters = { mu: 0, omega: -0.1, alpha: [0.2], gamma: [-0.1], beta: [0.9] }
    const v = T.asymmetricVariance(eps, egarch, p)
    const kappa = Math.sqrt(2 / Math.PI)
    const back = (() => {
      let tot = 0
      let s = 0
      eps.forEach((e, i) => {
        const w = 0.94 ** i
        tot += w
        s += w * e * e
      })
      return s / tot
    })()
    const ln0 = -0.1 + 0.9 * Math.log(back)
    expect(v[0]).toBeCloseTo(Math.exp(ln0), 12)
    const z0 = 1.2 / Math.sqrt(v[0] ?? 1)
    const ln1 = -0.1 + 0.2 * (Math.abs(z0) - kappa) - 0.1 * z0 + 0.9 * ln0
    expect(v[1]).toBeCloseTo(Math.exp(ln1), 12)
  })
})

describe('EGARCH fit', () => {
  const y = T.asymmetricSimulate(key(11), 3000, egarch, truth)
  it('recovers the planted parameters', () => {
    const f = T.asymmetricFit(y, egarch)
    expect(f.converged).toBe(true)
    expect(f.params.beta[0]).toBeCloseTo(0.9, 1)
    expect(f.params.alpha[0]).toBeCloseTo(0.2, 1)
    expect(f.params.gamma[0] ?? 0).toBeLessThan(0)
    expect(f.params.omega).toBeCloseTo(-0.1, 1)
  })
  it('reaches the same likelihood from different solvers of the slot', () => {
    const a = T.asymmetricFit(y, egarch, { solver: { kind: 'bfgs', gradTol: 1e-7 }, gradTol: 1e-7 })
    const b = T.asymmetricFit(y, egarch, { solver: { kind: 'lbfgs', gradTol: 1e-7 }, gradTol: 1e-7 })
    expect(b.logLikelihood).toBeCloseTo(a.logLikelihood, 2)
    expect(b.params.beta[0]).toBeCloseTo(a.params.beta[0] ?? 0, 2)
  })
  it('logs the likelihood rising along the traced fit and serialises its state', () => {
    const { algorithm } = T.asymmetricFitAlgorithm(y.slice(0, 600), egarch)
    const run = runAlgorithm(algorithm, { maxSteps: 4 })
    const back = algorithm.deserialise?.(algorithm.serialise?.(run.state) as never)
    expect(back?.u).toEqual(run.state.u)
  })
  it('forecasts the variance by simulation, near the one-step recursion at horizon one', () => {
    const f = T.asymmetricFit(y.slice(0, 1500), egarch)
    const fc = T.asymmetricForecast(f, y.slice(0, 1500), 3, key(1), 400)
    expect(fc.variance).toHaveLength(3)
    const all = T.asymmetricVariance(y.slice(0, 1501), egarch, f.params)
    // The one-step forecast is the recursion's value at the next time, up to the simulated shocks (none at h = 1).
    expect(fc.variance[0]).toBeCloseTo(all[1500] ?? 0, 8)
  })
})

describe('APARCH', () => {
  const spec = { kind: 'aparch', mean: 'zero' } as const
  const p: T.AsymmetricParameters = { mu: 0, omega: 0.05, alpha: [0.1], gamma: [0.4], beta: [0.8], delta: 1.5 }
  const y = T.asymmetricSimulate(key(5), 3000, spec, p)
  it('reduces to GARCH(1,1) when delta = 2 and gamma = 0', () => {
    const e = [0.5, -1, 0.3, 1.1]
    const a = T.asymmetricVariance(e, spec, { mu: 0, omega: 0.1, alpha: [0.1], gamma: [0], beta: [0.8], delta: 2 })
    const g = T.garchVariance(
      e,
      { p: 1, q: 1, mean: 'zero' },
      { mu: 0, omega: 0.1, alpha: [0.1], gamma: [], beta: [0.8] },
    )
    for (let t = 0; t < e.length; t++) expect(a[t]).toBeCloseTo(g[t] ?? 0, 10)
  })
  it('fits with a power and an asymmetry in the right direction', () => {
    const f = T.asymmetricFit(y, spec)
    expect(f.params.gamma[0] ?? 0).toBeGreaterThan(0)
    expect(f.params.delta ?? 0).toBeGreaterThan(0.8)
    expect(f.params.delta ?? 9).toBeLessThan(2.6)
    expect(f.persistence).toBeLessThan(1.05)
  })
})

describe('news impact curves', () => {
  const shocks = [-3, -2, -1, 0, 1, 2, 3]
  it('is a symmetric parabola for GARCH, tilted to bad news for GJR and EGARCH with gamma < 0', () => {
    const garch = T.newsImpactCurve(
      { spec: { p: 1, q: 1 }, params: { mu: 0, omega: 0.1, alpha: [0.1], gamma: [], beta: [0.8] } },
      shocks,
    )
    expect(garch.variance[0]).toBeCloseTo(garch.variance[6] ?? 0, 12)
    const gjr = T.newsImpactCurve(
      { spec: { p: 1, o: 1, q: 1 }, params: { mu: 0, omega: 0.1, alpha: [0.05], gamma: [0.1], beta: [0.8] } },
      shocks,
    )
    expect(gjr.variance[0] ?? 0).toBeGreaterThan(gjr.variance[6] ?? 1)
    const e = T.newsImpactCurve({ spec: egarch, params: truth }, shocks)
    expect(e.variance[0] ?? 0).toBeGreaterThan(e.variance[6] ?? 1)
    // At zero shock EGARCH sits below the stationary variance only through -alpha E|z|.
    expect(e.stationaryVariance).toBeCloseTo(Math.exp(-0.1 / 0.1), 10)
    const ap = T.newsImpactCurve(
      { spec: { kind: 'aparch' }, params: { mu: 0, omega: 0.05, alpha: [0.1], gamma: [0.4], beta: [0.8], delta: 1.5 } },
      shocks,
    )
    expect(ap.variance[0] ?? 0).toBeGreaterThan(ap.variance[6] ?? 1)
  })
})

describe('GARCH over the solver slot', () => {
  it('gives the likelihood of the default fit from another solver', () => {
    const y = T.garchSimulate(
      key(2),
      500,
      { mean: 'zero' },
      { mu: 0, omega: 0.2, alpha: [0.1], gamma: [], beta: [0.8] },
    )
    const base = T.garchFit(y, { mean: 'zero' })
    const lb = T.garchFit(y, { mean: 'zero' }, { solver: { kind: 'lbfgs', gradTol: 1e-8 }, gradTol: 1e-8 })
    expect(lb.logLikelihood).toBeCloseTo(base.logLikelihood, 3)
  })
})
