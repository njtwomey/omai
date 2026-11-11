// The coverage figure is tested so that it only goes up: a rule that lands raises it and the test is moved up with it.
import { describe, expect, it } from 'vitest'
import { epAtoms } from './ep-atom-list.ts'
import { epCoverage, vmpCoverage } from './ep-coverage.ts'

describe('epCoverage', () => {
  const c = epCoverage()

  it('lists every atom and every benchmark model with a reason for each that does not run', () => {
    expect(c.atoms.length).toBe(epAtoms.length)
    expect(c.benchmarks.length).toBeGreaterThanOrEqual(27)
    for (const r of [...c.atoms, ...c.benchmarks])
      if (r.status === 'planned') expect(r.reason.length).toBeGreaterThan(10)
  })

  it('has every atom doing what it declares', () => {
    expect(c.atoms.filter((r) => r.status !== r.declared).map((r) => r.name)).toEqual([])
  })

  it('runs on at least 27 benchmark models (27 once the stick-breaking mixture and LDA-lite map; 25 once EP runs inside a grid over the length scale of the two Gaussian process regressions; 9 of 27 in the audit of 2026-11-11, 10 once stochastic volatility mapped, 23 with the likelihood registry, apply, the projected priors, gates, ordered constraints and the shared precisions; 46 of 47 atoms with the prior atlas), and only goes up; 92 of 92 atoms with the covariance atoms (six: Wishart precision, bivariate correlation, correlated random intercepts and slopes, each basic and learned); 86 of 86 atoms once the summed-out mixtures with a random share and precision map (84 before); 84 of 86 atoms once the two-variable sites, the multiclass site, ordinal probit, quantile regression and the draw margin landed (72 before)', () => {
    expect(c.benchmarksRun).toBeGreaterThanOrEqual(27)
    expect(c.atomsRun).toBeGreaterThanOrEqual(92)
  })
})

describe('vmpCoverage', () => {
  const c = vmpCoverage()

  it('lists every atom, with the operators that lack a variational rule and their reason for each that does not run', () => {
    expect(c.atoms.length).toBe(epAtoms.length)
    for (const r of c.atoms) if (r.status === 'planned') expect(r.reason.length).toBeGreaterThan(10)
  })

  it('runs on at least 78 of 92 atoms (36 before the numeric likelihoods, scales, shapes, softmax, deterministic functions, gamma products and gates had variational messages), 76 of them to a fixed point, and only goes up', () => {
    expect(c.atomsRun).toBeGreaterThanOrEqual(78)
    expect(c.atomsConverged).toBeGreaterThanOrEqual(76)
    // the two atoms with a correlation (`mvn-sd-corr`) run under EP only: the expected log of a beta correlation has no closed form
    expect(c.behindEp).toBeLessThanOrEqual(8)
  })

  it('refuses with a reason what has no mean-field message: a hard comparison, the Weibull shape, a custom function', () => {
    const reasons = Object.fromEntries(c.atoms.filter((r) => r.status === 'planned').map((r) => [r.name, r.reason]))
    expect(reasons['ordered-pair']).toMatch(/greater-than.*KL\(q \|\| p\) is infinite/)
    expect(reasons['weibull-shape-learned']).toMatch(/double exponential/)
    expect(reasons['gp-length-scale']).toMatch(/opaque/)
  })

  it('agrees with the reference posterior where the mean field can: means within half a standard deviation, spreads not above the reference', () => {
    const names = [
      'gaussian',
      'gamma-precision',
      'student-t',
      'laplace',
      'poisson-log',
      'exp',
      'log-scale',
      'softmax-regression',
      'ordinal-probit',
      'quantile-regression',
      'weibull-learned',
      'student-t-df-learned',
      'prior-beta-strength-gamma',
      'prior-beta-two-shapes',
      'gate-weight-summed-out',
      'stick-breaking',
    ]
    const accurate = vmpCoverage({ accuracy: true, names })
    expect(accurate.atoms.length).toBe(names.length)
    for (const r of accurate.atoms) {
      expect(r.status, r.name).toBe('runs')
      expect(r.meanZ, r.name).toBeLessThan(0.5)
      expect(r.sdRatio?.[1], r.name).toBeLessThan(1.2)
      expect(r.sdRatio?.[0], r.name).toBeGreaterThan(0.5)
    }
  }, 60000)
})
