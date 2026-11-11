// The coverage is measured, never remembered: the invariants are that every atom and every benchmark model runs under EP (or says why
// not), and the regression net for the numbers themselves is the committed digest (`make ep-digest-check`), not a floor in a test.
import { describe, expect, it } from 'vitest'
import { benchmarkModels } from '../models/index.ts'
import { epAtoms } from './ep-atom-list.ts'
import { epCoverage, vmpCoverage } from './ep-coverage.ts'

describe('epCoverage', () => {
  const c = epCoverage()

  it('lists every atom and every benchmark model with a reason for each that does not run', () => {
    expect(c.atoms.length).toBe(epAtoms.length)
    expect(c.benchmarks.length).toBe(benchmarkModels().length)
    for (const r of [...c.atoms, ...c.benchmarks])
      if (r.status === 'planned') expect(r.reason.length).toBeGreaterThan(10)
  })

  it('has every atom doing what it declares', () => {
    expect(c.atoms.filter((r) => r.status !== r.declared).map((r) => r.name)).toEqual([])
  })

  it('runs every atom and every benchmark model, and says by which route a benchmark model runs', () => {
    expect(c.atomsRun).toBe(epAtoms.length)
    expect(c.benchmarksRun).toBe(c.benchmarks.length)
    for (const r of c.benchmarks) expect(['ep', 'ep-grid'], r.name).toContain(r.via)
  })
})

describe('vmpCoverage', () => {
  const c = vmpCoverage()

  it('lists every atom, with the operators that lack a variational rule and their reason for each that does not run', () => {
    expect(c.atoms.length).toBe(epAtoms.length)
    for (const r of c.atoms) if (r.status === 'planned') expect(r.reason.length).toBeGreaterThan(10)
  })

  it('counts consistently: the atoms it runs, those that settle, and those EP runs that it does not (the rule gaps)', () => {
    expect(c.atomsRun).toBe(c.atoms.filter((r) => r.status === 'runs').length)
    expect(c.atomsConverged).toBeLessThanOrEqual(c.atomsRun)
    expect(c.behindEp).toBe(c.atoms.filter((r) => r.epRuns && r.status !== 'runs').length)
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
