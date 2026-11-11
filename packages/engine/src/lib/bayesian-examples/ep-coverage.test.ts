// The coverage figure is tested so that it only goes up: a rule that lands raises it and the test is moved up with it.
import { describe, expect, it } from 'vitest'
import { epAtoms } from './ep-atom-list.ts'
import { epCoverage } from './ep-coverage.ts'

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

  it('runs on at least the nine benchmark models (9 of 27 in the audit of 2026-11-11, 10 once stochastic volatility mapped), and only goes up', () => {
    expect(c.benchmarksRun).toBeGreaterThanOrEqual(10)
    expect(c.atomsRun).toBeGreaterThanOrEqual(19)
  })
})
