import { describe, expect, it } from 'vitest'
import { epCoverage } from './ep-coverage.ts'
import { epGaps } from './ep-gaps.ts'

describe('the gaps of the EP suite', () => {
  const coverage = epCoverage()
  const rows = [...coverage.atoms, ...coverage.benchmarks]
  const named = new Set(epGaps.flatMap((g) => g.refuses))

  it('every setup and benchmark model EP refuses is named by a gap, so nothing is orphaned', () => {
    const orphans = rows.filter((r) => r.status !== 'runs' && !named.has(r.name)).map((r) => r.name)
    expect(orphans).toEqual([])
  })

  it('a gap names only items that exist and are still refused (a gap that has been closed is removed)', () => {
    const refused = new Set(rows.filter((r) => r.status !== 'runs').map((r) => r.name))
    const stale = [...named].filter((n) => !refused.has(n))
    expect(stale).toEqual([])
  })

  it('every gap says why, how it would be closed and how big that is', () => {
    expect(new Set(epGaps.map((g) => g.id)).size).toBe(epGaps.length)
    for (const g of epGaps) {
      expect(g.summary.length, g.id).toBeGreaterThan(40)
      expect(g.plan.length, g.id).toBeGreaterThan(40)
    }
  })
})
