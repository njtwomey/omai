import { describe, expect, it } from 'vitest'
import { epCoverage } from './ep-coverage.ts'
import { epDirections, epGaps, epGapsRefusing, unclaimedRefusals } from './ep-gaps.ts'

describe('the gaps of the EP suite', () => {
  const coverage = epCoverage()
  const refused = [...coverage.atoms, ...coverage.benchmarks].filter((r) => r.status !== 'runs')

  it('claim every setup and benchmark model EP refuses, so nothing is orphaned (whatever refuses today)', () => {
    expect(unclaimedRefusals(refused)).toEqual([])
  })

  it('derive what a gap refuses from the verdicts: a synthetic refusal is named by the gap whose marker its reason carries, and only by it', () => {
    const six = {
      name: 'ten-classes',
      reason: 'multiclass-site: 8 free scores are more than the six this rule integrates',
      missing: [],
    }
    const gaps = epGapsRefusing([six])
    expect(gaps.filter((g) => g.refuses.length > 0).map((g) => [g.id, g.refuses])).toEqual([
      ['softmax-many-classes', ['ten-classes']],
    ])
    // the marker is also read in the missing tags, not only the reason
    const tagged = { name: 'big-gp', reason: 'no', missing: ['scalar hyperparameters with known priors'] }
    expect(epGapsRefusing([tagged]).find((g) => g.id === 'grid-hyperparameters')?.refuses).toEqual(['big-gp'])
  })

  it('see a refusal nobody has written down: an item whose reason no gap carries is unclaimed', () => {
    expect(
      unclaimedRefusals([{ name: 'new-model', reason: 'a reason no gap carries', missing: ['something'] }]),
    ).toEqual(['new-model'])
  })

  it('the exported list names nothing: `refuses` is derived from the coverage, never kept', () => {
    expect(epGaps.every((g) => g.refuses.length === 0)).toBe(true)
  })

  it('every gap says why, how it would be closed and how big that is, in prose without counts or dates', () => {
    expect(new Set(epGaps.map((g) => g.id)).size).toBe(epGaps.length)
    for (const g of epGaps) {
      expect(g.summary.length, g.id).toBeGreaterThan(40)
      expect(g.plan.length, g.id).toBeGreaterThan(40)
      expect(g.summary + g.plan, g.id).not.toMatch(/\b20\d\d-\d\d-\d\d\b|agent-pp/)
    }
  })

  it('a gap that stands open does not begin by saying it is closed', () => {
    for (const g of epGaps) if (g.status !== 'queued') expect(g.summary, g.id).not.toMatch(/^Closed/)
  })

  it('the wider suite has distinct directions, each with what it is and what it needs', () => {
    expect(new Set(epDirections.map((d) => d.id)).size).toBe(epDirections.length)
    for (const d of epDirections) {
      expect(d.what.length, d.id).toBeGreaterThan(30)
      expect(d.needs.length, d.id).toBeGreaterThan(30)
    }
  })
})
