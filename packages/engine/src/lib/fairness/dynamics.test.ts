import { describe, expect, it } from 'vitest'
import { delayedImpact, labelShiftSweep, outcomeCurve, repeatedDecisions, shiftedDisparities } from './dynamics.ts'

const model = { scores: [300, 450, 600, 750], repay: [0.1, 0.5, 0.8, 0.95] }
const groups = [
  { name: 'advantaged', share: 0.7, distribution: [0.1, 0.2, 0.3, 0.4] },
  { name: 'disadvantaged', share: 0.3, distribution: [0.4, 0.3, 0.2, 0.1] },
]

describe('dynamics of decisions', () => {
  it('outcome curves start at zero, rise and fall, and TPR grows with selection', () => {
    const c = outcomeCurve(model, groups[1]?.distribution ?? [], 21)
    expect(c.meanChange[0]).toBe(0)
    const peak = Math.max(...c.meanChange)
    expect(peak).toBeGreaterThan(0)
    expect(c.meanChange.at(-1)).toBeLessThan(peak)
    expect(c.truePositiveRate.every((t, i) => i === 0 || t >= (c.truePositiveRate[i - 1] ?? 0))).toBe(true)
    expect(() => outcomeCurve(model, [1], 3)).toThrow(/one probability/)
  })

  it('demographic parity selects both groups alike; the lender loses utility to it', () => {
    const [mu, dp, eo] = delayedImpact(model, groups, 201)
    expect(dp?.selection[0]).toBe(dp?.selection[1])
    expect(mu?.utility).toBeGreaterThanOrEqual((dp?.utility ?? 0) - 1e-12)
    expect(mu?.utility).toBeGreaterThanOrEqual((eo?.utility ?? 0) - 1e-12)
  })

  it('repeated decisions keep each distribution a distribution and record every round', () => {
    const r = repeatedDecisions(model, groups, { rounds: 4, steps: 51, policy: 'demographicParity' })
    expect(r.meanScore).toHaveLength(5)
    expect(r.selection).toHaveLength(4)
    for (const round of r.distributions) for (const d of round) expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
  })

  it('under label shift error-rate gaps stay put while selection gaps move; unit weights change nothing', () => {
    const labels = [1, 1, 0, 0, 1, 1, 0, 0]
    const decisions = [1, 0, 1, 0, 1, 1, 0, 0]
    const groups = [0, 0, 0, 0, 1, 1, 1, 1]
    const plain = shiftedDisparities(
      labels,
      decisions,
      groups,
      labels.map(() => 1),
    )
    const s = labelShiftSweep(labels, decisions, groups, { group: 1, baseRates: [0.5, 0.2, 0.6] })
    const named = (d: readonly { name: string; value: number }[], n: string): number =>
      d.find((x) => x.name === n)?.value ?? Number.NaN
    expect(named(s.disparities[0] ?? [], 'demographic parity difference')).toBeCloseTo(
      named(plain, 'demographic parity difference'),
      12,
    )
    for (const d of s.disparities) expect(named(d, 'equal opportunity difference')).toBeCloseTo(0.5, 12)
    expect(named(s.disparities[1] ?? [], 'demographic parity difference')).not.toBeCloseTo(
      named(s.disparities[2] ?? [], 'demographic parity difference'),
      3,
    )
    expect(() => labelShiftSweep([1, 1], [1, 1], [0, 0], { group: 0, baseRates: [0.5] })).toThrow(/both classes/)
  })
})
