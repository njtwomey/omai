import { describe, expect, it } from 'vitest'
import { optimiseThresholds, thresholdDecisions } from './post-processing.ts'

const scores = [0.1, 0.3, 0.45, 0.6, 0.8, 0.2, 0.35, 0.5, 0.7, 0.9]
const labels = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
const groups = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

describe('threshold optimisation', () => {
  it('a single group keeps its best threshold', () => {
    const s = [0.1, 0.3, 0.6, 0.8]
    const r = optimiseThresholds(s, [0, 0, 1, 1], [0, 0, 0, 0], { gridSize: 100 })
    expect(r.yBest).toBe(1)
    expect(thresholdDecisions(r.rules, s, [0, 0, 0, 0])).toEqual([0, 0, 1, 1])
  })

  it('equalises the constrained rate in expectation', () => {
    for (const constraint of ['demographicParity', 'truePositiveRateParity', 'falsePositiveRateParity'] as const) {
      const r = optimiseThresholds(scores, labels, groups, { constraint, gridSize: 200 })
      const d = thresholdDecisions(r.rules, scores, groups)
      const rate = (g: number): number => {
        const rows = groups.flatMap((v, i) =>
          v !== g
            ? []
            : constraint === 'demographicParity'
              ? [i]
              : constraint === 'truePositiveRateParity'
                ? labels[i] === 1
                  ? [i]
                  : []
                : labels[i] === 0
                  ? [i]
                  : [],
        )
        return rows.reduce((s, i) => s + (d[i] ?? 0), 0) / rows.length
      }
      expect(rate(0)).toBeCloseTo(rate(1), 9)
      expect(rate(0)).toBeCloseTo(r.xBest, 9)
    }
  })

  it('equalised odds meets at a shared point, mixing in a constant decision', () => {
    const r = optimiseThresholds(scores, labels, groups, { constraint: 'equalisedOdds', gridSize: 200 })
    const d = thresholdDecisions(r.rules, scores, groups)
    const rate = (g: number, y: number): number => {
      const rows = groups.flatMap((v, i) => (v === g && labels[i] === y ? [i] : []))
      return rows.reduce((s, i) => s + (d[i] ?? 0), 0) / rows.length
    }
    expect(rate(0, 1)).toBeCloseTo(rate(1, 1), 9)
    expect(rate(0, 0)).toBeCloseTo(rate(1, 0), 9)
    expect(r.rules.every((rule) => rule.pIgnore !== undefined && rule.pIgnore >= 0 && rule.pIgnore <= 1)).toBe(true)
    expect(r.rulesAt(0.5)).toHaveLength(2)
  })

  it('decisions are probabilities, and a degenerate group is refused', () => {
    const r = optimiseThresholds(scores, labels, groups)
    expect(thresholdDecisions(r.rules, scores, groups).every((p) => p >= 0 && p <= 1)).toBe(true)
    expect(() => optimiseThresholds([0.1, 0.2], [1, 1], [0, 0])).toThrow(/one class/)
  })
})
