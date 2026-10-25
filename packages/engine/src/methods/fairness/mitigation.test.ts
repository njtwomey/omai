import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { biasedCredit } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import type { Tensor } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { fairnessAudit } from './audit.ts'
import { exponentiatedGradient, gridSearchReduction } from './in-processing.ts'
import { thresholdOptimizer } from './post-processing.ts'
import { disparateImpactRemover, reweighedFit, reweighing } from './preprocessing.ts'
import { fairnessFrontier, impossibilityStudy } from './studies.ts'

const data = generate(biasedCredit, { key: key(2), knobs: { n: 600, labelBias: 0.4 } }).data
const dp = (r: { disparities: readonly { name: string; value: number }[] }, name: string): number =>
  r.disparities.find((d) => d.name === name)?.value ?? Number.NaN

describe('post-processing', () => {
  const model = logisticRegression().fit(data)

  it('equalises selection rates in expectation (demographic parity)', () => {
    const t = thresholdOptimizer(model, data)
    const [before, after] = recordsOf(t.exposition, 'groupReport')
    expect(dp(before ?? { disparities: [] }, 'demographic parity difference')).toBeGreaterThan(0.1)
    expect(dp(after ?? { disparities: [] }, 'demographic parity difference')).toBeLessThan(0.01)
    expect(t.rules).toHaveLength(2)
    expect(t.decisions.every((p) => p >= 0 && p <= 1)).toBe(true)
    expect(recordsOf(t.exposition, 'curves')[0]?.marks?.[0]?.x).toBe(t.xBest)
  })

  it('equalises true- and false-positive rates in expectation (equalised odds)', () => {
    const t = thresholdOptimizer(model, data, { constraint: 'equalisedOdds' })
    const after = recordsOf(t.exposition, 'groupReport')[1]
    expect(dp(after ?? { disparities: [] }, 'equalised odds difference')).toBeLessThan(0.01)
    expect(t.predictProbabilities([0.9, 0.1], [0, 1])).toHaveLength(2)
  })
})

describe('in-processing', () => {
  it('exponentiated gradient trades a little error for a much smaller violation, traced', () => {
    const r = exponentiatedGradient(data, { iterations: 12, epsilon: 0.02, differenceBound: 0.02 })
    // fairlearn's stopping rule may end the game early, once the duality gap is small.
    expect(r.steps.length).toBeGreaterThan(5)
    expect(r.steps.length).toBeLessThanOrEqual(12)
    const last = r.steps[r.steps.length - 1]
    expect(last?.averageViolation).toBeLessThan(0.05)
    expect(last?.averageError).toBeGreaterThanOrEqual(r.baseline.error - 0.02)
    // A modest price for parity, not a coin flip.
    expect(last?.averageError).toBeLessThan(r.baseline.error + 0.1)
    expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
    expect(recordsOf(r.exposition, 'curves')).toHaveLength(2)
    expect(recordsOf(r.exposition, 'groupReport')).toHaveLength(2)
  })

  it('equalised odds moments work too', () => {
    const r = exponentiatedGradient(data, { iterations: 6, constraint: 'equalisedOdds' })
    expect(r.steps.at(-1)?.averageViolation).toBeLessThan(r.baseline.violation)
  })

  it('a grid of multipliers traces error against violation', () => {
    const g = gridSearchReduction(data, { gridSize: 5 })
    expect(g.points).toHaveLength(5)
    const v = g.points.map((p) => p.violation)
    expect(Math.min(...v)).toBeLessThan(Math.max(...v))
  })
})

describe('preprocessing', () => {
  it('reweighing makes the label independent of the group in the weighted data', () => {
    const r = reweighing(data)
    const w = r.weights
    const y = Array.from(data.y.data)
    const x = Array.from(data.x.data)
    const group = y.map((_, i) => x[4 * i + 3] ?? 0)
    const rate = (a: number): number => {
      let num = 0
      let den = 0
      for (const [i, g] of group.entries())
        if (g === a) {
          den += w[i] ?? 0
          num += (w[i] ?? 0) * (y[i] ?? 0)
        }
      return num / den
    }
    expect(rate(0)).toBeCloseTo(rate(1), 12)
    expect(r.cells).toHaveLength(4)
    expect(recordsOf(r.exposition, 'coefficientTable')).toHaveLength(1)
    const fit = reweighedFit(data)
    const [plain, re] = recordsOf(fit.exposition, 'groupReport')
    expect(dp(re ?? { disparities: [] }, 'demographic parity difference')).toBeLessThan(
      dp(plain ?? { disparities: [] }, 'demographic parity difference'),
    )
  })

  it('disparate-impact removal aligns the groups’ feature distributions', () => {
    const r = disparateImpactRemover(data)
    const x = Array.from((r.data.x as Tensor).data as Float64Array)
    const group = Array.from({ length: x.length / 4 }, (_, i) => x[4 * i + 3] ?? 0)
    const median = (a: number, j: number): number => {
      const v = group.flatMap((g, i) => (g === a ? [x[4 * i + j] ?? 0] : [])).sort((p, q) => p - q)
      return v[Math.floor(v.length / 2)] ?? 0
    }
    for (const j of [0, 1, 2]) expect(Math.abs(median(0, j) - median(1, j))).toBeLessThan(0.1)
    expect(recordsOf(r.exposition, 'curves')[0]?.series).toHaveLength(4)
    const audit = fairnessAudit(logisticRegression().fit(r.data), r.data)
    expect(audit.disparities.demographicParityDifference).toBeTypeOf('number')
  })
})

describe('studies', () => {
  it('the frontier rises with the allowed violation', () => {
    const f = fairnessFrontier(data, { gridSize: 7, thresholds: 15 })
    expect(f.thresholds.length).toBeGreaterThan(1)
    const acc = f.thresholds.map((p) => p.accuracy)
    expect(acc).toEqual([...acc].sort((a, b) => a - b))
    expect(recordsOf(f.exposition, 'curves')[0]?.series).toHaveLength(2)
    const [front] = recordsOf(f.exposition, 'paretoFront')
    expect(front?.families).toHaveLength(2)
    expect(f.chosen.violation).toBeLessThanOrEqual(0.05)
  })

  it('calibrated scores with unequal base rates cannot also have equal error rates', () => {
    const s = impossibilityStudy({ gaps: [0, 1.5], n: 3000 })
    const [none, big] = s.points
    expect(big?.calibrationGap).toBeLessThan(0.08)
    expect(big?.equalisedOdds).toBeGreaterThan((none?.equalisedOdds ?? 0) + 0.1)
    expect(big?.equalisedOddsAfter).toBeLessThan(0.02)
    expect(big?.predictiveParityAfter).toBeGreaterThan(0.1)
  })
})
