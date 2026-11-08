import { analyseThresholds } from 'omai-engine/metrics'
import { describe, expect, it } from 'vitest'
import { stillOf, viewFor } from './registry.ts'
import {
  defaultThreshold,
  scoreRange,
  thresholdCostPanel,
  thresholdPrPanel,
  thresholdRocPanel,
  thresholdScoresPanel,
  thresholdSweepView,
} from './thresholds.ts'

type Opt = { grid: unknown[]; series: { name?: string; data?: unknown[] }[] }

const y = [0, 0, 0, 1, 0, 1, 1, 1]
const s = [0.1, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9]
const costs = [
  [0, 1],
  [5, 0],
]
const priced = analyseThresholds(y, s, { costs })
const plain = analyseThresholds(y, s)

describe('threshold sweep views', () => {
  it('draws scores, cost, ROC and PR in one figure, or three panels without costs', () => {
    expect((thresholdSweepView(priced) as unknown as Opt).grid).toHaveLength(4)
    expect((thresholdSweepView(plain) as unknown as Opt).grid).toHaveLength(3)
    expect(viewFor('thresholdSweep').component).toBe('ThresholdSweepView')
    expect((stillOf(priced) as unknown as Opt).grid).toHaveLength(4)
  })

  it('defaults to the cheapest threshold, else the most accurate', () => {
    expect(defaultThreshold(priced)).toBe(priced.thresholds[priced.cheapest ?? 0])
    expect(defaultThreshold(plain)).toBe(0.4)
    expect(scoreRange(plain)).toEqual([0, 1])
    const [lo, hi] = scoreRange(analyseThresholds([0, 1], [-2, 3]))
    expect(lo).toBeLessThan(-2)
    expect(hi).toBeGreaterThan(3)
  })

  it('moves the operating point and the cost marker with the threshold', () => {
    const point = (p: { layers: readonly { series: readonly unknown[] }[] }): unknown =>
      (p.layers.at(-1)?.series[0] as { data?: unknown[] } | undefined)?.data?.[0]
    expect(point(thresholdRocPanel(priced, 'light', 0.5))).not.toEqual(point(thresholdRocPanel(priced, 'light', 0.85)))
    expect(point(thresholdCostPanel(priced, 'light', { threshold: 0.2 }))).not.toEqual(
      point(thresholdCostPanel(priced, 'light', { threshold: 0.75 })),
    )
    expect(thresholdCostPanel(plain, 'light').layers).toHaveLength(0)
    // Nobody predicted positive has no precision: no operating point on the PR panel.
    expect(thresholdPrPanel(priced, 'light', 2).layers).toHaveLength(2)
  })

  it('fixes the score axis by the histograms, so a dragged threshold does not move it', () => {
    const a = thresholdScoresPanel(priced, 'light', { threshold: 0.2 })
    const b = thresholdScoresPanel(priced, 'light', { threshold: 0.8 })
    expect(a.top).toBe(b.top)
    expect(a.options?.y?.limits).toEqual([0, a.top])
    expect(thresholdScoresPanel(analyseThresholds(y, s, { keepRows: false }), 'light').layers).toHaveLength(1)
  })
})
