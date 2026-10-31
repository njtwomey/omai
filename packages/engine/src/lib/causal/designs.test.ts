import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { panelEffects, syntheticControlPanel } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  differenceInDifferences,
  eventStudy,
  placeboInSpace,
  placeboInSpaceAlgorithm,
  splitPanel,
  syntheticControl,
  twoWayFixedEffects,
  wideOutcomes,
} from './index.ts'

const panelOf = (knobs: Record<string, number | string>, seed = 1) => {
  const g = generate(panelEffects, { key: key(seed), knobs: { units: 120, periods: 10, ...knobs } })
  return { panel: splitPanel(g.data), truth: g.truth }
}

describe('difference in differences', () => {
  it('recovers the effect when trends are parallel and is biased when they are not', () => {
    const ok = panelOf({ effect: 2, selection: 0.5, trend: 0, noise: 0.5 })
    const did = differenceInDifferences(ok.panel)
    expect(Math.abs(did.estimate.value - 2)).toBeLessThan(0.4)
    expect(did.estimate.interval?.[0]).toBeLessThan(2)
    expect(did.estimate.interval?.[1]).toBeGreaterThan(2)
    const broken = panelOf({ effect: 2, trend: 0.4, noise: 0.5 })
    const bad = twoWayFixedEffects(broken.panel)
    expect(bad.estimate.value - 2).toBeGreaterThan(0.8)
  })

  it('the two-by-two and the fixed-effects regression agree in a balanced simultaneous design', () => {
    const { panel } = panelOf({ effect: 1.5, noise: 0.3 }, 3)
    const post = { ...panel }
    const a = differenceInDifferences(post)
    const b = twoWayFixedEffects(post)
    // With one adoption date the two are the same regression up to the pre-period averaging.
    expect(a.estimate.value).toBeCloseTo(b.estimate.value, 1)
  })

  it('the hand-computed two-by-two', () => {
    const panel = {
      unit: [0, 0, 1, 1, 2, 2],
      period: [0, 1, 0, 1, 0, 1],
      treated: [0, 1, 0, 1, 0, 0],
      outcome: [1, 4, 2, 5, 1, 2],
    }
    const r = differenceInDifferences(panel)
    expect(r.estimate.value).toBeCloseTo(2, 12)
    expect(r.means.treatedPost).toBeCloseTo(4.5, 12)
  })
})

describe('event study', () => {
  it('leads are zero under parallel trends and the pre-trend test rejects a planted pre-trend', () => {
    const ok = eventStudy(panelOf({ effect: 2, noise: 0.5, trend: 0 }).panel, { leads: 3, lags: 3 })
    expect(ok.pretrend?.pValue).toBeGreaterThan(0.01)
    expect(ok.coefficients[ok.relativeTimes.indexOf(0)]).toBeGreaterThan(1.2)
    expect(ok.coefficients[ok.relativeTimes.indexOf(-1)]).toBe(0)
    const bad = eventStudy(panelOf({ effect: 2, noise: 0.3, trend: 0.5 }).panel, { leads: 3, lags: 3 })
    expect(bad.pretrend?.pValue).toBeLessThan(0.01)
  })

  it('tracks effects that grow with time since adoption', () => {
    const { panel, truth } = panelOf({ effect: 1, growth: 1, noise: 0.3, selection: 0 }, 5)
    const es = eventStudy(panel, { leads: 2, lags: 4 })
    const planted = (truth.extra?.['eventTimeEffects'] ?? {}) as Record<string, number>
    for (const e of [0, 2, 4]) {
      const j = es.relativeTimes.indexOf(e)
      expect(Math.abs((es.coefficients[j] ?? 0) - (planted[String(e)] ?? 0))).toBeLessThan(0.6)
    }
  })

  it('under staggered adoption with growing effects the static fixed-effects estimate is biased; the event studies are not', () => {
    const { panel, truth } = panelOf(
      { effect: 1, growth: 1, timing: 'staggered', noise: 0.3, selection: 0, units: 200, periods: 12, adopters: 0.6 },
      8,
    )
    const att = truth.effects?.att ?? 0
    expect(Math.abs(twoWayFixedEffects(panel).estimate.value - att)).toBeGreaterThan(0.5)
    const cs = eventStudy(panel, { leads: 2, lags: 4, method: 'callaway-santanna', resamples: 100 })
    const tw = eventStudy(panel, { leads: 2, lags: 4 })
    for (const e of [0, 1, 2, 3, 4]) {
      const planted = 1 + e
      expect(Math.abs((cs.coefficients[cs.relativeTimes.indexOf(e)] ?? 0) - planted)).toBeLessThan(0.3)
      expect(Math.abs((tw.coefficients[tw.relativeTimes.indexOf(e)] ?? 0) - planted)).toBeLessThan(0.3)
    }
    expect(cs.pretrend?.pValue).toBeGreaterThan(0.01)
  })
})

describe('synthetic control', () => {
  const g = generate(syntheticControlPanel, {
    key: key(2),
    knobs: { donors: 15, periods: 30, start: 20, effect: 3, noise: 0.05 },
  })
  const wide = wideOutcomes(splitPanel(g.data))
  const t = 30
  const j = 15
  const treated = Float64Array.from({ length: t }, (_, s) => wide.matrix[s * 16] ?? 0)
  const donors = Float64Array.from({ length: t * j }, (_, q) => wide.matrix[Math.floor(q / j) * 16 + 1 + (q % j)] ?? 0)
  const data = { treated, donors, donorCount: j, start: 20 }
  const extra = g.truth.extra as { donorWeights: number[]; effectPath: number[] }

  it('recovers the planted weights, the counterfactual and the effect', () => {
    const r = syntheticControl(data)
    expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(Math.min(...r.weights)).toBeGreaterThanOrEqual(0)
    for (const [i, w] of r.weights.entries()) expect(Math.abs(w - (extra.donorWeights[i] ?? 0))).toBeLessThan(0.1)
    expect(Math.abs(r.effect - (g.truth.effects?.att ?? 0))).toBeLessThan(0.15)
    expect(r.preRmspe).toBeLessThan(0.15)
    expect(r.ratio).toBeGreaterThan(10)
  })

  it('the placebo test ranks the real effect first and the algorithm takes one donor per step', () => {
    const p = placeboInSpace(data)
    expect(p.rank).toBe(1)
    expect(p.pValue).toBeCloseTo(1 / 16, 12)
    const run = runAlgorithm(placeboInSpaceAlgorithm(data), { maxSteps: 40, key: key(0) })
    expect(run.steps).toBe(15)
    expect(run.state.ratios.length).toBe(15)
  })

  it('without an effect the real unit does not stand out', () => {
    const quiet = generate(syntheticControlPanel, {
      key: key(3),
      knobs: { donors: 15, periods: 30, start: 20, effect: 0, noise: 0.3 },
    })
    const w = wideOutcomes(splitPanel(quiet.data))
    const tr = Float64Array.from({ length: t }, (_, s) => w.matrix[s * 16] ?? 0)
    const dn = Float64Array.from({ length: t * j }, (_, q) => w.matrix[Math.floor(q / j) * 16 + 1 + (q % j)] ?? 0)
    expect(placeboInSpace({ treated: tr, donors: dn, donorCount: j, start: 20 }).pValue).toBeGreaterThan(0.1)
  })
})
