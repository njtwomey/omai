import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { discontinuityDesign } from '../datasets/index.ts'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { densityTest, imbensKalyanaraman, rdBandwidthSweep, regressionDiscontinuity } from './index.ts'

const design = (knobs: Record<string, number | string>, seed = 1) => {
  const g = generate(discontinuityDesign, { key: key(seed), knobs: { n: 6000, ...knobs } })
  const m = toFlat(g.data.x)
  const running = Array.from({ length: m.length / 2 }, (_, i) => m[i * 2] ?? 0)
  const treated = Array.from({ length: m.length / 2 }, (_, i) => m[i * 2 + 1] ?? 0)
  return { running, treated, outcome: Array.from(toFlat(g.data.y)), g }
}

/** A jump of 1 at zero with curvature `c` on the right side only, so wide windows are biased. */
const asymmetric = (c: number, n = 6000) => {
  const z = toFlat(normal(key(11), [2 * n]))
  const running = Array.from({ length: n }, (_, i) => (z[i] ?? 0) * 2)
  const outcome = running.map((v, i) => 1 + 0.5 * v + (v >= 0 ? 1 + c * v * v : 0) + 0.5 * (z[n + i] ?? 0))
  return { running, outcome }
}

describe('regression discontinuity', () => {
  it('a sharp design recovers the jump where the naive difference of means does not', () => {
    const { running, outcome } = design({ effect: 2, slope: 1.5, curvature: 0.3 })
    const above = outcome.filter((_, i) => (running[i] ?? 0) >= 0)
    const below = outcome.filter((_, i) => (running[i] ?? 0) < 0)
    const naive = above.reduce((a, b) => a + b, 0) / above.length - below.reduce((a, b) => a + b, 0) / below.length
    expect(Math.abs(naive - 2)).toBeGreaterThan(1.5)
    const r = regressionDiscontinuity({ running, outcome })
    expect(Math.abs(r.effect.estimate.value - 2)).toBeLessThan(0.35)
    expect(r.effect.estimate.interval?.[0]).toBeLessThan(2)
    expect(r.effect.estimate.interval?.[1]).toBeGreaterThan(2)
    expect(r.bandwidth).toBeGreaterThan(0.1)
    expect(r.bandwidth).toBeLessThan(6)
  })

  it('a kink in the outcome function is not mistaken for a jump', () => {
    const { running, outcome } = design({ effect: 0, slopeChange: 2, curvature: 0 })
    expect(
      Math.abs(regressionDiscontinuity({ running, outcome }, { bandwidth: 1.5 }).effect.estimate.value),
    ).toBeLessThan(0.35)
  })

  it('a fuzzy design divides the jump by the jump in treatment', () => {
    const { running, treated, outcome } = design({ effect: 3, design: 'fuzzy', compliance: 0.7, slope: 1 }, 2)
    const r = regressionDiscontinuity({ running, outcome, treatment: treated }, { design: 'fuzzy', bandwidth: 1.5 })
    expect(Math.abs(r.effect.estimate.value - 3)).toBeLessThan(1.2)
    expect(r.firstStageJump).toBeGreaterThan(0.5)
    expect(r.firstStageJump).toBeLessThan(0.9)
    // The sharp estimate on the same data is the intention-to-treat effect: the effect times the first stage.
    const itt = regressionDiscontinuity({ running, outcome }, { bandwidth: 1.5 }).effect.estimate.value
    expect(Math.abs(itt - 3 * (r.firstStageJump ?? 0))).toBeLessThan(0.7)
    expect(r.outcomeJump).toBeCloseTo(itt, 10)
  })

  it('the Imbens-Kalyanaraman bandwidth shrinks as the curvature differs more across the cutoff', () => {
    const flat = asymmetric(0.05)
    const curved = asymmetric(2)
    expect(imbensKalyanaraman(curved.running, curved.outcome)).toBeLessThan(
      imbensKalyanaraman(flat.running, flat.outcome),
    )
  })

  it('the sweep shows the bias of wide windows when the curvature differs across the cutoff', () => {
    const { running, outcome } = asymmetric(1.5)
    const sweep = rdBandwidthSweep({ running, outcome }, [0.4, 1, 2.5, 5], { order: 1 })
    const err = sweep.map((p) => Math.abs(p.estimate - 1))
    expect(err[3]).toBeGreaterThan(err[1] ?? 0)
    expect(sweep[0]?.standardError).toBeGreaterThan(sweep[3]?.standardError ?? 0)
    expect(sweep[0]?.n).toBeLessThan(sweep[3]?.n ?? 0)
  })

  it('a quadratic local fit removes that bias', () => {
    const { running, outcome } = asymmetric(1.5)
    const wide = regressionDiscontinuity({ running, outcome }, { bandwidth: 4, order: 2 })
    expect(Math.abs(wide.effect.estimate.value - 1)).toBeLessThan(0.3)
  })

  it('the hand-computed jump', () => {
    const running = Array.from({ length: 200 }, (_, i) => (i - 100) / 25)
    const outcome = running.map((v) => 1 + 0.5 * v + (v >= 0 ? 2 : 0))
    const r = regressionDiscontinuity({ running, outcome }, { bandwidth: 2, kernel: 'uniform' })
    expect(r.effect.estimate.value).toBeCloseTo(2, 10)
    expect(r.left.slope).toBeCloseTo(0.5, 10)
  })
})

describe('density test', () => {
  it('does not reject for a smooth running variable and rejects after bunching just above the cutoff', () => {
    const { running } = design({}, 6)
    expect(densityTest(running).pValue).toBeGreaterThan(0.01)
    const moved = running.map((v, i) => (v < 0 && v > -0.4 && i % 2 === 0 ? -v : v))
    expect(densityTest(moved).pValue).toBeLessThan(0.01)
    expect(densityTest(moved).logDifference).toBeGreaterThan(0)
  })
})
