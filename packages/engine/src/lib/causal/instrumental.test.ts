import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { instrumentalVariables as ivGenerator } from '../datasets/index.ts'
import { key as makeKey, normal } from '../random/index.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { firstStageF, instrumentalVariables, liml, twoStageLeastSquares, waldEstimate } from './index.ts'

const study = (knobs: Record<string, number>, seed = 1) => {
  const g = generate(ivGenerator, { key: makeKey(seed), knobs: { n: 5000, effect: 1.5, ...knobs } })
  const m = toFlat(g.data.x)
  const n = m.length / 2
  return {
    z: Array.from({ length: n }, (_, i) => m[i * 2] ?? 0),
    d: Array.from({ length: n }, (_, i) => m[i * 2 + 1] ?? 0),
    y: Array.from(toFlat(g.data.y)),
    g,
  }
}

describe('instrumental variables on the planted generator', () => {
  const { z, d, y, g } = study({ strength: 1, confounding: 1 })
  const bias = (g.truth.extra?.['olsBias'] ?? 0) as number

  it('least squares carries the planted bias and two-stage least squares removes it', () => {
    const ols = instrumentalVariables({ outcome: y, endogenous: d, instruments: z }, { estimator: 'ols' })
    expect(Math.abs((ols.coefficients[0] ?? 0) - (1.5 + bias))).toBeLessThan(0.05)
    expect(bias).toBeGreaterThan(0.2)
    const iv = twoStageLeastSquares({ outcome: y, endogenous: d, instruments: z })
    expect(Math.abs((iv.coefficients[0] ?? 0) - 1.5)).toBeLessThan(0.1)
    expect(iv.effect.estimate.interval?.[0]).toBeLessThan(1.5)
    expect(iv.effect.estimate.interval?.[1]).toBeGreaterThan(1.5)
    expect(iv.firstStage[0]?.fStatistic).toBeGreaterThan(1000)
    expect(iv.firstStage[0]?.coefficients[0]).toBeCloseTo((g.truth.extra?.['firstStage'] ?? 0) as number, 1)
  })

  it('the Wald estimate with one instrument is the two-stage estimate, with a robust error', () => {
    const w = waldEstimate(y, d, z)
    const iv = twoStageLeastSquares({ outcome: y, endogenous: d, instruments: z })
    expect(w.estimate.value).toBeCloseTo(iv.coefficients[0] ?? 0, 8)
    expect(w.estimate.standardError).toBeGreaterThan(0)
    const robust = instrumentalVariables({ outcome: y, endogenous: d, instruments: z }, { covariance: 'HC0' })
    expect(w.estimate.standardError).toBeCloseTo(robust.standardErrors[0] ?? 0, 3)
  })

  it('a weak instrument shows in the first-stage F and in the width of the interval', () => {
    const weak = study({ strength: 0.08, confounding: 1 }, 2)
    const f = firstStageF({ outcome: weak.y, endogenous: weak.d, instruments: weak.z })
    expect(f).toBeLessThan(20)
    const strong = twoStageLeastSquares({ outcome: y, endogenous: d, instruments: z })
    const wk = twoStageLeastSquares({ outcome: weak.y, endogenous: weak.d, instruments: weak.z })
    expect((wk.standardErrors[0] ?? 0) / (strong.standardErrors[0] ?? 1)).toBeGreaterThan(5)
  })

  it('LIML equals two-stage least squares when exactly identified and Fuller shrinks kappa', () => {
    const a = twoStageLeastSquares({ outcome: y, endogenous: d, instruments: z })
    const b = liml({ outcome: y, endogenous: d, instruments: z })
    expect(b.kappa).toBeCloseTo(1, 8)
    expect(b.coefficients[0]).toBeCloseTo(a.coefficients[0] ?? 0, 8)
    const f = liml({ outcome: y, endogenous: d, instruments: z }, { fuller: 1 })
    expect(f.kappa).toBeLessThan(b.kappa)
  })

  it('Sargan accepts valid extra instruments and rejects an invalid one', () => {
    const extra = Array.from(toFlat(normal(makeKey(9), [z.length])), (v, i) => 0.5 * v + 0.5 * (z[i] ?? 0))
    const valid = instrumentalVariables({ outcome: y, endogenous: d, instruments: z.map((v, i) => [v, extra[i] ?? 0]) })
    expect(valid.sargan?.pValue).toBeGreaterThan(0.01)
    const conf = g.truth.latents?.['confounder']
    const u = toFlat(conf as Tensor)
    const invalid = instrumentalVariables({
      outcome: y,
      endogenous: d,
      instruments: z.map((v, i) => [v, (u[i] ?? 0) + 0.3 * v]),
    })
    expect(invalid.sargan?.pValue).toBeLessThan(0.01)
    expect(valid.sargan?.df).toBe(1)
  })

  it('with controls, an exogenous covariate enters both stages', () => {
    const w = Array.from(toFlat(normal(makeKey(10), [z.length])))
    const y2 = y.map((v, i) => v + 0.7 * (w[i] ?? 0))
    const r = instrumentalVariables({ outcome: y2, endogenous: d, instruments: z, exogenous: w.map((v) => [v]) })
    expect(r.names).toEqual(['endogenous0', 'exogenous0', 'intercept'])
    expect(Math.abs((r.coefficients[1] ?? 0) - 0.7)).toBeLessThan(0.08)
    expect(Math.abs((r.coefficients[0] ?? 0) - 1.5)).toBeLessThan(0.12)
  })
})
