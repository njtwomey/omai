/**
 * Inference on extended-family fits, checked by simulation: the 95% pointwise intervals of the linear predictor (from the
 * Bayesian posterior covariance) cover the planted truth about 95% of the time over many replicates, the p-values of a
 * term with no effect are about uniform, and those of a real effect are small.
 */

import { describe, expect, it } from 'vitest'
import { beta as betaDist } from '../dists/continuous/index.ts'
import { families } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { extendedModel, extendedTermTests, fitExtended, linear, predictExtended, s } from './index.ts'

const logistic = (v: number): number => 1 / (1 + Math.exp(-v))
const truth = (t: number): number => -0.1 + 1.1 * Math.sin(2 * Math.PI * t)

function replicate(seed: number, n: number, effect: number, phi = 15): { x: number[][]; y: number[] } {
  const u = Array.from(toFlat(betaDist(tensor(new Array(n).fill(1)), tensor(new Array(n).fill(1))).sample(key(seed))))
  const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n, u[i] ?? 0.5])
  const mu = x.map((r) => logistic(0.2 + effect * truth(r[0] ?? 0)))
  const draw = betaDist(tensor(mu.map((m) => m * phi)), tensor(mu.map((m) => (1 - m) * phi))).sample(key(seed + 7919))
  return { x, y: Array.from(toFlat(draw), (v) => Math.min(Math.max(v, 1e-6), 1 - 1e-6)) }
}

describe('inference on beta regression fits', () => {
  it('has pointwise 95% intervals of the linear predictor that cover the truth about 95% of the time over 30 datasets', () => {
    const n = 120
    const grid = Array.from({ length: 15 }, (_, i) => [0.05 + (0.9 * i) / 14, 0.5])
    let hit = 0
    let total = 0
    for (let r = 0; r < 30; r++) {
      const { x, y } = replicate(100 + r, n, 1)
      const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 8 })]])
      const pred = predictExtended(model, fitExtended(model), grid)
      grid.forEach((g, i) => {
        const centre = pred.eta[0]?.[i] ?? 0
        const half = 1.96 * (pred.se[0]?.[i] ?? 0)
        const target = 0.2 + truth(g[0] ?? 0)
        total++
        if (Math.abs(centre - target) <= half) hit++
      })
    }
    // Bayesian intervals of a penalised fit cover on average over the curve a little under the nominal level.
    expect(hit / total).toBeGreaterThan(0.88)
    expect(hit / total).toBeLessThan(0.99)
  }, 300000)

  it('gives about uniform p-values to a smooth of noise and small ones to a real effect', () => {
    const n = 120
    const nulls: number[] = []
    for (let r = 0; r < 40; r++) {
      const { x, y } = replicate(500 + r, n, 0)
      const model = extendedModel(x, y, families.betaRegression(), [[s(1, { k: 6 }), linear(0)]])
      const tests = extendedTermTests(model, fitExtended(model))
      nulls.push(tests.find((t) => t.label === 's(x1)')?.pValue ?? Number.NaN)
    }
    const small = nulls.filter((p) => p < 0.05).length / nulls.length
    expect(small).toBeLessThan(0.15)
    expect(nulls.reduce((a, b) => a + b, 0) / nulls.length).toBeGreaterThan(0.35)
    expect(nulls.reduce((a, b) => a + b, 0) / nulls.length).toBeLessThan(0.65)
    for (let r = 0; r < 5; r++) {
      const { x, y } = replicate(900 + r, n, 1)
      const model = extendedModel(x, y, families.betaRegression(), [[s(0, { k: 6 })]])
      const t = extendedTermTests(model, fitExtended(model))[0]
      expect(t?.pValue ?? 1).toBeLessThan(1e-3)
      expect(t?.edf ?? 0).toBeGreaterThan(2)
    }
  }, 300000)
})
