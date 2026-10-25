import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { biasedCredit } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { baseRates, biasAmplification, labelBias, proxyScores, representation } from './data-bias.ts'

describe('bias in data', () => {
  const { data, truth } = generate(biasedCredit, {
    key: key(4),
    knobs: { n: 3000, labelBias: 0.4, proxyStrength: 2.5, minorityShare: 0.25 },
  })
  const x = toFlat(data.x)
  const rows = Array.from({ length: 3000 }, (_, i) => [x[4 * i] ?? 0, x[4 * i + 1] ?? 0, x[4 * i + 2] ?? 0])
  const group = Array.from({ length: 3000 }, (_, i) => x[4 * i + 3] ?? 0)
  const y = toFlat(data.y)

  it('sees under-representation and the base-rate gap', () => {
    const r = representation(group, { populationShares: { '0': 0.5, '1': 0.5 } })
    expect(r.shares[1]).toBeCloseTo(0.25, 1)
    expect(r.ratios?.[1]).toBeLessThan(0.6)
    expect(() => representation(group, { populationShares: { '0': 1 } })).toThrow(/population share/)
    const b = baseRates(y, group, { reference: 0 })
    expect(b.statisticalParityDifference).toBeLessThan(-0.15)
    expect(b.disparateImpact).toBeLessThan(0.8)
    expect(() => baseRates(y, group, { reference: 7 })).toThrow(/no group/)
  })

  it('measures the label bias the generator put in, in the minority only', () => {
    const clean = truth?.latents?.['cleanLabels'] as number[]
    const l = labelBias(y, clean, group)
    expect(l.positiveToNegative[0]).toBe(0)
    expect(l.positiveToNegative[1]).toBeCloseTo(0.4, 1)
    expect(l.negativeToPositive).toEqual([0, 0])
  })

  it('finds the postcode as the proxy, and the features together predict the group', () => {
    const p = proxyScores(rows, group, { featureNames: ['income', 'repayment', 'postcode'] })
    expect(p.target).toBe('1')
    const best = p.auc.indexOf(Math.max(...p.auc))
    expect(p.features[best]).toBe('postcode')
    expect(p.dependence[2]).toBeGreaterThan(p.dependence[0] ?? 1)
    expect(p.combinedAuc).toBeGreaterThan(0.85)
  })

  it('amplification is zero for a model copying the labels and has the sign of exaggeration', () => {
    expect(biasAmplification(y, y, group).value).toBeCloseTo(0, 12)
    const harsh = Array.from(y, (v, i) => (group[i] === 1 ? 0 : v))
    expect(biasAmplification(y, harsh, group).value).toBeGreaterThan(0)
  })
})
