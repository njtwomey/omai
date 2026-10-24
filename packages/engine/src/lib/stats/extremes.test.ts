import { describe, expect, it } from 'vitest'
import { generalisedExtremeValue, generalisedPareto } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { fitGev, fitGpd, meanExcess, peaksOverThreshold, returnLevel } from './extremes.ts'

describe('extreme values', () => {
  it('fits GEV and GPD samples near their true parameters', () => {
    const f = fitGev(toFlat(generalisedExtremeValue(10, 2, 0.1).sample(key(21), [1500])))
    expect(f.params.loc).toBeCloseTo(10, 0)
    expect(f.params.scale).toBeCloseTo(2, 0)
    expect(Math.abs(f.params.shape - 0.1)).toBeLessThan(0.08)
    const rl = returnLevel(f, 50)
    expect(rl.interval?.[0]).toBeLessThan(rl.value)
    expect(rl.value).toBeGreaterThan(f.params.loc)
    const g = fitGpd(toFlat(generalisedPareto(0, 1, 0.2).sample(key(22), [1500])))
    expect(g.params.scale).toBeCloseTo(1, 0)
    expect(Math.abs(g.params.shape - 0.2)).toBeLessThan(0.1)
  })

  it('declusters peaks by runs', () => {
    const p = peaksOverThreshold([0, 5, 6, 0, 0, 7, 0, 8, 0], 4, { runLength: 2 })
    expect(p.indices).toEqual([2, 7])
    expect(p.rate).toBeCloseTo(2 / 9, 12)
  })

  it('has a flat mean excess equal to the scale for exponential tails', () => {
    const x = toFlat(generalisedPareto(0, 1, 0).sample(key(23), [20000]))
    for (const v of toFlat(meanExcess(x, { thresholds: [0, 1, 2] }).meanExcess)) expect(v).toBeCloseTo(1, 1)
  })
})
