import { describe, expect, it } from 'vitest'
import { bayesPredict, clusterAgreement, generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { sum, type Tensor, toFlat } from '../tensor/index.ts'
import { anisotropicBlobs, oldFaithful, rings, unequalVarianceBlobs } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('clustering datasets', () => {
  it('rings have the asked radii and noise', () => {
    const { data, truth } = generate(rings, { key: key(0), knobs: { n: 3000, rings: 3, noise: 0.1 } })
    const x = toFlat(data.x)
    const y = toFlat(data.y)
    const dev = y.map((c, i) => Math.hypot(x[2 * i] ?? 0, x[2 * i + 1] ?? 0) - (c + 1))
    expect(Math.sqrt(mean(dev.map((v) => v * v)))).toBeCloseTo(0.1, 2)
    expect(toFlat(truth.clusters as Tensor)).toEqual(y)
    expect(clusterAgreement(truth, toFlat(bayesPredict(truth, data.x)))).toBeGreaterThan(0.99)
    expect(truth.bayesError).toBeLessThan(0.01)
  })

  for (const g of [anisotropicBlobs, unequalVarianceBlobs]) {
    it(`${g.name}: posterior sums to one and the Bayes rule matches the Bayes error`, () => {
      const { data, truth } = generate(g, { key: key(1), knobs: { n: 3000, boxSize: 3 } })
      for (const s of toFlat(sum(truth.posterior?.(data.x) as Tensor, 1))) expect(s).toBeCloseTo(1, 10)
      const err = mean(toFlat(bayesPredict(truth, data.x)).map((p, i) => (p === toFlat(data.y)[i] ? 0 : 1)))
      const b = truth.bayesError ?? 0
      expect(Math.abs(err - b)).toBeLessThan(5 * Math.sqrt(Math.max(b * (1 - b), 1e-3) / 3000))
    })
  }

  it('unequal-variance blobs have their own spreads', () => {
    const { data, truth } = generate(unequalVarianceBlobs, { key: key(2), knobs: { n: 3000, shuffle: false } })
    const centres = truth.latents?.['centers'] as number[][]
    const x = toFlat(data.x)
    const y = toFlat(data.y)
    for (const [c, sd] of [1, 2.5, 0.5].entries()) {
      const dev = y.flatMap((l, i) => (l === c ? [(x[2 * i] ?? 0) - (centres[c]?.[0] ?? 0)] : []))
      expect(Math.sqrt(mean(dev.map((v) => v * v)))).toBeCloseTo(sd, 0)
    }
  })

  it('Old Faithful splits into short and long eruptions', () => {
    const { data, truth } = oldFaithful()
    expect(data.x.shape).toEqual([272, 2])
    expect(toFlat(data.y).filter((c) => c === 0).length).toBe(97)
    expect(toFlat(truth.clusters as Tensor)).toEqual(toFlat(data.y))
  })
})
