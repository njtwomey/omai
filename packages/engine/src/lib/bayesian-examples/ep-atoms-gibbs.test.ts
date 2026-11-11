// The references of the extended gate atoms: the Gibbs sampler of a mixture finds the components of well-separated data, the
// ordered form relabels every draw, and the clutter quadrature puts the far observations on the clutter and the near ones on the centre.
import { describe, expect, it } from 'vitest'
import { clutterPosterior, mixtureGibbs, samplesMarginal } from './ep-atoms-gibbs.ts'

describe('mixtureGibbs', () => {
  const x = [-2.1, -1.8, -2.4, -1.5, -2.0, -1.7, 2.2, 1.9, 3.1, 2.4, 2.6, 2.0]
  const spec = {
    x,
    means: [
      { mean: 0, sd: 4 },
      { mean: 0, sd: 4 },
    ],
    precision: { shape: 2, rate: 2 },
    weights: [2, 2],
    seed: 3,
    burn: 200,
    draws: 600,
  }

  it('finds the two components, ordered by their means, and the responsibilities split the data', () => {
    const g = mixtureGibbs({ ...spec, ordered: true })
    const m0 = samplesMarginal(g.mean[0] as number[])
    const m1 = samplesMarginal(g.mean[1] as number[])
    expect(m0.mean).toBeGreaterThan(-2.6)
    expect(m0.mean).toBeLessThan(-1.4)
    expect(m1.mean).toBeGreaterThan(1.6)
    expect(m1.mean).toBeLessThan(3)
    expect((g.mean[0] as number[]).every((v, i) => v <= ((g.mean[1] as number[])[i] as number))).toBe(true)
    expect((g.responsibility[0] as number[])[0]).toBeGreaterThan(0.95)
    expect((g.responsibility[11] as number[])[1]).toBeGreaterThan(0.95)
    expect(samplesMarginal(g.weight[0] as number[]).mean).toBeCloseTo(0.5, 1)
  })

  it('is the same chain for the same seed', () => {
    const a = mixtureGibbs({ ...spec, draws: 20, ordered: true })
    const b = mixtureGibbs({ ...spec, draws: 20, ordered: true })
    expect(a.mean).toEqual(b.mean)
  })

  it('refuses priors that do not match the components', () => {
    expect(() => mixtureGibbs({ ...spec, weights: [1, 1, 1], ordered: true })).toThrow(/one weight prior/)
  })
})

describe('clutterPosterior', () => {
  const y = [1.4, 1.6, 1.2, -3.1, 1.7, 1.5, 5.2, 1.1, 1.8, 1.3, 0.9, 1.6]
  const priors = {
    centre: 5,
    inlier: { shape: 2, rate: 1 },
    clutter: { shape: 2, rate: 20 },
    weight: { a: 2, b: 5 },
  } as const

  it('centres on the inliers, calls the far observations clutter and the near ones not', () => {
    const p = clutterPosterior(y, priors)
    expect(p.centre.mean).toBeGreaterThan(1.3)
    expect(p.centre.mean).toBeLessThan(1.6)
    expect(p.clutterProbability[3]).toBeGreaterThan(0.9)
    expect(p.clutterProbability[6]).toBeGreaterThan(0.9)
    expect(p.clutterProbability[0]).toBeLessThan(0.2)
    expect(p.weight.mean).toBeGreaterThan(0.1)
    expect(p.weight.mean).toBeLessThan(0.4)
    expect(p.inlier.mean).toBeGreaterThan(p.clutter.mean)
  })
})
