// Flow guides (planar, radial, coupling) against targets a Gaussian cannot match: they recover a Gaussian posterior's means as
// the full-rank guide does, and on a banana-shaped density (a non-Gaussian one) reach a higher bound than the best Gaussian.
import { describe, expect, it } from 'vitest'
import { MODELS } from '../../testing/probprog.ts'
import { normal } from '../dists/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { primalOf, square, sub, toFlat, type Value } from '../tensor/index.ts'
import { stlElbo } from './elbo-extra.ts'
import { svi } from './engines/svi.ts'
import { type AutoGuide, autoMultivariateNormal } from './guides.ts'
import { autoFlowNormal } from './guides-flow.ts'
import { type Model, sample } from './primitives.ts'

const lr = (t: number) => 0.04 / (1 + t / 400)
const banana: Model = () => {
  const a = sample('a', normal(0, 1))
  sample('b', normal(sub(square(a), 1), 0.3))
}
const tail = (r: { elbos: readonly number[] }): number => r.elbos.slice(-200).reduce((a, b) => a + b, 0) / 200
const fitBanana = (guide: AutoGuide<void>) =>
  tail(
    svi(banana, undefined, {
      key: key(1),
      guide,
      numSteps: 1200,
      optimizer: adam({ learningRate: lr }),
      elbo: stlElbo({ numParticles: 2 }),
    }),
  )

describe('flow guides', () => {
  const mvn = MODELS['mvn'] as Model
  for (const flow of ['planar', 'radial', 'coupling'] as const) {
    it(`${flow}: recovers the means of a Gaussian posterior`, { timeout: 30_000 }, () => {
      const r = svi(mvn, undefined, {
        key: key(5),
        guide: autoFlowNormal(mvn, { flow, numFlows: 2 }),
        numSteps: 600,
        optimizer: adam({ learningRate: lr }),
        elbo: stlElbo({ numParticles: 2 }),
      })
      expect(r.status).toBe('completed')
      const x = toFlat(primalOf(r.posterior.sample(key(6), 4000)['x'] as Value))
      const mean = [0, 1].map((j) => x.filter((_, i) => i % 2 === j).reduce((a, b) => a + b, 0) / 4000)
      // The conditioned mean, by hand: (-0.186, 0.501).
      expect(Math.abs((mean[0] ?? 0) + 0.186)).toBeLessThan(0.15)
      expect(Math.abs((mean[1] ?? 0) - 0.501)).toBeLessThan(0.15)
    })
  }

  it('beat the best Gaussian guide on a banana-shaped density, which the full-rank guide cannot follow', () => {
    const full = fitBanana(autoMultivariateNormal(banana))
    const planar = fitBanana(autoFlowNormal(banana, { flow: 'planar', numFlows: 3 }))
    // The density is normalised, so every bound is at most 0 and a better guide is closer to it.
    expect(planar).toBeGreaterThan(full + 0.1)
    expect(planar).toBeLessThan(1e-6)
  }, 30_000)

  it('refuses a coupling flow over one dimension', () => {
    const one: Model = () => sample('x', normal(0, 1))
    expect(() =>
      svi(one, undefined, { key: key(0), guide: autoFlowNormal(one, { flow: 'coupling' }), numSteps: 1 }),
    ).toThrow(/two or more/)
  })
})
