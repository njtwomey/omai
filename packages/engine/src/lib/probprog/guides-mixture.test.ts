// The mixture guide on a bimodal density: with the component enumerated it finds both modes and their weights, and reaches a
// bound near the log normaliser (0), where the best single Gaussian can cover one mode only (at most -log 2).
import { describe, expect, it } from 'vitest'
import { compose, normal } from '../dists/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { primalOf, toFlat, type Value } from '../tensor/index.ts'
import { traceEnumElbo } from './elbo-enum.ts'
import { stlElbo } from './elbo-extra.ts'
import { svi } from './engines/svi.ts'
import { autoMultivariateNormal } from './guides.ts'
import { autoMixtureNormal } from './guides-mixture.ts'
import { type Model, sample } from './primitives.ts'

const bimodal: Model = () => {
  sample('z', compose.mixtureSameFamily({ probs: [0.5, 0.5] }, normal([-3, 3], 0.5)))
}
const lr = (t: number) => 0.05 / (1 + t / 500)
const tail = (r: { elbos: readonly number[] }): number => r.elbos.slice(-200).reduce((a, b) => a + b, 0) / 200

describe('mixture guide', () => {
  it('finds both modes of a bimodal density and their weights', () => {
    const guide = autoMixtureNormal(bimodal, { components: 2 })
    const r = svi(bimodal, undefined, {
      key: key(2),
      guide,
      numSteps: 1500,
      optimizer: adam({ learningRate: lr }),
      elbo: traceEnumElbo({ numParticles: 4 }),
    })
    const locs = Array.from(toFlat(r.params['auto_locs'] as Value)).sort((a, b) => a - b)
    expect(Math.abs((locs[0] ?? 0) + 3)).toBeLessThan(0.3)
    expect(Math.abs((locs[1] ?? 0) - 3)).toBeLessThan(0.3)
    const logits = Array.from(toFlat(primalOf(r.params['auto_logits'] as Value)))
    expect(Math.abs((logits[0] ?? 0) - (logits[1] ?? 0))).toBeLessThan(0.5)
    expect(tail(r)).toBeGreaterThan(-0.35)
    // The posterior it reports has both modes.
    const draws = toFlat(primalOf(r.posterior.sample(key(5), 2000)['z'] as Value))
    expect(draws.filter((x) => x < 0).length / 2000).toBeGreaterThan(0.35)
    expect(draws.filter((x) => x > 0).length / 2000).toBeGreaterThan(0.35)
  }, 30_000)

  it('beats the best single Gaussian by more than a quarter of a nat', () => {
    const mix = svi(bimodal, undefined, {
      key: key(2),
      guide: autoMixtureNormal(bimodal),
      numSteps: 1500,
      optimizer: adam({ learningRate: lr }),
      elbo: traceEnumElbo({ numParticles: 4 }),
    })
    const gauss = svi(bimodal, undefined, {
      key: key(2),
      guide: autoMultivariateNormal(bimodal),
      numSteps: 1500,
      optimizer: adam({ learningRate: lr }),
      elbo: stlElbo({ numParticles: 4 }),
    })
    expect(tail(mix)).toBeGreaterThan(tail(gauss) + 0.25)
    expect(tail(gauss)).toBeLessThan(-0.6)
  }, 30_000)
})
