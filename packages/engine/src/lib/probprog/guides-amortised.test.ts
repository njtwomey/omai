// The amortised guide on a model with a known per-datum posterior: z_n ~ N(0, 1), x_n ~ N(w z_n + b, s), whose posterior of
// every z_n is a normal with a mean linear in x_n. The encoder learns it (and, being a function of the features, serves rows
// it was not trained on), its gap to a free per-datum fit is small, and a model with a nonlinear posterior needs capacity.
import { describe, expect, it } from 'vitest'
import { normal } from '../dists/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { add, mul, primalOf, tanh, tensor, toFlat, type Value } from '../tensor/index.ts'
import { stlElbo } from './elbo-extra.ts'
import { svi } from './engines/svi.ts'
import { amortisationGap, amortisedGuide } from './guides-amortised.ts'
import { type Model, observe, plate, sample } from './primitives.ts'

const W = 1.5
const B = 0.2
const S = 0.6
const N = 24
const x = Array.from({ length: N }, (_, i) => W * Math.sin(i * 0.7) * 1.3 + B + 0.4 * Math.cos(i * 1.9))
const linearModel: Model = () => {
  plate('n', N, () => {
    const z = sample('z', normal(0, 1))
    observe('x', normal(add(mul(z, W), B), S), x)
  })
}
const decaying = (lr: number) => adam({ learningRate: (t: number) => lr / (1 + t / 500) })

describe('amortised guide', () => {
  it('learns the per-datum posterior of a linear-Gaussian model with a small encoder', () => {
    const guide = amortisedGuide(linearModel, { sites: ['z'], input: () => tensor(x), hidden: [8] })
    const r = svi(linearModel, undefined, {
      key: key(1),
      guide,
      numSteps: 2500,
      optimizer: decaying(0.02),
      elbo: stlElbo({ numParticles: 2 }),
    })
    const enc = guide.encode(r.params, undefined)['z']
    const loc = toFlat(primalOf(enc?.loc as Value))
    const scale = toFlat(primalOf(enc?.scale as Value))
    const prec = 1 + (W * W) / (S * S)
    x.forEach((xn, n) => {
      expect(Math.abs((loc[n] ?? 0) - (W * (xn - B)) / (S * S) / prec), `mean of z_${n}`).toBeLessThan(0.06)
      expect(Math.abs((scale[n] ?? 0) - Math.sqrt(1 / prec)), `sd of z_${n}`).toBeLessThan(0.06)
    })
    // The weights are N-free: far fewer parameters than 2 N per-datum ones.
    expect(guide.encoderParameters(undefined)).toBeLessThan(2 * N + 20)
  })

  it('has a small gap to a free per-datum fit, and the posterior reads', () => {
    const guide = amortisedGuide(linearModel, { sites: ['z'], input: () => tensor(x), hidden: [8] })
    const r = svi(linearModel, undefined, {
      key: key(2),
      guide,
      numSteps: 2500,
      optimizer: decaying(0.02),
      elbo: stlElbo({ numParticles: 2 }),
    })
    const g = amortisationGap({
      model: linearModel,
      data: undefined,
      guide,
      params: r.params,
      key: key(3),
      steps: 1200,
      draws: 150,
    })
    expect(Math.abs(g.gap)).toBeLessThan(0.25)
    expect(r.posterior.sites).toEqual(['z'])
    expect(toFlat(primalOf(r.posterior.sample(key(4), 5)['z'] as Value))).toHaveLength(5 * N)
  })

  it("has an amortisation gap that shrinks with the encoder's capacity, on a nonlinear posterior", () => {
    const M = 40
    const zs = Array.from({ length: M }, (_, i) => -2 + (4 * i) / (M - 1))
    const xs = zs.map((z) => 3 * Math.tanh(1.5 * z) + z + 0.2 * Math.sin(5 * z))
    const model: Model = () => {
      plate('n', M, () => {
        const z = sample('z', normal(0, 1.5))
        observe('x', normal(add(mul(tanh(mul(z, 1.5)), 3), z), 0.3), xs)
      })
    }
    const gapFor = (hidden: readonly number[]): number => {
      const guide = amortisedGuide(model, { sites: ['z'], input: () => tensor(xs), hidden })
      const r = svi(model, undefined, {
        key: key(1),
        guide,
        numSteps: 2500,
        optimizer: adam({ learningRate: (t: number) => 0.03 / (1 + t / 600) }),
        elbo: stlElbo({ numParticles: 2 }),
      })
      return amortisationGap({ model, data: undefined, guide, params: r.params, key: key(3), steps: 1500, draws: 150 })
        .gap
    }
    const [linear, large] = [gapFor([]), gapFor([16])]
    expect(linear).toBeGreaterThan(5)
    expect(large).toBeLessThan(linear / 20)
    expect(large).toBeLessThan(0.3)
  }, 60_000)

  it('refuses a site that is not under one plate, and an unknown one', () => {
    const noPlate: Model = () => {
      sample('z', normal(0, 1))
    }
    expect(() =>
      svi(noPlate, undefined, {
        key: key(0),
        guide: amortisedGuide(noPlate, { sites: ['z'], input: () => tensor([1]) }),
        numSteps: 1,
      }),
    ).toThrow(/exactly one plate/)
    expect(() =>
      svi(linearModel, undefined, {
        key: key(0),
        guide: amortisedGuide(linearModel, { sites: ['q'], input: () => tensor(x) }),
        numSteps: 1,
      }),
    ).toThrow(/does not cover/)
  })
})
