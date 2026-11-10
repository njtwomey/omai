// The guide methods: a structured guide recovers the correlation of explaining away through the method's own session, a
// Laplace start is the posterior from the first step, the mixture guide finds both modes, the amortised guide learns the encoder,
// and each meets the acceptance criteria of a method (resume equals uninterrupted, snapshot per second, cancel within a step).
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { compose, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { add, mul, primalOf, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { type Config, session } from '../../trace/index.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import type { ModelAnalysis } from './_support.ts'
import type { InferenceMethod } from './method.ts'
import { amortisedSviMethod, mixtureSviMethod, structuredSviMethod } from './svi-guide-methods.ts'

const explaining: Model = () => {
  const a = sample('a', normal(0, 1))
  const b = sample('b', normal(0, 1))
  observe('y', normal(add(a, b), 0.5), 1.0)
}
const bimodal: Model = () => {
  sample('z', compose.mixtureSameFamily({ probs: [0.5, 0.5] }, normal([-3, 3], 0.5)))
}
const xs = Array.from({ length: 12 }, (_, i) => 1.5 * Math.sin(i * 0.9) + 0.2)
const local: Model = () =>
  plate('n', xs.length, () => {
    const z = sample('z', normal(0, 1))
    observe('x', normal(add(mul(z, 1.5), 0.2), 0.6), xs)
  })

const corr = (draws: Record<string, Value>): number => {
  const a = toFlat(primalOf(draws['a'] as Value))
  const b = toFlat(primalOf(draws['b'] as Value))
  const n = a.length
  const ma = a.reduce((s, x) => s + x, 0) / n
  const mb = b.reduce((s, x) => s + x, 0) / n
  let c = 0
  let va = 0
  let vb = 0
  for (let i = 0; i < n; i++) {
    c += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
    va += ((a[i] ?? 0) - ma) ** 2
    vb += ((b[i] ?? 0) - mb) ** 2
  }
  return c / Math.sqrt(va * vb)
}

describe('structuredSviMethod', () => {
  it('recovers the correlation of explaining away (-0.8), which mean-field cannot', () => {
    const m = structuredSviMethod(explaining)
    const s = session(m, { config: { learningRate: 0.05, numParticles: 2 }, steps: 1200, seed: 1, mode: 'eager' })
    const r = corr(m.posterior?.(s.state, s.config)?.sample(key(3), 6000) as Record<string, Value>)
    expect(Math.abs(r + 0.8)).toBeLessThan(0.06)
    const f = session(m, {
      config: { learningRate: 0.05, dependencies: 'none', within: 'diag' },
      steps: 1200,
      seed: 1,
      mode: 'eager',
    })
    expect(
      Math.abs(corr(m.posterior?.(f.state, f.config)?.sample(key(3), 6000) as Record<string, Value>)),
    ).toBeLessThan(0.05)
  })

  it('starts at the Laplace approximation when asked, and every guide family is on offer', () => {
    const m = structuredSviMethod(explaining)
    const s = session(m, {
      config: { init: 'laplace', guide: 'autoMultivariateNormal', learningRate: 1e-5 },
      steps: 3,
      seed: 1,
      mode: 'eager',
    })
    expect(
      Math.abs(corr(m.posterior?.(s.state, s.config)?.sample(key(3), 6000) as Record<string, Value>) + 0.8),
    ).toBeLessThan(0.05)
    for (const guide of [
      'autoLowRankMultivariateNormal',
      'autoFlowNormal',
      'autoIafNormal',
      'autoNormal',
      'autoStructured',
    ]) {
      const t = session(m, { config: { guide, init: 'laplace' }, steps: 3, seed: 1, mode: 'eager' })
      expect(Number.isFinite(t.state.elbo), guide).toBe(true)
    }
  })
})

describe('mixtureSviMethod', () => {
  it('finds both modes of a bimodal density, with the cells on the tape', () => {
    const m = mixtureSviMethod(bimodal)
    const s = session(m, { config: { learningRate: 0.05, numParticles: 4 }, steps: 1200, seed: 2, mode: 'eager' })
    const draws = toFlat(primalOf(m.posterior?.(s.state, s.config)?.sample(key(5), 2000)['z'] as Value))
    expect(draws.filter((x) => x < 0).length / 2000).toBeGreaterThan(0.3)
    expect(draws.filter((x) => x > 0).length / 2000).toBeGreaterThan(0.3)
  }, 30_000)
})

describe('amortisedSviMethod', () => {
  it('learns the encoder: each z_n has the exact posterior mean', () => {
    const m = amortisedSviMethod(local, { sites: ['z'], input: () => tensor(xs) })
    const s = session(m, {
      config: { learningRate: 0.02, hiddenLayers: 1, hiddenWidth: 8 },
      steps: 2500,
      seed: 1,
      mode: 'eager',
    })
    const post = m.posterior?.(s.state, s.config)
    const mean = toFlat(primalOf(post?.mean('z') as Value))
    const prec = 1 + (1.5 * 1.5) / (0.6 * 0.6)
    for (const [n, x] of xs.entries())
      expect(Math.abs((mean[n] ?? 0) - (1.5 * (x - 0.2)) / 0.36 / prec), `z_${n}`).toBeLessThan(0.1)
  }, 30_000)
})

describe('the acceptance criteria of a method', () => {
  type Any = InferenceMethod<unknown, Config, string, ModelAnalysis>
  const small = [
    ['structured', structuredSviMethod(explaining), { learningRate: 0.05 }, 40, 17],
    [
      'structured-laplace',
      structuredSviMethod(explaining),
      { learningRate: 0.05, init: 'laplace', guide: 'autoMultivariateNormal' },
      40,
      11,
    ],
    ['mixture', mixtureSviMethod(bimodal), { learningRate: 0.05 }, 40, 19],
    [
      'amortised',
      amortisedSviMethod(local, { sites: ['z'], input: () => tensor(xs) }),
      { learningRate: 0.05, hiddenWidth: 4 },
      40,
      23,
    ],
  ] as const
  for (const [name, method, config, steps, at] of small) {
    it(`${name}: a state restored through JSON continues to the same result`, () => {
      expectResumeEqual(method as unknown as Any, { config, steps, at })
    })
  }
  it('snapshots once per second of clock time and cancels between steps', () => {
    for (const [, method, config] of small) {
      const snaps = expectSnapshotCadence(method as unknown as Any, {
        config,
        steps: 400,
        budgetMs: 6000,
        everyMs: 1000,
        tick: 400,
      })
      expect(snaps.length).toBeGreaterThanOrEqual(2)
      expectCancelAndContinue(method as unknown as Any, { config, steps: 400, cancelAfterSlices: 1 })
    }
  })
})
