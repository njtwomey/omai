// The variational methods beyond `sviMethod`: each recovers a posterior it can be checked against (the conjugate normal, a
// binary latent with a closed-form posterior), logs the tape it declares, and meets the common acceptance criteria of a
// method (resume equals uninterrupted, a snapshot per second of the injected clock, cancel within a step, a verdict from
// `check`).
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS } from '../../../testing/probprog.ts'
import { bernoulli, normal } from '../../dists/index.ts'
import { item, mul, tensor } from '../../tensor/index.ts'
import { type Config, Log, session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { conjugateInference } from '../engines/conjugate.ts'
import type { Model } from '../primitives.ts'
import { observe, param, sample } from '../primitives.ts'
import type { ModelAnalysis } from './_support.ts'
import type { InferenceMethod } from './method.ts'
import {
  renyiSviMethod,
  stickingTheLandingMethod,
  traceEnumSviMethod,
  traceGraphSviMethod,
} from './svi-extra-methods.ts'

const normalMean = MODELS['normalMean'] as Model
const exact = conjugateInference(normalMean, undefined)
const m0 = item(exact.mean('mu'))

// A binary latent with one observation: P(c = 1 | y) is known, so the fitted guide can be checked.
const Y = 1.3
const lpdf = (x: number, m: number): number => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2
const post1 = (0.4 * Math.exp(lpdf(Y, 3))) / (0.4 * Math.exp(lpdf(Y, 3)) + 0.6 * Math.exp(lpdf(Y, 0)))
const binaryModel: Model = () => {
  const c = sample('c', bernoulli({ probs: 0.4 }))
  observe('y', normal(mul(c, 3), 1), Y)
}
const binaryGuide: Model = () => {
  sample('c', bernoulli({ logits: param('l', tensor(0)) }))
}

const fit = <S, K extends string>(
  method: InferenceMethod<S, Config, K, ModelAnalysis>,
  config: Partial<Config>,
  steps: number,
) => {
  const s = session(method, { config, steps, seed: 3, mode: 'eager' })
  const post = method.posterior?.(s.state, s.config)
  if (post === undefined) throw new Error('no posterior')
  return { s, post }
}

describe('the reparameterised estimators fit the conjugate posterior', () => {
  it('sticking the landing, the doubly reparameterised bound and the Renyi bound', () => {
    for (const [name, method, config] of [
      ['stl', stickingTheLandingMethod(normalMean), { learningRate: 0.05 }],
      ['dreg', stickingTheLandingMethod(normalMean), { estimator: 'dreg', numParticles: 6, learningRate: 0.05 }],
      ['renyi', renyiSviMethod(normalMean), { alpha: 0.5, learningRate: 0.05 }],
    ] as const) {
      const { post } = fit(method as never, config, 1500)
      expect(Math.abs(item(post.mean('mu')) - m0), name).toBeLessThan(0.15)
    }
  })

  it('the graph estimator and the enumerating one reduce to the plain ELBO on continuous latents', () => {
    for (const method of [traceGraphSviMethod(normalMean), traceEnumSviMethod(normalMean)]) {
      const { post } = fit(method as never, { learningRate: 0.05, numParticles: 4 }, 1200)
      expect(Math.abs(item(post.mean('mu')) - m0), method.name).toBeLessThan(0.15)
    }
  })
})

describe('the discrete latent, with a guide written by hand', () => {
  it('both find the posterior probability of the site, by score-function and by enumeration', () => {
    for (const [method, config] of [
      [
        traceGraphSviMethod(binaryModel, { guide: binaryGuide }),
        { learningRate: 0.05, numParticles: 4, baseline: 'leaveOneOut' },
      ],
      [
        traceGraphSviMethod(binaryModel, { guide: binaryGuide }),
        { learningRate: 0.05, numParticles: 2, baseline: 'learned' },
      ],
      [traceEnumSviMethod(binaryModel, { guide: binaryGuide }), { learningRate: 0.1 }],
    ] as const) {
      const s = session(method as never, { config: config as Config, steps: 700, seed: 2, mode: 'eager' })
      const l = item((s.state as { params: Record<string, never> }).params['l'] as never)
      expect(Math.abs(1 / (1 + Math.exp(-l)) - post1), method.name).toBeLessThan(0.08)
    }
  })

  it('puts the baselines, the gradient norm and the gradient variance on the tape', () => {
    const m = traceGraphSviMethod(binaryModel, { guide: binaryGuide })
    const s = session(m, {
      config: { varianceEvery: 5, varianceRepeats: 6, learningRate: 0.05 },
      steps: 12,
      mode: 'eager',
      logs: { 'svi/elbo': Log(), 'svi/gradNorm': Log(), 'svi/gradVariance': Log(), 'svi/diag/*': Log() },
    })
    const names = new Set(s.run().events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const p of ['svi/elbo', 'svi/gradNorm', 'svi/gradVariance', 'svi/diag/baseline.c', 'svi/diag/cost.c'])
      expect(names.has(p), p).toBe(true)
    const e = session(traceEnumSviMethod(binaryModel, { guide: binaryGuide }), {
      steps: 3,
      mode: 'eager',
      logs: { 'svi/diag/*': Log() },
    })
    const enumNames = new Set(e.run().events.flatMap((x) => (x.type === 'metric' ? [x.name] : [])))
    expect(enumNames.has('svi/diag/cells')).toBe(true)
  })
})

describe('the acceptance criteria of a method', () => {
  const small = [
    ['trace-graph', traceGraphSviMethod(normalMean), { learningRate: 0.05, numParticles: 2 }, 40, 17],
    ['trace-enum', traceEnumSviMethod(normalMean), { learningRate: 0.05 }, 40, 13],
    ['stl', stickingTheLandingMethod(normalMean), { learningRate: 0.05 }, 40, 11],
    ['renyi', renyiSviMethod(normalMean), { learningRate: 0.05 }, 40, 19],
    [
      'graph-discrete',
      traceGraphSviMethod(binaryModel, { guide: binaryGuide }),
      { learningRate: 0.05, baseline: 'learned', numParticles: 2 },
      40,
      23,
    ],
  ] as const
  type Any = InferenceMethod<unknown, Config, string, ModelAnalysis>
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
  it('says why it does not apply, and applies with a guide written by hand', () => {
    const discreteOnly = analyse(binaryModel, undefined)
    const refused = traceGraphSviMethod(binaryModel).check?.(discreteOnly)
    expect(refused?.applies).toBe(false)
    expect(refused?.reason).toMatch(/no continuous/)
    expect(traceGraphSviMethod(binaryModel, { guide: binaryGuide }).check?.(discreteOnly)?.applies).toBe(true)
    expect(() => session(traceEnumSviMethod(binaryModel), { steps: 2, mode: 'eager' })).toThrow(/no continuous/)
  })
})
