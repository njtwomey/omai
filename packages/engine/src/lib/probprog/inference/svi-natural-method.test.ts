// The natural-gradient methods: the Gaussian one reaches the conjugate posterior and logs the natural gradient, the conjugate one
// is batch CAVI at rate one and minibatch SVI otherwise; both are resumable, snapshot on the clock, cancel between steps and say
// why they do not apply.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS } from '../../../testing/probprog.ts'
import { gamma, poisson } from '../../dists/index.ts'
import { item } from '../../tensor/index.ts'
import { type Config, Log, session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { conjugateInference } from '../engines/conjugate.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import type { ModelAnalysis } from './_support.ts'
import type { InferenceMethod } from './method.ts'
import { conjugateNaturalSviMethod, naturalGradientSviMethod } from './svi-natural-method.ts'

const normalMean = MODELS['normalMean'] as Model
const exact = conjugateInference(normalMean, undefined)
const gammaPoisson = MODELS['gammaPoisson'] as Model

describe('naturalGradientSviMethod', () => {
  it('reaches the conjugate posterior in tens of steps and logs the natural gradient and the rate', () => {
    const m = naturalGradientSviMethod(normalMean)
    const s = session(m, {
      config: { numParticles: 16, forgetting: 0.6 },
      steps: 60,
      seed: 3,
      mode: 'eager',
      logs: { 'svi/elbo': Log(), 'svi/naturalGradNorm': Log(), 'svi/rate': Log() },
    })
    const post = m.posterior?.(s.state, s.config)
    expect(Math.abs(item(post?.mean('mu') as never) - item(exact.mean('mu')))).toBeLessThan(0.1)
    expect(Math.abs(item(post?.variance('mu') as never) - item(exact.variance('mu')))).toBeLessThan(0.15)
    const names = new Set(s.run().events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const p of ['svi/elbo', 'svi/naturalGradNorm', 'svi/rate']) expect(names.has(p), p).toBe(true)
  })

  it('says why it does not apply', () => {
    const verdict = naturalGradientSviMethod(normalMean).check?.(
      analyse(() => {
        sample('x', gamma(1, 1))
      }, undefined),
    )
    expect(verdict?.applies).toBe(true)
    expect(
      naturalGradientSviMethod(normalMean).check?.(analyse(MODELS['dirichletCategorical'] as Model, undefined))?.reason
        .length,
    ).toBeGreaterThan(5)
  })
})

describe('conjugateNaturalSviMethod', () => {
  it('is batch CAVI in one step: the exact posterior of a gamma-Poisson model', () => {
    const m = conjugateNaturalSviMethod(gammaPoisson)
    const s = session(m, { steps: 1, mode: 'eager' })
    const post = m.posterior?.(s.state, s.config)
    const truth = conjugateInference(gammaPoisson, undefined)
    expect(Math.abs(item(post?.mean('rate') as never) - item(truth.mean('rate')))).toBeLessThan(1e-12)
    expect(post?.form).toBe('variational')
  })

  it("refuses a model that is not conjugate, with the conjugate method's reason", () => {
    const logistic = MODELS['logistic'] as Model
    expect(() => session(conjugateNaturalSviMethod(logistic), { steps: 2, mode: 'eager' })).toThrow(/conjugate/)
    expect(conjugateNaturalSviMethod(logistic).check?.(analyse(logistic, undefined))?.applies).toBe(false)
  })
})

describe('the acceptance criteria of a method', () => {
  const counts = Array.from({ length: 60 }, (_, i) => 2 + ((i * 7) % 5))
  const minibatch: Model = () => {
    const r = sample('rate', gamma(2, 0.5))
    plate('n', counts.length, () => observe('k', poisson(r), counts))
  }
  type Any = InferenceMethod<unknown, Config, string, ModelAnalysis>
  const small = [
    ['natural', naturalGradientSviMethod(normalMean), { numParticles: 4 }, 40, 17],
    ['natural-fullrank', naturalGradientSviMethod(normalMean), { numParticles: 4, family: 'fullRank' }, 40, 13],
    ['conjugate-natural', conjugateNaturalSviMethod(minibatch), { batchSize: 10, forgetting: 0.8 }, 40, 19],
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
