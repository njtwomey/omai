// The Pathfinder method: it finds a Gaussian posterior, reads a partial approximation while the paths climb, logs its tape, and
// meets the acceptance criteria of a method (resume equals uninterrupted, snapshot per second of the injected clock, cancel
// within a step, a verdict from `check`).
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS } from '../../../testing/probprog.ts'
import { item } from '../../tensor/index.ts'
import { type Config, Log, session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { conjugateInference } from '../engines/conjugate.ts'
import type { Model } from '../primitives.ts'
import type { ModelAnalysis } from './_support.ts'
import type { InferenceMethod } from './method.ts'
import { pathfinderMethod } from './pathfinder-method.ts'

const normalMean = MODELS['normalMean'] as Model
const exact = conjugateInference(normalMean, undefined)

describe('pathfinderMethod', () => {
  it('finds the conjugate posterior, with the paths and the end on the tape', () => {
    const m = pathfinderMethod(normalMean)
    const s = session(m, {
      config: { paths: 2, maxIterations: 8, numDraws: 400 },
      steps: 100,
      seed: 1,
      mode: 'eager',
      logs: {
        'pathfinder/elbo': Log(),
        'pathfinder/iterate': Log(),
        'pathfinder/path/*': Log(),
        'pathfinder/paretoK': Log(),
        'pathfinder/ess': Log(),
      },
    })
    expect(s.status).toBe('converged')
    const post = m.posterior?.(s.state, s.config)
    expect(Math.abs(item(post?.mean('mu') as never) - item(exact.mean('mu')))).toBeLessThan(0.12)
    expect(Math.abs(item(post?.variance('mu') as never) - item(exact.variance('mu')))).toBeLessThan(0.12)
    const names = new Set(s.run().events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    for (const p of [
      'pathfinder/elbo',
      'pathfinder/iterate',
      'pathfinder/path/0',
      'pathfinder/path/1',
      'pathfinder/paretoK',
      'pathfinder/ess',
    ])
      expect(names.has(p), p).toBe(true)
  })

  it('reads a partial approximation while a path is still climbing', () => {
    const m = pathfinderMethod(MODELS['logistic'] as Model)
    const s = session(m, { steps: 3, seed: 1, mode: 'eager' })
    const post = m.posterior?.(s.state, s.config)
    expect(post?.status).toBe('stopped')
    expect(post?.sites).toEqual(['alpha', 'beta'])
  })

  it('says why it does not apply', () => {
    const v = pathfinderMethod(normalMean).check?.(analyse(MODELS['dirichletCategorical'] as Model, undefined))
    expect(v?.reason.length).toBeGreaterThan(5)
    expect(pathfinderMethod(normalMean).check?.(analyse(normalMean, undefined))?.applies).toBe(true)
  })
})

describe('the acceptance criteria of a method', () => {
  type Any = InferenceMethod<unknown, Config, string, ModelAnalysis>
  const small = [
    [
      'one-path',
      pathfinderMethod(MODELS['logistic'] as Model),
      { maxIterations: 12, elboDraws: 10, numDraws: 50 },
      40,
      3,
    ],
    [
      'multi-path',
      pathfinderMethod(MODELS['mvn'] as Model),
      { paths: 2, maxIterations: 12, elboDraws: 10, numDraws: 50 },
      40,
      4,
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
      expect(snaps.length).toBeGreaterThanOrEqual(1)
      expectCancelAndContinue(method as unknown as Any, { config, steps: 400, cancelAfterSlices: 1 })
    }
  })
})
