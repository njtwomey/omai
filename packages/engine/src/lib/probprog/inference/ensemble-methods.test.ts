// The ensemble methods: every walker of an ensemble is a chain of the posterior, the draws match closed forms, the tape adds the
// scale of the slice directions, and a run cancels within one iteration and resumes from a serialised state.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS, Y4 } from '../../../testing/probprog.ts'
import { session } from '../../trace/index.ts'
import { aiesMethod, ensembleSliceMethod } from './ensemble-methods.ts'

const exactMean = (() => {
  const precision = 1 / 4 + Y4.length / 2.25
  return Y4.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactSd = Math.sqrt(1 / (1 / 4 + Y4.length / 2.25))
const normalMean = MODELS['normalMean'] as never

describe.each([
  ['aies', aiesMethod, { numWalkers: 10, numWarmup: 100, numDraws: 1500 }],
  ['aies (differential)', aiesMethod, { numWalkers: 10, numWarmup: 100, numDraws: 1500, move: 'differential' }],
  ['ensemble-slice', ensembleSliceMethod, { numWalkers: 10, numWarmup: 100, numDraws: 1000 }],
  [
    'ensemble-slice (gaussian)',
    ensembleSliceMethod,
    { numWalkers: 10, numWarmup: 100, numDraws: 1000, move: 'gaussian' },
  ],
] as const)('%s', (_name, make, config) => {
  it('matches the closed-form posterior, with every walker a chain', () => {
    const m = make(normalMean)
    const s = session(m, { config, steps: 20000, mode: 'eager', seed: 1 })
    expect(s.status).toBe('converged')
    const post = m.posterior?.(s.state, s.config)
    expect(post?.chains?.numChains).toBe(10)
    const x = Array.from(post?.draws?.['mu']?.data ?? [])
    const mean = x.reduce((a, v) => a + v, 0) / x.length
    const sd = Math.sqrt(x.reduce((a, v) => a + (v - mean) ** 2, 0) / (x.length - 1))
    expect(Math.abs(mean - exactMean)).toBeLessThan(0.1)
    expect(Math.abs(sd - exactSd)).toBeLessThan(0.1)
  })
})

describe('the method convention', () => {
  const small = { numChains: 1, numWalkers: 6, numWarmup: 30, numDraws: 25 }
  for (const [name, make] of [
    ['aies', aiesMethod],
    ['ensemble-slice', ensembleSliceMethod],
  ] as const)
    it(`${name}: its tape, a cancel within one step, a resume that equals the run, and anytime snapshots`, () => {
      const m = make(normalMean)
      const s = session(m, { config: small, steps: 500, mode: 'eager', logs: 'all' })
      const paths = new Set(s.run().events.flatMap((e) => ('name' in e ? [e.name] : [])))
      for (const p of ['mcmc/accept', 'mcmc/kept', 'mcmc/logDensity']) expect(paths.has(p), p).toBe(true)
      if (name === 'ensemble-slice') expect(paths.has('ensemble/scale')).toBe(true)
      expect(s.t).toBe(1 + small.numWarmup + small.numDraws)
      expectCancelAndContinue(m, { config: small, steps: 500, cancelAfterSlices: 7 })
      expectResumeEqual(m, { config: small, steps: 500, at: 15 })
      expectResumeEqual(m, { config: small, steps: 500, at: 1 + small.numWarmup + 6 })
      expectSnapshotCadence(m, {
        config: { ...small, numDraws: 400 },
        steps: 5000,
        budgetMs: 2500,
        everyMs: 1000,
        tick: 100,
      })
    })

  it('moves the walkers of a two-dimensional model to the posterior mean', () => {
    const m = aiesMethod(MODELS['mvn'] as never)
    const s = session(m, {
      config: { numWalkers: 12, numWarmup: 200, numDraws: 800 },
      steps: 20000,
      mode: 'eager',
      seed: 3,
    })
    const post = m.posterior?.(s.state, s.config)
    expect(post?.chains?.dim).toBe(2)
    expect(post?.chains?.numChains).toBe(12)
  })
})
