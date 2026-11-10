// ChEES-HMC as a method: every chain of an ensemble is a chain of the posterior, the trajectory length is on the tape, and the run
// cancels within one iteration, resumes exactly and reads as anytime snapshots.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS, Y4 } from '../../../testing/probprog.ts'
import { session } from '../../trace/index.ts'
import { cheesMethod } from './chees-method.ts'

const exactMean = (() => {
  const precision = 1 / 4 + Y4.length / 2.25
  return Y4.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactSd = Math.sqrt(1 / (1 / 4 + Y4.length / 2.25))
const normalMean = MODELS['normalMean'] as never

describe('cheesMethod', () => {
  it('matches the closed-form posterior with every chain of the ensemble a chain of it', { timeout: 30_000 }, () => {
    const m = cheesMethod(normalMean)
    const s = session(m, {
      config: { numWalkers: 12, numWarmup: 300, numDraws: 400 },
      steps: 20000,
      mode: 'eager',
      seed: 1,
    })
    expect(s.status).toBe('converged')
    const post = m.posterior?.(s.state, s.config)
    expect(post?.chains?.numChains).toBe(12)
    const x = Array.from(post?.draws?.['mu']?.data ?? [])
    const mean = x.reduce((a, v) => a + v, 0) / x.length
    const sd = Math.sqrt(x.reduce((a, v) => a + (v - mean) ** 2, 0) / (x.length - 1))
    expect(Math.abs(mean - exactMean)).toBeLessThan(0.1)
    expect(Math.abs(sd - exactSd)).toBeLessThan(0.1)
  })

  const small = { numChains: 1, numWalkers: 4, numWarmup: 30, numDraws: 20 }
  it('puts the trajectory length on the tape, cancels within one step, resumes exactly and snapshots', () => {
    const m = cheesMethod(normalMean)
    const s = session(m, { config: small, steps: 500, mode: 'eager', logs: 'all' })
    const paths = new Set(s.run().events.flatMap((e) => ('name' in e ? [e.name] : [])))
    for (const p of ['chees/trajectoryLength', 'chees/numSteps', 'mcmc/accept', 'mcmc/stepSize'])
      expect(paths.has(p), p).toBe(true)
    expectCancelAndContinue(m, { config: small, steps: 500, cancelAfterSlices: 7 })
    expectResumeEqual(m, { config: small, steps: 500, at: 15 })
    expectResumeEqual(m, { config: small, steps: 500, at: 1 + small.numWarmup + 6 })
    expectSnapshotCadence(m, {
      config: { ...small, numDraws: 300 },
      steps: 5000,
      budgetMs: 2500,
      everyMs: 1000,
      tick: 100,
    })
  })
})
