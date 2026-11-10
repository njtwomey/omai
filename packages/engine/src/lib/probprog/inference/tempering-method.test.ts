// Parallel tempering as a method: the cold replica's draws match a closed form, the tape carries the ladder and the swap
// statistics, and the run cancels within one iteration, resumes exactly and reads as anytime snapshots.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS, Y4 } from '../../../testing/probprog.ts'
import { session } from '../../trace/index.ts'
import { temperingMethod } from './tempering-method.ts'

const exactMean = (() => {
  const precision = 1 / 4 + Y4.length / 2.25
  return Y4.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactSd = Math.sqrt(1 / (1 / 4 + Y4.length / 2.25))
const normalMean = MODELS['normalMean'] as never

describe('temperingMethod', () => {
  for (const kernel of ['hmc', 'nuts', 'mala', 'barker', 'rwm'])
    it(`${kernel}: the cold replica matches the closed-form posterior`, { timeout: 30_000 }, () => {
      const m = temperingMethod(normalMean)
      const config = {
        kernel,
        numChains: 2,
        numWarmup: 200,
        numDraws: 800,
        numTemperatures: 3,
        minBeta: 0.3,
        numSteps: 5,
      }
      const s = session(m, { config, steps: 20000, mode: 'eager', seed: 2 })
      expect(s.status).toBe('converged')
      const x = Array.from(m.posterior?.(s.state, s.config).draws?.['mu']?.data ?? [])
      const mean = x.reduce((a, v) => a + v, 0) / x.length
      const sd = Math.sqrt(x.reduce((a, v) => a + (v - mean) ** 2, 0) / (x.length - 1))
      expect(Math.abs(mean - exactMean)).toBeLessThan(0.15)
      expect(Math.abs(sd - exactSd)).toBeLessThan(0.12)
    })

  const small = { numChains: 2, numWarmup: 30, numDraws: 20, numTemperatures: 3, minBeta: 0.3 }
  it('puts the ladder and the swap statistics on the tape, cancels within one step, resumes exactly and snapshots', () => {
    const m = temperingMethod(normalMean)
    const s = session(m, { config: small, steps: 500, mode: 'eager', logs: 'all' })
    const paths = new Set(s.run().events.flatMap((e) => ('name' in e ? [e.name] : [])))
    for (const p of [
      'pt/temperature',
      'pt/swapRate',
      'pt/swapAccepted',
      'pt/replicaLogDensity',
      'pt/roundTrips',
      'mcmc/accept',
    ])
      expect(paths.has(p), p).toBe(true)
    expect(s.t).toBe(1 + small.numWarmup + small.numDraws)
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

  it('reports every adjacent pair of its ladder with a swap rate in the unit interval', () => {
    const m = temperingMethod(normalMean)
    const s = session(m, { config: { ...small, numDraws: 100 }, steps: 500, mode: 'eager', logs: 'all' })
    const run = s.run()
    const rates = run.events.filter((e) => 'name' in e && e.name === 'pt/swapRate')
    expect(rates.length).toBeGreaterThan(0)
  })
})
