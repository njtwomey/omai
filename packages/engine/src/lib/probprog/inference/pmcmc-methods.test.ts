// Particle MCMC against the exact answer: the AR(1) state-space benchmark with an unknown autoregression coefficient, whose
// posterior comes from the Kalman filter's exact marginal likelihood on a grid. Both methods recover its mean and spread within
// the chain's own Monte Carlo error, resume from a serialised state, snapshot on the cadence of a fake clock and cancel
// between steps; and `check` says why particle Gibbs does not apply to a model without a transition density.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { session } from '../../trace/index.ts'
import {
  ar1CoefficientPosterior,
  ar1CoefficientFamily as family,
  ar1Observations as Y,
} from '../models/ar1-parameters.ts'
import { particleGibbsMethod, pmmhMethod } from './pmcmc-methods.ts'

// The exact posterior of phi: the Kalman marginal likelihood on a fine grid under the uniform prior.
const exact = ar1CoefficientPosterior()

describe('particle marginal Metropolis-Hastings', () => {
  const method = pmmhMethod(family, Y)

  it('recovers the exact posterior of the autoregression coefficient within its Monte Carlo error', () => {
    const s = session(method, {
      steps: 500,
      mode: 'eager',
      seed: 1,
      config: { numIterations: 500, particles: 50, stepSize: 0.35 },
    })
    expect(s.status).toBe('completed')
    const row = method.posterior?.(s.state, s.config).summary()[0]
    expect(row?.name).toBe('phi')
    const ess = row?.ess ?? 1
    expect(Math.abs((row?.mean ?? 0) - exact.mean)).toBeLessThan((4 * exact.sd) / Math.sqrt(ess) + 0.01)
    expect(Math.abs((row?.sd ?? 0) / exact.sd - 1)).toBeLessThan(0.4)
    expect(s.state.accepted / s.state.t).toBeGreaterThan(0.1)
  })

  it('logs the acceptance, the likelihood estimate and the parameters on the tape', () => {
    const s = session(method, {
      steps: 5,
      mode: 'eager',
      config: { numIterations: 5, particles: 20 },
      logs: 'all',
    })
    const names = new Set(s.run().events.flatMap((e) => (e.type === 'metric' || e.type === 'tensor' ? [e.name] : [])))
    for (const p of ['pmcmc/accept', 'pmcmc/logLik', 'pmcmc/theta']) expect(names.has(p), p).toBe(true)
  })

  it('resumes from a serialised state, snapshots on the cadence of a clock and cancels between steps', () => {
    const config = { numIterations: 30, particles: 20, proposal: 'adaptive' as const }
    expectResumeEqual(method, { config, steps: 30, at: 12 })
    expectSnapshotCadence(method, { config, steps: 30, budgetMs: 400, everyMs: 100, tick: 10 })
    expectCancelAndContinue(method, { config, steps: 30, cancelAfterSlices: 4 })
  })

  it('stops after a step budget under the anytime driver and the posterior reads at every step', () => {
    const s = session(method, { steps: 200, config: { numIterations: 200, particles: 20 } })
    s.advance({ steps: 15 })
    expect(s.t).toBe(15)
    expect(method.posterior?.(s.state, s.config).status).toBe('stopped')
  })
})

describe('particle Gibbs with ancestor sampling', () => {
  const method = particleGibbsMethod(family, Y)

  it('recovers the exact posterior of the autoregression coefficient within its Monte Carlo error', () => {
    const s = session(method, {
      steps: 500,
      mode: 'eager',
      seed: 2,
      config: { numIterations: 500, particles: 24, thetaMoves: 3, stepSize: 0.3 },
    })
    expect(s.status).toBe('completed')
    const post = method.posterior?.(s.state, s.config)
    const row = post?.summary()[0]
    const ess = row?.ess ?? 1
    expect(Math.abs((row?.mean ?? 0) - exact.mean)).toBeLessThan((4 * exact.sd) / Math.sqrt(ess) + 0.01)
    expect(Math.abs((row?.sd ?? 0) / exact.sd - 1)).toBeLessThan(0.4)
    expect(s.state.path.length).toBe(Y.length + 1)
  })

  it('resumes from a serialised state, snapshots on the cadence of a clock and cancels between steps', () => {
    const config = { numIterations: 30, particles: 12 }
    expectResumeEqual(method, { config, steps: 30, at: 12 })
    expectSnapshotCadence(method, { config, steps: 30, budgetMs: 400, everyMs: 100, tick: 10 })
    expectCancelAndContinue(method, { config, steps: 30, cancelAfterSlices: 4 })
  })

  it('check says why a model without a transition density is refused, and the run throws that reason', () => {
    const { logTransition: _drop, ...blind } = family.model(new Float64Array([0.5]))
    const refused = particleGibbsMethod({ ...family, model: () => blind }, Y)
    const verdict = refused.check?.({ ...family, model: () => blind })
    expect(verdict?.applies).toBe(false)
    expect(verdict?.missing).toEqual(['logTransition'])
    expect(() => session(refused, { steps: 1, mode: 'eager' })).toThrow(/transition density/)
    expect(method.check?.({ ...family, logInitial: undefined as never })?.missing).toEqual(['logInitial'])
  })
})
