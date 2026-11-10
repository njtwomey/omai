// Nested sampling as a method: the evidence and the posterior of shared programs against the closed form (a normal mean, a
// linear-Gaussian observation of a correlated pair) and against the grid posterior's evidence (a bounded model with a
// uniform and a log-normal coordinate), within the error the run states; resumes from a serialised state, snapshots on the
// cadence of a fake clock, cancels between steps, and its `check` says why a discrete model does not apply.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { DISCRETE, MODELS, Y4 } from '../../../testing/probprog.ts'
import { item } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import { normalNormal } from '../validation/exact.ts'
import { gridPosterior } from '../validation/grid.ts'
import { nestedMethod } from './nested-method.ts'

const exactNormal = normalNormal({ priorMean: 0, priorSd: 2, sigma: 1.5, y: Y4 })

describe('nested sampling as a method', () => {
  it('finds the closed-form evidence and posterior of a normal mean within its stated error', () => {
    const method = nestedMethod(MODELS['normalMean'] as never)
    const s = session(method, { steps: 20000, mode: 'eager', seed: 1, config: { numLive: 100, innerSteps: 3 } })
    expect(s.status).toBe('converged')
    const post = method.posterior?.(s.state, s.config)
    const z = post?.logEvidence
    expect(Math.abs((z?.value ?? 0) - exactNormal.logEvidence)).toBeLessThan(4 * (z?.standardError ?? 0) + 0.02)
    expect(post?.status).toBe('converged')
    expect(Math.abs(item(post?.mean('mu') ?? 0) - item(exactNormal.posterior.mean()))).toBeLessThan(0.1)
    expect(Math.abs(item(post?.variance('mu') ?? 0) / item(exactNormal.posterior.variance()) - 1)).toBeLessThan(0.3)
  })

  it('finds the evidence of a correlated pair under a linear-Gaussian observation', () => {
    // y = x0 + 2 x1 + N(0, 1) with x ~ N((0, 1), [[1, .3], [.3, 2]]): y ~ N(2, 11.2) marginally, observed at 0.7.
    const exact = -0.5 * Math.log(2 * Math.PI * 11.2) - (0.7 - 2) ** 2 / (2 * 11.2)
    const method = nestedMethod(MODELS['mvn'] as never)
    const s = session(method, { steps: 20000, mode: 'eager', seed: 2, config: { numLive: 100, innerSteps: 3 } })
    const z = method.posterior?.(s.state, s.config).logEvidence
    expect(Math.abs((z?.value ?? 0) - exact)).toBeLessThan(4 * (z?.standardError ?? 0) + 0.02)
  })

  it('agrees with the grid posterior evidence on a bounded model (uniform and log-normal coordinates)', () => {
    const grid = gridPosterior(MODELS['bounded'] as never, undefined, { points: 61, maxRefinements: 0 })
    const method = nestedMethod(MODELS['bounded'] as never)
    const s = session(method, { steps: 20000, mode: 'eager', seed: 3, config: { numLive: 100, innerSteps: 3 } })
    const post = method.posterior?.(s.state, s.config)
    const z = post?.logEvidence
    expect(Math.abs((z?.value ?? 0) - grid.logEvidence)).toBeLessThan(4 * (z?.standardError ?? 0) + 0.02)
    // The posterior mean of the uniform coordinate, on the constrained scale, against the grid's.
    expect(Math.abs(item(post?.mean('u') ?? 0) - grid.marginal('u').mean)).toBeLessThan(0.2)
  }, 20_000)

  it('has an estimate, the error and the likelihood threshold at every step, and a partial posterior before the end', () => {
    const method = nestedMethod(MODELS['normalMean'] as never)
    const s = session(method, { steps: 20000, config: { numLive: 100 } })
    s.advance({ steps: 40 })
    const partial = method.posterior?.(s.state, s.config)
    expect(partial?.status).toBe('stopped')
    expect(Number.isFinite(partial?.logEvidence?.value)).toBe(true)
    expect(partial?.diagnostics['partial']).toBe(true)
    const tape = session(method, { steps: 30, mode: 'eager', config: { numLive: 100 }, logs: 'all' }).run()
    const names = new Set(tape.events.flatMap((e) => (e.type === 'metric' || e.type === 'tensor' ? [e.name] : [])))
    for (const p of ['ns/logZ', 'ns/logZError', 'ns/logVolume', 'ns/logL', 'ns/information', 'ns/evaluations'])
      expect(names.has(p), p).toBe(true)
  })

  it('resumes from a serialised state, snapshots on the cadence of a clock and cancels between steps', () => {
    const method = nestedMethod(MODELS['normalMean'] as never)
    const config = { numLive: 50, tolerance: 0.2 }
    expectResumeEqual(method, { config, steps: 600, at: 30 })
    expectSnapshotCadence(method, { config, steps: 600, budgetMs: 400, everyMs: 100, tick: 10 })
    expectCancelAndContinue(method, { config, steps: 600, cancelAfterSlices: 5 })
  })

  it('check says why a model with discrete latent sites does not apply, and a run throws that reason', () => {
    const verdict = nestedMethod(DISCRETE['mixture'] as never).check?.(analyse(DISCRETE['mixture'] as never, undefined))
    expect(verdict?.applies).toBe(false)
    expect(verdict?.reason).toMatch(/discrete/)
    expect(() => session(nestedMethod(DISCRETE['mixture'] as never), { steps: 1, mode: 'eager' })).toThrow(/discrete/)
    expect(
      nestedMethod(MODELS['normalMean'] as never).check?.(analyse(MODELS['normalMean'] as never, undefined))?.applies,
    ).toBe(true)
  })
})
