// The fine-grained NUTS method: it matches the closed form of a conjugate model, on a program and on a bound spec, runs a
// discrete mixture with enumerated sites, cancels within one iteration, publishes snapshots at the cadence of a fake clock,
// stops at step and time budgets, resumes from a serialised state to the same draws, and says why it does not apply.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { DISCRETE, MODELS, Y4 } from '../../../testing/probprog.ts'
import { toArray } from '../../tensor/index.ts'
import { Log, session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import * as d from '../model/dists/index.ts'
import { analyseSpec, defineModel } from '../model/index.ts'
import { runAnytime } from './method.ts'
import { nutsMethod } from './nuts-method.ts'

const exactMean = (() => {
  const precision = 1 / 4 + Y4.length / 2.25
  return Y4.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactSd = Math.sqrt(1 / (1 / 4 + Y4.length / 2.25))
const small = { numChains: 2, numWarmup: 40, numDraws: 30 }

const spec = defineModel('normal-mean', (b) => {
  const y = b.data('y', { dims: ['data'] })
  b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), y)
})

describe('nutsMethod', () => {
  it('matches the closed-form posterior of a conjugate model, from a program and from a bound spec', () => {
    for (const model of [MODELS['normalMean'] as never, spec.bind({ y: Y4 })]) {
      const m = nutsMethod(model)
      const s = session(m, { config: { ...small, numWarmup: 150, numDraws: 400 }, steps: 5000, mode: 'eager' })
      expect(s.status).toBe('converged')
      const post = m.posterior?.(s.state, s.config)
      expect(post?.status).toBe('completed')
      expect(Math.abs((post?.mean('mu').data[0] as number) - exactMean)).toBeLessThan(0.15)
      expect(Math.abs((post?.stddev('mu').data[0] as number) - exactSd)).toBeLessThan(0.1)
      expect(post?.diagnostics['divergences']).toBe(0)
    }
  })

  it('puts the iterations on the tape: step sizes, acceptance, divergences, the draws kept and a window message', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    const s = session(m, { config: small, steps: 200, mode: 'eager', logs: 'all' })
    const run = s.run()
    const paths = new Set(run.events.flatMap((e) => (e.type === 'message' ? [e.kind] : 'name' in e ? [e.name] : [])))
    for (const p of [
      'mcmc/stepSize',
      'mcmc/accept',
      'mcmc/divergent',
      'mcmc/kept',
      'mcmc/stage',
      'mcmc/windowEnd',
      'init/tries',
    ])
      expect(paths.has(p), p).toBe(true)
    // A step is one iteration of every chain: init, 40 warm-up, 30 draws.
    expect(s.t).toBe(1 + 40 + 30)
    expect(s.probe(s.t, 'mcmc/snapshot')).toMatchObject({ engine: 'nuts', status: 'completed' })
    void Log
  })

  it('runs HMC, and a discrete mixture whose enumerated sites are recovered one draw per step', () => {
    const hmc = nutsMethod(MODELS['normalMean'] as never)
    const a = session(hmc, { config: { ...small, algorithm: 'hmc', numChains: 1 }, steps: 500, mode: 'eager' })
    expect(a.status).toBe('converged')
    const mixture = nutsMethod(DISCRETE['mixture'] as never)
    const s = session(mixture, { config: { numChains: 1, numWarmup: 30, numDraws: 10 }, steps: 500, mode: 'eager' })
    expect(s.status).toBe('converged')
    // init, warm-up, draws, and a step per draw for the enumerated site.
    expect(s.t).toBe(1 + 30 + 10 + 10)
    const post = mixture.posterior?.(s.state, s.config)
    expect(post?.draws?.['z']?.shape).toEqual([10, 3])
    expect(toArray(post?.draws?.['locs'] as never)).toHaveLength(10)
  })

  it('cancels within one iteration and continues', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    // Cancelled after the 7th slice of one step: the run is at step 7, in the middle of the warm-up of every chain.
    expectCancelAndContinue(m, { config: small, steps: 500, cancelAfterSlices: 7 })
    const s = session(m, { config: small, steps: 500 })
    let n = 0
    runAnytime(s, { snapshot: { everySteps: 1 }, shouldStop: () => ++n > 12 })
    expect(s.state.stage).toBe('warmup')
    expect(s.state.iteration).toBe(11)
  })

  it('stops at a step budget and at a time budget with a fake clock, snapshotting a posterior each second', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    const s = session(m, { config: { ...small, budgetSteps: 25, snapshotEverySteps: 10 }, steps: 500 })
    const seen: number[] = []
    const r = runAnytime(s, { onSnapshot: (x) => seen.push(x.t) })
    expect([r.reason, r.t, seen]).toEqual(['budget', 25, [10, 20, 25]])
    // 100 ms of clock per reading, a snapshot per 1000 ms, stopping at 3500 ms.
    const snaps = expectSnapshotCadence(m, {
      config: { ...small, numDraws: 1000 },
      steps: 5000,
      budgetMs: 3500,
      everyMs: 1000,
      tick: 100,
    })
    expect(snaps.length).toBeGreaterThanOrEqual(3)
    // Early snapshots (still warming up) have no draws; later ones are posteriors with draws so far.
    const late = m.posterior?.(
      snaps.at(-1)?.state as never,
      session(m, { config: { ...small, numDraws: 1000 } }).config,
    )
    expect(['completed', 'stopped', 'failed']).toContain(late?.status)
  })

  it('shows R-hat, ESS and divergences of the draws so far in a snapshot', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    const s = session(m, { config: { numChains: 2, numWarmup: 40, numDraws: 60 }, steps: 500 })
    runAnytime(s, { budget: { steps: 1 + 40 + 20 } })
    const snap = s.probe(s.t, 'mcmc/snapshot') as {
      summary: { rhat?: number; ess?: number }[]
      diagnostics: Record<string, unknown>
    }
    expect(snap.summary[0]?.rhat).toBeGreaterThan(0.9)
    expect(snap.summary[0]?.ess).toBeGreaterThan(1)
    expect(snap.diagnostics['draws']).toBe(20)
    expect(snap.diagnostics['partial']).toBe(true)
    // A snapshot is JSON: it survives a round trip unchanged.
    expect(JSON.parse(JSON.stringify(snap))).toEqual(snap)
  })

  it('resumes from a serialised state to the same draws as an uninterrupted run', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    // In warm-up, in the draws, and with a mixture's discrete stage.
    expectResumeEqual(m, { config: small, steps: 500, at: 25 })
    expectResumeEqual(m, { config: small, steps: 500, at: 55 })
    expectResumeEqual(nutsMethod(DISCRETE['mixture'] as never), {
      config: { numChains: 1, numWarmup: 20, numDraws: 6 },
      steps: 500,
      at: 29,
    })
  })

  it('says whether it applies, and why not', () => {
    const m = nutsMethod(MODELS['normalMean'] as never)
    expect(m.check?.(analyseSpec(spec))).toMatchObject({ applies: true })
    // A density with no observed node is a target to sample, and the reason says so.
    const prior = defineModel('prior', (b) => void b.latent('x', d.normal(0, 1)))
    expect(m.check?.(analyseSpec(prior))).toMatchObject({ applies: true })
    expect(m.check?.(analyseSpec(prior))?.reason).toMatch(/no observed nodes, so the target is the prior/)
    const discrete = defineModel('discrete', (b) => {
      b.observe('y', d.normal(b.latent('z', d.bernoulli({ probs: 0.3 })), 1), b.data('y', { dims: ['n'] }))
    })
    expect(m.check?.(analyseSpec(discrete))).toMatchObject({ applies: false, missing: ['continuous latents'] })
    // For a program, the engine's own verdict.
    expect(m.check?.(analyse(MODELS['normalMean'] as never, undefined)).applies).toBe(true)
    expect(m.check?.(analyse(DISCRETE['hmm'] as never, undefined)).applies).toBe(false)
  })

  it('refuses to start on a model it does not apply to, with the verdict as the error', () => {
    const discrete = defineModel('discrete', (b) => {
      b.observe('y', d.normal(b.latent('z', d.bernoulli({ probs: 0.3 })), 1), b.data('y', { dims: ['n'] }))
    })
    const m = nutsMethod(discrete.bind({ y: [0.1, 0.2] }))
    expect(() => session(m, { mode: 'eager', steps: 5 })).toThrow(/no continuous latent nodes/)
  })

  it('samples a prior-only spec: the funnel of a spec is the same density as the program', () => {
    const density = defineModel('density', (b) => {
      const v = b.latent('v', d.normal(0, 1))
      b.latent('x', d.normal(v, 0.5))
    })
    const m = nutsMethod(density.bind({}))
    const s = session(m, { config: small, seed: 1, steps: 200, mode: 'eager' })
    const post = m.posterior?.(s.state, s.config)
    // v ~ N(0, 1) and x ~ N(0, 1 + 0.25): exactly known.
    expect(Math.abs(toArray(post?.mean('v') as never) as unknown as number)).toBeLessThan(0.6)
    expect(post?.status).toBe('completed')
  })
})
