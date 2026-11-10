// The kernel methods: each matches a closed-form posterior on a small model, puts its own tape on the run, cancels within one
// iteration, resumes from a serialised state to the same draws, publishes anytime snapshots, and says why it does not apply.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { MODELS, Y4 } from '../../../testing/probprog.ts'
import { type Config, session } from '../../trace/index.ts'
import { analyse } from '../analysis.ts'
import * as d from '../model/dists/index.ts'
import { analyseSpec, defineModel } from '../model/index.ts'
import {
  barkerMethod,
  ellipticalSliceMethod,
  hmcMethod,
  malaMethod,
  mclmcMethod,
  rwmMethod,
  sghmcMethod,
  sgldMethod,
  sliceMethod,
} from './mcmc-methods.ts'
import { runAnytime } from './method.ts'

const exactMean = (() => {
  const precision = 1 / 4 + Y4.length / 2.25
  return Y4.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactSd = Math.sqrt(1 / (1 / 4 + Y4.length / 2.25))

// The `mvn` program: x ~ N((0, 1), [[1, 0.3], [0.3, 2]]), y = x0 + 2 x1 + N(0, 1) observed at 0.7.
const mvnTruth = (() => {
  const s = [
    [1, 0.3],
    [0.3, 2],
  ] as const
  const det = s[0][0] * s[1][1] - s[0][1] * s[1][0]
  const inv = [
    [s[1][1] / det, -s[0][1] / det],
    [-s[1][0] / det, s[0][0] / det],
  ] as const
  const a = [1, 2] as const
  const p = [
    [inv[0][0] + a[0] * a[0], inv[0][1] + a[0] * a[1]],
    [inv[1][0] + a[1] * a[0], inv[1][1] + a[1] * a[1]],
  ] as const
  const pdet = p[0][0] * p[1][1] - p[0][1] * p[1][0]
  const cov = [
    [p[1][1] / pdet, -p[0][1] / pdet],
    [-p[1][0] / pdet, p[0][0] / pdet],
  ] as const
  const b = [inv[0][1] * 1 + a[0] * 0.7, inv[1][1] * 1 + a[1] * 0.7] as const
  return {
    mean: [cov[0][0] * b[0] + cov[0][1] * b[1], cov[1][0] * b[0] + cov[1][1] * b[1]],
    sd: [Math.sqrt(cov[0][0]), Math.sqrt(cov[1][1])],
  }
})()

const Y10 = [0.2, -0.4, 1.1, 0.7, -1.3, 0.5, 2.0, -0.1, 0.9, 0.3]
const subTruth = { mean: Y10.reduce((a, v) => a + v, 0) / 11, sd: 1 / Math.sqrt(11) }

const normalMean = MODELS['normalMean'] as never
const run = (
  method: ReturnType<typeof rwmMethod>,
  config: Config,
  steps = 20000,
): {
  readonly mean: number
  readonly sd: number
  readonly status: string | undefined
  readonly post: ReturnType<NonNullable<typeof method.posterior>>
} => {
  const s = session(method, { config, steps, mode: 'eager', seed: 3 })
  const post = method.posterior?.(s.state, s.config) as never
  const x = Array.from((post as { draws: Record<string, { data: Float64Array }> }).draws['mu']?.data ?? [])
  const n = x.length
  const mean = x.reduce((a, v) => a + v, 0) / n
  const sd = Math.sqrt(x.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1))
  return { mean, sd, status: s.status, post }
}

describe('kernel methods on a conjugate model', () => {
  const cases: [string, (m: never) => ReturnType<typeof rwmMethod>, Config][] = [
    ['rwm', rwmMethod, { numChains: 2, numWarmup: 300, numDraws: 1500 }],
    ['mala', malaMethod, { numChains: 2, numWarmup: 200, numDraws: 800 }],
    ['barker', barkerMethod, { numChains: 2, numWarmup: 200, numDraws: 1500 }],
    ['hmc', hmcMethod, { numChains: 2, numWarmup: 200, numDraws: 600, numSteps: 5 }],
    ['slice', sliceMethod, { numChains: 2, numWarmup: 100, numDraws: 800 }],
    ['elliptical-slice', ellipticalSliceMethod, { numChains: 2, numWarmup: 50, numDraws: 800 }],
  ]
  for (const [name, make, config] of cases)
    it(`${name} matches the closed-form posterior`, () => {
      const r = run(make(normalMean), config)
      expect(r.status).toBe('converged')
      expect(Math.abs(r.mean - exactMean)).toBeLessThan(0.15)
      expect(Math.abs(r.sd - exactSd)).toBeLessThan(0.1)
    })
})

describe('mclmcMethod', () => {
  it('matches the linear-Gaussian posterior of a two-dimensional model, to within its step-size bias', () => {
    const m = mclmcMethod(MODELS['mvn'] as never)
    const s = session(m, {
      config: { numChains: 2, numWarmup: 400, numDraws: 1500 },
      steps: 20000,
      mode: 'eager',
      seed: 2,
    })
    expect(s.status).toBe('converged')
    const post = m.posterior?.(s.state, s.config)
    const x = Array.from(post?.draws?.['x']?.data ?? [])
    const n = x.length / 2
    for (const j of [0, 1]) {
      const v = Float64Array.from({ length: n }, (_, i) => x[2 * i + j] ?? 0)
      const mean = v.reduce((a, b) => a + b, 0) / n
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1))
      expect(Math.abs(mean - (mvnTruth.mean[j] as number))).toBeLessThan(0.15)
      expect(Math.abs(sd - (mvnTruth.sd[j] as number))).toBeLessThan(0.15)
    }
  })

  it('needs two unconstrained coordinates, and says so', () => {
    expect(() => session(mclmcMethod(normalMean), { config: { numWarmup: 20 }, steps: 10 })).toThrow(
      /two unconstrained/,
    )
  })
})

describe('elliptical slice on a model with several Gaussian latents', () => {
  it('matches the multivariate normal posterior and the logistic model applies', () => {
    const m = ellipticalSliceMethod(MODELS['mvn'] as never)
    const s = session(m, {
      config: { numChains: 2, numWarmup: 50, numDraws: 1500 },
      steps: 20000,
      mode: 'eager',
      seed: 4,
    })
    const x = Array.from(m.posterior?.(s.state, s.config).draws?.['x']?.data ?? [])
    const n = x.length / 2
    const mean0 = Array.from({ length: n }, (_, i) => x[2 * i] ?? 0).reduce((a, b) => a + b, 0) / n
    expect(Math.abs(mean0 - (mvnTruth.mean[0] as number))).toBeLessThan(0.15)
    expect(
      ellipticalSliceMethod(MODELS['logistic'] as never).check?.(analyse(MODELS['logistic'] as never, undefined))
        .applies,
    ).toBe(true)
  })

  it('says why it does not apply to a hierarchical or non-Gaussian model', () => {
    const m = ellipticalSliceMethod(MODELS['schools'] as never)
    const v = m.check?.(analyse(MODELS['schools'] as never, undefined))
    expect(v).toMatchObject({ applies: false, missing: ['Gaussian priors'] })
    expect(
      ellipticalSliceMethod(MODELS['gammaPoisson'] as never).check?.(
        analyse(MODELS['gammaPoisson'] as never, undefined),
      ).applies,
    ).toBe(false)
    const spec = defineModel('hier', (b) => {
      const mu = b.latent('mu', d.normal(0, 1))
      b.observe('y', d.normal(b.latent('theta', d.normal(mu, 1)), 1), b.data('y', { dims: ['n'] }))
    })
    expect(m.check?.(analyseSpec(spec)).applies).toBe(false)
    expect(() => session(m, { steps: 5 })).toThrow(/elliptical-slice/)
  })
})

describe('the stochastic-gradient methods', () => {
  const sub = MODELS['subsampled'] as never
  for (const [name, make, config] of [
    ['sgld', sgldMethod, { numChains: 2, numWarmup: 300, numDraws: 2500, stepSize: 0.005 }],
    ['sghmc', sghmcMethod, { numChains: 2, numWarmup: 300, numDraws: 1500, stepSize: 0.01, numSteps: 5, friction: 1 }],
  ] as const)
    it(`${name} samples the posterior of a subsampled plate, with minibatch gradients`, () => {
      const m = make(sub)
      const s = session(m, { config, steps: 20000, mode: 'eager', seed: 5 })
      const x = Array.from(m.posterior?.(s.state, s.config).draws?.['mu']?.data ?? [])
      const mean = x.reduce((a, v) => a + v, 0) / x.length
      const sd = Math.sqrt(x.reduce((a, v) => a + (v - mean) ** 2, 0) / (x.length - 1))
      expect(Math.abs(mean - subTruth.mean)).toBeLessThan(0.2)
      // Minibatch noise widens the draws a little: the spread is the posterior's within a factor.
      expect(sd).toBeGreaterThan(subTruth.sd * 0.6)
      expect(sd).toBeLessThan(subTruth.sd * 2)
    })

  it('uses the exact gradient on a program with no subsampled plate', () => {
    const m = sgldMethod(normalMean)
    const s = session(m, {
      config: { numChains: 2, numWarmup: 200, numDraws: 3000, stepSize: 0.02 },
      steps: 20000,
      mode: 'eager',
    })
    const x = Array.from(m.posterior?.(s.state, s.config).draws?.['mu']?.data ?? [])
    expect(Math.abs(x.reduce((a, v) => a + v, 0) / x.length - exactMean)).toBeLessThan(0.2)
  })
})

describe('the method convention', () => {
  const small = { numChains: 2, numWarmup: 40, numDraws: 30 }
  const methods = [
    ['rwm', rwmMethod, small, normalMean],
    ['mala', malaMethod, small, normalMean],
    ['barker', barkerMethod, small, normalMean],
    ['hmc', hmcMethod, small, normalMean],
    ['slice', sliceMethod, small, normalMean],
    ['elliptical-slice', ellipticalSliceMethod, small, normalMean],
    ['mclmc', mclmcMethod, { ...small, numWarmup: 60 }, MODELS['mvn'] as never],
    ['sgld', sgldMethod, { ...small, stepSize: 0.01 }, MODELS['subsampled'] as never],
  ] as const
  for (const [name, make, config, model] of methods) {
    it(`${name}: its tape, a cancel within one step, a resume that equals the run, and anytime snapshots`, () => {
      const m = make(model)
      const s = session(m, { config, steps: 500, mode: 'eager', logs: 'all' })
      const run0 = s.run()
      const paths = new Set(run0.events.flatMap((e) => (e.type === 'message' ? [e.kind] : 'name' in e ? [e.name] : [])))
      for (const p of ['mcmc/accept', 'mcmc/kept', 'mcmc/stage', 'mcmc/logDensity', 'init/tries'])
        expect(paths.has(p), p).toBe(true)
      // A step is one iteration of every chain: init, the warm-up, the draws.
      expect(s.t).toBe(1 + config.numWarmup + config.numDraws)
      expect(s.probe(s.t, 'mcmc/snapshot')).toMatchObject({ engine: name, status: 'completed' })
      expectCancelAndContinue(m, { config, steps: 500, cancelAfterSlices: 7 })
      expectResumeEqual(m, { config, steps: 500, at: 20 })
      expectResumeEqual(m, { config, steps: 500, at: 1 + config.numWarmup + 8 })
      const snaps = expectSnapshotCadence(m, {
        config: { ...config, numDraws: 400 },
        steps: 5000,
        budgetMs: 2500,
        everyMs: 1000,
        tick: 100,
      })
      expect(snaps.length).toBeGreaterThanOrEqual(2)
    })
  }

  it('stops at a step budget and says whether it applies', () => {
    const m = rwmMethod(normalMean)
    const s = session(m, { config: { ...small, budgetSteps: 12, snapshotEverySteps: 5 }, steps: 500 })
    const seen: number[] = []
    const r = runAnytime(s, { onSnapshot: (x) => seen.push(x.t) })
    expect([r.reason, r.t, seen]).toEqual(['budget', 12, [5, 10, 12]])
    const noLatent = defineModel('none', (b) => void b.observe('y', d.normal(0, 1), b.data('y', { dims: ['n'] })))
    expect(m.check?.(analyseSpec(noLatent))).toMatchObject({ applies: false, missing: ['continuous latents'] })
  })
})

describe('mass matrix variants', () => {
  it('hmc learns a dense and a low-rank mass matrix of the correlated posterior, and the draws match the closed form', {
    timeout: 30_000,
  }, () => {
    for (const config of [{ dense: true }, { dense: true, lowRank: 1 }, { dense: false }]) {
      const m = hmcMethod(MODELS['mvn'] as never)
      const s = session(m, {
        config: { numChains: 2, numWarmup: 300, numDraws: 500, numSteps: 6, ...config },
        steps: 20000,
        mode: 'eager',
        seed: 7,
      })
      const x = Array.from(m.posterior?.(s.state, s.config).draws?.['x']?.data ?? [])
      const n = x.length / 2
      const mean0 = Array.from({ length: n }, (_, i) => x[2 * i] ?? 0).reduce((a, b) => a + b, 0) / n
      expect(Math.abs(mean0 - (mvnTruth.mean[0] as number)), JSON.stringify(config)).toBeLessThan(0.15)
      const parameters = m.posterior?.(s.state, s.config).diagnostics['parameters'] as { inverseMassMatrix: number[] }[]
      expect(parameters[0]?.inverseMassMatrix.length).toBe(config.dense ? 4 : 2)
    }
  })
})
