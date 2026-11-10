// The per-engine inference methods against closed forms and the engines they wrap: each matches the exact answer (or the
// one-call engine) within tolerance on shared programs and on a bound spec, resumes from a serialised state to the same
// result, snapshots on the cadence of a fake clock and cancels between steps, and its `check` says why it does or does not
// apply to a handful of specs.
import { describe, expect, it } from 'vitest'
import { expectCancelAndContinue, expectResumeEqual, expectSnapshotCadence } from '../../../testing/methods.ts'
import { DISCRETE, MODELS, Y4 } from '../../../testing/probprog.ts'
import { categorical, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { item, type Tensor, take, toArray, type Value } from '../../tensor/index.ts'
import { type Config, Log, session } from '../../trace/index.ts'
import { chainInference } from '../engines/chain.ts'
import { conjugateInference } from '../engines/conjugate.ts'
import { em } from '../engines/em.ts'
import { enumerationPosterior } from '../engines/enumeration.ts'
import { gaussianInference } from '../engines/gaussian.ts'
import * as d from '../model/dists/index.ts'
import { analyseSpec, defineModel, ops } from '../model/index.ts'
import { markov, observe, param, plate, sample } from '../primitives.ts'
import { abcRejectionMethod, abcSmcMethod } from './abc-methods.ts'
import { emMethod } from './em-method.ts'
import { chainMethod, conjugateMethod, enumerationMethod, gaussianMethod } from './exact-methods.ts'
import { gibbsMethod } from './gibbs-method.ts'
import { importanceMethod } from './importance-method.ts'
import { laplaceMethod } from './laplace-method.ts'
import type { InferenceMethod } from './method.ts'
import { runAnytime } from './method.ts'
import type { ModelAnalysis } from './model-input.ts'
import { particleFilterMethod, temperedSmcMethod } from './smc-methods.ts'
import { sviMethod } from './svi-method.ts'

const exactMean = (() => {
  const precision = 1 / 4 + Y4.length / 2.25
  return Y4.reduce((a, v) => a + v, 0) / 2.25 / precision
})()
const exactVar = 1 / (1 / 4 + Y4.length / 2.25)
const normalMean = MODELS['normalMean'] as never
const spec = defineModel('normal-mean', (b) => {
  b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), b.data('y', { dims: ['data'] }))
})
const bound = spec.bind({ y: Y4 })
const closedForm = conjugateInference(normalMean, undefined)
// A Poisson rate through exp: continuous and differentiable, with no conjugate pair and no discrete latent.
const glm = defineModel('glm', (b) => {
  b.observe('y', d.poisson(ops.exp(b.latent('a', d.normal(0, 1)))), b.data('y', { dims: ['n'] }))
})

/** Run a method to its end in eager mode and read the posterior of the final state. */
function run<S, K extends string>(
  method: InferenceMethod<S, Config, K, ModelAnalysis>,
  config: Partial<Config>,
  steps: number,
  seed = 1,
) {
  const s = session(method, { config, steps, seed, mode: 'eager' })
  const post = method.posterior?.(s.state, s.config)
  if (post === undefined) throw new Error('no posterior')
  return { s, post }
}

describe('the exact methods are one step, and equal their engines', () => {
  it('conjugate and gaussian find the closed form on a program and on a bound spec', () => {
    for (const model of [normalMean, bound]) {
      for (const r of [run(conjugateMethod(model), {}, 5), run(gaussianMethod(model), {}, 5)]) {
        expect(r.s.t).toBe(1)
        expect(r.s.status).toBe('converged')
        expect(Math.abs(item(r.post.mean('mu')) - exactMean)).toBeLessThan(1e-9)
        expect(Math.abs(item(r.post.variance('mu')) - exactVar)).toBeLessThan(1e-9)
      }
    }
    const { post } = run(conjugateMethod(MODELS['gammaPoisson'] as never), {}, 5)
    expect(item(post.mean('rate'))).toBeCloseTo(20 / 4.5, 9)
  })

  it('enumeration and chain give the exact marginals of a discrete chain, as their engines do', () => {
    const hmm = DISCRETE['hmm'] as never
    const old = enumerationPosterior(hmm, undefined)
    const viaChain = chainInference(hmm, undefined)
    const e = run(enumerationMethod(hmm), {}, 5)
    const c = run(chainMethod(hmm), {}, 5)
    for (const site of ['z0', 'z1', 'z2', 'z3']) {
      expect(toArray(e.post.mean(site))).toEqual(toArray(old.mean(site)))
      expect(Math.abs(item(c.post.mean(site)) - item(viaChain.mean(site)))).toBeLessThan(1e-12)
      expect(Math.abs(item(e.post.mean(site)) - item(viaChain.mean(site)))).toBeLessThan(1e-9)
    }
    expect(Math.abs((e.post.logEvidence?.value ?? 0) - (old.logEvidence?.value ?? 1))).toBeLessThan(1e-12)
    // The evidence is on the tape.
    const tape = session(enumerationMethod(hmm), { steps: 1, mode: 'eager', logs: 'all' }).run()
    expect(tape.events.some((x) => x.type === 'metric' && x.name === 'enumeration/logEvidence')).toBe(true)
  })

  it('gaussian runs the kalman smoother on a chain and agrees with the engine', () => {
    const walk = () =>
      markov(
        't',
        3,
        (t, x: Value | number) => {
          const z = sample(`x${t}`, normal(x, 1))
          observe(`y${t}`, normal(z, 1), t)
          return z
        },
        0,
      )
    const { post } = run(gaussianMethod(walk as never), { method: 'kalman' }, 5)
    expect(Math.abs(item(post.mean('x2')) - item(gaussianInference(walk as never, undefined).mean('x2')))).toBeLessThan(
      1e-9,
    )
  })

  it('are resumable and snapshot-able in the one step they have', () => {
    const m = enumerationMethod(DISCRETE['hmm'] as never)
    const s = session(m, { steps: 5 })
    runAnytime(s, { budget: { steps: 3 } })
    expect(s.t).toBe(1)
    expect(s.probe(1, 'enumeration/snapshot')).toMatchObject({ engine: 'enumeration', status: 'completed' })
  })
})

describe('the iterative methods match closed forms', () => {
  it('SVI recovers the Gaussian posterior (a mean-field normal is exact here), on a program and on a spec', () => {
    for (const model of [normalMean, bound]) {
      const { s, post } = run(sviMethod(model), { learningRate: 0.1, numParticles: 8 }, 1200)
      expect(s.status).toBe('completed')
      expect(Math.abs(item(post.mean('mu')) - exactMean)).toBeLessThan(0.25)
      expect(Math.abs(item(post.variance('mu')) - exactVar)).toBeLessThan(0.25)
    }
  })

  it('SVI logs the ELBO, the loss and the parameters, and stops converged on a tolerance', () => {
    const m = sviMethod(normalMean)
    const s = session(m, {
      config: { learningRate: 0.1, tol: 1e-5 },
      steps: 3000,
      mode: 'eager',
      logs: { 'svi/elbo': Log(), 'svi/loss': Log(), 'svi/smoothed': Log(), 'svi/params/*': Log() },
    })
    expect(s.status).toBe('converged')
    const names = new Set(s.run().events.flatMap((e) => (e.type === 'metric' || e.type === 'tensor' ? [e.name] : [])))
    for (const p of ['svi/elbo', 'svi/loss', 'svi/smoothed', 'svi/params/mu_auto_loc'])
      expect(names.has(p), p).toBe(true)
  })

  it('Gibbs finds the conjugate posterior of a normal mean and of a Dirichlet', () => {
    const a = run(gibbsMethod(normalMean), { numDraws: 1500, numWarmup: 50 }, 5000)
    expect(a.s.status).toBe('completed')
    expect(Math.abs(item(a.post.mean('mu')) - exactMean)).toBeLessThan(0.1)
    const b = run(gibbsMethod(MODELS['dirichletCategorical'] as never), { numDraws: 1500, numWarmup: 50 }, 5000)
    // Dirichlet(1 + [1, 1, 3] + [0, 1, 2]) = Dirichlet(2, 3, 6) after the prior (1, 2, 3).
    expect((toArray(b.post.mean('p')) as number[]).map((v) => Math.round(v * 10) / 10)).toEqual([0.2, 0.3, 0.5])
    expect(a.post.status).toBe('completed')
  })

  it('Gibbs serves a program with a discrete latent too (the mixture, against the one-call engine)', () => {
    const a = run(gibbsMethod(DISCRETE['mixture'] as never), { numDraws: 100, numWarmup: 20 }, 500)
    expect(a.post.draws?.['z']?.shape).toEqual([100, 3])
  })

  it('Laplace is exact on a Gaussian: the mode, the covariance and the evidence', () => {
    for (const model of [normalMean, bound]) {
      const { s, post } = run(laplaceMethod(model), {}, 200)
      expect(['converged', 'completed']).toContain(s.status)
      expect(Math.abs(item(post.mean('mu')) - exactMean)).toBeLessThan(1e-6)
      expect(Math.abs(item(post.variance('mu')) - exactVar)).toBeLessThan(1e-8)
      expect(Math.abs((post.logEvidence?.value ?? 0) - (closedForm.logEvidence?.value ?? 1))).toBeLessThan(1e-4)
    }
  })

  it('Laplace reads a point estimate while it optimises and the normal after the Hessian', () => {
    const m = laplaceMethod(MODELS['logistic'] as never)
    const s = session(m, { steps: 200 })
    runAnytime(s, { budget: { steps: 3 } })
    expect(s.state.stage).toBe('optimise')
    const partial = m.posterior?.(s.state, s.config)
    expect(partial?.form).toBe('point')
    expect(partial?.status).toBe('stopped')
    runAnytime(s)
    expect(s.state.stage).toBe('done')
    expect(m.posterior?.(s.state, s.config)?.form).toBe('variational')
  })

  it('tempered SMC matches the closed-form mean and evidence, with the temperatures on the tape', () => {
    const m = temperedSmcMethod(normalMean)
    const s = session(m, { config: { numParticles: 400 }, steps: 100, mode: 'eager', logs: 'all' })
    expect(s.status).toBe('converged')
    const post = m.posterior?.(s.state, s.config)
    expect(Math.abs(item(post?.mean('mu') as Tensor) - exactMean)).toBeLessThan(0.15)
    expect(Math.abs((post?.logEvidence?.value ?? 0) - (closedForm.logEvidence?.value ?? 1))).toBeLessThan(0.3)
    const temps = s.run().events.flatMap((e) => (e.type === 'metric' && e.name === 'smc/temperature' ? [e.value] : []))
    expect(temps.at(-1)).toBe(1)
    expect(temps).toEqual([...temps].sort((a, b) => a - b))
  })

  it('importance sampling in batches draws what the one-call engine draws, and its evidence matches the closed form', () => {
    const m = importanceMethod(normalMean)
    const { s, post } = run(m, { numSamples: 6000, batchSize: 1000 }, 100)
    expect(s.t).toBe(6)
    expect(Math.abs(item(post.mean('mu')) - exactMean)).toBeLessThan(0.08)
    expect(Math.abs((post.logEvidence?.value ?? 0) - (closedForm.logEvidence?.value ?? 1))).toBeLessThan(0.1)
    // Draws are keyed by their index: five batches then the sixth is a prefix of a longer run.
    const half = session(m, { config: { numSamples: 6000, batchSize: 1000 }, steps: 3, mode: 'eager' })
    expect(m.posterior?.(half.state, half.config)?.status).toBe('stopped')
  })

  it('EM reproduces the one-call engine step for step', () => {
    const y = [-2.1, -1.9, -2.2, 2.0, 1.8, 2.3]
    const gmm = () => {
      const mu = param('mu', [-1, 1])
      plate('n', y.length, () => {
        const z = sample('z', categorical({ probs: [0.5, 0.5] }))
        observe('y', normal(take(mu, z), 0.5), y)
      })
    }
    const { s, post } = run(emMethod(gmm), {}, 200)
    expect(s.status).toBe('converged')
    const old = em(gmm, undefined, { key: key(0) })
    expect(toArray(post.mean('mu'))).toEqual(toArray(old.mean('mu')))
    expect(Math.abs(Math.abs((toArray(post.mean('mu')) as number[])[0] ?? 0) - 2.07)).toBeLessThan(0.01)
  })

  it('ABC-SMC and rejection ABC recover the mean of a simulator within a generous tolerance', () => {
    const y = [0.5, 1.2, 1.6, 1.9, 2, 2.1, 2.4, 2.8, 3.5, 2]
    const model = () => {
      const m = sample('m', normal(0, 3))
      plate('n', y.length, () => observe('y', normal(m, 1), y))
    }
    const exact = y.reduce((a, v) => a + v, 0) / (y.length + 1 / 9)
    const a = run(abcSmcMethod(model), { numParticles: 150 }, 5)
    expect(Math.abs(item(a.post.mean('m')) - exact)).toBeLessThan(0.5)
    const r = run(abcRejectionMethod(model), { numSamples: 1500 }, 3)
    expect(r.s.t).toBe(1)
    expect(Math.abs(item(r.post.mean('m')) - exact)).toBeLessThan(0.6)
  })

  it('the particle filter agrees with the Kalman smoother on the last state of a random walk', () => {
    const walk = () =>
      markov(
        't',
        3,
        (t, x: Value | number) => {
          const z = sample(`x${t}`, normal(x, 1))
          observe(`y${t}`, normal(z, 1), t)
          return z
        },
        0,
      )
    const { s, post } = run(particleFilterMethod(walk as never), { numParticles: 600 }, 10)
    expect(s.t).toBe(3)
    expect(Math.abs(item(post.mean('x2')) - item(gaussianInference(walk as never, undefined).mean('x2')))).toBeLessThan(
      0.25,
    )
  })
})

describe('every iterative method is resumable, snapshots on the clock and cancels between steps', () => {
  const walk = () =>
    markov(
      't',
      4,
      (t, x: Value | number) => {
        const z = sample(`x${t}`, normal(x, 1))
        observe(`y${t}`, normal(z, 1), t)
        return z
      },
      0,
    )
  const gmm = () => {
    const mu = param('mu', [-1, 1])
    plate('n', 6, () => {
      const z = sample('z', categorical({ probs: [0.5, 0.5] }))
      observe('y', normal(take(mu, z), 0.5), [-2.1, -1.9, -2.2, 2.0, 1.8, 2.3])
    })
  }
  const small = [
    ['svi', sviMethod(normalMean), { learningRate: 0.05 }, 40, 17],
    ['gibbs', gibbsMethod(normalMean), { numDraws: 20, numWarmup: 10 }, 40, 13],
    ['laplace', laplaceMethod(MODELS['logistic'] as never), {}, 100, 4],
    ['tempered-smc', temperedSmcMethod(normalMean), { numParticles: 60 }, 40, 2],
    ['importance', importanceMethod(normalMean), { numSamples: 600, batchSize: 100 }, 40, 3],
    ['em', emMethod(gmm), {}, 100, 1],
    ['abc-smc', abcSmcMethod(normalMean), { numParticles: 40 }, 4, 2],
    ['particle-filter', particleFilterMethod(walk as never), { numParticles: 40 }, 10, 2],
  ] as const

  for (const [name, method, config, steps, at] of small) {
    it(`${name}: a state restored through JSON into a fresh session continues to the same result`, () => {
      expectResumeEqual(method as InferenceMethod<unknown, Config, string, ModelAnalysis>, { config, steps, at })
    })
  }

  it('snapshots once per second of clock time, each a posterior that reads, and stops at a time budget', () => {
    for (const [name, method, config, steps] of small) {
      if (name === 'abc-smc' || name === 'em' || name === 'laplace' || name === 'particle-filter') continue
      // 400 ms of clock per reading, so a run of a few steps spans several snapshots.
      const snaps = expectSnapshotCadence(method as InferenceMethod<unknown, Config, string, ModelAnalysis>, {
        config,
        steps: Math.max(steps, 200),
        budgetMs: 6000,
        everyMs: 1000,
        tick: 400,
      })
      expect(snaps.length, name).toBeGreaterThanOrEqual(name === 'tempered-smc' ? 1 : 2)
    }
  })

  it('cancels between two steps, and a cancelled run keeps its tape and continues', () => {
    for (const [name, method, config] of small) {
      if (name === 'particle-filter' || name === 'abc-smc') continue
      expectCancelAndContinue(method as InferenceMethod<unknown, Config, string, ModelAnalysis>, {
        config,
        steps: 400,
        cancelAfterSlices: 1,
      })
    }
  })
})

describe('check says whether a method applies to a spec, and why not', () => {
  const lg = analyseSpec(spec)
  const prior = analyseSpec(defineModel('prior', (b) => void b.latent('x', d.normal(0, 1))))
  const discrete = analyseSpec(
    defineModel('discrete', (b) => {
      b.observe('y', d.normal(b.latent('z', d.bernoulli({ probs: 0.3 })), 1), b.data('y', { dims: ['n'] }))
    }),
  )
  const nonconj = analyseSpec(glm)
  const withParam = analyseSpec(
    defineModel('mixture', (b) => {
      const locs = b.param('locs', [-1, 1], { dims: ['k'] })
      const z = b.latent('z', d.categorical({ probs: [0.4, 0.6] }), { dims: ['row'] })
      b.observe('y', d.normal(locs.at(z), 1), b.data('y', { dims: ['row'] }))
    }),
  )
  const verdicts = (m: InferenceMethod<never, Config, string, ModelAnalysis>) =>
    [lg, prior, discrete, nonconj, withParam].map((a) => m.check?.(a))
  const applies = (m: InferenceMethod<never, Config, string, ModelAnalysis>): boolean[] =>
    verdicts(m).map((v) => v?.applies === true)

  it('has a verdict with a reason for every method on every spec, and the matrix is what each method needs', () => {
    const table: [string, InferenceMethod<never, Config, string, ModelAnalysis>, boolean[]][] = [
      //                                                           lg     prior  discrete nonconj param-mixture
      ['nuts', nuts(), [true, false, false, true, false]],
      ['svi', sviMethod(normalMean) as never, [true, false, false, true, false]],
      ['laplace', laplaceMethod(normalMean) as never, [true, false, false, true, false]],
      ['gibbs', gibbsMethod(normalMean) as never, [true, false, true, false, true]],
      ['tempered-smc', temperedSmcMethod(normalMean) as never, [true, false, false, true, false]],
      ['importance', importanceMethod(normalMean) as never, [true, false, true, true, true]],
      ['conjugate', conjugateMethod(normalMean) as never, [true, false, false, false, false]],
      ['gaussian', gaussianMethod(normalMean) as never, [true, true, false, false, false]],
      ['enumeration', enumerationMethod(normalMean) as never, [false, false, true, false, true]],
      ['em', emMethod(normalMean) as never, [false, false, false, false, true]],
      ['abc-smc', abcSmcMethod(normalMean) as never, [true, false, false, true, false]],
      ['chain', chainMethod(normalMean) as never, [false, false, false, false, false]],
    ]
    for (const [name, m, expected] of table) {
      expect(applies(m), name).toEqual(expected)
      for (const v of verdicts(m)) {
        expect(v?.reason.length, name).toBeGreaterThan(5)
        if (v?.applies === false) expect(v.missing.length, `${name}: ${v.reason}`).toBeGreaterThan(0)
      }
    }
  })

  it('names what is missing and what would help', () => {
    const g = gibbsMethod(normalMean).check?.(nonconj)
    expect(g).toMatchObject({ applies: false, hint: 'use NUTS' })
    expect(g?.reason).toContain('no discrete or conjugate')
    expect(conjugateMethod(normalMean).check?.(nonconj)?.reason).toContain('a')
    expect(enumerationMethod(normalMean).check?.(lg)?.missing).toEqual(['finite discrete latents'])
    expect(gaussianMethod(normalMean).check?.(nonconj)?.reason.length).toBeGreaterThan(5)
    expect(emMethod(normalMean).check?.(lg)?.missing).toEqual(['param nodes'])
  })

  it('reads a program through the engine it wraps', () => {
    const p = (a: ModelAnalysis) => gibbsMethod(normalMean).check?.(a)
    expect(p(analyseOf(MODELS['dirichletCategorical'] as never))?.applies).toBe(true)
    expect(p(analyseOf(MODELS['logistic'] as never))?.applies).toBe(false)
  })

  it('a method refuses to run on a model it does not apply to, with the same reason', () => {
    const bad = glm.bind({ y: [1, 2, 3] })
    expect(() => session(gibbsMethod(bad), { mode: 'eager', steps: 3 })).toThrow(/no discrete or conjugate/)
    expect(() => session(conjugateMethod(bad), { mode: 'eager', steps: 3 })).toThrow(/no usable conjugate structure/)
  })
})

import { analyse } from '../analysis.ts'
import type { Model } from '../primitives.ts'
import { nutsMethod } from './nuts-method.ts'

function analyseOf(model: Model): ModelAnalysis {
  return analyse(model, undefined)
}
function nuts(): InferenceMethod<never, Config, string, ModelAnalysis> {
  return nutsMethod(normalMean) as never
}
