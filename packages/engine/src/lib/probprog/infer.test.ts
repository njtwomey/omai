import { describe, expect, it } from 'vitest'
import { DISCRETE, MODELS } from '../../testing/probprog.ts'
import { gamma, normal, poisson } from '../dists/index.ts'
import { key, normal as normalDraws } from '../random/index.ts'
import { add, exp, item, mul, take, tensor, toFlat, type Value } from '../tensor/index.ts'
import { policies, recorder } from '../trace/index.ts'
import {
  conjugateInference,
  definePosterior,
  enumerationEngine,
  explainPlan,
  factor,
  infer,
  logLikelihood,
  type Model,
  markov,
  nutsEngine,
  observe,
  plan,
  plate,
  pointPosterior,
  pointwiseLogLikelihood,
  posteriorPredictive,
  posteriorPredictiveCheck,
  priorPredictive,
  psisLoo,
  sample,
  samplePosterior,
  simulator,
  waic,
  weightedPosterior,
} from './index.ts'

const first = (m: Model): string => plan(m, undefined).entries[0]?.engine.name ?? ''

describe('the planner', () => {
  it('prefers exact engines and explains its ranking', () => {
    expect(first(MODELS['gammaPoisson'] as Model)).toBe('conjugate')
    expect(first(DISCRETE['mixture'] as Model)).not.toBe('enumeration')
    expect(first(DISCRETE['hmm'] as Model)).toBe('chain')
    const walk: Model = () => {
      markov(
        't',
        3,
        (t, x: Value | number) => {
          const z = sample(`x${t}`, normal(mul(x, 0.5), 1))
          observe(`y${t}`, normal(z, 1), t)
          return z
        },
        0,
      )
    }
    expect(first(walk)).toBe('gaussian')
    expect(first(MODELS['schools'] as Model)).toBe('nuts')
    const text = explainPlan(plan(MODELS['schools'] as Model, undefined))
    expect(text).toMatch(/^1\. nuts \(mcmc/)
    expect(text).toMatch(/enumeration \(does not apply\): continuous latent sites/)
  })

  it('respects a budget and lets the caller choose', () => {
    const p = plan(MODELS['schools'] as Model, undefined, { budget: 1e5 })
    expect(p.entries[0]?.engine.name).not.toBe('nuts')
    expect(p.entries.find((e) => e.engine.name === 'nuts')?.withinBudget).toBe(false)
    const sim: Model = () => {
      const m = sample('m', normal(0, 1))
      observe(
        'y',
        simulator('s', (k) => add(m, normalDraws(k, []))),
        0.5,
      )
    }
    expect(first(sim)).toBe('abc')
  })
})

describe('infer', () => {
  it('runs the planned engine as a traced child of an infer run', () => {
    const rec = recorder()
    const post = infer(MODELS['gammaPoisson'] as Model, undefined, { tracer: rec, policy: policies.standard })
    expect(post.engine).toBe('conjugate')
    expect(String(post.diagnostics['reason'])).toMatch(/conjugate prior/)
    expect(rec.run().events.some((e) => e.type === 'message' && e.kind === 'plan')).toBe(true)
  })

  it('takes an engine by name or value, and falls back when one fails', () => {
    const model = MODELS['normalMean'] as Model
    const byName = infer(model, undefined, { engine: 'laplace', key: key(0) })
    expect(byName.engine).toBe('laplace')
    expect(() => infer(model, undefined, { engine: 'nope' })).toThrow(/no engine/)
    const failing = {
      ...nutsEngine(),
      name: 'broken',
      applies: () => ({ applies: true, cost: 0, quality: 'exact' as const, reason: 'always' }),
      run: () => definePosterior({ form: 'samples', engine: 'broken', sites: [], status: 'failed', draws: {} }),
    }
    const post = infer(model, undefined, {
      engines: [failing, enumerationEngine(), ...[nutsEngine({ numChains: 1, numWarmup: 100, numDraws: 100 })]],
      key: key(1),
    })
    expect(post.engine).toBe('nuts')
    expect(post.diagnostics['tried']).toEqual([
      { engine: 'broken', error: 'failed' },
      { engine: 'nuts', status: 'completed' },
    ])
  })

  it('reports when nothing applies', () => {
    const nothing: Model = () => {
      factor('f', 0)
    }
    expect(() => infer(nothing, undefined, { engines: [enumerationEngine()] })).toThrow(/no engine applies/)
  })
})

describe('posteriors', () => {
  it('answers moments and quantiles the same way whatever the form', () => {
    const draws = samplePosterior({ x: tensor([1, 2, 3, 4]) }, { engine: 't' })
    expect(toFlat(draws.quantiles('x', [0.5]))).toEqual([2.5])
    const weighted = weightedPosterior({ x: tensor([0, 1]) }, [Math.log(1), Math.log(3)], { engine: 't' })
    expect(item(weighted.mean('x'))).toBeCloseTo(0.75, 14)
    expect(item(weighted.variance('x'))).toBeCloseTo(0.1875, 14)
    const point = pointPosterior({ mu: 2 }, { engine: 't' })
    expect(toFlat(point.sample(key(0), 3)['mu'] ?? tensor([]))).toEqual([2, 2, 2])
    const exact = conjugateInference(MODELS['normalMean'] as Model, undefined)
    const q = toFlat(exact.quantiles('mu', [0.5]))
    expect(q[0]).toBeCloseTo(item(exact.mean('mu')), 12)
    expect(exact.summary()[0]?.name).toBe('mu')
    expect(() => exact.mean('nope')).toThrow(/no site/)
  })
})

describe('predictives and criticism', () => {
  const y = [1.2, 0.4, 2.3, 1.7]
  const model = (data: { readonly y?: readonly number[]; readonly n: number }): void => {
    const mu = sample('mu', normal(0, 2))
    plate('n', data.n, () => sample('y', normal(mu, 1), { obs: data.y }))
  }
  const post = conjugateInference(model, { y, n: y.length })

  it('draws prior and posterior predictive data and pointwise log-likelihoods', () => {
    const prior = priorPredictive(model, { y, n: 4 }, { key: key(0), numSamples: 50 })
    expect(prior['y']?.shape).toEqual([50, 4])
    const rep = posteriorPredictive(model, post, { y, n: 4 }, { key: key(1), numSamples: 400 })
    const values = toFlat(rep['y'] ?? tensor([]))
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    expect(Math.abs(mean - item(post.mean('mu')))).toBeLessThan(0.15)
    // New covariates: predict five points where the data had four.
    expect(posteriorPredictive(model, post, { n: 5 }, { key: key(2), numSamples: 3 })['y']?.shape).toEqual([3, 5])
    const ll = logLikelihood(model, post, { y, n: 4 }, { key: key(3), numSamples: 200 })
    expect(ll['y']?.shape).toEqual([200, 4])
    const m = pointwiseLogLikelihood(ll)
    expect(m.shape).toEqual([200, 4])
    const w = waic(m)
    const l = psisLoo(m)
    // WAIC and LOO agree closely on a well-specified model with weak observations.
    expect(Math.abs(w.elpd - l.elpd)).toBeLessThan(0.1)
    const check = posteriorPredictiveCheck(y, rep['y'] ?? tensor([]), (d) => item(exp(mul(take(d, 0), 0))))
    expect(check.pValue).toBe(1)
  })

  it('discards implausible data in a posterior predictive check', () => {
    const counts = [0, 0, 0, 20, 0]
    const pois = (): void => {
      const r = sample('r', gamma(1, 1))
      plate('n', counts.length, () => observe('k', poisson(r), counts))
    }
    const p = conjugateInference(pois, undefined)
    const rep = posteriorPredictive(pois, p, undefined, { key: key(4), numSamples: 500 })
    const max = (d: ReturnType<typeof tensor>) => toFlat(d).reduce((a, b) => Math.max(a, b), 0)
    expect(posteriorPredictiveCheck(counts, rep['k'] ?? tensor([]), max).pValue).toBeLessThan(0.02)
  })
})
