import { describe, expect, it } from 'vitest'
import { expectPosteriorMoments, MODELS } from '../../../testing/probprog.ts'
import { bernoulli, beta, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { add, asTensor, item, mul, tensor, toFlat } from '../../tensor/index.ts'
import { analyse } from '../analysis.ts'
import { factor, type Model, observe, plate, sample } from '../primitives.ts'
import { conjugateInference } from './conjugate.ts'
import { inferDiscrete, mcmc, nutsEngine } from './nuts.ts'

const logN = (x: number, m: number, s: number): number =>
  -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI)

describe('NUTS on programs', () => {
  it('recovers a conjugate normal posterior within Monte Carlo error', () => {
    const model = MODELS['normalMean'] as Model
    const exact = conjugateInference(model, undefined)
    const post = mcmc(model, undefined, { key: key(1), numChains: 2, numWarmup: 300, numDraws: 400 })
    expect(post.status).toBe('completed')
    expect(post.diagnostics['divergences']).toBe(0)
    expectPosteriorMoments(post, 'mu', {
      mean: exact.mean('mu').data[0] ?? 0,
      sd: Math.sqrt(exact.variance('mu').data[0] ?? 0),
    })
    expect(post.chains?.numChains).toBe(2)
    expect(post.summary()[0]?.rhat).toBeLessThan(1.05)
  })

  it('samples a constrained site on its unconstrained scale', () => {
    const flips = [1, 1, 0, 1, 1, 0, 1]
    const coin = (): void => {
      const p = sample('p', beta(2, 2))
      plate('n', flips.length, () => observe('y', bernoulli({ probs: p }), flips))
    }
    const post = mcmc(coin, undefined, {
      key: key(2),
      numChains: 1,
      numWarmup: 300,
      numDraws: 600,
      algorithm: 'hmc',
      numSteps: 8,
    })
    // Beta(2 + 5, 2 + 2).
    const a = 7
    const b = 4
    expectPosteriorMoments(post, 'p', { mean: a / (a + b), sd: Math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1))) })
    expect(post.engine).toBe('hmc')
  })

  it('sums an enumerated discrete site out and recovers it by inferDiscrete', () => {
    const y = 1.4
    const model = (): void => {
      const mu = sample('mu', normal(0, 1))
      const c = sample('c', bernoulli({ probs: 0.5 }))
      observe('y', normal(add(mu, mul(c, 2)), 1), y)
    }
    // p(c | y) from y ~ N(2c, sqrt 2); p(mu | c, y) = N((y - 2c) / 2, 1 / sqrt 2).
    const w1 = Math.exp(logN(y, 2, Math.SQRT2))
    const w0 = Math.exp(logN(y, 0, Math.SQRT2))
    const p1 = w1 / (w0 + w1)
    const mean = (1 - p1) * (y / 2) + p1 * ((y - 2) / 2)
    const post = mcmc(model, undefined, { key: key(3), numChains: 1, numWarmup: 300, numDraws: 800 })
    expectPosteriorMoments(post, 'mu', { mean })
    const c = toFlat(post.draws?.['c'] ?? tensor([]))
    const freq = c.reduce((s, v) => s + v, 0) / c.length
    // Draws are autocorrelated through mu; 6 binomial standard errors is a wide but principled band.
    expect(Math.abs(freq - p1)).toBeLessThan(6 * Math.sqrt((p1 * (1 - p1)) / c.length))
    expect(post.diagnostics['enumerated']).toEqual(['c'])
  })

  it('draws discrete sites from their exact conditionals', () => {
    const model = (): void => {
      const c = sample('c', bernoulli({ probs: 0.5 }))
      observe('y', normal(mul(c, 2), 1), 2)
    }
    const p1 = Math.exp(logN(2, 2, 1)) / (Math.exp(logN(2, 2, 1)) + Math.exp(logN(2, 0, 1)))
    let ones = 0
    for (let i = 0; i < 400; i++) ones += item(asTensor(inferDiscrete(model, undefined, {}, key(i))['c'] ?? 0))
    expect(Math.abs(ones / 400 - p1)).toBeLessThan(4 * Math.sqrt((p1 * (1 - p1)) / 400))
  })

  it('ends with status failed when no starting point has a finite density', () => {
    const broken = (): void => {
      sample('x', normal(0, 1))
      factor('impossible', Number.NEGATIVE_INFINITY)
    }
    const post = mcmc(broken, undefined, { key: key(0), numChains: 1, numWarmup: 10, numDraws: 10 })
    expect(post.status).toBe('failed')
    expect(String(post.diagnostics['reason'])).toMatch(/starting point/)
  })

  it('applies to differentiable programs only', () => {
    const engine = nutsEngine()
    expect(engine.applies(analyse(MODELS['schools'] as Model, undefined)).applies).toBe(true)
    const discrete = (): void => {
      sample('k', bernoulli({ probs: 0.3 }))
    }
    expect(engine.applies(analyse(discrete, undefined)).applies).toBe(false)
  })
})
