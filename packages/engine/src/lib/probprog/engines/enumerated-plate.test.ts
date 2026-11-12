import { describe, expect, it } from 'vitest'
import { discreteUniform, gamma, poisson } from '../../dists/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { key } from '../../random/index.ts'
import { arange, item, less, tensor, toFlat, where } from '../../tensor/index.ts'
import { logJoint } from '../density.ts'
import { enumerate } from '../handlers.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops, toProgram } from '../model/index.ts'
import { type Model, observe, plate, sample } from '../primitives.ts'
import { mcmc } from './nuts.ts'

// A switch-point model: the rate is `a` before the switch year `s` and `b` from then on, over a plate of years.
const counts = [4, 5, 4, 6, 3, 4, 5, 4, 1, 0, 2, 1, 1, 0, 2, 1, 1, 2, 0, 1]
const T = counts.length
const [ALPHA, BETA] = [2, 0.5]
const total = (from: number, to: number): number => counts.slice(from, to).reduce((acc, c) => acc + c, 0)

/** log of the integral of a gamma(ALPHA, BETA) rate against Poisson counts summing to `sum` over `n` years. */
const logMarginal = (sum: number, n: number): number =>
  ALPHA * Math.log(BETA) - item(lgamma(ALPHA)) + item(lgamma(ALPHA + sum)) - (ALPHA + sum) * Math.log(BETA + n)

const logFact = counts.reduce((acc, c) => acc + item(lgamma(c + 1)), 0)

/** Brute-force posterior of the switch year, summed over the T + 1 values with the rates integrated out exactly. */
function exactSwitch(): number[] {
  const logp = Array.from({ length: T + 1 }, (_, s) => logMarginal(total(0, s), s) + logMarginal(total(s, T), T - s))
  const top = Math.max(...logp)
  const w = logp.map((v) => Math.exp(v - top))
  const z = w.reduce((acc, v) => acc + v, 0)
  return w.map((v) => v / z)
}

/** log p(y, a, b) with s summed out, by direct summation over s. */
function bruteLogJoint(a: number, b: number): number {
  const prior = (r: number): number =>
    ALPHA * Math.log(BETA) - item(lgamma(ALPHA)) + (ALPHA - 1) * Math.log(r) - BETA * r
  const terms = Array.from({ length: T + 1 }, (_, s) => {
    const lik = counts.reduce((acc, c, t) => {
      const r = t < s ? a : b
      return acc + c * Math.log(r) - r
    }, -logFact)
    return -Math.log(T + 1) + lik
  })
  const top = Math.max(...terms)
  return prior(a) + prior(b) + top + Math.log(terms.reduce((acc, v) => acc + Math.exp(v - top), 0))
}

const programModel = (): void => {
  const s = sample('s', discreteUniform(0, T))
  const a = sample('a', gamma(ALPHA, BETA))
  const b = sample('b', gamma(ALPHA, BETA))
  plate('t', T, (t) => observe('y', poisson(where(less(t, s), a, b)), tensor(counts)))
}

const specModel = (): Model => {
  const def = defineModel('switch', (m) => {
    const y = m.data('y', { dims: ['year'], support: 'count' })
    const year = m.data('year', { dims: ['year'] })
    const prior = m.const(new Array<number>(T + 1).fill(1 / (T + 1)))
    const s = m.latent('s', d.categorical({ probs: prior }))
    const a = m.latent('a', d.gamma(ALPHA, BETA))
    const b = m.latent('b', d.gamma(ALPHA, BETA))
    m.observe('y', d.poisson(ops.where(ops.lt(year, s), a, b)), y)
  })
  return toProgram(def.bind({ y: counts, year: toFlat(arange(T)) as number[] }))
}

describe.each([
  ['a program', programModel],
  ['a model written in the model language', specModel()],
])('a scalar discrete latent used inside a plated observation, %s', (_name, model) => {
  it('sums the switch year out of the log joint exactly', () => {
    for (const [a, b] of [
      [4.5, 1.1],
      [3, 2],
    ] as const) {
      const got = item(logJoint(enumerate(model), undefined, { a, b }).logJoint)
      expect(got).toBeCloseTo(bruteLogJoint(a, b), 9)
    }
  })

  it('has NUTS with the switch year enumerated agree with the brute-force posterior of the switch year', () => {
    const post = mcmc(model, undefined, { key: key(7), numChains: 2, numWarmup: 150, numDraws: 300 })
    expect(post.status).toBe('completed')
    expect(post.diagnostics['enumerated']).toEqual(['s'])
    const exact = exactSwitch()
    const draws = toFlat(post.draws?.['s'] ?? tensor([]))
    const freq = exact.map((_, s) => draws.filter((v) => v === s).length / draws.length)
    // The switch year is sharp (the rate drops at year 8): compare the mass within one year of the mode, and the mean.
    const mode = exact.indexOf(Math.max(...exact))
    const near = (p: readonly number[]): number => (p[mode - 1] ?? 0) + (p[mode] ?? 0) + (p[mode + 1] ?? 0)
    expect(Math.abs(near(freq) - near(exact))).toBeLessThan(0.08)
    const mean = (p: readonly number[]): number => p.reduce((acc, v, s) => acc + v * s, 0)
    expect(Math.abs(mean(freq) - mean(exact))).toBeLessThan(0.3)
  })
})

describe('a discrete latent over a plate, indexing a vector of locations', () => {
  it('sums each row out of the log joint exactly', () => {
    const ys = [-2, 2.2, 1.9, -1.8]
    const model = toProgram(
      defineModel('mixture', (m) => {
        const y = m.data('y', { dims: ['row'] })
        const mu = m.latent('mu', d.normal(0, 3), { dims: ['k'] })
        const z = m.latent('z', d.categorical({ probs: [0.4, 0.6] }), { dims: ['row'] })
        m.observe('y', d.normal(mu.at(z), 1), y)
      }).bind({ y: ys }, { sizes: { k: 2 } }),
    )
    const mu = [-1.5, 2]
    const logN = (x: number, m: number, sd: number): number =>
      -0.5 * ((x - m) / sd) ** 2 - Math.log(sd) - 0.5 * Math.log(2 * Math.PI)
    const prior = mu.reduce((acc, v) => acc + logN(v, 0, 3), 0)
    const rows = ys.reduce(
      (acc, y) => acc + Math.log(0.4 * Math.exp(logN(y, mu[0] ?? 0, 1)) + 0.6 * Math.exp(logN(y, mu[1] ?? 0, 1))),
      0,
    )
    expect(item(logJoint(enumerate(model), undefined, { mu: tensor(mu) }).logJoint)).toBeCloseTo(prior + rows, 9)
  })
})
