import { describe, expect, it } from 'vitest'
import { logSumExpOf } from '../../base/index.ts'
import {
  bernoulli,
  beta,
  categorical,
  gamma,
  inverseGamma,
  multivariateNormal,
  normal,
  poisson,
} from '../../dists/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { key } from '../../random/index.ts'
import { item, mul, sqrt, take, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { analyse } from '../analysis.ts'
import { markov, observe, plate, sample } from '../primitives.ts'
import { conjugateEngine, conjugateInference } from './conjugate.ts'
import { enumerationEngine, enumerationPosterior } from './enumeration.ts'
import { gaussianEngine, gaussianInference } from './gaussian.ts'

const logN = (x: number, m: number, s: number): number =>
  -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI)

describe('enumeration engine', () => {
  const W = [0.4, 0.6]
  const MU = [-1, 1.5]
  const Y = [-0.7, 1.2, 0.1]
  const mixture = (y: number[]): void => {
    plate('data', y.length, () => {
      const z = sample('z', categorical({ probs: W }))
      observe('y', normal(take(tensor(MU), z), 1), y)
    })
  }

  it('gives exact marginals of plated local sites and the evidence', () => {
    const post = enumerationPosterior(mixture, Y)
    const probs = toFlat(post.mean('z'))
    let logZ = 0
    Y.forEach((y, i) => {
      const lw = [0, 1].map((k) => Math.log(W[k] ?? 0) + logN(y, MU[k] ?? 0, 1))
      const z = logSumExpOf(lw)
      logZ += z
      expect(probs[i]).toBeCloseTo(Math.exp((lw[1] ?? 0) - z), 12)
    })
    expect(post.logEvidence?.value).toBeCloseTo(logZ, 10)
    expect(post.marginal('z').batchShape).toEqual([3])
  })

  it('samples the joint posterior of a chain exactly', () => {
    const trans = [
      [0.8, 0.2],
      [0.3, 0.7],
    ]
    const ys = [1, 1, 0, 1]
    const chain = (): void => {
      markov(
        't',
        ys.length,
        (t, prev: Value | undefined) => {
          const z = sample(`z${t}`, categorical({ probs: prev === undefined ? [0.5, 0.5] : take(tensor(trans), prev) }))
          observe(`y${t}`, bernoulli({ probs: take(tensor([0.1, 0.9]), z) }), ys[t] ?? 0)
          return z
        },
        undefined as Value | undefined,
      )
    }
    const post = enumerationPosterior(chain, undefined)
    // Brute-force joint over 16 paths.
    const joint = new Map<string, number>()
    for (let m = 0; m < 16; m++) {
      let lp = 0
      let prev = -1
      for (let t = 0; t < 4; t++) {
        const z = (m >> t) & 1
        lp += Math.log(prev < 0 ? 0.5 : (trans[prev]?.[z] ?? 0))
        lp += Math.log(z === 1 ? (ys[t] === 1 ? 0.9 : 0.1) : ys[t] === 1 ? 0.1 : 0.9)
        prev = z
      }
      joint.set(String(m), lp)
    }
    const logZ = logSumExpOf([...joint.values()])
    expect(post.logEvidence?.value).toBeCloseTo(logZ, 10)
    // P(z0 = z1 = z2 = z3 = 1) against the frequency of joint draws.
    const draws = post.sample(key(3), 4000)
    const all = Array.from({ length: 4000 }, (_, i) =>
      [0, 1, 2, 3].every((t) => toFlat(draws[`z${t}`] ?? tensor([]))[i] === 1),
    )
    const p = Math.exp((joint.get('15') ?? 0) - logZ)
    const freq = all.filter((x) => x).length / 4000
    expect(Math.abs(freq - p)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / 4000))
    expect(post.logProb?.({ z0: 1, z1: 1, z2: 1, z3: 1 })).toBeCloseTo(Math.log(p), 10)
  })

  it('applies only to finite discrete programs', () => {
    expect(enumerationEngine().applies(analyse(mixture, Y)).applies).toBe(true)
    const continuous = (): void => {
      sample('x', normal(0, 1))
    }
    expect(enumerationEngine().applies(analyse(continuous, undefined)).applies).toBe(false)
  })
})

describe('conjugate engine', () => {
  it('matches the beta-binomial update and its evidence', () => {
    const flips = [1, 0, 1, 1, 1]
    const coin = (y: number[]): void => {
      const p = sample('p', beta(2, 3))
      plate('n', y.length, () => observe('y', bernoulli({ probs: p }), y))
    }
    const a = analyse(coin, flips)
    expect(conjugateEngine().applies(a).applies).toBe(true)
    const post = conjugateInference(coin, flips, a)
    expect(item(post.mean('p'))).toBeCloseTo(6 / 10, 14)
    // Evidence: B(2 + 4, 3 + 1) / B(2, 3).
    const lbeta = (x: number, y: number): number => lg(x) + lg(y) - lg(x + y)
    expect(post.logEvidence?.value).toBeCloseTo(lbeta(6, 4) - lbeta(2, 3), 10)
  })

  it('handles gamma-Poisson and inverse-gamma variance structures together', () => {
    const model = (): void => {
      const rate = sample('rate', gamma(2, 1))
      plate('counts', 3, () => observe('k', poisson(rate), [3, 5, 4]))
      const v = sample('v', inverseGamma(3, 2))
      plate('points', 2, () => observe('x', normal(1, sqrt(v)), [0.5, 2]))
    }
    const post = conjugateInference(model, undefined)
    expect(item(post.mean('rate'))).toBeCloseTo((2 + 12) / (1 + 3), 12)
    // Inverse gamma(3 + 1, 2 + (0.25 + 1) / 2): mean b / (a - 1).
    expect(item(post.mean('v'))).toBeCloseTo((2 + 0.625) / 3, 12)
  })
})

const lg = (x: number): number => item(lgamma(x))

describe('gaussian engine', () => {
  const ys = [0.3, 1.4, 0.8, 2.1, 1.7]
  const walk = (): void => {
    markov(
      't',
      ys.length,
      (t, x: Value | number) => {
        const z = sample(`x${t}`, normal(mul(x, 0.9), t === 0 ? 2 : 0.5))
        observe(`y${t}`, normal(z, 0.7), ys[t] ?? 0)
        return z
      },
      0 as Value | number,
    )
  }

  it('agrees between Kalman smoothing, dense conditioning and belief propagation', () => {
    const a = analyse(walk, undefined)
    expect(a.linearGaussian.linear).toBe(true)
    expect(gaussianEngine().applies(a).applies).toBe(true)
    const kalman = gaussianInference(walk, undefined, { method: 'kalman', analysis: a })
    const joint = gaussianInference(walk, undefined, { method: 'joint', analysis: a })
    const bp = gaussianInference(walk, undefined, { method: 'bp', analysis: a })
    for (let t = 0; t < ys.length; t++) {
      expect(item(kalman.mean(`x${t}`))).toBeCloseTo(item(joint.mean(`x${t}`)), 10)
      expect(item(kalman.variance(`x${t}`))).toBeCloseTo(item(joint.variance(`x${t}`)), 10)
      expect(item(bp.mean(`x${t}`))).toBeCloseTo(item(joint.mean(`x${t}`)), 8)
      expect(item(bp.variance(`x${t}`))).toBeCloseTo(item(joint.variance(`x${t}`)), 8)
    }
    expect(kalman.logEvidence?.value).toBeCloseTo(joint.logEvidence?.value ?? 0, 10)
    // Backward sampling reproduces the smoothed moments.
    const d = kalman.sample(key(5), 4000)
    const x2 = toFlat(d['x2'] ?? tensor([]))
    const m = x2.reduce((s, v) => s + v, 0) / x2.length
    const sd = Math.sqrt(item(joint.variance('x2')))
    expect(Math.abs(m - item(joint.mean('x2')))).toBeLessThan((4 * sd) / Math.sqrt(4000))
  })

  it('conditions multivariate sites', () => {
    const model = (): void => {
      const x = sample(
        'x',
        multivariateNormal([0, 0], {
          covariance: [
            [1, 0.5],
            [0.5, 2],
          ],
        }),
      )
      observe('y', normal(take(x, 0), 0.5), 1)
    }
    const post = gaussianInference(model, undefined)
    // Conditioning N(0, Sigma) on x0 + noise = 1: gain Sigma[:, 0] / (Sigma00 + 0.25).
    const g = [1 / 1.25, 0.5 / 1.25]
    expect(toFlat(post.mean('x'))[0]).toBeCloseTo(g[0] ?? 0, 12)
    expect(toFlat(post.mean('x'))[1]).toBeCloseTo(g[1] ?? 0, 12)
  })
})
