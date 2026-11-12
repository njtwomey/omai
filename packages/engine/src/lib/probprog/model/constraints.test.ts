// Constraints on Gaussian latents (ordered, positive, bounds) compiled to comparison factors and run under EP, against the
// exact posterior on a grid and the benchmark truth of the ordered Gaussian mixture.
import { describe, expect, it } from 'vitest'
import { runEp } from '../../../testing/factor-graph.ts'
import { type MpState, marginalsOf } from '../../message-passing/index.ts'
import { gaussianMixtureOrdered } from '../models/index.ts'
import * as d from './dists/index.ts'
import { defineModel, type toFactorGraph } from './index.ts'

function run(bound: Parameters<typeof toFactorGraph>[0], tol = 1e-9) {
  const { g, r } = runEp(bound, { tol: tol, damping: 0.3, maxSteps: 600, verdict: true })
  return marginalsOf(g, r.state as MpState)
}

describe('the ordered constraint', () => {
  it('two ordered means with a Gaussian likelihood: EP is the truncated posterior, against a two-dimensional grid', () => {
    const m = defineModel('ordered-pair', (b) => {
      const k = b.range('k', { size: 2 })
      const mu = b.latent('mu', d.normal(0, 2), { dims: [k], constraint: { kind: 'ordered' } })
      b.observe('y', d.normal(mu, 0.8), b.data('y', { dims: [k] }))
    })
    const ys = [1.4, 0.3]
    const marginals = run(m.bind({ y: ys }))
    // The exact posterior of (mu0, mu1): the two Gaussian likelihoods and priors on the region mu0 < mu1 (the observations have
    // opposite order, so the constraint binds).
    const n = 700
    const lo = -6
    const h = 12 / n
    let z = 0
    const e = [0, 0]
    const e2 = [0, 0]
    for (let i = 0; i <= n; i++)
      for (let j = i; j <= n; j++) {
        const a = lo + i * h
        const c = lo + j * h
        const w = Math.exp(
          -0.5 *
            ((a / 2) ** 2 + (c / 2) ** 2 + ((ys[0] as number) - a) ** 2 / 0.64 + ((ys[1] as number) - c) ** 2 / 0.64),
        )
        z += w
        e[0] = (e[0] as number) + w * a
        e[1] = (e[1] as number) + w * c
        e2[0] = (e2[0] as number) + w * a * a
        e2[1] = (e2[1] as number) + w * c * c
      }
    for (const k of [0, 1]) {
      const mean = (e[k] as number) / z
      const sd = Math.sqrt((e2[k] as number) / z - mean * mean)
      const got = marginals[`mu[${k}]`] as Record<string, number>
      expect(got['mean']).toBeCloseTo(mean, 1)
      expect(Math.sqrt(got['variance'] as number)).toBeCloseTo(sd, 1)
    }
  })

  it('the ordered Gaussian mixture benchmark runs under EP and recovers the means of its quadrature truth', () => {
    const bound = gaussianMixtureOrdered.bound()
    const truth = gaussianMixtureOrdered.truth()
    const marginals = run(bound, 1e-7)
    for (const k of [0, 1]) {
      const oracle = truth.marginals[`mu[${k}]`]
      const got = marginals[`mu[${k}]`] as Record<string, number>
      expect(got['mean']).toBeCloseTo(oracle?.mean as number, 1)
      expect(Math.sqrt(got['variance'] as number)).toBeCloseTo(oracle?.sd as number, 1)
    }
  })
})

describe('bounds on a Gaussian latent', () => {
  it('a positive Gaussian (a half normal) is the truncated Gaussian: moment matching against the exact posterior', () => {
    const m = defineModel('positive', (b) => {
      const s = b.latent('s', d.normal(0, 1), { constraint: { kind: 'positive' } })
      b.observe('y', d.normal(s, 1), b.data('y', { dims: ['row'] }))
    })
    const ys = [-0.4, 0.2, -0.9]
    const marginals = run(m.bind({ y: ys }))
    let z = 0
    let e = 0
    let e2 = 0
    for (let i = 1; i <= 40000; i++) {
      const s = i * 0.0002
      const w = Math.exp(-0.5 * s * s - 0.5 * ys.reduce((a, y) => a + (y - s) ** 2, 0))
      z += w
      e += w * s
      e2 += w * s * s
    }
    const mean = e / z
    const got = marginals['s'] as Record<string, number>
    expect(got['mean']).toBeCloseTo(mean, 2)
    expect(Math.sqrt(got['variance'] as number)).toBeCloseTo(Math.sqrt(e2 / z - mean * mean), 2)
  })
})
