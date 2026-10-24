import { describe, expect, it } from 'vitest'
import { logNdtr } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import * as P from '../index.ts'

describe('discrete expectation propagation', () => {
  const grid = P.isingModel({ rows: 3, cols: 3, coupling: 0.35, field: { s_0_0: 0.4, s_2_2: -0.3 } })

  it('has the fixed points of loopy belief propagation at power 1', () => {
    const ep = P.expectationPropagation(grid, { tol: 1e-12 })
    const bp = P.beliefPropagation(grid, { tol: 1e-13 })
    expect(ep.converged).toBe(true)
    for (const v of Object.keys(bp.marginals)) {
      expect(P.factorValues(ep.marginals[v] as P.Factor)[1]).toBeCloseTo(
        P.factorValues(bp.marginals[v] as P.Factor)[1] ?? 0,
        9,
      )
    }
    expect(ep.logPartition).toBeCloseTo(bp.logPartition, 8)
    const damped = P.expectationPropagation(grid, { damping: 0.5, tol: 1e-12, maxIter: 2000 })
    expect(P.factorValues(damped.marginals['s_1_1'] as P.Factor)[1]).toBeCloseTo(
      P.factorValues(bp.marginals['s_1_1'] as P.Factor)[1] ?? 0,
      9,
    )
  })

  it('runs power EP and stays exact on a tree for any power', () => {
    const half = P.expectationPropagation(grid, { power: 0.5, maxIter: 2000 })
    expect(half.converged).toBe(true)
    expect(Number.isNaN(half.logPartition)).toBe(true)
    const exact = P.junctionTreeMarginals(grid)
    for (const v of Object.keys(exact)) {
      expect(
        Math.abs(
          (P.factorValues(half.marginals[v] as P.Factor)[1] ?? 0) - (P.factorValues(exact[v] as P.Factor)[1] ?? 0),
        ),
      ).toBeLessThan(0.05)
    }
    expect(() => P.expectationPropagation(grid, { power: 0 })).toThrow(/power/)
  })
})

describe('Gaussian expectation propagation', () => {
  it('is exact for Gaussian sites (Bayesian linear regression)', () => {
    const prior = {
      mean: [0.5, -1],
      covariance: [
        [2, 0.3],
        [0.3, 1],
      ],
    }
    const data: [number[], number][] = [
      [[1, 0.2], 1.1],
      [[0.4, -1], 0.3],
      [[-0.7, 0.5], -0.9],
    ]
    const noise = 0.25
    const r = P.gaussianEp(
      prior,
      data.map(([w, y]) => ({ kind: 'gaussian' as const, weights: w, observation: y, variance: noise })),
    )
    // The same posterior by Gaussian factor algebra: a linear-Gaussian network y_i = w_i . x + noise.
    const g = P.linearGaussianBayesNet(
      [
        { name: 'x0', parents: [], weights: [], intercept: 0.5 - (0.3 / 1) * -1, variance: 2 - 0.09 },
        { name: 'x1', parents: [], weights: [], intercept: -1, variance: 1 },
        ...data.map(([w, _], i) => ({
          name: `y${i}`,
          parents: ['x0', 'x1'],
          weights: w,
          intercept: 0,
          variance: noise,
        })),
      ].map((n, i) => (i === 0 ? { ...n, parents: ['x1'], weights: [0.3] } : n)) as P.LinearGaussianNode[],
    )
    const exact = P.linearGaussianPosterior(g, ['x0', 'x1'], Object.fromEntries(data.map(([, y], i) => [`y${i}`, y])))
    for (const k of [0, 1]) expect(r.mean[k]).toBeCloseTo(exact.mean[k] ?? 0, 10)
    expect(r.covariance[0]?.[1]).toBeCloseTo(exact.covariance[0]?.[1] ?? 0, 10)
    expect(r.logMarginalLikelihood).toBeCloseTo(exact.logMass, 9)
  })

  it('matches one probit site exactly, by closed form and by quadrature', () => {
    const prior = { mean: [0.3], covariance: [[2]] }
    const closed = P.gaussianEp(prior, [{ kind: 'probit', weights: [1], label: 1 }])
    const quad = P.gaussianEp(prior, [{ kind: 'custom', weights: [1], logLikelihood: (z) => item(logNdtr(z)) }])
    expect(closed.logMarginalLikelihood).toBeCloseTo(item(logNdtr(0.3 / Math.sqrt(3))), 12)
    expect(quad.logMarginalLikelihood).toBeCloseTo(closed.logMarginalLikelihood, 8)
    expect(quad.mean[0]).toBeCloseTo(closed.mean[0] ?? 0, 8)
    expect(quad.covariance[0]?.[0]).toBeCloseTo(closed.covariance[0]?.[0] ?? 0, 8)
  })

  it('approximates a two-site probit posterior closely (against brute-force integration)', () => {
    const sites = [
      { kind: 'probit' as const, weights: [1, 0.5], label: 1 as const },
      { kind: 'probit' as const, weights: [-0.3, 1], label: -1 as const },
      { kind: 'probit' as const, weights: [0.8, 0.8], label: 1 as const },
    ]
    const r = P.gaussianEp(
      {
        mean: [0, 0],
        covariance: [
          [1, 0],
          [0, 1],
        ],
      },
      sites,
      { damping: 0.2 },
    )
    expect(r.converged).toBe(true)
    // Riemann sums of the unnormalised posterior on a grid.
    let z = 0
    let m0 = 0
    let m1 = 0
    const h = 0.02
    for (let a = -6; a <= 6; a += h) {
      for (let b = -6; b <= 6; b += h) {
        let lp = -(a * a + b * b) / 2 - Math.log(2 * Math.PI)
        for (const s of sites) lp += item(logNdtr(s.label * ((s.weights[0] ?? 0) * a + (s.weights[1] ?? 0) * b)))
        const w = Math.exp(lp) * h * h
        z += w
        m0 += w * a
        m1 += w * b
      }
    }
    expect(Math.abs((r.mean[0] ?? 0) - m0 / z)).toBeLessThan(0.02)
    expect(Math.abs((r.mean[1] ?? 0) - m1 / z)).toBeLessThan(0.02)
    expect(Math.abs(r.logMarginalLikelihood - Math.log(z))).toBeLessThan(0.01)
    const power = P.gaussianEp(
      {
        mean: [0, 0],
        covariance: [
          [1, 0],
          [0, 1],
        ],
      },
      sites,
      { power: 0.5 },
    )
    expect(power.converged).toBe(true)
    expect(Math.abs((power.mean[0] ?? 0) - m0 / z)).toBeLessThan(0.05)
    expect(() =>
      P.gaussianEp({ mean: [0], covariance: [[1]] }, [{ kind: 'probit', weights: [1, 2], label: 1 }]),
    ).toThrow(/weights/)
  })
})
