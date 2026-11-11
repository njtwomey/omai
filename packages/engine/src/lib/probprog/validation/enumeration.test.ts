// The enumeration oracle of the spike and slab regression against closed forms and a brute-force grid.
import { describe, expect, it } from 'vitest'
import { spikeSlabPosterior } from '../ep/ep-atoms-mixtures.ts'
import { mixtureOfRegressions, spikeSlabRegression, zeroInflatedPoisson } from './enumeration.ts'

describe('spikeSlabRegression', () => {
  it('one coefficient over several rows is the two-component closed form on the sufficient statistics', () => {
    const x = [0.5, -1, 2, 1.2]
    const y = [0.9, -1.4, 2.3, 1.1]
    const sigma = 0.6
    const sxx = x.reduce((a, v) => a + v * v, 0)
    const yEff = x.reduce((a, v, i) => a + v * (y[i] as number), 0) / sxx
    const closed = spikeSlabPosterior(0.3, 2, 0.1, yEff, sigma / Math.sqrt(sxx))
    const got = spikeSlabRegression({
      X: x.map((v) => [v]),
      y,
      inclusion: 0.3,
      slab: 2,
      spike: 0.1,
      noise: { sd: sigma },
    })
    expect(got.inclusion[0]).toBeCloseTo(closed.p1, 10)
    expect(got.mean[0]).toBeCloseTo(closed.selected.mean, 10)
    expect(got.sd[0]).toBeCloseTo(closed.selected.sd, 10)
    expect(got.subsets).toBe(2)
  })

  it('an unknown precision (a gamma prior on a grid) agrees with a two-dimensional brute-force sum', () => {
    const x = Array.from({ length: 12 }, (_, i) => Math.sin(i + 1) * 1.5)
    const y = x.map((v, i) => 0.9 * v + 0.4 * Math.cos(5 * i))
    const [shape, rate, pi, slab, spike] = [2, 0.5, 0.4, 1.5, 0.2]
    const got = spikeSlabRegression({
      X: x.map((v) => [v]),
      y,
      inclusion: pi,
      slab,
      spike,
      noise: { shape, rate },
      gridPoints: 121,
    })
    // The sum over (w, log tau) of prior x likelihood, on a fine grid.
    let z = 0
    let m = 0
    let t = 0
    let inc = 0
    const norm = (w: number, s: number): number => Math.exp(-0.5 * (w / s) ** 2) / (s * Math.sqrt(2 * Math.PI))
    for (let a = 0; a <= 400; a++) {
      const lt = -6 + (a * 12) / 400
      const tau = Math.exp(lt)
      const prior = Math.exp(shape * lt - rate * tau) // gamma density in log tau, up to a constant
      for (let b = 0; b <= 800; b++) {
        const w = -6 + (b * 12) / 800
        const lik = x.reduce((s, xi, i) => s + ((y[i] as number) - xi * w) ** 2, 0)
        const like = tau ** (x.length / 2) * Math.exp(-0.5 * tau * lik)
        const slabW = pi * norm(w, slab)
        const wt = prior * like * (slabW + (1 - pi) * norm(w, spike))
        z += wt
        m += wt * w
        t += wt * tau
        inc += prior * like * slabW
      }
    }
    expect(got.mean[0]).toBeCloseTo(m / z, 2)
    expect(got.inclusion[0]).toBeCloseTo(inc / z, 2)
    expect(got.precision).toBeCloseTo(t / z, 1)
  })

  it('finds the active coefficients of a small sparse problem', () => {
    const X = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), Math.sin(3 * i + 1), Math.cos(i / 2)])
    const y = X.map((r, i) => 2 * (r[0] as number) - 1.5 * (r[2] as number) + 0.05 * Math.sin(7 * i))
    const got = spikeSlabRegression({ X, y, inclusion: 0.3, slab: 3, spike: 0.05, noise: { shape: 2, rate: 0.1 } })
    expect(got.inclusion.map((v) => v > 0.5)).toEqual([true, false, true, false])
    expect(got.mean[0]).toBeCloseTo(2, 1)
  })
})

describe('zeroInflatedPoisson', () => {
  it('with every count positive, the inflation is Beta(a, b + n) whatever the rate and no count is a structural zero', () => {
    const post = zeroInflatedPoisson({
      x: [0, 1, -1, 0.5, 2],
      y: [1, 3, 1, 2, 9],
      inflation: { a: 1, b: 1 },
      coefficientSd: 2,
    })
    expect(post.pi.mean).toBeCloseTo(1 / 7, 3)
    expect(post.pi.sd).toBeCloseTo(Math.sqrt((1 * 6) / (49 * 8)), 3)
    expect(post.structural.every((v) => v === 0)).toBe(true)
  })

  it('a run of zeros under a rate that is large is a structural zero', () => {
    const post = zeroInflatedPoisson({
      x: [0, 0, 0, 0, 0, 0, 0, 0],
      y: [0, 0, 0, 12, 9, 11, 10, 8],
      inflation: { a: 1, b: 1 },
      coefficientSd: 2,
    })
    expect(post.structural[0]).toBeGreaterThan(0.95)
    expect(post.b0.mean).toBeGreaterThan(2)
  })
})

describe('mixtureOfRegressions', () => {
  const norm = (y: number, m: number, v: number): number =>
    Math.exp((-0.5 * (y - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)

  it('one row: each class has the weight of its marginal density at the response, and the class posterior is the conjugate one', () => {
    const [x, y, s0, s] = [1.5, 2, 1.2, 0.7]
    const post = mixtureOfRegressions({
      X: [[x]],
      y: [y],
      priorMean: [[0.5], [-2]],
      priorSd: s0,
      noiseSd: s,
      weights: [0.3, 0.7],
    })
    const w = [0.3 * norm(y, x * 0.5, x * x * s0 * s0 + s * s), 0.7 * norm(y, x * -2, x * x * s0 * s0 + s * s)]
    expect(post.assignments).toBe(2)
    expect(post.membership[0]?.[0]).toBeCloseTo((w[0] as number) / ((w[0] as number) + (w[1] as number)), 10)
    expect(post.logEvidence).toBeCloseTo(Math.log((w[0] as number) + (w[1] as number)), 10)
    // Given class 0 the coefficient is the conjugate posterior of a normal prior and one observation.
    const precision = 1 / (s0 * s0) + (x * x) / (s * s)
    const first = post.coefficients[0]?.[0]?.components.find((c) => c.weight > 0.01)
    expect(first?.mean).toBeCloseTo((0.5 / (s0 * s0) + (x * y) / (s * s)) / precision, 10)
    expect(first?.sd).toBeCloseTo(Math.sqrt(1 / precision), 10)
  })

  it('a single class is a Bayesian linear regression: the evidence of two rows is the Gaussian marginal', () => {
    const post = mixtureOfRegressions({ X: [[1], [1]], y: [1, 2], priorMean: [[0]], priorSd: 1, noiseSd: 1 })
    // y ~ N(0, [[2, 1], [1, 2]]): the density at (1, 2) is exp(-0.5 * 14/3 ... ) by the 2 x 2 inverse.
    const det = 3
    const quad = (2 * 1 - 2 * 1 * 2 + 2 * 4) / det
    expect(post.logEvidence).toBeCloseTo(-Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * quad, 10)
    expect(post.membership.every((r) => r[0] === 1 || Math.abs((r[0] as number) - 1) < 1e-12)).toBe(true)
  })
})
