import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { jacfwd } from '../autodiff/index.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import {
  add,
  item,
  maximum,
  mean,
  mul,
  sliceAxis,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { feynmanKac } from './calculus.ts'
import {
  besselProcess,
  cir,
  geometricBrownianMotion,
  langevinDiffusion,
  mertonJumpDiffusion,
  mertonPrice,
  ornsteinUhlenbeck,
  simulateDiffusion,
  simulateJumpDiffusion,
  underdampedLangevin,
  vasicek,
  vasicekBondPrice,
} from './diffusions.ts'
import { timeGrid, valuesAt } from './paths.ts'
import { sdeint } from './sde.ts'

function expectMoments(xs: readonly number[], m: number, v: number): void {
  const n = xs.length
  expect(Math.abs(meanOf(xs) - m)).toBeLessThan(4 * Math.sqrt(v / n) + 1e-12)
  // the sample variance has standard error about v sqrt(2 / n) for light tails; allow a little kurtosis
  expect(Math.abs(varianceOf(xs) - v)).toBeLessThan(5 * v * Math.sqrt(2 / n))
}

describe('exact simulation of named diffusions', () => {
  it('Ornstein-Uhlenbeck matches its transition moments on a coarse grid', () => {
    const ou = ornsteinUhlenbeck({ theta: 1.5, mu: 1, sigma: 0.7 })
    const p = simulateDiffusion(key(1), ou, -1, timeGrid(2, 2), { paths: 4000 })
    const end = toFlat(valuesAt(p, -1))
    expectMoments(end, item(ou.mean(-1, 2)), item(ou.variance(-1, 2)))
    expect(item(ou.stationary().variance())).toBeCloseTo(0.49 / 3, 14)
  })

  it('agrees with an SDE scheme on the same process', () => {
    const ou = ornsteinUhlenbeck({ theta: 1.5, mu: 1, sigma: 0.7 })
    const p = simulateDiffusion(key(2), ou, -1, timeGrid(2, 200), { paths: 3000, exact: false })
    expectMoments(toFlat(valuesAt(p, -1)), item(ou.mean(-1, 2)), item(ou.variance(-1, 2)))
  })

  it('geometric Brownian motion stays positive with the log-normal moments', () => {
    const g = geometricBrownianMotion({ mu: 0.1, sigma: 0.3 })
    const p = simulateDiffusion(key(3), g, 2, timeGrid(1, 3), { paths: 4000 })
    const end = toFlat(valuesAt(p, -1))
    expect(end.every((x) => x > 0)).toBe(true)
    expectMoments(end, item(g.mean(2, 1)), item(g.variance(2, 1)))
  })

  it('CIR stays positive with its closed-form moments and settles into the gamma law', () => {
    const c = cir({ kappa: 2, theta: 0.5, sigma: 0.6 })
    const p = simulateDiffusion(key(4), c, 0.1, timeGrid(1, 2), { paths: 3000 })
    const end = toFlat(valuesAt(p, -1))
    expect(end.every((x) => x > 0)).toBe(true)
    expectMoments(end, item(c.mean(0.1, 1)), item(c.variance(0.1, 1)))
    const far = toFlat(valuesAt(simulateDiffusion(key(5), c, 0.1, [0, 20], { paths: 3000 }), -1))
    expectMoments(far, item(c.stationary().mean()), item(c.stationary().variance()))
  })

  it('a Bessel process of dimension 3 has E[R_t^2] = r0^2 + 3 t, like the norm of 3-d Brownian motion', () => {
    const b = besselProcess({ delta: 3 })
    const p = simulateDiffusion(key(6), b, 0.5, timeGrid(1, 2), { paths: 2000 })
    const sq = toFlat(valuesAt(p, -1)).map((r) => r * r)
    expect(Math.abs(meanOf(sq) - item(b.meanSquare(0.5, 1)))).toBeLessThan(4 * Math.sqrt(varianceOf(sq) / 2000))
  })

  it('Vasicek is an Ornstein-Uhlenbeck short rate whose bond price is the Feynman-Kac expectation', () => {
    const params = { a: 1.2, b: 0.04, sigma: 0.05 }
    const v = vasicek(params)
    expect(v.name).toBe('vasicek')
    const fk = feynmanKac(
      key(7),
      { sde: v, payoff: (y) => add(zeros([y.shape[0] ?? 1]), 1), potential: (_t, y) => sum(y, 1) },
      [0.03],
      2,
      { steps: 50, paths: 2000 },
    )
    const exact = item(vasicekBondPrice(0.03, 2, params))
    // left-point sums bias the integral by O(dt); the Monte Carlo error is tiny for bonds
    expect(Math.abs(fk.estimate.value - exact)).toBeLessThan(4 * (fk.estimate.standardError ?? 0) + 2e-3)
  })
})

function _req<T>(f: T | undefined): T {
  if (f === undefined) throw new Error('missing')
  return f
}

describe('pathwise derivatives of exact simulations', () => {
  it('differentiates OU and GBM paths in their parameters', () => {
    const t = timeGrid(1, 3)
    const ou = (theta: Value, sigma: Value): Value =>
      valuesAt(simulateDiffusion(key(8), ornsteinUhlenbeck({ theta, mu: 0.5, sigma }), 1, t, { paths: 3 }), -1)
    expectConformant(ou, [tensor(1.2), tensor(0.4)])
    const gbm = (mu: Value, sigma: Value): Value =>
      valuesAt(simulateDiffusion(key(9), geometricBrownianMotion({ mu, sigma }), 1, t, { paths: 3 }), -1)
    expectConformant(gbm, [tensor(0.05), tensor(0.3)])
  })

  it('differentiates CIR transitions by implicit reparameterisation, unbiased for d E[X_t] / d kappa', () => {
    // The sampler is a Poisson mixture, so finite differences of one draw do not apply; the implicit gradient is the
    // derivative at a fixed CDF level, whose average is the derivative of the mean, -t (x0 - theta) e^{-kappa t}.
    const n = 1000
    const per = (kappa: Value): Value =>
      valuesAt(simulateDiffusion(key(10), cir({ kappa, theta: 0.5, sigma: 0.4 }), 0.3, [0, 0.5], { paths: n }), -1)
    const each = toFlat(jacfwd(per)(tensor(1.5)))
    const g = meanOf(each)
    const want = 0.5 * 0.2 * Math.exp(-0.75)
    expect(Math.abs(g - want)).toBeLessThan(4 * Math.sqrt(varianceOf(each) / n))
  })
})

describe('Langevin dynamics', () => {
  it('overdamped dynamics in a quadratic well have stationary variance 1 / (beta k)', () => {
    const l = langevinDiffusion({ potential: (x) => mul(sum(square(x), 1), 1), beta: 2 })
    expect(toFlat(l.logStationary([[1, 1]]))).toEqual([-4])
    const p = sdeint(key(11), l, zeros([2000, 1]), timeGrid(5, 250))
    const end = toFlat(valuesAt(p, -1))
    expectMoments(end, 0, 0.25)
  })

  it('underdamped dynamics thermalise the velocity to variance 1 / beta', () => {
    const l = underdampedLangevin({ potential: (x) => mul(sum(square(x), 1), 0.5), friction: 2, beta: 1 })
    const p = sdeint(key(12), l, zeros([2000, 2]), timeGrid(6, 300))
    const end = valuesAt(p, -1)
    expectMoments(toFlat(sliceAxis(end, 1, 1, 2)), 0, 1)
    expectMoments(toFlat(sliceAxis(end, 1, 0, 1)), 0, 1)
    expect(() => l.drift(0, zeros([1, 3]))).toThrow()
  })
})

describe('jump diffusions', () => {
  const params = { mu: 0.05, sigma: 0.2, rate: 1.5, jumpMean: -0.1, jumpStd: 0.15 }

  it("Merton's exact simulation has E[S_T] = S_0 e^{mu T} and prices calls like the series", () => {
    const jd = mertonJumpDiffusion(params)
    const p = simulateJumpDiffusion(key(13), jd, 100, timeGrid(1, 2), { paths: 8000 })
    const end = toFlat(valuesAt(p, -1))
    const sd = Math.sqrt(varianceOf(end) / end.length)
    expect(Math.abs(meanOf(end) - 100 * Math.exp(0.05))).toBeLessThan(4 * sd)
    const payoff = end.map((s) => Math.max(s - 100, 0) * Math.exp(-0.05))
    const exact = item(mertonPrice(100, 100, 1, 0.05, params))
    expect(Math.abs(meanOf(payoff) - exact)).toBeLessThan(4 * Math.sqrt(varianceOf(payoff) / payoff.length))
  })

  it('a pure jump process has the compound Poisson moments', () => {
    const still = {
      drift: (_t: number, y: Value) => zeros(y.shape),
      diffusion: (_t: number, y: Value) => zeros(y.shape),
    }
    const p = simulateJumpDiffusion(
      key(14),
      { name: 'cp', sde: still, rate: 3, jumpSize: normal(0.5, 0.2) },
      0,
      timeGrid(2, 4),
      {
        paths: 4000,
      },
    )
    const end = toFlat(valuesAt(p, -1))
    // mean lambda t E[J], variance lambda t E[J^2]
    expectMoments(end, 3, 6 * 0.29)
    expect(() =>
      simulateJumpDiffusion(key(0), { name: 'bad', sde: still, rate: -1, jumpSize: normal(0, 1) }, 0, [0, 1]),
    ).toThrow()
  })

  it('the Euler scheme with multiplicative jumps approximates Merton', () => {
    const jd = mertonJumpDiffusion(params)
    const { exactStep: _ignored, ...scheme } = jd
    const p = simulateJumpDiffusion(key(15), scheme, 100, timeGrid(1, 100), { paths: 3000 })
    const end = toFlat(valuesAt(p, -1))
    expect(Math.abs(meanOf(end) - 100 * Math.exp(0.05))).toBeLessThan(4 * Math.sqrt(varianceOf(end) / 3000) + 0.5)
  })

  it('discounted payoffs differentiate in the spot (pathwise delta)', () => {
    const jd = mertonJumpDiffusion(params)
    const f = (s0: Value): Value =>
      mean(maximum(sub(valuesAt(simulateJumpDiffusion(key(16), jd, s0, [0, 1], { paths: 50 }), -1), 100), 0))
    expectConformant(f, [tensor(105)], { vmap: false })
  })
})
