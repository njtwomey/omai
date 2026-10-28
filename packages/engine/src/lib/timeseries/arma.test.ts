import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { simulateArma } from '../../testing/series.ts'
import { kalmanLogLikelihood } from '../dynamics/filtering/index.ts'
import { item } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { constrainStationary, pacfToAr } from './_params.ts'
import * as T from './index.ts'

const reals = (k: number) =>
  fc.array(fc.double({ min: -2, max: 2, noNaN: true, noDefaultInfinity: true }), { minLength: k, maxLength: k })
const stationary = (k: number) => reals(k).map((u) => Array.from(constrainStationary(u)))
const invertible = (k: number) => reals(k).map((u) => Array.from(constrainStationary(u), (v) => -v))

describe('ARMA theory', () => {
  it('has the closed forms of an AR(1) and an MA(1)', () => {
    const acfAr = T.armaAcf([0.8], [], 5)
    acfAr.forEach((v, k) => {
      expect(v).toBeCloseTo(0.8 ** k, 12)
    })
    const acfMa = T.armaAcf([], [0.5], 3)
    expect(acfMa[1]).toBeCloseTo(0.5 / 1.25, 12)
    expect(acfMa[2]).toBeCloseTo(0, 12)
    expect(T.armaAutocovariance([], [0.5], 4, 1)[0]).toBeCloseTo(5, 12)
  })

  it('refuses a non-stationary autoregression', () => {
    expect(() => T.armaAutocovariance([1.2], [], 1, 3)).toThrow()
    expect(T.isStationary([0.5, 0.4])).toBe(true)
    expect(T.isStationary([0.5, 0.6])).toBe(false)
    expect(T.isInvertible([0.4, 0.3])).toBe(true)
    expect(T.isInvertible([-1.5])).toBe(false)
  })

  it('has psi and pi weights that multiply to one', () => {
    fc.assert(
      fc.property(stationary(2), invertible(2), (ar, ma) => {
        const psi = T.armaPsiWeights(ar, ma, 40)
        const pi = T.armaPiWeights(ar, ma, 40)
        // x = psi(B) e and e = x - pi(B) x, so psi(B)(1 - pi(B)) = 1.
        for (let k = 0; k < 20; k++) {
          let s = psi[k] ?? 0
          for (let j = 1; j <= k; j++) s -= (pi[j - 1] ?? 0) * (psi[k - j] ?? 0)
          expect(s).toBeCloseTo(k === 0 ? 1 : 0, 6)
        }
      }),
      { numRuns: 25 },
    )
  })

  it('has autocovariances that satisfy the Yule-Walker recursion at high lags', () => {
    fc.assert(
      fc.property(stationary(2), invertible(1), (ar, ma) => {
        const g = T.armaAutocovariance(ar, ma, 2, 12)
        for (let k = 3; k <= 12; k++) {
          const lhs = g[k] ?? 0
          const rhs = (ar[0] ?? 0) * (g[k - 1] ?? 0) + (ar[1] ?? 0) * (g[k - 2] ?? 0)
          expect(lhs).toBeCloseTo(rhs, 8)
        }
      }),
      { numRuns: 25 },
    )
  })

  it('has partial autocorrelations that cut off after p for an AR(p)', () => {
    const p = T.armaPacf([0.5, -0.3, 0.2], [], 6)
    expect(Math.abs(p[4] ?? 1)).toBeLessThan(1e-12)
    expect(Math.abs(p[5] ?? 1)).toBeLessThan(1e-12)
    expect(p[3]).toBeCloseTo(0.2, 12)
  })

  it('converts between partial autocorrelations and coefficients', () => {
    fc.assert(
      fc.property(reals(4), (u) => {
        const r = u.map((v) => v / Math.sqrt(1 + v * v))
        const back = T.arToPartialAutocorrelations(pacfToAr(r))
        back.forEach((v, i) => {
          expect(v).toBeCloseTo(r[i] ?? 0, 9)
        })
      }),
    )
  })
})

describe('the exact likelihood', () => {
  it('agrees across the fast filter, the library Kalman filter and the innovations algorithm', () => {
    fc.assert(
      fc.property(stationary(2), invertible(1), fc.double({ min: 0.3, max: 3, noNaN: true }), (ar, ma, sigma2) => {
        const y = simulateArma(3, 30, ar, ma, 0, 1)
        const fast = T.armaLogLikelihood(y, ar, ma, sigma2)
        const kalman = item(
          kalmanLogLikelihood(
            T.armaStateSpace(ar, ma, sigma2),
            y.map((v) => [v]),
          ),
        )
        expect(fast).toBeCloseTo(kalman, 8)
        // The innovations form: sum_t log N(x_t; xhat_t, v_(t-1)).
        const gamma = T.armaAutocovariance(ar, ma, sigma2, y.length)
        const { theta, v } = T.innovations(gamma)
        let ll = 0
        const e: number[] = []
        for (let t = 0; t < y.length; t++) {
          let xhat = 0
          for (let j = 1; j <= t; j++) xhat += (theta[t]?.[j - 1] ?? 0) * (e[t - j] ?? 0)
          e.push((y[t] ?? 0) - xhat)
          ll += -0.5 * (Math.log(2 * Math.PI * (v[t] ?? 1)) + (e[t] ?? 0) ** 2 / (v[t] ?? 1))
        }
        expect(fast).toBeCloseTo(ll, 6)
      }),
      { numRuns: 12 },
    )
  })

  it('skips missing observations', () => {
    const y = simulateArma(4, 40, [0.5], [0.3])
    const withGap = y.map((v, t) => (t === 10 ? Number.NaN : v))
    const full = T.armaLogLikelihood(y, [0.5], [0.3], 1)
    const gap = T.armaLogLikelihood(withGap, [0.5], [0.3], 1)
    expect(Number.isFinite(gap)).toBe(true)
    expect(gap).not.toBeCloseTo(full, 3)
    expect(T.armaFilter(withGap, [0.5], [0.3]).nobs).toBe(39)
  })
})

describe('the innovations algorithm', () => {
  it('runs one row per step and tends to the moving-average coefficients', () => {
    const gamma = T.armaAutocovariance([], [0.6], 1, 30)
    const run = runAlgorithm(T.innovationsAlgorithm(gamma), { maxSteps: 30 })
    expect(run.state.theta).toHaveLength(31)
    expect(run.state.theta[30]?.[0]).toBeCloseTo(0.6, 4)
    expect(run.state.v[30]).toBeCloseTo(1, 4)
  })

  it('serialises its state', async () => {
    const { expectResumable } = await import('../../testing/series.ts')
    expectResumable(T.innovationsAlgorithm(T.armaAutocovariance([0.5], [0.3], 1, 12)), 5, 4)
  })
})
