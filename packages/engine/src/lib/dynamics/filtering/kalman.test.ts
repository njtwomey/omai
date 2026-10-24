import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, reshape, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { kalmanFilter, kalmanLogLikelihood, nees, nisTest, rtsSmoother, steadyStateKalman } from './kalman.ts'

describe('the Kalman filter', () => {
  it('equals batch weighted least squares on a static linear-Gaussian model (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 2, maxLength: 8 }),
        fc.double({ min: 0.1, max: 4, noNaN: true }),
        (zs, r) => {
          // x is a fixed 2-vector observed through h_t = [1, t]; prior N(0, 10 I).
          const model = {
            F: [
              [1, 0],
              [0, 1],
            ],
            H: [[1, 0]],
            Q: [
              [0, 0],
              [0, 0],
            ],
            R: [[r]],
            x0: [0, 0],
            P0: [
              [10, 0],
              [0, 10],
            ],
          }
          // A time-varying H is the same as stacking the observations: use the information form directly.
          let a = 0.1
          let b = 0
          let c = 0.1
          let u = 0
          let v = 0
          zs.forEach((z, t) => {
            a += 1 / r
            b += t / r
            c += (t * t) / r
            u += z / r
            v += (t * z) / r
          })
          const det = a * c - b * b
          const ls = [(c * u - b * v) / det, (a * v - b * u) / det]
          // Run the filter with the time-varying observation row by augmenting time into F: x = [x1, x2], z = x1 + t x2.
          let x = [0, 0]
          let P = [10, 0, 0, 10]
          zs.forEach((z, t) => {
            const f = kalmanFilter({ ...model, H: [[1, t]], x0: x, P0: [P.slice(0, 2), P.slice(2)] }, [[z]])
            x = toFlat(f.means)
            P = toFlat(f.covariances)
          })
          return Math.abs((x[0] ?? 0) - (ls[0] ?? 0)) < 1e-8 && Math.abs((x[1] ?? 0) - (ls[1] ?? 0)) < 1e-8
        },
      ),
      { numRuns: 30 },
    )
  })

  const cv = {
    F: [
      [1, 1],
      [0, 1],
    ],
    H: [[1, 0]],
    Q: [
      [0.01, 0],
      [0, 0.01],
    ],
    R: [[1]],
    x0: [0, 0],
    P0: [
      [1, 0],
      [0, 1],
    ],
  }

  it('skips missing observations and smooths through gaps', () => {
    const z = [[1], [Number.NaN], [3], [Number.NaN], [5]]
    const f = kalmanFilter(cv, z)
    expect(f.observed).toEqual([1, 0, 1, 0, 1])
    expect(Number.isNaN(f.nis[1] ?? 0)).toBe(true)
    const s = rtsSmoother(cv, f)
    const m = toFlat(s.means)
    // The smoothed position at the gap lies between its neighbours.
    expect((m[2] ?? 0) > (m[0] ?? 0) && (m[2] ?? 0) < (m[4] ?? 0)).toBe(true)
  })

  it('the log-likelihood is differentiable in the noise variance', () => {
    const z = [[0.5], [1.7], [2.2], [3.9]]
    const ll = (r: Value): Value => kalmanLogLikelihood({ ...cv, R: reshape(r, [1, 1]) }, z)
    const g = item(grad(ll)(tensor(1.3)))
    const h = 1e-6
    const fd = (item(ll(tensor(1.3 + h))) - item(ll(tensor(1.3 - h)))) / (2 * h)
    expect(Math.abs(g - fd)).toBeLessThan(1e-6)
  })

  it('the time-varying covariance converges to the steady state, and a correct model is consistent', () => {
    const ss = steadyStateKalman(cv.F, cv.H, cv.Q, cv.R)
    const T = 200
    const noise = toFlat(normal(key(5), [T, 3]))
    const xs: number[][] = []
    const zs: number[][] = []
    let x = [0, 0]
    for (let t = 0; t < T; t++) {
      x = [(x[0] ?? 0) + (x[1] ?? 0) + 0.1 * (noise[3 * t] ?? 0), (x[1] ?? 0) + 0.1 * (noise[3 * t + 1] ?? 0)]
      xs.push(x)
      zs.push([(x[0] ?? 0) + (noise[3 * t + 2] ?? 0)])
    }
    const f = kalmanFilter(cv, zs)
    const Plast = toFlat(f.predictedCovariances).slice(-4)
    toFlat(ss.prior).forEach((v, i) => {
      expect(Math.abs(v - (Plast[i] ?? 0))).toBeLessThan(1e-6)
    })
    expect(nisTest(f).consistent).toBe(true)
    const e = nees(f, xs)
    expect(e.reduce((a, b) => a + b, 0) / T).toBeLessThan(3)
  })
})
