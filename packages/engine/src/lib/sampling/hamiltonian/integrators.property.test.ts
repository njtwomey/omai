import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { numericTarget } from '../target.ts'
import {
  euclideanMetric,
  hamiltonian,
  INTEGRATOR_COEFFICIENTS,
  type IntegratorScheme,
  type IntegratorState,
  integrator,
  integratorState,
} from './integrators.ts'

// A non-Gaussian 2-D target with an exact gradient: x0 ~ N(0, 4), x1 | x0 ~ N(x0^2 / 2, 1).
const banana = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 / 4 + ((x[1] ?? 0) - 0.5 * (x[0] ?? 0) ** 2) ** 2), 2, {
  grad: (x) => {
    const a = x[0] ?? 0
    const r = (x[1] ?? 0) - 0.5 * a * a
    return [-a / 4 + r * a, -r]
  },
})

const SCHEMES = Object.keys(INTEGRATOR_COEFFICIENTS) as IntegratorScheme[]
const METRICS = [
  euclideanMetric(1, 2),
  euclideanMetric([1.5, 0.6], 2),
  euclideanMetric(
    [
      [1, 0.3],
      [0.3, 0.5],
    ],
    2,
  ),
]

const coordinate = fc.double({ min: -1.5, max: 1.5, noNaN: true })
const start = fc.tuple(coordinate, coordinate, coordinate, coordinate)

function run(
  scheme: IntegratorScheme,
  metric: (typeof METRICS)[number],
  s: IntegratorState,
  eps: number,
  n: number,
): IntegratorState {
  const step = integrator(banana, metric, scheme)
  let out = s
  for (let i = 0; i < n; i++) out = step(out, eps)
  return out
}

describe('symplectic integrators', () => {
  it('are time-reversible: integrating back with negated momentum returns to the start', () => {
    fc.assert(
      fc.property(start, fc.constantFrom(...SCHEMES), fc.integer({ min: 0, max: 2 }), ([a, b, c, d], scheme, m) => {
        const metric = METRICS[m] ?? euclideanMetric(1, 2)
        const s0 = integratorState(banana, [a, b], [c, d])
        const s1 = run(scheme, metric, s0, 0.05, 8)
        const back = run(scheme, metric, { ...s1, momentum: s1.momentum.map((v) => -v) }, 0.05, 8)
        for (let i = 0; i < 2; i++) {
          expect(back.position[i]).toBeCloseTo(s0.position[i] ?? 0, 9)
          expect(-(back.momentum[i] ?? 0)).toBeCloseTo(s0.momentum[i] ?? 0, 9)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('preserve phase-space volume: the Jacobian of the flow has determinant 1', () => {
    fc.assert(
      fc.property(start, fc.constantFrom(...SCHEMES), fc.integer({ min: 0, max: 2 }), (z, scheme, m) => {
        const metric = METRICS[m] ?? euclideanMetric(1, 2)
        const flow = (v: readonly number[]): number[] => {
          const s = run(
            scheme,
            metric,
            integratorState(banana, [v[0] ?? 0, v[1] ?? 0], [v[2] ?? 0, v[3] ?? 0]),
            0.05,
            5,
          )
          return [...s.position, ...s.momentum]
        }
        // Central differences of the 4 x 4 Jacobian, then its determinant by elimination.
        const h = 1e-5
        const jac = Array.from({ length: 4 }, (_, j) => {
          const up = [...z]
          const down = [...z]
          up[j] = (up[j] ?? 0) + h
          down[j] = (down[j] ?? 0) - h
          const fu = flow(up)
          const fd = flow(down)
          return fu.map((v, i) => (v - (fd[i] ?? 0)) / (2 * h))
        })
        // jac[j][i] = d out_i / d in_j: the transpose has the same determinant.
        const a = jac.map((r) => [...r])
        let det = 1
        for (let k = 0; k < 4; k++) {
          let p = k
          for (let r = k + 1; r < 4; r++) if (Math.abs(a[r]?.[k] ?? 0) > Math.abs(a[p]?.[k] ?? 0)) p = r
          if (p !== k) {
            const t = a[k] ?? []
            a[k] = a[p] ?? []
            a[p] = t
            det = -det
          }
          const pivot = a[k]?.[k] ?? 0
          det *= pivot
          for (let r = k + 1; r < 4; r++) {
            const f = (a[r]?.[k] ?? 0) / pivot
            for (let c = k; c < 4; c++) {
              const row = a[r] ?? []
              row[c] = (row[c] ?? 0) - f * (a[k]?.[c] ?? 0)
            }
          }
        }
        expect(det).toBeCloseTo(1, 5)
      }),
      { numRuns: RUNS },
    )
  })

  it('conserve energy to second order (velocity Verlet) and better for higher-order schemes', () => {
    const gaussian = numericTarget((x) => -0.5 * ((x[0] ?? 0) ** 2 + (x[1] ?? 0) ** 2), 2, {
      grad: (x) => x.map((v) => -v),
    })
    const metric = euclideanMetric(1, 2)
    const error = (scheme: IntegratorScheme, eps: number): number => {
      const step = integrator(gaussian, metric, scheme)
      let s = integratorState(gaussian, [1, 0.5], [0.3, -0.8])
      const h0 = hamiltonian(s, metric)
      let worst = 0
      for (let t = 0; t < Math.round(2 / eps); t++) {
        s = step(s, eps)
        worst = Math.max(worst, Math.abs(hamiltonian(s, metric) - h0))
      }
      return worst
    }
    // Halving the step size divides the energy error by about 4 for a second-order scheme.
    const ratio = error('velocityVerlet', 0.2) / error('velocityVerlet', 0.1)
    expect(ratio).toBeGreaterThan(3.5)
    expect(ratio).toBeLessThan(4.5)
    expect(error('yoshida', 0.2)).toBeLessThan(error('velocityVerlet', 0.2))
    expect(error('mcLachlan', 0.2)).toBeLessThan(error('velocityVerlet', 0.2))
  })
})
