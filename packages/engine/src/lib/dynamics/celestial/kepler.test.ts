import { describe, expect, it } from 'vitest'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import {
  conicKind,
  eccentricFromTrue,
  hyperbolicFromTrue,
  keplerAlgorithm,
  keplerBesselSeries,
  keplerLagrangeSeries,
  keplerResidual,
  keplerSolvers,
  LAPLACE_LIMIT,
  meanFromTrue,
  solveKepler,
  solveKeplerParabolic,
  trueFromEccentric,
  trueFromHyperbolic,
  wrapAngle,
} from './kepler.ts'

describe('anomaly conversions', () => {
  it('round-trips the true and eccentric anomalies on an ellipse, on the branch of the input', () => {
    for (const e of [0, 0.3, 0.9]) {
      for (const nu of [-3, -1, 0, 0.5, 2, 3.1, 4, 7]) {
        expect(trueFromEccentric(eccentricFromTrue(nu, e), e)).toBeCloseTo(nu, 12)
      }
    }
  })

  it('round-trips the true and hyperbolic anomalies inside the asymptotes', () => {
    const e = 2
    const limit = Math.acos(-1 / e)
    for (const nu of [-1.9, -0.5, 0, 1, 2]) {
      expect(Math.abs(nu)).toBeLessThan(limit)
      expect(trueFromHyperbolic(hyperbolicFromTrue(nu, e), e)).toBeCloseTo(nu, 12)
    }
    expect(() => hyperbolicFromTrue(limit + 0.01, e)).toThrow()
  })

  it('has the mean anomaly at the apsides: 0 at periapsis, pi at apoapsis', () => {
    expect(meanFromTrue(0, 0.6)).toBe(0)
    expect(meanFromTrue(Math.PI, 0.6)).toBeCloseTo(Math.PI, 14)
  })

  it('wraps angles into (-pi, pi]', () => {
    expect(wrapAngle(Math.PI)).toBe(Math.PI)
    expect(wrapAngle(-Math.PI)).toBe(Math.PI)
    expect(wrapAngle(7)).toBeCloseTo(7 - 2 * Math.PI, 14)
  })

  it('classifies the conic by its eccentricity', () => {
    expect([conicKind(0), conicKind(0.99), conicKind(1), conicKind(1.5)]).toEqual([
      'elliptic',
      'elliptic',
      'parabolic',
      'hyperbolic',
    ])
    expect(() => conicKind(-1)).toThrow()
  })
})

describe('Kepler solvers', () => {
  // The root of E - e sin E = M is checked through its residual, not by comparing one method with another.
  it('solves the elliptic equation to machine precision for every method over a grid, M any real', () => {
    for (const method of keplerSolvers.names) {
      // Fixed-point iteration contracts at e cos E, which is not below 1 in the limit: it is the method that fails there.
      for (const e of method === 'fixed-point' ? [0, 0.2, 0.7, 0.95] : [0, 0.2, 0.7, 0.95, 0.999]) {
        for (const M of [-7.5, -3, -0.2, 0, 1e-6, 0.5, 2, 3.14, 5.5, 12]) {
          const s = solveKepler(M, e, { method: method as never, maxIterations: 20000, tol: 1e-13 })
          expect(Math.abs(keplerResidual(s.anomaly, M, e))).toBeLessThan(1e-12 * Math.max(1, Math.abs(M)))
        }
      }
    }
  })

  it('solves the hyperbolic equation, residual and symmetry in M', () => {
    for (const method of ['newton', 'halley', 'bisection', 'secant', 'fixed-point'] as const) {
      for (const e of [1.01, 1.5, 5]) {
        for (const M of [-30, -1, 0, 0.1, 2, 100]) {
          const s = solveKepler(M, e, { method, maxIterations: 20000, tol: 1e-13 })
          expect(s.kind).toBe('hyperbolic')
          expect(Math.abs(keplerResidual(s.anomaly, M, e))).toBeLessThan(1e-11 * Math.max(1, Math.abs(M)))
          expect(s.anomaly).toBeCloseTo(-solveKepler(-M, e, { method, maxIterations: 20000, tol: 1e-13 }).anomaly, 10)
        }
      }
    }
  })

  it('solves Barker’s equation exactly and meets the other conics at e = 1', () => {
    for (const nu of [-2.5, -1, 0, 0.3, 1.7, 3]) {
      expect(solveKeplerParabolic(meanFromTrue(nu, 1))).toBeCloseTo(nu, 12)
    }
    expect(solveKepler(2, 1)).toMatchObject({ kind: 'parabolic', iterations: 0 })
  })

  it('converges at each method’s order: Newton quadratically, Halley cubically, bisection by bits, fixed point linearly', () => {
    const errors = (method: 'newton' | 'halley' | 'bisection' | 'fixed-point'): number[] => {
      const rec = recorder()
      runAlgorithm(keplerAlgorithm({ M: 1, e: 0.5 }, { method, tol: 1e-15 }), { maxSteps: 80, tracer: rec })
      return metricSeries(rec.run(), 'kepler/error').values
    }
    const newton = errors('newton')
    const halley = errors('halley')
    // Quadratic: the error squares (times a constant near e / (2 (1 - e cos E)) ~ 0.4 here).
    expect(newton[1] as number).toBeLessThan(0.5 * (newton[0] as number) ** 2)
    expect(newton[2] as number).toBeLessThan(0.5 * (newton[1] as number) ** 2)
    // Cubic: the error cubes, so Halley is at least as good as Newton two steps in.
    expect(halley[1] as number).toBeLessThan(newton[1] as number)
    const bisection = errors('bisection')
    // The bracket [M, M + e] has width 0.5, which halves each step.
    expect(bisection[9] as number).toBeLessThan(0.5 / 2 ** 9)
    expect(bisection.length).toBeGreaterThan(40)
    const fixed = errors('fixed-point')
    // Linear with ratio e cos E ~ 0.5 * cos(1.4987) = 0.036: each step gains well over one digit but never squares.
    expect((fixed[3] as number) / (fixed[2] as number)).toBeGreaterThan(0.01)
    expect((fixed[3] as number) / (fixed[2] as number)).toBeLessThan(0.1)
  })

  it('shows fixed-point iteration crawling as e approaches 1 while Newton does not', () => {
    const steps = (method: 'newton' | 'fixed-point'): number => {
      const alg = keplerAlgorithm({ M: 0.01, e: 0.999 }, { method, tol: 1e-12 })
      return runAlgorithm(alg, { maxSteps: 100000 }).steps
    }
    expect(steps('newton')).toBeLessThan(15)
    expect(steps('fixed-point')).toBeGreaterThan(20 * steps('newton'))
  })

  it('logs the iterate, residual, error and digits, with the error measured against Newton run to the last digit', () => {
    const rec = recorder()
    runAlgorithm(keplerAlgorithm({ M: 2, e: 0.8 }), { maxSteps: 50, tracer: rec })
    const run = rec.run()
    const digits = metricSeries(run, 'kepler/digits').values
    expect(digits.at(-1)).toBeGreaterThan(13)
    expect(metricSeries(run, 'kepler/residual').values.at(-1)).toBeLessThan(1e-13)
    const E = metricSeries(run, 'kepler/anomaly').values.at(-1) as number
    expect(Math.abs(keplerResidual(E, 2, 0.8))).toBeLessThan(1e-13)
  })

  it('reports the unreduced anomaly for M outside [0, pi]', () => {
    const rec = recorder()
    runAlgorithm(keplerAlgorithm({ M: -10, e: 0.4 }), { maxSteps: 50, tracer: rec })
    const E = metricSeries(rec.run(), 'kepler/anomaly').values.at(-1) as number
    expect(Math.abs(keplerResidual(E, -10, 0.4))).toBeLessThan(1e-12)
  })

  it('serialises mid-run and resumes to the same iterate', () => {
    const alg = keplerAlgorithm({ M: 1, e: 0.9 }, { method: 'secant' })
    const straight = runAlgorithm(alg, { maxSteps: 4 })
    const half = runAlgorithm(alg, { maxSteps: 2 })
    const restored = alg.deserialise?.(alg.serialise?.(half.state) as never) as typeof half.state
    expect(restored).toEqual(half.state)
    const resumed = alg.step(alg.step(restored, { log: () => undefined } as never), { log: () => undefined } as never)
    expect(resumed.x).toBe(straight.state.x)
  })

  it('refuses the parabola and non-finite input in the iterative solver', () => {
    expect(() => keplerAlgorithm({ M: 1, e: 1 })).toThrow()
    expect(() => keplerAlgorithm({ M: Number.NaN, e: 0.5 })).toThrow()
  })

  it('is a table of modes with a knob space', () => {
    expect(keplerSolvers.names).toEqual(['newton', 'halley', 'bisection', 'secant', 'fixed-point'])
    const alg = keplerSolvers.make({ M: 1, e: 0.3 }, { method: 'halley', tol: 1e-14, start: 'auto' }, null as never)
    expect(alg.name).toBe('kepler')
  })
})

describe('series solutions', () => {
  it('sums the Fourier-Bessel series to machine precision for moderate e', () => {
    for (const e of [0.1, 0.3, 0.5]) {
      for (const M of [0.3, 1, 2.5, -1.2]) {
        const exact = solveKepler(M, e).anomaly
        expect(keplerBesselSeries(M, e, 80)).toBeCloseTo(exact, 11)
      }
    }
  })

  it('has the Fourier-Bessel series converge more slowly as e grows', () => {
    const exact = solveKepler(1, 0.8).anomaly
    const errs = [10, 30, 60, 120].map((n) => Math.abs(keplerBesselSeries(1, 0.8, n) - exact))
    for (let k = 1; k < errs.length; k++) expect(errs[k] as number).toBeLessThan(errs[k - 1] as number)
    expect(errs[3] as number).toBeLessThan(1e-8)
  })

  it('has the Lagrange series gain about a factor e in accuracy per order at a small eccentricity', () => {
    const e = 0.05
    const M = 1.1
    const exact = solveKepler(M, e).anomaly
    const errs = [1, 2, 3, 4, 5].map((k) => Math.abs(keplerLagrangeSeries(M, e, k) - exact))
    for (let k = 1; k < errs.length; k++) expect(errs[k] as number).toBeLessThan(errs[k - 1] as number)
    // Truncating after e^k leaves a term of order e^(k + 1) with a modest coefficient.
    for (const [i, err] of errs.entries()) expect(err).toBeLessThan(2 * e ** (i + 2))
  })

  it('has the Lagrange series diverge beyond the Laplace limit: the high orders get worse at some M', () => {
    expect(LAPLACE_LIMIT).toBeCloseTo(0.6627434, 6)
    const e = 0.9
    const M = Math.PI / 2
    const exact = solveKepler(M, e).anomaly
    const errs = [1, 2, 3, 4, 5].map((k) => Math.abs(keplerLagrangeSeries(M, e, k) - exact))
    // At e = 0.9 the truncations do not converge: the order-5 error is no smaller than the order-3 one.
    expect(errs[4] as number).toBeGreaterThan(0.1 * (errs[2] as number))
    expect(() => keplerLagrangeSeries(1, 0.2, 6)).toThrow()
  })
})
