import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../trace/index.ts'
import * as S from './index.ts'

const ones = (n: number): number[] => new Array<number>(n).fill(1)

describe('stationary iterations', () => {
  const n = 15
  const a = S.poisson1d(n)
  const c = Math.cos(Math.PI / (n + 1))

  it('has the known spectral radii on the 1-D Poisson matrix', () => {
    // Jacobi: cos(pi h); Gauss-Seidel: its square; SOR at the optimal omega: omega - 1.
    expect(S.stationarySpectralRadius(a, { method: 'jacobi' })).toBeCloseTo(c, 10)
    expect(S.stationarySpectralRadius(a, { method: 'gauss-seidel' })).toBeCloseTo(c * c, 10)
    const omega = S.sorOptimalOmega(c)
    expect(S.stationarySpectralRadius(a, { method: 'sor', omega })).toBeCloseTo(omega - 1, 6)
  })

  it('shrinks the residual at the rate of the spectral radius', () => {
    const r = S.gaussSeidel(a, ones(n), { rtol: 1e-10, maxIterations: 5000 })
    expect(r.converged).toBe(true)
    const h = r.history
    const k = h.length - 1
    const observed = ((h[k] ?? 1) / (h[k - 30] ?? 1)) ** (1 / 30)
    expect(observed).toBeCloseTo(c * c, 2)
  })

  it('orders the methods by speed, and SOR at its optimal omega is the fastest', () => {
    const iterations = (m: S.StationaryMethod, omega?: number): number =>
      S.stationarySolve(a, ones(n), {
        method: m,
        ...(omega === undefined ? {} : { omega }),
        rtol: 1e-8,
        maxIterations: 5000,
      }).iterations
    const jac = iterations('jacobi')
    const gs = iterations('gauss-seidel')
    const best = iterations('sor', S.sorOptimalOmega(c))
    expect(gs).toBeLessThan(jac)
    expect(best).toBeLessThan(gs / 3)
    // Under- and over-relaxing away from the optimum is slower.
    expect(iterations('sor', 1.1)).toBeGreaterThan(best)
    expect(iterations('sor', 1.95)).toBeGreaterThan(best)
  })

  it('damped Jacobi smooths high frequencies and barely touches low ones', () => {
    const modeAmplitude = (k: number): number => {
      const x0 = Array.from({ length: n }, (_, i) => Math.sin(((i + 1) * k * Math.PI) / (n + 1)))
      // Solve A e = 0 from the error x0: one sweep multiplies the mode by its eigenvalue of the iteration matrix.
      const after = S.stationarySolve(a, new Array<number>(n).fill(0), {
        method: 'jacobi',
        omega: 2 / 3,
        x0,
        rtol: 0,
        maxIterations: 1,
      }).x
      return Math.max(...toFlat(after).map(Math.abs)) / Math.max(...x0.map(Math.abs))
    }
    expect(modeAmplitude(1)).toBeGreaterThan(0.9)
    expect(modeAmplitude(n)).toBeLessThan(0.4)
    expect(modeAmplitude(Math.ceil((3 * n) / 4))).toBeLessThan(0.4)
  })

  it('reports divergence', () => {
    const r = S.jacobi(
      [
        [1, 2],
        [3, 1],
      ],
      [1, 1],
      { maxIterations: 2000 },
    )
    expect(r.converged).toBe(false)
  })

  it('serialises its state: pausing and resuming gives the same iterate', () => {
    const alg: Algorithm<S.StationaryState> = S.stationaryAlgorithm(a, ones(n), { method: 'sor', omega: 1.4 })
    const straight = runAlgorithm(alg, { maxSteps: 12, seed: 1 })
    const first = runAlgorithm(alg, { maxSteps: 5, seed: 1 })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(first.state)))
    const resumed = runAlgorithm(alg, { maxSteps: 7, seed: 1, start: { state: json, step: 5, serialised: true } })
    expect(resumed.state.x).toEqual(straight.state.x)
    expect(resumed.state.history).toEqual(straight.state.history)
  })

  it('rejects a zero diagonal and an unstable SOR relaxation', () => {
    expect(() =>
      S.jacobi(
        [
          [0, 1],
          [1, 2],
        ],
        [1, 1],
      ),
    ).toThrow(/diagonal/)
    expect(() => S.sor(a, ones(n), { omega: 2 })).toThrow(/omega/)
  })
})
