import { describe, expect, it } from 'vitest'
import { add, mul, sin, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { convergenceOrder } from './convergence.ts'
import { etdrk4, exponentialEuler, phi, phiMatrix, semilinearField } from './exponential.ts'
import { solveIvp } from './solve-ivp.ts'

describe('exponential integrators', () => {
  it('phi functions are continuous across the series switch', () => {
    for (const z of [-0.999999, -1.000001, 0.9999999, 1.0000001]) {
      const a = phi(z, 3)
      const b = phi(z + (z > 0 ? 1e-7 : -1e-7), 3)
      a.forEach((v, i) => {
        expect(Math.abs(v - (b[i] ?? 0))).toBeLessThan(1e-6)
      })
    }
    const z = -2.5
    expect(phi(z, 2)[2]).toBeCloseTo((Math.exp(z) - 1 - z) / (z * z), 14)
  })

  it('phiMatrix of a diagonal matrix matches the scalar phis', () => {
    const mats = phiMatrix(
      [
        [-1.5, 0],
        [0, 0.3],
      ],
      3,
    )
    for (let k = 0; k <= 3; k++) {
      const m = toFlat(mats[k] as Value)
      expect(m[0]).toBeCloseTo(phi(-1.5, 3)[k] ?? 0, 12)
      expect(m[3]).toBeCloseTo(phi(0.3, 3)[k] ?? 0, 12)
    }
  })

  // y' = -50 y + sin(y) / 2 + cos(3 t): stiff linear part, smooth nonlinearity and forcing.
  const problem = {
    linear: [-50],
    nonlinear: (t: number, y: Value): Value => add(mul(sin(y), 0.5), Math.cos(3 * t)),
  }
  const f = semilinearField(problem)
  const reference =
    toFlat(solveIvp(f, [0, 1], [1], { method: 'Radau', rtol: 1e-13, atol: 1e-15, tEval: [1] }).y)[0] ?? 0

  it('ETDRK4 is fourth order and exponential Euler first order on a stiff problem', () => {
    const err = (make: typeof etdrk4) => (dt: number) =>
      Math.abs((toFlat(solveIvp(f, [0, 1], [1], { method: make(problem, { dt }), tEval: [1] }).y)[0] ?? 0) - reference)
    // In the stiff regime (h L = -5 .. -1.25) Cox-Matthews ETDRK4 shows its known stiff order reduction to about 3.
    expect(convergenceOrder(err(etdrk4), [0.1, 0.05, 0.025]).order).toBeGreaterThan(2.9)
    const mild = { linear: [-2], nonlinear: problem.nonlinear }
    const g = semilinearField(mild)
    const ref = toFlat(solveIvp(g, [0, 1], [1], { method: 'DOP853', rtol: 1e-13, atol: 1e-15, tEval: [1] }).y)[0] ?? 0
    const e4 = (dt: number): number =>
      Math.abs((toFlat(solveIvp(g, [0, 1], [1], { method: etdrk4(mild, { dt }), tEval: [1] }).y)[0] ?? 0) - ref)
    expect(convergenceOrder(e4, [0.2, 0.1, 0.05]).order).toBeGreaterThan(3.7)
    expect(Math.abs(convergenceOrder(err(exponentialEuler), [0.05, 0.025, 0.0125]).order - 1)).toBeLessThan(0.3)
  })

  it('a matrix linear part', () => {
    const m = {
      linear: [
        [-2, 1],
        [0, -100],
      ],
      nonlinear: (_t: number, y: Value): Value => mul(y, 0),
    }
    const y = toFlat(
      solveIvp(semilinearField(m), [0, 1], tensor([1, 1]), { method: etdrk4(m, { dt: 0.25 }), tEval: [1] }).y,
    )
    const exact = toFlat(
      solveIvp(semilinearField(m), [0, 1], tensor([1, 1]), { method: 'Radau', rtol: 1e-12, atol: 1e-14, tEval: [1] }).y,
    )
    expect(Math.abs((y[0] ?? 0) - (exact[0] ?? 0))).toBeLessThan(1e-9)
  })
})
