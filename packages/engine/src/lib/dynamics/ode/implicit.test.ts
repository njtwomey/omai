import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { item, matmul, mul, neg, sliceAxis, stack, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { aStability, stepperOrder } from './convergence.ts'
import {
  BACKWARD_EULER,
  bdf,
  dirk,
  IMPLICIT_MIDPOINT,
  radau,
  rosenbrock23,
  SDIRK2,
  SDIRK3,
  stiffness,
  TRAPEZOID,
} from './implicit.ts'
import { type OdeMethodName, solveIvp } from './solve-ivp.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))
const robertson = (_t: number, y: Value): Value => {
  const [a, b, c] = [at(y, 0), at(y, 1), at(y, 2)]
  const r1 = mul(a, 0.04)
  const r2 = mul(mul(b, c), 1e4)
  const r3 = mul(mul(b, b), 3e7)
  return stack([sub(r2, r1), sub(sub(r1, r2), r3), r3])
}
const vdp =
  (mu: number) =>
  (_t: number, y: Value): Value => {
    const [x, v] = [at(y, 0), at(y, 1)]
    return stack([v, sub(mul(mul(sub(1, mul(x, x)), v), mu), x)])
  }

describe('implicit solvers', () => {
  for (const method of ['Radau', 'BDF', 'LSODA', 'Rosenbrock23'] as OdeMethodName[]) {
    it(`${method} solves Robertson's problem and conserves mass`, () => {
      const sol = solveIvp(robertson, [0, 1e3], [1, 0, 0], { method, rtol: 1e-6, atol: 1e-10 })
      expect(sol.success).toBe(true)
      const y = toFlat(sol.ys.at(-1) as Value)
      expect(Math.abs(y.reduce((s, v) => s + v, 0) - 1)).toBeLessThan(1e-6)
      expect(sol.nSteps).toBeLessThan(2000)
    })
  }

  it('stiff van der Pol (mu = 1000) takes few steps with Radau and BDF', () => {
    for (const method of ['Radau', 'BDF'] as OdeMethodName[]) {
      const sol = solveIvp(vdp(1000), [0, 300], [2, 0], { method, rtol: 1e-5, atol: 1e-8 })
      expect(sol.success).toBe(true)
      expect(sol.nSteps).toBeLessThan(1500)
    }
  })

  it("a mass matrix: 2 y1' = -y1, y2' = -y2", () => {
    for (const make of [radau, bdf, rosenbrock23]) {
      const sol = solveIvp((_t, y) => neg(y), [0, 1], [1, 1], {
        method: make({
          mass: [
            [2, 0],
            [0, 1],
          ],
          rtol: 1e-8,
          atol: 1e-10,
        }),
      })
      const y = toFlat(sol.ys.at(-1) as Value)
      expect(Math.abs((y[0] ?? 0) - Math.exp(-0.5))).toBeLessThan(1e-6)
      expect(Math.abs((y[1] ?? 0) - Math.exp(-1))).toBeLessThan(1e-6)
    }
  })

  for (const t of [BACKWARD_EULER, TRAPEZOID, IMPLICIT_MIDPOINT, SDIRK2, SDIRK3]) {
    it(`${t.name} converges at order ${t.order}`, () => {
      const study = stepperOrder(
        (dt) => dirk(t, { dt }),
        (_t, y) => neg(y),
        [0, 1],
        [1],
        [Math.exp(-1)],
        [0.1, 0.05, 0.025],
      )
      expect(Math.abs(study.order - t.order)).toBeLessThan(0.3)
    })
  }

  it('A- and L-stability of the implicit tableaus', () => {
    expect(aStability(BACKWARD_EULER).lStable).toBe(true)
    expect(aStability(SDIRK2).lStable).toBe(true)
    expect(aStability(TRAPEZOID).aStable).toBe(true)
    expect(aStability(TRAPEZOID).lStable).toBe(false)
    expect(aStability(SDIRK3).aStable).toBe(true)
  })

  it('differentiates through Radau, BDF and SDIRK steps', () => {
    const k0 = 0.7
    const exact = -Math.exp(-k0)
    for (const method of ['Radau', 'BDF', 'Rosenbrock23'] as OdeMethodName[]) {
      const g = grad((k: Value) =>
        sum(solveIvp((_t, y) => neg(mul(k, y)), [0, 1], [1], { method, rtol: 1e-9, atol: 1e-12, tEval: [1] }).y),
      )
      expect(Math.abs(item(g(tensor(k0))) - exact)).toBeLessThan(5e-5)
    }
    const fixed = grad((k: Value) =>
      sum(solveIvp((_t, y) => neg(mul(k, y)), [0, 1], [1], { method: dirk(SDIRK3, { dt: 0.01 }), tEval: [1] }).y),
    )
    expect(Math.abs(item(fixed(tensor(k0))) - exact)).toBeLessThan(1e-5)
  })

  it('detects stiffness from the Jacobian', () => {
    const a = tensor([
      [-1, 0],
      [0, -1000],
    ])
    const r = stiffness((_t, y) => matmul(a, y), 0, [1, 1], { h: 0.01 })
    expect(r.ratio).toBeCloseTo(1000)
    expect(r.stiff).toBe(true)
    expect(stiffness((_t, y) => matmul(a, y), 0, [1, 1], { h: 1e-4 }).stiff).toBe(false)
  })

  it('reports failure when Newton cannot converge', () => {
    const s = dirk(BACKWARD_EULER, { dt: 10, maxNewton: 1, newtonTol: 1e-300 })
    const st = s.step(s.init((_t, y) => mul(mul(y, y), y), 0, tensor([3]), 10))
    expect(st.status).toBe('failed')
  })
})
