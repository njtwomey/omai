import { describe, expect, it } from 'vitest'
import { add, mul, neg, sliceAxis, stack, sub, sum, toFlat, type Value } from '../../tensor/index.ts'
import { consistentInitialConditions, solveDae } from './dae.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))

describe('differential-algebraic equations', () => {
  it('makes initial conditions consistent', () => {
    const dae = { f: (_t: number, y: Value) => y, g: (_t: number, y: Value, z: Value) => sub(mul(z, z), at(y, 0)) }
    const z = consistentInitialConditions(dae, 0, [4], [1])
    expect(toFlat(z)[0]).toBeCloseTo(2, 10)
  })

  it('keeps the algebraic constraint along the solution (Radau and BDF)', () => {
    // y' = -y + z, 0 = z - sin(t) y: the constraint couples a forcing into the state.
    const dae = {
      f: (_t: number, y: Value, z: Value): Value => add(neg(y), z),
      g: (t: number, y: Value, z: Value): Value => sub(z, mul(y, Math.sin(t))),
    }
    for (const method of ['Radau', 'BDF'] as const) {
      const s = solveDae(dae, [0, 3], [1], [0.3], { method, rtol: 1e-8, atol: 1e-10, tEval: [1, 2, 3] })
      expect(s.success).toBe(true)
      const y = toFlat(s.yDiff)
      const z = toFlat(s.zAlg)
      ;[1, 2, 3].forEach((t, i) => {
        expect(Math.abs((z[i] ?? 0) - (y[i] ?? 0) * Math.sin(t))).toBeLessThan(1e-7)
        // y' = y (sin t - 1): y = exp(1 - cos t - t)
        expect(Math.abs((y[i] ?? 0) - Math.exp(1 - Math.cos(t) - t))).toBeLessThan(1e-5)
      })
    }
  })

  it('a pendulum in index-1 form (tension as the algebraic variable)', () => {
    // q = (x, y), v = (u, w); x'' = -T x, y'' = -T y - 1 with T from the differentiated constraint.
    const dae = {
      f: (_t: number, s: Value, T: Value): Value => {
        const [x, y, u, w, tt] = [at(s, 0), at(s, 1), at(s, 2), at(s, 3), at(T, 0)]
        return stack([u, w, neg(mul(tt, x)), sub(neg(mul(tt, y)), 1)])
      },
      g: (_t: number, s: Value, T: Value): Value => {
        const [x, y, u, w, tt] = [at(s, 0), at(s, 1), at(s, 2), at(s, 3), at(T, 0)]
        // d^2/dt^2 (x^2 + y^2) = 0: u^2 + w^2 - T (x^2 + y^2) - y = 0
        return stack([sub(sub(add(mul(u, u), mul(w, w)), mul(tt, add(mul(x, x), mul(y, y)))), y)])
      },
    }
    const theta = 0.5
    const s = solveDae(dae, [0, 2], [Math.sin(theta), -Math.cos(theta), 0, 0], [0], { rtol: 1e-8, atol: 1e-10 })
    expect(s.success).toBe(true)
    const last = toFlat(s.yDiff).slice(-4)
    expect(Math.abs((last[0] ?? 0) ** 2 + (last[1] ?? 0) ** 2 - 1)).toBeLessThan(1e-5)
  })
})
