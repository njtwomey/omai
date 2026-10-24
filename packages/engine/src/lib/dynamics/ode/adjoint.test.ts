import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad, jvp } from '../../autodiff/index.ts'
import { item, mul, sliceAxis, stack, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { type AdjointMethod, compareGradientMethods, forwardSensitivities, odeintAdjoint } from './adjoint.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))
// Lotka-Volterra with four parameters.
const lotka = (_t: number, y: Value, p: Value): Value => {
  const [a, b] = [at(y, 0), at(y, 1)]
  const [alpha, beta, delta, gamma] = [at(p, 0), at(p, 1), at(p, 2), at(p, 3)]
  return stack([sub(mul(alpha, a), mul(mul(beta, a), b)), sub(mul(mul(delta, a), b), mul(gamma, b))])
}
const y0 = tensor([2, 1])
const p0 = tensor([1.5, 1, 1, 3])
const times = [0, 0.5, 1.5]

describe('adjoint and forward sensitivities', () => {
  for (const adjoint of ['backsolve', 'interpolated', 'checkpointed'] as AdjointMethod[]) {
    it(`the ${adjoint} adjoint passes checkPrimitive (vjp, jvp, dot product and vmap)`, () => {
      const f = (y: Value, p: Value): Value => odeintAdjoint(lotka, y, times, p, { rtol: 1e-10, atol: 1e-12, adjoint })
      const report = checkPrimitive(f, [y0, p0], { rtol: 1e-5, atol: 1e-7 })
      expect(report.failures).toEqual([])
    })
  }

  it('forward sensitivities match the jvp of the solve and finite differences', () => {
    const s = forwardSensitivities(lotka, y0, times, p0, { rtol: 1e-10, atol: 1e-12 })
    const tangent = jvp(
      (p: Value) => odeintAdjoint(lotka, y0, times, p, { rtol: 1e-10, atol: 1e-12 }),
      p0,
      tensor([0, 1, 0, 0]),
    ).tangent
    const t = toFlat(tangent)
    // dy(t_2)/d beta, both components
    expect(Math.abs((t[4] ?? 0) - (s.dp.data[2 * 8 + 1] ?? 0))).toBeLessThan(1e-7)
    expect(Math.abs((t[5] ?? 0) - (s.dp.data[2 * 8 + 4 + 1] ?? 0))).toBeLessThan(1e-7)
    const h = 1e-6
    const plus = toFlat(odeintAdjoint(lotka, y0, times, tensor([1.5, 1 + h, 1, 3]), { rtol: 1e-12, atol: 1e-14 }))
    const minus = toFlat(odeintAdjoint(lotka, y0, times, tensor([1.5, 1 - h, 1, 3]), { rtol: 1e-12, atol: 1e-14 }))
    expect(Math.abs(((plus[4] ?? 0) - (minus[4] ?? 0)) / (2 * h) - (t[4] ?? 0))).toBeLessThan(1e-5)
  })

  it('every gradient method agrees with the reference', () => {
    const loss = (ys: Value): Value => sum(mul(ys, ys))
    const results = compareGradientMethods(lotka, y0, times, p0, loss, { rtol: 1e-9, atol: 1e-11 })
    for (const r of results) expect(r.error).toBeLessThan(1e-5)
    const backsolve = results.find((r) => r.method === 'backsolve')
    const direct = results.find((r) => r.method === 'direct')
    expect(backsolve?.stored ?? 0).toBeLessThan(direct?.stored ?? 0)
  })

  it('times[0] passes the initial state through', () => {
    const ys = odeintAdjoint(lotka, y0, [0], p0)
    expect(toFlat(ys)).toEqual([2, 1])
    const g = grad((y: Value) => sum(odeintAdjoint(lotka, y, [0, 0], p0)))(y0)
    expect(toFlat(g)).toEqual([2, 2])
    expect(item(sum(odeintAdjoint(lotka, y0, [0, 1], p0)))).toBeGreaterThan(0)
  })
})
