import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { item, mul, neg, sliceAxis, stack, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { solveIvp } from './solve-ivp.ts'
import { taylor, taylorSeriesCoefficients } from './taylor.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))
const lotka = (_t: number, y: Value): Value => {
  const [a, b] = [at(y, 0), at(y, 1)]
  return stack([sub(mul(a, 1.5), mul(a, b)), sub(mul(a, b), mul(b, 3))])
}

describe('Taylor-series integration', () => {
  it('coefficients of the harmonic oscillator are those of cos and sin', () => {
    const c = taylorSeriesCoefficients((y) => stack([at(y, 1), neg(at(y, 0))]), tensor([1, 0]), 6)
    const fact = [1, 1, 2, 6, 24, 120, 720]
    const cosSeries = [1, 0, -1, 0, 1, 0, -1]
    c.forEach((v, k) => {
      expect(toFlat(v)[0]).toBeCloseTo((cosSeries[k] ?? 0) / (fact[k] ?? 1), 14)
    })
  })

  it('matches a tight DOP853 solution on Lotka-Volterra', () => {
    const ref = toFlat(solveIvp(lotka, [0, 2], [10, 5], { method: 'DOP853', rtol: 1e-13, atol: 1e-13, tEval: [2] }).y)
    const sol = solveIvp(lotka, [0, 2], [10, 5], {
      method: taylor({ order: 16, rtol: 1e-12, atol: 1e-12 }),
      tEval: [2],
    })
    const y = toFlat(sol.y)
    expect(Math.abs((y[0] ?? 0) - (ref[0] ?? 0)) / (ref[0] ?? 1)).toBeLessThan(1e-9)
    expect(sol.nSteps).toBeLessThan(200)
  })

  it('is differentiable through its steps', () => {
    const g = grad((k: Value) =>
      sum(solveIvp((_t, y) => neg(mul(k, y)), [0, 1], [1], { method: taylor(), tEval: [1] }).y),
    )
    expect(Math.abs(item(g(tensor(0.5))) + Math.exp(-0.5))).toBeLessThan(1e-8)
  })

  it('rejects low orders', () => {
    expect(() => taylor({ order: 1 })).toThrow()
  })
})
