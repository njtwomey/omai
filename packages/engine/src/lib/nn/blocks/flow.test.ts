import { describe, expect, it } from 'vitest'
import { expectClose } from '../../../testing/close.ts'
import { minimise } from '../../../testing/optimise.ts'
import { checkPrimitive } from '../../autodiff/index.ts'
import { odeSolver } from '../../dynamics/ode/index.ts'
import { expm, slogdet } from '../../numerics/linalg/index.ts'
import { key, normal } from '../../random/index.ts'
import { add, item, matmul, mean, mul, sum, tensor, toFlat, transpose, type Value } from '../../tensor/index.ts'
import { Linear } from '../module/layers.ts'
import { CNF, standardNormalLogProb } from './flow.ts'

const rk4 = odeSolver('RK4', { dt: 0.125 })
const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)

/** A flow whose field is the linear map z -> z A^T (no bias, time independent). */
function linearFlow(a: number[][], trace: 'exact' | 'hutchinson' = 'exact'): CNF {
  const field = new Linear(a.length, a.length, { bias: false, key: key(0) })
  field.loadStateDict({ weight: tensor(a) })
  return new CNF(a.length, { solver: rk4, field, trace })
}

describe('continuous normalising flow', () => {
  it('for a linear field the map is the matrix exponential and the log determinant is the trace', () => {
    const a = [
      [0.3, -0.8, 0.1],
      [0.5, -0.2, 0.4],
      [-0.3, 0.2, 0.1],
    ]
    const flow = linearFlow(a)
    const x = n(1, [4, 3])
    const out = flow.forward({ x })
    expectClose(out.z, matmul(x, transpose(expm(tensor(a)))), 1e-6)
    const traceA = a.reduce((t, row, i) => t + (row[i] as number), 0)
    for (const v of toFlat(out.logDet)) expect(v).toBeCloseTo(traceA, 12)
    // log |det exp(A)| = tr(A): the integrated trace is the exact log determinant of the map.
    expect(item(slogdet(expm(tensor(a))).logabsdet)).toBeCloseTo(traceA, 10)
  })

  it('the log determinant of a nonlinear flow matches the finite-difference Jacobian of the map', () => {
    const flow = new CNF(2, { solver: rk4, hidden: [8], key: key(2) })
    const x = tensor([[0.4, -0.3]])
    const out = flow.forward({ x })
    const h = 1e-6
    const z = (p: number[]): number[] => toFlat(flow.forward({ x: tensor([p]) }).z)
    const jac = [0, 1].map((j) => {
      const plus = z([0.4 + (j === 0 ? h : 0), -0.3 + (j === 1 ? h : 0)])
      const minus = z([0.4 - (j === 0 ? h : 0), -0.3 - (j === 1 ? h : 0)])
      return plus.map((v, i) => (v - (minus[i] as number)) / (2 * h))
    })
    const [[a, b], [c, d]] = jac as [[number, number], [number, number]]
    const det = a * d - b * c
    expect(item(out.logDet)).toBeCloseTo(Math.log(Math.abs(det)), 6)
  })

  it('the density integrates to one and sampling inverts the map', () => {
    const flow = new CNF(1, { solver: rk4, hidden: [10], key: key(3) })
    let total = 0
    const h = 0.05
    for (let x = -9; x <= 9; x += h) total += Math.exp(item(flow.logProb(flow.params, [[x]]))) * h
    expect(total).toBeCloseTo(1, 3)
    const x = n(4, [5, 1])
    const out = flow.forward({ x })
    expectClose(flow.sample(flow.params, out.z), x, 1e-4)
  })

  it('Hutchinson is exact for a diagonal Jacobian and unbiased in general', () => {
    const exact = new CNF(3, { solver: rk4, hidden: [6], key: key(6) })
    const stochastic = new CNF(3, { solver: rk4, hidden: [6], trace: 'hutchinson', key: key(6) })
    const rows = 6000
    const z = add(mul(n(7, [rows, 3]), 0), tensor([[0.3, -0.5, 0.8]]))
    const truth = toFlat(exact.jacobianTrace(exact.params, 0.2, z))[0] as number
    const eps = n(8, [rows, 3])
    const estimates = toFlat(stochastic.jacobianTrace(stochastic.params, 0.2, z, eps))
    const m = estimates.reduce((a, v) => a + v, 0) / rows
    const sd = Math.sqrt(estimates.reduce((a, v) => a + (v - m) ** 2, 0) / (rows - 1))
    expect(Math.abs(m - truth)).toBeLessThan((4 * sd) / Math.sqrt(rows))
    expect(sd).toBeGreaterThan(0)
    // Rademacher noise on a field with a diagonal Jacobian gives zero variance.
    const field = new Linear(3, 3, { bias: false, key: key(9) })
    field.loadStateDict({
      weight: tensor([
        [0.5, 0, 0],
        [0, -1, 0],
        [0, 0, 0.25],
      ]),
    })
    const diag = new CNF(3, { solver: rk4, field, trace: 'hutchinson', noise: 'rademacher' })
    const out = diag.forward({ x: n(10, [4, 3]), key: key(11) })
    for (const v of toFlat(out.logDet)) expect(v).toBeCloseTo(-0.25, 12)
  })

  it('maximum likelihood recovers the log scale of a Gaussian with a linear field', () => {
    const data = mul(n(12, [400, 1]), 2)
    const flow = new CNF(1, { solver: rk4, field: new Linear(1, 1, { bias: false, key: key(13) }) })
    const loss = (p: Record<string, unknown>): Value => mean(mul(flow.logProb(p, data), -1))
    const fit = minimise(flow.params, loss, 150, 0.05)
    const a = toFlat((fit.params['field'] as Record<string, Value>)['weight'] as Value)[0] as number
    // z(1) = exp(a) x, so the maximiser of the log-likelihood is a = -log(mean x^2) / 2.
    const second = toFlat(data).reduce((s, v) => s + v * v, 0) / 400
    expect(a).toBeCloseTo(-0.5 * Math.log(second), 3)
    expect(a).toBeLessThan(-0.5)
  })

  it('is differentiable in the field parameters, with the exact and the stochastic trace', () => {
    for (const trace of ['exact', 'hutchinson'] as const) {
      const flow = new CNF(2, { solver: odeSolver('RK4', { dt: 0.5 }), hidden: [3], trace, key: key(14) })
      const x = tensor([
        [0.3, -0.2],
        [0.1, 0.5],
      ])
      const w = ((flow.params['field'] as Record<string, unknown>)['0'] as Record<string, Value>)['weight'] as Value
      const f = (p: Value): Value =>
        flow.logProb(
          {
            field: {
              ...(flow.params['field'] as object),
              '0': {
                weight: p,
                bias: ((flow.params['field'] as Record<string, unknown>)['0'] as Record<string, Value>)['bias'],
              },
            },
          },
          x,
          { key: key(15) },
        )
      expect(checkPrimitive(f, [w], { rtol: 1e-4 }).failures, trace).toEqual([])
    }
  })

  it('the standard normal base has the usual log density', () => {
    expect(item(standardNormalLogProb([[0, 0]]))).toBeCloseTo(-Math.log(2 * Math.PI), 14)
    expect(sum(standardNormalLogProb([[1], [2]])).shape).toEqual([])
  })
})
