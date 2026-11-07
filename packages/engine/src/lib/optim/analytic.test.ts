import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { sum, type TensorLike, tensor, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { analyticObjective, finiteDifferenceHessian } from './analytic.ts'
import { flatObjective } from './minimize/objective.ts'
import { smoothSolvers } from './solvers.ts'

// f(x) = cosh(x0 - 1) + (x1 + 2)^4 / 4 + x0 x1 / 10, with its derivatives written out.
const spec = {
  value: (x: Float64Array): number =>
    Math.cosh((x[0] ?? 0) - 1) + ((x[1] ?? 0) + 2) ** 4 / 4 + ((x[0] ?? 0) * (x[1] ?? 0)) / 10,
  gradient: (x: Float64Array): number[] => [
    Math.sinh((x[0] ?? 0) - 1) + (x[1] ?? 0) / 10,
    ((x[1] ?? 0) + 2) ** 3 + (x[0] ?? 0) / 10,
  ],
  hessian: (x: Float64Array): number[] => [Math.cosh((x[0] ?? 0) - 1), 0.1, 0.1, 3 * ((x[1] ?? 0) + 2) ** 2],
}

describe('analyticObjective', () => {
  it('gives the solvers the written gradient and Hessian', () => {
    const flat = flatObjective(analyticObjective(spec), [0.3, -1])
    const x = [0.3, -1]
    expect(flat.value(x)).toBeCloseTo(spec.value(Float64Array.from(x)), 12)
    expect(Array.from(flat.valueAndGrad(x).grad)).toEqual(spec.gradient(Float64Array.from(x)))
    expect(Array.from(flat.hessian(x))).toEqual(spec.hessian(Float64Array.from(x)))
    expect(Array.from(flat.hvp(x, [1, 2]))).toEqual([
      (spec.hessian(Float64Array.from(x))[0] ?? 0) + 2 * 0.1,
      0.1 + 2 * (spec.hessian(Float64Array.from(x))[3] ?? 0),
    ])
  })

  it('differences the gradient when there is no Hessian', () => {
    const h = finiteDifferenceHessian((x) => spec.gradient(x), [0.3, -1]).data
    const exact = spec.hessian(Float64Array.from([0.3, -1]))
    for (let k = 0; k < 4; k++) expect(h[k]).toBeCloseTo(exact[k] ?? 0, 6)
    const { hessian: _drop, ...without } = spec
    const flat = flatObjective(analyticObjective(without), [0.3, -1])
    for (let k = 0; k < 4; k++) expect(flat.hessian([0.3, -1])[k]).toBeCloseTo(exact[k] ?? 0, 6)
  })

  it('is minimised by every smooth solver to the same point', () => {
    const objective = analyticObjective(spec)
    const reference = runAlgorithm(
      smoothSolvers.make({ objective, x0: [0, 0] }, { kind: 'newton', gradTol: 1e-12 }, key(0)),
      { maxSteps: 100 },
    )
    const x = (reference.state as { x: Float64Array }).x
    expect(Math.max(...spec.gradient(x).map(Math.abs))).toBeLessThan(1e-10)
    for (const config of [
      { kind: 'bfgs', gradTol: 1e-9 },
      { kind: 'lbfgs', gradTol: 1e-9 },
      { kind: 'conjugateGradient', gradTol: 1e-9 },
    ]) {
      const run = runAlgorithm(smoothSolvers.make({ objective, x0: [0, 0] }, config, key(0)), { maxSteps: 500 })
      const y = (run.state as { x: Float64Array }).x
      expect(y[0], config.kind).toBeCloseTo(x[0] ?? 0, 5)
      expect(y[1], config.kind).toBeCloseTo(x[1] ?? 0, 5)
    }
  })
})

describe('tensor-like inputs (migration, D-009)', () => {
  it('the point takes a tensor, a typed array or a list and the Hessian is an [m, m] tensor; a matrix is a shape error; a traced one is not differentiable', () => {
    const at = (x: TensorLike): number[] =>
      toFlat(finiteDifferenceHessian((v) => [2 * (v[0] ?? 0), 6 * (v[1] ?? 0)], x))
    const expected = at([1, 1])
    expect(at(Float64Array.of(1, 1))).toEqual(expected)
    expect(at(tensor([1, 1]))).toEqual(expected)
    expect(finiteDifferenceHessian((v) => v, [1, 1, 1]).shape).toEqual([3, 3])
    expect(() => at(tensor([[1, 1]]))).toThrow(expect.objectContaining({ code: 'shape' }))
    expect(() =>
      grad((a: Value) => {
        at(a as TensorLike)
        return sum(a)
      })(tensor([1, 1])),
    ).toThrow(expect.objectContaining({ code: 'not-differentiable' }))
  })
})
