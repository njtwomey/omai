import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad } from '../../autodiff/index.ts'
import { matmul, sum, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from './index.ts'

const FUNCTIONS: Readonly<Record<string, (x: number) => number>> = {
  exp: Math.exp,
  sin: Math.sin,
  log: Math.log,
  atan: Math.atan,
  poly: (x) => x ** 5 - 3 * x ** 2 + 1,
  gaussian: (x) => Math.exp(-(x ** 2)),
}
const POINTS: Readonly<Record<string, number>> = { exp: 1, sin: 0.7, log: 2.5, atan: -0.3, poly: 1.3, gaussian: 0.4 }

const vectorFunction = ([a = 0, b = 0, c = 0]: number[]): number[] => [a * a * b, 5 * a + Math.sin(b), Math.exp(a * c)]
const rosenbrock = ([a = 0, b = 0, c = 0]: number[]): number =>
  (1 - a) ** 2 + 100 * (b - a * a) ** 2 + (1 - b) ** 2 + 100 * (c - b * b) ** 2

describe('finite differences against numpy and scipy.differentiate', () => {
  for (const c of loadFixture('numerics/differentiate/finite_differences').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'derivative') {
        const name = p['fn_name'] as string
        expectMatches(
          D.derivative(FUNCTIONS[name] as (x: number) => number, POINTS[name] ?? 0).df,
          c.output,
          c.rtol,
          c.atol,
          c.name,
        )
      } else if (c.fn === 'gradient') {
        const spacing =
          c.inputs[1] === undefined ? ((p['spacing'] as number | undefined) ?? 1) : (decode(c.inputs[1]) as Tensor)
        const g = D.gradient(decode(c.inputs[0]) as Tensor, {
          spacing,
          edgeOrder: p['edgeOrder'] as 1 | 2,
          axis: (p['axis'] as number | undefined) ?? 0,
        })
        expectMatches(g, c.output, c.rtol, c.atol, c.name)
      } else if (c.fn === 'jacobian') {
        expectMatches(
          D.jacobianFd(vectorFunction, toFlat(decode(c.inputs[0]) as Tensor)),
          c.output,
          c.rtol,
          c.atol,
          c.name,
        )
      } else {
        expectMatches(D.hessianFd(rosenbrock, toFlat(decode(c.inputs[0]) as Tensor)), c.output, c.rtol, c.atol, c.name)
      }
    })
  }
})

describe('finite-difference tools', () => {
  it('Fornberg weights reproduce the classical stencils', () => {
    expect(toFlat(D.finiteDifferenceWeights(1, [-1, 0, 1]))).toEqual([-0.5, 0, 0.5])
    const fourth = toFlat(D.finiteDifferenceWeights(1, [-2, -1, 0, 1, 2]))
    ;[1 / 12, -2 / 3, 0, 2 / 3, -1 / 12].forEach((w, i) => {
      expect(fourth[i] ?? 0).toBeCloseTo(w, 14)
    })
    const oneSided = toFlat(D.finiteDifferenceWeights(1, [0, 1, 2]))
    ;[-1.5, 2, -0.5].forEach((w, i) => {
      expect(oneSided[i] ?? 0).toBeCloseTo(w, 14)
    })
    // Weights for the second derivative are exact on cubics over an uneven stencil.
    const stencil = [-0.3, 0.1, 0.7, 1.6]
    const w = toFlat(D.finiteDifferenceWeights(2, stencil, 0.2))
    const f = (x: number): number => 2 * x ** 3 - x * x + 4
    expect(stencil.reduce((s, x, i) => s + (w[i] ?? 0) * f(x), 0)).toBeCloseTo(12 * 0.2 - 2, 11)
  })

  it('the point and stencil take tensor-likes (migration, D-009); traced input is rejected', () => {
    const calls: [string, (v: TensorLike) => number[]][] = [
      ['jacobianFd', (v) => toFlat(D.jacobianFd(([a, b]) => [(a ?? 0) * (b ?? 0), (a ?? 0) + (b ?? 0)], v))],
      ['hessianFd', (v) => toFlat(D.hessianFd(([a, b]) => (a ?? 0) ** 2 * (b ?? 0), v))],
      ['finiteDifferenceWeights', (v) => toFlat(D.finiteDifferenceWeights(1, v))],
    ]
    for (const [name, f] of calls) {
      const v = [0.5, 1.5]
      const expected = f(v)
      expect(f(Float64Array.from(v)), name).toEqual(expected)
      expect(f(tensor(v)), name).toEqual(expected)
      expect(() => f(tensor([v, v])), name).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(() => grad((x: Value) => sum(tensor(f(x as TensorLike))))(tensor(v)), name).toThrow(
        expect.objectContaining({ code: 'not-differentiable' }),
      )
    }
  })

  it('complex-step derivatives are accurate to machine precision', () => {
    const C = D.complexOps
    const f = (z: D.ComplexNumber): D.ComplexNumber =>
      C.div(C.exp(z), C.sqrt(C.add(C.pow(C.sin(z), 3), C.pow(C.cos(z), 3))))
    const real = (x: number): number => Math.exp(x) / Math.sqrt(Math.sin(x) ** 3 + Math.cos(x) ** 3)
    // Squire and Trapp's test function at x = 1.5, against Ridders' extrapolation.
    const reference = D.derivative(real, 1.5, { initialStep: 0.1 }).df
    expect(Math.abs(D.complexStep(f, 1.5) - reference) / Math.abs(reference)).toBeLessThan(1e-12)
  })

  it('difference matrices match gradient and the second difference', () => {
    const xs = Array.from({ length: 7 }, (_, i) => 0.3 * i)
    const y = tensor(xs.map((x) => x ** 3))
    const viaMatrix = toFlat(matmul(D.differenceMatrix(7, { h: 0.3 }), y))
    const viaGradient = toFlat(D.gradient(y, { spacing: 0.3, edgeOrder: 2 }))
    viaMatrix.forEach((v, i) => {
      expect(v).toBeCloseTo(viaGradient[i] ?? 0, 12)
    })
    // The second-difference matrix is exact for cubics inside and at the second-order edges for quadratics.
    const quadratic = tensor(xs.map((x) => x * x))
    toFlat(matmul(D.differenceMatrix(7, { order: 2, h: 0.3 }), quadratic)).forEach((v) => {
      expect(v).toBeCloseTo(2, 10)
    })
  })

  it('derivative reports a small error estimate', () => {
    const r = D.derivative(Math.sin, 0.7)
    expect(r.error).toBeLessThan(1e-10)
    expect(r.table.length).toBeGreaterThan(2)
  })
})
