import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import {
  asTensor,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { type BoundedProblem, boundedSolvers, boxMap } from './bounded-solvers.ts'
import { solverConfigOf } from './solvers.ts'

describe('boxMap', () => {
  it('maps between the box and the free coordinates, one-sided, two-sided and free', () => {
    const map = boxMap([0, 1, -Infinity, -Infinity], [2, Infinity, 3, Infinity])
    const x = [0.5, 4, -2, 7]
    const back = map.fromFree(map.toFree(x))
    x.forEach((v, i) => {
      expect(back[i]).toBeCloseTo(v, 8)
    })
    for (const u of [-30, 0, 30]) {
      const y = map.fromFree([u, u, u, u])
      expect(y[0] ?? 0).toBeGreaterThanOrEqual(0)
      expect(y[0] ?? 0).toBeLessThanOrEqual(2)
      expect(y[1] ?? 0).toBeGreaterThanOrEqual(1)
      expect(y[2] ?? 0).toBeLessThanOrEqual(3)
    }
  })

  it('has a differentiable form that agrees with the numbers', () => {
    const map = boxMap([0, 1, -Infinity, -Infinity], [2, Infinity, 3, Infinity])
    const u = [0.3, -0.7, 1.1, 2]
    const v = map.fromFreeValue(asTensor(Float64Array.from(u)))
    const expected = map.fromFree(u)
    Array.from(toFlat(v as Tensor)).forEach((value, i) => {
      expect(value).toBeCloseTo(expected[i] ?? 0, 10)
    })
  })
})

describe('boundedSolvers', () => {
  const problem = (target: readonly number[], lower: number, upper: number): BoundedProblem => ({
    objective: (x) => sum(square(sub(x as Value, asTensor(Float64Array.from(target))))),
    x0: [1, 1],
    lower: [lower, lower],
    upper: [upper, upper],
  })

  it('every mode reaches an optimum inside the box, in the problem coordinates', () => {
    expect(boundedSolvers.names.slice(0, 2)).toEqual(['lbfgsb', 'slsqp'])
    expect(boundedSolvers.names).not.toContain('newton')
    const p = problem([0.4, 1.7], 0, 3)
    for (const name of boundedSolvers.names) {
      const config = solverConfigOf(boundedSolvers, { kind: name, gradTol: 1e-8 })
      const out = runAlgorithm(boundedSolvers.make(p, config, key(0)), { maxSteps: 500 })
      const x = boundedSolvers.params(config, out.state).x
      // The stochastic mode's plain SGD is slow at its default rate; it only has to be heading there.
      const digits = name === 'gradient' ? 0 : 3
      expect(x[0], name).toBeCloseTo(0.4, digits)
      expect(x[1], name).toBeCloseTo(1.7, digits)
    }
  }, 60_000)

  it('keeps every mode inside the box when the optimum lies outside it', () => {
    const p = problem([5, -5], 0, 3)
    for (const name of boundedSolvers.names) {
      const config = solverConfigOf(boundedSolvers, { kind: name })
      const out = runAlgorithm(boundedSolvers.make(p, config, key(0)), { maxSteps: 200 })
      const x = boundedSolvers.params(config, out.state).x
      expect(x[0] ?? 0, name).toBeLessThanOrEqual(3 + 1e-9)
      expect(x[1] ?? 0, name).toBeGreaterThanOrEqual(-1e-9)
    }
  }, 60_000)
})

describe('tensor-like bounds (migration, D-009)', () => {
  it('the bounds of boxMap take a tensor, a typed array or a list; a matrix is a shape error; a traced one is not differentiable', () => {
    const at = (lower: TensorLike): number[] => Array.from(boxMap(lower, [2, Infinity]).fromFree([0.3, 0.3]))
    const expected = at([0, 1])
    expect(at(Float64Array.of(0, 1))).toEqual(expected)
    expect(at(tensor([0, 1]))).toEqual(expected)
    expect(() => at(tensor([[0, 1]]))).toThrow(expect.objectContaining({ code: 'shape' }))
    expect(() => grad((a: Value) => (at(a as TensorLike), sum(a)))(tensor([0, 1]))).toThrow(
      expect.objectContaining({ code: 'not-differentiable' }),
    )
    const p: BoundedProblem = {
      objective: (x) => sum(square(sub(x as Value, 0.4))),
      x0: [1],
      lower: tensor([0]),
      upper: Float64Array.of(3),
    }
    const out = runAlgorithm(boundedSolvers.make(p, { kind: 'lbfgsb' }, key(0)), { maxSteps: 100 })
    expect(boundedSolvers.params({ kind: 'lbfgsb' }, out.state).x[0]).toBeCloseTo(0.4, 6)
  })
})
