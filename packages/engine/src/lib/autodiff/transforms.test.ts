import { describe, expect, it } from 'vitest'
import { add, exp, mul, primalOf, sin, sum, tensor, toArray, type Value } from '../tensor/index.ts'
import { grad, gradAndAux, hessian, hvp, jacfwd, jacrev, jvp, valueAndGrad, vhp, vjp } from './transforms.ts'

describe('transforms', () => {
  it('differentiates a polynomial', () => {
    const f = (x: Value) => add(mul(mul(x, x), x), mul(x, 2))
    expect(toArray(grad(f)(tensor(2)))).toBeCloseTo(14)
  })

  it('returns value and gradient over a pytree', () => {
    const { value, grad: g } = valueAndGrad((p: { a: Value; b: Value }) => sum(mul(p.a, p.b)))({
      a: tensor([1, 2]),
      b: tensor([3, 4]),
    })
    expect(toArray(value)).toBe(11)
    expect(toArray(g.a)).toEqual([3, 4])
    expect(toArray(g.b)).toEqual([1, 2])
  })

  it('computes second derivatives by nesting', () => {
    const f = (x: Value) => mul(mul(x, x), x)
    const d2 = grad((x: Value) => grad(f)(x))(tensor(3))
    expect(toArray(d2)).toBeCloseTo(18)
  })

  it('handles closures over outer traced values (perturbation confusion)', () => {
    // d/dx [ x * d/dy (x + y) ] = d/dx [ x * 1 ] = 1
    const outer = grad((x: Value) => mul(x, grad((y: Value) => add(x, y))(tensor(1))))(tensor(5))
    expect(toArray(outer)).toBeCloseTo(1)
  })

  it('forward and reverse agree', () => {
    const f = (x: Value) => sin(mul(x, exp(x)))
    const x = tensor([0.3, -0.7])
    expect(toArray(jacfwd(f)(x))).toEqual(toArray(jacrev(f)(x)))
    const t = jvp(f, x, tensor([1, 0])).tangent
    const { pullback } = vjp(f, x)
    expect((toArray(pullback(tensor([1, 0]))) as number[])[0]).toBeCloseTo((toArray(t) as number[])[0] as number)
  })

  it('passes auxiliary data through the gradient', () => {
    const { grad: g, aux } = gradAndAux((p: { w: Value }) => {
      const loss = sum(mul(p.w, p.w))
      return [loss, { loss: toArray(primalOf(loss)) }] as const
    })({ w: tensor([1, 3]) })
    expect(toArray(g.w)).toEqual([2, 6])
    expect(aux).toEqual({ loss: 10 })
  })

  it('vector-Hessian products agree with Hessian-vector products and the Hessian', () => {
    const f = (x: Value) => sum(mul(sin(x), mul(x, x)))
    const x = tensor([0.3, -1.2, 2])
    const v = tensor([1, -2, 0.5])
    const left = toArray(vhp(f, x, v)) as number[]
    const right = toArray(hvp(f, x, v)) as number[]
    const h = toArray(hessian(f)(x)) as number[][]
    left.forEach((value, i) => {
      expect(value).toBeCloseTo(right[i] as number, 12)
      const column = h.reduce((acc, row, j) => acc + (row[i] as number) * ([1, -2, 0.5][j] as number), 0)
      expect(value).toBeCloseTo(column, 12)
    })
  })

  it('computes a Hessian', () => {
    expect(toArray(hessian((x: Value) => sum(mul(mul(x, x), x)))(tensor([1, 2])))).toEqual([
      [6, 0],
      [0, 12],
    ])
  })
})
