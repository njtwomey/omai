import { describe, expect, it } from 'vitest'
import { add, exp, mul, sin, sum, tensor, toArray, type Value } from '../tensor/index.ts'
import { grad, hessian, jacfwd, jacrev, jvp, valueAndGrad, vjp } from './transforms.ts'

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

  it('computes a Hessian', () => {
    expect(toArray(hessian((x: Value) => sum(mul(mul(x, x), x)))(tensor([1, 2])))).toEqual([
      [6, 0],
      [0, 12],
    ])
  })
})
