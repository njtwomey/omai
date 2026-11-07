import { describe, expect, it } from 'vitest'
import {
  add,
  arctan2,
  bind1,
  definePrimitive,
  div,
  matmul,
  maximum,
  minimum,
  mod,
  mul,
  pow,
  sub,
  sum,
  tensor,
  toArray,
  type Value,
} from '../tensor/index.ts'
import { grad, valueAndGrad } from './transforms.ts'

// Reverse mode tells a rule which inputs need a cotangent: the rules of the binary operations and of matmul skip the
// rest, and the gradient of the others is exactly what it was when both were computed.
const A = tensor([
  [0.5, 1.5, 2],
  [3, 0.25, 1],
])
const B = tensor([
  [1.25, 2, 0.75],
  [1, 3, 0.5],
])
const ROW = tensor([1.5, 0.5, 2])
const M = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
])

const OPS: readonly (readonly [string, (x: Value, y: Value) => Value, Value, Value])[] = [
  ['add', add, A, B],
  ['sub', sub, A, B],
  ['mul', mul, A, B],
  ['div', div, A, B],
  ['pow', pow, A, B],
  ['maximum', maximum, A, B],
  ['minimum', minimum, A, B],
  ['mod', mod, A, B],
  ['arctan2', arctan2, A, B],
  ['mul broadcast', mul, A, ROW],
  ['div broadcast', div, ROW, A],
  ['matmul', matmul, A, M],
]

describe('reverse mode passes `needs` to vjp rules', () => {
  it.each(OPS)('%s: the gradient with respect to one operand is the same bits as with both', (_name, op, x, y) => {
    const f = (p: { a: Value; b: Value }): Value => sum(mul(op(p.a, p.b), op(p.a, p.b)))
    const both = valueAndGrad(f)({ a: x, b: y }).grad
    const onlyA = grad((a: Value) => f({ a, b: y }))(x)
    const onlyB = grad((b: Value) => f({ a: x, b }))(y)
    expect(toArray(onlyA)).toEqual(toArray(both.a))
    expect(toArray(onlyB)).toEqual(toArray(both.b))
    // second order, through the same rules
    const d2 = grad((a: Value) => sum(grad((u: Value) => f({ a: u, b: y }))(a)))(x)
    const d2Both = grad((a: Value) => sum(valueAndGrad(f)({ a, b: y }).grad.a))(x)
    expect(toArray(d2)).toEqual(toArray(d2Both))
  })

  it('hands the rule `needs`: false for a constant operand, true for a tracked one, and every input of other uses', () => {
    const seen: (readonly boolean[] | undefined)[] = []
    const probe = definePrimitive<null>({
      name: 'probe',
      abstractEval: ([a]) => [a as never],
      impl: ([a, b]) => [mul(a as never, b as never) as never],
      vjp: (gs, xs, _p, _o, needs) => {
        seen.push(needs)
        return [mul(gs[0] as Value, xs[1] as Value), mul(gs[0] as Value, xs[0] as Value)]
      },
    })
    const f = (x: Value, c: Value): Value => sum(bind1(probe, [x, c], null))
    expect(toArray(grad((x: Value) => f(x, tensor([2, 3])))(tensor([1, 1])))).toEqual([2, 3])
    expect(seen.at(-1)).toEqual([true, false])
    grad((c: Value) => f(tensor([1, 1]), c))(tensor([2, 3]))
    expect(seen.at(-1)).toEqual([false, true])
    valueAndGrad((p: { x: Value; c: Value }) => f(p.x, p.c))({ x: tensor([1, 1]), c: tensor([2, 3]) })
    expect(seen.at(-1)).toEqual([true, true])
  })
})
