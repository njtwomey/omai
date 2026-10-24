import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import {
  add,
  bind1,
  cast,
  cos,
  definePrimitive,
  floor,
  type MaybeZero,
  matmul,
  mul,
  primalOf,
  round,
  sigmoid,
  sin,
  sum,
  type Tensor,
  tanh,
  tensor,
  toArray,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { checkPrimitive } from './check.ts'
import { checkpoint, customJvp, straightThrough } from './custom.ts'
import { grad, hessian, jvp, valueAndGrad } from './transforms.ts'
import { vmap } from './vmap.ts'

const rand = (shape: readonly number[], seed: number): Tensor => normal(key(seed), shape) as Tensor

function expectClose(got: Value, want: Value, tol = 1e-10): void {
  expect(got.shape).toEqual(want.shape)
  const g = toFlat(got)
  const w = toFlat(want)
  for (let i = 0; i < w.length; i++) expect(Math.abs((g[i] ?? 0) - (w[i] ?? 0))).toBeLessThanOrEqual(tol)
}

describe('customJvp', () => {
  // A logistic function whose rule uses the output, as a library would for stability.
  const logistic = customJvp({
    name: 'logisticTest',
    forward: (x) => sigmoid(x),
    jvp: (_xs, [t], out) => (t === null || t === undefined ? null : mul(t, mul(out, add(mul(out, -1), 1)))),
  })

  it('passes the conformance check in both modes, from a single forward rule', () => {
    expect(checkPrimitive(logistic, [tensor([-1.5, 0.2, 2])]).failures).toEqual([])
  })

  it('gives higher derivatives by nesting', () => {
    const want = hessian((x: Value) => sum(sigmoid(x)))(tensor([0.3, -0.4]))
    expectClose(hessian((x: Value) => sum(logistic(x)))(tensor([0.3, -0.4])), want)
  })

  it('transposes rules with several inputs, ignoring integer inputs', () => {
    const scaled = customJvp({
      name: 'scaledTest',
      forward: (x, y, k) => mul(mul(x, y), cast(k as Value, 'float64')),
      jvp: ([x, y, k], [tx, ty]) => {
        const kf = cast(k as Value, 'float64')
        const a: MaybeZero = tx === null || tx === undefined ? null : mul(mul(tx, y as Value), kf)
        const b: MaybeZero = ty === null || ty === undefined ? null : mul(mul(x as Value, ty), kf)
        return a === null ? b : b === null ? a : add(a, b)
      },
    })
    const k = tensor([3], { dtype: 'int32' })
    expect(checkPrimitive((x, y) => scaled(x, y, k), [tensor([1, 2]), tensor([0.5, -1])]).failures).toEqual([])
  })
})

describe('checkpoint', () => {
  const w = rand([3, 3], 1)
  const block = (x: Value): Value => tanh(matmul(sin(x), w))

  it('leaves values and derivatives unchanged under every policy', () => {
    const x = rand([2, 3], 2)
    const want = grad((v: Value) => sum(block(block(v))))(x)
    for (const policy of ['nothing', 'matmuls', 'everything'] as const) {
      const c = checkpoint(block, { policy })
      expectClose(c(x), block(x), 0)
      expectClose(grad((v: Value) => sum(c(c(v))))(x), want)
      expect(checkPrimitive(c, [x]).failures).toEqual([])
    }
  })

  it('recomputes the forward pass in the backward pass, except what the policy saves', () => {
    let calls = 0
    const counted = definePrimitive<null>({
      name: 'counted',
      abstractEval: ([v]) => [v as NonNullable<typeof v>],
      impl: ([v]) => {
        calls += 1
        return [(v as Tensor).backend.map(v as Tensor, Math.sin, 'float64')]
      },
      jvp: () => [null],
      vjp: ([g], [v]) => [g === null || g === undefined ? null : mul(g, cos(v as Value))],
    })
    const f = (x: Value): Value => mul(bind1(counted, [x], null), x)
    const x = tensor([0.1, 0.2])
    const run = (policy: 'nothing' | 'everything'): number => {
      calls = 0
      grad((v: Value) => sum(checkpoint(f, { policy })(v)))(x)
      return calls
    }
    expect(run('nothing')).toBe(2)
    expect(run('everything')).toBe(1)
    expect(run('nothing')).toBe(2)
    const byPredicate = checkpoint(f, { policy: (p) => p === counted })
    calls = 0
    grad((v: Value) => sum(byPredicate(v)))(x)
    expect(calls).toBe(1)
  })

  it('works in forward mode, under vmap and with integer inputs', () => {
    const x = rand([3], 3)
    const t = rand([3], 4)
    const c = checkpoint(block)
    expectClose(jvp(c, x, t).tangent, jvp(block, x, t).tangent)
    const xs = rand([4, 3], 5)
    expectClose(vmap(grad((v: Value) => sum(c(v))))(xs), vmap(grad((v: Value) => sum(block(v))))(xs))
    const shift = checkpoint((v, n) => add(v, cast(n as Value, 'float64')))
    expect(toArray(grad((v: Value) => sum(shift(v, tensor([2], { dtype: 'int32' }))))(tensor([1])))).toEqual([1])
  })

  it('lets the checkpointed function read concrete values', () => {
    const adaptive = checkpoint((v) => mul(v, Math.ceil(toFlat(primalOf(v))[0] ?? 0)))
    expect(toArray(grad((v: Value) => sum(adaptive(v)))(tensor([1.5])))).toEqual([2])
  })
})

describe('straightThrough', () => {
  it('has the value of the function and the derivative of the identity', () => {
    const q = straightThrough((v) => mul(round(mul(v, 4)), 0.25))
    const x = tensor([0.1, 0.33, -0.6])
    expect(toArray(q(x))).toEqual([0, 0.25, -0.5])
    const { value, grad: g } = valueAndGrad((v: Value) => sum(mul(q(v), v)))(x)
    expect(toArray(value)).toBeCloseTo(0.25 * 0.33 + 0.5 * 0.6)
    // d/dv [q(v) v] with dq/dv = 1: q(v) + v
    expectClose(g, add(q(x), x))
    expect(toArray(jvp(straightThrough(floor), tensor(1.7), tensor(3)).tangent)).toBe(3)
  })
})
