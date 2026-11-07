import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import {
  abs,
  add,
  arctan2,
  argmax,
  argsort,
  bind1,
  broadcastTo,
  cast,
  concat,
  cos,
  cumprod,
  cumsum,
  definePrimitive,
  diagonal,
  div,
  dot,
  einsum,
  exp,
  expandDims,
  flip,
  floor,
  greater,
  item,
  kron,
  log,
  logsumexp,
  matmul,
  max,
  maximum,
  mean,
  mul,
  oneHot,
  outer,
  pad,
  pow,
  primalOf,
  prod,
  repeat,
  reshape,
  roll,
  scatterAdd,
  segmentSum,
  sigmoid,
  sin,
  slice,
  softmax,
  sort,
  split,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  take,
  takeAlongAxis,
  tanh,
  tensor,
  tile,
  toArray,
  topk,
  trace,
  transpose,
  type Value,
  variance,
  where,
} from '../tensor/index.ts'
import { customVjp } from './custom.ts'
import { grad, hessian, jacrev, jvp, valueAndGrad, vjp } from './transforms.ts'
import { vmap } from './vmap.ts'

const rand = (shape: readonly number[], seed: number): Tensor => normal(key(seed), shape) as Tensor
const pos = (shape: readonly number[], seed: number): Value => add(abs(rand(shape, seed)), 0.5)

function exampleOf(x: Value, axis: number, b: number): Value {
  return squeeze(take(x, [b], axis), axis)
}

/** The reference: call `f` on each example and stack the results along `outAxis`. */
function loop(
  f: (...xs: Value[]) => Value,
  args: readonly Value[],
  axes: readonly (number | null)[],
  outAxis = 0,
): Value {
  const i = axes.findIndex((a) => a !== null)
  const ax = axes[i] as number
  const first = args[i] as Value
  const n = first.shape[ax < 0 ? first.shape.length + ax : ax] ?? 0
  const outs: Value[] = []
  for (let b = 0; b < n; b++) {
    outs.push(
      f(
        ...args.map((x, j) => {
          const a = axes[j] ?? null
          return a === null ? x : exampleOf(x, a < 0 ? x.shape.length + a : a, b)
        }),
      ),
    )
  }
  return stack(outs, outAxis)
}

function expectClose(got: Value, want: Value, tol = 1e-12): void {
  expect(got.shape).toEqual(want.shape)
  const g = (got as Tensor).data
  const w = (want as Tensor).data
  for (let i = 0; i < w.length; i++) {
    const a = g[i] ?? Number.NaN
    const b = w[i] ?? Number.NaN
    if (Number.isNaN(b)) expect(Number.isNaN(a)).toBe(true)
    else expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)))
  }
}

function agrees(f: (...xs: Value[]) => Value, args: readonly Value[], axes: readonly (number | null)[]): void {
  expectClose(vmap(f, axes)(...args), loop(f, args, axes))
}

describe('vmap against a loop', () => {
  const x = rand([4, 3], 1)
  const y = rand([4, 3], 2)
  const yt = rand([3, 4], 2)
  const m = rand([4, 3, 5], 3)

  it('elementwise unary operations', () => {
    for (const f of [exp, sin, cos, tanh, sigmoid, floor, abs, (v: Value) => log(add(abs(v), 1))]) {
      agrees(f, [x], [0])
      agrees(f, [x], [1])
    }
    agrees(sqrt, [pos([4, 3], 4)], [0])
  })

  it('binary operations with broadcasting and shared operands', () => {
    for (const f of [add, sub, mul, div, maximum, arctan2, (a: Value, b: Value) => pow(abs(a), b)]) {
      agrees(f, [x, y], [0, 0])
      agrees(f, [x, yt], [0, 1])
      agrees(f, [x, rand([3], 5)], [0, null])
      agrees(f, [rand([4], 5), x], [null, 1])
      agrees(f, [m, rand([3, 1], 6)], [0, null])
      agrees(f, [m, rand([4, 5], 22)], [0, 0])
      agrees(f, [rand([5, 4], 7), m], [1, 0])
    }
  })

  it('comparisons, where and casts', () => {
    agrees((a, b) => cast(greater(a, b), 'float64'), [x, yt], [0, 1])
    agrees((a, b) => where(greater(a, 0), a, b), [x, rand([3], 8)], [0, null])
    agrees((a, b) => where(greater(b, 0), 1, a), [m, rand([3, 5], 23)], [2, 1])
    agrees((a) => cast(cast(mul(a, 10), 'int32'), 'float64'), [x], [1])
  })

  it('reductions and scans', () => {
    for (const axis of [0, 1, -1]) {
      agrees((a) => sum(a, axis), [m], [axis === 0 ? 1 : 0])
      agrees((a) => mean(a, axis, true), [m], [2])
      agrees((a) => max(a, axis), [m], [1])
      agrees((a) => cast(argmax(a, axis), 'float64'), [m], [0])
      agrees((a) => cumsum(a, axis), [m], [1])
      agrees((a) => cumprod(a, axis), [m], [2])
      agrees((a) => logsumexp(a, axis), [m], [0])
      agrees((a) => softmax(a, axis), [m], [2])
    }
    agrees((a) => sum(a), [m], [1])
    agrees((a) => prod(a, [0, 1]), [m], [2])
    agrees((a) => variance(a, 0), [m], [0])
  })

  it('shape operations', () => {
    agrees((a) => reshape(a, [15]), [m], [0])
    agrees((a) => reshape(a, [5, 4]), [m], [1])
    agrees((a) => transpose(a), [m], [1])
    agrees((a) => swapaxes(a, 0, 1), [m], [2])
    agrees((a) => broadcastTo(a, [2, 3]), [x], [0])
    agrees((a) => broadcastTo(a, [2, 3, 4]), [rand([4, 3, 1], 9)], [0])
    agrees((a, b) => concat([a, b], 0), [x, rand([2], 10)], [1, null])
    agrees((a, b) => stack([a, b], 1), [x, yt], [0, 1])
    agrees((a) => pad(a, [[1, 2]], 3), [x], [0])
    agrees(
      (a) =>
        pad(a, [
          [1, 0],
          [0, 2],
        ]),
      [m],
      [1],
    )
    agrees((a) => flip(a, 1), [m], [0])
    agrees((a) => roll(a, 2, 1), [m], [2])
    agrees((a) => tile(a, [2, 1]), [m], [0])
    agrees((a) => repeat(a, 2, 0), [m], [1])
    agrees((a) => expandDims(a, -1), [m], [0])
    agrees((a) => split(a, [1, 3], 1)[1] as Value, [m], [0])
  })

  it('indexing operations', () => {
    const idx = tensor([2, 0, 0, 1], { dtype: 'int32' })
    const perEx = tensor(
      [
        [0, 2],
        [1, 1],
        [2, 0],
        [0, 0],
      ],
      { dtype: 'int32' },
    )
    agrees((a) => slice(a, [1, 0], [3, 4], [1, 2]), [m], [0])
    agrees((a) => slice(a, [2], [undefined], [-1]), [x], [0])
    agrees((a) => take(a, [2, 0], 1), [m], [0])
    agrees((a) => take(a, [1, 1, 0], 0), [m], [2])
    agrees((a, i) => take(a, i), [x, idx], [null, 0])
    agrees((a, i) => take(a, i, 1), [m, perEx], [null, 0])
    agrees((a, i) => take(a, i, 1), [m, perEx], [0, 0])
    agrees((a, i) => take(a, i, 0), [rand([3, 4, 5], 24), perEx], [1, 0])
    agrees((a, i) => takeAlongAxis(a, i, 0), [yt, reshape(idx, [4, 1])], [1, 0])
    agrees((a, i) => takeAlongAxis(a, i, 0), [rand([3], 11), reshape(idx, [4, 1])], [null, 0])
    agrees((a, i) => segmentSum(a, i, 3), [rand([4, 2, 2], 12), perEx], [0, 0])
    agrees((a, i) => segmentSum(a, i, 3), [rand([2, 2], 12), perEx], [null, 0])
    agrees((a, i) => segmentSum(a, i, 3), [rand([4, 2, 2], 12), tensor([0, 2], { dtype: 'int32' })], [1, null])
    agrees((u, i) => scatterAdd([3], i, u, 0), [rand([4, 2], 13), perEx], [0, 0])
    agrees((a) => sort(a, 0), [m], [1])
    agrees((a) => cast(argsort(a, -1, true), 'float64'), [m], [0])
    agrees((a) => topk(a, 2).values, [m], [2])
    agrees((i) => oneHot(i, 3), [idx], [0])
  })

  it('products and contractions', () => {
    const a = rand([4, 3, 3], 14)
    agrees(matmul, [a, rand([3, 2], 15)], [0, null])
    agrees(matmul, [rand([2, 3], 15), rand([3, 2, 4], 15)], [null, 2])
    agrees(matmul, [a, rand([4, 3, 2], 16)], [0, 0])
    agrees(matmul, [a, rand([3, 3, 2], 16)], [1, 1])
    agrees(matmul, [rand([4, 2, 3, 3], 17), a], [0, 0])
    agrees(matmul, [a, rand([3, 4], 18)], [0, 1])
    agrees(dot, [x, yt], [0, 1])
    agrees(outer, [x, rand([2], 19)], [0, null])
    agrees((p, q) => einsum('ij,jk->ik', p, q), [a, rand([3, 2], 20)], [0, null])
    agrees((p, q) => kron(p, q), [a, rand([2, 2], 21)], [2, null])
    agrees((p) => trace(p), [a], [0])
    agrees((p) => diagonal(p), [a], [1])
  })

  it('places the batch where out-axes say and passes shared arguments through', () => {
    const f = (a: Value, b: Value): Value => mul(a, b)
    expectClose(vmap(f, [0, null], 1)(x, rand([3], 5)), loop(f, [x, rand([3], 5)], [0, null], 1))
    const consts = vmap((_a: Value) => tensor([1, 2]))(x)
    expect(toArray(consts)).toEqual([
      [1, 2],
      [1, 2],
      [1, 2],
      [1, 2],
    ])
    const tree = vmap((p: { a: Value; b: Value }) => [sum(p.a), mul(p.b, 2)], 0, [0, 1])({ a: x, b: y })
    expect((tree[0] as Value).shape).toEqual([4])
    expect((tree[1] as Value).shape).toEqual([3, 4])
  })

  it('falls back to a loop for primitives without a batch rule', () => {
    const cube = definePrimitive<null>({
      name: 'cube',
      abstractEval: ([v]) => [v as NonNullable<typeof v>],
      impl: ([v]) => [(v as Tensor).backend.map(v as Tensor, (z) => z * z * z, 'float64')],
      jvp: ([v], [t]) => [t === null || t === undefined ? null : mul(t, mul(3, mul(v as Value, v as Value)))],
      vjp: ([g], [v]) => [g === null || g === undefined ? null : mul(g, mul(3, mul(v as Value, v as Value)))],
    })
    const f = (v: Value): Value => sum(bind1(cube, [v], null))
    agrees(f, [m], [1])
    expectClose(vmap(grad(f))(x), loop(grad(f), [x], [0]))
    expect(vmap((v: Value) => bind1(cube, [v], null))(rand([0, 2], 1)).shape).toEqual([0, 2])
  })

  it('loops over the examples of a function that needs concrete values', () => {
    // The number of terms depends on the data, so this cannot be traced as a batch.
    const series = (v: Value): Value => {
      const n = Math.ceil(item(primalOf(max(abs(v)))) * 4)
      let total: Value = mul(v, 0)
      for (let j = 0; j < n; j++) total = add(total, pow(v, j))
      return total
    }
    const xs = rand([4, 3], 50)
    agrees(series, [xs], [0])
    agrees(series, [xs], [1])
    expectClose(
      vmap(grad((v: Value) => sum(series(v))))(xs),
      loop(
        grad((v: Value) => sum(series(v))),
        [xs],
        [0],
      ),
    )
    const pair = vmap((v: Value) => [series(v), sum(v)], 0, [1, 0])(xs)
    expect((pair[0] as Value).shape).toEqual([3, 4])
    expect((pair[1] as Value).shape).toEqual([4])
  })

  it('reports a missing derivative rule instead of looping over the examples and running the function twice', () => {
    let calls = 0
    const noRule = definePrimitive<null>({
      name: 'noRuleForVmap',
      abstractEval: ([v]) => [v as NonNullable<typeof v>],
      impl: ([v]) => [v as Tensor],
    })
    const f = (v: Value): Value => {
      calls += 1
      return sum(bind1(noRule, [v], null))
    }
    let caught: unknown
    try {
      vmap(grad(f))(x)
    } catch (e) {
      caught = e
    }
    expect(caught).toMatchObject({ code: 'not-differentiable', details: { primitive: 'noRuleForVmap' } })
    expect(calls).toBe(1)
  })

  it('rejects mismatched batch sizes and unbatched calls', () => {
    expect(() => vmap(add)(x, rand([3, 3], 1))).toThrow(/batch sizes/)
    expect(() => vmap(add, null)(x, y)).toThrow(/no argument/)
  })
})

describe('vmap composes with differentiation', () => {
  const f = (w: Value, xi: Value): Value => sum(tanh(matmul(xi, w)))
  const w = rand([3, 2], 30)
  const xs = rand([5, 3], 31)

  it('per-example gradients: vmap of grad', () => {
    const g = (xi: Value, wv: Value): Value => grad((v: Value) => f(v, xi))(wv)
    expectClose(vmap(g, [0, null])(xs, w), loop(g, [xs, w], [0, null]))
  })

  it('the gradient of a vectorised function', () => {
    const total = (wv: Value): Value => sum(vmap((xi: Value) => f(wv, xi))(xs))
    const reference = grad((wv: Value) => f(wv, xs))(w)
    expectClose(grad(total)(w), reference, 1e-12)
  })

  it('reverse mode through a vectorised gather with batched indices', () => {
    const labels = tensor([0, 1, 1, 0, 1], { dtype: 'int32' })
    const loss = (wv: Value): Value =>
      sum(vmap((xi: Value, l: Value) => take(matmul(xi, wv), expandDims(l, 0)), [0, 0])(xs, labels))
    const direct = (wv: Value): Value => sum(takeAlongAxis(matmul(xs, wv), reshape(labels, [5, 1]), 1))
    expectClose(grad(loss)(w), grad(direct)(w))
  })

  it('forward mode both ways', () => {
    const t = rand([3, 2], 32)
    const inner = (xi: Value): Value => jvp((v: Value) => f(v, xi), w, t).tangent
    expectClose(vmap(inner)(xs), loop(inner, [xs], [0]))
    const outer = jvp((v: Value) => vmap((xi: Value) => f(v, xi))(xs), w, t).tangent
    expectClose(outer, loop(inner, [xs], [0]))
  })

  it('per-example Hessians and Jacobians', () => {
    const h = (xi: Value): Value => hessian((v: Value) => sum(mul(sin(v), v)))(xi)
    expectClose(vmap(h)(xs), loop(h, [xs], [0]))
    const j = (xi: Value): Value => jacrev((v: Value) => mul(exp(v), v))(xi)
    expectClose(vmap(j)(xs), loop(j, [xs], [0]))
  })

  it('grad of vmap of grad', () => {
    const g = (v: Value): Value => sum(vmap(grad((u: Value) => sum(mul(mul(u, u), v))))(xs))
    const vs = rand([3], 33)
    const want = grad((v: Value) => sum(mul(mul(xs, 2), v)))(vs)
    expectClose(grad(g)(vs), want)
  })

  it('custom derivative rules survive vectorisation', () => {
    const softplus = customVjp({
      name: 'softplusTest',
      forward: (v) => log(add(exp(v), 1)),
      vjp: (gr, [v]) => [mul(gr, sigmoid(v as Value))],
    })
    const g = vmap(grad((v: Value) => sum(softplus(v))))
    expectClose(g(xs), sigmoid(xs))
    const { value, grad: gw } = valueAndGrad((v: Value) => sum(vmap((xi: Value) => softplus(dot(xi, v)))(xs)))(
      rand([3], 34),
    )
    expect(value.shape).toEqual([])
    expect(gw.shape).toEqual([3])
  })

  it('vjp of a vectorised function matches a loop of vjps', () => {
    const fx = (v: Value): Value => mul(sin(v), v)
    const { pullback } = vjp(vmap(fx), xs)
    const ct = rand([5, 3], 35)
    expectClose(
      pullback(ct),
      loop((a: Value, c: Value) => vjp(fx, a).pullback(c), [xs, ct], [0, 0]),
    )
  })
})

describe('nested vmap', () => {
  it('maps over two axes', () => {
    const a = rand([3, 4], 40)
    const b = rand([5, 4], 41)
    const pair = vmap((u: Value) => vmap((v: Value) => dot(u, v))(b))(a)
    expectClose(pair, matmul(a, transpose(b)))
  })

  it('handles closures over the outer batched value', () => {
    const xs = rand([3], 42)
    const ys = rand([4], 43)
    const out = vmap((xv: Value) => vmap((yv: Value) => mul(xv, yv))(ys))(xs)
    expectClose(out, outer(xs, ys))
    const swapped = vmap((xv: Value) => vmap((yv: Value) => sub(xv, yv), 0, 0)(ys), 0, 1)(xs)
    expectClose(swapped, transpose(sub(expandDims(xs, 1), expandDims(ys, 0))))
  })

  it('nests with grad in between', () => {
    const xs = rand([3, 2], 44)
    const ys = rand([4, 2], 45)
    const g = vmap((xv: Value) => vmap((yv: Value) => grad((v: Value) => sum(mul(mul(v, v), yv)))(xv))(ys))(xs)
    const want = mul(mul(expandDims(xs, 1), 2), expandDims(ys, 0))
    expectClose(g, want)
  })
})
