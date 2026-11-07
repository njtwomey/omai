import { describe, expect, it } from 'vitest'
import {
  abs,
  add,
  arccos,
  arccosh,
  arcsin,
  arcsinh,
  arctan,
  arctan2,
  arctanh,
  bind1,
  broadcastTo,
  concat,
  cos,
  cosh,
  cumprod,
  cumsum,
  definePrimitive,
  div,
  exp,
  expm1,
  greater,
  log,
  log1p,
  log2,
  log10,
  logsumexp,
  matmul,
  max,
  maximum,
  mul,
  type Primitive,
  pad,
  pow,
  reshape,
  sigmoid,
  sin,
  sinh,
  slice,
  softmax,
  sqrt,
  sub,
  sum,
  type Tensor,
  take,
  tan,
  tanh,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import { jet, taylorCoefficients, taylorPolynomial } from './taylor.ts'
import { grad, jvp } from './transforms.ts'
import { vmap } from './vmap.ts'

/** Normalised Taylor coefficients of f(x + t v) by nested forward mode (the reference). */
function nested(f: (x: Value) => Value, x: Value, v: Value, order: number): Value[] {
  const out: Value[] = [f(x)]
  let g = (t: Value): Value => f(add(x, mul(t, v)))
  let factorial = 1
  for (let k = 1; k <= order; k++) {
    const prev = g
    g = (t: Value): Value => jvp(prev, t, tensor(1)).tangent
    factorial *= k
    out.push(mul(g(tensor(0)), 1 / factorial))
  }
  return out
}

function expectSeries(got: readonly Value[], want: readonly (Value | number)[], tol = 1e-10): void {
  expect(got.length).toBe(want.length)
  got.forEach((g, k) => {
    const w = want[k] as Value | number
    const gs = toFlat(g)
    const ws = typeof w === 'number' ? [w] : toFlat(w)
    expect(gs.length).toBe(ws.length)
    gs.forEach((value, i) => {
      const target = ws[i] ?? Number.NaN
      expect(Math.abs(value - target)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(target)))
    })
  })
}

const scalarSeries = (f: (x: Value) => Value, x: number, order: number): Value[] =>
  taylorCoefficients(f, tensor(x), tensor(1), order)

describe('Taylor coefficients against known series at 0', () => {
  const factorials = [1, 1, 2, 6, 24, 120, 720]
  it('exponential and logarithmic functions', () => {
    expectSeries(
      scalarSeries(exp, 0, 6),
      factorials.map((f) => 1 / f),
    )
    expectSeries(scalarSeries(expm1, 0, 4), [0, 1, 1 / 2, 1 / 6, 1 / 24])
    expectSeries(scalarSeries(log1p, 0, 5), [0, 1, -1 / 2, 1 / 3, -1 / 4, 1 / 5])
    expectSeries(
      scalarSeries((x) => log(add(x, 1)), 0, 5),
      [0, 1, -1 / 2, 1 / 3, -1 / 4, 1 / 5],
    )
    expectSeries(
      scalarSeries((x) => sqrt(add(x, 1)), 0, 4),
      [1, 1 / 2, -1 / 8, 1 / 16, -5 / 128],
    )
    expectSeries(
      scalarSeries((x) => div(1, sub(1, x)), 0, 5),
      [1, 1, 1, 1, 1, 1],
    )
    expectSeries(
      scalarSeries((x) => pow(add(x, 1), 0.5), 0, 4),
      [1, 1 / 2, -1 / 8, 1 / 16, -5 / 128],
    )
  })

  it('trigonometric and hyperbolic functions', () => {
    expectSeries(scalarSeries(sin, 0, 5), [0, 1, 0, -1 / 6, 0, 1 / 120])
    expectSeries(scalarSeries(cos, 0, 4), [1, 0, -1 / 2, 0, 1 / 24])
    expectSeries(scalarSeries(tan, 0, 5), [0, 1, 0, 1 / 3, 0, 2 / 15])
    expectSeries(scalarSeries(sinh, 0, 5), [0, 1, 0, 1 / 6, 0, 1 / 120])
    expectSeries(scalarSeries(cosh, 0, 4), [1, 0, 1 / 2, 0, 1 / 24])
    expectSeries(scalarSeries(tanh, 0, 5), [0, 1, 0, -1 / 3, 0, 2 / 15])
    expectSeries(scalarSeries(sigmoid, 0, 3), [1 / 2, 1 / 4, 0, -1 / 48])
  })

  it('inverse functions', () => {
    expectSeries(scalarSeries(arctan, 0, 5), [0, 1, 0, -1 / 3, 0, 1 / 5])
    expectSeries(scalarSeries(arcsin, 0, 5), [0, 1, 0, 1 / 6, 0, 3 / 40])
    expectSeries(scalarSeries(arccos, 0, 3), [Math.PI / 2, -1, 0, -1 / 6])
    expectSeries(scalarSeries(arcsinh, 0, 5), [0, 1, 0, -1 / 6, 0, 3 / 40])
    expectSeries(scalarSeries(arctanh, 0, 5), [0, 1, 0, 1 / 3, 0, 1 / 5])
  })

  it('integer powers are exact at zero', () => {
    expectSeries(
      scalarSeries((x) => pow(x, 3), 0, 4),
      [0, 0, 0, 1, 0],
    )
    expectSeries(
      scalarSeries((x) => pow(x, 0), 0, 2),
      [1, 0, 0],
    )
    expectSeries(
      scalarSeries((x) => mul(x, x), 0, 3),
      [0, 0, 1, 0],
    )
  })
})

describe('Taylor coefficients against nested forward mode', () => {
  const x = tensor([0.3, -0.8, 1.4])
  const v = tensor([1, 0.5, -2])
  const cases: [string, (x: Value) => Value][] = [
    [
      'elementwise chain',
      (u) => add(mul(exp(sin(u)), log(add(mul(u, u), 1))), add(log2(add(mul(u, u), 1)), log10(add(mul(u, u), 2)))),
    ],
    ['powers', (u) => add(pow(add(abs(u), 0.5), 2.5), pow(2, u))],
    ['inverse functions', (u) => add(arccosh(add(mul(u, u), 1.5)), arctan(u))],
    ['quotients', (u) => div(sin(u), add(cosh(u), 1))],
    ['reductions and softmax', (u) => add(softmax(u), logsumexp(u))],
    ['structure', (u) => sum(mul(reshape(concat([u, pad(slice(u, [1]), [[1, 0]])]), [2, 3]), 2), 0)],
    ['gathers and broadcasting', (u) => mul(take(u, [2, 0, 0, 1]), sum(broadcastTo(u, [2, 3])))],
    [
      'matrix products',
      (u) => matmul(tanh(reshape(concat([u, u]), [2, 3])), transpose(reshape(concat([u, u]), [2, 3]))),
    ],
    ['selection', (u) => where(greater(u, 0), exp(u), mul(u, u))],
    ['fallback: atan2, max, cumprod', (u) => add(arctan2(u, add(mul(u, u), 1)), mul(max(u), cumprod(u)))],
    ['fallback: maximum and cumsum', (u) => cumsum(maximum(sin(u), mul(u, 0.2)))],
  ]
  for (const [name, f] of cases) {
    it(name, () => {
      expectSeries(taylorCoefficients(f, x, v, 4), nested(f, x, v, 4), 1e-9)
    })
  }

  it('propagates through the jvp rule of primitives without a Taylor rule, to high order', () => {
    // A primitive whose derivative is itself: its Taylor fallback recurses through its own jvp rule.
    const myExp: Primitive<null> = definePrimitive<null>({
      name: 'myExp',
      abstractEval: ([u]) => [u as NonNullable<typeof u>],
      impl: ([u]) => [(u as Tensor).backend.map(u as Tensor, Math.exp, 'float64')],
      jvp: ([u], [t]) => [t === null || t === undefined ? null : mul(t, bind1(myExp, [u as Value], null))],
    })
    const want = [1]
    for (let k = 1; k <= 10; k++) want.push((want[k - 1] as number) / k)
    expectSeries(
      scalarSeries((u) => bind1(myExp, [u], null), 0, 10),
      want,
      1e-12,
    )
  })

  it('falls back for primitives without a Taylor rule', () => {
    const cube = definePrimitive<null>({
      name: 'cubeTaylor',
      abstractEval: ([u]) => [u as NonNullable<typeof u>],
      impl: ([u]) => [(u as Tensor).backend.map(u as Tensor, (z) => z * z * z, 'float64')],
      jvp: ([u], [t]) => [t === null || t === undefined ? null : mul(t, mul(3, mul(u as Value, u as Value)))],
      vjp: ([g], [u]) => [g === null || g === undefined ? null : mul(g, mul(3, mul(u as Value, u as Value)))],
    })
    const f = (u: Value): Value => sin(bind1(cube, [u], null))
    expectSeries(taylorCoefficients(f, x, v, 4), nested(f, x, v, 4), 1e-9)
  })
})

describe('jet', () => {
  it('uses derivative coefficients (the jax.experimental.jet convention)', () => {
    const { out, series } = jet(exp, [tensor(0)], [[tensor(1), tensor(0), tensor(0)]])
    expect(toArray(out)).toBe(1)
    expectSeries(series, [1, 1, 1])
  })

  it('propagates curves with higher-order input terms through several inputs', () => {
    // x(t) = 0.5 + t + t^2 (x' = 1, x'' = 2), y(t) = 2 - 3t (y' = -3)
    const f = (a: Value, b: Value): Value => mul(sin(a), exp(mul(a, b)))
    const { series } = jet(
      f,
      [0.5, 2],
      [
        [1, 2, 0],
        [-3, 0, 0],
      ],
    )
    const g = (t: Value): Value => f(add(add(0.5, t), mul(t, t)), sub(2, mul(t, 3)))
    const d1 = (t: Value): Value => jvp(g, t, tensor(1)).tangent
    const d2 = (t: Value): Value => jvp(d1, t, tensor(1)).tangent
    const d3 = (t: Value): Value => jvp(d2, t, tensor(1)).tangent
    expectSeries(series, [d1(tensor(0)), d2(tensor(0)), d3(tensor(0))], 1e-10)
  })

  it('rejects series of different lengths', () => {
    expect(() => jet(add, [1, 2], [[1], [1, 2]])).toThrow(/same length/)
  })
})

describe('Taylor mode composes with the other transforms', () => {
  it('gradients of Taylor coefficients', () => {
    // c_2(x) = f''(x) / 2 for f = sin, so d c_2 / dx = -cos(x) / 2
    const c2 = (x: Value): Value => taylorCoefficients(sin, x, tensor(1), 2)[2] as Value
    expect(toArray(grad(c2)(tensor(0.7)))).toBeCloseTo(-Math.cos(0.7) / 2, 12)
  })

  it('vectorised Taylor coefficients', () => {
    const xs = tensor([0.1, 0.5, 0.9])
    const c3 = vmap((x: Value) => taylorCoefficients(tanh, x, tensor(1), 3)[3] as Value)(xs)
    const want = toFlat(xs).map((x) => {
      const t = Math.tanh(x)
      // d^3/dx^3 tanh = -2 (1 - t^2) (1 - 3 t^2)
      return (-2 * (1 - t * t) * (1 - 3 * t * t)) / 6
    })
    expectSeries([c3], [tensor(want)], 1e-12)
  })
})

describe('taylorPolynomial', () => {
  it('approximates the function near the centre', () => {
    const p = taylorPolynomial(exp, 1, 8)
    expect(p.coefficients[0]).toBeCloseTo(Math.E, 14)
    expect(p.coefficients[3]).toBeCloseTo(Math.E / 6, 14)
    const at = toFlat(p.evaluate(tensor([0.9, 1, 1.2])))
    ;[0.9, 1, 1.2].forEach((x, i) => {
      expect(at[i]).toBeCloseTo(Math.exp(x), 8)
    })
  })

  it('rejects functions with several outputs', () => {
    expect(() => taylorPolynomial((x) => concat([reshape(x, [1]), reshape(x, [1])]), 0, 2)).toThrow(/single value/)
  })
})
