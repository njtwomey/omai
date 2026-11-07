// Taylor-mode propagation rules for the core primitives. A series is [x_0, x_1, ..., x_K]: the primal and the
// normalised Taylor coefficients of x(t) = sum_k x_k t^k (x_k = x^(k)(0) / k!), `null` for a zero coefficient. Each rule
// is the standard recurrence for its function (Griewank and Walther, "Evaluating Derivatives", chapter 13), written
// with operations so that the coefficients are themselves differentiable.
import { MdmlError } from '../base/index.ts'
import {
  type Aval,
  add,
  arccos,
  arccosh,
  arcsin,
  arcsinh,
  arctan,
  arctanh,
  bind1,
  broadcastTo,
  cast,
  concat,
  cos,
  cosh,
  cumsum,
  div,
  exp,
  expm1,
  isTensor,
  log,
  log1p,
  log2,
  log10,
  type MaybeZero,
  matmulPrimitive,
  mul,
  neg,
  type Primitive,
  pad,
  pow,
  reshape,
  scatterAdd,
  segmentSum,
  shapesEqual,
  sigmoid,
  sin,
  sinh,
  slice,
  sqrt,
  square,
  sub,
  sum,
  type Trace,
  type Traced,
  take,
  takeAlongAxis,
  tan,
  tanh,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'

/** A truncated Taylor series: the primal first, then coefficients 1..K (`null` for zero). */
export type Series = readonly MaybeZero[]

/** Output series of a primitive, given its input series, parameters, primal outputs and the order. */
export type JetRule = (series: readonly Series[], params: unknown, outputs: readonly Value[], order: number) => Series[]

const addM = (a: MaybeZero, b: MaybeZero): MaybeZero => (a === null ? b : b === null ? a : add(a, b))
const subM = (a: MaybeZero, b: MaybeZero): MaybeZero => (b === null ? a : a === null ? neg(b) : sub(a, b))
const mulM = (a: MaybeZero, b: MaybeZero): MaybeZero => (a === null || b === null ? null : mul(a, b))
const scaleM = (a: MaybeZero, c: number): MaybeZero => (a === null ? null : mul(a, c))
const at = (s: Series, k: number): MaybeZero => s[k] ?? null
const primal = (s: Series): Value => s[0] as Value

/** Coefficient m of the product of two series. */
function cauchy(a: Series, b: Series, m: number): MaybeZero {
  let acc: MaybeZero = null
  for (let j = 0; j <= m; j++) acc = addM(acc, mulM(at(a, j), at(b, m - j)))
  return acc
}

function mulSeries(a: Series, b: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) y.push(cauchy(a, b, k))
  return y
}

/** The series of y with y' = x' u(y), y(0) = y0: k y_k = sum_{j=1..k} j x_j u_{k-j}, where u_m may use y_0..y_m. */
function ode(x: Series, y0: Value, order: number, u: (y: Series, m: number) => MaybeZero): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    let acc: MaybeZero = null
    for (let j = 1; j <= k; j++) acc = addM(acc, mulM(scaleM(at(x, j), j / k), u(y, k - j)))
    y.push(acc)
  }
  return y
}

/** The series of y with y' h = x', y(0) = y0: k h_0 y_k = k x_k - sum_{j=1..k-1} (k - j) y_{k-j} h_j. */
function quotient(x: Series, h: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    let acc: MaybeZero = at(x, k)
    for (let j = 1; j < k; j++) acc = subM(acc, mulM(scaleM(at(y, k - j), (k - j) / k), at(h, j)))
    y.push(acc === null ? null : div(acc, primal(h)))
  }
  return y
}

/** y = x / z: y_k = (x_k - sum_{j=1..k} z_j y_{k-j}) / z_0. */
function divSeries(x: Series, z: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    let acc: MaybeZero = at(x, k)
    for (let j = 1; j <= k; j++) acc = subM(acc, mulM(at(z, j), at(y, k - j)))
    y.push(acc === null ? null : div(acc, primal(z)))
  }
  return y
}

/** y = sqrt(x): y_k = (x_k - sum_{j=1..k-1} y_j y_{k-j}) / (2 y_0). */
function sqrtSeries(x: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    let acc: MaybeZero = at(x, k)
    for (let j = 1; j < k; j++) acc = subM(acc, mulM(at(y, j), at(y, k - j)))
    y.push(acc === null ? null : div(acc, mul(y0, 2)))
  }
  return y
}

/** y = log(x): y_k = (x_k - (1/k) sum_{j=1..k-1} j y_j x_{k-j}) / x_0. */
function logSeries(x: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    let acc: MaybeZero = at(x, k)
    for (let j = 1; j < k; j++) acc = subM(acc, mulM(scaleM(at(y, j), j / k), at(x, k - j)))
    y.push(acc === null ? null : div(acc, primal(x)))
  }
  return y
}

/** y = log_b(x) from the natural-log series, scaled by 1 / ln b (`y0` is the primal of the base-b logarithm). */
function logBase(x: Series, y0: Value, lnBase: number, order: number): MaybeZero[] {
  const ln = logSeries(x, log(primal(x)), order)
  return ln.map((c, k) => (k === 0 ? y0 : c === null ? null : div(c, lnBase)))
}

/** sin and cos (sign -1), or sinh and cosh (sign 1), together: s' = c x', c' = sign s x'. */
function sinCos(x: Series, s0: Value, c0: Value, order: number, sign: 1 | -1): [MaybeZero[], MaybeZero[]] {
  const s: MaybeZero[] = [s0]
  const c: MaybeZero[] = [c0]
  for (let k = 1; k <= order; k++) {
    let sk: MaybeZero = null
    let ck: MaybeZero = null
    for (let j = 1; j <= k; j++) {
      sk = addM(sk, mulM(scaleM(at(x, j), j / k), at(c, k - j)))
      ck = addM(ck, mulM(scaleM(at(x, j), (sign * j) / k), at(s, k - j)))
    }
    s.push(sk)
    c.push(ck)
  }
  return [s, c]
}

/** The series of c + sign x^2. */
function shiftedSquare(x: Series, c: 1 | -1, sign: 1 | -1, order: number): MaybeZero[] {
  const h: MaybeZero[] = [add(c, mul(square(primal(x)), sign))]
  for (let m = 1; m <= order; m++) h.push(scaleM(cauchy(x, x, m), sign))
  return h
}

/** The series of sqrt(c + sign x^2). */
function rootOf(x: Series, c: 1 | -1, sign: 1 | -1, order: number): MaybeZero[] {
  const inner = shiftedSquare(x, c, sign, order)
  return sqrtSeries(inner, sqrt(primal(inner)), order)
}

/** The series of x^n for a non-negative integer n, by repeated squaring (exact where x_0 = 0). */
function integerPower(x: Series, n: number, y0: Value, order: number): MaybeZero[] {
  let result: Series | null = null
  let base: Series = x
  for (let e = n; e > 0; e = Math.floor(e / 2)) {
    if (e % 2 === 1) result = result === null ? base : mulSeries(result, base, mul(primal(result), primal(base)), order)
    if (e > 1) base = mulSeries(base, base, mul(primal(base), primal(base)), order)
  }
  return [y0, ...Array.from({ length: order }, (_, k) => (result === null ? null : at(result, k + 1)))]
}

/** A concrete scalar non-negative integer exponent, if the exponent is one. */
function smallInteger(b: Value): number | null {
  if (!isTensor(b) || b.data.length !== 1) return null
  const v = b.data[0] ?? Number.NaN
  return Number.isInteger(v) && v >= 0 && v <= 64 ? v : null
}

function powSeries(a: Series, b: Series, y0: Value, order: number): Series {
  if (b.slice(1).every((c) => c === null)) {
    const n = smallInteger(primal(b))
    if (n !== null) return integerPower(a, n, y0, order)
    // x y' = c y x': k x_0 y_k = sum_{j=1..k} (c j - (k - j)) x_j y_{k-j}
    const c = primal(b)
    const y: MaybeZero[] = [y0]
    for (let k = 1; k <= order; k++) {
      let acc: MaybeZero = null
      for (let j = 1; j <= k; j++) acc = addM(acc, mulM(mulM(at(a, j), sub(mul(c, j / k), (k - j) / k)), at(y, k - j)))
      y.push(acc === null ? null : div(acc, primal(a)))
    }
    return y
  }
  // a^b = exp(b log a)
  const la = logSeries(a, log(primal(a)), order)
  const p = mulSeries(b, la, mul(primal(b), primal(la)), order)
  return ode(p, y0, order, (y, m) => at(y, m))
}

/** Broadcast every coefficient to the shape of its primal (coefficients of broadcast inputs keep the input shape). */
export function fitShapes(s: Series): MaybeZero[] {
  const y0 = primal(s)
  return s.map((c, k) => (k === 0 || c === null || shapesEqual(c.shape, y0.shape) ? c : broadcastTo(c, y0.shape)))
}

// Find the primitive an operation binds by running it once on abstract placeholders (most primitives are not
// exported). This runs at module load and touches no shared state.
function primitiveOf(op: (...xs: Value[]) => Value, avals: readonly Aval[]): Primitive<unknown> {
  let found: Primitive<unknown> | null = null
  const trace: Trace = {
    level: Number.MAX_SAFE_INTEGER,
    lift: (x) => make(x),
    process: <P>(p: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      found ??= p as Primitive<unknown>
      return p.abstractEval(inputs, params).map(make)
    },
  }
  function make(a: Aval): Traced {
    return { kind: 'traced', shape: a.shape, dtype: a.dtype, trace }
  }
  op(...avals.map(make))
  if (found === null) throw new MdmlError('value', 'jet: an operation bound no primitive')
  return found
}

const f2: Aval = { shape: [2], dtype: 'float64' }
const f21: Aval = { shape: [2, 1], dtype: 'float64' }
const f22: Aval = { shape: [2, 2], dtype: 'float64' }
const i2: Aval = { shape: [2], dtype: 'int32' }
const i21: Aval = { shape: [2, 1], dtype: 'int32' }
const unary = (op: (x: Value) => Value): Primitive<unknown> => primitiveOf(op, [f2])
const binary = (op: (a: Value, b: Value) => Value): Primitive<unknown> => primitiveOf(op, [f2, f2])

/** Primitives that are linear (or affine) in their inputs jointly: each coefficient maps through the jvp rule. */
export const LINEAR: ReadonlySet<Primitive<unknown>> = new Set([
  binary(add),
  binary(sub),
  unary(neg),
  unary((x) => reshape(x, [2, 1])),
  primitiveOf((x) => transpose(x), [f22]),
  unary((x) => broadcastTo(x, [3, 2])),
  binary((a, b) => concat([a, b])),
  unary((x) => pad(x, [[1, 1]], 2)),
  unary((x) => slice(x, [1])),
  unary((x) => sum(x)),
  unary((x) => cumsum(x)),
  unary((x) => cast(x, 'float32')),
  primitiveOf((x, i) => take(x, i), [f2, i2]),
  primitiveOf((x, i) => takeAlongAxis(x, i, 0), [f21, i21]),
  primitiveOf((x, i) => segmentSum(x, i, 3), [f2, i2]),
  primitiveOf((x, i) => scatterAdd([3, 1], i, x, 0), [f21, i21]),
])

function matmulSeries(a: Series, b: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    let acc: MaybeZero = null
    for (let j = 0; j <= k; j++) {
      const x = at(a, j)
      const z = at(b, k - j)
      if (x !== null && z !== null) acc = addM(acc, bind1(matmulPrimitive, [x, z], null))
    }
    y.push(acc)
  }
  return y
}

function whereSeries(c: Value, a: Series, b: Series, y0: Value, order: number): MaybeZero[] {
  const y: MaybeZero[] = [y0]
  for (let k = 1; k <= order; k++) {
    const ak = at(a, k)
    const bk = at(b, k)
    y.push(ak === null && bk === null ? null : where(c, ak ?? 0, bk ?? 0))
  }
  return y
}

const one =
  (rule: (x: Series, y0: Value, order: number) => Series): JetRule =>
  (series, _p, outputs, order) => [rule(series[0] ?? [], outputs[0] as Value, order)]
const two =
  (rule: (a: Series, b: Series, y0: Value, order: number) => Series): JetRule =>
  (series, _p, outputs, order) => [rule(series[0] ?? [], series[1] ?? [], outputs[0] as Value, order)]

/** Rules for the nonlinear core primitives. */
export const RULES: ReadonlyMap<Primitive<unknown>, JetRule> = new Map<Primitive<unknown>, JetRule>([
  [binary(mul), two(mulSeries)],
  [binary(div), two(divSeries)],
  [binary(pow), two(powSeries)],
  [matmulPrimitive as Primitive<unknown>, two(matmulSeries)],
  [
    primitiveOf((c, a, b) => where(c, a, b), [{ shape: [2], dtype: 'bool' }, f2, f2]),
    (series, _p, outputs, order) => [
      whereSeries(primal(series[0] ?? []), series[1] ?? [], series[2] ?? [], outputs[0] as Value, order),
    ],
  ],
  [unary(exp), one((x, y0, order) => ode(x, y0, order, (y, m) => at(y, m)))],
  [unary(expm1), one((x, y0, order) => ode(x, y0, order, (y, m) => (m === 0 ? add(y0, 1) : at(y, m))))],
  [unary(log), one(logSeries)],
  [unary(log1p), one((x, y0, order) => logSeries([add(primal(x), 1), ...x.slice(1)], y0, order))],
  // log_b x = ln x / ln b: the logarithm's series scaled, with the exact primal.
  [unary(log2), one((x, y0, order) => logBase(x, y0, Math.LN2, order))],
  [unary(log10), one((x, y0, order) => logBase(x, y0, Math.LN10, order))],
  [unary(sqrt), one(sqrtSeries)],
  [unary(sin), one((x, y0, order) => sinCos(x, y0, cos(primal(x)), order, -1)[0])],
  [unary(cos), one((x, y0, order) => sinCos(x, sin(primal(x)), y0, order, -1)[1])],
  [unary(sinh), one((x, y0, order) => sinCos(x, y0, cosh(primal(x)), order, 1)[0])],
  [unary(cosh), one((x, y0, order) => sinCos(x, sinh(primal(x)), y0, order, 1)[1])],
  // tan' = 1 + tan^2, tanh' = 1 - tanh^2, sigmoid' = sigmoid - sigmoid^2 (as y' = x' u)
  [unary(tan), one((x, y0, order) => ode(x, y0, order, (y, m) => (m === 0 ? add(1, square(y0)) : cauchy(y, y, m))))],
  [
    unary(tanh),
    one((x, y0, order) => ode(x, y0, order, (y, m) => (m === 0 ? sub(1, square(y0)) : subM(null, cauchy(y, y, m))))),
  ],
  [unary(sigmoid), one((x, y0, order) => ode(x, y0, order, (y, m) => subM(at(y, m), cauchy(y, y, m))))],
  // y' h = x' with h = 1 + x^2, sqrt(1 - x^2), sqrt(1 + x^2), sqrt(x^2 - 1), 1 - x^2
  [unary(arctan), one((x, y0, order) => quotient(x, shiftedSquare(x, 1, 1, order), y0, order))],
  [unary(arcsin), one((x, y0, order) => quotient(x, rootOf(x, 1, -1, order), y0, order))],
  [
    unary(arccos),
    one((x, y0, order) =>
      quotient(
        x.map((c, k) => (k === 0 || c === null ? c : neg(c))),
        rootOf(x, 1, -1, order),
        y0,
        order,
      ),
    ),
  ],
  [unary(arcsinh), one((x, y0, order) => quotient(x, rootOf(x, 1, 1, order), y0, order))],
  [unary(arccosh), one((x, y0, order) => quotient(x, rootOf(x, -1, 1, order), y0, order))],
  [unary(arctanh), one((x, y0, order) => quotient(x, shiftedSquare(x, 1, -1, order), y0, order))],
])

/** Apply a linear primitive to each coefficient through its jvp rule. */
export function linearSeries<P>(
  primitive: Primitive<P>,
  series: readonly Series[],
  params: P,
  outputs: readonly Value[],
  order: number,
): MaybeZero[][] {
  const rule = primitive.jvp
  if (rule === undefined) throw new MdmlError('not-differentiable', `primitive ${primitive.name} has no jvp rule`)
  const primals = series.map(primal)
  const out: MaybeZero[][] = outputs.map((o) => [o])
  for (let k = 1; k <= order; k++) {
    const ts = series.map((s) => at(s, k))
    const ys = ts.every((t) => t === null) ? outputs.map(() => null) : rule(primals, ts, params, outputs)
    for (let i = 0; i < out.length; i++) out[i]?.push(ys[i] ?? null)
  }
  return out
}
