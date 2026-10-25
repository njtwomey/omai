/**
 * Taylor-mode differentiation (like `jax.experimental.jet`): pushing a truncated power series through a function in one
 * pass to get higher derivatives along a direction. Provides the jet trace, `jet`, `taylorCoefficients` and
 * `taylorPolynomial`, the local model behind Taylor-series figures and Taylor methods.
 */

import { MdmlError } from '../base/index.ts'
import {
  add,
  asTensor,
  bind,
  isFloat,
  isTraced,
  type MaybeZero,
  mul,
  type Primitive,
  sizeOf,
  sub,
  type Tensor,
  type TensorLike,
  type Trace,
  type Traced,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { fitShapes, LINEAR, linearSeries, RULES, type Series } from './_jet-rules.ts'
import { withLevel } from './_level.ts'

/**
 * A value inside a Taylor-mode trace: its primal and the normalised Taylor coefficients `x_1..x_K` of a curve through it
 * (`x(t) = x_0 + x_1 t + ... + x_K t^K`), `null` for zero.
 */
export interface JetValue extends Traced {
  readonly primal: Value
  readonly terms: readonly MaybeZero[]
}

/**
 * A Taylor-mode trace (truncated power series) at one level.
 *
 * @concept taylor-mode-autodiff
 */
export interface JetTrace extends Trace {
  readonly kind: 'jet'
  readonly order: number
  jetOf(primal: Value, terms: readonly MaybeZero[]): JetValue
}

// The fallback for primitives without a Taylor rule, through their forward-mode rule: if y(t) = p(x(t)), then
// y'(t) = jvp_p(x(t), x'(t)), so one Taylor pass of order K - 1 through the jvp rule (with x' as the tangent, whose
// coefficients are (m + 1) x_{m+1}) gives k y_k as coefficient k - 1 of y'. The rule is made of operations, which have
// Taylor rules of their own (or recurse to lower order), so the cost stays polynomial in K.
function jvpSeries<P>(
  primitive: Primitive<P>,
  series: readonly Series[],
  params: P,
  outputs: readonly Value[],
  order: number,
): MaybeZero[][] {
  const rule = primitive.jvp
  if (rule === undefined) {
    throw new MdmlError('not-differentiable', `primitive ${primitive.name} has no derivative rule`, {
      primitive: primitive.name,
    })
  }
  const lower = order - 1
  const derivative = withLevel((level) => {
    const trace = jetTrace(level, lower)
    const xs = series.map((s) => trace.jetOf(s[0] as Value, s.slice(1, order)))
    const ts = series.map((s) => {
      const first = s[1] ?? null
      if (s.slice(1).every((c) => c === null)) return null
      const rest = Array.from({ length: lower }, (_, m) => {
        const c = s[m + 2] ?? null
        return c === null ? null : mul(c, m + 2)
      })
      return trace.jetOf(first ?? zeros(s[0]?.shape ?? [], { dtype: (s[0] as Value).dtype }), rest)
    })
    const outs = bind(primitive, xs, params)
    const dys = rule(xs, ts, params, outs)
    return outputs.map((_, i) => {
      const dy = dys[i] ?? null
      if (dy === null) return new Array<MaybeZero>(order).fill(null)
      const ours = isTraced(dy) && dy.trace === trace
      const head = ours ? (dy as JetValue).primal : dy
      const tail = ours ? (dy as JetValue).terms : []
      return [head, ...Array.from({ length: lower }, (_, m) => tail[m] ?? null)]
    })
  })
  return outputs.map((o, i) => {
    const dy = derivative[i] ?? []
    const floating = isFloat(o.dtype)
    return [
      o,
      ...Array.from({ length: order }, (_, m) => {
        const c = dy[m] ?? null
        return c === null || !floating ? null : mul(c, 1 / (m + 1))
      }),
    ]
  })
}

/**
 * Create a Taylor-mode trace. Use {@link jet} or {@link taylorCoefficients} rather than this directly.
 *
 * @param level - The trace's level.
 * @param order - The number of coefficients carried after the primal.
 * @returns The trace.
 * @concept taylor-mode-autodiff
 * @example
 * ```ts
 * import { jetTrace } from 'mdml-engine/autodiff'
 * jetTrace(1, 3).order // => 3
 * ```
 */
export function jetTrace(level: number, order: number): JetTrace {
  const trace: JetTrace = {
    kind: 'jet',
    level,
    order,
    lift: (x) => make(x, []),
    jetOf: (primal, terms) => make(primal, terms),
    process: <P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      const jin = inputs as readonly JetValue[]
      const primals = jin.map((x) => x.primal)
      const outputs = bind(primitive, primals, params)
      const constant = jin.every((x) => x.terms.every((t) => t === null))
      if (constant || primitive.zeroDerivative === true) return outputs.map((o) => make(o, []))
      const series: Series[] = jin.map((x) => [
        x.primal,
        ...Array.from({ length: order }, (_, k) => x.terms[k] ?? null),
      ])
      const key = primitive as Primitive<unknown>
      const rule = RULES.get(key)
      const out = LINEAR.has(key)
        ? linearSeries(primitive, series, params, outputs, order)
        : rule !== undefined
          ? rule(series, params, outputs, order)
          : jvpSeries(primitive, series, params, outputs, order)
      return out.map((s) => {
        const fitted = fitShapes(s)
        return make(fitted[0] as Value, fitted.slice(1))
      })
    },
  }
  function make(primal: Value, terms: readonly MaybeZero[]): JetValue {
    return { kind: 'traced', dtype: primal.dtype, shape: primal.shape, trace, primal, terms }
  }
  return trace
}

/** Propagate normalised coefficients (x_k = x^(k)(0) / k!) through `f`. */
function propagate(
  f: (...xs: Value[]) => Value,
  primals: readonly Value[],
  terms: readonly (readonly MaybeZero[])[],
  order: number,
): { readonly out: Value; readonly terms: readonly Value[] } {
  return withLevel((level) => {
    const trace = jetTrace(level, order)
    const result = f(...primals.map((p, i) => trace.jetOf(p, terms[i] ?? [])))
    const ours = isTraced(result) && result.trace === trace
    const out = ours ? (result as JetValue).primal : result
    const outTerms = Array.from({ length: order }, (_, k) => {
      const t = ours ? ((result as JetValue).terms[k] ?? null) : null
      return t ?? zeros(out.shape, { dtype: out.dtype })
    })
    return { out, terms: outTerms }
  })
}

/**
 * Taylor-mode differentiation (like `jax.experimental.jet`): push a truncated power series through `f` in one pass.
 * If the inputs follow a curve `x(t)` with derivatives `series[i] = [x'(0), x''(0), ..., x^(K)(0)]`, the result is
 * `f(x(0))` and the derivatives of `f(x(t))` at 0 up to order K. Core operations have Taylor rules (exponentials,
 * logarithms, powers, roots, trigonometric and hyperbolic functions and their inverses, products, quotients, matrix
 * products, structural operations); other primitives propagate the series through their forward-mode rule.
 *
 * @param f - A function of several values returning one value.
 * @param primals - The point `x(0)`.
 * @param series - For each input, its derivatives along the curve, orders 1 to K.
 * @returns The output and its derivatives along the curve, orders 1 to K.
 * @throws MdmlError with code `value` when the series have different lengths.
 * @concept taylor-mode-autodiff
 * @example
 * ```ts
 * import { jet } from 'mdml-engine/autodiff'
 * import { sin, tensor, toArray } from 'mdml-engine/tensor'
 * const { out, series } = jet(sin, [tensor(0)], [[tensor(1), tensor(0), tensor(0)]])
 * toArray(out) // => 0
 * series.map((s) => toArray(s)) // => [1, 0, -1]
 * ```
 */
export function jet(
  f: (...xs: Value[]) => Value,
  primals: readonly TensorLike[],
  series: readonly (readonly TensorLike[])[],
): { readonly out: Value; readonly series: readonly Value[] } {
  const order = series[0]?.length ?? 0
  if (series.length !== primals.length || series.some((s) => s.length !== order)) {
    throw new MdmlError('value', 'jet: give one series of the same length per input')
  }
  const ps = primals.map((p) => asTensor(p))
  let factorial = 1
  const scale = Array.from({ length: order }, (_, k) => {
    factorial *= k + 1
    return factorial
  })
  const terms = series.map((s) => s.map((d, k) => mul(asTensor(d), 1 / (scale[k] ?? 1))))
  const { out, terms: outTerms } = propagate(f, ps, terms, order)
  return { out, series: outTerms.map((t, k) => mul(t, scale[k] ?? 1)) }
}

/**
 * The Taylor coefficients of `f` along a direction: `c_k = (1/k!) d^k/dt^k f(x + t v)` at `t = 0`, for `k = 0..order`,
 * so that `f(x + t v) = c_0 + c_1 t + c_2 t^2 + ...`. One Taylor-mode pass gives every order.
 *
 * @param f - A function of one value.
 * @param x - The point.
 * @param v - The direction, with the shape of `x`.
 * @param order - The highest order.
 * @returns The coefficients `[f(x), c_1, ..., c_order]`, each with the shape of `f(x)`.
 * @concept taylor-mode-autodiff
 * @example
 * ```ts
 * import { taylorCoefficients } from 'mdml-engine/autodiff'
 * import { exp, tensor, toArray } from 'mdml-engine/tensor'
 * taylorCoefficients(exp, tensor(0), tensor(1), 3).map((c) => toArray(c)) // => [1, 1, 0.5, 0.16666666666666666]
 * ```
 */
export function taylorCoefficients(f: (x: Value) => Value, x: TensorLike, v: TensorLike, order: number): Value[] {
  const point = asTensor(x)
  const direction = asTensor(v)
  const terms = Array.from({ length: order }, (_, k) => (k === 0 ? direction : null))
  const { out, terms: outTerms } = propagate((y) => f(y), [point], [terms], order)
  return [out, ...outTerms]
}

/** A Taylor polynomial of a scalar function around a centre. */
export interface TaylorPolynomial {
  readonly center: number
  /** `coefficients[k]` multiplies `(x - center)^k`. */
  readonly coefficients: readonly number[]
  /** Evaluate the polynomial elementwise (differentiable in `x`). */
  evaluate(x: TensorLike): Value
}

/**
 * The Taylor polynomial of degree `order` of a scalar function around `center`, from one Taylor-mode pass: the curve
 * behind Taylor-series figures and the local model of Taylor methods.
 *
 * @param f - A function from a scalar to a scalar.
 * @param center - The expansion point.
 * @param order - The degree.
 * @returns The centre, the coefficients and an evaluator.
 * @throws MdmlError with code `shape` when `f` does not return a single value.
 * @concept taylor-series
 * @example
 * ```ts
 * import { taylorPolynomial } from 'mdml-engine/autodiff'
 * import { exp, tensor, toArray } from 'mdml-engine/tensor'
 * const p = taylorPolynomial(exp, 0, 3)
 * p.coefficients // => [1, 1, 0.5, 0.16666666666666666]
 * toArray(p.evaluate(tensor([0, 1]))) // => [1, 2.6666666666666665]
 * ```
 */
export function taylorPolynomial(f: (x: Value) => Value, center: number, order: number): TaylorPolynomial {
  const cs = taylorCoefficients(f, center, 1, order)
  const coefficients = cs.map((c) => {
    if (sizeOf(c.shape) !== 1) throw new MdmlError('shape', 'taylorPolynomial: f must return a single value')
    return toFlat(c as Tensor)[0] ?? Number.NaN
  })
  const evaluate = (x: TensorLike): Value => {
    const h = sub(asTensor(x), center)
    let acc: Value = mul(h, 0)
    for (let k = coefficients.length - 1; k >= 0; k--) acc = add(mul(acc, h), coefficients[k] ?? 0)
    return acc
  }
  return { center, coefficients, evaluate }
}
