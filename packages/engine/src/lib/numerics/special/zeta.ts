/**
 * Zeta functions and exponential-type integrals: the Riemann and Hurwitz zeta functions, the Lambert W function, the
 * exponential integrals E_n and Ei, the sine and cosine integrals, and the Fresnel integrals, matching scipy.special.
 */

import {
  add,
  cos,
  div,
  equal,
  exp,
  mul,
  neg,
  promoteFloat,
  sin,
  square,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as k from './_kernels.ts'
import * as misc from './_misc.ts'

// d/ds by term-by-term differentiation of the Euler-Maclaurin sum (its own derivatives use finite differences).
const hurwitzZetaDs = special('hurwitzZetaDs', k.hurwitzZetaDs, [null, null])
const hurwitzZetaP = special('hurwitzZeta', k.hurwitzZeta, [
  ([s, q]) => hurwitzZetaDs(s as Value, q as Value),
  ([s, q]) => neg(mul(s as Value, hurwitzZeta(add(s as Value, 1), q as Value))),
])
const zetaP = special('zeta', k.zeta, [([s]) => hurwitzZetaDs(s as Value, 1)])
const lambertWP = special('lambertW', k.lambertW, [(_, w) => div(1, mul(exp(w), add(w, 1)))])
const expiP = special('expi', k.expi, [([x]) => div(exp(x as Value), x as Value)])
const expnPs = new Map<number, (x: TensorLike) => Value>()

// sin(x) / x and cos(x) / x with the removable singularity of the first handled.
const sinc = (x: Value): Value => {
  const zero = equal(x, 0)
  const safe = where(zero, 1, x)
  return where(zero, 1, div(sin(safe), safe))
}
const siP = special('si', (x) => misc.sici(x).si, [([x]) => sinc(x as Value)])
const ciP = special('ci', (x) => misc.sici(x).ci, [([x]) => div(cos(x as Value), x as Value)])
const halfPi = (x: Value): Value => mul(square(x), Math.PI / 2)
const fresnelSP = special('fresnelS', (x) => misc.fresnel(x).s, [([x]) => sin(halfPi(x as Value))])
const fresnelCP = special('fresnelC', (x) => misc.fresnel(x).c, [([x]) => cos(halfPi(x as Value))])

/**
 * The Riemann zeta function for s > 1.
 *
 * @param s - Argument.
 * @returns zeta(s), differentiable (the derivative by differentiating the Euler-Maclaurin sum term by term).
 * @concept special-functions
 * @example
 * ```ts
 * import { zeta } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(zeta(2)) - Math.PI ** 2 / 6) < 1e-15 // => true
 * ```
 */
export function zeta(s: TensorLike): Value {
  return zetaP(s)
}

/**
 * The Hurwitz zeta function zeta(s, q) = sum_{k >= 0} (k + q)^{-s}, for s > 1 and q > 0 (Infinity for q a non-positive
 * integer; for s = Infinity the limit: 0 for q > 1, 1 for q = 1, Infinity for q < 1).
 *
 * @param s - Exponent.
 * @param q - Shift.
 * @returns zeta(s, q), differentiable in both arguments.
 * @example
 * ```ts
 * import { hurwitzZeta } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(hurwitzZeta(2, 2)) - (Math.PI ** 2 / 6 - 1)) < 1e-15 // => true
 * ```
 */
export function hurwitzZeta(s: TensorLike, q: TensorLike): Value {
  return hurwitzZetaP(s, q)
}

/**
 * The principal branch of the Lambert W function: the w >= -1 with w exp(w) = x, for x >= -1/e.
 *
 * @param x - Argument.
 * @returns W(x), differentiable (W' = 1 / (exp(W) (1 + W))).
 * @example
 * ```ts
 * import { lambertW } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(lambertW(Math.E)) - 1) < 1e-15 // => true
 * ```
 */
export function lambertW(x: TensorLike): Value {
  return lambertWP(x)
}

/**
 * The generalised exponential integral E_n(x) = int_1^inf exp(-x t) / t^n dt.
 *
 * @param n - Order (a non-negative integer).
 * @param x - Argument.
 * @returns E_n(x), differentiable in x (E_n' = -E_{n-1}).
 * @example
 * ```ts
 * import { expn } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * item(expn(1, 0)) // => Infinity
 * ```
 */
export function expn(n: number, x: TensorLike): Value {
  let f = expnPs.get(n)
  if (f === undefined) {
    const rule =
      n === 0
        ? ([v]: readonly Value[]) => {
            const x = v as Value
            return neg(div(mul(exp(neg(x)), add(x, 1)), square(x)))
          }
        : ([v]: readonly Value[]) => neg(expn(n - 1, v as Value))
    f = special(`expn${n}`, (v) => k.expn(n, v), [rule])
    expnPs.set(n, f)
  }
  return f(x)
}

/**
 * The exponential integral Ei(x) (principal value for x > 0).
 *
 * @param x - Argument.
 * @returns Ei(x), differentiable (Ei' = exp(x) / x).
 * @example
 * ```ts
 * import { expi } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * item(expi(0)) // => -Infinity
 * ```
 */
export function expi(x: TensorLike): Value {
  return expiP(x)
}

/**
 * The sine and cosine integrals Si(x) = int_0^x sin(t) / t dt and Ci(x) = gamma + log x + int_0^x (cos t - 1) / t dt
 * (for negative x, Ci returns the real part, Ci(|x|), as scipy does).
 *
 * @param x - Argument.
 * @returns `[Si(x), Ci(x)]`, both differentiable.
 * @example
 * ```ts
 * import { sici } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * item(sici(0)[0]) // => 0
 * ```
 */
export function sici(x: TensorLike): [Value, Value] {
  const [v] = promoteFloat(x) as [Value]
  return [siP(v), ciP(v)]
}

/**
 * The Fresnel integrals S(x) = int_0^x sin(pi t^2 / 2) dt and C(x) = int_0^x cos(pi t^2 / 2) dt.
 *
 * @param x - Argument.
 * @returns `[S(x), C(x)]` (scipy's order), both differentiable.
 * @example
 * ```ts
 * import { fresnel } from 'omai-engine/numerics/special'
 * import { item } from 'omai-engine/tensor'
 * item(fresnel(Infinity)[1]) // => 0.5
 * ```
 */
export function fresnel(x: TensorLike): [Value, Value] {
  const [v] = promoteFloat(x) as [Value]
  return [fresnelSP(v), fresnelCP(v)]
}
