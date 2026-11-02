/**
 * The gamma function family as differentiable elementwise operations, scipy.special style: gamma, its logarithm and
 * reciprocal, digamma, trigamma and polygamma, the beta function and its logarithm, factorials and binomial
 * coefficients with their logarithms, extended to real arguments.
 */

import { add, mul, neg, sub, type TensorLike, type Value } from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as k from './_kernels.ts'

const gammaP = special('gamma', k.gamma, [([x], out) => mul(out, digamma(x as Value))])
const lgammaP = special('lgamma', k.lgamma, [([x]) => digamma(x as Value)])
const rgammaP = special('rgamma', k.rgamma, [([x], out) => neg(mul(out, digamma(x as Value)))])
const digammaP = special('digamma', k.digamma, [([x]) => polygamma(1, x as Value)])
const polygammaPs = new Map<number, (x: TensorLike) => Value>()
// psi(x + h) - psi(x) as a primitive, accurate when h is small relative to x (the partials of lbeta and lbinom).
const digammaDiffP = special('digammaDiff', k.digammaDiff, [
  ([x, h]) => sub(polygamma(1, add(x as Value, h as Value)), polygamma(1, x as Value)),
  ([x, h]) => polygamma(1, add(x as Value, h as Value)),
])
const lbetaP = special('lbeta', k.lbeta, [
  ([a, b]) => neg(digammaDiffP(a as Value, b as Value)),
  ([a, b]) => neg(digammaDiffP(b as Value, a as Value)),
])
const betaP = special('beta', k.beta, [
  ([a, b], out) => neg(mul(out, digammaDiffP(a as Value, b as Value))),
  ([a, b], out) => neg(mul(out, digammaDiffP(b as Value, a as Value))),
])
// Continuous extensions through the gamma function: d/dn and d/dk of log C(n, k).
const lbinomN = ([n, kk]: readonly Value[]): Value => digammaDiffP(add(sub(n as Value, kk as Value), 1), kk as Value)
const lbinomK = ([n, kk]: readonly Value[]): Value =>
  sub(digamma(add(sub(n as Value, kk as Value), 1)), digamma(add(kk as Value, 1)))
const lbinomP = special('lbinom', k.lbinom, [lbinomN, lbinomK])
const binomP = special('binom', k.binom, [(xs, out) => mul(out, lbinomN(xs)), (xs, out) => mul(out, lbinomK(xs))])
const factorialKernel = (n: number): number => (n < 0 ? 0 : Number.isInteger(n) ? k.factorial(n) : k.gamma(n + 1))
const factorialP = special('factorial', factorialKernel, [([n], out) => mul(out, digamma(add(n as Value, 1)))])
const lfactorialP = special('lfactorial', (n) => k.lgamma(n + 1), [([n]) => digamma(add(n as Value, 1))])

/**
 * The gamma function, with poles at the non-positive integers (NaN there).
 *
 * @param x - Input.
 * @returns Gamma(x), differentiable (Gamma' = Gamma psi).
 * @concept special-functions
 * @index function Gamma function, differentiable; its log, digamma and beta live beside it.
 * @example
 * ```ts
 * import { gamma } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(gamma(5)) // => 24
 * ```
 */
export function gamma(x: TensorLike): Value {
  return gammaP(x)
}

/**
 * log |Gamma(x)|, without overflow.
 *
 * @param x - Input.
 * @returns log |Gamma(x)|, differentiable (derivative psi).
 * @example
 * ```ts
 * import { lgamma } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(lgamma(1)) // => 0
 * ```
 */
export function lgamma(x: TensorLike): Value {
  return lgammaP(x)
}

/**
 * The reciprocal gamma function 1 / Gamma(x), an entire function (zero at the poles of Gamma).
 *
 * @param x - Input.
 * @returns 1 / Gamma(x), differentiable away from the non-positive integers.
 * @example
 * ```ts
 * import { rgamma } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(rgamma(-2)) // => 0
 * ```
 */
export function rgamma(x: TensorLike): Value {
  return rgammaP(x)
}

/**
 * The digamma function psi(x) = d/dx log Gamma(x).
 *
 * @param x - Input.
 * @returns psi(x), differentiable (derivative trigamma).
 * @example
 * ```ts
 * import { digamma } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(digamma(1)) + 0.5772156649015329) < 1e-15 // => true
 * ```
 */
export function digamma(x: TensorLike): Value {
  return digammaP(x)
}

/**
 * The polygamma function psi^(n)(x), the n-th derivative of digamma.
 *
 * @param n - Order (a non-negative integer).
 * @param x - Input.
 * @returns psi^(n)(x), differentiable in x (derivative psi^(n+1)).
 * @example
 * ```ts
 * import { polygamma } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(polygamma(1, 1)) - Math.PI ** 2 / 6) < 1e-15 // => true
 * ```
 */
export function polygamma(n: number, x: TensorLike): Value {
  if (n === 0) return digamma(x)
  let f = polygammaPs.get(n)
  if (f === undefined) {
    f = special(`polygamma${n}`, (v) => k.polygamma(n, v), [([v]) => polygamma(n + 1, v as Value)])
    polygammaPs.set(n, f)
  }
  return f(x)
}

/**
 * The trigamma function psi'(x).
 *
 * @param x - Input.
 * @returns psi'(x), differentiable.
 * @example
 * ```ts
 * import { trigamma } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(trigamma(1)) - Math.PI ** 2 / 6) < 1e-15 // => true
 * ```
 */
export function trigamma(x: TensorLike): Value {
  return polygamma(1, x)
}

/**
 * The beta function B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b).
 *
 * @param a - First argument.
 * @param b - Second argument.
 * @returns B(a, b), differentiable.
 * @example
 * ```ts
 * import { beta } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(beta(1, 1)) // => 1
 * ```
 */
export function beta(a: TensorLike, b: TensorLike): Value {
  return betaP(a, b)
}

/**
 * log |B(a, b)|, accurate when either argument is large.
 *
 * @param a - First argument.
 * @param b - Second argument.
 * @returns log B(a, b), differentiable.
 * @example
 * ```ts
 * import { lbeta } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(lbeta(1, 1)) // => 0
 * ```
 */
export function lbeta(a: TensorLike, b: TensorLike): Value {
  return lbetaP(a, b)
}

/**
 * n! for non-negative integers, Gamma(n + 1) for other non-negative reals, and 0 for negative n (as scipy).
 *
 * @param n - Input.
 * @returns n!, differentiable through Gamma(n + 1).
 * @example
 * ```ts
 * import { factorial } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(factorial(5)) // => 120
 * ```
 */
export function factorial(n: TensorLike): Value {
  return factorialP(n)
}

/**
 * log(n!) = log Gamma(n + 1).
 *
 * @param n - Input.
 * @returns log n!, differentiable.
 * @example
 * ```ts
 * import { lfactorial } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(lfactorial(0)) // => 0
 * ```
 */
export function lfactorial(n: TensorLike): Value {
  return lfactorialP(n)
}

/**
 * The binomial coefficient C(n, k), exact for moderate integers and extended to reals through the gamma function.
 *
 * @param n - Upper argument.
 * @param k - Lower argument.
 * @returns C(n, k), differentiable through the gamma extension.
 * @example
 * ```ts
 * import { binom } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(binom(5, 2)) // => 10
 * ```
 */
export function binom(n: TensorLike, k: TensorLike): Value {
  return binomP(n, k)
}

/**
 * log C(n, k), without overflow.
 *
 * @param n - Upper argument.
 * @param k - Lower argument.
 * @returns log C(n, k), differentiable.
 * @example
 * ```ts
 * import { lbinom } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(lbinom(4, 0)) // => 0
 * ```
 */
export function lbinom(n: TensorLike, k: TensorLike): Value {
  return lbinomP(n, k)
}
