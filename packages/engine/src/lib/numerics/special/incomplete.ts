/**
 * Incomplete gamma and beta functions, the CDFs of the gamma and beta distributions: the regularised lower and upper
 * incomplete gamma functions with log forms that stay finite where they underflow, their inverses, and the regularised
 * incomplete beta function with its inverse.
 */

import { add, div, exp, log, log1p, mul, neg, sub, type TensorLike, type Value } from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as inc from './_incomplete.ts'
import { betaincDa, betaincDb, gammaincDa, logGammainccDa, logGammaincDa } from './_partials.ts'
import { lbeta, lgamma } from './gamma.ts'

// Densities written with operations, so their own derivatives are exact.
const gammaDensity = (a: Value, x: Value): Value => exp(sub(sub(mul(sub(a, 1), log(x)), x), lgamma(a)))
const logGammaDensity = (a: Value, x: Value): Value => sub(sub(mul(sub(a, 1), log(x)), x), lgamma(a))
const betaDensity = (a: Value, b: Value, x: Value): Value =>
  exp(sub(add(mul(sub(a, 1), log(x)), mul(sub(b, 1), log1p(neg(x)))), lbeta(a, b)))

const gammaincP = special('gammainc', inc.gammainc, [
  ([a, x]) => gammaincDa(a as Value, x as Value),
  ([a, x]) => gammaDensity(a as Value, x as Value),
])
const gammainccP = special('gammaincc', inc.gammaincc, [
  ([a, x]) => neg(gammaincDa(a as Value, x as Value)),
  ([a, x]) => neg(gammaDensity(a as Value, x as Value)),
])
const logGammaincP = special('logGammainc', inc.logGammainc, [
  ([a, x]) => logGammaincDa(a as Value, x as Value),
  ([a, x], out) => exp(sub(logGammaDensity(a as Value, x as Value), out)),
])
const logGammainccP = special('logGammaincc', inc.logGammaincc, [
  ([a, x]) => logGammainccDa(a as Value, x as Value),
  ([a, x], out) => neg(exp(sub(logGammaDensity(a as Value, x as Value), out))),
])
// Inverses by the implicit function theorem: with F(a, x) = p, dx/dp = 1 / F_x and dx/da = -F_a / F_x.
const gammaincinvP = special('gammaincinv', inc.gammaincinv, [
  ([a], x) => neg(div(gammaincDa(a as Value, x), gammaDensity(a as Value, x))),
  ([a], x) => div(1, gammaDensity(a as Value, x)),
])
const gammainccinvP = special('gammainccinv', inc.gammainccinv, [
  ([a], x) => neg(div(gammaincDa(a as Value, x), gammaDensity(a as Value, x))),
  ([a], x) => neg(div(1, gammaDensity(a as Value, x))),
])
const betaincP = special('betainc', inc.betainc, [
  ([a, b, x]) => betaincDa(a as Value, b as Value, x as Value),
  ([a, b, x]) => betaincDb(a as Value, b as Value, x as Value),
  ([a, b, x]) => betaDensity(a as Value, b as Value, x as Value),
])
const betaincinvP = special('betaincinv', inc.betaincinv, [
  ([a, b], x) => neg(div(betaincDa(a as Value, b as Value, x), betaDensity(a as Value, b as Value, x))),
  ([a, b], x) => neg(div(betaincDb(a as Value, b as Value, x), betaDensity(a as Value, b as Value, x))),
  ([a, b], x) => div(1, betaDensity(a as Value, b as Value, x)),
])

/**
 * The regularised lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a), the CDF of a unit-scale gamma
 * distribution.
 *
 * Differentiable in both arguments; the derivative in the shape `a` is computed by differentiating the series and
 * continued fraction term by term, so it stays accurate in both tails (it drives implicit reparameterisation of gamma
 * samples).
 *
 * @param a - Shape (positive).
 * @param x - Argument (non-negative).
 * @returns P(a, x).
 * @concept special-functions
 * @example
 * ```ts
 * import { gammainc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(gammainc(1, 2)) - (1 - Math.exp(-2))) < 1e-15 // => true
 * ```
 */
export function gammainc(a: TensorLike, x: TensorLike): Value {
  return gammaincP(a, x)
}

/**
 * The regularised upper incomplete gamma function Q(a, x) = 1 - P(a, x), accurate where it is tiny.
 *
 * @param a - Shape (positive).
 * @param x - Argument (non-negative).
 * @returns Q(a, x), differentiable in both arguments.
 * @example
 * ```ts
 * import { gammaincc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(gammaincc(1, 2)) - Math.exp(-2)) < 1e-15 // => true
 * ```
 */
export function gammaincc(a: TensorLike, x: TensorLike): Value {
  return gammainccP(a, x)
}

/**
 * log P(a, x), finite where P underflows.
 *
 * @param a - Shape.
 * @param x - Argument.
 * @returns log P(a, x), differentiable.
 * @example
 * ```ts
 * import { logGammainc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(logGammainc(500, 1)) < -2000 // => true
 * ```
 */
export function logGammainc(a: TensorLike, x: TensorLike): Value {
  return logGammaincP(a, x)
}

/**
 * log Q(a, x), finite where Q underflows.
 *
 * @param a - Shape.
 * @param x - Argument.
 * @returns log Q(a, x), differentiable.
 * @example
 * ```ts
 * import { logGammaincc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(logGammaincc(1, 1000)) // => -1000
 * ```
 */
export function logGammaincc(a: TensorLike, x: TensorLike): Value {
  return logGammainccP(a, x)
}

/**
 * The inverse of P in its second argument: the x with P(a, x) = p.
 *
 * @param a - Shape.
 * @param p - Probability.
 * @returns x, differentiable in a and p (implicit function theorem).
 * @example
 * ```ts
 * import { gammaincinv } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(gammaincinv(1, 0.5)) - Math.LN2) < 1e-15 // => true
 * ```
 */
export function gammaincinv(a: TensorLike, p: TensorLike): Value {
  return gammaincinvP(a, p)
}

/**
 * The inverse of Q in its second argument: the x with Q(a, x) = q, accurate for tiny q.
 *
 * @param a - Shape.
 * @param q - Upper-tail probability.
 * @returns x, differentiable in a and q.
 * @example
 * ```ts
 * import { gammainccinv } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(gammainccinv(1, 1e-100)) - 100 * Math.LN10) < 1e-12 // => true
 * ```
 */
export function gammainccinv(a: TensorLike, q: TensorLike): Value {
  return gammainccinvP(a, q)
}

/**
 * The regularised incomplete beta function I_x(a, b), the CDF of a beta distribution. Differentiable in all three
 * arguments, with shape derivatives carried through the continued fraction.
 *
 * @param a - First shape (positive).
 * @param b - Second shape (positive).
 * @param x - Argument in [0, 1].
 * @returns I_x(a, b).
 * @example
 * ```ts
 * import { betainc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(betainc(2, 2, 0.5)) - 0.5) < 1e-15 // => true
 * ```
 */
export function betainc(a: TensorLike, b: TensorLike, x: TensorLike): Value {
  return betaincP(a, b, x)
}

/**
 * The inverse of I in its last argument: the x with I_x(a, b) = p.
 *
 * @param a - First shape.
 * @param b - Second shape.
 * @param p - Probability.
 * @returns x, differentiable in a, b and p.
 * @example
 * ```ts
 * import { betaincinv } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(betaincinv(2, 2, 0.5)) - 0.5) < 1e-15 // => true
 * ```
 */
export function betaincinv(a: TensorLike, b: TensorLike, p: TensorLike): Value {
  return betaincinvP(a, b, p)
}
