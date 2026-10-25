/**
 * Elementwise bijectors, applied to each coordinate independently: identity, exponential, softplus, sigmoid, tanh,
 * affine, power and the standard normal CDF, each with a numerically stable log-Jacobian. They map the real line onto
 * the positive reals, the unit interval and other simple supports.
 */

import { logit, logSigmoid, ndtr, ndtri, normalLogPdf } from '../../numerics/special/index.ts'
import {
  abs,
  add,
  arctanh,
  div,
  expm1,
  exp as expOp,
  log,
  mul,
  neg,
  pow,
  primalOf,
  promoteFloat,
  sigmoid as sigmoidOp,
  softplus as softplusOp,
  sub,
  type TensorLike,
  tanh as tanhOp,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { type Bijector, defineBijector } from './bijector.ts'

const zerosLikeValue = (x: Value): Value => mul(x, 0)

/**
 * The identity bijector.
 *
 * @returns The bijector (log-determinant 0).
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.identity().forward(2)) // => 2
 * ```
 */
export function identity(): Bijector {
  return defineBijector({
    name: 'identity',
    sign: 1,
    forward: (x) => x,
    inverse: (y) => y,
    forwardLogDetJacobian: zerosLikeValue,
  })
}

/**
 * The exponential bijector from the real line onto the positive reals: y = exp(x), log |dy/dx| = x.
 *
 * @returns The bijector.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.exp().forwardLogDetJacobian(1.5)) // => 1.5
 * ```
 */
export function exp(): Bijector {
  return defineBijector({
    name: 'exp',
    codomain: { kind: 'positive' },
    sign: 1,
    forward: expOp,
    inverse: log,
    forwardLogDetJacobian: (x) => x,
    inverseLogDetJacobian: (y) => neg(log(y)),
  })
}

/**
 * The softplus bijector onto the positive reals: y = log(1 + exp(x)), inverse x = y + log(-expm1(-y)), log |dy/dx| =
 * log sigmoid(x). Gentler than exp for large inputs.
 *
 * @returns The bijector.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.softplus().forward(0)) === Math.LN2 // => true
 * ```
 */
export function softplus(): Bijector {
  return defineBijector({
    name: 'softplus',
    codomain: { kind: 'positive' },
    sign: 1,
    forward: softplusOp,
    inverse: (y) => add(y, log(neg(expm1(neg(y))))),
    forwardLogDetJacobian: logSigmoid,
  })
}

/**
 * The logistic sigmoid bijector onto (0, 1): y = 1 / (1 + exp(-x)), inverse logit, log |dy/dx| = log sigmoid(x) +
 * log sigmoid(-x).
 *
 * @returns The bijector.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.sigmoid().forward(0)) // => 0.5
 * ```
 */
export function sigmoid(): Bijector {
  return defineBijector({
    name: 'sigmoid',
    codomain: { kind: 'unitInterval' },
    sign: 1,
    forward: sigmoidOp,
    inverse: logit,
    forwardLogDetJacobian: (x) => add(logSigmoid(x), logSigmoid(neg(x))),
  })
}

/**
 * The hyperbolic tangent bijector onto (-1, 1): log |dy/dx| = 2 (log 2 - x - softplus(-2x)), stable for large |x|.
 *
 * @returns The bijector.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.tanh().forwardLogDetJacobian(0)) // => 0
 * ```
 */
export function tanh(): Bijector {
  return defineBijector({
    name: 'tanh',
    codomain: { kind: 'interval', low: promoteFloat(-1)[0] as Value, high: promoteFloat(1)[0] as Value },
    sign: 1,
    forward: tanhOp,
    inverse: arctanh,
    forwardLogDetJacobian: (x) => mul(sub(sub(Math.LN2, x), softplusOp(mul(x, -2))), 2),
  })
}

/** The sign of a concrete scale (0 when mixed or traced without a value). */
function signOf(scale: Value): 1 | -1 | 0 {
  let values: number[]
  try {
    values = toFlat(primalOf(scale))
  } catch {
    return 0
  }
  if (values.every((v) => v > 0)) return 1
  if (values.every((v) => v < 0)) return -1
  return 0
}

/**
 * The affine bijector y = loc + scale x (elementwise, scale nonzero): log |dy/dx| = log |scale|.
 *
 * @param loc - Shift.
 * @param scale - Scale (nonzero).
 * @returns The bijector, differentiable in loc and scale.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.affine(1, 2).forward(3)) // => 7
 * ```
 */
export function affine(loc: TensorLike = 0, scale: TensorLike = 1): Bijector {
  const [shift, s] = promoteFloat(loc, scale) as [Value, Value]
  return defineBijector({
    name: 'affine',
    sign: signOf(s),
    forward: (x) => add(shift, mul(s, x)),
    inverse: (y) => div(sub(y, shift), s),
    forwardLogDetJacobian: (x) => add(log(abs(s)), mul(x, 0)),
    inverseLogDetJacobian: (y) => sub(mul(y, 0), log(abs(s))),
  })
}

/**
 * The power bijector y = x^p on the positive reals (p nonzero): log |dy/dx| = log |p| + (p - 1) log x.
 *
 * @param exponent - The power p.
 * @returns The bijector.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.power(2).inverse(9)) // => 3
 * ```
 */
export function power(exponent: TensorLike): Bijector {
  const [p] = promoteFloat(exponent) as [Value]
  return defineBijector({
    name: 'power',
    domain: { kind: 'positive' },
    codomain: { kind: 'positive' },
    sign: signOf(p),
    forward: (x) => pow(x, p),
    inverse: (y) => pow(y, div(1, p)),
    forwardLogDetJacobian: (x) => add(log(abs(p)), mul(sub(p, 1), log(x))),
  })
}

/**
 * The standard normal CDF as a bijector from the real line onto (0, 1): y = Phi(x), inverse Phi^{-1}, log |dy/dx| =
 * log phi(x) (the Gaussian copula's marginal map).
 *
 * @returns The bijector.
 * @example
 * ```ts
 * import { bijectors } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * item(bijectors.normalCdf().forward(0)) // => 0.5
 * ```
 */
export function normalCdf(): Bijector {
  return defineBijector({
    name: 'normalCdf',
    codomain: { kind: 'unitInterval' },
    sign: 1,
    forward: ndtr,
    inverse: ndtri,
    forwardLogDetJacobian: normalLogPdf,
  })
}
