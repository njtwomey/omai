// Scalar special functions for the message algebra: thin wrappers that take and return plain numbers over the tensor
// special functions of numerics/special (messages are number arrays, so the hot loops stay free of tensors), and the
// inverse digamma that moment matching onto a gamma, beta or Dirichlet needs.

import { logAddExp } from '../../base/index.ts'
import { digamma, lgamma, polygamma } from '../../numerics/special/_kernels.ts'
import { erfcx, logNdtr, ndtr, normalLogPdf } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { ImproperMessageError } from '../core/errors.ts'

/** The standard normal distribution function. */
export const Phi = (x: number): number => item(ndtr(x))

/** log of the standard normal distribution function, accurate in the lower tail. */
export const logPhi = (x: number): number => item(logNdtr(x))

/** The standard normal density. */
export const phi = (x: number): number => Math.exp(item(normalLogPdf(x)))

/** log of the standard normal density. */
export const logPhiDensity = (x: number): number => item(normalLogPdf(x))

/** The ratio phi(z) / Phi(z), stable in the lower tail where both underflow. */
export const millsRatio = (z: number): number => Math.exp(item(normalLogPdf(z)) - item(logNdtr(z)))

/** exp(x^2) erfc(x), without overflow. */
export const erfcxOf = (x: number): number => item(erfcx(x))

// The scalar kernels of the tensor functions (the same numbers, without the tensor round trip: a Dirichlet projection calls
// them in a Newton loop for every message).
export const digammaOf = (x: number): number => digamma(x)
export const trigammaOf = (x: number): number => polygamma(1, x)
export const lgammaOf = (x: number): number => lgamma(x)

/**
 * The inverse of the digamma function: the x > 0 with digamma(x) = y (Minka's starting value, then Newton's method,
 * quadratically convergent).
 *
 * @param y - A real number.
 * @returns x.
 * @throws OmaiError with code `value` for a non-finite y.
 */
export function inverseDigamma(y: number): number {
  if (!Number.isFinite(y)) throw new ImproperMessageError('inverseDigamma: the argument must be finite')
  let x = y >= -2.22 ? Math.exp(y) + 0.5 : -1 / (y + 0.5772156649015329)
  for (let i = 0; i < 12; i++) {
    const step = (digammaOf(x) - y) / trigammaOf(x)
    const next = x - step
    x = next > 0 ? next : x / 2
    if (Math.abs(step) < 1e-14 * Math.abs(x)) break
  }
  return x
}

/** softplus, `log(1 + exp(x))`, on plain numbers without overflow. */
export const softplusOf = (x: number): number => logAddExp(0, x)

/** The log of the logistic function, `-softplus(-x)`, on plain numbers, accurate for very negative `x`. */
export const logSigmoidOf = (x: number): number => -logAddExp(0, -x)
