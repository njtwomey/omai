/**
 * Pieces shared by the conditional-variance models: the `arch` backcast of the pre-sample variance and the constant
 * of the Student-t log-density.
 */
import { lgamma } from '../numerics/special/index.ts'
import { item } from '../tensor/index.ts'

/** The pre-sample value of the squared residuals: an exponentially weighted mean of the first 75 (as `arch`). */
export function backcastOf(eps: ArrayLike<number>): number {
  const tau = Math.min(75, eps.length)
  let total = 0
  let s = 0
  for (let i = 0; i < tau; i++) {
    const w = 0.94 ** i
    total += w
    s += w * (eps[i] ?? 0) ** 2
  }
  return s / total
}

/** The constant part of the Student-t log-density: lgamma((nu + 1) / 2) - lgamma(nu / 2) - log(pi (nu - 2)) / 2. */
export function studentConstant(nu: number): number {
  return item(lgamma((nu + 1) / 2)) - item(lgamma(nu / 2)) - 0.5 * Math.log(Math.PI * (nu - 2))
}
