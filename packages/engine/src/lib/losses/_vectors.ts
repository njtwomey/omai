import { abs, add, div, maximum, pow, sqrt, square, sub, sum, type Value } from '../tensor/index.ts'

/**
 * x / max(||x||_2, eps) along the last axis (torch.nn.functional.normalize). The clamp sits under the square root,
 * sqrt(max(sum x^2, eps^2)), so an all-zero row has a finite gradient (0, and 1 / eps through the numerator): the
 * gradient of the norm at zero is infinite, and a clamp placed above it passes a zero that makes it NaN.
 */
export function l2Normalise(x: Value, eps = 1e-12): Value {
  return div(x, sqrt(maximum(sum(square(x), -1, true), eps * eps)))
}

/** ||a - b + eps||_p along the last axis (torch.nn.functional.pairwise_distance). */
export function pairwiseDistance(a: Value, b: Value, p = 2, eps = 1e-6): Value {
  const d = add(sub(a, b), eps)
  if (p === 2) return sqrt(sum(square(d), -1))
  if (p === 1) return sum(abs(d), -1)
  return pow(sum(pow(abs(d), p), -1), 1 / p)
}
