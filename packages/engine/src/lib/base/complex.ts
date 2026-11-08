/**
 * Arithmetic on complex numbers as `[re, im]` pairs (`Complex`), for the code that works on plain numbers: sums,
 * products, quotients, absolute values, the exponential and the principal logarithm. Tensor-valued complex data is
 * `ComplexValue` in `tensor`; the object form `{ re, im }` of the complex-step derivative is `complexOps` in
 * `numerics/differentiate`, which is built on these.
 */

import type { Complex } from './vectors.ts'

/**
 * The sum of two complex numbers.
 *
 * @param a - The first number.
 * @param b - The second number.
 * @returns `a + b`.
 * @example
 * ```ts
 * import { cadd } from 'omai-engine/base'
 * cadd([1, 2], [3, -5]) // => [4, -3]
 * ```
 */
export const cadd = (a: Complex, b: Complex): Complex => [a[0] + b[0], a[1] + b[1]]

/**
 * The difference of two complex numbers.
 *
 * @param a - The minuend.
 * @param b - The subtrahend.
 * @returns `a - b`.
 * @example
 * ```ts
 * import { csub } from 'omai-engine/base'
 * csub([1, 2], [3, -5]) // => [-2, 7]
 * ```
 */
export const csub = (a: Complex, b: Complex): Complex => [a[0] - b[0], a[1] - b[1]]

/**
 * The product of two complex numbers.
 *
 * @param a - The first factor.
 * @param b - The second factor.
 * @returns `a * b`.
 * @example
 * ```ts
 * import { cmul } from 'omai-engine/base'
 * cmul([1, 2], [3, 4]) // => [-5, 10]
 * ```
 */
export const cmul = (a: Complex, b: Complex): Complex => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]

/**
 * The quotient of two complex numbers (by the textbook formula: no scaling against overflow).
 *
 * @param a - The numerator.
 * @param b - The denominator.
 * @returns `a / b`.
 * @example
 * ```ts
 * import { cdiv } from 'omai-engine/base'
 * cdiv([-5, 10], [3, 4]) // => [1, 2]
 * ```
 */
export const cdiv = (a: Complex, b: Complex): Complex => {
  const d = b[0] * b[0] + b[1] * b[1]
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}

/**
 * The modulus of a complex number.
 *
 * @param a - The number.
 * @returns `|a|`.
 * @example
 * ```ts
 * import { cabs } from 'omai-engine/base'
 * cabs([3, 4]) // => 5
 * ```
 */
export const cabs = (a: Complex): number => Math.hypot(a[0], a[1])

/**
 * The complex exponential.
 *
 * @param a - The exponent.
 * @returns `exp(a)`.
 * @example
 * ```ts
 * import { cexp } from 'omai-engine/base'
 * cexp([0, 0]) // => [1, 0]
 * ```
 */
export const cexp = (a: Complex): Complex => {
  const r = Math.exp(a[0])
  return [r * Math.cos(a[1]), r * Math.sin(a[1])]
}

/**
 * The principal complex logarithm.
 *
 * @param a - A nonzero number.
 * @returns `log |a| + i arg a`, the argument in `(-pi, pi]`.
 * @example
 * ```ts
 * import { clog } from 'omai-engine/base'
 * clog([1, 0]) // => [0, 0]
 * ```
 */
export const clog = (a: Complex): Complex => [Math.log(Math.hypot(a[0], a[1])), Math.atan2(a[1], a[0])]
