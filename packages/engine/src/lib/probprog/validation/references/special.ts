/**
 * The scalar functions the reference answers are written with: the log-gamma of a number, a plain sum, the normal density and distribution
 * function (the library's `ndtr`, and a fast approximation for a grid that calls it millions of times).
 */

import { lgamma, ndtr } from '../../../numerics/special/index.ts'
import { item } from '../../../tensor/index.ts'

/** `log Gamma(x)` of a number. */
export const lg = (x: number): number => item(lgamma(x))

/** The sum of a list of numbers, left to right. */
export const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

/**
 * The normal density.
 *
 * @param x - The point.
 * @param mean - The mean.
 * @param sd - The standard deviation.
 * @returns The density.
 * @example
 * ```ts
 * import { normalPdf } from 'omai-engine/probprog/validation'
 * normalPdf(0, 0, 1).toFixed(4) // => '0.3989'
 * ```
 */
export function normalPdf(x: number, mean: number, sd: number): number {
  return Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
}

/**
 * The standard normal distribution function.
 *
 * @param x - The point.
 * @returns P(Z <= x).
 * @example
 * ```ts
 * import { normalCdf } from 'omai-engine/probprog/validation'
 * normalCdf(0) // => 0.5
 * ```
 */
export function normalCdf(x: number): number {
  return item(ndtr(Math.max(-40, Math.min(40, x))))
}

/**
 * The standard normal distribution function in plain arithmetic (Abramowitz and Stegun 26.2.17, error under 1e-7): the fast
 * variant for a grid. Where the accuracy of the library matters, `normalCdf` (the library's `ndtr`) is the one.
 *
 * @param x - The point.
 * @returns P(Z <= x), to about 1e-7.
 * @example
 * ```ts
 * import { normalCdfFast } from 'omai-engine/probprog/validation'
 * normalCdfFast(0).toFixed(4) // => '0.5000'
 * ```
 */
export function normalCdfFast(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x))
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
  const tail = Math.exp(-0.5 * x * x) * 0.3989422804014327 * poly
  return x >= 0 ? 1 - tail : tail
}
