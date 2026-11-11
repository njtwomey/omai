/**
 * The small helpers the atoms of this folder share: the log-gamma of a number, a plain sum, a value computed once, and the
 * normal distribution function in plain arithmetic for a reference grid that calls it millions of times. Not exported from the
 * folder; a helper that more than one file needs lives here and not in a copy.
 */

import { lgamma } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'

/** `log Gamma(x)` of a number. */
export const lg = (x: number): number => item(lgamma(x))

/** The sum of a list of numbers, left to right. */
export const sum = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)

/** A value computed the first time it is asked for (a reference grid that a calibration over many seeds never reads). */
export function once<T>(make: () => T): () => T {
  let value: T | undefined
  return () => (value ??= make())
}

/**
 * The standard normal distribution function in plain arithmetic (Abramowitz and Stegun 26.2.17, error under 1e-7): the fast
 * variant for a grid. Where the accuracy of the library matters, `normalCdf` (the library's `ndtr`) is the one.
 */
export function normalCdfFast(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x))
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
  const tail = Math.exp(-0.5 * x * x) * 0.3989422804014327 * poly
  return x >= 0 ? 1 - tail : tail
}
