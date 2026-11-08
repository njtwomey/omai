// Relational parameter checks that a one-parameter constraint cannot say (low < high, mode inside the range, a symmetric
// covariance), and the one place that reads a possibly traced parameter as plain numbers.
import { OmaiError } from '../base/index.ts'
import {
  abs,
  asTensor,
  greater,
  less,
  lessEqual,
  logicalOr,
  primalOf,
  sub,
  swapaxes,
  toFlat,
  type Value,
} from '../tensor/index.ts'

/** The concrete values of a parameter, or `undefined` when it has none (a value traced inside `vmap`). */
export function concreteFlat(v: Value): number[] | undefined {
  try {
    return toFlat(primalOf(v))
  } catch {
    return undefined
  }
}

/** Whether any element of a boolean tensor of concrete parameters is true (`undefined`: not concrete, skip). */
function anyTrue(mask: Value): boolean {
  return concreteFlat(mask)?.some((x) => Boolean(x)) === true
}

/**
 * Check that `low < high` elementwise (after broadcasting).
 *
 * @param family - The distribution's name, for the message.
 * @param lowName - The lower parameter's name.
 * @param low - The lower parameter.
 * @param highName - The upper parameter's name.
 * @param high - The upper parameter.
 * @throws OmaiError with code `value`.
 */
export function validateOrdered(family: string, lowName: string, low: Value, highName: string, high: Value): void {
  if (anyTrue(lessEqual(primalOf(high), primalOf(low))))
    throw new OmaiError('value', `${family}: ${lowName} must be below ${highName}`, { parameter: lowName })
}

/**
 * Check that `low <= value <= high` elementwise (after broadcasting).
 *
 * @param family - The distribution's name, for the message.
 * @param name - The checked parameter's name.
 * @param value - The checked parameter.
 * @param bounds - The lower and upper bounds as `[name, value]` pairs.
 * @throws OmaiError with code `value`.
 */
export function validateBetween(
  family: string,
  name: string,
  value: Value,
  [lowName, low]: readonly [string, Value],
  [highName, high]: readonly [string, Value],
): void {
  const outside = logicalOr(less(primalOf(value), primalOf(low)), greater(primalOf(value), primalOf(high)))
  if (anyTrue(outside))
    throw new OmaiError('value', `${family}: ${name} must lie between ${lowName} and ${highName}`, { parameter: name })
}

/**
 * Check that a (batch of) square matrices is symmetric to a relative tolerance of 1e-8 (concrete values only).
 *
 * @param family - The distribution's name, for the message.
 * @param name - The matrix's name.
 * @param m - The matrix `[..., d, d]`.
 * @throws OmaiError with code `value`.
 */
export function validateSymmetric(family: string, name: string, m: Value): void {
  const t = asTensor(primalOf(m))
  const flat = concreteFlat(t)
  if (flat === undefined || t.shape.length < 2) return
  const skew = concreteFlat(abs(sub(t, swapaxes(t, -1, -2)))) ?? []
  const scale = flat.reduce((s, x) => Math.max(s, Math.abs(x)), 1)
  if (skew.some((x) => x > 1e-8 * scale))
    throw new OmaiError('value', `${family}: ${name} must be symmetric`, { parameter: name })
}
