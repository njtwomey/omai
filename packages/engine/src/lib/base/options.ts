/**
 * The option shapes and defaults shared across the library, so every loss, metric and test reads them the same way: loss
 * and metric reductions, the alternative hypothesis of a test, and confidence levels.
 */

import { MdmlError } from './errors.ts'

/** How a loss or metric reduces over samples. */
export type Reduction = 'mean' | 'sum' | 'none'

/**
 * The default reduction: `'mean'`.
 *
 * @returns The default.
 * @example
 * ```ts
 * import { defaultReduction } from 'mdml-engine/base'
 * defaultReduction // => 'mean'
 * ```
 */
export const defaultReduction: Reduction = 'mean'

/** The alternative hypothesis of a test. */
export type Alternative = 'two-sided' | 'less' | 'greater'

/**
 * The default alternative: `'two-sided'`.
 *
 * @returns The default.
 * @example
 * ```ts
 * import { defaultAlternative } from 'mdml-engine/base'
 * defaultAlternative // => 'two-sided'
 * ```
 */
export const defaultAlternative: Alternative = 'two-sided'

/** The coverage of a confidence or credible interval, strictly between 0 and 1 (0.95 for a 95% interval). */
export type ConfidenceLevel = number

/**
 * The default confidence level: 0.95.
 *
 * @returns The default.
 * @example
 * ```ts
 * import { defaultConfidenceLevel } from 'mdml-engine/base'
 * defaultConfidenceLevel // => 0.95
 * ```
 */
export const defaultConfidenceLevel: ConfidenceLevel = 0.95

/**
 * A confidence level, defaulted and checked to lie strictly between 0 and 1.
 *
 * @param level - The level, or undefined for the default.
 * @returns The level.
 * @throws MdmlError with code `value` when the level is outside (0, 1).
 * @example
 * ```ts
 * import { resolveConfidenceLevel } from 'mdml-engine/base'
 * resolveConfidenceLevel() // => 0.95
 * resolveConfidenceLevel(0.9) // => 0.9
 * ```
 */
export function resolveConfidenceLevel(level: ConfidenceLevel = defaultConfidenceLevel): ConfidenceLevel {
  if (!(level > 0 && level < 1)) throw new MdmlError('value', `confidence level must lie in (0, 1), got ${level}`)
  return level
}

/**
 * A numeric option read from a record that may hold anything, or be missing: the number when it is one, else the
 * fallback.
 *
 * @param options - The record of options, if any.
 * @param name - The option's name.
 * @param fallback - The value when the option is missing or not a number.
 * @returns The option.
 * @example
 * ```ts
 * import { numberOr } from 'mdml-engine/base'
 * numberOr({ g: 9.8, label: 'earth' }, 'g', 10) // => 9.8
 * numberOr({ g: 'high' }, 'g', 10) // => 10
 * ```
 */
export function numberOr(
  options: Readonly<Record<string, unknown>> | undefined,
  name: string,
  fallback: number,
): number {
  const v = options?.[name]
  return typeof v === 'number' ? v : fallback
}
