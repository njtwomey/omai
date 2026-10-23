import { WhiteboxError } from './errors.ts'

/** Convergence tolerances shared by iterative methods. */
export interface Tolerance {
  readonly atol: number
  readonly rtol: number
  readonly maxIter: number
}

/** How a loss or metric reduces over samples. */
export type Reduction = 'mean' | 'sum' | 'none'

/**
 * The default tolerances: `atol` 1e-10, `rtol` 1e-8, `maxIter` 1000.
 *
 * @returns The defaults.
 * @example
 * ```ts
 * import { defaultTolerance } from 'whitebox-engine/base'
 * defaultTolerance.maxIter // => 1000
 * ```
 */
export const defaultTolerance: Tolerance = { atol: 1e-10, rtol: 1e-8, maxIter: 1000 }

/**
 * Fill in missing tolerance fields from the defaults.
 *
 * @param partial - Any subset of the fields.
 * @returns A complete tolerance.
 * @example
 * ```ts
 * import { resolveTolerance } from 'whitebox-engine/base'
 * resolveTolerance({ maxIter: 50 }).rtol // => 1e-8
 * ```
 */
export function resolveTolerance(partial: Partial<Tolerance> = {}): Tolerance {
  return { ...defaultTolerance, ...partial }
}

/**
 * The default reduction: `'mean'`.
 *
 * @returns The default.
 * @example
 * ```ts
 * import { defaultReduction } from 'whitebox-engine/base'
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
 * import { defaultAlternative } from 'whitebox-engine/base'
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
 * import { defaultConfidenceLevel } from 'whitebox-engine/base'
 * defaultConfidenceLevel // => 0.95
 * ```
 */
export const defaultConfidenceLevel: ConfidenceLevel = 0.95

/**
 * A confidence level, defaulted and checked to lie strictly between 0 and 1.
 *
 * @param level - The level, or undefined for the default.
 * @returns The level.
 * @throws WhiteboxError with code `value` when the level is outside (0, 1).
 * @example
 * ```ts
 * import { resolveConfidenceLevel } from 'whitebox-engine/base'
 * resolveConfidenceLevel() // => 0.95
 * resolveConfidenceLevel(0.9) // => 0.9
 * ```
 */
export function resolveConfidenceLevel(level: ConfidenceLevel = defaultConfidenceLevel): ConfidenceLevel {
  if (!(level > 0 && level < 1)) throw new WhiteboxError('value', `confidence level must lie in (0, 1), got ${level}`)
  return level
}
