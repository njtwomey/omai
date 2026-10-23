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
