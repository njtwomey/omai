/**
 * The one guard the routines of `optim` use where a failure of a linear-algebra call or of a bracket search is part of
 * the method (a singular basis, a Gram matrix that is not positive definite, no bracket along a coordinate) and is
 * answered by a fallback: they catch an error and call `rethrowUnless` first, so only the failures meant are absorbed
 * and anything else (an error of the user's objective, a bug, an exhausted memory) reaches the caller. Private to the
 * domain.
 */

import { type ErrorCode, OmaiError } from '../base/index.ts'

/**
 * Rethrow an error unless it is an `OmaiError` with one of the codes a method recovers from.
 *
 * @param error - What a `catch` clause caught.
 * @param codes - The codes the method recovers from.
 * @throws The error itself, when it is not an `OmaiError` or its code is not among `codes`.
 */
export function rethrowUnless(error: unknown, ...codes: readonly ErrorCode[]): void {
  if (!(error instanceof OmaiError && codes.includes(error.code))) throw error
}
