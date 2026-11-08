/**
 * The library's single error type, `OmaiError`, with machine-readable codes that callers branch on instead of
 * message text, and the guards built on it: `invariant`, `assertFinite` and `assertShape` for checking conditions,
 * numbers and shapes at function boundaries.
 */

/**
 * The category of a {@link OmaiError}: callers branch on this, never on message text. `value` is an invalid argument (a
 * number out of range, too few observations, a bad choice); `shape` dimensions that disagree; `domain` a kernel
 * hyperparameter outside its range (the kernels' contract); `schema` data that does not parse; `not-implemented` a
 * capability the thing truly lacks.
 */
export type ErrorCode =
  | 'shape'
  | 'dtype'
  | 'value'
  | 'domain'
  | 'schema'
  | 'convergence'
  | 'singular'
  | 'not-implemented'
  | 'not-differentiable'

/**
 * The single error type thrown by omai, with a machine-readable code and structured details.
 *
 * @param code - The category of the failure.
 * @param message - A human-readable explanation that names the offending values.
 * @param details - Structured context, such as the shapes involved.
 * @returns The error (use with `throw`).
 * @concept errors
 * @index structure The library's single error type with machine-readable codes.
 * @example
 * ```ts
 * import { OmaiError } from 'omai-engine/base'
 * new OmaiError('shape', 'shapes [2] and [3] do not broadcast').code // => 'shape'
 * ```
 */
export class OmaiError extends Error {
  readonly code: ErrorCode
  readonly details: Readonly<Record<string, unknown>>

  constructor(code: ErrorCode, message: string, details: Readonly<Record<string, unknown>> = {}) {
    super(message)
    this.name = 'OmaiError'
    this.code = code
    this.details = details
  }
}

/**
 * Test whether a value is a {@link OmaiError}, optionally of a given code.
 *
 * @param error - Any caught value.
 * @param code - When given, the code the error must have.
 * @returns `true` for a OmaiError (with that code).
 * @example
 * ```ts
 * import { isOmaiError, OmaiError } from 'omai-engine/base'
 * isOmaiError(new OmaiError('value', 'bad'), 'value') // => true
 * isOmaiError(new Error('plain')) // => false
 * ```
 */
export function isOmaiError(error: unknown, code?: ErrorCode): error is OmaiError {
  return error instanceof OmaiError && (code === undefined || error.code === code)
}

/**
 * Throw a {@link OmaiError} unless a condition holds.
 *
 * @param condition - The condition that must be true.
 * @param code - The error code on failure.
 * @param message - The message on failure.
 * @param details - Structured context attached to the error.
 * @returns Nothing; narrows `condition` to true.
 * @example
 * ```ts
 * import { invariant } from 'omai-engine/base'
 * invariant(1 + 1 === 2, 'value', 'arithmetic is broken') // => undefined
 * ```
 */
export function invariant(
  condition: boolean,
  code: ErrorCode,
  message: string,
  details: Readonly<Record<string, unknown>> = {},
): asserts condition {
  if (!condition) throw new OmaiError(code, message, details)
}

/**
 * Check that a number is finite.
 *
 * @param value - The number.
 * @param name - The name used in the error message.
 * @returns The number, unchanged.
 * @throws OmaiError with code `value` when the number is NaN or infinite.
 * @example
 * ```ts
 * import { assertFinite } from 'omai-engine/base'
 * assertFinite(2.5, 'scale') // => 2.5
 * ```
 */
export function assertFinite(value: number, name: string): number {
  if (!Number.isFinite(value)) throw new OmaiError('value', `${name} must be finite, got ${value}`, { [name]: value })
  return value
}

/**
 * Check that a shape matches an expected pattern, where `-1` in the pattern matches any length.
 *
 * @param shape - The actual shape.
 * @param expected - The expected shape, with `-1` as a wildcard.
 * @param name - The name used in the error message.
 * @returns Nothing.
 * @throws OmaiError with code `shape` on a mismatch.
 * @example
 * ```ts
 * import { assertShape } from 'omai-engine/base'
 * assertShape([3, 2], [-1, 2], 'x') // => undefined
 * ```
 */
export function assertShape(shape: readonly number[], expected: readonly number[], name: string): void {
  const ok = shape.length === expected.length && expected.every((n, i) => n === -1 || n === shape[i])
  if (!ok) {
    throw new OmaiError('shape', `${name} must have shape [${expected.join(', ')}], got [${shape.join(', ')}]`, {
      shape,
      expected,
    })
  }
}
