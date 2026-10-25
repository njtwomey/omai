/**
 * The library's single error type, `WhiteboxError`, with machine-readable codes that callers branch on instead of
 * message text, and the guards built on it: `invariant`, `assertFinite` and `assertShape` for checking conditions,
 * numbers and shapes at function boundaries.
 */

/** The category of a {@link WhiteboxError}: callers branch on this, never on message text. */
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
 * The single error type thrown by whitebox-js, with a machine-readable code and structured details.
 *
 * @param code - The category of the failure.
 * @param message - A human-readable explanation that names the offending values.
 * @param details - Structured context, such as the shapes involved.
 * @returns The error (use with `throw`).
 * @concept errors
 * @example
 * ```ts
 * import { WhiteboxError } from 'whitebox-engine/base'
 * new WhiteboxError('shape', 'shapes [2] and [3] do not broadcast').code // => 'shape'
 * ```
 */
export class WhiteboxError extends Error {
  readonly code: ErrorCode
  readonly details: Readonly<Record<string, unknown>>

  constructor(code: ErrorCode, message: string, details: Readonly<Record<string, unknown>> = {}) {
    super(message)
    this.name = 'WhiteboxError'
    this.code = code
    this.details = details
  }
}

/**
 * Test whether a value is a {@link WhiteboxError}, optionally of a given code.
 *
 * @param error - Any caught value.
 * @param code - When given, the code the error must have.
 * @returns `true` for a WhiteboxError (with that code).
 * @example
 * ```ts
 * import { isWhiteboxError, WhiteboxError } from 'whitebox-engine/base'
 * isWhiteboxError(new WhiteboxError('value', 'bad'), 'value') // => true
 * isWhiteboxError(new Error('plain')) // => false
 * ```
 */
export function isWhiteboxError(error: unknown, code?: ErrorCode): error is WhiteboxError {
  return error instanceof WhiteboxError && (code === undefined || error.code === code)
}

/**
 * Throw a {@link WhiteboxError} unless a condition holds.
 *
 * @param condition - The condition that must be true.
 * @param code - The error code on failure.
 * @param message - The message on failure.
 * @param details - Structured context attached to the error.
 * @returns Nothing; narrows `condition` to true.
 * @example
 * ```ts
 * import { invariant } from 'whitebox-engine/base'
 * invariant(1 + 1 === 2, 'value', 'arithmetic is broken') // => undefined
 * ```
 */
export function invariant(
  condition: boolean,
  code: ErrorCode,
  message: string,
  details: Readonly<Record<string, unknown>> = {},
): asserts condition {
  if (!condition) throw new WhiteboxError(code, message, details)
}

/**
 * Check that a number is finite.
 *
 * @param value - The number.
 * @param name - The name used in the error message.
 * @returns The number, unchanged.
 * @throws WhiteboxError with code `value` when the number is NaN or infinite.
 * @example
 * ```ts
 * import { assertFinite } from 'whitebox-engine/base'
 * assertFinite(2.5, 'scale') // => 2.5
 * ```
 */
export function assertFinite(value: number, name: string): number {
  if (!Number.isFinite(value))
    throw new WhiteboxError('value', `${name} must be finite, got ${value}`, { [name]: value })
  return value
}

/**
 * Check that a shape matches an expected pattern, where `-1` in the pattern matches any length.
 *
 * @param shape - The actual shape.
 * @param expected - The expected shape, with `-1` as a wildcard.
 * @param name - The name used in the error message.
 * @returns Nothing.
 * @throws WhiteboxError with code `shape` on a mismatch.
 * @example
 * ```ts
 * import { assertShape } from 'whitebox-engine/base'
 * assertShape([3, 2], [-1, 2], 'x') // => undefined
 * ```
 */
export function assertShape(shape: readonly number[], expected: readonly number[], name: string): void {
  const ok = shape.length === expected.length && expected.every((n, i) => n === -1 || n === shape[i])
  if (!ok) {
    throw new WhiteboxError('shape', `${name} must have shape [${expected.join(', ')}], got [${shape.join(', ')}]`, {
      shape,
      expected,
    })
  }
}
