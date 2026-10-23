import { z } from 'zod'

/** A JSON value: what can cross a boundary (files, messages, traces) without loss. */
export type Json = null | boolean | number | string | readonly Json[] | { readonly [key: string]: Json }

/**
 * Schema for any JSON value, used inside larger boundary schemas.
 *
 * @returns Parses any JSON-compatible value.
 * @example
 * ```ts
 * import { JsonSchema } from 'whitebox-engine/base'
 * JsonSchema.parse({ a: [1, 'x', null] }) // => { a: [1, 'x', null] }
 * ```
 */
export const JsonSchema: z.ZodType<Json> = z.lazy(() =>
  z.union([z.null(), z.boolean(), z.number(), z.string(), z.array(JsonSchema), z.record(z.string(), JsonSchema)]),
)

/**
 * Encode a number for JSON, writing non-finite values as the strings `'inf'`, `'-inf'` and `'nan'` (the convention the
 * Python oracle uses too).
 *
 * @param x - A number.
 * @returns The number, or its string encoding when it is not finite.
 * @example
 * ```ts
 * import { encodeNumber } from 'whitebox-engine/base'
 * encodeNumber(-Infinity) // => '-inf'
 * encodeNumber(1.5) // => 1.5
 * ```
 */
export function encodeNumber(x: number): number | string {
  if (Number.isNaN(x)) return 'nan'
  if (x === Infinity) return 'inf'
  if (x === -Infinity) return '-inf'
  return x
}

/**
 * Decode a number written by {@link encodeNumber}.
 *
 * @param x - A number or one of `'inf'`, `'-inf'`, `'nan'`.
 * @returns The number.
 * @example
 * ```ts
 * import { decodeNumber } from 'whitebox-engine/base'
 * decodeNumber('inf') // => Infinity
 * ```
 */
export function decodeNumber(x: number | string): number {
  if (typeof x === 'number') return x
  if (x === 'inf') return Infinity
  if (x === '-inf') return -Infinity
  return Number.NaN
}
