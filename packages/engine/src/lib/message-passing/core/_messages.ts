/**
 * The small conventions every operator (and the engine) repeats, in one place: reading an argument of the context, the
 * flat message, a site divided by the power of its factor, the evidence that is not a number while a message has no
 * density yet, and the text of a caught error. Internal to message passing.
 */

import { ImproperMessageError, ParameterError } from './errors.ts'
import type { Arg, Eta, OperatorContext } from './types.ts'

/** The argument of port `i` of an operator context. */
export const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg

/** Whether a message carries no information (every natural parameter is zero). */
export const isFlat = (eta: Eta | undefined): boolean => eta?.every((v) => v === 0) === true

/** A site message divided by the power of its factor (the site of the unpowered factor is returned as it is). */
export const perPower = (site: Eta, power: number): Eta => (power === 1 ? site : site.map((v) => v / power))

/** The evidence of an operator rule, `NaN` while a message has no density yet (the flat start of a run); any other error propagates. */
export function evidenceOrNaN(compute: () => number): number {
  try {
    return compute()
  } catch (err) {
    if (err instanceof ImproperMessageError) return Number.NaN
    throw err
  }
}

/**
 * A key for a cache: the text of a value in which every number is written in full (`NaN`, `Infinity` and `-Infinity` stay
 * distinct, which JSON does not do), arrays and objects recursively, so two values have one key exactly when they are the same numbers.
 */
export function keyOf(value: unknown): string {
  if (typeof value === 'number') return String(value)
  if (Array.isArray(value) || value instanceof Float64Array)
    return `[${Array.from(value as ArrayLike<unknown>, keyOf).join(',')}]`
  if (value !== null && typeof value === 'object')
    return `{${Object.entries(value)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([k, v]) => `${JSON.stringify(k)}:${keyOf(v)}`)
      .join(',')}}`
  return JSON.stringify(value) ?? 'undefined'
}

/** The value of `compute` kept in the context's cache under `key` (computed every time when the context has none). A throw is not kept. */
export function memoised<T>(c: OperatorContext, key: string, compute: () => T): T {
  const memo = c.memo
  if (memo === undefined) return compute()
  if (memo.has(key)) return memo.get(key) as T
  const value = compute()
  memo.set(key, value)
  return value
}

/** The text of a caught value: the message of an error, else the value as a string. */
export const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))

/** Refuse a damping outside [0, 1): the message is `who: damping must be in [0, 1), got ...` (code `value`). */
export function assertDamping(damping: number, who: string): void {
  if (!(damping >= 0 && damping < 1)) throw new ParameterError(`${who}: damping must be in [0, 1), got ${damping}`)
}
