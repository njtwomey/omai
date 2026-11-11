/**
 * The small conventions every operator (and the engine) repeats, in one place: reading an argument of the context, the
 * flat message, a site divided by the power of its factor, the evidence that is not a number while a message has no
 * density yet, and the text of a caught error. Internal to message passing.
 */

import { ImproperMessageError } from './args.ts'
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

/** The text of a caught value: the message of an error, else the value as a string. */
export const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e))
