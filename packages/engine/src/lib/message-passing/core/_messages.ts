/**
 * The small conventions every operator (and the engine) repeats, in one place: reading an argument of the context, the
 * flat message, a site divided by the power of its factor, the evidence that is not a number while a message has no
 * density yet, and the text of a caught error. Internal to message passing.
 */

import type { Arg, OperatorContext } from './types.ts'

/** The argument of port `i` of an operator context. */
export const arg = (c: OperatorContext, i: number): Arg => c.args[i] as Arg
