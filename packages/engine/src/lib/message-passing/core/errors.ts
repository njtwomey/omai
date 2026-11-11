/**
 * The two errors of message passing, told apart because they mean opposite things to the engine. An
 * `ImproperMessageError` is a numerical fact of a run (a message or a quantity has no density at this point of the
 * iteration: a cavity with a negative precision, a tilted distribution with no mass); the engine catches it and repairs,
 * counts it and goes on. A `ParameterError` is a mistake in the model or its configuration (a parameter that is missing,
 * of the wrong type or out of range, a port that cannot be observed, a setting that is not valid); it is never swallowed,
 * so a typo ends the run where it is made instead of looking like a failure of the numbers. Both keep the `value` code of
 * `OmaiError`, so a caller that catches by code is unaffected.
 */

import { OmaiError } from '../../base/index.ts'

/**
 * Thrown by an operator, a family or a numeric rule that cannot form a message because a quantity it needs is improper at
 * this point of the run; the engine counts it and repairs (a flat message, a damped retry, a restart). Only this error is
 * caught by the engine.
 *
 * @index structure The error of a numerical failure the engine repairs (an improper message), not of a misconfiguration.
 */
export class ImproperMessageError extends OmaiError {
  constructor(message: string) {
    super('value', message)
    this.name = 'ImproperMessageError'
  }
}

/**
 * Thrown for a mistake in a model or a setting: a factor parameter that is missing, unknown, of the wrong type or out of
 * range, an argument that cannot be observed, an invalid option. Never caught by the engine, so it reaches the caller (at
 * graph compile time for the parameters of every operator).
 *
 * @index structure The error of a misconfigured graph or setting, never swallowed by the engine.
 */
export class ParameterError extends OmaiError {
  constructor(message: string) {
    super('value', message)
    this.name = 'ParameterError'
  }
}
