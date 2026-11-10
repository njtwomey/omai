/**
 * The projection slot of the engine: which message rule of the operators is asked for and what the operators are
 * shown. Moment matching (expectation propagation, KL(p || q)), the expected-log rule (variational message passing,
 * KL(q || p)) and the sum-product rule (belief propagation) are the three that exist now; a variation that is a
 * different projection (an alpha-divergence, a Laplace or quadrature site) is another entry of the same shape.
 */

import type { Mode } from './types.ts'

/**
 * A projection: the rule an operator is asked for (`hook`), what it is shown as the incoming message of each port
 * (`arguments`: the cavity, the belief with the factor's own message divided out; or the belief itself), and the
 * estimate of the log evidence or bound that goes with it.
 *
 * @concept message-passing-projections
 * @index structure The projection slot: moment matching, expected log, sum-product.
 */
export interface Projection {
  readonly name: string
  /** The operator rule: `ep` moment matching, `vmp` expected log factor, `bp` sum-product. */
  readonly hook: Mode
  /** What an operator sees of each port: the cavity (EP, BP) or the belief (VMP). */
  readonly arguments: 'cavity' | 'belief'
  /** The estimate this projection reports: the EP free energy (also the Bethe value for BP) or the variational bound. */
  readonly evidence: 'ep-energy' | 'elbo'
  readonly description: string
  /**
   * The alpha of an alpha-divergence projection (`alphaProjection`): the update minimises D_alpha of the tilted
   * distribution instead of matching its moments (alpha = 1, KL(p || q), is expectation propagation). Absent for the
   * three projections of the table. Only factors whose operator supports a power (`FactorOperator.power`) read it; the
   * others keep their own message rule.
   */
  readonly alpha?: number
}

/** Moment matching: expectation propagation. */
export const momentProjection: Projection = {
  name: 'moment',
  hook: 'ep',
  arguments: 'cavity',
  evidence: 'ep-energy',
  description: 'match the moments of the tilted distribution (minimise KL(p || q)): expectation propagation',
}

/** The expected log factor: variational message passing. */
export const klqpProjection: Projection = {
  name: 'klqp',
  hook: 'vmp',
  arguments: 'belief',
  evidence: 'elbo',
  description: 'the expected log factor under the other variables (minimise KL(q || p)): variational message passing',
}

/** The sum-product rule: belief propagation. */
export const sumProductProjection: Projection = {
  name: 'sum-product',
  hook: 'bp',
  arguments: 'cavity',
  evidence: 'ep-energy',
  description: 'sum the factor against the incoming messages: belief propagation (exact on trees)',
}

/**
 * The projections by name.
 *
 * @concept message-passing-projections
 * @index table The projection slot's options: moment, klqp, sum-product.
 * @example
 * ```ts
 * import { projections } from 'omai-engine/message-passing'
 * Object.keys(projections) // => ['moment', 'klqp', 'sum-product']
 * ```
 */
export const projections: Readonly<Record<string, Projection>> = {
  moment: momentProjection,
  klqp: klqpProjection,
  'sum-product': sumProductProjection,
}
