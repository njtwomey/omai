/**
 * The numeric operator: a factor given only by its log density over Gaussian variables, whose EP messages are found by
 * a panelled Gauss-Legendre product rule over the arguments (each with its own incoming message as the weight, then
 * zoomed onto where the tilted mass is). The fallback for the factors with no closed form: a robust likelihood, a
 * non-conjugate link, the clutter problem.
 */

import { OmaiError } from '../../../base/index.ts'
import { allProper, gaussianOf, massOf, scalarOf } from '../../core/args.ts'
import type {
  Arg,
  FactorOperator,
  GaussianBase,
  OperatorContext,
  Params,
  Port,
  TiltedMoments,
  TiltedRule,
} from '../../core/types.ts'
import { quadratureRule } from '../../numerics/numeric-rules.ts'
import { numericExpectedLog, numericMessage, quadraturePorts } from '../kernel/variational-numeric.ts'

/** What {@link numericOperator} builds the operator from. */
export interface NumericOperatorOptions {
  readonly name: string
  readonly description?: string
  /** The number of Gaussian arguments (1 to 3: the grid is a product over them). */
  readonly arity: number
  /** log f at a point, one coordinate per argument, and the factor's parameters (its data), so one operator serves every factor of a kind. */
  readonly logDensity: (x: readonly number[], params: Params) => number
  /** The rule that integrates the factor against the cavities (default: the panelled quadrature); a run's `Projection.numeric` overrides it. */
  readonly rule?: TiltedRule
}

/**
 * An operator for a factor given by its log density over Gaussian variables. EP projects the quadrature moments of the
 * tilted distribution onto a Gaussian (the engine does the projection); the factor's mass under the incoming messages
 * is the same quadrature. Arguments may be observed. The operator supports a power (fractional and power EP, the
 * alpha-divergence projection): the factor raised to `OperatorContext.power` is what is integrated.
 *
 * @param options - The name, arity and log density.
 * @returns The operator, to be put in the registry the graph is compiled against.
 * @throws OmaiError with code `value` for an arity outside 1 to 3.
 * @concept factor-operators
 * @index function An EP operator for any density over Gaussian variables, by quadrature.
 * @example
 * ```ts
 * import { gaussian, gaussianEta, messageArg, numericOperator } from 'omai-engine/message-passing'
 * // exp(-x^2 / 2) as a factor: the tilted distribution of N(0, 1) is N(0, 1/2).
 * const op = numericOperator({ name: 'bell', arity: 1, logDensity: ([x]) => -0.5 * (x as number) ** 2 })
 * const t = op.tilted?.({ mode: 'ep', to: 0, args: [messageArg(gaussian, gaussianEta(0, 1))], params: {}, power: 1 })
 * t?.moments[1]?.toFixed(6) // => '0.500000'
 * ```
 */
export function numericOperator(options: NumericOperatorOptions): FactorOperator {
  const n = options.arity
  if (!(Number.isInteger(n) && n >= 1 && n <= 3))
    throw new OmaiError('value', 'numericOperator: the arity must be 1, 2 or 3')
  const ports: Port[] = Array.from({ length: n }, (_, i) => ({ name: `x${i}`, accepts: ['gaussian'], constant: true }))
  const tilted = (c: OperatorContext): TiltedMoments => {
    const bases = c.args.map(
      (a: Arg): GaussianBase => (a.value === undefined ? gaussianOf(a) : { mean: scalarOf(a), variance: 0 }),
    )
    const rule = c.rule ?? options.rule ?? quadratureRule
    let t: ReturnType<TiltedRule['moments']>
    try {
      t = rule.moments({ bases, to: c.to, logf: (x) => c.power * options.logDensity(x, c.params) })
    } catch (err) {
      if (err instanceof OmaiError && err.code === 'value')
        throw new OmaiError('value', `${options.name}: ${err.message}`)
      throw err
    }
    let logZ = t.logMass
    for (const a of c.args) logZ += massOf(a)
    return { moments: [t.mean, t.variance + t.mean * t.mean], logZ }
  }
  return {
    name: options.name,
    description: options.description ?? `a factor given by its log density over ${n} Gaussian variable(s)`,
    ports,
    support: { ep: 'numeric', vmp: 'numeric' },
    readsTarget: true,
    power: true,
    tilted,
    // The mean field of a factor given by its log density: the non-conjugate message (natural gradient of the expected log).
    vmp: (c) => ({
      message: numericMessage(quadraturePorts(c.args), c.to, (x) => options.logDensity(x, c.params)),
    }),
    expectedLog: (c) => numericExpectedLog(quadraturePorts(c.args), (x) => options.logDensity(x, c.params)),
    logEvidence: (c) => (allProper(c.args) ? tilted({ ...c, to: 0 }).logZ : Number.NaN),
  }
}
