/**
 * Power and fractional expectation propagation (Minka 2004; Wiegerinck and Heskes 2002) as an algorithm of its own: the
 * engine of `engine.ts` with a power on the factors. A factor with power p is approximated by an exponential-family
 * message whose p-th power stands in for the factor to the power p: the cavity removes only the p-th power of the
 * factor's own message, the tilted distribution is the factor to the power p times that cavity, and the new message is
 * the moment match divided by p. Power 1 is expectation propagation; a power below one makes each update a smaller
 * step, which damps the divergence of EP on difficult models and moves the fixed points toward the covering,
 * mass-seeking end of the alpha-divergence family. The powers are per factor (`powers` by factor id, `power` for the
 * rest), and apply to the factors whose operator supports one (`FactorOperator.power`: the probit and logistic links,
 * `numericOperator`, and a `gate` of closed-form branches over one variable); the others keep power 1. On a model with one Gaussian variable and Gaussian factors every power
 * gives the exact posterior, and the library's `gaussianEp` of `pgm` (Gaussian-latent models, power in (0, 1]) is the
 * reference where the two overlap.
 *
 * @index module Power and fractional EP: per-factor powers on the message-passing engine, as an algorithm and a method.
 */

import { OmaiError } from '../base/index.ts'
import { dict, real } from '../space/index.ts'
import { type Algorithm, type Config, defineMethod, type Method } from '../trace/index.ts'
import { type MpState, messagePassingAlgorithm } from './engine.ts'
import type { CompiledFactor, CompiledGraph } from './graph.ts'
import {
  type EpPath,
  epContract,
  epDefaults,
  type MessagePassingOptions,
  messagePassingOptions,
  messagePassingSettings,
  messagePassingSpace,
} from './named.ts'
import { momentProjection } from './projections.ts'
import { takesPower } from './types.ts'

/** The knobs of power EP: the common ones and the powers. */
export interface PowerEpOptions extends MessagePassingOptions {
  /** The power of every factor whose operator supports one and that `powers` does not name (default 0.5). */
  readonly power?: number
  /** Powers by factor id (`f<i>` or the id the factor was given), which override `power`. */
  readonly powers?: Readonly<Record<string, number>>
}

/** The defaults of power EP: those of EP, with the power one half (fractional EP). */
export const powerEpDefaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>> & { readonly power: number } =
  {
    ...epDefaults,
    power: 0.5,
  }

/**
 * The same graph with the powers of its factors replaced: `powerOf` is asked for every factor, and a factor whose
 * operator does not support a power may only be given 1.
 *
 * @param g - The compiled graph.
 * @param powerOf - The power of a factor.
 * @returns A graph that shares everything with `g` but the powers.
 * @throws OmaiError with code `value` for a power that is not positive and finite, or one asked of an operator that has no power.
 * @concept expectation-propagation
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, withPowers } from 'omai-engine/message-passing'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'bernoulli' }, observed: 1 }],
 *     factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
 *   },
 *   defaultOperators(),
 * )
 * withPowers(g, (f) => (f.operator.name === 'probit' ? 0.5 : 1)).factors.map((f) => f.power) // => [1, 0.5]
 * ```
 */
export function withPowers(g: CompiledGraph, powerOf: (f: CompiledFactor) => number): CompiledGraph {
  return {
    ...g,
    factors: g.factors.map((f) => {
      const power = powerOf(f)
      if (!(power > 0 && Number.isFinite(power)))
        throw new OmaiError('value', `power EP: the power of ${f.id} must be positive, got ${power}`)
      if (power !== 1 && !takesPower(f.operator, f.params))
        throw new OmaiError('value', `power EP: ${f.id} (${f.operator.name}) does not support a power`)
      return power === f.power ? f : { ...f, power }
    }),
  }
}

/**
 * Power expectation propagation (Minka 2004) on a factor graph, with fractional EP (every supporting factor at one
 * power below 1) as its default. The cavity of a factor with power p is the belief divided by the p-th power of the
 * factor's message, the tilted distribution is the factor to the power p times the cavity, and the message is the
 * moment match divided by p; the evidence on the tape is the power-EP energy
 * `sum_a log Z_a / p_a + sum_v (1 - sum_a 1 / p_a) A(belief_v)`. Power 1 everywhere is `epAlgorithm`; every power is
 * exact for Gaussian factors on one Gaussian variable; a power below one steps less and damps EP's oscillation at the
 * price of fixed points that are not EP's. An improper update is repaired as in EP and counted on the tape. The state is
 * the messages, so a run serialises, resumes and has anytime snapshots as every engine run does.
 *
 * @param g - The compiled graph.
 * @param options - The powers, plus schedule, granularity, damping, improper policy, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @throws OmaiError with code `value` for a power that is not positive, or one set on a factor whose operator has none.
 * @concept power-ep
 * @reference Minka, T. (2004). Power EP. Microsoft Research Technical Report MSR-TR-2004-149.
 * @reference Wiegerinck, W. and Heskes, T. (2002). Fractional belief propagation. Advances in Neural Information Processing Systems 15.
 * @index algorithm Power and fractional EP: per-factor powers on the message-passing engine.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, marginalsOf, powerEpAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'bernoulli' }, observed: 1 }],
 *     factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
 *   },
 *   defaultOperators(),
 * )
 * // One probit site at power one half: the posterior mean is 0.5639, where EP (power 1) gives 0.5642.
 * const r = runAlgorithm(powerEpAlgorithm(g, { power: 0.5 }), { maxSteps: 200 })
 * Number(marginalsOf(g, r.state).x?.mean).toFixed(4) // => '0.5639'
 * ```
 */
export function powerEpAlgorithm(g: CompiledGraph, options: PowerEpOptions = {}): Algorithm<MpState, EpPath> {
  const power = options.power ?? powerEpDefaults.power
  const named = options.powers ?? {}
  const ids = new Set(g.factors.map((f) => f.id))
  for (const id of Object.keys(named)) {
    if (!ids.has(id)) throw new OmaiError('value', `power EP: there is no factor with the id ${id}`)
  }
  const powered = withPowers(g, (f) => named[f.id] ?? (takesPower(f.operator, f.params) ? power : 1))
  const alg = messagePassingAlgorithm(powered, messagePassingSettings(momentProjection, epDefaults, options))
  return { ...alg, name: 'power-ep' } as Algorithm<MpState, EpPath>
}

/**
 * Power EP on a graph as a method: the common knobs and the `power` as a space, the tape as {@link epContract}, so a
 * session, a worker or a power slider drives it by configuration (per-factor powers are an option of
 * {@link powerEpAlgorithm}, not of the space).
 *
 * @param g - The compiled graph.
 * @returns The method, named `power-ep`.
 * @concept power-ep
 * @index function The power EP method of a graph: a power knob beside the EP ones, typed tape.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, powerEpMethod } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * Object.keys(powerEpMethod(g).config.spaces).includes('power') // => true
 * ```
 */
export function powerEpMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  const base = messagePassingSpace(powerEpDefaults)
  return defineMethod<MpState, Config, EpPath>({
    name: 'power-ep',
    config: dict({
      ...base.spaces,
      power: real(0.05, 1, {
        default: powerEpDefaults.power,
        description: 'the power of every factor that supports one (1 is EP, below 1 is fractional EP)',
      }),
    }),
    make: (c) => {
      const power = typeof c['power'] === 'number' ? c['power'] : undefined
      return powerEpAlgorithm(g, { ...messagePassingOptions(c), ...(power === undefined ? {} : { power }) })
    },
    contract: epContract,
  })
}
