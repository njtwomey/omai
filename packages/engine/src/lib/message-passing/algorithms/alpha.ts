/**
 * Alpha-divergence message passing (Minka 2005): expectation propagation whose local update minimises the
 * alpha-divergence between the tilted distribution (the factor times its cavity) and the approximation, instead of the
 * Kullback-Leibler divergence KL(p || q) that moment matching minimises. One knob, `alpha`, runs the family: alpha = 1 is
 * expectation propagation (KL(p || q), which covers all of the tilted mass, even where that makes q too wide: it is
 * mass-covering and zero-avoiding), and alpha toward 0 approaches KL(q || p), the variational objective (which sits on
 * one mode of the tilted distribution and underestimates its width: mode-seeking and zero-forcing). In between, alpha
 * = 1/2 is the Hellinger distance. The update of an edge is the fixed point of "moment-match the distribution
 * proportional to f^alpha cavity^alpha q^(1 - alpha)", which is the stationarity condition of D_alpha over an
 * exponential family; the first sweep of that iteration at alpha = 1 is the EP update, so the sweep of alphas
 * interpolates between the two methods the library already has.
 *
 * @index module Alpha-divergence message passing: the alpha slot of the engine, as an algorithm and a method.
 */

import { dict, real } from '../../space/index.ts'
import { type Algorithm, type Config, defineMethod, type Method } from '../../trace/index.ts'
import { ParameterError } from '../core/errors.ts'
import type { CompiledGraph } from '../core/graph.ts'
import { takesPower } from '../core/types.ts'
import type { Projection } from '../engine/projections.ts'
import { messagePassingAlgorithm } from '../engine/run.ts'
import type { MpState } from '../engine/state.ts'
import {
  type EpPath,
  epContract,
  epDefaults,
  type MessagePassingOptions,
  messagePassingOptions,
  messagePassingSettings,
  messagePassingSpace,
} from './named.ts'

/** The knobs of alpha-divergence message passing: the common ones and `alpha`. */
export interface AlphaEpOptions extends MessagePassingOptions {
  /** The alpha of the divergence in (0, 1]: 1 is EP (KL(p || q)), toward 0 the variational direction (KL(q || p)). */
  readonly alpha?: number
}

/** The defaults of alpha-divergence message passing: those of EP with alpha one half (the Hellinger divergence). */
export const alphaEpDefaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>> & { readonly alpha: number } =
  {
    ...epDefaults,
    alpha: 0.5,
  }

/**
 * The projection of an alpha-divergence: the slot of the engine that, for every factor whose operator supports a power,
 * replaces the moment match by the minimiser of D_alpha(f cavity || q) in the family (alpha = 1 is the moment match).
 * Factors without a power keep their own message rule.
 *
 * @param alpha - The alpha in (0, 1].
 * @returns The projection, to be used as `EngineSettings.projection`.
 * @throws OmaiError with code `value` for an alpha outside (0, 1].
 * @concept alpha-divergence
 * @index function The alpha-divergence projection slot: alpha 1 is EP, toward 0 variational.
 * @example
 * ```ts
 * import { alphaProjection } from 'omai-engine/message-passing'
 * alphaProjection(0.5).name // => 'alpha(0.5)'
 * ```
 */
export function alphaProjection(alpha: number): Projection {
  if (!(alpha > 0 && alpha <= 1)) throw new ParameterError(`alpha: alpha must be in (0, 1], got ${alpha}`)
  return {
    name: `alpha(${alpha})`,
    hook: 'ep',
    arguments: 'cavity',
    evidence: 'ep-energy',
    alpha,
    description: `minimise the alpha-divergence with alpha = ${alpha} of the tilted distribution (1: EP, toward 0: variational)`,
  }
}

/**
 * Alpha-divergence message passing (Minka 2005) on a factor graph: every factor whose operator supports a power
 * (`probit`, `logistic`, `numericOperator`) is projected by the alpha-divergence minimiser rather than by moment
 * matching, with the cavity and the other arguments mixed in as `cavity^alpha q^(1 - alpha)`; the other factors (priors,
 * linear Gaussian factors, tables) keep their closed-form rules. At `alpha = 1` the run is `epAlgorithm`. Smaller alpha
 * gives narrower, mode-seeking beliefs (at alpha -> 0 the variational ones), larger alpha wider, mass-covering ones.
 * The tape is EP's, without the evidence (the free energy of an alpha-divergence is not the EP one and is not
 * reported); an update that fails to converge in 200 fixed-point steps keeps its last iterate. The state is the
 * messages, so a run serialises, resumes and has anytime snapshots as every engine run does.
 *
 * @param g - The compiled graph; its factors must all have power 1.
 * @param options - `alpha`, plus schedule, granularity, damping, improper policy, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @throws OmaiError with code `value` for an alpha outside (0, 1], a factor with a power, or a graph none of whose factors supports a power.
 * @concept alpha-divergence
 * @reference Minka, T. (2005). Divergence measures and message passing. Microsoft Research Technical Report MSR-TR-2005-173.
 * @reference Hernandez-Lobato, J. M. et al., Black-box alpha divergence minimization (ICML 2016).
 * @index algorithm Alpha-divergence message passing: one alpha knob from EP (1) toward variational (0).
 * @example
 * ```ts
 * import { alphaEpAlgorithm, compileGraph, defaultOperators, gaussianPrior, marginalsOf } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'bernoulli' }, observed: 1 }],
 *     factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
 *   },
 *   defaultOperators(),
 * )
 * // alpha = 1 is EP (0.5642), a smaller alpha moves the mean a little.
 * const mean = (alpha: number): string => Number(marginalsOf(g, runAlgorithm(alphaEpAlgorithm(g, { alpha }), { maxSteps: 400 }).state).x?.mean).toFixed(4)
 * [mean(1), mean(0.5)] // => ['0.5642', '0.5639']
 * ```
 */
export function alphaEpAlgorithm(g: CompiledGraph, options: AlphaEpOptions = {}): Algorithm<MpState, EpPath> {
  const alpha = options.alpha ?? alphaEpDefaults.alpha
  const projection = alphaProjection(alpha)
  for (const f of g.factors) {
    if (f.power !== 1)
      throw new ParameterError(
        `alpha: ${f.id} has a power of its own; alpha-divergence updates take factors of power 1`,
      )
  }
  if (alpha !== 1 && !g.factors.some((f) => takesPower(f.operator, f.params)))
    throw new ParameterError('alpha: no factor of the graph supports a power, so there is nothing to project by alpha')
  const alg = messagePassingAlgorithm(g, messagePassingSettings(projection, epDefaults, options))
  return { ...alg, name: 'alpha-ep' } as Algorithm<MpState, EpPath>
}

/**
 * Alpha-divergence message passing on a graph as a method: the common knobs and `alpha` as a space, the tape as
 * {@link epContract}, so a session, a worker or an alpha slider drives it by configuration.
 *
 * @param g - The compiled graph.
 * @returns The method, named `alpha-ep`.
 * @concept alpha-divergence
 * @index function The alpha-divergence method of a graph: an alpha knob beside the EP ones, typed tape.
 * @example
 * ```ts
 * import { alphaEpMethod, compileGraph, defaultOperators, gaussianPrior } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * alphaEpMethod(g).name // => 'alpha-ep'
 * ```
 */
export function alphaEpMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  const base = messagePassingSpace(alphaEpDefaults)
  return defineMethod<MpState, Config, EpPath>({
    name: 'alpha-ep',
    config: dict({
      ...base.spaces,
      alpha: real(0.05, 1, {
        default: alphaEpDefaults.alpha,
        description: 'the alpha of the divergence: 1 is EP (mass-covering), toward 0 variational (mode-seeking)',
      }),
    }),
    make: (c) => {
      const alpha = typeof c['alpha'] === 'number' ? c['alpha'] : undefined
      return alphaEpAlgorithm(g, { ...messagePassingOptions(c), ...(alpha === undefined ? {} : { alpha }) })
    },
    contract: epContract,
  })
}
