/**
 * Numerical projections: EP with the numeric operators of a graph integrated by a rule of your choice. The rules
 * themselves (the panelled quadrature, Gauss-Hermite, unscented, Laplace and importance sampling) are in
 * `numerics/numeric-rules.ts`; this file is the projection that selects one for the engine's `numeric` slot, and the
 * algorithm and method that run EP with it.
 */

import { OmaiError } from '../../base/index.ts'
import { dict, int, real, union } from '../../space/index.ts'
import { type Algorithm, type Config, defineMethod, type Method } from '../../trace/index.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { TiltedRule } from '../core/types.ts'
import { type MpState, messagePassingAlgorithm } from '../engine/engine.ts'
import type { Projection } from '../engine/projections.ts'
import { numericRules, quadratureRule } from '../numerics/numeric-rules.ts'
import {
  type EpPath,
  epContract,
  epDefaults,
  type MessagePassingOptions,
  messagePassingOptions,
  messagePassingSettings,
  messagePassingSpace,
} from './named.ts'

/**
 * The numerical projection: moment matching by EP's rules, with every numeric operator of the graph told to find its
 * tilted moments by the given rule (the projection of the engine's `numeric` slot).
 *
 * @param rule - The rule.
 * @returns The projection.
 * @concept numerical-projections
 * @index function The numerical projection slot: moment matching with a chosen numeric rule.
 * @example
 * ```ts
 * import { laplaceRule, numericProjection } from 'omai-engine/message-passing'
 * numericProjection(laplaceRule).name // => 'numeric(laplace)'
 * ```
 */
export function numericProjection(rule: TiltedRule): Projection {
  return {
    name: `numeric(${rule.name})`,
    hook: 'ep',
    arguments: 'cavity',
    evidence: 'ep-energy',
    numeric: rule,
    description: `moment matching with the numeric operators integrated by the ${rule.name} rule`,
  }
}

/** The knobs of numeric EP: the common ones and the rule with its own knobs. */
export interface NumericEpOptions extends MessagePassingOptions {
  /** The numeric rule (default: the panelled quadrature of the numeric operator). */
  readonly rule?: TiltedRule
}

/**
 * Expectation propagation with numeric factors integrated by a rule of your choice (`quadratureRule`, `gaussHermiteRule`,
 * `laplaceRule`, `sampleRule`, `unscentedRule`): the factors of the graph that have closed forms keep them; the numeric
 * operators (a robust likelihood, a non-conjugate link, the clutter problem) find their tilted moments, and the evidence
 * of the run, by the rule. The tape is EP's; the evidence is the EP energy with the rule's masses (so, with Laplace, the
 * Laplace approximation of each factor's mass). The rules are pure functions, so a run serialises and resumes to the same
 * state and has anytime snapshots as every engine run does.
 *
 * @param g - The compiled graph.
 * @param options - The rule, plus schedule, granularity, damping, improper policy, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @concept numerical-projections
 * @reference Minka, T. (2001). Expectation propagation for approximate Bayesian inference (UAI 2001).
 * @reference Smola, A., Vishwanathan, V. and Eskin, E. (2003). Laplace propagation (NIPS 2003).
 * @index algorithm EP with numeric factors projected by quadrature, Laplace, sample or sigma-point rules.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussHermiteRule, gaussianPrior, marginalsOf, numericEpAlgorithm, numericOperator } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const bell = numericOperator({ name: 'bell', arity: 1, logDensity: ([x]) => -0.5 * (x as number) ** 2 })
 * const g = compileGraph(
 *   { variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1), { operator: 'bell', args: ['x'] }] },
 *   { ...defaultOperators(), bell },
 * )
 * const r = runAlgorithm(numericEpAlgorithm(g, { rule: gaussHermiteRule(16) }), { maxSteps: 50 })
 * Number(marginalsOf(g, r.state).x?.variance).toFixed(4) // => '0.5000'
 * ```
 */
export function numericEpAlgorithm(g: CompiledGraph, options: NumericEpOptions = {}): Algorithm<MpState, EpPath> {
  const rule = options.rule ?? quadratureRule
  const alg = messagePassingAlgorithm(g, messagePassingSettings(numericProjection(rule), epDefaults, options))
  return { ...alg, name: 'numeric-ep' } as Algorithm<MpState, EpPath>
}

/**
 * Numeric EP on a graph as a method: the common knobs and the rule (a tagged union: each rule brings its own knobs) as
 * a space, the tape as {@link epContract}.
 *
 * @param g - The compiled graph.
 * @returns The method, named `numeric-ep`.
 * @concept numerical-projections
 * @index function The numeric EP method of a graph: a rule knob (quadrature, Gauss-Hermite, Laplace, sample, unscented).
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, numericEpMethod } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * numericEpMethod(g).name // => 'numeric-ep'
 * ```
 */
export function numericEpMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  const base = messagePassingSpace(epDefaults)
  return defineMethod<MpState, Config, EpPath>({
    name: 'numeric-ep',
    config: dict({
      ...base.spaces,
      rule: union(
        {
          quadrature: dict({}),
          'gauss-hermite': dict({ points: int(2, 200, { default: 32, description: 'nodes per argument' }) }),
          laplace: dict({}),
          sample: dict({
            samples: int(2, 1_000_000, {
              default: 2048,
              log: true,
              description: 'importance samples (antithetic pairs)',
            }),
            seed: int(0, 1_000_000, { default: 0, description: 'the seed of the fixed draws' }),
          }),
          unscented: dict({ kappa: real(0.1, 10, { default: 2, description: 'the spread of the sigma points' }) }),
        },
        { default: 'quadrature', description: 'how numeric operators find their tilted moments' },
      ),
    }),
    make: (c) => {
      const r = c['rule']
      const knobs: Record<string, number> = {}
      let kind = 'quadrature'
      if (typeof r === 'object' && r !== null && !Array.isArray(r)) {
        for (const [k, v] of Object.entries(r)) {
          if (k === 'kind' && typeof v === 'string') kind = v
          else if (typeof v === 'number') knobs[k] = v
        }
      }
      const make = numericRules[kind]
      if (make === undefined) throw new OmaiError('value', `numeric EP: unknown rule ${kind}`)
      return numericEpAlgorithm(g, { ...messagePassingOptions(c), rule: make(knobs) })
    },
    contract: epContract,
  })
}
