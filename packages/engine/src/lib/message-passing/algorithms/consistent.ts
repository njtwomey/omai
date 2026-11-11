/**
 * Expectation consistent (EC) approximate inference (Opper and Winther 2005; its Ising form is the adaptive TAP
 * approximation of Opper and Winther 2001) for a model that is a Gaussian times a product of one-dimensional sites,
 * `p(x) proportional to f_q(x) prod_i f_i(x_i)`: a Gaussian coupling `f_q` of any covariance (the dense part, kept exact)
 * and non-Gaussian single-variable factors (a probit or logistic label, a spin pinned to plus or minus one). EC
 * approximates the sites by Gaussians that are diagonal, chosen so that the Gaussian `q ~ f_q prod_i (site_i)`, the
 * factorised `r_i ~ f_i (cavity_i)` and the diagonal Gaussian `s` they are glued to agree on their first and second moments
 * (the expectation consistency), and estimates `log Z = log Z_q + sum_i (log Z_ri - log Z_si)`. Its fixed points are those
 * of expectation propagation with the same sites (Minka 2001, and Opper and Winther 2005): this is EP on a graph with one
 * Gaussian vector and one unit-vector row per site, with the parallel (flooding) schedule of the EC iteration, which is
 * what `ecAlgorithm` runs. It is a bespoke algorithm with its own knobs, not a switch: the schedule is parallel by
 * default and damped, the evidence on the tape is the EC estimate, and `expectationConsistentGraph` builds the graph of a
 * Gaussian-times-sites model.
 *
 * @index module Expectation consistent inference (adaptive TAP): a Gaussian times one-dimensional sites, as EP with parallel updates.
 */

import { OmaiError } from '../base/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { type Algorithm, type Config, defineMethod, type Method } from '../trace/index.ts'
import { type MpState, messagePassingAlgorithm } from './engine.ts'
import { type CompiledGraph, compileGraph } from './graph.ts'
import {
  type EpPath,
  epContract,
  epDefaults,
  type MessagePassingOptions,
  messagePassingOptions,
  messagePassingSettings,
  messagePassingSpace,
} from './named.ts'
import { defaultOperators } from './operators/registry.ts'
import { momentProjection } from './projections.ts'
import type { FactorSpec, VariableSpec } from './types.ts'

/** A one-dimensional site of an expectation consistent model. */
export type EcSite =
  | { readonly kind: 'probit'; readonly label: 0 | 1; readonly noise?: number }
  | { readonly kind: 'logistic'; readonly label: 0 | 1 }
  | { readonly kind: 'spin'; readonly field?: number }

/** A Gaussian times sites: the Gaussian's precision and shift (`exp(-x'Px/2 + h'x)`, row-major) and a site for each coordinate. */
export interface EcModel {
  /** The precision matrix P of the Gaussian part, row-major d by d (positive definite). */
  readonly precision: readonly number[]
  /** The shift h of the Gaussian part (default zero). */
  readonly shift?: readonly number[]
  /** The sites, one for each coordinate. */
  readonly sites: readonly EcSite[]
}

/**
 * The factor graph of a Gaussian-times-sites model: one Gaussian vector `x` with the Gaussian part as its prior and one
 * unit-vector row per site (`probit-row`, `logistic-row` or `spin-row` with `a = e_i`), the form expectation consistent
 * inference runs on.
 *
 * @param model - The Gaussian part and the sites.
 * @param operators - The operator registry (default: the library's).
 * @returns The compiled graph, with the vector named `x`.
 * @throws OmaiError with code `shape` for a precision or shift of the wrong size, `value` for a precision that is not positive definite.
 * @concept expectation-consistent
 * @index function The factor graph of a Gaussian part times one-dimensional sites, for EC inference.
 * @example
 * ```ts
 * import { expectationConsistentGraph } from 'omai-engine/message-passing'
 * const g = expectationConsistentGraph({ precision: [2, -0.5, -0.5, 2], sites: [{ kind: 'spin' }, { kind: 'spin', field: 0.3 }] })
 * g.factors.map((f) => f.operator.name) // => ['prior', 'spin-row', 'spin-row']
 * ```
 */
export function expectationConsistentGraph(
  model: EcModel,
  operators: Parameters<typeof compileGraph>[1] = defaultOperators(),
): CompiledGraph {
  const d = model.sites.length
  if (model.precision.length !== d * d)
    throw new OmaiError('shape', `expectation consistent: the precision needs ${d * d} entries`)
  if (model.shift !== undefined && model.shift.length !== d)
    throw new OmaiError('shape', `expectation consistent: the shift needs ${d} entries`)
  const variables: VariableSpec[] = [{ name: 'x', family: { kind: 'gaussian-mv', size: d } }]
  const factors: FactorSpec[] = [
    {
      operator: 'prior',
      args: ['x'],
      params: { eta: [...(model.shift ?? new Array<number>(d).fill(0)), ...model.precision] },
    },
  ]
  for (const [i, site] of model.sites.entries()) {
    const a = Array.from({ length: d }, (_, k) => (k === i ? 1 : 0))
    if (site.kind === 'spin') {
      factors.push({ operator: 'spin-row', args: ['x'], params: { a, field: site.field ?? 0 } })
    } else {
      variables.push({ name: `y${i}`, family: { kind: 'bernoulli' }, observed: site.label })
      factors.push({
        operator: site.kind === 'probit' ? 'probit-row' : 'logistic-row',
        args: [`y${i}`, 'x'],
        params: site.kind === 'probit' && site.noise !== undefined ? { a, noise: site.noise } : { a },
      })
    }
  }
  return compileGraph({ variables, factors }, operators)
}

/** The knobs of expectation consistent inference. */
export type EcOptions = MessagePassingOptions

/** The defaults of expectation consistent inference: parallel (flooding) updates, damped. */
export const ecDefaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>> = {
  ...epDefaults,
  schedule: 'flooding',
  damping: 0.3,
  maxSweeps: 500,
}

/**
 * Expectation consistent inference on a Gaussian-times-sites graph (see `expectationConsistentGraph`): the moment matching
 * of every site against its cavity, all sites at once from the current Gaussian, damped. The tape is EP's: the
 * evidence is the EC estimate `log Z_q + sum_i (log Z_ri - log Z_si)` (the EP energy at the same fixed point), the
 * marginal of `x` is the Gaussian `q` with its full covariance, and the improper counts say how often a site's update would
 * have made `q` improper (the usual failure of the parallel iteration, repaired by the improper policy). A run serialises,
 * resumes to the same state and has anytime snapshots as every engine run does.
 *
 * @param g - The compiled graph of a Gaussian-times-sites model.
 * @param options - Schedule (default parallel), granularity, damping, improper policy, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @concept expectation-consistent
 * @reference Opper, M. and Winther, O. (2005). Expectation consistent approximate inference. Journal of Machine Learning Research 6.
 * @reference Opper, M. and Winther, O. (2001). Adaptive and self-averaging Thouless-Anderson-Palmer mean-field theory for probabilistic modeling. Physical Review E 64.
 * @index algorithm Expectation consistent (adaptive TAP) inference: a Gaussian times 1-D sites.
 * @example
 * ```ts
 * import { ecAlgorithm, expectationConsistentGraph, marginalsOf } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // two weakly coupled spins with a field on the first: the magnetisation of the first is 0.664, which is tanh(0.8) to three places
 * const g = expectationConsistentGraph({ precision: [2, -0.1, -0.1, 2], sites: [{ kind: 'spin', field: 0.8 }, { kind: 'spin' }] })
 * const r = runAlgorithm(ecAlgorithm(g), { maxSteps: 200 })
 * (marginalsOf(g, r.state).x?.mean as number[])[0]?.toFixed(3) // => '0.664'
 * ```
 */
export function ecAlgorithm(g: CompiledGraph, options: EcOptions = {}): Algorithm<MpState, EpPath> {
  const alg = messagePassingAlgorithm(g, messagePassingSettings(momentProjection, ecDefaults, options))
  return { ...alg, name: 'ec' } as Algorithm<MpState, EpPath>
}

/**
 * Expectation consistent inference on a graph as a method: the knobs as a space (the parallel or serial schedule, damping,
 * improper policy, tolerance, sweep cap), the tape as {@link epContract}.
 *
 * @param g - The compiled graph.
 * @returns The method, named `ec`.
 * @concept expectation-consistent
 * @index function The expectation consistent method of a graph: knobs as a space, typed tape.
 * @example
 * ```ts
 * import { ecMethod, expectationConsistentGraph } from 'omai-engine/message-passing'
 * ecMethod(expectationConsistentGraph({ precision: [1], sites: [{ kind: 'spin' }] })).name // => 'ec'
 * ```
 */
export function ecMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  const base = messagePassingSpace(ecDefaults)
  return defineMethod<MpState, Config, EpPath>({
    name: 'ec',
    config: dict({
      ...base.spaces,
      schedule: choice(['flooding', 'serial'] as const, {
        default: 'flooding',
        description: 'all sites at once from the current Gaussian (EC), or one at a time (sequential EP)',
      }),
      maxSweeps: int(1, 10_000, { default: ecDefaults.maxSweeps, description: 'the sweep cap' }),
      damping: real(0, 0.95, { default: ecDefaults.damping, description: 'weight of the old site in the new' }),
    }),
    make: (c) => ecAlgorithm(g, messagePassingOptions(c)),
    contract: epContract,
  })
}
