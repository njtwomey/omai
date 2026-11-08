/**
 * Interventions and the do-operator by graph surgery: `doIntervention` cuts an intervened variable from its parents and
 * fixes its value, giving interventional distributions P(y | do(x)) of Bayes networks, average causal effects, and
 * interventional means and covariances of linear-Gaussian networks.
 */

import { OmaiError } from '../../base/index.ts'
import { type Factor, factorValues } from '../factor.ts'
import { variableElimination } from '../inference/elimination.ts'
import { type BayesNet, bayesNet, bayesNode } from '../models/bayes-net.ts'
import {
  type LinearGaussianBayesNet,
  linearGaussianBayesNet,
  linearGaussianPosterior,
} from '../models/linear-gaussian.ts'
import { type DiscreteAssignment, type LinearAssignment, type Scm, scm } from './scm.ts'

/** Interventions: the value each intervened variable is set to (a state name or index; a number for linear models). */
export type Interventions = Readonly<Record<string, string | number>>

function stateIndex(states: readonly string[], v: string | number, name: string): number {
  const i = typeof v === 'number' ? v : states.indexOf(v)
  if (!(Number.isInteger(i) && i >= 0 && i < states.length))
    throw new OmaiError('value', `do: ${name} has no state ${String(v)}`)
  return i
}

/**
 * The do-operator by graph surgery (Pearl 2009): the mutilated model in which each intervened variable loses its
 * parents and is fixed at its value, everything else unchanged. Works on Bayes networks (a point-mass CPT), structural
 * causal models (a constant assignment) and linear-Gaussian networks (a near-deterministic root at the value).
 *
 * @param model - The model.
 * @param interventions - do(X = x) for each intervened variable.
 * @returns The mutilated model, of the same kind.
 * @throws OmaiError with code `value` for unknown variables or states.
 * @concept interventions
 * @example
 * ```ts
 * import { bayesNet } from 'omai-engine/pgm'
 * import { doIntervention } from 'omai-engine/pgm/causal'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] }])
 * const cut = doIntervention(bn, { b: '1' }) as typeof bn
 * cut.nodes.map((n) => [n.name, n.parents, n.cpt]) // => [['a', [], [0.5, 0.5]], ['b', [], [0, 1]]]
 * ```
 */
export function doIntervention(
  model: BayesNet | Scm | LinearGaussianBayesNet,
  interventions: Interventions,
): BayesNet | Scm | LinearGaussianBayesNet {
  const names = new Set(model.kind === 'scm' ? model.variables.map((v) => v.name) : model.nodes.map((n) => n.name))
  for (const v of Object.keys(interventions))
    if (!names.has(v)) throw new OmaiError('value', `do: unknown variable ${v}`)
  if (model.kind === 'bayesNet') {
    return bayesNet(
      model.nodes.map((n) => {
        const value = interventions[n.name]
        if (value === undefined) return n
        const s = stateIndex(n.states, value, n.name)
        return { ...n, parents: [], cpt: n.states.map((_, i) => (i === s ? 1 : 0)) }
      }),
    )
  }
  if (model.kind === 'linearGaussianBayesNet') {
    return linearGaussianBayesNet(
      model.nodes.map((n) => {
        const value = interventions[n.name]
        if (value === undefined) return n
        if (typeof value !== 'number') throw new OmaiError('value', `do: ${n.name} needs a numeric value`)
        return { ...n, parents: [], weights: [], intercept: value, variance: 1e-300 }
      }),
    )
  }
  return scm(
    model.variables.map((v): DiscreteAssignment | LinearAssignment => {
      const value = interventions[v.name]
      if (value === undefined) return v
      if (v.kind === 'discrete') {
        const s = stateIndex(v.states, value, v.name)
        return { ...v, parents: [], noise: [1], assign: [s] }
      }
      if (typeof value !== 'number') throw new OmaiError('value', `do: ${v.name} needs a numeric value`)
      return {
        ...v,
        parents: [],
        weights: [],
        intercept: value,
        noise: { distribution: v.noise.distribution, scale: 0 },
      }
    }),
  )
}

/**
 * An interventional distribution P(query | do(X = x), evidence) of a Bayes network: inference in the mutilated network
 * (pgmpy's CausalInference.query with `do`).
 *
 * @param bn - The network.
 * @param query - The query variables.
 * @param interventions - The interventions.
 * @param evidence - Observed states after the intervention.
 * @returns The distribution over the query.
 * @throws OmaiError with code `value` when a query variable is intervened on.
 * @concept interventions
 * @example
 * ```ts
 * import { bayesNet, factorValues } from 'omai-engine/pgm'
 * import { interventionalDistribution } from 'omai-engine/pgm/causal'
 * // Seeing b = 1 is evidence about a; setting b = 1 is not.
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [0.9, 0.1, 0.2, 0.8] }])
 * factorValues(interventionalDistribution(bn, ['a'], { b: '1' })) // => [0.5, 0.5]
 * ```
 */
export function interventionalDistribution(
  bn: BayesNet,
  query: readonly string[],
  interventions: Interventions,
  evidence: Readonly<Record<string, string | number>> = {},
): Factor {
  for (const q of query)
    if (q in interventions) throw new OmaiError('value', `interventionalDistribution: ${q} is intervened on`)
  const cut = doIntervention(bn, interventions) as BayesNet
  return variableElimination(cut, query, { evidence }).posterior
}

/**
 * The average causal effect of a binary contrast on an outcome state: P(Y = y | do(X = treated)) - P(Y = y | do(X =
 * control)) in a Bayes network.
 *
 * @param bn - The network.
 * @param treatment - The treatment variable.
 * @param outcome - The outcome variable.
 * @param contrast - The treated and control states, and the outcome state of interest.
 * @returns The causal risk difference.
 * @concept interventions
 * @example
 * ```ts
 * import { bayesNet } from 'omai-engine/pgm'
 * import { averageCausalEffect } from 'omai-engine/pgm/causal'
 * const bn = bayesNet([{ name: 'x', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'y', states: ['0', '1'], parents: ['x'], cpt: [0.9, 0.1, 0.4, 0.6] }])
 * Math.round(averageCausalEffect(bn, 'x', 'y', { treated: '1', control: '0', outcome: '1' }) * 1e12) / 1e12 // => 0.5
 * ```
 */
export function averageCausalEffect(
  bn: BayesNet,
  treatment: string,
  outcome: string,
  contrast: { readonly treated: string | number; readonly control: string | number; readonly outcome: string | number },
): number {
  const y = stateIndex(bayesNode(bn, outcome).states, contrast.outcome, outcome)
  const p = (x: string | number): number =>
    factorValues(interventionalDistribution(bn, [outcome], { [treatment]: x }))[y] ?? 0
  return p(contrast.treated) - p(contrast.control)
}

/**
 * The interventional mean and covariance of some variables of a linear-Gaussian network: E[query | do(X = x)] by
 * conditioning the mutilated network (in which the intervened variables are fixed).
 *
 * @param g - The network.
 * @param query - The query variables.
 * @param interventions - Numeric interventions.
 * @returns The mean and covariance of the query under the intervention.
 * @concept interventions
 * @example
 * ```ts
 * import { linearGaussianBayesNet } from 'omai-engine/pgm'
 * import { linearInterventionalMoments } from 'omai-engine/pgm/causal'
 * const g = linearGaussianBayesNet([
 *   { name: 'x', parents: [], weights: [], intercept: 0, variance: 1 },
 *   { name: 'y', parents: ['x'], weights: [2], intercept: 1, variance: 1 },
 * ])
 * linearInterventionalMoments(g, ['y'], { x: 3 }).mean.map((v) => Math.round(v * 1e9) / 1e9) // => [7]
 * ```
 */
export function linearInterventionalMoments(
  g: LinearGaussianBayesNet,
  query: readonly string[],
  interventions: Readonly<Record<string, number>>,
): { readonly mean: readonly number[]; readonly covariance: readonly (readonly number[])[] } {
  const cut = doIntervention(g, interventions) as LinearGaussianBayesNet
  const m = linearGaussianPosterior(cut, query, interventions)
  return { mean: m.mean, covariance: m.covariance }
}
