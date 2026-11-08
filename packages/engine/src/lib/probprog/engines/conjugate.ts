/**
 * Closed-form inference for conjugate programs, where every latent site is a conjugate prior of its observed children
 * (beta-Bernoulli, normal-normal, gamma-Poisson and the rest), so each posterior is a conjugate update. Gathering each
 * structure's data, the posteriors, the closed-form posterior predictive and the engine for the planner.
 */

import { OmaiError } from '../../base/index.ts'
import {
  bernoulli,
  betaBinomial,
  categorical,
  type Distribution,
  dirichletMultinomial,
  multivariateNormal,
  negativeBinomial,
  normal,
  studentT,
} from '../../dists/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  item,
  matmul,
  primalOf,
  reshape,
  sqrt,
  square,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { unexpanded } from '../_expand.ts'
import { analyse, type ConjugateStructure, fixedParameters, type ProgramAnalysis } from '../analysis.ts'
import { logJoint } from '../density.ts'
import type { ProgramTrace } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/** The observations and known parameters of one conjugate structure, from a trace. */
export interface ConjugateData {
  /** The children's values stacked along a new first axis. */
  readonly data: Value
  readonly fixed: Readonly<Record<string, Value>>
}

/**
 * Gather a conjugate structure's observations (stacked on the first axis, the prior's batch and the likelihood's event
 * on the rest) and the likelihood's known parameters from a trace.
 *
 * @param t - A trace of the program.
 * @param c - The structure.
 * @returns The data and known parameters.
 * @throws OmaiError with code `value` when the children's shapes do not line up with the prior, or the known
 *   parameters vary across observations.
 * @concept conjugate-priors
 * @example
 * ```ts
 * import { analyse, conjugateData, observe, plate, sample } from 'omai-engine/probprog'
 * import { gamma, poisson } from 'omai-engine/dists'
 * const model = () => { const r = sample('r', gamma(1, 1)); plate('n', 3, () => observe('k', poisson(r), [1, 4, 2])) }
 * const a = analyse(model, undefined)
 * conjugateData(a.prototype, a.conjugate[0]).data.shape // => [3]
 * ```
 */
export function conjugateData(t: ProgramTrace, c: ConjugateStructure | undefined): ConjugateData {
  if (c === undefined) throw new OmaiError('value', 'conjugateData: no conjugate structure')
  const prior = t.sites.get(c.prior)
  if (prior?.type !== 'sample') throw new OmaiError('value', `conjugateData: no site ${c.prior}`)
  const thetaBatch = prior.value.shape.slice(0, prior.value.shape.length - prior.fn.eventShape.length)
  const pieces: Value[] = []
  let fixed: Record<string, Value> | undefined
  for (const name of c.children) {
    const child = t.sites.get(name)
    if (child?.type !== 'sample') throw new OmaiError('value', `conjugateData: no site ${name}`)
    const rest = [...thetaBatch, ...child.fn.eventShape]
    const v = child.value
    const tail = v.shape.slice(v.shape.length - rest.length)
    if (v.shape.length < rest.length || tail.some((n, i) => n !== rest[i]))
      throw new OmaiError('value', `conjugateData: ${name} does not line up with the prior ${c.prior}`)
    pieces.push(reshape(v, [-1, ...rest]))
    const f = fixedParameters(unexpanded(child.fn), c.pair)
    if (f === undefined) throw new OmaiError('value', `conjugateData: ${name} lacks a known parameter`)
    for (const [k, x] of Object.entries(f))
      if (product(x.shape) > Math.max(1, product(prior.value.shape)) * Math.max(1, product(child.fn.eventShape)))
        throw new OmaiError('value', `conjugateData: the known ${k} of ${name} varies across observations`)
    fixed = fixed ?? f
  }
  return { data: concat(pieces), fixed: fixed ?? {} }
}

/**
 * The closed-form posterior of each conjugate structure.
 *
 * @param t - A trace (its children's values are the data).
 * @param structures - The conjugate structures.
 * @returns The posterior distribution of each prior site.
 * @concept conjugate-priors
 * @example
 * ```ts
 * import { analyse, conjugatePosteriors, observe, plate, sample } from 'omai-engine/probprog'
 * import { gamma, poisson } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const r = sample('r', gamma(1, 1)); plate('n', 3, () => observe('k', poisson(r), [1, 4, 2])) }
 * const a = analyse(model, undefined)
 * // Gamma(1 + 7, 1 + 3): mean 2.
 * item(conjugatePosteriors(a.prototype, a.conjugate)['r']?.mean() ?? 0) // => 2
 * ```
 */
export function conjugatePosteriors(
  t: ProgramTrace,
  structures: readonly ConjugateStructure[],
): Record<string, Distribution> {
  const out: Record<string, Distribution> = {}
  for (const c of structures) {
    const prior = t.sites.get(c.prior)
    if (prior?.type !== 'sample') continue
    const { data, fixed } = conjugateData(t, c)
    out[c.prior] = c.pair.posterior(prior.fn, data, fixed)
  }
  return out
}

/**
 * The posterior predictive distribution of one new observation of a conjugate structure, in closed form where the
 * library has the family (Bernoulli and beta-binomial from a beta, negative binomial from a gamma-Poisson, normal from
 * a normal mean, Student t from an inverse-gamma variance, categorical and Dirichlet-multinomial from a Dirichlet,
 * multivariate normal from a normal mean vector); undefined otherwise (use Monte Carlo through `predictive`).
 *
 * @param c - The structure.
 * @param posterior - The posterior of its prior site.
 * @param fixed - The likelihood's known parameters.
 * @returns The predictive distribution, or undefined.
 * @concept conjugate-priors
 * @example
 * ```ts
 * import { analyse, conjugatePosteriors, conjugatePredictive, observe, plate, sample } from 'omai-engine/probprog'
 * import { beta, bernoulli } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const p = sample('p', beta(1, 1)); plate('n', 4, () => observe('y', bernoulli({ probs: p }), [1, 1, 0, 1])) }
 * const a = analyse(model, undefined)
 * const c = a.conjugate[0]
 * const post = conjugatePosteriors(a.prototype, a.conjugate)['p']
 * c !== undefined && post !== undefined ? item(conjugatePredictive(c, post, {})?.mean() ?? 0) : 0 // => 0.6666666666666666
 * ```
 */
export function conjugatePredictive(
  c: ConjugateStructure,
  posterior: Distribution,
  fixed: Readonly<Record<string, TensorLike>>,
): Distribution | undefined {
  const p = posterior.params
  const get = (name: string): Value => {
    const v = p[name]
    if (v === undefined) throw new OmaiError('value', `conjugatePredictive: the posterior has no ${name}`)
    return v
  }
  const known = (name: string): Value => {
    const v = fixed[name]
    if (v === undefined) throw new OmaiError('value', `conjugatePredictive: no known ${name}`)
    return primalOf(asTensor(v))
  }
  const key = `${c.likelihood}.${c.parameter}`
  switch (key) {
    case 'bernoulli.probs':
      return bernoulli({ probs: div(get('a'), add(get('a'), get('b'))) })
    case 'binomial.probs':
      return betaBinomial(known('n'), get('a'), get('b'))
    case 'poisson.rate':
      return negativeBinomial(get('concentration'), div(get('rate'), add(get('rate'), 1)))
    case 'normal.loc':
      return normal(get('loc'), sqrt(add(square(get('scale')), square(known('scale')))))
    case 'normal.variance':
      return studentT(
        add(get('concentration'), get('concentration')),
        known('loc'),
        sqrt(div(get('scale'), get('concentration'))),
      )
    case 'categorical.probs': {
      const alpha = get('concentration')
      return categorical({ probs: div(alpha, sum(alpha, -1, true)) })
    }
    case 'multinomial.probs': {
      const n = fixed['n']
      if (n === undefined) return undefined
      return dirichletMultinomial(item(asTensor(n)), get('concentration'))
    }
    case 'multivariateNormal.loc': {
      const l = get('scaleTril')
      const cov = add(matmul(l, swapaxes(l, -1, -2)), known('covariance'))
      return multivariateNormal(get('loc'), { covariance: cov })
    }
    default:
      return undefined
  }
}

/**
 * Closed-form inference for a conjugate program: every latent site is the conjugate prior of its observed children,
 * so each posterior is a conjugate update, the joint posterior is their product, and the evidence follows exactly from
 * the candidate's formula log p(x) = log p(theta, x) - log p(theta | x) at the posterior mean.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param analysis - Its analysis (default: analysed here).
 * @returns The exact posterior; `diagnostics.predictive` lists the sites with a closed-form predictive.
 * @throws OmaiError with code `value` when the program is not fully conjugate.
 * @concept conjugate-priors
 * @example
 * ```ts
 * import { conjugateInference, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { item } from 'omai-engine/tensor'
 * const model = (y: number[]) => { const mu = sample('mu', normal(0, 1)); plate('n', y.length, () => observe('y', normal(mu, 1), y)) }
 * // Posterior N(sum y / (n + 1), 1 / (n + 1)).
 * item(conjugateInference(model, [1, 2, 3]).mean('mu')) // => 1.5
 * ```
 */
export function conjugateInference<D = void>(model: Model<D>, data: D, analysis?: ProgramAnalysis): Posterior {
  const a = analysis ?? analyse(model, data)
  if (!a.fullyConjugate) throw new OmaiError('value', 'conjugateInference: the program is not fully conjugate')
  const t = a.prototype
  const marginals = conjugatePosteriors(t, a.conjugate)
  const at: Record<string, Value> = {}
  for (const [name, d] of Object.entries(marginals)) at[name] = primalOf(d.mean())
  const joint = item(logJoint(model, data, at).logJoint)
  const posteriorAt = Object.entries(marginals).reduce((s, [name, d]) => s + item(sum(d.logProb(at[name] as Value))), 0)
  const predictive = a.conjugate.filter((c) => {
    const post = marginals[c.prior]
    if (post === undefined) return false
    try {
      return conjugatePredictive(c, post, conjugateData(t, c).fixed) !== undefined
    } catch {
      return false
    }
  })
  return definePosterior({
    form: 'exact',
    engine: 'conjugate',
    sites: Object.keys(marginals),
    marginals,
    logEvidence: { value: joint - posteriorAt, standardError: 0, method: 'conjugate (candidate formula)' },
    diagnostics: {
      structures: a.conjugate.map((c) => `${c.priorFamily}-${c.likelihood} on ${c.prior}`),
      predictive: predictive.flatMap((c) => c.children),
    },
    sampler: (key: Key, n: number) => {
      const out: Record<string, Tensor> = {}
      Object.entries(marginals).forEach(([name, d], i) => {
        out[name] = primalOf(d.sample(foldIn(key, i), [n]))
      })
      return out
    },
    logProb: (values) =>
      Object.entries(marginals).reduce((s, [name, d]) => {
        const v = values[name]
        if (v === undefined) throw new OmaiError('value', `conjugate posterior: no value for ${name}`)
        return s + toFlat(sum(d.logProb(v))).reduce((x, y) => x + y, 0)
      }, 0),
  })
}

/**
 * The conjugate engine: closed-form posteriors for programs whose every latent site is a conjugate prior of its
 * observed children (cost: the number of observations).
 *
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { analyse, conjugateEngine, observe, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const a = analyse(() => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 0) }, undefined)
 * conjugateEngine().applies(a).applies // => true
 * ```
 */
export function conjugateEngine(): InferenceEngine {
  return {
    name: 'conjugate',
    form: 'exact',
    description: 'closed-form posteriors and predictives of conjugate programs',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'exact', 0, 'observed sites have no likelihood')
      if (!a.fullyConjugate) {
        const missing = a.latent.filter((n) => !a.conjugate.some((c) => c.prior === n))
        return feasibility(
          false,
          'exact',
          0,
          missing.length > 0
            ? `no conjugate structure for ${missing.join(', ')}`
            : 'conjugate structures overlap or the program has factors',
        )
      }
      return feasibility(
        true,
        'exact',
        Math.max(1, a.observations),
        `every latent site is a conjugate prior (${a.conjugate.map((c) => `${c.priorFamily}-${c.likelihood}`).join(', ')})`,
      )
    },
    run: <D>(model: Model<D>, data: D, a: ProgramAnalysis, _options: EngineRunOptions) =>
      conjugateInference(model, data, a),
  }
}
