/**
 * Item response theory scoring as a spec: twelve respondents answer eight items of known difficulty and discrimination
 * (the 2PL model, `P(correct) = sigmoid(a_j (theta_p - b_j))`), the abilities `theta_p ~ N(0, 1)` are the latents. The
 * abilities are independent given the items, so each posterior is one-dimensional and the truth is a quadrature per
 * respondent; the library's EAP estimate (`irtAbility` of `omai-engine/pgm`, a coarser grid) is an independent check of the
 * posterior means. The responses are simulated with the library's `irtSimulate`. With the items unknown (the calibration
 * problem, whose maximum-marginal-likelihood fit `fitIrt` computes) the posterior is a 28-dimensional density with no
 * independent truth, so it is not a benchmark here.
 */

import { type IrtItem, irtSimulate } from '../../pgm/index.ts'
import type { Key } from '../../random/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { type MarginalOracle, weightedOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'

/** The eight items: discriminations 0.8 to 2, difficulties -1.5 to 1.5. */
export const IRT_ITEMS: readonly IrtItem[] = [
  { a: 0.8, b: -1.5, c: 0 },
  { a: 1.2, b: -1, c: 0 },
  { a: 1, b: -0.5, c: 0 },
  { a: 1.6, b: 0, c: 0 },
  { a: 1.4, b: 0.3, c: 0 },
  { a: 0.9, b: 0.8, c: 0 },
  { a: 2, b: 1, c: 0 },
  { a: 1.1, b: 1.5, c: 0 },
]

function generateResponses(key: Key, size = 12): BenchmarkData {
  const { responses, theta } = irtSimulate(key, size, IRT_ITEMS)
  return {
    data: { a: IRT_ITEMS.map((i) => i.a), b: IRT_ITEMS.map((i) => i.b), y: responses },
    generating: { theta },
  }
}

/**
 * The posterior of each respondent's ability by quadrature: the product of the N(0, 1) prior and the response likelihood on
 * a grid of `points` values over `[-8, 8]`, and the log evidence as the sum over respondents of the log integral.
 *
 * @param a - The discriminations.
 * @param b - The difficulties.
 * @param y - The responses, `[person][item]`.
 * @param points - Grid size (default 4001).
 * @returns The truth: an oracle for each `theta[p]` and the log evidence.
 * @concept item-response-theory
 * @example
 * ```ts
 * import { irtScoringTruth } from 'omai-engine/probprog/models'
 * const t = irtScoringTruth([1], [0], [[1], [0]], 801)
 * [Object.keys(t.marginals), t.marginals['theta[0]'].mean > 0, t.marginals['theta[1]'].mean < 0] // => [['theta[0]', 'theta[1]'], true, true]
 * ```
 */
export function irtScoringTruth(
  a: readonly number[],
  b: readonly number[],
  y: readonly (readonly number[])[],
  points = 4001,
): BenchmarkTruth {
  const step = 16 / (points - 1)
  const grid = Float64Array.from({ length: points }, (_, i) => -8 + i * step)
  const marginals: Record<string, MarginalOracle> = {}
  let logEvidence = 0
  y.forEach((row, p) => {
    const weights = Float64Array.from(grid, (t) => {
      let logp = -0.5 * t * t - 0.5 * Math.log(2 * Math.PI)
      row.forEach((r, j) => {
        const eta = (a[j] as number) * (t - (b[j] as number))
        // log sigmoid(eta) and log sigmoid(-eta), stable.
        logp += r === 1 ? -Math.log1p(Math.exp(-eta)) : -Math.log1p(Math.exp(eta))
      })
      return Math.exp(logp)
    })
    logEvidence += Math.log(weights.reduce((s, v) => s + v, 0) * step)
    marginals[`theta[${p}]`] = weightedOracle(`theta[${p}]`, grid, weights)
  })
  return {
    kind: 'quadrature',
    source: `an independent quadrature per respondent (${points} points over [-8, 8]) of N(theta; 0, 1) times the 2PL response likelihood; the abilities are independent given the items, so the joint is the product`,
    logEvidence,
    marginals,
  }
}

/**
 * Item response theory scoring (2PL, items known): twelve respondents, eight items, `y_pj ~ Bernoulli(sigmoid(a_j (theta_p -
 * b_j)))` with `theta_p ~ N(0, 1)`. Each ability's posterior is one-dimensional, so the truth is exact to the quadrature;
 * expectation propagation approximates the logistic sites by Gaussians, NUTS and tempered SMC sample, and the Laplace
 * fit and mean-field guide are close because the posteriors are near normal.
 *
 * @concept item-response-theory
 * @index dataset 2PL item response scoring with known items; one-dimensional quadrature truth for each respondent.
 * @example
 * ```ts
 * import { itemResponseScoring } from 'omai-engine/probprog/models'
 * [itemResponseScoring.bound().sizes, itemResponseScoring.truth(undefined, 'fast').kind] // => [{ person: 12, item: 8 }, 'quadrature']
 * ```
 */
export const itemResponseScoring: BenchmarkModel = defineBenchmark({
  name: 'item-response-scoring',
  title: 'Item response theory (2PL scoring)',
  description:
    'Respondents with latent abilities answer items of known difficulty and discrimination through a logistic link: independent one-dimensional posteriors with a quadrature truth.',
  tags: ['irt', 'psychometrics', 'logistic', 'binary-data', 'plates', 'quadrature'],
  references: [
    'Birnbaum (1968), Some latent trait models and their use in inferring an examinee’s ability',
    'Bock and Aitkin (1981), Marginal maximum likelihood estimation of item parameters',
    'Gelman et al., Bayesian Data Analysis, 3rd ed., section on hierarchical logistic models',
  ],
  definition: defineModel('item-response-scoring', (m) => {
    const person = m.range('person')
    const item = m.range('item')
    const a = m.data('a', { dims: [item], support: 'positive', role: 'covariate' })
    const b = m.data('b', { dims: [item], role: 'covariate' })
    const y = m.data('y', { dims: [person, item], support: 'binary', role: 'response' })
    const theta = m.latent('theta', d.normal(0, 1), { dims: [person], doc: 'the ability of each respondent' })
    m.observe('y', d.bernoulli({ logits: a.mul(theta.sub(b)) }), y)
  }),
  generate: generateResponses,
  truth: (data, quality) =>
    irtScoringTruth(
      data.data['a'] as number[],
      data.data['b'] as number[],
      data.data['y'] as number[][],
      quality === 'fast' ? 801 : 4001,
    ),
  plans: (scale) => [
    nutsPlan(scale, { note: 'twelve independent near-normal posteriors: agrees within Monte Carlo error' }),
    temperedPlan(scale, {
      expect: 'report',
      note: 'tempered particles over twelve abilities: consistent, with Monte Carlo error larger than NUTS',
    }),
    {
      kind: 'ep',
      config: {},
      steps: 200,
      expect: 'agree',
      note: 'expectation propagation projects each logistic site onto a Gaussian: close to the quadrature on these near-normal posteriors',
      tolerances: { meanError: 0.1, meanZ: 6, sdRatio: [0.9, 1.1], ks: 0.1 },
    },
    laplacePlan(scale, { note: 'the posteriors are near normal: the mode and curvature are close' }),
    sviPlan(scale, {
      note: 'independent abilities, so the mean-field family is right in form; a mean-field normal fitted by stochastic gradients: close, with optimisation noise',
    }),
  ],
})
