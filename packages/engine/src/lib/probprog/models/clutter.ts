/**
 * Minka's clutter problem (2001): observations are draws of a hidden location `theta` (a standard normal around it) mixed
 * with clutter (a wider normal around zero), `y_i ~ (1 - w) N(theta, 1) + w N(0, 10)`, with `theta ~ N(0, 100)`; the
 * posterior of `theta` is a mixture of up to 2^n Gaussians, so it is bimodal or heavy-tailed where the data are ambiguous,
 * and it is one-dimensional, which makes the exact posterior a quadrature. It is the showcase of expectation propagation:
 * each clutter factor is a numeric site (`numericOperator` of `omai-engine/message-passing`, its tilted moments integrated by
 * a rule), and EP's Gaussian is close to the exact moments where a Laplace fit (one mode) and a mean-field guide are not.
 * The spec writes the mixture with a Bernoulli indicator per observation (`where` picks the mean and the scale), which
 * NUTS sums out and the grid enumerates; `clutterEp` runs EP on the same data with the numeric operator.
 */

import {
  compileGraph,
  defaultOperators,
  gaussHermiteRule,
  gaussianPrior,
  laplaceRule,
  marginalsOf,
  numericEpAlgorithm,
  numericOperator,
  quadratureRule,
} from '../../message-passing/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { add, arange, mul, toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type MarginalOracle, weightedOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan } from './plans.ts'

const W = 0.5
const THETA = 2
const CLUTTER_VARIANCE = 10
const PRIOR_VARIANCE = 100

/** The centre of the distractor cluster: observations that the model can explain as inliers of a different location, or as clutter. */
const DISTRACTOR = 8

function generateClutter(key: Key, size = 12): BenchmarkData {
  const e = toFlat(drawNormal(foldIn(key, 0), [size]))
  const c = toFlat(drawNormal(foldIn(key, 1), [size]))
  const u = toFlat(drawUniform(foldIn(key, 2), [size]))
  // About 60% of the observations are near the location, the rest near a distractor cluster; some of each are clutter draws.
  const inliers = Math.ceil(0.6 * size)
  const clutter = Array.from(u, (v) => v < 0.2)
  return {
    data: {
      y: clutter.map((isClutter, i) =>
        isClutter ? Math.sqrt(CLUTTER_VARIANCE) * (c[i] ?? 0) : (i < inliers ? THETA : DISTRACTOR) + (e[i] ?? 0),
      ),
    },
    generating: { theta: THETA, distractor: DISTRACTOR, clutter: clutter.map((v) => (v ? 1 : 0)) },
  }
}

/** The clutter factor of one observation as a function of `theta`. */
const density = (y: number, theta: number): number =>
  ((1 - W) * Math.exp(-0.5 * (y - theta) ** 2)) / Math.sqrt(2 * Math.PI) +
  (W * Math.exp((-0.5 * y * y) / CLUTTER_VARIANCE)) / Math.sqrt(2 * Math.PI * CLUTTER_VARIANCE)

/**
 * The exact posterior of the location by quadrature: the prior `N(0, 100)` times the product over the observations of the
 * mixture density `(1 - w) N(y; theta, 1) + w N(y; 0, 10)` (the indicators summed out in closed form) on a grid of `points` values over
 * `[-60, 60]` (six prior standard deviations: with few observations the posterior keeps a flat tail, the all-clutter explanation, that
 * a narrower box would cut), and the evidence as the integral.
 *
 * @param y - The observations.
 * @param points - Grid size (default 12001).
 * @returns The truth: an oracle for `theta` and the log evidence.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { clutterTruth } from 'omai-engine/probprog/models'
 * const t = clutterTruth([1.8, 2.3, 2.0, 1.6], 801)
 * Math.abs((t.marginals['theta']?.mean ?? 0) - 2) < 0.5 // => true
 * ```
 */
export function clutterTruth(y: readonly number[], points = 12001): BenchmarkTruth {
  const step = 120 / (points - 1)
  const grid = toFlat(add(mul(arange(points), step), -60))
  const logw = Float64Array.from(
    grid,
    (t) =>
      (-0.5 * t * t) / PRIOR_VARIANCE -
      0.5 * Math.log(2 * Math.PI * PRIOR_VARIANCE) +
      y.reduce((a, v) => a + Math.log(density(v, t)), 0),
  )
  const max = Math.max(...logw)
  const weights = Float64Array.from(logw, (v) => Math.exp(v - max))
  const oracle: MarginalOracle = weightedOracle('theta', grid, weights)
  return {
    kind: 'quadrature',
    source: `a grid of ${points} values of theta over [-60, 60] (spacing ${step.toFixed(4)}), the product of the prior and the n mixture densities, the clutter indicators summed out in closed form; the evidence is the trapezoid integral`,
    logEvidence: max + Math.log(weights.reduce((a, v) => a + v, 0) * step),
    marginals: { theta: oracle },
  }
}

/** A rule for the numeric clutter sites of {@link clutterEp}. */
export type ClutterRule = 'quadrature' | 'gauss-hermite' | 'laplace'

/**
 * Expectation propagation on the clutter problem with numeric sites: a Gaussian prior `N(0, 100)` on `theta` and one numeric factor
 * per observation (the mixture density), the tilted moments of each integrated by the rule (the engine's panelled
 * quadrature, a Gauss-Hermite rule of 32 points, or the Laplace rule).
 *
 * @param y - The observations.
 * @param rule - The numeric rule (default `quadrature`).
 * @returns EP's Gaussian for `theta`: its mean and standard deviation, and whether the run converged.
 * @concept numerical-projections
 * @example
 * ```ts
 * import { clutterEp } from 'omai-engine/probprog/models'
 * const r = clutterEp([1.8, 2.3, 2.0, 1.6, 2.4])
 * [r.converged, Math.abs(r.mean - 2) < 0.5] // => [true, true]
 * ```
 */
export function clutterEp(
  y: readonly number[],
  rule: ClutterRule = 'quadrature',
): { readonly mean: number; readonly sd: number; readonly converged: boolean } {
  const site = numericOperator({
    name: 'clutter',
    arity: 1,
    logDensity: ([theta], p) => Math.log(density(p['y'] as number, theta as number)),
  })
  const g = compileGraph(
    {
      variables: [{ name: 'theta', family: { kind: 'gaussian' } }],
      factors: [
        gaussianPrior('theta', 0, PRIOR_VARIANCE),
        ...y.map((v) => ({ operator: 'clutter', args: ['theta'], params: { y: v } })),
      ],
    },
    { ...defaultOperators(), clutter: site },
  )
  const rules = { quadrature: quadratureRule, 'gauss-hermite': gaussHermiteRule(32), laplace: laplaceRule }
  const r = runAlgorithm(numericEpAlgorithm(g, { rule: rules[rule], damping: 0.3, tol: 1e-9 }), { maxSteps: 400 })
  const m = marginalsOf(g, r.state)['theta'] as unknown as { mean: number; variance: number }
  return { mean: m.mean, sd: Math.sqrt(m.variance), converged: r.status === 'converged' }
}

/**
 * Minka's clutter problem with a distractor: twelve observations, seven near a location `theta = 2` and five near a second cluster at 8
 * (about a fifth of all of them replaced by clutter from `N(0, 10)`), `theta ~ N(0, 100)`, and the model `y ~ (1 - w) N(theta, 1) + w
 * N(0, 10)` with `w = 1/2`. The data can be explained either as inliers of `theta` near 2 with the second cluster as clutter, or the
 * other way round, so the posterior of `theta` has two modes (the one near 8, with more of the data, higher): a mixture of Gaussians
 * whose exact form is a one-dimensional quadrature. NUTS has to cross between the modes; the Laplace fit and the mean-field guide are
 * one normal at one mode; expectation propagation with numeric sites (`clutterEp`) finds a Gaussian too. Every plan is reported, none
 * required to agree: what each does is the finding (`benchmark/clutter-ep` tabulates it).
 *
 * @concept posterior-validation
 * @index dataset Minka's clutter problem with a distractor: bimodal robust location, quadrature truth.
 * @example
 * ```ts
 * import { clutterProblem } from 'omai-engine/probprog/models'
 * [clutterProblem.bound().sizes, clutterProblem.truth(undefined, 'fast').kind] // => [{ row: 12 }, 'quadrature']
 * ```
 */
export const clutterProblem: BenchmarkModel = defineBenchmark({
  name: 'clutter-problem',
  title: 'The clutter problem',
  description:
    "Minka's robust location model with a distractor cluster: observations from a normal around an unknown location mixed with wide clutter, whose one-dimensional posterior has two modes and whose exact form is a quadrature; the EP showcase.",
  tags: ['robust', 'mixture', 'discrete-latent', 'ep', 'one-dimensional', 'multimodal', 'quadrature'],
  references: [
    'Minka (2001), Expectation propagation for approximate Bayesian inference (UAI), the clutter problem',
    'Minka (2001), A family of algorithms for approximate Bayesian inference (thesis), chapter 3',
  ],
  definition: defineModel('clutter-problem', (b) => {
    const row = b.range('row')
    const y = b.data('y', { dims: [row], role: 'response' })
    const theta = b.latent('theta', d.normal(0, Math.sqrt(PRIOR_VARIANCE)), { doc: 'the location' })
    const z = b.latent('z', d.bernoulli({ probs: W }), { dims: [row], doc: 'whether the observation is clutter' })
    const isClutter = z.gt(0.5)
    b.observe('y', d.normal(ops.where(isClutter, 0, theta), ops.where(isClutter, Math.sqrt(CLUTTER_VARIANCE), 1)), y)
  }),
  generate: (key, size) => generateClutter(key, size),
  truth: (data, quality) => clutterTruth(data.data['y'] as number[], quality === 'fast' ? 4001 : 12001),
  plans: (scale) => [
    nutsPlan(scale, {
      expect: 'report',
      note: 'NUTS with the clutter indicators summed out: a one-dimensional bimodal posterior, so how the chains share the modes is the finding',
    }),
    laplacePlan(scale, { note: 'one normal at the mode it finds: misses the second mode and the mass between them' }),
    sviPlan(scale, { note: 'a mean-field normal on theta: one normal for a bimodal posterior' }),
  ],
})
