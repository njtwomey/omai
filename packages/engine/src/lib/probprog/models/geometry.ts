/**
 * The geometry benchmarks, targets with no data whose difficulty is their shape: Neal's funnel (centred and non-centred), Haario's
 * banana and an equal two-mode Gaussian mixture in two dimensions. They have no observed nodes, so the spec-level capability
 * checks of the methods refuse them (a prior-only spec has no likelihood) and the methods are made on the compiled program,
 * whose checks are the engines' own. The truth is analytic: the funnel and the banana are defined by their conditional
 * structure, so iid draws of an exact sampler give the marginals (with the exact moments), and the mixture's marginals are
 * normal mixtures. The expected results are the classical ones: NUTS diverges in the centred funnel's neck and stays
 * in one mode of the mixture, a normal approximation misses the curvature of the banana, a mean-field guide is the wrong
 * shape for all three.
 */

import { normal as drawNormal, foldIn, key as makeKey } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { gibbsPlan, importancePlan, laplacePlan, nutsPlan, sviPlan } from './plans.ts'
import { mixtureOracle, samplesOracle } from './truth.ts'

/** The key of the iid draws that make the oracles of the targets with no CDF in closed form (fixed: a truth is a constant). */
const key0 = makeKey(20260810)

/** The data of a target with no data: nothing, and the parameters that define it. */
const noData = (generating: BenchmarkData['generating']): BenchmarkData => ({ data: {}, generating })

const FUNNEL_SCALE = 3
const FUNNEL_COORDS = 9
/** The number of iid draws that make an oracle of a target with no CDF in closed form. */
const ORACLE_DRAWS = 100_000

/**
 * Neal's funnel, centred: `v ~ N(0, 3²)` and nine coordinates `x_i | v ~ N(0, e^v)` (Neal 2003, section 8). The density
 * is a funnel, wide at large `v` and a needle at small `v`: a fixed step size is too large in the neck and too small in
 * the mouth, so NUTS diverges and under-explores the neck (`v` is too large), a normal approximation cannot be right
 * (the density is unbounded as `v` falls) and a mean-field guide has the wrong shape. The truth is analytic: `v` is
 * normal, `x_i` is a normal scale mixture with mean 0 and sd `exp(9 / 4)`.
 *
 * @concept posterior-validation
 * @index dataset Neal's funnel, centred: analytic truth; NUTS diverges in the neck.
 * @example
 * ```ts
 * import { funnelCentred } from 'omai-engine/probprog/models'
 * funnelCentred.target // => 'density'
 * ```
 */
export const funnelCentred: BenchmarkModel = defineBenchmark({
  name: 'funnel-centred',
  title: "Neal's funnel, centred",
  description:
    "Neal's funnel in ten dimensions, written directly: v ~ N(0, 9) and nine coordinates x_i ~ N(0, e^v). The neck defeats a fixed step size.",
  tags: ['funnel', 'geometry', 'divergences', 'centred', 'density', 'analytic', 'hierarchical-prior'],
  references: [
    'Neal (2003), Slice sampling, Annals of Statistics 31 (section 8, the funnel)',
    'Betancourt and Girolami (2015), Hamiltonian Monte Carlo for hierarchical models',
    'the `funnelDensity` of omai-engine/datasets (the same density in two dimensions, whose log density the tests compare with)',
  ],
  target: 'density',
  definition: defineModel('funnel-centred', (m) => {
    const v = m.latent('v', d.normal(0, FUNNEL_SCALE), { doc: 'the log variance of the coordinates' })
    m.latent('x', d.normal(0, ops.exp(v.mul(0.5))), {
      dims: [m.range('coord', { size: FUNNEL_COORDS })],
    })
  }),
  generate: () => noData({ scale: FUNNEL_SCALE, coordinates: FUNNEL_COORDS }),
  truth: () => funnelTruth(),
  plans: (scale) => [
    nutsPlan(scale, {
      expect: 'differ',
      quantities: ['v'],
      note: 'divergent transitions in the neck and too small a spread of v: the sampler never reaches the narrowest part (the non-centred funnel has none)',
      config: { targetAcceptance: 0.9 },
    }),
    sviPlan(scale, {
      expect: 'differ',
      quantities: ['v', 'x[0]'],
      note: 'a mean-field normal cannot be a funnel: v and x are dependent in scale, not in location',
    }),
  ],
})

/**
 * Neal's funnel, non-centred: `v ~ N(0, 3²)`, `z_i ~ N(0, 1)` and `x_i = e^(v / 2) z_i`, the same distribution of `(v, x)`
 * with the funnel in a deterministic transform instead of the density the sampler sees. NUTS has no divergences and
 * agrees with the truth; a mean-field guide is exact for `(v, z)` (independent normals), so it agrees too.
 *
 * @concept posterior-validation
 * @index dataset Neal's funnel, non-centred: the same law of (v, x) with the funnel removed from the density.
 * @example
 * ```ts
 * import { funnelNonCentred } from 'omai-engine/probprog/models'
 * funnelNonCentred.definition?.spec.nodes.map((n) => n.kind) // => ['latent', 'latent', 'derived']
 * ```
 */
export const funnelNonCentred: BenchmarkModel = defineBenchmark({
  name: 'funnel-non-centred',
  title: "Neal's funnel, non-centred",
  description:
    "Neal's funnel with x_i = e^(v/2) z_i and z_i standard normal: the same law as the centred funnel, with no funnel in the density.",
  tags: ['funnel', 'geometry', 'non-centred', 'reparametrisation', 'density', 'analytic'],
  references: [
    'Neal (2003), Slice sampling, Annals of Statistics 31 (section 8, the funnel)',
    'Papaspiliopoulos, Roberts and Skoeld (2007), A general framework for the parametrization of hierarchical models',
  ],
  target: 'density',
  definition: defineModel('funnel-non-centred', (m) => {
    const v = m.latent('v', d.normal(0, FUNNEL_SCALE), { doc: 'the log variance of the coordinates' })
    const z = m.latent('z', d.normal(0, 1), { dims: [m.range('coord', { size: FUNNEL_COORDS })] })
    m.derived('x', ops.exp(v.mul(0.5)).mul(z))
  }),
  generate: () => noData({ scale: FUNNEL_SCALE, coordinates: FUNNEL_COORDS }),
  truth: () => funnelTruth(),
  plans: (scale) => [
    nutsPlan(scale, {
      quantities: ['v', 'x[0]'],
      note: 'no funnel in the density: no divergences, agrees with the truth',
      tolerances: { meanError: 0.15, sdRatio: [0.8, 1.2] },
    }),
    sviPlan(scale, {
      quantities: ['v'],
      note: 'v and z are independent normals, the mean-field family is exact for them',
    }),
  ],
})

/** The funnel's marginals: `v` normal, `x_i` a scale mixture (iid draws of the exact sampler, with the exact moments). */
function funnelTruth(): BenchmarkTruth {
  const z = toFlat(drawNormal(foldIn(key0, 1), [ORACLE_DRAWS, 2]))
  const x = new Float64Array(ORACLE_DRAWS)
  for (let i = 0; i < ORACLE_DRAWS; i++) x[i] = Math.exp((FUNNEL_SCALE * (z[2 * i] ?? 0)) / 2) * (z[2 * i + 1] ?? 0)
  const exactSd = Math.exp((FUNNEL_SCALE * FUNNEL_SCALE) / 4)
  return {
    kind: 'analytic',
    source: `v is N(0, ${FUNNEL_SCALE ** 2}) exactly; x_i | v ~ N(0, e^v) is a scale mixture with mean 0 and sd exp(${(FUNNEL_SCALE * FUNNEL_SCALE) / 4}), its CDF the empirical one of ${ORACLE_DRAWS} iid draws of the exact sampler`,
    marginals: {
      v: mixtureOracle('v', [{ weight: 1, mean: 0, sd: FUNNEL_SCALE }]),
      ...Object.fromEntries(
        Array.from({ length: FUNNEL_COORDS }, (_, i) => [
          `x[${i}]`,
          samplesOracle(`x[${i}]`, x, { mean: 0, sd: exactSd }),
        ]),
      ),
    },
  }
}

const BANANA = { a: 2, b: 0.5 }

/**
 * Haario's banana: `x_0 ~ N(0, 2²)` and `x_1 | x_0 ~ N(0.5 (x_0² - 4), 1)`, a curved, strongly nonlinear ridge in two
 * dimensions (Haario, Saksman and Tamminen 1999). The marginals are known (`x_0` normal; `x_1` has mean 0 and variance
 * `2 b² a⁴ + 1 = 9`), the shape is not Gaussian: a normal approximation at the mode (the Laplace fit) is a poor description of it
 * and a mean-field guide cannot represent the dependence, while NUTS follows the ridge with a reduced effective sample size.
 *
 * @concept posterior-validation
 * @index dataset Haario's banana: a curved correlated 2-D ridge; analytic marginals; Laplace and mean field are wrong.
 * @example
 * ```ts
 * import { banana } from 'omai-engine/probprog/models'
 * banana.truth().marginals['x1']?.sd // => 3
 * ```
 */
export const banana: BenchmarkModel = defineBenchmark({
  name: 'banana',
  title: "Haario's banana",
  description:
    'A curved two-dimensional ridge: x0 ~ N(0, 4), x1 | x0 ~ N(0.5 (x0^2 - 4), 1). Marginals are known; the shape is far from Gaussian.',
  tags: ['geometry', 'curved', 'density', 'analytic', 'two-dimensional', 'correlated'],
  references: [
    'Haario, Saksman and Tamminen (1999), Adaptive proposal distribution for random walk Metropolis algorithm, Computational Statistics 14',
    'the `bananaDensity` of omai-engine/datasets (the same density, whose log density the tests compare with)',
  ],
  target: 'density',
  definition: defineModel('banana', (m) => {
    const x0 = m.latent('x0', d.normal(0, BANANA.a))
    m.latent('x1', d.normal(ops.mul(BANANA.b, ops.sub(x0.square(), BANANA.a * BANANA.a)), 1))
  }),
  generate: () => noData({ ...BANANA }),
  truth: () => {
    const z = toFlat(drawNormal(foldIn(key0, 2), [ORACLE_DRAWS, 2]))
    const x1 = new Float64Array(ORACLE_DRAWS)
    for (let i = 0; i < ORACLE_DRAWS; i++) {
      const x0 = BANANA.a * (z[2 * i] ?? 0)
      x1[i] = BANANA.b * (x0 * x0 - BANANA.a * BANANA.a) + (z[2 * i + 1] ?? 0)
    }
    return {
      kind: 'analytic',
      source: `x0 is N(0, ${BANANA.a ** 2}) exactly; x1 has mean 0 and variance 2 b² a⁴ + 1 = ${2 * BANANA.b ** 2 * BANANA.a ** 4 + 1}, its CDF the empirical one of ${ORACLE_DRAWS} iid draws of the exact sampler`,
      marginals: {
        x0: mixtureOracle('x0', [{ weight: 1, mean: 0, sd: BANANA.a }]),
        x1: samplesOracle('x1', x1, { mean: 0, sd: Math.sqrt(2 * BANANA.b ** 2 * BANANA.a ** 4 + 1) }),
      },
    }
  },
  plans: (scale) => [
    nutsPlan(scale, {
      warmup: 800,
      draws: 1500,
      expect: 'report',
      note: 'follows the ridge but with divergences in the arms and an effective sample size of a few percent of the draws: the spread of x1 is off by about a fifth',
    }),
    laplacePlan(scale, {
      expect: 'differ',
      note: 'a normal at the mode (0, -2): a straight ellipse for a curved ridge, wrong in the spread of x1 and in its skew',
    }),
    sviPlan(scale, {
      expect: 'differ',
      note: 'a mean-field normal: no dependence between x0 and x1, so it cannot represent the ridge',
    }),
  ],
})

const LOW = -3
const HIGH = 3
const MODES = [
  [LOW, LOW],
  [HIGH, HIGH],
]
const MODE_SD = 1

/**
 * An equal mixture of two well separated Gaussians in two dimensions, at `(-3, -3)` and `(3, 3)` with unit sd, written as
 * a discrete component and the coordinates given it (the component is summed out by NUTS). The marginal of each coordinate
 * is the even mixture of `N(-3, 1)` and `N(3, 1)`. A single chain stays in one mode (the barrier between them is about
 * eight standard deviations of the target's own scale), so NUTS reports a unimodal posterior, two chains in different modes
 * disagree (R-hat is poor), and nothing here recovers the mode weights; a Laplace fit and a mean-field guide are mode-seeking,
 * they choose one. A tempered sampler would recover the weights (the method belongs to a later release).
 *
 * @concept posterior-validation
 * @index dataset Two equal Gaussian modes in 2-D: the multimodal target for mode weights and R-hat.
 * @example
 * ```ts
 * import { bimodal } from 'omai-engine/probprog/models'
 * bimodal.truth().marginals['x[0]']?.cdf(0) // => 0.5
 * ```
 */
export const bimodal: BenchmarkModel = defineBenchmark({
  name: 'bimodal',
  title: 'Two Gaussian modes',
  description:
    'An equal mixture of two unit Gaussians at (-3, -3) and (3, 3): the target whose mode weights a single chain cannot find.',
  tags: ['multimodal', 'geometry', 'density', 'analytic', 'two-dimensional', 'discrete-latent'],
  references: [
    'Neal (1996), Sampling from multimodal distributions using tempered transitions, Statistics and Computing 6',
    'Earl and Deem (2005), Parallel tempering: theory, applications and new perspectives',
  ],
  target: 'density',
  definition: defineModel('bimodal', (m) => {
    const coord = m.range('coord', { size: 2 })
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { doc: 'the mode: 0 for (-3, -3), 1 for (3, 3)' })
    // Both coordinates share the label: the centre is z (HIGH - LOW) + LOW in each (the modes lie on the diagonal).
    m.latent('x', d.normal(z.mul(HIGH - LOW).add(LOW), MODE_SD), { dims: [coord] })
  }),
  generate: () => noData({ modes: MODES, sd: MODE_SD }),
  truth: () => ({
    kind: 'analytic',
    source: 'each coordinate is the even mixture of N(-3, 1) and N(3, 1): a closed-form normal mixture CDF',
    marginals: Object.fromEntries(
      [0, 1].map((i) => [
        `x[${i}]`,
        mixtureOracle(`x[${i}]`, [
          { weight: 1, mean: MODES[0]?.[i] ?? 0, sd: MODE_SD },
          { weight: 1, mean: MODES[1]?.[i] ?? 0, sd: MODE_SD },
        ]),
      ]),
    ),
  }),
  plans: (scale) => [
    nutsPlan(scale, {
      expect: 'report',
      note: 'one mode per chain: the weights are those of the chains (R-hat is poor), not one half each',
    }),
    gibbsPlan(scale, {
      expect: 'differ',
      note: 'the component given the point, the point given the component: stuck in the mode it starts in',
    }),
    importancePlan(scale, {
      expect: 'agree',
      draws: 20000,
      note: 'no observed node: the prior is the target and the weights are all one, so the draws are exact',
    }),
  ],
})
