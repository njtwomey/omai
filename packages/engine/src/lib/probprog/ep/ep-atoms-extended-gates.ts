/**
 * The extended twins of the gate atoms: the clutter problem, a gate between two random values, a mixture with an index into
 * random components, and spike and slab, each with the numbers the basic atom fixes now unknown (the width of every component,
 * the precision of the clutter, the weight of the switch, the share of active coefficients, the scale of the slab) with priors,
 * and the data simulated from a stated truth. The gate of Minka and Winn carries it: each branch is a Gaussian with a gamma
 * precision of its own, and the switch has a beta (or Dirichlet) weight. The references are the exact posterior by quadrature
 * where the unknowns are few (the clutter problem, the sparse sequence) and a Gibbs sampler for the mixtures; the pages
 * show what EP recovers of the truth, and where it does not (components that overlap are read as one).
 */

import { foldIn, key, normal, uniform } from '../../random/index.ts'
import { dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { type AtomProblem, type AtomSupport, defineAtom, type EpAtom, type ExactMarginal } from './ep-atoms.ts'
import { simulationKnobs } from './ep-atoms-extended-conjugate.ts'
import { clutterPosterior, mixtureGibbs, outlierMixturePosterior, samplesMarginal } from './ep-atoms-gibbs.ts'
import { normalPdf } from './ep-atoms-quad.ts'

/** The knobs every extended atom has (the seed and the number of observations), with this atom's most observations. */
const learnedKnobs = (n: number, max: number): Record<string, ReturnType<typeof int>> => ({
  ...simulationKnobs(n),
  n: int(8, max, { default: n, label: 'observations' }),
})

const probability = (p: number): ExactMarginal => ({ mean: p, sd: Math.sqrt(p * (1 - p)), pdf: () => 0 })

const sig = (v: number, digits = 3): number => Number(v.toPrecision(digits))

// ---------------------------------------------------------------------------------------------------------------------
// Clutter with the parameters of the clutter and of the inliers unknown.

/** The priors of the extended clutter problem, stated on the page and in the code. */
const CLUTTER_PRIORS = {
  centre: 5,
  inlier: { shape: 2, rate: 1 },
  clutter: { shape: 2, rate: 20 },
  weight: { a: 2, b: 5 },
} as const

function clutterLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueTheta, trueInlierSd] = [configNumber(k, 'trueTheta'), configNumber(k, 'trueInlierSd')]
  const [trueClutterSd, trueWeight] = [configNumber(k, 'trueClutterSd'), configNumber(k, 'trueWeight')]
  // region clutter-learned
  // the ground truth the data are drawn from: the model does not know it
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])))
  const e = Array.from(toFlat(normal(foldIn(key(seed), 1), [n])))
  const isClutter = u.map((v) => v < trueWeight)
  const y = e.map((v, i) => (isClutter[i] ? trueClutterSd * v : trueTheta + trueInlierSd * v))
  const definition = defineModel('clutter-learned', (m) => {
    const row = m.range('i')
    // the centre, the precision of the inliers and of the clutter, the share of clutter
    const theta = m.latent('theta', d.normal(0, 5))
    const inlier = m.latent('inlier', d.gamma(2, 1))
    const clutter = m.latent('clutter', d.gamma(2, 20))
    const share = m.latent('share', d.beta(2, 5))
    // is observation i clutter?
    const z = m.latent('z', d.bernoulli({ probs: share }), { dims: [row] })
    // each branch of the gate has its own centre and its own precision
    const centre = ops.where(z, 0, theta)
    const precision = ops.where(z, clutter, inlier)
    const sd = ops.div(1, ops.sqrt(precision))
    m.observe('y', d.normal(centre, sd), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  const post = clutterPosterior(y, CLUTTER_PRIORS)
  const at = y.map((_, i) => i + 1)
  const band = (c: number, w: number): number[] => [c + w, c + w]
  const x = [0, n + 1]
  const [inSd, outSd] = [trueInlierSd, trueClutterSd]
  return {
    definition,
    data: { y },
    equations: [
      `truth: centre ${trueTheta}, inlier sd ${trueInlierSd} (precision ${sig(1 / inSd ** 2)}), clutter sd ${trueClutterSd} (precision ${sig(1 / outSd ** 2)}), clutter share ${trueWeight}; ${n} observations, seed ${seed}`,
      'theta ~ N(0, 5²)   (the centre)',
      'inlier ~ Gamma(2, 1)   (precision of the inliers: mean 2, sd 1.4)',
      'clutter ~ Gamma(2, 20)   (precision of the clutter: mean 0.1, a standard deviation of about 3: wider than the inliers)',
      'share ~ Beta(2, 5)   (the share of clutter: a minority, mean 0.29)',
      'z_i ~ Bernoulli(share)   (z_i = 1: observation i is clutter)',
      'y_i ~ z_i ? N(0, 1 / clutter) : N(theta, 1 / inlier)   (a gate per observation; each branch has its own precision)',
    ],
    quantities: [
      { name: 'theta', label: 'theta (the centre)', support: 'real', truth: trueTheta },
      { name: 'inlier', label: 'inlier precision', support: 'positive', truth: 1 / inSd ** 2 },
      { name: 'clutter', label: 'clutter precision', support: 'positive', truth: 1 / outSd ** 2 },
      { name: 'share', label: 'share of clutter', support: 'unit', truth: trueWeight },
      ...at.map((i) => ({
        name: `z[${i - 1}]`,
        label: `observation ${i} is clutter`,
        support: 'probability' as const,
        hidden: true,
      })),
    ],
    exact: {
      theta: post.centre,
      inlier: post.inlier,
      clutter: post.clutter,
      share: post.weight,
      ...Object.fromEntries(post.clutterProbability.map((p, i) => [`z[${i}]`, probability(p)])),
    },
    referenceSource: 'quadrature over the centre, the two precisions and the share, the switches summed out',
    figure: {
      caption:
        'The simulated observations, coloured by what EP believes of each (clutter, a draw from the wide component around 0, or an inlier, a detection of the centre). Green: the truth (the centre, the inlier band and the clutter band at two sds); thick and dashed: what EP learned.',
      xLabel: 'observation',
      yLabel: 'value',
      xLimits: x as [number, number],
      points: [
        {
          name: 'observations',
          x: at,
          y,
          group: isClutter.map((c) => (c ? 1 : 0)),
          groupNames: { 0: 'inlier', 1: 'clutter' },
          groupFrom: (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
        },
      ],
      curves: [
        { name: 'true centre', colour: 'tab2', x, y: band(trueTheta, 0) },
        { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, 2 * inSd) },
        { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, -2 * inSd) },
        { name: 'true clutter band', colour: 'tab1', x, y: band(0, 2 * outSd) },
        { name: 'true clutter band', colour: 'tab1', x, y: band(0, -2 * outSd) },
      ],
      answer: (m) => {
        const centre = m['theta'] ?? Number.NaN
        const w = 2 / Math.sqrt(m['inlier'] ?? Number.NaN)
        const o = 2 / Math.sqrt(m['clutter'] ?? Number.NaN)
        return [
          { name: 'theta', x, y: band(centre, 0) },
          { name: 'theta + 2 inlier sd', x, y: band(centre, w) },
          { name: 'theta - 2 inlier sd', x, y: band(centre, -w) },
          { name: 'clutter + 2 sd', x, y: band(0, o) },
          { name: 'clutter - 2 sd', x, y: band(0, -o) },
        ]
      },
    },
  }
}

/**
 * Minka's clutter problem with everything unknown: the centre, the precision of the inliers, the precision of the clutter (a gamma
 * that keeps it wider than the inliers) and the share of clutter (a beta), learned from data simulated from a stated truth. Each
 * observation is a gate between a Gaussian at the centre with the inlier precision and a Gaussian at zero with the clutter
 * precision, so EP reports the probability that each observation is clutter as well as the parameters.
 *
 * @index estimator EP atom: clutter with centre, both precisions and share learned.
 * @example
 * ```ts
 * import { clutterLearnedAtom } from 'omai-engine/probprog/ep'
 * clutterLearnedAtom.extends // => 'clutter'
 * ```
 */
export const clutterLearnedAtom: EpAtom = defineAtom({
  name: 'clutter-learned',
  title: 'Clutter: parameters learned',
  extends: 'clutter',
  recovery: {
    seeds: 12,
    coverage: 0.6,
    zrms: 1.8,
    reason:
      'EP matches the exact posterior closely, so over seeds drawn from the truth its intervals hold it about nine times in ten (40 seeds: 0.88 to 1.0 per parameter); 12 seeds, to keep the test quick, leave a binomial spread of about 0.1',
  },
  what: 'The clutter problem of the basic atom with the numbers it fixes made unknown: the width of the inliers, the width of the clutter and the share of clutter all have priors (the clutter kept wider than the inliers), and the data are simulated from a stated truth. EP, through a gate per observation whose branches each have a gamma precision, recovers all four parameters and says which observations are clutter.',
  factors: [
    'prior (gaussian, gamma, beta)',
    'bernoulli (a switch with a beta probability)',
    'select gate per observation (gaussian-gamma-precision branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: {
    meanZ: 0.15,
    sdRatio: [0.65, 1.2],
    ks: 0.12,
    reason:
      'every observation is a two-way gate and EP matches the moments of each, carrying one Gaussian, two gammas and a beta for the four unknowns; against the exact posterior by quadrature the parameters agree to a few hundredths of a posterior sd (spreads within 2%) and the responsibilities to 0.01 (the spread of a nearly certain responsibility, a coin, differs in the third decimal, which the lower bound allows)',
  },
  knobs: dict({
    ...learnedKnobs(50, 80),
    trueTheta: real(-3, 3, { default: 1.5, step: 0.1, label: 'true centre' }),
    trueInlierSd: real(0.2, 1.5, { default: 0.5, step: 0.05, label: 'true inlier sd' }),
    trueClutterSd: real(1.5, 8, { default: 4, step: 0.5, label: 'true clutter sd' }),
    trueWeight: real(0.05, 0.5, { default: 0.25, step: 0.01, label: 'true share of clutter' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  problem: clutterLearnedProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Mixtures with unknown components.

/** The true mixture the data are drawn from: weights, means and sds of each component. */
interface TrueMixture {
  readonly weights: readonly number[]
  readonly means: readonly number[]
  readonly sds: readonly number[]
}

const mixtureDensity =
  (t: TrueMixture) =>
  (x: number): number =>
    t.weights.reduce((a, w, k) => a + w * normalPdf(x, t.means[k] as number, t.sds[k] as number), 0)

/** Draws from a mixture: which component, then the value. */
function simulateMixture(seed: number, n: number, t: TrueMixture): { y: number[]; from: number[] } {
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])))
  const e = Array.from(toFlat(normal(foldIn(key(seed), 1), [n])))
  const from = u.map((v) => {
    let acc = 0
    for (let k = 0; k < t.weights.length - 1; k++) {
      acc += t.weights[k] as number
      if (v < acc) return k
    }
    return t.weights.length - 1
  })
  return { y: e.map((v, i) => (t.means[from[i] as number] as number) + (t.sds[from[i] as number] as number) * v), from }
}

/** A figure of mixture data: the points along the axis (coloured by component), the true density and the fitted one. */
function mixtureFigure(
  y: readonly number[],
  from: readonly number[],
  truth: TrueMixture,
  fitted: (m: Readonly<Record<string, number>>) => TrueMixture,
  names: Record<number, string>,
  groupFrom?: (m: Readonly<Record<string, number>>) => readonly number[],
): NonNullable<AtomProblem['figure']> {
  const lo = Math.min(...y, ...truth.means.map((m, k) => m - 3 * (truth.sds[k] as number)))
  const hi = Math.max(...y, ...truth.means.map((m, k) => m + 3 * (truth.sds[k] as number)))
  const xs = Array.from({ length: 160 }, (_, i) => lo + ((hi - lo) * i) / 159)
  const density = mixtureDensity(truth)
  const top = Math.max(...xs.map(density))
  return {
    caption:
      'The simulated observations along the axis (coloured by the component each was drawn from), the true mixture density (green) and the mixture EP learned from the means of its beliefs (thick, dashed). The densities are drawn in the units of the vertical axis, scaled to the picture.',
    xLabel: 'value',
    yLabel: 'density',
    xLimits: [lo, hi],
    yLimits: [-0.12 * top, 1.15 * top],
    points: [
      {
        name: 'observations',
        x: [...y],
        y: y.map((_, i) => -0.04 * top - 0.06 * top * ((i * 7) % 5) * 0.2),
        group: [...from],
        groupNames: names,
        ...(groupFrom === undefined ? {} : { groupFrom }),
      },
    ],
    curves: [{ name: 'true density', colour: 'tab2', x: xs, y: xs.map(density) }],
    answer: (m) => {
      const f = fitted(m)
      const g = mixtureDensity(f)
      return [{ name: 'EP mixture', x: xs, y: xs.map(g) }]
    },
  }
}

const sdOf = (precision: number | undefined): number => 1 / Math.sqrt(precision ?? Number.NaN)

/** The truths of a mixture's quantities, in the model's names (`mu[k]`, `tau[k]`, `weights[k]`). */
function mixtureQuantities(
  truth: TrueMixture,
  weightName: (k: number) => string,
  meanName: (k: number) => string,
  precisionName: (k: number) => string,
  label: (k: number) => string,
): AtomProblem['quantities'] {
  const K = truth.weights.length
  return [
    ...Array.from({ length: K }, (_, k) => ({
      name: meanName(k),
      label: `mean of ${label(k)}`,
      support: 'real' as AtomSupport,
      truth: truth.means[k] as number,
    })),
    ...Array.from({ length: K }, (_, k) => ({
      name: precisionName(k),
      label: `precision of ${label(k)}`,
      support: 'positive' as AtomSupport,
      truth: 1 / (truth.sds[k] as number) ** 2,
    })),
    ...Array.from({ length: K }, (_, k) => ({
      name: weightName(k),
      label: `weight of ${label(k)}`,
      support: 'unit' as AtomSupport,
      truth: truth.weights[k] as number,
    })),
  ]
}

/** The mixture a set of beliefs' means stand for. */
function fittedMixture(
  m: Readonly<Record<string, number>>,
  K: number,
  weightName: (k: number) => string,
  meanName: (k: number) => string,
  precisionName: (k: number) => string,
): TrueMixture {
  const ks = Array.from({ length: K }, (_, k) => k)
  return {
    weights: ks.map((k) => m[weightName(k)] ?? Number.NaN),
    means: ks.map((k) => m[meanName(k)] ?? Number.NaN),
    sds: ks.map((k) => sdOf(m[precisionName(k)])),
  }
}

function mixtureLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const K = configNumber(k, 'components')
  const gap = configNumber(k, 'separation')
  const truth: TrueMixture =
    K === 2
      ? { weights: [0.6, 0.4], means: [-gap / 2, gap / 2], sds: [0.8, 1.2] }
      : { weights: [0.4, 0.35, 0.25], means: [-gap, 0, gap], sds: [0.8, 1, 1.2] }
  const { y, from } = simulateMixture(seed, n, truth)
  const alpha = Array.from({ length: K }, () => 2)
  // region mixture-components-learned
  const definition = defineModel('mixture-components-learned', (m) => {
    const row = m.range('i')
    const comp = m.range('component', { size: K })
    // the means (in increasing order, which names the components), their precisions, the weights
    const mu = m.latent('mu', d.normal(0, 4), { dims: [comp], constraint: { kind: 'ordered' } })
    const tau = m.latent('tau', d.gamma(2, 2), { dims: [comp] })
    const weights = m.latent('weights', d.dirichlet(alpha))
    // which component each observation comes from
    const z = m.latent('z', d.categorical({ probs: weights }), { dims: [row] })
    const sd = ops.div(1, ops.sqrt(tau.at(z)))
    m.observe('y', d.normal(mu.at(z), sd), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  const gibbs = mixtureGibbs({
    x: y,
    means: Array.from({ length: K }, () => ({ mean: 0, sd: 4 })),
    precision: { shape: 2, rate: 2 },
    weights: alpha,
    seed: seed + 1000,
    burn: 300,
    draws: 1500,
    ordered: true,
  })
  const names = (kk: number): string => `component ${kk + 1}`
  const [wn, mn, pn] = [(c: number) => `weights[${c}]`, (c: number) => `mu[${c}]`, (c: number) => `tau[${c}]`]
  const exact: Record<string, ExactMarginal> = {}
  for (let c = 0; c < K; c++) {
    exact[mn(c)] = samplesMarginal(gibbs.mean[c] as number[])
    exact[pn(c)] = samplesMarginal(gibbs.precision[c] as number[])
    exact[wn(c)] = samplesMarginal(gibbs.weight[c] as number[])
  }
  return {
    definition,
    data: { y },
    bindOptions: { sizes: { component: K } },
    equations: [
      `truth: weights ${truth.weights.join(', ')}, means ${truth.means.map((v) => sig(v)).join(', ')}, sds ${truth.sds.join(', ')}; ${n} observations, seed ${seed}`,
      `mu_1 < ... < mu_${K}, each ~ N(0, 4²)   (an ordered vector: the first has the smallest mean)`,
      'tau_k ~ Gamma(2, 2)   (the precision of each component: mean 1)',
      `weights ~ Dirichlet(${alpha.join(', ')})`,
      'z_i ~ Categorical(weights)   (which component)',
      'y_i ~ N(mu[z_i], 1 / tau[z_i])   (the mean and the precision both picked by the index)',
    ],
    quantities: mixtureQuantities(truth, wn, mn, pn, names),
    exact,
    referenceSource: `a Gibbs sampler of the same model (1500 draws after 300), the means relabelled in increasing order`,
    figure: mixtureFigure(
      y,
      from,
      truth,
      (m) => fittedMixture(m, K, wn, mn, pn),
      Object.fromEntries(Array.from({ length: K }, (_, c) => [c, names(c)])),
    ),
  }
}

/**
 * A mixture of two or three Gaussians whose means, precisions and weights are all unknown (ordered means, gamma precisions, a
 * Dirichlet on the weights), learned from data simulated from a stated truth: the index gate carries the mean and the precision of
 * the chosen component together. The separation knob shows when EP recovers the components and when it settles on one reading of
 * overlapping ones.
 *
 * @index estimator EP atom: mixture with unknown means, widths and weights, learned.
 * @example
 * ```ts
 * import { mixtureComponentsLearnedAtom } from 'omai-engine/probprog/ep'
 * mixtureComponentsLearnedAtom.extends // => 'mixture-components'
 * ```
 */
export const mixtureComponentsLearnedAtom: EpAtom = defineAtom({
  name: 'mixture-components-learned',
  title: 'Random index into random components: parameters learned',
  extends: 'mixture-components',
  recovery: {
    seeds: 10,
    coverage: 0.6,
    zrms: 1.8,
    reason:
      'at the default separation of 4 the intervals hold the truth 0.75 to 1.0 of the time per parameter (20 seeds); the floor leaves the spread of 10 seeds, and the page says honestly that closer components (a separation of 2.5 or less) are covered less often',
  },
  what: 'The index gate of the basic atom, mu[z], with the means, the widths (gamma precisions) and the weights (a Dirichlet) all unknown, the means ordered so the components are named, and the data simulated from a stated truth. The separation knob moves the components together: EP recovers them when they are apart, and reports a confident reading of the overlap when they are not.',
  factors: [
    'prior (gaussian, gamma, dirichlet)',
    'ordered constraint (greater-than between neighbours)',
    'discrete (a switch with a Dirichlet probability)',
    'index gate (gaussian-gamma-precision branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.7, 1.2],
    ks: 0.15,
    reason:
      'for well-separated components EP is within a fifth of a posterior sd of a Gibbs sampler of the same model, a little overconfident (spreads about 0.9 of the sampler); where the components overlap the sampler is wider (spreads 0.5 to 0.7) because EP settles on one reading of the labels',
  },
  knobs: dict({
    ...learnedKnobs(45, 120),
    components: int(2, 3, { default: 2, label: 'components' }),
    separation: real(1, 6, { default: 4, step: 0.25, label: 'distance between the true means' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  problem: mixtureLearnedProblem,
})

function gateRandomLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const gap = configNumber(k, 'separation')
  const truth: TrueMixture = { weights: [0.65, 0.35], means: [-gap / 2, gap / 2], sds: [0.8, 1.2] }
  const { y, from } = simulateMixture(seed, n, truth)
  // region gate-random-learned
  const definition = defineModel('gate-random-learned', (m) => {
    const row = m.range('i')
    // the two components: where each is centred (a prior on each) and how wide (a precision)
    const low = m.latent('low', d.normal(-2, 2))
    const high = m.latent('high', d.normal(2, 2))
    const lowPrecision = m.latent('lowPrecision', d.gamma(2, 2))
    const highPrecision = m.latent('highPrecision', d.gamma(2, 2))
    // the weight of the second component, and the switch of each observation
    const share = m.latent('share', d.beta(2, 2))
    const z = m.latent('z', d.bernoulli({ probs: share }), { dims: [row] })
    // the switch selects the centre and the precision of its component together
    const centre = ops.where(z, high, low)
    const precision = ops.where(z, highPrecision, lowPrecision)
    const sd = ops.div(1, ops.sqrt(precision))
    m.observe('y', d.normal(centre, sd), m.data('y', { dims: [row], role: 'response' }))
  })
  // endregion
  const gibbs = mixtureGibbs({
    x: y,
    means: [
      { mean: -2, sd: 2 },
      { mean: 2, sd: 2 },
    ],
    precision: { shape: 2, rate: 2 },
    weights: [2, 2],
    seed: seed + 1000,
    burn: 300,
    draws: 1500,
    ordered: false,
  })
  const at = y.map((_, i) => i + 1)
  const exact: Record<string, ExactMarginal> = {
    low: samplesMarginal(gibbs.mean[0] as number[]),
    high: samplesMarginal(gibbs.mean[1] as number[]),
    lowPrecision: samplesMarginal(gibbs.precision[0] as number[]),
    highPrecision: samplesMarginal(gibbs.precision[1] as number[]),
    share: samplesMarginal(gibbs.weight[1] as number[]),
    ...Object.fromEntries(gibbs.responsibility.map((r, i) => [`z[${i}]`, probability(r[1] as number)])),
  }
  const figure = mixtureFigure(
    y,
    from,
    truth,
    (m) => ({
      weights: [1 - (m['share'] ?? Number.NaN), m['share'] ?? Number.NaN],
      means: [m['low'] ?? Number.NaN, m['high'] ?? Number.NaN],
      sds: [sdOf(m['lowPrecision']), sdOf(m['highPrecision'])],
    }),
    { 0: 'low component', 1: 'high component' },
    (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
  )
  return {
    definition,
    data: { y },
    equations: [
      `truth: weights ${truth.weights.join(', ')}, means ${truth.means.map((v) => sig(v)).join(', ')}, sds ${truth.sds.join(', ')}; ${n} observations, seed ${seed}`,
      'low ~ N(-2, 2²),  high ~ N(2, 2²)   (the two components, told apart by their priors)',
      'lowPrecision, highPrecision ~ Gamma(2, 2)   (their widths: mean 1)',
      'share ~ Beta(2, 2)   (the weight of the high component)',
      'z_i ~ Bernoulli(share)',
      'y_i ~ z_i ? N(high, 1 / highPrecision) : N(low, 1 / lowPrecision)   (a select gate: centre and precision together)',
    ],
    quantities: [
      { name: 'low', label: 'mean of the low component', support: 'real', truth: truth.means[0] as number },
      { name: 'high', label: 'mean of the high component', support: 'real', truth: truth.means[1] as number },
      {
        name: 'lowPrecision',
        label: 'precision of the low component',
        support: 'positive',
        truth: 1 / (truth.sds[0] as number) ** 2,
      },
      {
        name: 'highPrecision',
        label: 'precision of the high component',
        support: 'positive',
        truth: 1 / (truth.sds[1] as number) ** 2,
      },
      { name: 'share', label: 'weight of the high component', support: 'unit', truth: truth.weights[1] as number },
      ...at.map((i) => ({
        name: `z[${i - 1}]`,
        label: `observation ${i} is from the high component`,
        support: 'probability' as const,
        hidden: true,
      })),
    ],
    exact,
    referenceSource: 'a Gibbs sampler of the same model (1500 draws after 300)',
    figure,
  }
}

/**
 * A switch between two random Gaussians, written as a select, with everything about them unknown: the two means, their two
 * widths (gamma precisions) and the weight of the switch (a beta), learned from data simulated from a stated truth. The twin
 * of the gate between two random values and of the gate-weight atoms, with the components' widths learned too.
 *
 * @index estimator EP atom: select gate with unknown means, widths and weight, learned.
 * @example
 * ```ts
 * import { gateRandomLearnedAtom } from 'omai-engine/probprog/ep'
 * gateRandomLearnedAtom.extends // => 'gate-random'
 * ```
 */
export const gateRandomLearnedAtom: EpAtom = defineAtom({
  name: 'gate-random-learned',
  title: 'Gate between two random values: parameters learned',
  extends: 'gate-random',
  recovery: {
    seeds: 10,
    coverage: 0.6,
    zrms: 1.8,
    reason:
      'at the default separation of 4 the intervals hold the truth 0.75 to 1.0 of the time per parameter (20 seeds); the floor leaves the spread of 10 seeds, and the page says honestly that closer components are covered less often',
  },
  what: 'The gate between two random values of the basic atom, with the two means, the two widths and the weight of the switch all unknown (a prior on each), and the data simulated from a stated truth: the switch picks the centre and the precision of its component together, and EP also says which component each observation came from.',
  factors: [
    'prior (gaussian, gamma, beta)',
    'bernoulli (a switch with a beta probability)',
    'select gate per observation (gaussian-gamma-precision branches)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: {
    meanZ: 0.3,
    sdRatio: [0.6, 1.2],
    ks: 0.15,
    reason:
      'for well-separated components EP is within a fifth of a posterior sd of a Gibbs sampler of the same model, a little overconfident (the parameters within 12% of its spread, the responsibilities, which are coins, within 0.03); where the components overlap the sampler is wider because EP settles on one reading',
  },
  knobs: dict({
    ...learnedKnobs(40, 120),
    separation: real(1, 6, { default: 4, step: 0.25, label: 'distance between the true means' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  problem: gateRandomLearnedProblem,
})

// ---------------------------------------------------------------------------------------------------------------------
// Spike and slab with the inclusion probability and the slab width unknown.

/** The coefficients that are really active in the simulated sequence, and their values. */
const ACTIVE: readonly (readonly [number, number])[] = [
  [1, 3.2],
  [4, -2.4],
  [6, 4.1],
]
const SPIKE_SD = 0.1

function spikeSlabLearnedProblem(k: Config): AtomProblem {
  const [seed, p] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const noise = configNumber(k, 'noise')
  // region spike-and-slab-learned
  // the ground truth: three coefficients are active, the rest are zero
  const values = Array.from({ length: p }, (_, j) => ACTIVE.find(([at]) => at === j)?.[1] ?? 0)
  const e = Array.from(toFlat(normal(foldIn(key(seed), 0), [p])))
  const y = values.map((v, j) => v + noise * (e[j] ?? 0))
  const definition = defineModel('spike-and-slab-learned', (m) => {
    const j = m.range('j')
    // how often a coefficient is active, and the precision of the active ones
    const share = m.latent('share', d.beta(1.5, 3.5))
    const slabPrecision = m.latent('slabPrecision', d.gamma(2, 8))
    // is coefficient j in the slab?
    const z = m.latent('z', d.bernoulli({ probs: share }), { dims: [j] })
    const slabSd = ops.div(1, ops.sqrt(slabPrecision))
    const slab = m.latent('slab', d.normal(0, slabSd), { dims: [j] })
    const spike = m.latent('spike', d.normal(0, 0.1), { dims: [j] })
    // each coefficient is whichever component its switch selects
    m.observe('y', d.normal(ops.where(z, slab, spike), noise), m.data('y', { dims: [j] }))
  })
  // endregion
  const active = values.map((v) => v !== 0)
  const trueShare = active.filter(Boolean).length / p
  const rms = Math.sqrt(values.filter((v) => v !== 0).reduce((a, v) => a + v * v, 0) / active.filter(Boolean).length)
  // The exact posterior: given the share and the slab precision the coefficients are independent two-component mixtures.
  const [aS, bS, aT, bT] = [1.5, 3.5, 2, 8]
  const spikeVar = noise ** 2 + SPIKE_SD ** 2
  const ps = Array.from({ length: 300 }, (_, i) => (i + 0.5) / 300)
  const lts = Array.from(
    { length: 300 },
    (_, i) => Math.log(0.002) + ((Math.log(4) - Math.log(0.002)) * (i + 0.5)) / 300,
  )
  const ts = lts.map(Math.exp)
  const logs: number[] = []
  const pips: number[][] = []
  let top = Number.NEGATIVE_INFINITY
  for (const t of ts) {
    const slabVar = noise ** 2 + 1 / t
    for (const q of ps) {
      let lp = (aS - 1) * Math.log(q) + (bS - 1) * Math.log(1 - q) + aT * Math.log(t) - bT * t
      const pip: number[] = []
      for (const v of y) {
        const s1 = q * normalPdf(v, 0, Math.sqrt(slabVar))
        const s0 = (1 - q) * normalPdf(v, 0, Math.sqrt(spikeVar))
        lp += Math.log(s0 + s1)
        pip.push(s1 / (s0 + s1))
      }
      logs.push(lp)
      pips.push(pip)
      if (lp > top) top = lp
    }
  }
  const mass = logs.map((l) => Math.exp(l - top))
  const norm = mass.reduce((a, b) => a + b, 0)
  const shareMass = ps.map((_, i) => ts.reduce((a, __, jj) => a + (mass[jj * ps.length + i] as number), 0))
  const precisionMass = ts.map((_, jj) => ps.reduce((a, __, i) => a + (mass[jj * ps.length + i] as number), 0))
  const pipMean = y.map((_, j) => mass.reduce((a, m, c) => a + (m * ((pips[c] as number[])[j] as number)) / norm, 0))
  const shareMarginal = weightedMarginal(ps, shareMass)
  const precisionExact = weightedMarginal(ts, precisionMass)
  const at = y.map((_, j) => j + 1)
  const x = [0, p + 1]
  return {
    definition,
    data: { y },
    equations: [
      `truth: coefficients ${ACTIVE.map(([j, v]) => `${j + 1}: ${v}`).join(', ')} active (share ${sig(trueShare)}), the other ${p - ACTIVE.length} zero; noise sd ${noise} known; seed ${seed}`,
      'share ~ Beta(1.5, 3.5)   (the inclusion probability: mean 0.3)',
      'slabPrecision ~ Gamma(2, 8)   (the precision of the active coefficients: mean 0.25, a slab sd of about 2)',
      'z_j ~ Bernoulli(share)   (z_j = 1: coefficient j is in the slab)',
      'slab_j ~ N(0, 1 / slabPrecision),  spike_j ~ N(0, 0.1²)   (a wide and a narrow component)',
      `y_j ~ N(z_j ? slab_j : spike_j, ${noise}²), j = 1..${p}`,
    ],
    quantities: [
      { name: 'share', label: 'share of active coefficients', support: 'unit', truth: trueShare },
      {
        name: 'slabPrecision',
        label: 'slab precision (1 / rms of the active coefficients squared)',
        support: 'positive',
        truth: 1 / rms ** 2,
      },
      ...at.map((i) => ({
        name: `z[${i - 1}]`,
        label: `coefficient ${i} is active`,
        support: 'probability' as const,
      })),
    ],
    exact: {
      share: shareMarginal,
      slabPrecision: precisionExact,
      ...Object.fromEntries(pipMean.map((v, j) => [`z[${j}]`, probability(v)])),
    },
    referenceSource: 'the posterior of the share and the slab precision on a grid, the coefficients summed out',
    figure: {
      caption:
        'The observed value of each coefficient, coloured by whether it is truly active (marks) and by what EP believes (P(slab) above one half). The green lines are the truth at zero and the true values of the active ones.',
      xLabel: 'coefficient',
      yLabel: 'observed value',
      xLimits: x as [number, number],
      points: [
        {
          name: 'observed y',
          x: at,
          y: [...y],
          group: active.map((a) => (a ? 1 : 0)),
          groupNames: { 0: 'inactive', 1: 'active' },
          groupFrom: (m) => at.map((i) => ((m[`z[${i - 1}]`] ?? 0) > 0.5 ? 1 : 0)),
        },
        { name: 'true value', x: at, y: values },
      ],
      notes: [`The truly active coefficients are ${ACTIVE.map(([j]) => j + 1).join(', ')}.`],
    },
  }
}

/** A marginal from weights on points (not nested in a grid): moments and a piecewise-constant density. */
function weightedMarginal(points: readonly number[], weights: readonly number[]): ExactMarginal {
  const z = weights.reduce((a, b) => a + b, 0)
  const mean = points.reduce((a, v, i) => a + (v * (weights[i] as number)) / z, 0)
  const sd = Math.sqrt(points.reduce((a, v, i) => a + ((v - mean) ** 2 * (weights[i] as number)) / z, 0))
  const live = points.filter((_, i) => (weights[i] as number) > 1e-5 * Math.max(...weights))
  return {
    mean,
    sd,
    pdf: (xx) => {
      let nearest = 0
      for (let i = 1; i < points.length; i++)
        if (Math.abs((points[i] as number) - xx) < Math.abs((points[nearest] as number) - xx)) nearest = i
      const left = points[Math.max(0, nearest - 1)] as number
      const right = points[Math.min(points.length - 1, nearest + 1)] as number
      return (weights[nearest] as number) / z / ((right - left) / 2)
    },
    range: [Math.min(...live), Math.max(...live)],
  }
}

/**
 * A sparse sequence of coefficients, each a switch between a narrow spike and a wide slab, with the inclusion probability (a
 * beta) and the precision of the slab (a gamma) unknown, learned from data simulated from a stated set of active coefficients:
 * EP recovers which coefficients are active, how common they are and how large.
 *
 * @index estimator EP atom: spike and slab with unknown share and slab width, learned.
 * @example
 * ```ts
 * import { spikeSlabLearnedAtom } from 'omai-engine/probprog/ep'
 * spikeSlabLearnedAtom.extends // => 'spike-and-slab'
 * ```
 */
export const spikeSlabLearnedAtom: EpAtom = defineAtom({
  name: 'spike-and-slab-learned',
  title: 'Spike and slab: parameters learned',
  extends: 'spike-and-slab',
  recovery: {
    seeds: 30,
    coverage: 0.7,
    zrms: 1.6,
    reason:
      'with ten coefficients and three active, the share and the slab precision are learned from a handful of values and their posteriors are skewed, which EP carries as a beta and a gamma; 30 seeds leave a spread of about 0.06',
  },
  what: 'The spike and slab gate of the basic atom on a sequence of coefficients, with the inclusion probability (a beta) and the precision of the slab (a gamma) unknown, and the data simulated from a stated set of active coefficients: EP recovers which are active, how common they are and how wide the slab is.',
  factors: [
    'prior (beta, gamma)',
    'bernoulli (a switch with a beta probability)',
    'gaussian-gamma-precision (the slab)',
    'select gate (spike and slab components)',
    'gaussian likelihood',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: {
    meanZ: 0.2,
    sdRatio: [0.8, 1.2],
    ks: 0.1,
    reason:
      'the share and the slab precision are tied through the switches, which EP carries as a beta and a gamma; the posterior of the precision is skewed and EP is a little narrower than the exact one (0.89 of its sd)',
  },
  knobs: dict({
    seed: simulationKnobs(10)['seed'] as ReturnType<typeof int>,
    n: int(7, 14, { default: 10, label: 'coefficients' }),
    noise: real(0.2, 1.2, { default: 0.5, step: 0.05, label: 'noise sd (known)' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  problem: spikeSlabLearnedProblem,
})

function spikeSlabMarginalLearnedProblem(k: Config): AtomProblem {
  const base = spikeSlabLearnedProblem(k)
  const [seed, p] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const noise = configNumber(k, 'noise')
  // region spike-and-slab-marginal-learned
  // the ground truth: three coefficients are active, the rest are zero
  const coefficients = Array.from({ length: p }, (_, j) => ACTIVE.find(([at]) => at === j)?.[1] ?? 0)
  const draw = Array.from(toFlat(normal(foldIn(key(seed), 0), [p])))
  const observed = coefficients.map((v, j) => v + noise * (draw[j] ?? 0))
  const definition = defineModel('spike-and-slab-marginal-learned', (m) => {
    const j = m.range('j')
    // how often a coefficient is active, and the precision of the active ones
    const share = m.latent('share', d.beta(1.5, 3.5))
    const slabPrecision = m.latent('slabPrecision', d.gamma(2, 8))
    const slabSd = ops.div(1, ops.sqrt(slabPrecision))
    // each coefficient has the two-component prior, its switch summed out
    const w = m.latent('w', d.normalSpikeSlab(share, slabSd, 0.1), { dims: [j] })
    m.observe('y', d.normal(w, noise), m.data('y', { dims: [j] }))
  })
  // endregion
  return {
    ...base,
    definition,
    data: { y: observed },
    equations: base.equations
      .map((line) =>
        line.startsWith('z_j ~')
          ? 'w_j ~ share N(0, 1 / slabPrecision) + (1 - share) N(0, 0.1²)   (the switch summed out)'
          : line,
      )
      .filter((line) => !line.startsWith('slab_j ~')),
    quantities: base.quantities.filter((q) => q.support !== 'probability'),
    exact: Object.fromEntries(Object.entries(base.exact ?? {}).filter(([name]) => !name.startsWith('z['))),
  }
}

/**
 * The same sparse sequence with each coefficient's switch summed out (the library's spike and slab family) and the inclusion
 * probability and the slab precision unknown: the compiler gives each coefficient the gate of its spike and slab over a switch
 * that the posterior does not report, the same graph as the explicit-switch twin, so EP agrees with the exact posterior of the
 * share and the slab precision (the coefficients summed out) about as closely as the twin does.
 *
 * @index estimator EP atom: summed-out spike and slab, learned share and width.
 * @example
 * ```ts
 * import { spikeSlabMarginalLearnedAtom } from 'omai-engine/probprog/ep'
 * spikeSlabMarginalLearnedAtom.expected // => 'runs'
 * ```
 */
export const spikeSlabMarginalLearnedAtom: EpAtom = defineAtom({
  name: 'spike-and-slab-marginal-learned',
  title: 'Spike and slab, switch summed out: parameters learned',
  extends: 'spike-and-slab-marginal',
  recovery: {
    seeds: 30,
    coverage: 0.7,
    zrms: 1.6,
    reason:
      'the same posterior and the same floor as the explicit-switch twin (spike and slab: parameters learned): the share and the slab precision are learned from a handful of values',
  },
  what: 'The spike and slab prior with the switch summed out (the mixture form, no variable for the probability of the slab) on a sequence of coefficients, with the inclusion probability and the slab precision unknown, and the data simulated from a stated set of active coefficients: EP recovers how common the active coefficients are and how wide the slab is.',
  factors: [
    'prior (beta, gamma)',
    'bernoulli (a hidden switch with a beta probability)',
    'gate of spike and slab components (gaussian-gamma-precision slab)',
    'gaussian likelihood',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: {
    meanZ: 0.2,
    sdRatio: [0.8, 1.2],
    ks: 0.1,
    reason:
      'the same posterior as the explicit-switch twin: EP carries a beta and a gamma (measured: 0.006 and 0.003 posterior sds, spreads 0.98)',
  },
  knobs: dict({
    seed: simulationKnobs(10)['seed'] as ReturnType<typeof int>,
    n: int(7, 14, { default: 10, label: 'coefficients' }),
    noise: real(0.2, 1.2, { default: 0.5, step: 0.05, label: 'noise sd (known)' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  problem: spikeSlabMarginalLearnedProblem,
})

function clutterMixtureLearnedProblem(k: Config): AtomProblem {
  const [seed, n] = [configNumber(k, 'seed'), configNumber(k, 'n')]
  const [trueTheta, trueSd, trueWeight] = [
    configNumber(k, 'trueTheta'),
    configNumber(k, 'trueSd'),
    configNumber(k, 'trueWeight'),
  ]
  const width = 10
  // region clutter-mixture-learned
  // the ground truth: outliers are draws around the same centre, ten times wider
  const u = Array.from(toFlat(uniform(foldIn(key(seed), 0), [n])))
  const e = Array.from(toFlat(normal(foldIn(key(seed), 1), [n])))
  const isOutlier = u.map((v) => v < trueWeight)
  const y = e.map((v, i) => trueTheta + trueSd * v * (isOutlier[i] ? width : 1))
  const definition = defineModel('clutter-mixture-learned', (m) => {
    const row = m.range('i')
    // the centre, the precision of the inliers, the share of outliers
    const mu = m.latent('mu', d.normal(0, 5))
    const precision = m.latent('precision', d.gamma(2, 1))
    const share = m.latent('share', d.beta(2, 5))
    // each observation is an inlier or, ten times wider, an outlier: the switch summed out
    const sd = ops.div(1, ops.sqrt(precision))
    const observed = m.data('y', { dims: [row], role: 'response' })
    m.observe('y', d.normalOutliers(mu, sd, share, 10), observed)
  })
  // endregion
  const priors = { centre: 5, precision: { shape: 2, rate: 1 }, share: { a: 2, b: 5 }, width } as const
  const post = outlierMixturePosterior(y, priors)
  const at = y.map((_, i) => i + 1)
  const x = [0, n + 1]
  const band = (c: number, w: number): number[] => [c + w, c + w]
  return {
    definition,
    data: { y },
    equations: [
      `truth: centre ${trueTheta}, inlier sd ${trueSd} (precision ${sig(1 / trueSd ** 2)}), outlier share ${trueWeight}, outliers ${width} times wider; ${n} observations, seed ${seed}`,
      'mu ~ N(0, 5²)   (the centre)',
      'precision ~ Gamma(2, 1)   (of the inliers)',
      'share ~ Beta(2, 5)   (the share of outliers)',
      `y_i ~ (1 - share) N(mu, 1 / precision) + share N(mu, ${width}² / precision)   (the switch summed out)`,
    ],
    quantities: [
      { name: 'mu', label: 'mu (the centre)', support: 'real', truth: trueTheta },
      { name: 'precision', label: 'inlier precision', support: 'positive', truth: 1 / trueSd ** 2 },
      { name: 'share', label: 'share of outliers', support: 'unit', truth: trueWeight },
    ],
    exact: { mu: post.centre, precision: post.precision, share: post.share },
    referenceSource: 'quadrature over the centre, the precision and the share, the switches summed out',
    figure: {
      caption:
        'The simulated observations (outliers drawn ten times wider around the same centre, marked) and the centre with the inlier band at two sds: green, the truth; thick and dashed, what EP believes.',
      xLabel: 'observation',
      yLabel: 'value',
      xLimits: x as [number, number],
      points: [
        {
          name: 'observations',
          x: at,
          y,
          group: isOutlier.map((o) => (o ? 1 : 0)),
          groupNames: { 0: 'inlier', 1: 'outlier' },
        },
      ],
      curves: [
        { name: 'true centre', colour: 'tab2', x, y: band(trueTheta, 0) },
        { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, 2 * trueSd) },
        { name: 'true inlier band', colour: 'tab2', x, y: band(trueTheta, -2 * trueSd) },
      ],
      answer: (m) => {
        const centre = m['mu'] ?? Number.NaN
        const w = 2 / Math.sqrt(m['precision'] ?? Number.NaN)
        return [
          { name: 'mu', x, y: band(centre, 0) },
          { name: 'mu + 2 inlier sd', x, y: band(centre, w) },
          { name: 'mu - 2 inlier sd', x, y: band(centre, -w) },
        ]
      },
    },
  }
}

/**
 * The inlier and outlier mixture with the switches summed out (the library's `normalOutliers`) and the centre, the width of the
 * inliers and the share of outliers all unknown, learned from data simulated from a stated truth. The compiler reads each
 * observation as the gate of an inlier and an outlier branch over a switch the posterior does not report (the graph of the
 * explicit-switch twin, clutter: parameters learned), and EP agrees with the exact posterior (quadrature over the centre, the
 * precision and the share, the switches summed out).
 *
 * @index estimator EP atom: summed-out outlier mixture, learned width and share.
 * @example
 * ```ts
 * import { clutterMixtureLearnedAtom } from 'omai-engine/probprog/ep'
 * clutterMixtureLearnedAtom.expected // => 'runs'
 * ```
 */
export const clutterMixtureLearnedAtom: EpAtom = defineAtom({
  name: 'clutter-mixture-learned',
  title: 'Inlier and outlier mixture, switches summed out: parameters learned',
  extends: 'clutter-mixture',
  recovery: {
    seeds: 20,
    coverage: 0.7,
    zrms: 1.6,
    reason: 'the same floor as the explicit-switch twin (clutter: parameters learned)',
  },
  what: 'The inlier and outlier mixture likelihood of the basic atom with the centre, the width of the inliers and the share of outliers all unknown (priors), and the data simulated from a stated truth. The switches are summed out, so EP gives beliefs about the centre, the precision and the share, and no probability that an observation is an outlier.',
  factors: [
    'prior (gaussian, gamma, beta)',
    'bernoulli (a hidden switch per observation with a beta probability)',
    'gate of inlier and outlier branches (gaussian-gamma-precision)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: {
    meanZ: 0.5,
    sdRatio: [0.5, 1.6],
    ks: 0.3,
    reason:
      'the posterior can be lumpy and EP matches only its mean and variance with a Gaussian, a gamma and a beta (measured on the default data: within 0.02 posterior sds, spreads 0.96 to 1.0)',
  },
  knobs: dict({
    ...learnedKnobs(50, 80),
    trueTheta: real(-3, 3, { default: 1.5, step: 0.1, label: 'true centre' }),
    trueSd: real(0.2, 1.5, { default: 0.5, step: 0.05, label: 'true inlier sd' }),
    trueWeight: real(0.05, 0.5, { default: 0.25, step: 0.01, label: 'true share of outliers' }),
  }),
  ep: { damping: 0.5, tol: 1e-4 },
  problem: clutterMixtureLearnedProblem,
})
