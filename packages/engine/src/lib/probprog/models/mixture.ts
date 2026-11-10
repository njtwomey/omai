/**
 * The two-component Gaussian mixture with unknown means, known weights (one half each) and known unit noise: a finite
 * discrete latent per row (the component label) and the two means over a `component` dimension, gathered by the label with
 * `at`. NUTS sums the labels out (they are finite discrete latents), Gibbs samples them. Written with symmetric priors
 * the posterior has two modes that are each other's relabelling, which is the label switching problem: a raw
 * summary of a component's mean is bimodal and a single chain sees only one mode, while a label-switching-aware summary
 * (the sorted means, `mu(1) < mu(2)`) is the same in every mode. The ordered variant constrains `mu_0 < mu_1`, which
 * removes the second mode. The truth is a grid over the two means around one mode (the labels summed out exactly):
 * that mode is the ordered posterior, and the unconstrained posterior is its even mixture with the relabelling.
 * `gaussianMixtureModel` is the same idea for any number of components with unknown weights and a shared spread, for data of
 * unit scale (a real dataset standardised), with no closed-form truth.
 */

import { OmaiError } from '../../base/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, type ModelDefinition, toProgram } from '../model/index.ts'
import { gridPosterior } from '../validation/index.ts'
import {
  type BenchmarkData,
  type BenchmarkModel,
  type BenchmarkTruth,
  defineBenchmark,
  type TruthQuality,
} from './benchmark.ts'
import { gibbsPlan, laplacePlan, nutsPlan, sviPlan } from './plans.ts'
import { averageOracle, renamed } from './truth.ts'

const MEANS = [-2, 2]

function generateMixture(key: Key, size = 40): BenchmarkData {
  const u = toFlat(drawUniform(foldIn(key, 0), [size]))
  const e = toFlat(drawNormal(foldIn(key, 1), [size]))
  const label = Array.from(u, (v) => (v < 0.5 ? 0 : 1))
  return {
    data: { y: label.map((l, i) => (MEANS[l] ?? 0) + (e[i] ?? 0)) },
    options: { sizes: { component: 2 } },
    generating: { mu: MEANS, weights: [0.5, 0.5], sd: 1, label },
  }
}

/**
 * The sorted component means of a mixture's draws: `mu(1)` is the smaller of `mu[0]` and `mu[1]` in each draw and `mu(2)`
 * the larger. A relabelling of the components leaves them unchanged, so they are what a label-switching-aware comparison
 * reads.
 *
 * @param draws - Draws by quantity name, with `mu[0]` and `mu[1]` of the same length.
 * @returns The sorted pair, `mu(1)` and `mu(2)`.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { sortedComponentMeans } from 'omai-engine/probprog/models'
 * const s = sortedComponentMeans({ 'mu[0]': [2, -1], 'mu[1]': [-2, 3] })
 * [Array.from(s['mu(1)'] ?? []), Array.from(s['mu(2)'] ?? [])] // => [[-2, -1], [2, 3]]
 * ```
 */
export function sortedComponentMeans(draws: Readonly<Record<string, ArrayLike<number>>>): Record<string, Float64Array> {
  const a = draws['mu[0]']
  const b = draws['mu[1]']
  if (a === undefined || b === undefined) return {}
  const lo = new Float64Array(a.length)
  const hi = new Float64Array(a.length)
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? Number.NaN
    const y = b[i] ?? Number.NaN
    lo[i] = Math.min(x, y)
    hi[i] = Math.max(x, y)
  }
  return { 'mu(1)': lo, 'mu(2)': hi }
}

/** The options of {@link gaussianMixtureModel}. */
export interface GaussianMixtureOptions {
  /** The number of components, 1 or more. */
  readonly components: number
  /** Constrain the means to increase (`mu_0 < mu_1 < ...`), which removes the label switching. Default false. */
  readonly ordered?: boolean
  /** The standard deviation of the normal prior of each mean (default 2). */
  readonly meanScale?: number
}

/**
 * A one-dimensional Gaussian mixture of `components` normals with everything unknown: the weights (a flat Dirichlet), the
 * means (`N(0, meanScale^2)`, optionally ordered), one spread the components share (`LogNormal(-1, 0.75)`, a median of 0.37) and a discrete
 * component label per row, which the samplers sum out or sample. It is written for data of about unit scale, so
 * standardise the data first. Left unordered the posterior has `components!` modes that are relabellings of one another:
 * the label switching a chain, a sampler or a guide runs into.
 *
 * @param options - The number of components, whether to order the means and the prior scale of the means.
 * @returns The definition; bind it to `{ y: [...] }`.
 * @throws OmaiError with code `value` for fewer than one component.
 * @concept model-spec
 * @index function A K-component Gaussian mixture with unknown weights, means and a shared spread, optionally ordered.
 * @example
 * ```ts
 * import { gaussianMixtureModel } from 'omai-engine/probprog/models'
 * gaussianMixtureModel({ components: 3, ordered: true }).bind({ y: [-1, 0.2, 1.4, 0.3] }).sizes // => { row: 4, component: 3 }
 * ```
 */
export function gaussianMixtureModel(options: GaussianMixtureOptions): ModelDefinition {
  const { components, ordered = false, meanScale = 2 } = options
  if (!(Number.isInteger(components) && components >= 1))
    throw new OmaiError('value', 'gaussianMixtureModel: components is a whole number from 1')
  return defineModel(`gaussian-mixture-${components}${ordered ? '-ordered' : ''}`, (m) => {
    const row = m.range('row')
    const k = m.range('component', { size: components })
    const y = m.data('y', { dims: [row], role: 'response' })
    const weights = m.latent('weights', d.dirichlet(Array.from({ length: components }, () => 1)), {
      doc: 'the mixture weights',
    })
    const mu = m.latent('mu', d.normal(0, meanScale), {
      dims: [k],
      ...(ordered ? { constraint: { kind: 'ordered' as const } } : {}),
      doc: 'the component means',
    })
    const sigma = m.latent('sigma', d.logNormal(-1, 0.75), { doc: 'the spread every component shares' })
    const z = m.latent('z', d.categorical({ probs: weights }), { dims: [row], doc: 'the component of each row' })
    m.observe('y', d.normal(mu.at(z), sigma), y)
  })
}

const mixtureSpec = (name: string, ordered: boolean) =>
  defineModel(name, (m) => {
    const row = m.range('row')
    const k = m.range('component', { size: 2 })
    const y = m.data('y', { dims: [row], role: 'response' })
    const mu = m.latent('mu', d.normal(0, 3), {
      dims: [k],
      ...(ordered ? { constraint: { kind: 'ordered' as const } } : {}),
      doc: 'the component means',
    })
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row], doc: 'the component of each row' })
    m.observe('y', d.normal(mu.at(z), 1), y)
  })

const cache = new WeakMap<
  BenchmarkData,
  Map<
    TruthQuality,
    { low: BenchmarkTruth['marginals'][string]; high: BenchmarkTruth['marginals'][string]; truth: BenchmarkTruth }
  >
>()

/** One mode of the posterior by quadrature: the marginals of the smaller and the larger mean, and the evidence of that mode. */
function modeTruth(
  data: BenchmarkData,
  quality: TruthQuality,
): { low: BenchmarkTruth['marginals'][string]; high: BenchmarkTruth['marginals'][string]; truth: BenchmarkTruth } {
  const hit = cache.get(data)?.get(quality)
  if (hit !== undefined) return hit
  const spec = mixtureSpec('gaussian-mixture', false)
  const grid = gridPosterior(
    toProgram(spec.bind(data.data, data.options)),
    undefined,
    quality === 'fast' ? { points: 31, maxRefinements: 0 } : { points: 61, maxRefinements: 1 },
  )
  const a = grid.marginal('mu[0]')
  const b = grid.marginal('mu[1]')
  const [low, high] = a.mean <= b.mean ? [a, b] : [b, a]
  const truth: BenchmarkTruth = {
    kind: 'quadrature',
    source: `grid quadrature of the two means around one mode, the component labels summed out exactly (${grid.points} points per axis, evidence error ${grid.evidenceError.toExponential(1)}); that mode is the ordered posterior and the unconstrained one is its even mixture with the relabelling`,
    logEvidence: grid.logEvidence,
    marginals: {},
  }
  const out = { low, high, truth }
  const byQuality = cache.get(data) ?? new Map()
  byQuality.set(quality, out)
  cache.set(data, byQuality)
  return out
}

/**
 * The Gaussian mixture with the means ordered (`mu_0 < mu_1`): the label switching removed by a constraint. Every method
 * that is consistent agrees with the quadrature truth, and the raw means are the sorted ones.
 *
 * @concept posterior-validation
 * @index dataset Two-component Gaussian mixture, means ordered; discrete labels summed out; the truth is a 2-D grid.
 * @example
 * ```ts
 * import { gaussianMixtureOrdered } from 'omai-engine/probprog/models'
 * gaussianMixtureOrdered.bound().sizes // => { component: 2, row: 40 }
 * ```
 */
export const gaussianMixtureOrdered: BenchmarkModel = defineBenchmark({
  name: 'gaussian-mixture-ordered',
  title: 'Gaussian mixture, ordered means',
  description:
    'Two normal components with known weights and unit noise, unknown means constrained mu_0 < mu_1 and a discrete label per row: the mixture without label switching.',
  tags: ['mixture', 'discrete-latent', 'enumeration', 'ordered', 'quadrature', 'two-dimensional'],
  references: [
    'Betancourt (2017), Identifying Bayesian mixture models (the Stan case study)',
    'Stephens (2000), Dealing with label switching in mixture models, JRSS B 62',
    'Bishop (2006), Pattern Recognition and Machine Learning, chapter 9',
  ],
  definition: mixtureSpec('gaussian-mixture-ordered', true),
  generate: generateMixture,
  truth: (data, quality) => {
    const { low, high, truth } = modeTruth(data, quality)
    return { ...truth, marginals: { 'mu[0]': renamed(low, 'mu[0]'), 'mu[1]': renamed(high, 'mu[1]') } }
  },
  plans: (scale) => [
    nutsPlan(scale, {
      note: 'the labels are summed out and the ordering removes the second mode: agrees with the truth',
    }),
    gibbsPlan(scale, {
      draws: 1500,
      note: 'the means have no closed-form conditional under the ordering, so they are updated by slice steps on the ordered scale (the labels are exact): agrees with the truth',
    }),
    laplacePlan(scale, { note: 'a normal at the mode of the ordered density: close, the labels summed out' }),
    sviPlan(scale, {
      note: 'a mean-field normal on the unconstrained coordinates (the first mean and the log gap): close in the means, but mu[0] too narrow and mu[1], their sum, too wide',
    }),
  ],
})

/**
 * The Gaussian mixture with exchangeable means and the labels free to switch: the posterior has two modes that are each
 * other's relabelling. The truth is the even mixture of the two modes' marginals; the sorted means `mu(1)`, `mu(2)` are
 * the label-switching-aware summary every consistent method agrees on, while a single NUTS chain explores one mode, so its
 * raw `mu[0]` is unimodal where the truth is bimodal (the table reports it).
 *
 * @concept posterior-validation
 * @index dataset Two-component Gaussian mixture with label switching; the truth is the even two-mode mixture; sorted summaries.
 * @example
 * ```ts
 * import { gaussianMixture } from 'omai-engine/probprog/models'
 * Object.keys(gaussianMixture.truth(undefined, 'fast').marginals) // => ['mu[0]', 'mu[1]', 'mu(1)', 'mu(2)']
 * ```
 */
export const gaussianMixture: BenchmarkModel = defineBenchmark({
  name: 'gaussian-mixture',
  title: 'Gaussian mixture, label switching',
  description:
    'The two-component mixture with exchangeable means: a bimodal posterior (the relabelling), a multimodal target for the raw means and a unimodal one for the sorted means.',
  tags: ['mixture', 'discrete-latent', 'multimodal', 'label-switching', 'quadrature', 'two-dimensional'],
  references: [
    'Stephens (2000), Dealing with label switching in mixture models, JRSS B 62',
    'Jasra, Holmes and Stephens (2005), Markov chain Monte Carlo methods and the label switching problem in Bayesian mixture modeling, Statistical Science 20',
    'Betancourt (2017), Identifying Bayesian mixture models (the Stan case study)',
  ],
  definition: mixtureSpec('gaussian-mixture', false),
  generate: generateMixture,
  summarise: sortedComponentMeans,
  truth: (data, quality) => {
    const { low, high, truth } = modeTruth(data, quality)
    const half = (name: string): ReturnType<typeof averageOracle> =>
      averageOracle(name, [
        { weight: 1, oracle: low },
        { weight: 1, oracle: high },
      ])
    return {
      ...truth,
      marginals: {
        'mu[0]': half('mu[0]'),
        'mu[1]': half('mu[1]'),
        'mu(1)': renamed(low, 'mu(1)'),
        'mu(2)': renamed(high, 'mu(2)'),
      },
    }
  },
  plans: (scale) => [
    nutsPlan(scale, {
      quantities: ['mu(1)', 'mu(2)'],
      note: 'the sorted means are invariant to the relabelling: every chain agrees on them, whichever mode it started in',
    }),
    nutsPlan(scale, {
      expect: 'report',
      quantities: ['mu[0]', 'mu[1]'],
      note: 'the raw means: the truth is bimodal and chains are split between the modes (or all in one), so the draws are not the truth and R-hat is poor',
    }),
    gibbsPlan(scale, {
      expect: 'report',
      draws: 1500,
      quantities: ['mu(1)', 'mu(2)'],
      note: 'sorted means: agrees when the sweep does not stick',
    }),
    laplacePlan(scale, {
      quantities: ['mu(1)', 'mu(2)'],
      note: 'a normal at one mode: the sorted means are right, the raw means are one mode only',
    }),
  ],
})
