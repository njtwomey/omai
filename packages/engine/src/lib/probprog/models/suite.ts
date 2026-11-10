/**
 * The benchmark suite as a list: `benchmarkModels` returns every model of the folder, in the order the documents use, and
 * `whereBenchmarks` selects by tags or text the way the dataset registry's `whereDatasets` does. Nothing registers on import;
 * an app, a worker's module or a showcase page asks for the list and filters it. Each model carries a title, a
 * description, tags, references, a data generator and its truth, which is what a showcase page needs to present it.
 */

import { OmaiError } from '../../base/index.ts'
import { additionalBenchmarkModels } from './additional.ts'
import type { BenchmarkModel } from './benchmark.ts'
import { betaBernoulli, gammaPoisson, normalNormal } from './conjugate.ts'
import { banana, bimodal, funnelCentred, funnelNonCentred } from './geometry.ts'
import { logisticRegression, negativeBinomialRegression, poissonRegression, probitRegression } from './glm.ts'
import { radon } from './hierarchical.ts'
import { gaussianMixture, gaussianMixtureOrdered } from './mixture.ts'
import { trueskillThreePlayers } from './ranking.ts'
import { eightSchoolsCentred, eightSchoolsNonCentred } from './schools.ts'
import { ar1StateSpace } from './state-space.ts'

/**
 * Every benchmark model of the suite: the conjugate trio, logistic, probit, Poisson and negative binomial regression, eight
 * schools in two parametrisations, the Gaussian mixture with and without label switching, the funnel in two parametrisations,
 * the banana, the two-mode target, the radon regression, a three-player TrueSkill and the AR(1) state space, then the models of
 * `additionalBenchmarkModels` (stochastic volatility, a Gaussian HMM, Gaussian process regression in two parametrisations, a
 * topic model, a Bayesian neural network, item response scoring, a stick-breaking mixture and the clutter problem).
 *
 * @returns The models, a fresh array.
 * @concept model-spec
 * @index table The benchmark models: specs with truth, generators, tags and method plans.
 * @example
 * ```ts
 * import { benchmarkModels } from 'omai-engine/probprog/models'
 * benchmarkModels().map((m) => m.name).slice(0, 3) // => ['beta-bernoulli', 'normal-normal', 'gamma-poisson']
 * ```
 */
export function benchmarkModels(): BenchmarkModel[] {
  return [
    betaBernoulli,
    normalNormal,
    gammaPoisson,
    logisticRegression,
    probitRegression,
    poissonRegression,
    negativeBinomialRegression,
    eightSchoolsCentred,
    eightSchoolsNonCentred,
    gaussianMixtureOrdered,
    gaussianMixture,
    funnelCentred,
    funnelNonCentred,
    banana,
    bimodal,
    radon,
    trueskillThreePlayers,
    ar1StateSpace,
    ...additionalBenchmarkModels(),
  ]
}

/** A query over the benchmark models. */
export interface BenchmarkQuery {
  /** Models that have every one of these tags. */
  readonly tags?: readonly string[]
  /** Models whose name, title, description or tags contain this text (case-insensitive). */
  readonly text?: string
}

/**
 * The benchmark models that match a query (all of the tags, and the text anywhere in the name, title, description or tags).
 *
 * @param models - The models to search.
 * @param query - The tags and text.
 * @returns The matching models, in order.
 * @concept model-spec
 * @example
 * ```ts
 * import { benchmarkModels, whereBenchmarks } from 'omai-engine/probprog/models'
 * whereBenchmarks(benchmarkModels(), { tags: ['funnel'] }).map((m) => m.name) // => ['eight-schools-centred', 'funnel-centred', 'funnel-non-centred']
 * ```
 */
export function whereBenchmarks(models: readonly BenchmarkModel[], query: BenchmarkQuery = {}): BenchmarkModel[] {
  const text = query.text?.toLowerCase()
  return models.filter(
    (m) =>
      (query.tags ?? []).every((t) => m.tags.includes(t)) &&
      (text === undefined ||
        [m.name, m.title, m.description, ...m.tags].some((field) => field.toLowerCase().includes(text))),
  )
}

/**
 * One benchmark model by name.
 *
 * @param name - The model's name.
 * @param models - The models to look in (default the suite).
 * @returns The model.
 * @throws OmaiError with code `value` listing the names that exist.
 * @concept model-spec
 * @example
 * ```ts
 * import { benchmarkModel } from 'omai-engine/probprog/models'
 * benchmarkModel('banana').title // => "Haario's banana"
 * ```
 */
export function benchmarkModel(name: string, models: readonly BenchmarkModel[] = benchmarkModels()): BenchmarkModel {
  const m = models.find((x) => x.name === name)
  if (m === undefined)
    throw new OmaiError('value', `no benchmark model called '${name}' (known: ${models.map((x) => x.name).join(', ')})`)
  return m
}
