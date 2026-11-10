/**
 * The benchmark models added after the first suite, as a list of their own: `additionalBenchmarkModels` returns them in the
 * order of their campaigns, so the periodic validation can run them after the campaigns that already have tables (a
 * campaign's random key follows from its position, and appending leaves every earlier table as it was). The suite
 * (`benchmarkModels`) holds the first models followed by these.
 */

import type { BenchmarkModel } from './benchmark.ts'
import { clutterProblem } from './clutter.ts'
import { gaussianProcessNonCentred, gaussianProcessRegression } from './gaussian-process.ts'
import { itemResponseScoring } from './irt.ts'
import { bayesianNeuralNetwork } from './neural-network.ts'
import { stickBreakingMixture } from './nonparametric.ts'
import { gaussianHmm, stochasticVolatility } from './series.ts'
import { ldaLite } from './topics.ts'

/**
 * The benchmark models added after the first suite: recurrences written as scans (stochastic volatility, a hidden Markov
 * model) and the rest of the plan's list.
 *
 * @returns The models, a fresh array.
 * @concept model-spec
 * @index table The later benchmark models: scans, Gaussian processes, LDA, a neural network, IRT, clutter.
 * @example
 * ```ts
 * import { additionalBenchmarkModels } from 'omai-engine/probprog/models'
 * additionalBenchmarkModels().map((m) => m.name).slice(0, 2) // => ['stochastic-volatility', 'gaussian-hmm']
 * ```
 */
export function additionalBenchmarkModels(): BenchmarkModel[] {
  return [
    stochasticVolatility,
    gaussianHmm,
    gaussianProcessRegression,
    gaussianProcessNonCentred,
    ldaLite,
    bayesianNeuralNetwork,
    itemResponseScoring,
    stickBreakingMixture,
    clutterProblem,
  ]
}
