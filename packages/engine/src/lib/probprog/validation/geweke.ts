/**
 * Geweke's joint distribution test (Geweke 2004) of a posterior sampler. Two simulators of the joint distribution of
 * parameters and data are compared: the marginal-conditional one draws parameters from the prior and data from the
 * model, independently every time; the successive-conditional one alternates the transition under test (which leaves
 * the posterior of the parameters given the data invariant) with a fresh draw of the data given the parameters. If the
 * transition is correct both have the same joint distribution, so every statistic of parameters and data has the same
 * mean under both. A bug in the transition shows as a significant difference, with no need for a known posterior.
 */

import { OmaiError } from '../../base/index.ts'
import { ndtr } from '../../numerics/special/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { ess } from '../../sampling/index.ts'
import { item, tensor } from '../../tensor/index.ts'

/** Options of {@link gewekeTest}. */
export interface GewekeOptions<S, D> {
  /** Draw the parameters from the prior and the data from the model given them (independent draws). */
  readonly simulate: (key: Key) => { readonly state: S; readonly data: D }
  /** Draw new data from the model given the parameters. */
  readonly resimulateData: (state: S, key: Key) => D
  /** The transition under test: new parameters given the old and the data, leaving the posterior invariant. */
  readonly transition: (state: S, data: D, key: Key) => S
  /** Statistics of the parameters and the data to compare (include squares to test the spread). */
  readonly statistics: Readonly<Record<string, (state: S, data: D) => number>>
  /** The number of independent draws of the marginal-conditional simulator. */
  readonly numSamples: number
  /** The number of iterations of the successive-conditional chain (after `burnIn`). */
  readonly numIterations: number
  readonly burnIn?: number
  readonly key: Key
}

/** The comparison of one statistic. */
export interface GewekeStatistic {
  readonly name: string
  /** The mean under the marginal-conditional simulator. */
  readonly meanMarginal: number
  /** The mean under the successive-conditional simulator. */
  readonly meanSuccessive: number
  /** The effective sample size of the successive-conditional chain. */
  readonly ess: number
  /** (difference of means) / standard error. */
  readonly z: number
  /** The two-sided normal p-value of z. */
  readonly pValue: number
}

/** The result of {@link gewekeTest}. */
export interface GewekeResult {
  readonly statistics: readonly GewekeStatistic[]
  /** The smallest p-value against `alpha / number of statistics` (Bonferroni). */
  readonly passes: boolean
  readonly alpha: number
}

/**
 * Run Geweke's joint distribution test of a transition operator.
 *
 * @param options - The two simulators, the transition, the statistics and the sizes.
 * @param alpha - The level, Bonferroni-adjusted over the statistics (default 0.01).
 * @returns The comparison of every statistic and the verdict.
 * @throws OmaiError with code `value` for fewer than 100 samples or iterations.
 * @concept posterior-validation
 * @index function Geweke joint-distribution test of a posterior transition operator.
 * @example
 * ```ts
 * import { gewekeTest } from 'omai-engine/probprog/validation'
 * import { key, normal } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * // m ~ N(0, 1), y ~ N(m, 1); the exact transition draws m from N(y / 2, 1 / 2).
 * const z = (k: Parameters<typeof normal>[0]): number => item(normal(k, []))
 * const result = gewekeTest({
 *   key: key(0), numSamples: 2000, numIterations: 2000,
 *   simulate: (k) => { const m = z(k); return { state: m, data: m + z(key(1)) } },
 *   resimulateData: (m, k) => m + z(k),
 *   transition: (_m, y, k) => y / 2 + Math.SQRT1_2 * z(k),
 *   statistics: { m: (m) => m, y: (_m, y) => y },
 * })
 * result.statistics.length // => 2
 * ```
 */
export function gewekeTest<S, D>(options: GewekeOptions<S, D>, alpha = 0.01): GewekeResult {
  if (options.numSamples < 100 || options.numIterations < 100)
    throw new OmaiError('value', 'gewekeTest: at least 100 samples and iterations are needed')
  const names = Object.keys(options.statistics)
  const marginal = names.map(() => new Float64Array(options.numSamples))
  for (let i = 0; i < options.numSamples; i++) {
    const { state, data } = options.simulate(foldIn(foldIn(options.key, 0), i))
    names.forEach((n, j) => {
      ;(marginal[j] as Float64Array)[i] = (options.statistics[n] as (s: S, d: D) => number)(state, data)
    })
  }
  const total = options.numIterations
  const burn = options.burnIn ?? 100
  const chain = names.map(() => new Float64Array(total))
  const start = options.simulate(foldIn(foldIn(options.key, 1), 0))
  let state = start.state
  let data = start.data
  for (let t = -burn; t < total; t++) {
    const k = foldIn(foldIn(options.key, 2), t + burn)
    state = options.transition(state, data, foldIn(k, 0))
    data = options.resimulateData(state, foldIn(k, 1))
    if (t < 0) continue
    names.forEach((n, j) => {
      ;(chain[j] as Float64Array)[t] = (options.statistics[n] as (s: S, d: D) => number)(state, data)
    })
  }
  const stats = names.map((name, j): GewekeStatistic => {
    const a = marginal[j] as Float64Array
    const b = chain[j] as Float64Array
    const mean = (x: Float64Array): number => x.reduce((s, v) => s + v, 0) / x.length
    const variance = (x: Float64Array, m: number): number => x.reduce((s, v) => s + (v - m) ** 2, 0) / (x.length - 1)
    const ma = mean(a)
    const mb = mean(b)
    const raw = ess(tensor([Array.from(b)]))
    const e = Number.isFinite(raw) ? Math.max(1, Math.min(total, raw)) : 1
    const se = Math.sqrt(variance(a, ma) / a.length + variance(b, mb) / e)
    const z = se === 0 ? 0 : (mb - ma) / se
    return { name, meanMarginal: ma, meanSuccessive: mb, ess: e, z, pValue: 2 * item(ndtr(-Math.abs(z))) }
  })
  return { statistics: stats, passes: stats.every((s) => s.pValue >= alpha / stats.length), alpha }
}
