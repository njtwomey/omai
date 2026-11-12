/**
 * Simulation-based calibration (Talts, Betancourt, Simpson, Vehtari and Gelman 2018): draw parameters from the prior
 * and data from the model, fit the data with the method under test, and rank the true parameter among `L` posterior
 * draws. If the method is exact (and the draws independent) the rank is uniform on {0, ..., L} whatever the prior and
 * the model, so the ranks of many replications must look uniform: a chi-square test on the histogram, and a test on
 * the maximum difference of the empirical CDF of the ranks from the uniform CDF (the statistic of the ECDF-difference
 * plot), whose null distribution is simulated from the exact rank distribution with a fixed key. A method that is too
 * confident gives a U-shaped histogram, one that is too wide a hump, a biased one a slope.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, randint } from '../../random/index.ts'
import { chisquare } from '../../stats/tests/index.ts'
import { arange, div, floor, full, mul, take, tensor, toFlat, zeros } from '../../tensor/index.ts'

/** One simulated dataset: its data and the true values of the quantities to rank. */
export interface SbcSimulation<D> {
  readonly data: D
  readonly truth: Readonly<Record<string, number>>
}

/** Options of {@link sbc}. */
export interface SbcOptions<D> {
  /** Draw the parameters from the prior and the data from the model. */
  readonly simulate: (key: Key) => SbcSimulation<D>
  /** Fit the data and return posterior draws of each quantity (at least `numRanks`, thinned evenly to `numRanks`). */
  readonly fit: (data: D, key: Key) => Readonly<Record<string, ArrayLike<number>>>
  /** The number of replications. */
  readonly replications: number
  /** The number of draws `L` each rank is taken among: ranks are in 0..L. */
  readonly numRanks: number
  readonly key: Key
  /** Histogram bins (default the largest divisor of L + 1 not above 20). */
  readonly bins?: number
  /** Level of the two tests together, Bonferroni-adjusted over the quantities (default 0.01). */
  readonly alpha?: number
  /** Simulations of the null distribution of the ECDF statistic (default 4000). */
  readonly nullSimulations?: number
}

/** What the ranks of one quantity say. */
export interface SbcQuantity {
  readonly name: string
  /** The rank of the truth among the draws, one per replication. */
  readonly ranks: readonly number[]
  readonly numRanks: number
  /** Counts of the ranks in equal bins of the values 0..L. */
  readonly histogram: readonly number[]
  readonly chiSquare: { readonly statistic: number; readonly pValue: number }
  /** The largest difference of the ECDF of the ranks from the uniform CDF, and its simulated p-value. */
  readonly ecdf: { readonly statistic: number; readonly pValue: number }
  /** The smaller of the two p-values, doubled (capped at one). */
  readonly pValue: number
  /** A reading of the histogram when the quantity fails: too confident, too wide, or biased up or down. */
  readonly diagnosis: string
  readonly passes: boolean
}

/** The result of {@link sbc}. */
export interface SbcResult {
  readonly replications: number
  readonly numRanks: number
  readonly alpha: number
  readonly quantities: readonly SbcQuantity[]
  /** Every quantity passes. */
  readonly passes: boolean
}

/**
 * The exact distribution of a rank among `L` draws when the method is exact: uniform on {0, ..., L}.
 *
 * @param numRanks - L.
 * @returns The probabilities of 0..L.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { rankDistribution } from 'omai-engine/probprog/validation'
 * rankDistribution(3) // => [0.25, 0.25, 0.25, 0.25]
 * ```
 */
export function rankDistribution(numRanks: number): number[] {
  return toFlat(full([numRanks + 1], 1 / (numRanks + 1)))
}

/**
 * The rank of a true value among draws: how many draws lie below it (ties count half, rounded down).
 *
 * @param truth - The true value.
 * @param draws - The posterior draws.
 * @returns The rank, from 0 to the number of draws.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { rankOf } from 'omai-engine/probprog/validation'
 * rankOf(2.5, [1, 2, 3, 4]) // => 2
 * ```
 */
export function rankOf(truth: number, draws: ArrayLike<number>): number {
  let below = 0
  let ties = 0
  for (let i = 0; i < draws.length; i++) {
    const d = draws[i] ?? Number.NaN
    if (d < truth) below += 1
    else if (d === truth) ties += 1
  }
  return below + Math.floor(ties / 2)
}

function thinTo(draws: ArrayLike<number>, n: number, name: string): number[] {
  if (draws.length < n)
    throw new OmaiError('value', `sbc: ${name} has ${draws.length} draws, fewer than numRanks = ${n}`)
  return toFlat(take(tensor(Array.from(draws)), floor(div(mul(arange(n), draws.length), n))))
}

function defaultBins(values: number): number {
  for (let b = Math.min(20, values); b >= 2; b--) if (values % b === 0) return b
  return 1
}

/** The maximum |ECDF - uniform CDF| of ranks on 0..L. */
function ecdfStatistic(ranks: readonly number[], numRanks: number): number {
  const counts = toFlat(zeros([numRanks + 1]))
  for (const r of ranks) counts[r] = (counts[r] ?? 0) + 1
  let acc = 0
  let d = 0
  for (let i = 0; i <= numRanks; i++) {
    acc += counts[i] ?? 0
    d = Math.max(d, Math.abs(acc / ranks.length - (i + 1) / (numRanks + 1)))
  }
  return d
}

/** The p-value of the ECDF statistic against `simulations` replications of uniform ranks, from a fixed key. */
function ecdfPValue(statistic: number, replications: number, numRanks: number, simulations: number): number {
  const root = makeKey(20_180_425)
  let atLeast = 0
  for (let s = 0; s < simulations; s++) {
    const r = Array.from(randint(foldIn(root, s), [replications], 0, numRanks + 1).data)
    if (ecdfStatistic(r, numRanks) >= statistic - 1e-12) atLeast += 1
  }
  return (1 + atLeast) / (1 + simulations)
}

function diagnose(ranks: readonly number[], numRanks: number): string {
  const u = ranks.map((r) => (r + 0.5) / (numRanks + 1))
  const m = u.reduce((a, b) => a + b, 0) / u.length
  const v = u.reduce((a, b) => a + (b - m) ** 2, 0) / u.length
  const sdMean = Math.sqrt(1 / 12 / u.length)
  if (Math.abs(m - 0.5) > 3 * sdMean && Math.abs(m - 0.5) > Math.abs(v - 1 / 12))
    return m > 0.5 ? 'truth ranks high: the posterior is biased low' : 'truth ranks low: the posterior is biased high'
  return v > 1 / 12 ? 'U-shaped: the posterior is too narrow or biased' : 'hump: the posterior is too wide'
}

/**
 * Simulation-based calibration of an inference method: for each replication draw the parameters and data
 * (`simulate`), fit the data (`fit`), and rank each quantity's true value among `numRanks` posterior draws; then test
 * the ranks for uniformity. Thin autocorrelated draws before ranking (the rank test assumes independent draws): a
 * `fit` returning more than `numRanks` draws is thinned evenly.
 *
 * @param options - The simulator, the fit, the replications, `numRanks`, the key and the levels.
 * @returns The ranks, histograms, p-values and the verdict.
 * @throws OmaiError with code `value` for fewer than 10 replications, or a fit with too few draws or a missing quantity.
 * @concept posterior-validation
 * @index function Simulation-based calibration of an inference method with the exact rank distribution.
 * @example
 * ```ts
 * import { sbc } from 'omai-engine/probprog/validation'
 * import { foldIn, key } from 'omai-engine/random'
 * import { normal } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * // m ~ N(0, 1), y ~ N(m, 1): the posterior is N(y / 2, 1 / 2). The exact sampler is calibrated.
 * const result = sbc({
 *   key: key(1), replications: 100, numRanks: 19,
 *   simulate: (k) => { const m = item(normal(foldIn(k, 0), [])); return { data: m + item(normal(foldIn(k, 1), [])), truth: { m } } },
 *   fit: (y, k) => ({ m: Array.from(normal(k, [19], { loc: y / 2, scale: Math.SQRT1_2 }).data) }),
 * })
 * result.passes // => true
 * ```
 */
export function sbc<D>(options: SbcOptions<D>): SbcResult {
  const { replications, numRanks } = options
  if (!Number.isInteger(replications) || replications < 10)
    throw new OmaiError('value', 'sbc: at least 10 replications are needed')
  if (!Number.isInteger(numRanks) || numRanks < 1)
    throw new OmaiError('value', 'sbc: numRanks must be a positive integer')
  const ranks: Record<string, number[]> = {}
  for (let r = 0; r < replications; r++)
    for (const [name, rank] of Object.entries(sbcReplication(options, r))) ranks[name] = [...(ranks[name] ?? []), rank]
  return sbcSummary(ranks, options)
}

/**
 * One replication of simulation-based calibration: the parameters and data of replication `r` (from the key folded in
 * with `r`), the fit, and the rank of each true value among `numRanks` draws. {@link sbc} is `replications` of these
 * and {@link sbcSummary}; the one replication is the step of a run that shows the ranks as they arrive.
 *
 * @param options - The simulator, the fit, `numRanks` and the key.
 * @param r - The replication's index.
 * @returns The rank of each quantity's true value.
 * @throws OmaiError with code `value` for a fit with too few draws or a missing quantity.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { sbcReplication } from 'omai-engine/probprog/validation'
 * import { foldIn, key, normal } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const ranks = sbcReplication({
 *   key: key(1), numRanks: 19,
 *   simulate: (k) => { const m = item(normal(foldIn(k, 0), [])); return { data: m + item(normal(foldIn(k, 1), [])), truth: { m } } },
 *   fit: (y, k) => ({ m: Array.from(normal(k, [19], { loc: y / 2, scale: Math.SQRT1_2 }).data) }),
 * }, 0)
 * ranks['m'] !== undefined && ranks['m'] >= 0 && ranks['m'] <= 19 // => true
 * ```
 */
export function sbcReplication<D>(
  options: Pick<SbcOptions<D>, 'simulate' | 'fit' | 'numRanks' | 'key'>,
  r: number,
): Record<string, number> {
  const kr = foldIn(options.key, r)
  const sim = options.simulate(foldIn(kr, 0))
  const fit = options.fit(sim.data, foldIn(kr, 1))
  const out: Record<string, number> = {}
  for (const [name, truth] of Object.entries(sim.truth)) {
    const draws = fit[name]
    if (draws === undefined) throw new OmaiError('value', `sbc: the fit returned no draws of ${name}`)
    out[name] = rankOf(truth, thinTo(draws, options.numRanks, name))
  }
  return out
}

/**
 * The tests of the ranks gathered by {@link sbcReplication}: the histogram, the chi-square test, the ECDF statistic with its
 * simulated p-value, the diagnosis of a failure and the verdict, for any number of replications from ten.
 *
 * @param ranks - The ranks of each quantity, one per replication.
 * @param options - `numRanks` and the levels, bins and null simulations of {@link sbc}.
 * @returns The result.
 * @throws OmaiError with code `value` for fewer than 10 replications.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { sbcSummary } from 'omai-engine/probprog/validation'
 * import { arange, mod, toFlat } from 'omai-engine/tensor'
 * const ranks = toFlat(mod(arange(40), 20))
 * sbcSummary({ m: ranks }, { numRanks: 19, nullSimulations: 200 }).passes // => true
 * ```
 */
export function sbcSummary(
  ranks: Readonly<Record<string, readonly number[]>>,
  options: Pick<SbcOptions<unknown>, 'numRanks' | 'bins' | 'alpha' | 'nullSimulations'>,
): SbcResult {
  const { numRanks } = options
  const names = Object.keys(ranks)
  const replications = ranks[names[0] ?? '']?.length ?? 0
  if (replications < 10) throw new OmaiError('value', 'sbc: at least 10 replications are needed')
  const alpha = options.alpha ?? 0.01
  const bins = options.bins ?? defaultBins(numRanks + 1)
  const quantities = names.map((name): SbcQuantity => {
    const rs = [...(ranks[name] ?? [])]
    const histogram = toFlat(zeros([bins]))
    const width = (numRanks + 1) / bins
    for (const r of rs)
      histogram[Math.min(bins - 1, Math.floor(r / width))] =
        (histogram[Math.min(bins - 1, Math.floor(r / width))] ?? 0) + 1
    const chi = chisquare(tensor(histogram))
    const statistic = ecdfStatistic(rs, numRanks)
    const ecdfP = ecdfPValue(statistic, replications, numRanks, options.nullSimulations ?? 4000)
    const pValue = Math.min(1, 2 * Math.min(chi.pValue, ecdfP))
    const passes = Math.min(chi.pValue, ecdfP) >= alpha / (2 * names.length)
    return {
      name,
      ranks: rs,
      numRanks,
      histogram,
      chiSquare: { statistic: chi.statistic, pValue: chi.pValue },
      ecdf: { statistic, pValue: ecdfP },
      pValue,
      diagnosis: passes ? 'uniform' : diagnose(rs, numRanks),
      passes,
    }
  })
  return { replications, numRanks, alpha, quantities, passes: quantities.every((q) => q.passes) }
}

/**
 * The calibration result as a markdown table (one row per quantity, with the rank histogram).
 *
 * @param result - The result of {@link sbc}.
 * @returns The markdown.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { sbcMarkdown } from 'omai-engine/probprog/validation'
 * sbcMarkdown({ replications: 0, numRanks: 1, alpha: 0.01, quantities: [], passes: true }).split('\n').length // => 2
 * ```
 */
export function sbcMarkdown(result: SbcResult): string {
  const head = ['quantity', 'chi-square p', 'ECDF p', 'verdict', 'rank histogram']
  const lines = [`| ${head.join(' | ')} |`, `| ${head.map(() => '---').join(' | ')} |`]
  for (const q of result.quantities)
    lines.push(
      `| ${q.name} | ${q.chiSquare.pValue.toPrecision(2)} | ${q.ecdf.pValue.toPrecision(2)} | ${q.passes ? 'passes' : `FAILS (${q.diagnosis})`} | ${q.histogram.join(' ')} |`,
    )
  return lines.join('\n')
}
