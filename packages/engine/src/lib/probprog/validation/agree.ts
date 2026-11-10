/**
 * The agreement harness: compare any set of posterior approximations with an oracle (a closed form, a grid, a long
 * reference run) or with each other, per quantity, and return a table record: the mean's error in posterior standard
 * deviations and in Monte Carlo standard errors, the ratio of standard deviations, the Kolmogorov distance and the
 * first Wasserstein distance (from `stats/tests`) to the oracle's CDF, and the KL divergence of the oracle from the
 * approximation's histogram over the oracle's equal-mass bins. Which methods must agree and which are expected to
 * differ is a statement about this table, made by the caller.
 */

import { OmaiError } from '../../base/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import { wassersteinDistance } from '../../stats/tests/index.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import type { Posterior } from '../posterior.ts'
import { type MarginalOracle, weightedOracle } from './oracle.ts'

/** One approximation to compare: weighted draws of each scalar quantity. */
export interface Approximation {
  /** The method's name in the table. */
  readonly name: string
  /** Draws by quantity name (`mu`, `theta[1]`); all the same length. */
  readonly draws: Readonly<Record<string, ArrayLike<number>>>
  /** Weights of the draws (importance weights; default equal). */
  readonly weights?: ArrayLike<number>
  /** Effective sample size by quantity (default: Kish's from the weights, else the number of draws). */
  readonly ess?: Readonly<Record<string, number>>
}

/** Options of {@link fromPosterior}. */
export interface FromPosteriorOptions {
  /** The key for posteriors without stored draws (exact, variational): draws are made with `sample`. */
  readonly key?: Key
  /** How many draws to make for those (default 4000). */
  readonly numDraws?: number
  /** Only these sites (default all). */
  readonly sites?: readonly string[]
}

/**
 * An approximation from a {@link Posterior}: its draws (or `numDraws` draws of its sampler) split into scalar
 * quantities named `site` or `site[i]`, with the effective sample size of Markov chains from the posterior's own
 * summary and Kish's size for weighted draws.
 *
 * @param name - The method's name.
 * @param posterior - The posterior.
 * @param options - Key and number of draws for sampler-based posteriors, and the sites to keep.
 * @returns The approximation.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { observe, sample } from 'omai-engine/probprog'
 * import { conjugateInference } from 'omai-engine/probprog'
 * import { fromPosterior } from 'omai-engine/probprog/validation'
 * import { normal } from 'omai-engine/dists'
 * const model = () => { const m = sample('m', normal(0, 1)); observe('y', normal(m, 1), 1.2) }
 * Object.keys(fromPosterior('exact', conjugateInference(model, undefined)).draws) // => ['m']
 * ```
 */
export function fromPosterior(name: string, posterior: Posterior, options: FromPosteriorOptions = {}): Approximation {
  const stored = posterior.draws
  const n = options.numDraws ?? 4000
  const draws = stored ?? posterior.sample(options.key ?? makeKey(0), n)
  const rows = new Map(posterior.summary().map((r) => [r.name, r]))
  const out: Record<string, Float64Array> = {}
  const ess: Record<string, number> = {}
  for (const [site, value] of Object.entries(draws)) {
    if (options.sites !== undefined && !options.sites.includes(site)) continue
    const rows0 = value.shape[0] ?? 0
    const size = rows0 === 0 ? 0 : value.data.length / rows0
    for (let j = 0; j < size; j++) {
      const q = size === 1 ? site : `${site}[${j}]`
      out[q] = Float64Array.from({ length: rows0 }, (_, i) => value.data[i * size + j] ?? Number.NaN)
      const e = rows.get(q)?.ess
      if (e !== undefined) ess[q] = e
    }
  }
  const weights =
    posterior.logWeights === undefined ? undefined : Float64Array.from(toFlat(posterior.logWeights), Math.exp)
  return { name, draws: out, ess, ...(weights === undefined ? {} : { weights }) }
}

/** One row of the agreement table: one method, one quantity. */
export interface AgreementRow {
  readonly method: string
  readonly quantity: string
  readonly mean: number
  readonly sd: number
  /** (mean - oracle mean) / oracle sd: the error in posterior standard deviations. */
  readonly meanError: number
  /** (mean - oracle mean) / (sd / sqrt(ess)): the error in Monte Carlo standard errors. */
  readonly meanZ: number
  /** sd / oracle sd. */
  readonly sdRatio: number
  /** The Kolmogorov distance sup |F_n - F| of the weighted draws from the oracle's CDF. */
  readonly ks: number
  /** The first Wasserstein distance to the oracle in oracle standard deviations. */
  readonly wasserstein: number
  /** KL(oracle || approximation) with the approximation as a histogram on the oracle's equal-mass bins. */
  readonly kl: number
  readonly ess: number
  readonly numDraws: number
}

/** The agreement table. */
export interface AgreementTable {
  /** What the approximations were compared with. */
  readonly oracle: string
  readonly quantities: readonly string[]
  readonly rows: readonly AgreementRow[]
}

/** Options of {@link agree}. */
export interface AgreeOptions {
  /** The oracles by quantity name (a closed form or a grid). Without them, `reference` plays that part. */
  readonly oracle?: Readonly<Record<string, MarginalOracle>>
  /** The name of the approximation that is the reference when there is no oracle (default the first). */
  readonly reference?: string
  /** Quantities to compare (default every oracle's, or every one the approximations share). */
  readonly quantities?: readonly string[]
  /** Equal-mass bins of the KL (default 20). */
  readonly bins?: number
  /** The oracle's quantile points used for the Wasserstein distance (default 2000). */
  readonly quantilePoints?: number
}

function sortedWeighted(
  draws: ArrayLike<number>,
  weights: ArrayLike<number> | undefined,
): { x: number[]; w: number[] } {
  const n = draws.length
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (draws[a] ?? 0) - (draws[b] ?? 0))
  const total = weights === undefined ? n : Array.from(weights).reduce((a, b) => a + b, 0)
  return {
    x: order.map((i) => draws[i] ?? 0),
    w: order.map((i) => (weights === undefined ? 1 : (weights[i] ?? 0)) / total),
  }
}

function kolmogorov(x: readonly number[], w: readonly number[], cdf: (v: number) => number): number {
  let acc = 0
  let d = 0
  for (let i = 0; i < x.length; i++) {
    const F = cdf(x[i] ?? 0)
    d = Math.max(d, Math.abs(acc - F), Math.abs(acc + (w[i] ?? 0) - F))
    acc += w[i] ?? 0
  }
  return d
}

function binnedKl(x: readonly number[], w: readonly number[], oracle: MarginalOracle, bins: number): number {
  const edges = Array.from({ length: bins - 1 }, (_, b) => oracle.quantile((b + 1) / bins))
  const mass = new Array<number>(bins).fill(0)
  let bin = 0
  for (let i = 0; i < x.length; i++) {
    while (bin < bins - 1 && (x[i] ?? 0) > (edges[bin] ?? Number.POSITIVE_INFINITY)) bin += 1
    mass[bin] = (mass[bin] ?? 0) + (w[i] ?? 0)
  }
  const n = x.length
  let kl = 0
  for (const m of mass) {
    const q = (m * n + 0.5) / (n + 0.5 * bins)
    kl += (1 / bins) * Math.log(1 / bins / q)
  }
  return kl
}

/**
 * Compare approximations with an oracle, or with a reference approximation, quantity by quantity.
 *
 * @param approximations - The methods to compare.
 * @param options - The oracles (or a reference method), the quantities and the resolution of the distances.
 * @returns The agreement table.
 * @throws OmaiError with code `value` when a quantity has no oracle and no reference draws, or an approximation lacks it.
 * @concept posterior-validation
 * @index function Compare posterior approximations with an oracle or each other: moments, KS, Wasserstein, KL.
 * @example
 * ```ts
 * import { agree, distributionOracle } from 'omai-engine/probprog/validation'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const exact = normal(0.6, Math.sqrt(0.5))
 * const draws = Array.from({ length: 2000 }, (_, i) => item(exact.sample(key(i), [])))
 * const table = agree([{ name: 'iid', draws: { m: draws } }], { oracle: { m: distributionOracle('m', exact) } })
 * table.rows[0]?.ks !== undefined && (table.rows[0]?.ks ?? 1) < 0.05 // => true
 * ```
 */
export function agree(approximations: readonly Approximation[], options: AgreeOptions = {}): AgreementTable {
  if (approximations.length === 0) throw new OmaiError('value', 'agree: no approximations to compare')
  const bins = options.bins ?? 20
  const points = options.quantilePoints ?? 2000
  const reference =
    options.oracle === undefined
      ? (approximations.find((a) => a.name === options.reference) ?? approximations[0])
      : undefined
  const names =
    options.quantities ??
    (options.oracle === undefined ? Object.keys(reference?.draws ?? {}) : Object.keys(options.oracle))
  const rows: AgreementRow[] = []
  for (const q of names) {
    let oracle = options.oracle?.[q]
    if (oracle === undefined) {
      const draws = reference?.draws[q]
      if (draws === undefined) throw new OmaiError('value', `agree: no oracle and no reference draws for ${q}`)
      oracle = weightedOracle(q, draws, reference?.weights)
    }
    const targets = Float64Array.from({ length: points }, (_, i) => oracle.quantile((i + 0.5) / points))
    for (const a of approximations) {
      if (options.oracle === undefined && a === reference) continue
      const draws = a.draws[q]
      if (draws === undefined) throw new OmaiError('value', `agree: ${a.name} has no draws of ${q}`)
      const { x, w } = sortedWeighted(draws, a.weights)
      const mean = x.reduce((s, v, i) => s + v * (w[i] ?? 0), 0)
      const sd = Math.sqrt(x.reduce((s, v, i) => s + (w[i] ?? 0) * (v - mean) ** 2, 0))
      const kish = 1 / w.reduce((s, v) => s + v * v, 0)
      const ess = a.ess?.[q] ?? kish
      rows.push({
        method: a.name,
        quantity: q,
        mean,
        sd,
        meanError: (mean - oracle.mean) / oracle.sd,
        meanZ: (mean - oracle.mean) / (sd / Math.sqrt(ess)),
        sdRatio: sd / oracle.sd,
        ks: kolmogorov(x, w, oracle.cdf),
        wasserstein: wassersteinDistance(tensor(x), tensor(Array.from(targets)), { uWeights: tensor(w) }) / oracle.sd,
        kl: binnedKl(x, w, oracle, bins),
        ess,
        numDraws: x.length,
      })
    }
  }
  return {
    oracle: options.oracle === undefined ? `reference run ${reference?.name ?? ''}` : 'oracle',
    quantities: names,
    rows,
  }
}

/** Thresholds of {@link agreementWithin}. */
export interface AgreementTolerances {
  /** Largest |mean error| in posterior sds (default 0.1). */
  readonly meanError?: number
  /** Largest |z| of the mean in Monte Carlo standard errors (default 4). */
  readonly meanZ?: number
  /** Allowed range of the sd ratio (default 0.9 to 1.1). */
  readonly sdRatio?: readonly [number, number]
  /** Largest Kolmogorov distance (default 0.1). */
  readonly ks?: number
}

/**
 * The rows of a table that are outside the tolerances: a method that must agree has none.
 *
 * @param table - The agreement table.
 * @param tolerances - The thresholds.
 * @returns The offending rows, each with the measures that failed.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { agreementWithin } from 'omai-engine/probprog/validation'
 * const row = { method: 'a', quantity: 'x', mean: 0, sd: 1, meanError: 0, meanZ: 0, sdRatio: 1, ks: 0.01, wasserstein: 0, kl: 0, ess: 100, numDraws: 100 }
 * agreementWithin({ oracle: 'oracle', quantities: ['x'], rows: [row] }).length // => 0
 * ```
 */
export function agreementWithin(
  table: AgreementTable,
  tolerances: AgreementTolerances = {},
): { readonly row: AgreementRow; readonly failed: readonly string[] }[] {
  const [lo, hi] = tolerances.sdRatio ?? [0.9, 1.1]
  const out: { row: AgreementRow; failed: string[] }[] = []
  for (const row of table.rows) {
    const failed: string[] = []
    if (!(Math.abs(row.meanError) <= (tolerances.meanError ?? 0.1))) failed.push('meanError')
    if (!(Math.abs(row.meanZ) <= (tolerances.meanZ ?? 4))) failed.push('meanZ')
    if (!(row.sdRatio >= lo && row.sdRatio <= hi)) failed.push('sdRatio')
    if (!(row.ks <= (tolerances.ks ?? 0.1))) failed.push('ks')
    if (failed.length > 0) out.push({ row, failed })
  }
  return out
}

const fmt = (v: number): string => (Number.isFinite(v) ? Number(v.toPrecision(3)).toString() : String(v))

/**
 * The agreement table as a markdown table.
 *
 * @param table - The agreement table.
 * @returns The markdown.
 * @concept posterior-validation
 * @example
 * ```ts
 * import { agreementMarkdown } from 'omai-engine/probprog/validation'
 * agreementMarkdown({ oracle: 'oracle', quantities: [], rows: [] }).split('\n')[0] // => '| method | quantity | mean | sd | mean error (sd) | mean z | sd ratio | KS | W1 (sd) | KL | ESS |'
 * ```
 */
export function agreementMarkdown(table: AgreementTable): string {
  const head = [
    'method',
    'quantity',
    'mean',
    'sd',
    'mean error (sd)',
    'mean z',
    'sd ratio',
    'KS',
    'W1 (sd)',
    'KL',
    'ESS',
  ]
  const lines = [`| ${head.join(' | ')} |`, `| ${head.map(() => '---').join(' | ')} |`]
  for (const r of table.rows)
    lines.push(
      `| ${[r.method, r.quantity, fmt(r.mean), fmt(r.sd), fmt(r.meanError), fmt(r.meanZ), fmt(r.sdRatio), fmt(r.ks), fmt(r.wasserstein), fmt(r.kl), fmt(r.ess)].join(' | ')} |`,
    )
  return lines.join('\n')
}
