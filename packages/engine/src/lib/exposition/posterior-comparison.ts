/**
 * The exposition record of a comparison of posterior approximations (`posteriorComparison`): for each quantity the
 * oracle's distribution (a closed form, a grid or a long reference run, as its CDF on a grid) with each method's
 * distribution (equal-mass quantiles of its weighted draws, or a normal given by its mean and sd) so that a view lays
 * them over one another as densities, ECDFs or the difference of each ECDF from the oracle's, and the agreement table
 * (mean error in posterior sds, sd ratio, Kolmogorov and Wasserstein distances, KL) as a marked table whose cells
 * beyond tolerance are flagged. It is built from the validation kernel's `agree` table and oracles by their shape
 * alone (`AgreementTableLike`, `MarginalLike`), so a benchmark campaign is shown without the record depending on the
 * kernel; a variational run's final posterior against the truth (`ElboRun`) is the same overlay.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'
import { type CellMark, type MarkedTable, markedTable } from './tables.ts'

/** The marginal of one quantity by shape: the validation kernel's `MarginalOracle`. */
export interface MarginalLike {
  readonly name: string
  readonly mean: number
  readonly sd: number
  /** P(X <= x). */
  cdf(x: number): number
  /** The p-quantile, for p in (0, 1). */
  quantile(p: number): number
}

/** One approximation as weighted draws of each quantity: the validation kernel's `Approximation`. */
export interface DrawsLike {
  readonly name: string
  readonly draws: Readonly<Record<string, ArrayLike<number>>>
  readonly weights?: ArrayLike<number>
}

/** One approximation as a normal per quantity, as a variational method with a mean-field normal guide reports it. */
export interface NormalsLike {
  readonly name: string
  readonly normal: Readonly<Record<string, { readonly mean: number; readonly sd: number }>>
}

/** One row of the kernel's agreement table (`AgreementRow`). */
export interface AgreementRowLike {
  readonly method: string
  readonly quantity: string
  readonly mean: number
  readonly sd: number
  readonly meanError: number
  readonly meanZ: number
  readonly sdRatio: number
  readonly ks: number
  readonly wasserstein: number
  readonly kl: number
  readonly ess: number
  readonly numDraws: number
}

/** The kernel's agreement table (`AgreementTable`). */
export interface AgreementTableLike {
  readonly oracle: string
  readonly quantities: readonly string[]
  readonly rows: readonly AgreementRowLike[]
}

/** The distribution of a quantity under one method: equal-mass quantiles of its draws, or a normal. */
export interface ComparedMethod {
  readonly name: string
  readonly mean: number
  readonly sd: number
  /** The quantiles at the levels `(i + 0.5) / n` of the method's (weighted) draws, ascending: an equal-mass sample of its distribution. */
  readonly quantiles?: readonly number[]
  /** A normal approximation, drawn as its density. */
  readonly normal?: { readonly mean: number; readonly sd: number }
}

/** What the methods are compared with for one quantity: the oracle's CDF on a grid. */
export interface ComparedTruth {
  readonly mean: number
  readonly sd: number
  /** The grid, ascending, and the CDF on it: the density is its slope. */
  readonly x: readonly number[]
  readonly cdf: readonly number[]
}

/** One quantity: the oracle (when there is one) and every method's distribution. */
export interface ComparedQuantity {
  readonly name: string
  readonly truth?: ComparedTruth
  readonly methods: readonly ComparedMethod[]
}

/** One row of the agreement table, as the record keeps it (a measure the kernel could not compute is NaN). */
export type ComparisonRow = AgreementRowLike

/** How far a measure may be before its cell is flagged, as `agreementWithin` of the kernel takes them. */
export interface ComparisonTolerances {
  /** Largest |mean error| in posterior sds. */
  readonly meanError: number
  /** Largest |z| of the mean in Monte Carlo standard errors. */
  readonly meanZ: number
  /** The range of the sd ratio. */
  readonly sdRatio: readonly [number, number]
  /** Largest Kolmogorov distance. */
  readonly ks: number
}

/** The tolerances of the kernel's `agreementWithin`. */
export const DEFAULT_TOLERANCES: ComparisonTolerances = { meanError: 0.1, meanZ: 4, sdRatio: [0.9, 1.1], ks: 0.1 }

/** Posterior approximations against an oracle, quantity by quantity, with the agreement table. */
export interface PosteriorComparison {
  readonly kind: 'posteriorComparison'
  /** What the methods are compared with (`oracle`, `reference run nuts`). */
  readonly oracle: string
  /** The methods, in order. */
  readonly methods: readonly string[]
  readonly quantities: readonly ComparedQuantity[]
  /** The rows of the agreement table, one per method and quantity (empty when none was given). */
  readonly agreement: readonly ComparisonRow[]
  readonly tolerances: ComparisonTolerances
}

/** Options of {@link posteriorComparison}. */
export interface PosteriorComparisonOptions {
  /** What the methods are compared with, for the caption (default the table's, else `oracle`). */
  readonly oracle?: string
  /** The oracles by quantity name (a closed form, a grid, a reference run). */
  readonly truth?: Readonly<Record<string, MarginalLike>>
  /** The agreement table of the kernel (`agree`): its rows become the record's. */
  readonly table?: AgreementTableLike
  /** The quantities to show (default the table's, else the oracles', else the first method's). */
  readonly quantities?: readonly string[]
  /** The quantile points of each method's draws (default 200). */
  readonly quantilePoints?: number
  /** The points of the oracle's CDF grid (default 200). */
  readonly gridPoints?: number
  /** The tolerances beyond which a cell of the table is flagged (default those of the kernel). */
  readonly tolerances?: Partial<ComparisonTolerances>
}

const isNormals = (a: DrawsLike | NormalsLike): a is NormalsLike => 'normal' in a

/** The `n` quantiles at the levels `(i + 0.5) / n` of weighted draws, interpolating between the sorted draws as the kernel's `weightedOracle` does. */
function weightedQuantiles(draws: ArrayLike<number>, weights: ArrayLike<number> | undefined, n: number): number[] {
  const m = draws.length
  const order = Array.from({ length: m }, (_, i) => i).sort((a, b) => (draws[a] ?? 0) - (draws[b] ?? 0))
  const x = order.map((i) => draws[i] ?? 0)
  const w = order.map((i) => weights?.[i] ?? 1)
  const total = w.reduce((a, b) => a + b, 0)
  // The cumulative mass at the middle of each draw's own.
  const F: number[] = []
  let acc = 0
  for (const v of w) {
    F.push((acc + v / 2) / total)
    acc += v
  }
  return Array.from({ length: n }, (_, i) => {
    const p = (i + 0.5) / n
    if (p <= (F[0] ?? 0)) return x[0] ?? 0
    if (p >= (F[m - 1] ?? 1)) return x[m - 1] ?? 0
    let lo = 0
    let hi = m - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if ((F[mid] ?? 0) <= p) lo = mid
      else hi = mid
    }
    const span = (F[lo + 1] ?? 0) - (F[lo] ?? 0)
    const t = span === 0 ? 0 : (p - (F[lo] ?? 0)) / span
    return (x[lo] ?? 0) + t * ((x[lo + 1] ?? 0) - (x[lo] ?? 0))
  })
}

function momentsOf(draws: ArrayLike<number>, weights: ArrayLike<number> | undefined): { mean: number; sd: number } {
  let total = 0
  let mean = 0
  for (let i = 0; i < draws.length; i++) {
    const w = weights?.[i] ?? 1
    total += w
    mean += w * (draws[i] ?? 0)
  }
  mean /= total
  let v = 0
  for (let i = 0; i < draws.length; i++) v += (weights?.[i] ?? 1) * ((draws[i] ?? 0) - mean) ** 2
  return { mean, sd: Math.sqrt(v / total) }
}

/**
 * The record of a comparison of posterior approximations, from their draws (or normals), the oracles of the quantities
 * and the kernel's agreement table.
 *
 * @param approximations - The methods: weighted draws by quantity (`DrawsLike`, the kernel's `Approximation`), or a normal per quantity (`NormalsLike`).
 * @param options - The oracles, the agreement table, the quantities and the resolutions.
 * @returns The record.
 * @throws OmaiError `value` when a quantity named has no oracle and no method gives it, or a method has no draws of a quantity
 *   it is to show; `shape` for an empty list of methods.
 * @index function Posterior approximations against an oracle per quantity, with the agreement table, from `agree`'s inputs.
 * @example
 * ```ts
 * import { posteriorComparison } from 'omai-engine/exposition'
 * const oracle = { name: 'm', mean: 0, sd: 1, cdf: (x: number) => 1 / (1 + Math.exp(-1.702 * x)), quantile: (p: number) => Math.log(p / (1 - p)) / 1.702 }
 * const draws = Array.from({ length: 400 }, (_, i) => oracle.quantile((i + 0.5) / 400))
 * const r = posteriorComparison([{ name: 'iid', draws: { m: draws } }, { name: 'vi', normal: { m: { mean: 0.1, sd: 0.9 } } }], { truth: { m: oracle }, quantilePoints: 50, gridPoints: 30 })
 * [r.methods, r.quantities[0]?.truth?.x.length, r.quantities[0]?.methods[0]?.quantiles?.length] // => [['iid', 'vi'], 30, 50]
 * ```
 */
export function posteriorComparison(
  approximations: readonly (DrawsLike | NormalsLike)[],
  options: PosteriorComparisonOptions = {},
): PosteriorComparison {
  if (approximations.length === 0) throw new OmaiError('shape', 'posteriorComparison: no methods to compare')
  const first = approximations[0]
  const names =
    options.quantities ??
    options.table?.quantities ??
    (options.truth === undefined
      ? first === undefined
        ? []
        : Object.keys(isNormals(first) ? first.normal : first.draws)
      : Object.keys(options.truth))
  const points = Math.max(2, options.quantilePoints ?? 200)
  const gridPoints = Math.max(3, options.gridPoints ?? 200)
  const quantities = names.map((name): ComparedQuantity => {
    const methods = approximations.flatMap((a): ComparedMethod[] => {
      if (isNormals(a)) {
        const n = a.normal[name]
        return n === undefined ? [] : [{ name: a.name, mean: n.mean, sd: n.sd, normal: { mean: n.mean, sd: n.sd } }]
      }
      const draws = a.draws[name]
      if (draws === undefined) return []
      if (draws.length < 2)
        throw new OmaiError('value', `posteriorComparison: ${a.name} has fewer than two draws of ${name}`)
      return [{ name: a.name, ...momentsOf(draws, a.weights), quantiles: weightedQuantiles(draws, a.weights, points) }]
    })
    if (methods.length === 0) throw new OmaiError('value', `posteriorComparison: no method gives ${name}`)
    const oracle = options.truth?.[name]
    let truth: ComparedTruth | undefined
    if (oracle !== undefined) {
      // The grid covers the oracle's bulk and every method's quantiles, so a method far from the truth is in view.
      let lo = oracle.quantile(0.0005)
      let hi = oracle.quantile(0.9995)
      for (const m of methods) {
        const spread = m.normal === undefined ? (m.quantiles ?? []) : [m.mean - 3 * m.sd, m.mean + 3 * m.sd]
        for (const v of spread) {
          if (v < lo) lo = v
          if (v > hi) hi = v
        }
      }
      const x = Array.from({ length: gridPoints }, (_, i) => lo + ((hi - lo) * i) / (gridPoints - 1))
      truth = { mean: oracle.mean, sd: oracle.sd, x, cdf: x.map((v) => oracle.cdf(v)) }
    }
    return { name, ...(truth === undefined ? {} : { truth }), methods }
  })
  const rows = (options.table?.rows ?? []).filter((r) => names.includes(r.quantity))
  return seal({
    kind: 'posteriorComparison',
    oracle: options.oracle ?? options.table?.oracle ?? 'oracle',
    methods: approximations.map((a) => a.name),
    quantities,
    agreement: rows.map((r) => ({ ...r })),
    tolerances: { ...DEFAULT_TOLERANCES, ...options.tolerances },
  })
}

/** A number for a table cell: a measure the kernel could not compute (NaN, infinite) is missing. */
const cell = (v: number): number | null => (Number.isFinite(v) ? v : null)

/**
 * The agreement table of a comparison as a marked table: one row per method and quantity with the error of the mean in
 * posterior standard deviations, the ratio of standard deviations, the Kolmogorov distance, the Wasserstein distance in
 * posterior sds, the KL divergence and the effective sample size. A measure beyond its tolerance is flagged, bad when
 * three times beyond (a method that must agree and does not); the caption says what is compared with what.
 *
 * @param record - The comparison.
 * @returns The table.
 * @example
 * ```ts
 * import { posteriorComparison, comparisonTable } from 'omai-engine/exposition'
 * const row = { method: 'vi', quantity: 'm', mean: 0, sd: 0.5, meanError: 0.02, meanZ: 1, sdRatio: 0.5, ks: 0.05, wasserstein: 0.1, kl: 0.1, ess: 100, numDraws: 100 }
 * const r = posteriorComparison([{ name: 'vi', normal: { m: { mean: 0, sd: 0.5 } } }], { table: { oracle: 'oracle', quantities: ['m'], rows: [row] } })
 * comparisonTable(r).marks?.map((m) => [m.col, m.status]) // => [[3, 'bad']]
 * ```
 */
export function comparisonTable(record: PosteriorComparison): MarkedTable {
  const t = record.tolerances
  const marks: CellMark[] = []
  const flag = (row: number, col: number, status: 'warn' | 'bad', note: string): void => {
    marks.push({ row, col, status, note })
  }
  const cells = record.agreement.map((r, row) => {
    if (Math.abs(r.meanError) > t.meanError || Math.abs(r.meanZ) > t.meanZ)
      flag(
        row,
        2,
        Math.abs(r.meanError) > 3 * t.meanError ? 'bad' : 'warn',
        `the mean is ${r.meanError.toPrecision(2)} posterior sds off (z = ${Number.isFinite(r.meanZ) ? r.meanZ.toPrecision(2) : 'n/a'} Monte Carlo errors)`,
      )
    const [lo, hi] = t.sdRatio
    if (!(r.sdRatio >= lo && r.sdRatio <= hi))
      flag(
        row,
        3,
        r.sdRatio < 1 - 3 * (1 - lo) || r.sdRatio > 1 + 3 * (hi - 1) ? 'bad' : 'warn',
        r.sdRatio < 1 ? 'the approximation is too narrow' : 'the approximation is too wide',
      )
    if (r.ks > t.ks) flag(row, 4, r.ks > 3 * t.ks ? 'bad' : 'warn', 'the CDFs differ by more than the tolerance')
    return [
      r.method,
      r.quantity,
      cell(r.meanError),
      cell(r.sdRatio),
      cell(r.ks),
      cell(r.wasserstein),
      cell(r.kl),
      cell(r.ess),
    ]
  })
  return markedTable({
    columns: ['method', 'quantity', 'mean error (sd)', 'sd ratio', 'KS', 'W1 (sd)', 'KL', 'ESS'],
    cells,
    marks,
    digits: 3,
    caption: `compared with ${record.oracle}; flagged: mean error over ${t.meanError} sd, sd ratio outside ${t.sdRatio[0]} to ${t.sdRatio[1]}, KS over ${t.ks}`,
  })
}
