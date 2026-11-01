/**
 * Tests for cointegration of integrated series, statsmodels style: the Engle-Granger two-step test (regress one series
 * on the others with the chosen deterministic terms, then an augmented Dickey-Fuller test of the residual spread with
 * MacKinnon's p-values and critical values for N series) and Johansen's trace and maximum-eigenvalue tests of the
 * cointegration rank (the reduced-rank eigenproblem of the residual product-moment matrices, with the critical-value
 * tables of MacKinnon, Haug and Michelis), plus the choice of the rank from either sequence of tests.
 */

import { MdmlError } from '../base/index.ts'
import { key as makeKey } from '../random/index.ts'
import { type AdfOptions, adfuller, mackinnoncrit, mackinnonp, type TrendTerms } from '../stats/tests/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { JOHANSEN_MAX_EIGENVALUE, JOHANSEN_TRACE } from './_johansen-tables.ts'
import {
  choleskyLower,
  columnsOf,
  invert,
  type Matrix,
  multiply,
  residualsOn,
  symmetricEigen,
  transpose,
} from './_matrix.ts'
import { leastSquares } from './_ml.ts'

/** The regressors of a series as rows (`[time][series]`), or a single series as a flat array. */
export type SeriesColumns = ArrayLike<number> | ArrayLike<ArrayLike<number>>

function columnsFrom(y: SeriesColumns, who: string): Float64Array[] {
  const first = (y as ArrayLike<unknown>)[0]
  if (typeof first === 'number') return [Float64Array.from(y as ArrayLike<number>)]
  const rows = y as ArrayLike<ArrayLike<number>>
  if (rows.length === 0) throw new MdmlError('value', `${who}: no observations`)
  return columnsOf(Array.from(rows, (r) => Array.from(r)))
}

/** Options of {@link engleGranger}. */
export interface EngleGrangerOptions extends Pick<AdfOptions, 'maxlag' | 'autolag'> {
  /** The deterministic terms of the cointegrating regression (default `'c'`). */
  readonly trend?: TrendTerms
}

/** The result of {@link engleGranger}. */
export interface EngleGrangerResult {
  /** The t statistic of the unit-root test on the spread. */
  readonly statistic: number
  /** MacKinnon's approximate asymptotic p-value (small rejects no cointegration). */
  readonly pValue: number
  /** The 1%, 5% and 10% critical values (NaN without a constant or trend: not tabulated). */
  readonly critical: readonly [number, number, number]
  /** The cointegrating regression's coefficients: the other series' slopes, then the trend terms. */
  readonly coefficients: readonly number[]
  /** The spread, the residual of the cointegrating regression. */
  readonly spread: readonly number[]
  /** The lags the unit-root test used. */
  readonly lags: number
  /** The R squared of the cointegrating regression. */
  readonly rSquared: number
}

/**
 * The augmented Engle-Granger two-step test of no cointegration (statsmodels `coint`): `y0` on `y1` (and the trend
 * terms) by least squares, then the Dickey-Fuller regression without deterministic terms on the residual spread,
 * its statistic referred to MacKinnon's tables for N = 1 + the number of regressors series.
 *
 * @param y0 - The dependent series.
 * @param y1 - The other series: a flat array for one, rows `[time][series]` for several.
 * @param options - The trend terms, and the lag selection of the unit-root test.
 * @returns The statistic, p-value, critical values, the cointegrating vector and the spread.
 * @example
 * ```ts
 * import { engleGranger } from 'mdml-engine/timeseries'
 * // x is a random walk and y = 2x + a stationary alternating spread: cointegrated.
 * const x = [0, 0.5, 0.2, 0.9, 1.5, 1.1, 1.8, 2.6, 2.1, 2.9, 3.8, 3.3, 4.1, 4.9, 4.4, 5.3, 6.1, 5.6, 6.2, 7.0]
 * const y = x.map((v, i) => 2 * v + (i % 2 === 0 ? 0.3 : -0.3))
 * engleGranger(y, x, { autolag: null, maxlag: 0 }).coefficients.length // => 2
 * ```
 */
export function engleGranger(
  y0: ArrayLike<number>,
  y1: SeriesColumns,
  options: EngleGrangerOptions = {},
): EngleGrangerResult {
  const trend = options.trend ?? 'c'
  const others = columnsFrom(y1, 'engleGranger')
  const n = y0.length
  if (others.some((c) => c.length !== n)) throw new MdmlError('shape', 'engleGranger: series differ in length')
  const regressors: Float64Array[] = [...others]
  const terms = trend === 'n' ? 0 : trend.length
  if (terms >= 1) regressors.push(new Float64Array(n).fill(1))
  if (terms >= 2) regressors.push(Float64Array.from({ length: n }, (_, i) => i + 1))
  if (terms >= 3) regressors.push(Float64Array.from({ length: n }, (_, i) => (i + 1) ** 2))
  const beta = leastSquares(regressors, y0)
  const spread = Float64Array.from({ length: n }, (_, i) => {
    let fit = 0
    for (let j = 0; j < regressors.length; j++) fit += (beta[j] ?? 0) * (regressors[j]?.[i] ?? 0)
    return (y0[i] ?? 0) - fit
  })
  const mean = y0.length === 0 ? 0 : Array.from(y0).reduce((a, b) => a + b, 0) / n
  let tss = 0
  let rss = 0
  for (let i = 0; i < n; i++) {
    tss += ((y0[i] ?? 0) - mean) ** 2
    rss += (spread[i] ?? 0) ** 2
  }
  const rSquared = 1 - rss / tss
  const N = others.length + 1
  if (rSquared >= 1 - 100 * Math.sqrt(Number.EPSILON))
    return {
      statistic: Number.NEGATIVE_INFINITY,
      pValue: 0,
      critical: [Number.NaN, Number.NaN, Number.NaN],
      coefficients: beta,
      spread: Array.from(spread),
      lags: 0,
      rSquared,
    }
  const adf = adfuller(spread, {
    regression: 'n',
    ...(options.maxlag === undefined ? {} : { maxlag: options.maxlag }),
    ...(options.autolag === undefined ? {} : { autolag: options.autolag }),
  })
  const critical: [number, number, number] =
    trend === 'n' ? [Number.NaN, Number.NaN, Number.NaN] : mackinnoncrit(n - 1, trend, N)
  return {
    statistic: adf.statistic,
    pValue: mackinnonp(adf.statistic, trend, N),
    critical,
    coefficients: beta,
    spread: Array.from(spread),
    lags: Number(adf.assumptions?.['usedlag'] ?? 0),
    rSquared,
  }
}

/** The deterministic order of a Johansen test: -1 none, 0 a constant, 1 a constant and a linear trend. */
export type JohansenDeterministicOrder = -1 | 0 | 1

/** The result of {@link johansen}. */
export interface JohansenResult {
  /** The eigenvalues of the reduced-rank problem, largest first (squared canonical correlations). */
  readonly eigenvalues: readonly number[]
  /** The eigenvectors as columns (`[variable][vector]`), normalised so that v' S_kk v = I. */
  readonly eigenvectors: Matrix
  /** The trace statistics for ranks 0, 1, ..., K - 1. */
  readonly traceStatistics: readonly number[]
  /** The maximum-eigenvalue statistics for ranks 0, 1, ..., K - 1. */
  readonly maxEigenvalueStatistics: readonly number[]
  /** The trace critical values at 90%, 95% and 99% for each rank. */
  readonly traceCritical: readonly (readonly number[])[]
  /** The maximum-eigenvalue critical values at 90%, 95% and 99% for each rank. */
  readonly maxEigenvalueCritical: readonly (readonly number[])[]
  /** The effective sample size. */
  readonly nobs: number
}

/** The residuals of a matrix's columns from a polynomial trend in the scaled time (the identity for order -1). */
function detrend(cols: readonly Float64Array[], order: number): Float64Array[] {
  if (order < 0) return cols.map((c) => Float64Array.from(c))
  const n = cols[0]?.length ?? 0
  const t = Float64Array.from({ length: n }, (_, i) => (n === 1 ? 0 : -1 + (2 * i) / (n - 1)))
  const basis: Float64Array[] = []
  for (let d = order; d >= 0; d--) basis.push(Float64Array.from(t, (v) => v ** d))
  return residualsOn(cols, basis)
}

/** The lagged differences `[dx_(t-1), ..., dx_(t-k)]` for t = k..T-1, as columns. */
function laggedDifferences(dx: readonly Float64Array[], k: number): Float64Array[] {
  const n = dx[0]?.length ?? 0
  const out: Float64Array[] = []
  for (let lag = 1; lag <= k; lag++)
    for (const c of dx) out.push(Float64Array.from({ length: n - k }, (_, i) => c[i + k - lag] ?? 0))
  return out
}

/** The product-moment matrices and the generalised eigenproblem of the Johansen procedure on residual columns. */
export function reducedRankEigen(
  r0: readonly Float64Array[],
  rk: readonly Float64Array[],
): { values: number[]; vectors: Matrix; skk: Matrix; s00: Matrix; sk0: Matrix } {
  const t = r0[0]?.length ?? 0
  const moment = (a: readonly Float64Array[], b: readonly Float64Array[]): Matrix =>
    a.map((x) =>
      b.map((y) => {
        let s = 0
        for (let i = 0; i < t; i++) s += (x[i] ?? 0) * (y[i] ?? 0)
        return s / t
      }),
    )
  const skk = moment(rk, rk)
  const sk0 = moment(rk, r0)
  const s00 = moment(r0, r0)
  const sig = multiply(multiply(sk0, invert(s00)), transpose(sk0))
  const L = choleskyLower(skk)
  const Li = invert(L)
  const c = multiply(multiply(Li, sig), transpose(Li))
  const { values, vectors } = symmetricEigen(c)
  return { values, vectors: multiply(transpose(Li), vectors), skk, s00, sk0 }
}

/**
 * Johansen's cointegration rank tests (statsmodels `coint_johansen`): the trace and maximum-eigenvalue statistics of
 * the squared canonical correlations between the differences and the lagged levels, each after removing the lagged
 * differences and the deterministic terms, against the MacKinnon-Haug-Michelis critical values.
 *
 * @param y - The series as rows `[time][series]`.
 * @param detOrder - The deterministic order: -1 none, 0 constant, 1 constant and linear trend (default 0).
 * @param kArDiff - The number of lagged differences in the model (default 1).
 * @returns The eigenvalues, eigenvectors, statistics and critical values.
 * @example
 * ```ts
 * import { johansen } from 'mdml-engine/timeseries'
 * const y = [[0, 0.1], [1, 0.9], [1.5, 1.6], [3, 2.8], [3.4, 3.5], [5, 4.7], [5.1, 5.2], [6.8, 6.4], [7.2, 7.1], [9, 8.5], [9.5, 9.6], [10.9, 10.4]]
 * johansen(y, 0, 1).eigenvalues.length // => 2
 * ```
 */
export function johansen(
  y: ArrayLike<ArrayLike<number>>,
  detOrder: JohansenDeterministicOrder = 0,
  kArDiff = 1,
): JohansenResult {
  const cols = columnsFrom(y, 'johansen')
  const K = cols.length
  const nTot = cols[0]?.length ?? 0
  if (nTot < K * (kArDiff + 2) + 5) throw new MdmlError('value', 'johansen: too few observations')
  const f = detOrder > -1 ? 0 : detOrder
  const level = detrend(cols, detOrder)
  const dx = level.map((c) => Float64Array.from({ length: nTot - 1 }, (_, i) => (c[i + 1] ?? 0) - (c[i] ?? 0)))
  const zAll = laggedDifferences(dx, kArDiff)
  const z = detrend(zAll, f)
  const dxk = detrend(
    dx.map((c) => c.slice(kArDiff)),
    f,
  )
  const r0 = residualsOn(dxk, z)
  const lx = detrend(
    level.map((c) => c.slice(0, nTot - kArDiff).slice(1)),
    f,
  )
  const rk = residualsOn(lx, z)
  const t = rk[0]?.length ?? 0
  const { values, vectors } = reducedRankEigen(r0, rk)
  // Normalise the sign of the first non-zero entry to positive, as statsmodels does.
  const first = vectors.flat().find((v) => v !== 0) ?? 1
  const d = vectors.map((row) => row.map((v) => (first < 0 ? -v : v)))
  const trace: number[] = []
  const maxeig: number[] = []
  const cvt: number[][] = []
  const cvm: number[][] = []
  for (let i = 0; i < K; i++) {
    let s = 0
    for (let j = i; j < K; j++) s += Math.log(1 - (values[j] ?? 0))
    trace.push(-t * s)
    maxeig.push(-t * Math.log(1 - (values[i] ?? 0)))
    cvt.push(JOHANSEN_TRACE[detOrder][K - i - 1]?.slice() ?? [Number.NaN, Number.NaN, Number.NaN])
    cvm.push(JOHANSEN_MAX_EIGENVALUE[detOrder][K - i - 1]?.slice() ?? [Number.NaN, Number.NaN, Number.NaN])
  }
  return {
    eigenvalues: values,
    eigenvectors: d,
    traceStatistics: trace,
    maxEigenvalueStatistics: maxeig,
    traceCritical: cvt,
    maxEigenvalueCritical: cvm,
    nobs: t,
  }
}

/** The state of {@link cointegrationRankAlgorithm}: the statistics and critical values, and the ranks tested. */
export interface RankState {
  /** The next rank to test (ranks 0..r-1 failed to be accepted). */
  readonly r: number
  /** The test statistic and its critical value at every rank. */
  readonly statistics: readonly number[]
  readonly critical: readonly number[]
  /** The accepted rank, `null` until one is. */
  readonly rank: number | null
}

/** The paths {@link cointegrationRankAlgorithm} logs, as a type. */
export type RankPath = 'select/rank' | 'select/statistic' | 'select/critical' | 'select/accepted'

/** What {@link cointegrationRankAlgorithm} logs, by path: the rank tested, its statistic, its critical value, and 1 when accepted. */
export const rankContract: Contract<RankState, RankPath> = defineContract({
  'select/rank': scalar({ description: 'the rank tested at this step' }),
  'select/statistic': scalar({ description: 'the Johansen statistic for that rank' }),
  'select/critical': scalar({ description: 'its critical value at the significance level' }),
  'select/accepted': scalar({ description: '1 when the statistic does not exceed the critical value, else 0' }),
})

/**
 * The sequence of Johansen tests that chooses a cointegration rank, as a steppable algorithm: the statistics are found
 * once (at the start); each step tests one rank, from 0 upward, against its critical value and stops at the first it
 * does not reject.
 *
 * @param y - The series as rows.
 * @param detOrder - The deterministic order (default 0).
 * @param kArDiff - The lagged differences (default 1).
 * @param method - `'trace'` (default) or `'maxeig'`.
 * @param signif - The significance level: 0.1, 0.05 (default) or 0.01.
 * @returns The algorithm; it is done at the accepted rank, or after the last rank.
 * @example
 * ```ts
 * import { cointegrationRankAlgorithm } from 'mdml-engine/timeseries'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const y = [[0, 0.1], [1, 0.9], [1.5, 1.6], [3, 2.8], [3.4, 3.5], [5, 4.7], [5.1, 5.2], [6.8, 6.4], [7.2, 7.1], [9, 8.5], [9.5, 9.6], [10.9, 10.4]]
 * runAlgorithm(cointegrationRankAlgorithm(y), { maxSteps: 3 }).state.statistics.length // => 2
 * ```
 */
export function cointegrationRankAlgorithm(
  y: ArrayLike<ArrayLike<number>>,
  detOrder: JohansenDeterministicOrder = 0,
  kArDiff = 1,
  method: 'trace' | 'maxeig' = 'trace',
  signif: 0.1 | 0.05 | 0.01 = 0.05,
): Algorithm<RankState, RankPath> {
  const col = signif === 0.1 ? 0 : signif === 0.05 ? 1 : 2
  const r = johansen(y, detOrder, kArDiff)
  const statistics = method === 'trace' ? r.traceStatistics : r.maxEigenvalueStatistics
  const critical = (method === 'trace' ? r.traceCritical : r.maxEigenvalueCritical).map((row) => row[col] ?? Number.NaN)
  return {
    name: 'cointegration-rank-selection',
    config: { detOrder, kArDiff, method, signif, steps: statistics.length },
    init: () => ({ r: 0, statistics: [...statistics], critical, rank: null }),
    step: (s, ctx) => {
      const accepted = (s.statistics[s.r] ?? 0) <= (s.critical[s.r] ?? 0)
      if (ctx.wants('select/rank')) ctx.log('select/rank', s.r)
      if (ctx.wants('select/statistic')) ctx.log('select/statistic', s.statistics[s.r] ?? 0)
      if (ctx.wants('select/critical')) ctx.log('select/critical', s.critical[s.r] ?? 0)
      if (ctx.wants('select/accepted')) ctx.log('select/accepted', accepted ? 1 : 0)
      return { ...s, r: s.r + 1, rank: accepted ? s.r : null }
    },
    done: (s) => (s.rank !== null || s.r >= s.statistics.length ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RankState,
  }
}

/**
 * The cointegration rank chosen by a sequence of Johansen tests (statsmodels `select_coint_rank`; {@link cointegrationRankAlgorithm} run to the end): the smallest rank
 * whose statistic does not exceed its critical value at the significance level.
 *
 * @param y - The series as rows.
 * @param detOrder - The deterministic order (default 0).
 * @param kArDiff - The lagged differences (default 1).
 * @param method - `'trace'` (default) or `'maxeig'`.
 * @param signif - The significance level: 0.1, 0.05 (default) or 0.01.
 * @returns The rank and the statistics tested.
 * @example
 * ```ts
 * import { selectCointegrationRank } from 'mdml-engine/timeseries'
 * const y = [[0, 0.1], [1, 0.9], [1.5, 1.6], [3, 2.8], [3.4, 3.5], [5, 4.7], [5.1, 5.2], [6.8, 6.4], [7.2, 7.1], [9, 8.5], [9.5, 9.6], [10.9, 10.4]]
 * selectCointegrationRank(y).rank >= 0 // => true
 * ```
 */
export function selectCointegrationRank(
  y: ArrayLike<ArrayLike<number>>,
  detOrder: JohansenDeterministicOrder = 0,
  kArDiff = 1,
  method: 'trace' | 'maxeig' = 'trace',
  signif: 0.1 | 0.05 | 0.01 = 0.05,
): { readonly rank: number; readonly statistics: readonly number[]; readonly critical: readonly number[] } {
  const algorithm = cointegrationRankAlgorithm(y, detOrder, kArDiff, method, signif)
  const state = runAlgorithm(algorithm, { maxSteps: algorithm.init(makeKey(0)).statistics.length + 1 }).state
  return { rank: state.rank ?? state.statistics.length, statistics: state.statistics, critical: state.critical }
}
