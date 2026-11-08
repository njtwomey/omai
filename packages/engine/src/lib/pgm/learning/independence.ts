/**
 * Conditional independence tests for constraint-based structure learning, pgmpy style: the chi-square and G-tests for
 * discrete variables, Fisher's z and the partial-correlation t test for continuous ones, and an independence oracle
 * over a data set that the PC and FCI learners query.
 */

import { OmaiError } from '../../base/index.ts'
import { lstsq } from '../../numerics/linalg/index.ts'
import { chdtrc, ndtr, stdtr } from '../../numerics/special/index.ts'
import { item, toArray } from '../../tensor/index.ts'
import { code, type DataRow, numericColumns } from './_data.ts'

/** The outcome of a conditional independence test. */
export interface IndependenceTest {
  readonly statistic: number
  readonly pValue: number
  /** Degrees of freedom (when the test has them). */
  readonly dof?: number
  /** Whether the p-value is at least the significance level (independence is not rejected). */
  readonly independent: boolean
}

/** Options of the independence tests. */
export interface IndependenceOptions {
  /** Significance level (default 0.01, as pgmpy's PC). */
  readonly significance?: number
  /** States of the discrete variables (default: the observed values). */
  readonly states?: Readonly<Record<string, readonly string[]>>
}

/** The power-divergence statistic over strata of z: lambda 1 is Pearson's chi-square, 0 the G (log-likelihood) test. */
function powerDivergence(
  data: readonly DataRow[],
  x: string,
  y: string,
  z: readonly string[],
  lambda: 0 | 1,
  options: IndependenceOptions,
): IndependenceTest {
  const d = code(data, [x, y, ...z], options.states)
  const kx = d.states[x]?.length ?? 1
  const ky = d.states[y]?.length ?? 1
  const zc = z.map((v) => d.states[v]?.length ?? 1)
  const strata = zc.reduce((a, b) => a * b, 1)
  const observed = new Float64Array(strata * kx * ky)
  const cx = d.codes[x] ?? new Int32Array(0)
  const cy = d.codes[y] ?? new Int32Array(0)
  const cz = z.map((v) => d.codes[v] ?? new Int32Array(0))
  for (let i = 0; i < d.n; i++) {
    let s = 0
    let ok = (cx[i] ?? -1) >= 0 && (cy[i] ?? -1) >= 0
    for (const [m, col] of cz.entries()) {
      const c = col[i] ?? -1
      if (c < 0) ok = false
      s = s * (zc[m] ?? 1) + c
    }
    if (!ok) continue
    const idx = (s * kx + (cx[i] ?? 0)) * ky + (cy[i] ?? 0)
    observed[idx] = (observed[idx] ?? 0) + 1
  }
  let stat = 0
  let dof = 0
  for (let s = 0; s < strata; s++) {
    const cell = (a: number, b: number): number => observed[(s * kx + a) * ky + b] ?? 0
    const rows = Array.from({ length: kx }, (_, a) =>
      Array.from({ length: ky }, (_, b) => cell(a, b)).reduce((p, q) => p + q, 0),
    )
    const cols = Array.from({ length: ky }, (_, b) =>
      Array.from({ length: kx }, (_, a) => cell(a, b)).reduce((p, q) => p + q, 0),
    )
    const total = rows.reduce((p, q) => p + q, 0)
    if (total === 0) continue
    const df = Math.max(rows.filter((r) => r > 0).length - 1, 0) * Math.max(cols.filter((c) => c > 0).length - 1, 0)
    dof += df
    for (let a = 0; a < kx; a++) {
      for (let b = 0; b < ky; b++) {
        const e = ((rows[a] ?? 0) * (cols[b] ?? 0)) / total
        if (!(e > 0)) continue
        let o = cell(a, b)
        // Yates' continuity correction on 2 x 2 strata, as scipy.stats.chi2_contingency.
        if (df === 1) {
          const diff = e - o
          o += Math.min(0.5, Math.abs(diff)) * Math.sign(diff)
        }
        stat += lambda === 1 ? ((o - e) * (o - e)) / e : o > 0 ? 2 * o * Math.log(o / e) : 0
      }
    }
  }
  const pValue = dof === 0 ? Number.NaN : item(chdtrc(dof, stat))
  return { statistic: stat, pValue, dof, independent: pValue >= (options.significance ?? 0.01) }
}

/**
 * Pearson's chi-square test of conditional independence of two discrete variables given others (pgmpy's ChiSquare):
 * the statistic summed over the strata of the conditioning set, with (active rows - 1)(active columns - 1) degrees of
 * freedom per stratum and Yates' correction on 2 x 2 strata.
 *
 * @param data - Discrete rows.
 * @param x - First variable.
 * @param y - Second variable.
 * @param z - The conditioning set.
 * @param options - Significance level and states.
 * @returns The statistic, p-value, degrees of freedom and the decision.
 * @concept independence-tests
 * @example
 * ```ts
 * import { chiSquareTest } from 'omai-engine/pgm/learning'
 * const data = Array.from({ length: 40 }, (_, i) => ({ a: i % 2, b: i % 2, c: Math.floor(i / 2) % 2 }))
 * [chiSquareTest(data, 'a', 'b', []).independent, chiSquareTest(data, 'a', 'c', []).independent] // => [false, true]
 * ```
 */
export function chiSquareTest(
  data: readonly DataRow[],
  x: string,
  y: string,
  z: readonly string[] = [],
  options: IndependenceOptions = {},
): IndependenceTest {
  return powerDivergence(data, x, y, z, 1, options)
}

/**
 * The G-test (log-likelihood ratio) of conditional independence of two discrete variables given others (pgmpy's GSq),
 * stratified like {@link chiSquareTest}.
 *
 * @param data - Discrete rows.
 * @param x - First variable.
 * @param y - Second variable.
 * @param z - The conditioning set.
 * @param options - Significance level and states.
 * @returns The statistic, p-value, degrees of freedom and the decision.
 * @concept independence-tests
 * @example
 * ```ts
 * import { gSquareTest } from 'omai-engine/pgm/learning'
 * const data = Array.from({ length: 40 }, (_, i) => ({ a: i % 2, b: i % 2 }))
 * gSquareTest(data, 'a', 'b').independent // => false
 * ```
 */
export function gSquareTest(
  data: readonly DataRow[],
  x: string,
  y: string,
  z: readonly string[] = [],
  options: IndependenceOptions = {},
): IndependenceTest {
  return powerDivergence(data, x, y, z, 0, options)
}

/** The partial correlation of x and y given z: the correlation of their least-squares residuals on [1, z]. */
function partialCorrelation(data: readonly DataRow[], x: string, y: string, z: readonly string[]): number {
  const cols = numericColumns(data, [x, y, ...z])
  const n = data.length
  const residual = (v: Float64Array): number[] => {
    if (z.length === 0) {
      const m = v.reduce((a, b) => a + b, 0) / n
      return Array.from(v, (t) => t - m)
    }
    const design = Array.from({ length: n }, (_, i) => [1, ...z.map((_, k) => cols[k + 2]?.[i] ?? 0)])
    const beta = toArray(lstsq(design, Array.from(v))) as number[]
    return Array.from(v, (t, i) => t - (design[i] ?? []).reduce((s, d, k) => s + d * (beta[k] ?? 0), 0))
  }
  const rx = residual(cols[0] ?? new Float64Array(n))
  const ry = residual(cols[1] ?? new Float64Array(n))
  const mx = rx.reduce((a, b) => a + b, 0) / n
  const my = ry.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < n; i++) {
    const a = (rx[i] ?? 0) - mx
    const b = (ry[i] ?? 0) - my
    sxy += a * b
    sxx += a * a
    syy += b * b
  }
  return sxy / Math.sqrt(sxx * syy)
}

/**
 * Fisher's z test of conditional independence of two continuous variables given others (pgmpy's FisherZ): the partial
 * correlation r from regression residuals, z = sqrt(n - |Z| - 3) atanh(r), two-sided normal p-value.
 *
 * @param data - Rows of numbers.
 * @param x - First variable.
 * @param y - Second variable.
 * @param z - The conditioning set.
 * @param options - Significance level.
 * @returns The statistic, p-value and the decision.
 * @throws OmaiError with code `value` when there are too few rows for the conditioning set.
 * @concept independence-tests
 * @example
 * ```ts
 * import { fisherZTest } from 'omai-engine/pgm/learning'
 * const data = Array.from({ length: 50 }, (_, i) => ({ a: i, b: 2 * i + Math.sin(i), c: Math.cos(7 * i) }))
 * [fisherZTest(data, 'a', 'b').independent, fisherZTest(data, 'a', 'c').independent] // => [false, true]
 * ```
 */
export function fisherZTest(
  data: readonly DataRow[],
  x: string,
  y: string,
  z: readonly string[] = [],
  options: Pick<IndependenceOptions, 'significance'> = {},
): IndependenceTest {
  const n = data.length
  if (n - z.length - 3 <= 0) throw new OmaiError('value', 'fisherZTest: too few rows for the conditioning set')
  const r = Math.min(Math.max(partialCorrelation(data, x, y, z), -0.999999), 0.999999)
  const statistic = Math.sqrt(n - z.length - 3) * Math.atanh(r)
  const pValue = 2 * item(ndtr(-Math.abs(statistic)))
  return { statistic, pValue, independent: pValue >= (options.significance ?? 0.01) }
}

/**
 * The partial-correlation t test of conditional independence of two continuous variables given others (pgmpy's
 * Pearsonr): t = r sqrt(dof / (1 - r^2)) with dof = n - |Z| - 2.
 *
 * @param data - Rows of numbers.
 * @param x - First variable.
 * @param y - Second variable.
 * @param z - The conditioning set.
 * @param options - Significance level.
 * @returns The partial correlation (as the statistic), p-value, degrees of freedom and the decision.
 * @concept independence-tests
 * @example
 * ```ts
 * import { partialCorrelationTest } from 'omai-engine/pgm/learning'
 * const data = Array.from({ length: 50 }, (_, i) => ({ a: i, b: 2 * i + Math.sin(i) }))
 * partialCorrelationTest(data, 'a', 'b').statistic > 0.99 // => true
 * ```
 */
export function partialCorrelationTest(
  data: readonly DataRow[],
  x: string,
  y: string,
  z: readonly string[] = [],
  options: Pick<IndependenceOptions, 'significance'> = {},
): IndependenceTest {
  const n = data.length
  const r = partialCorrelation(data, x, y, z)
  const dof = n - z.length - 2
  const t = r * Math.sqrt(dof / (1 - r * r))
  const pValue = 2 * item(stdtr(dof, -Math.abs(t)))
  return { statistic: r, pValue, dof, independent: pValue >= (options.significance ?? 0.01) }
}

/** A conditional independence test by name, or any function deciding whether x and y are independent given z. */
export type CiTest =
  | 'chiSquare'
  | 'gSquare'
  | 'fisherZ'
  | 'partialCorrelation'
  | ((x: string, y: string, z: readonly string[]) => boolean)

/**
 * A conditional independence oracle over a data set, as the constraint-based learners use it.
 *
 * @param data - The rows.
 * @param test - The test (a name, or a decision function used as is).
 * @param options - Significance level and states.
 * @returns A function deciding independence of x and y given z (memoised).
 * @concept independence-tests
 * @example
 * ```ts
 * import { independenceOracle } from 'omai-engine/pgm/learning'
 * const data = Array.from({ length: 40 }, (_, i) => ({ a: i % 2, b: i % 2 }))
 * independenceOracle(data, 'chiSquare')('a', 'b', []) // => false
 * ```
 */
export function independenceOracle(
  data: readonly DataRow[],
  test: CiTest,
  options: IndependenceOptions = {},
): (x: string, y: string, z: readonly string[]) => boolean {
  if (typeof test === 'function') return test
  const run =
    test === 'chiSquare'
      ? chiSquareTest
      : test === 'gSquare'
        ? gSquareTest
        : test === 'fisherZ'
          ? fisherZTest
          : partialCorrelationTest
  const cache = new Map<string, boolean>()
  return (x, y, z) => {
    const [a, b] = x < y ? [x, y] : [y, x]
    const key = `${a}|${b}|${[...z].sort().join(',')}`
    let v = cache.get(key)
    if (v === undefined) {
      v = run(data, a, b, [...z].sort(), options).independent
      cache.set(key, v)
    }
    return v
  }
}
