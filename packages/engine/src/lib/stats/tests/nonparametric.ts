/**
 * Rank-based (nonparametric) tests that do not assume normality, scipy.stats style: Mann-Whitney U, Wilcoxon
 * signed-rank, Kruskal-Wallis and Friedman tests with exact or tie-corrected p-values and effect sizes, the Nemenyi
 * post-hoc comparisons that follow them (as scikit-posthocs), the sign test and the Brunner-Munzel test.
 */

import { MdmlError, type TestResult } from '../../base/index.ts'
import { studentisedRange } from '../../dists/index.ts'
import { item, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { rankKernel } from '../_ranks.ts'
import { chi2Sf, groupsOf, mat, normCdf, normSf, paired, rowsOf, sample, tCdf, tSf } from '../_util.ts'
import { median } from '../quantiles.ts'
import { tieCorrection, tieTerm } from '../ranks.ts'
import { permutationTest } from '../resampling.ts'
import { binomTest } from './proportions.ts'
import { pValueFromTails, resolveTestOptions, type TestOptions, testResult } from './result.ts'

/** How a rank test computes its p-value. */
export type RankTestMethod = 'auto' | 'exact' | 'asymptotic'

/** The exact null distribution of the Mann-Whitney U statistic for sample sizes m and n (no ties): P(U = u). */
function mannWhitneyPmf(m: number, n: number): Float64Array {
  // Coefficients of the Gaussian binomial [m + n choose m]_q = prod_{i=1}^{m} (1 - q^{n+i}) / (1 - q^i).
  const size = m * n + 1
  let c = new Float64Array(size)
  c[0] = 1
  for (let i = 1; i <= m; i++) {
    const next = Float64Array.from(c)
    const shift = n + i
    for (let u = size - 1; u >= shift; u--) next[u] = (next[u] ?? 0) - (c[u - shift] ?? 0)
    for (let u = i; u < size; u++) next[u] = (next[u] ?? 0) + (next[u - i] ?? 0)
    c = next
  }
  let total = 0
  for (const v of c) total += v
  return c.map((v) => v / total)
}

/** The exact null distribution of the Wilcoxon signed-rank statistic for n untied, non-zero differences. */
function wilcoxonPmf(n: number): Float64Array {
  const size = (n * (n + 1)) / 2 + 1
  const c = new Float64Array(size)
  c[0] = 1
  for (let k = 1; k <= n; k++) for (let s = size - 1; s >= k; s--) c[s] = (c[s] ?? 0) + (c[s - k] ?? 0)
  const total = 2 ** n
  return c.map((v) => v / total)
}

const tail = (pmf: Float64Array, from: number, to: number): number => {
  let s = 0
  for (let i = Math.max(0, from); i <= Math.min(pmf.length - 1, to); i++) s += pmf[i] ?? 0
  return s
}

/** Options of {@link mannWhitneyU}. */
export interface MannWhitneyOptions extends TestOptions {
  /** `'auto'` (default) is exact when both samples are small (one at most 8) and untied, as scipy. */
  readonly method?: RankTestMethod
  /** Continuity correction of the normal approximation (default true). */
  readonly useContinuity?: boolean
}

/**
 * The Mann-Whitney U test (scipy.stats.mannwhitneyu): U1 = R1 - n1 (n1 + 1) / 2 for the first sample, with the exact
 * null distribution or the tie-corrected normal approximation; the effect size is the rank-biserial correlation
 * 2 U1 / (n1 n2) - 1.
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - method, useContinuity, alternative.
 * @returns U1 and its p-value.
 * @index function Mann-Whitney U rank test.
 * @example
 * ```ts
 * import { mannWhitneyU } from 'mdml-engine/stats/tests'
 * mannWhitneyU([1, 2, 3], [4, 5, 6]).statistic // => 0
 * mannWhitneyU([1, 2, 3], [4, 5, 6]).pValue // => 0.1
 * ```
 */
export function mannWhitneyU(x: TensorLike, y: TensorLike, options: MannWhitneyOptions = {}): TestResult {
  const { alternative } = resolveTestOptions(options)
  const a = sample('mannWhitneyU', x)
  const b = sample('mannWhitneyU', y)
  const n1 = a.length
  const n2 = b.length
  const all = Float64Array.from([...a, ...b])
  const ranks = rankKernel(all)
  let R1 = 0
  for (let i = 0; i < n1; i++) R1 += ranks[i] ?? 0
  const U1 = R1 - (n1 * (n1 + 1)) / 2
  const U2 = n1 * n2 - U1
  const ties = tieTerm(all)
  let method = options.method ?? 'auto'
  if (method === 'auto') method = (n1 > 8 && n2 > 8) || ties > 0 ? 'asymptotic' : 'exact'
  let p: number
  if (method === 'exact') {
    const pmf = mannWhitneyPmf(n1, n2)
    const sf = (u: number): number => tail(pmf, Math.ceil(u - 1e-9), n1 * n2)
    if (alternative === 'greater') p = sf(U1)
    else if (alternative === 'less') p = sf(U2)
    else p = 2 * sf(Math.max(U1, U2))
  } else {
    const n = n1 + n2
    const mu = (n1 * n2) / 2
    const s = Math.sqrt(((n1 * n2) / 12) * (n + 1 - ties / (n * (n - 1))))
    let num = U1 - mu
    if (options.useContinuity !== false) {
      const sign = alternative === 'greater' ? 1 : alternative === 'less' ? -1 : Math.sign(num)
      num -= 0.5 * sign
    }
    const z = num / s
    p = pValueFromTails(alternative, normCdf(z), normSf(z))
  }
  return testResult({
    test: 'Mann-Whitney U test',
    statistic: U1,
    pValue: Math.min(1, Math.max(0, p)),
    alternative,
    method,
    effectSize: { name: 'rank-biserial correlation', value: (2 * U1) / (n1 * n2) - 1 },
  })
}

/** How the Wilcoxon signed-rank test treats zero differences (scipy's zero_method). */
export type ZeroMethod = 'wilcox' | 'pratt' | 'zsplit'

/** Options of {@link wilcoxon}. */
export interface WilcoxonOptions extends TestOptions {
  readonly zeroMethod?: ZeroMethod
  /** Continuity correction of the normal approximation (default false, as scipy). */
  readonly correction?: boolean
  /**
   * `'auto'` (default, scipy's rule): exact for n <= 50 without ties or zeros, an exact sign-flip permutation test for n
   * <= 13 otherwise, and the normal approximation beyond.
   */
  readonly method?: RankTestMethod
}

function signedRankStatistic(d: Float64Array, zeroMethod: ZeroMethod): { rPlus: number; rMinus: number } {
  const kept = zeroMethod === 'wilcox' ? d.filter((v) => v !== 0) : d
  const r = rankKernel(Float64Array.from(kept, Math.abs))
  let rPlus = 0
  let rMinus = 0
  let rZero = 0
  for (let i = 0; i < kept.length; i++) {
    const v = kept[i] ?? 0
    if (v > 0) rPlus += r[i] ?? 0
    else if (v < 0) rMinus += r[i] ?? 0
    else rZero += r[i] ?? 0
  }
  if (zeroMethod === 'zsplit') {
    rPlus += rZero / 2
    rMinus += rZero / 2
  }
  return { rPlus, rMinus }
}

/**
 * The Wilcoxon signed-rank test (scipy.stats.wilcoxon) of the differences x - y (or of x alone): the sum of ranks of
 * the positive differences, with the exact distribution, an exact sign-flip permutation distribution for small tied
 * samples, or the tie- and zero-corrected normal approximation. The statistic is min(R+, R-) for two-sided tests and
 * R+ otherwise, as scipy reports it.
 *
 * @param x - Differences, or first measurements.
 * @param y - Second measurements (optional).
 * @param options - zeroMethod (default `'wilcox'`, dropping zeros), correction, method, alternative.
 * @returns The statistic and its p-value; the effect size is the matched-pairs rank-biserial correlation.
 * @example
 * ```ts
 * import { wilcoxon } from 'mdml-engine/stats/tests'
 * wilcoxon([1.5, 2, -0.5, 3, 4.5, 2.5]).pValue // => 0.0625
 * ```
 */
export function wilcoxon(x: TensorLike, y?: TensorLike, options: WilcoxonOptions = {}): TestResult {
  const { alternative } = resolveTestOptions(options)
  const zeroMethod = options.zeroMethod ?? 'wilcox'
  const d =
    y === undefined
      ? sample('wilcoxon', x)
      : (() => {
          const [a, b] = paired('wilcoxon', x, y)
          return Float64Array.from(a, (v, i) => v - (b[i] ?? 0))
        })()
  const { rPlus, rMinus } = signedRankStatistic(d, zeroMethod)
  const nZero = d.filter((v) => v === 0).length
  const kept = zeroMethod === 'wilcox' ? d.filter((v) => v !== 0) : d
  const count = kept.length
  if (count === 0) throw new MdmlError('value', 'wilcoxon: every difference is zero')
  const ties = tieTerm(Float64Array.from(kept, Math.abs))
  let mn = (count * (count + 1)) / 4
  let se = count * (count + 1) * (2 * count + 1)
  let tieCorrect = ties
  if (zeroMethod === 'pratt') {
    mn -= (nZero * (nZero + 1)) / 4
    se -= nZero * (nZero + 1) * (2 * nZero + 1)
    // scipy drops the zeros' tie group from the correction.
    if (nZero > 1) tieCorrect -= nZero ** 3 - nZero
  }
  se = Math.sqrt((se - tieCorrect / 2) / 24)
  let method: RankTestMethod | 'permutation' = options.method ?? 'auto'
  if (method === 'auto') {
    if (d.length > 50) method = 'asymptotic'
    else if (ties === 0 && nZero === 0) method = 'exact'
    else if (d.length <= 13) method = 'permutation'
    else method = 'asymptotic'
  }
  let p: number
  if (method === 'exact') {
    const pmf = wilcoxonPmf(count)
    const total = (count * (count + 1)) / 2
    const cdf = (k: number): number => tail(pmf, 0, k)
    const sf = (k: number): number => tail(pmf, k, total)
    if (alternative === 'less') p = cdf(Math.ceil(rPlus))
    else if (alternative === 'greater') p = sf(Math.floor(rPlus))
    else p = Math.min(1, 2 * Math.min(sf(Math.floor(rPlus)), cdf(Math.ceil(rPlus))))
  } else if (method === 'permutation') {
    p = permutationTest([d], (s) => signedRankStatistic(s, zeroMethod).rPlus, {
      permutationType: 'samples',
      alternative,
    }).pValue
  } else {
    let z = (rPlus - mn) / se
    if (options.correction === true) {
      const sign = alternative === 'greater' ? 1 : alternative === 'less' ? -1 : Math.sign(z)
      z -= (sign * 0.5) / se
    }
    p = pValueFromTails(alternative, normCdf(z), normSf(z))
  }
  const total = rPlus + rMinus
  return testResult({
    test: 'Wilcoxon signed-rank test',
    statistic: alternative === 'two-sided' ? Math.min(rPlus, rMinus) : rPlus,
    pValue: p,
    alternative,
    method,
    effectSize: { name: 'matched-pairs rank-biserial correlation', value: total > 0 ? (rPlus - rMinus) / total : 0 },
  })
}

/**
 * The Kruskal-Wallis H test (scipy.stats.kruskal): a one-way ANOVA on ranks, tie-corrected, against chi-square with
 * k - 1 degrees of freedom; the effect size is epsilon squared H / (N - 1).
 *
 * @param groups - Two or more samples.
 * @returns H and its p-value.
 * @example
 * ```ts
 * import { kruskal } from 'mdml-engine/stats/tests'
 * kruskal([[1, 2, 3], [4, 5, 6]]).statistic // => 3.857143
 * ```
 */
export function kruskal(groups: readonly TensorLike[]): TestResult {
  const gs = groupsOf('kruskal', groups, 1)
  const all = Float64Array.from(gs.flatMap((g) => Array.from(g)))
  const N = all.length
  const ranks = rankKernel(all)
  let s = 0
  let offset = 0
  for (const g of gs) {
    let r = 0
    for (let i = 0; i < g.length; i++) r += ranks[offset + i] ?? 0
    s += (r * r) / g.length
    offset += g.length
  }
  const H = ((12 / (N * (N + 1))) * s - 3 * (N + 1)) / tieCorrection(all)
  const k = gs.length
  return testResult({
    test: 'Kruskal-Wallis test',
    statistic: H,
    pValue: chi2Sf(k - 1, H),
    df: k - 1,
    effectSize: { name: 'epsilon squared', value: H / (N - 1) },
  })
}

function rowRanks(name: string, data: TensorLike): { n: number; k: number; ranks: Float64Array; ties: number } {
  const { rows: n, cols: k, data: x } = rowsOf(name, data)
  if (n < 1 || k < 2) throw new MdmlError('value', `${name}: needs a blocks x treatments matrix with k >= 2`)
  const ranks = new Float64Array(n * k)
  let ties = 0
  for (let i = 0; i < n; i++) {
    const row = x.subarray(i * k, (i + 1) * k)
    ranks.set(rankKernel(row), i * k)
    ties += tieTerm(row)
  }
  return { n, k, ranks, ties }
}

/**
 * The Friedman test for repeated measures (scipy.stats.friedmanchisquare): ranks within each block, tie-corrected, against
 * chi-square with k - 1 degrees of freedom; the effect size is Kendall's W.
 *
 * @param data - An n x k matrix: blocks (subjects) in rows, treatments in columns.
 * @returns The chi-square statistic and its p-value.
 * @example
 * ```ts
 * import { friedman } from 'mdml-engine/stats/tests'
 * friedman([[1, 2, 3], [1, 3, 2], [1, 2, 3], [2, 1, 3]]).statistic // => 4.5
 * ```
 */
export function friedman(data: TensorLike): TestResult {
  const { n, k, ranks, ties } = rowRanks('friedman', data)
  const R = new Float64Array(k)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) R[j] = (R[j] ?? 0) + (ranks[i * k + j] ?? 0)
  let ssr = 0
  for (const r of R) ssr += r * r
  const c = 1 - ties / (n * k * (k * k - 1))
  const chi2 = ((12 / (n * k * (k + 1))) * ssr - 3 * n * (k + 1)) / c
  return testResult({
    test: 'Friedman test',
    statistic: chi2,
    pValue: chi2Sf(k - 1, chi2),
    df: k - 1,
    effectSize: { name: "Kendall's W", value: chi2 / (n * (k - 1)) },
  })
}

/** The result of the Nemenyi post-hoc tests: pairwise p-values and the studentised statistics. */
export interface PosthocResult {
  /** Pairwise studentised range statistics q, k x k. */
  readonly statistic: Tensor
  /** Pairwise p-values, k x k (1 on the diagonal). */
  readonly pValue: Tensor
}

function rangeSf(k: number): (q: number) => number {
  const dist = studentisedRange(k, Number.POSITIVE_INFINITY)
  return (q) => (q === 0 ? 1 : item(dist.sf(q)))
}

/**
 * The Nemenyi post-hoc test after a Friedman test (scikit-posthocs posthoc_nemenyi_friedman): mean ranks compared by
 * q = |R_i - R_j| / sqrt(k (k + 1) / (6 n)), with p-values from the studentised range distribution with infinite
 * degrees of freedom at q sqrt(2).
 *
 * @param data - An n x k matrix: blocks in rows, treatments in columns.
 * @returns The pairwise statistics and p-values.
 * @example
 * ```ts
 * import { nemenyiFriedman } from 'mdml-engine/stats/tests'
 * nemenyiFriedman([[1, 2, 3], [1, 3, 2], [1, 2, 3], [2, 1, 3]]).statistic.shape // => [3, 3]
 * ```
 */
export function nemenyiFriedman(data: TensorLike): PosthocResult {
  const { n, k, ranks } = rowRanks('nemenyiFriedman', data)
  const mean = new Float64Array(k)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) mean[j] = (mean[j] ?? 0) + (ranks[i * k + j] ?? 0) / n
  const scale = Math.sqrt((k * (k + 1)) / (6 * n))
  const sf = rangeSf(k)
  const q = new Float64Array(k * k)
  const p = new Float64Array(k * k).fill(1)
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      if (i === j) continue
      const v = Math.abs((mean[i] ?? 0) - (mean[j] ?? 0)) / scale
      q[i * k + j] = v
      p[i * k + j] = sf(v * Math.SQRT2)
    }
  return { statistic: mat(q, k, k), pValue: mat(p, k, k) }
}

/**
 * The Nemenyi post-hoc test after a Kruskal-Wallis test (scikit-posthocs posthoc_nemenyi): mean ranks of independent
 * groups compared by q = |R_i - R_j| / sqrt(N (N + 1) / 12 (1 / n_i + 1 / n_j)) with the tie correction, and p-values
 * from the studentised range distribution with infinite degrees of freedom at q sqrt(2).
 *
 * @param groups - Two or more samples.
 * @returns The pairwise statistics and p-values.
 * @example
 * ```ts
 * import { nemenyi } from 'mdml-engine/stats/tests'
 * nemenyi([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).pValue.shape // => [3, 3]
 * ```
 */
export function nemenyi(groups: readonly TensorLike[]): PosthocResult {
  const gs = groupsOf('nemenyi', groups, 1)
  const all = Float64Array.from(gs.flatMap((g) => Array.from(g)))
  const N = all.length
  const ranks = rankKernel(all)
  const k = gs.length
  const mean = new Float64Array(k)
  let offset = 0
  gs.forEach((g, j) => {
    for (let i = 0; i < g.length; i++) mean[j] = (mean[j] ?? 0) + (ranks[offset + i] ?? 0) / g.length
    offset += g.length
  })
  const c = tieCorrection(all)
  const sf = rangeSf(k)
  const q = new Float64Array(k * k)
  const p = new Float64Array(k * k).fill(1)
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      if (i === j) continue
      const scale = Math.sqrt(((N * (N + 1)) / 12) * c * (1 / (gs[i]?.length ?? 1) + 1 / (gs[j]?.length ?? 1)))
      const v = Math.abs((mean[i] ?? 0) - (mean[j] ?? 0)) / scale
      q[i * k + j] = v
      p[i * k + j] = sf(v * Math.SQRT2)
    }
  return { statistic: mat(q, k, k), pValue: mat(p, k, k) }
}

/**
 * The sign test of a median (statsmodels.stats.descriptivestats.sign_test): M = (#above - #below) / 2, with the exact
 * binomial p-value of the smaller count (ties with mu0 dropped).
 *
 * @param x - The sample.
 * @param mu0 - The hypothesised median (default 0).
 * @param options - alternative (two-sided by default; one-sided use the binomial tail of the count above).
 * @returns M and its p-value.
 * @example
 * ```ts
 * import { signTest } from 'mdml-engine/stats/tests'
 * signTest([1, 2, 3, 4, 5, 6, 7, -1], 0).pValue // => 0.0703125
 * ```
 */
export function signTest(x: TensorLike, mu0 = 0, options: TestOptions = {}): TestResult {
  const { alternative } = resolveTestOptions(options)
  const a = sample('signTest', x, 1, true)
  let pos = 0
  let neg = 0
  for (const v of a) {
    if (v > mu0) pos++
    else if (v < mu0) neg++
  }
  const n = pos + neg
  if (n === 0) throw new MdmlError('value', 'signTest: every value equals mu0')
  const p =
    alternative === 'two-sided'
      ? binomTest(Math.min(pos, neg), n, 0.5).pValue
      : binomTest(pos, n, 0.5, { alternative }).pValue
  return testResult({
    test: 'sign test',
    statistic: (pos - neg) / 2,
    pValue: p,
    alternative,
    estimate: { value: median(a), method: 'sample median' },
  })
}

/**
 * The Brunner-Munzel test of stochastic equality (scipy.stats.brunnermunzel), robust to unequal variances: the
 * studentised difference of mean placements, referred to Student's t with Satterthwaite degrees of freedom (or the
 * normal); the effect size is the estimated P(X < Y) + P(X = Y) / 2.
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - distribution (`'t'`, default, or `'normal'`) and alternative.
 * @returns The W statistic and its p-value.
 * @example
 * ```ts
 * import { brunnerMunzel } from 'mdml-engine/stats/tests'
 * brunnerMunzel([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1], [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]).statistic // => 3.137467
 * ```
 */
export function brunnerMunzel(
  x: TensorLike,
  y: TensorLike,
  options: TestOptions & { readonly distribution?: 't' | 'normal' } = {},
): TestResult {
  const { alternative } = resolveTestOptions(options)
  const a = sample('brunnerMunzel', x, 2)
  const b = sample('brunnerMunzel', y, 2)
  const nx = a.length
  const ny = b.length
  const rc = rankKernel(Float64Array.from([...a, ...b]))
  const rx = rankKernel(a)
  const ry = rankKernel(b)
  const mean = (v: ArrayLike<number>, from: number, to: number): number => {
    let s = 0
    for (let i = from; i < to; i++) s += v[i] ?? 0
    return s / (to - from)
  }
  const rcx = mean(rc, 0, nx)
  const rcy = mean(rc, nx, nx + ny)
  const rxm = mean(rx, 0, nx)
  const rym = mean(ry, 0, ny)
  let Sx = 0
  for (let i = 0; i < nx; i++) Sx += ((rc[i] ?? 0) - (rx[i] ?? 0) - rcx + rxm) ** 2
  Sx /= nx - 1
  let Sy = 0
  for (let i = 0; i < ny; i++) Sy += ((rc[nx + i] ?? 0) - (ry[i] ?? 0) - rcy + rym) ** 2
  Sy /= ny - 1
  const w = (nx * ny * (rcy - rcx)) / ((nx + ny) * Math.sqrt(nx * Sx + ny * Sy))
  let p: number
  let df: number | undefined
  if (options.distribution === 'normal') p = pValueFromTails(alternative, normCdf(-w), normSf(-w))
  else {
    df = (nx * Sx + ny * Sy) ** 2 / ((nx * Sx) ** 2 / (nx - 1) + (ny * Sy) ** 2 / (ny - 1))
    p = pValueFromTails(alternative, tCdf(df, -w), tSf(df, -w))
  }
  return testResult({
    test: 'Brunner-Munzel test',
    statistic: w,
    pValue: p,
    alternative,
    df,
    effectSize: { name: 'P(X < Y) + P(X = Y) / 2', value: (rcy - (ny + 1) / 2) / nx },
  })
}
