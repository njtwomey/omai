/**
 * Measures of association between variables, each with its test where one exists: Pearson, Spearman and Kendall
 * correlations (scipy.stats), the covariance and correlation matrices of numpy, partial correlation given covariates
 * and the point-biserial correlation (as pingouin), and the distance correlation of Szekely, Rizzo and Bakirov, which
 * is zero exactly when the variables are independent.
 */

import { type Estimate, meanOf, type TestResult, WhiteboxError } from '../base/index.ts'
import { bootstrapIndices, foldIn, type Key, key as makeKey } from '../random/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { pinvSym } from './_linear.ts'
import { betaInc, mat, normCdf, normPpf, normSf, paired, type Rows, rowsOf, tCdf, tSf } from './_util.ts'
import { quantile } from './quantiles.ts'
import { ranksOf } from './ranks.ts'
import { intervalEstimate, pValueFromTails, resolveTestOptions, type TestOptions, testResult } from './tests/result.ts'

/** Pearson's r of two number arrays (centred dot products, scaled by their largest deviations against overflow). */
function pearsonR(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length
  const ma = meanOf(a)
  const mb = meanOf(b)
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    const da = (a[i] ?? 0) - ma
    const db = (b[i] ?? 0) - mb
    sab += da * db
    saa += da * da
    sbb += db * db
  }
  if (saa === 0 || sbb === 0) return Number.NaN
  return Math.min(1, Math.max(-1, sab / Math.sqrt(saa) / Math.sqrt(sbb)))
}

/** A Fisher-z interval tanh(atanh(r) -+ z se) for a correlation, one-sided for one-sided alternatives. */
function fisherInterval(
  r: number,
  se: number,
  alternative: TestOptions['alternative'],
  level: number,
): [number, number] {
  const z = Math.atanh(r)
  if (alternative === 'less') return [-1, Math.tanh(z + normPpf(level) * se)]
  if (alternative === 'greater') return [Math.tanh(z - normPpf(level) * se), 1]
  const c = normPpf(0.5 + level / 2)
  return [Math.tanh(z - c * se), Math.tanh(z + c * se)]
}

/**
 * Pearson's correlation coefficient with its test (scipy.stats.pearsonr): the exact p-value from the null
 * distribution of r for bivariate normal data, a beta(n/2 - 1, n/2 - 1) on [-1, 1], and Fisher's z interval.
 *
 * @param x - First variable `[N]` (N >= 2).
 * @param y - Second variable `[N]`.
 * @param options - alternative and level.
 * @returns r as the statistic (and estimate, with its interval) and the p-value (df N - 2).
 * @example
 * ```ts
 * import { pearson } from 'whitebox-engine/stats'
 * pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5]).statistic // => 0.774597
 * ```
 */
export function pearson(x: TensorLike, y: TensorLike, options: TestOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const [a, b] = paired('pearson', x, y, 2)
  const n = a.length
  const r = pearsonR(a, b)
  const ab = n / 2 - 1
  let pValue: number
  if (n === 2) pValue = Number.isNaN(r) ? Number.NaN : 1
  else {
    const u = (r + 1) / 2
    pValue = pValueFromTails(alternative, betaInc(ab, ab, u), betaInc(ab, ab, 1 - u))
  }
  const interval: [number, number] = n > 3 ? fisherInterval(r, 1 / Math.sqrt(n - 3), alternative, level) : [-1, 1]
  return testResult({
    test: "Pearson's correlation",
    statistic: r,
    pValue,
    alternative,
    df: n - 2,
    estimate: intervalEstimate(r, interval, level, { method: "Fisher's z" }),
  })
}

/**
 * Spearman's rank correlation with its test (scipy.stats.spearmanr): Pearson's r of the average ranks, the p-value from
 * t = r sqrt((n - 2) / (1 - r^2)) on n - 2 degrees of freedom, and a Fisher z interval with Bonett and Wright's (2000)
 * variance (1 + r^2 / 2) / (n - 3).
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param options - alternative and level.
 * @returns rho with its interval and p-value.
 * @example
 * ```ts
 * import { spearman } from 'whitebox-engine/stats'
 * spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]).statistic // => 0.820783
 * ```
 */
export function spearman(x: TensorLike, y: TensorLike, options: TestOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const [a, b] = paired('spearman', x, y, 2)
  const n = a.length
  const r = pearsonR(ranksOf(a), ranksOf(b))
  const dof = n - 2
  const t = r * Math.sqrt(Math.max(0, dof / ((r + 1) * (1 - r))))
  const pValue = pValueFromTails(alternative, tCdf(dof, t), tSf(dof, t))
  const interval: [number, number] =
    n > 3 ? fisherInterval(r, Math.sqrt((1 + (r * r) / 2) / (n - 3)), alternative, level) : [-1, 1]
  return testResult({
    test: "Spearman's rank correlation",
    statistic: r,
    pValue,
    alternative,
    df: dof,
    estimate: intervalEstimate(r, interval, level, { method: "Fisher's z, Bonett-Wright variance" }),
  })
}

// Strict inversions of a sequence, by merge sort (Knight's O(n log n) count of discordant pairs).
function inversions(v: Float64Array): number {
  const n = v.length
  let a = Float64Array.from(v)
  let b = new Float64Array(n)
  let count = 0
  for (let width = 1; width < n; width *= 2) {
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(lo + width, n)
      const hi = Math.min(lo + 2 * width, n)
      let i = lo
      let j = mid
      let k = lo
      while (i < mid && j < hi) {
        if ((a[j] ?? 0) < (a[i] ?? 0)) {
          count += mid - i
          b[k++] = a[j++] ?? 0
        } else b[k++] = a[i++] ?? 0
      }
      while (i < mid) b[k++] = a[i++] ?? 0
      while (j < hi) b[k++] = a[j++] ?? 0
    }
    ;[a, b] = [b, a]
  }
  return count
}

function tieStats(sorted: ArrayLike<number>): [number, number, number] {
  let t0 = 0
  let t1 = 0
  let t2 = 0
  let i = 0
  while (i < sorted.length) {
    let j = i
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[i]) j++
    const c = j - i + 1
    if (c > 1) {
      t0 += (c * (c - 1)) / 2
      t1 += c * (c - 1) * (c - 2)
      t2 += c * (c - 1) * (2 * c + 5)
    }
    i = j + 1
  }
  return [t0, t1, t2]
}

// scipy's exact null distribution of the number of concordant pairs (Kendall 1970), no ties.
function kendallExactP(n: number, concordant: number, alternative: TestOptions['alternative']): number {
  const total = (n * (n - 1)) / 2
  const inRight = concordant >= total - concordant
  const c = Math.min(concordant, total - concordant)
  let prob: number
  let mass: number
  if (n === 1) {
    prob = 1
    mass = 1
  } else if (n === 2) {
    prob = 1
    mass = 0.5
  } else {
    // new[j] = number of permutations with j inversions (scaled by 1/j! as scipy does for large n).
    let row = new Float64Array(c + 1)
    row[0] = 1
    if (c >= 1) row[1] = 1
    for (let j = 3; j <= n; j++) {
      const next = new Float64Array(c + 1)
      let s = 0
      for (let i = 0; i <= c; i++) {
        s += row[i] ?? 0
        next[i] = s / j
      }
      if (j <= c) for (let i = c; i >= j; i--) next[i] = (next[i] ?? 0) - (next[i - j] ?? 0)
      row = next
    }
    let sum = 0
    for (const v of row) sum += v
    prob = sum
    mass = (row[c] ?? 0) / 2
    if (4 * c === n * (n - 1) && alternative === 'two-sided') prob = 1
  }
  if (alternative !== 'two-sided') {
    if (inRight === (alternative === 'greater')) prob /= 2
    else prob = 1 - prob / 2 + mass
  }
  return Math.min(1, Math.max(0, prob))
}

/** Options of {@link kendallTau}. */
export interface KendallOptions extends TestOptions {
  /** tau-b (default, corrected for ties) or Stuart's tau-c (for rectangular tables). */
  readonly variant?: 'b' | 'c'
  /** `'auto'` (exact without ties for n <= 33, as scipy), `'exact'` or `'asymptotic'`. */
  readonly method?: 'auto' | 'exact' | 'asymptotic'
}

/**
 * Kendall's tau with its test (scipy.stats.kendalltau): Knight's O(n log n) count of discordant pairs, tau-b or tau-c,
 * the exact null distribution without ties for small samples and otherwise the tie-corrected normal approximation, and
 * a Fisher z interval with Fieller, Hartley and Pearson's variance 0.437 / (n - 4).
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param options - variant, method, alternative and level.
 * @returns tau with its interval and p-value.
 * @example
 * ```ts
 * import { kendallTau } from 'whitebox-engine/stats'
 * kendallTau([1, 2, 3, 4], [1, 3, 2, 4]).statistic // => 0.6666666666666666
 * kendallTau([1, 2, 3, 4], [1, 3, 2, 4]).pValue // => 0.3333333333333333
 * ```
 */
export function kendallTau(x: TensorLike, y: TensorLike, options: KendallOptions = {}): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const [a, b] = paired('kendallTau', x, y, 2)
  const n = a.length
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (i, j) => (a[i] ?? 0) - (a[j] ?? 0) || (b[i] ?? 0) - (b[j] ?? 0),
  )
  const xs = Float64Array.from(order, (i) => a[i] ?? 0)
  const ys = Float64Array.from(order, (i) => b[i] ?? 0)
  const dis = inversions(ys)
  let ntie = 0
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && xs[j + 1] === xs[i] && ys[j + 1] === ys[i]) j++
    const c = j - i + 1
    ntie += (c * (c - 1)) / 2
    i = j + 1
  }
  const [xtie, x0, x1] = tieStats(xs)
  const [ytie, y0, y1] = tieStats(Float64Array.from(b).sort())
  const tot = (n * (n - 1)) / 2
  if (xtie === tot || ytie === tot)
    return testResult({ test: "Kendall's tau", statistic: Number.NaN, pValue: Number.NaN, alternative })
  const conMinusDis = tot - xtie - ytie + ntie - 2 * dis
  let tau: number
  if (options.variant === 'c') {
    const distinct = (v: Float64Array): number => new Set(v).size
    const m = Math.min(distinct(a), distinct(b))
    tau = (2 * conMinusDis) / ((n * n * (m - 1)) / m)
  } else tau = conMinusDis / Math.sqrt(tot - xtie) / Math.sqrt(tot - ytie)
  tau = Math.min(1, Math.max(-1, tau))
  let method = options.method ?? 'auto'
  if (method === 'exact' && (xtie !== 0 || ytie !== 0))
    throw new WhiteboxError('value', 'kendallTau: ties found; the exact method cannot be used')
  if (method === 'auto')
    method = xtie === 0 && ytie === 0 && (n <= 33 || Math.min(dis, tot - dis) <= 1) ? 'exact' : 'asymptotic'
  let pValue: number
  if (method === 'exact') pValue = kendallExactP(n, tot - dis, alternative)
  else {
    const m = n * (n - 1)
    const variance = (m * (2 * n + 5) - x1 - y1) / 18 + (2 * xtie * ytie) / m + (x0 * y0) / (9 * m * (n - 2))
    const z = conMinusDis / Math.sqrt(variance)
    pValue = pValueFromTails(alternative, normCdf(z), normSf(z))
  }
  const interval: [number, number] =
    n > 4 ? fisherInterval(tau, Math.sqrt(0.437 / (n - 4)), alternative, level) : [-1, 1]
  return testResult({
    test: "Kendall's tau",
    statistic: tau,
    pValue,
    alternative,
    method,
    estimate: intervalEstimate(tau, interval, level, { method: "Fisher's z, Fieller-Hartley-Pearson variance" }),
  })
}

/** Options of {@link cov} and {@link corrcoef}. */
export interface CovOptions {
  /** Rows are variables and columns observations (default true, numpy's convention). */
  readonly rowvar?: boolean
  /** Delta degrees of freedom (default 1). */
  readonly ddof?: number
  /** Integer frequency weights per observation. */
  readonly fweights?: TensorLike
  /** Reliability (analytic) weights per observation. */
  readonly aweights?: TensorLike
}

function variables(name: string, x: TensorLike, y: TensorLike | undefined, rowvar: boolean): Rows {
  const read = (v: TensorLike): Rows => {
    const t = asTensor(v)
    const m = rowsOf(name, v)
    if (t.shape.length === 1) return { rows: 1, cols: m.rows, data: m.data }
    if (rowvar) return m
    const tr = new Float64Array(m.rows * m.cols)
    for (let i = 0; i < m.rows; i++) for (let j = 0; j < m.cols; j++) tr[j * m.rows + i] = m.data[i * m.cols + j] ?? 0
    return { rows: m.cols, cols: m.rows, data: tr }
  }
  const a = read(x)
  if (y === undefined) return a
  const b = read(y)
  if (a.cols !== b.cols) throw new WhiteboxError('shape', `${name}: x and y have different numbers of observations`)
  return { rows: a.rows + b.rows, cols: a.cols, data: Float64Array.from([...a.data, ...b.data]) }
}

/**
 * The covariance matrix (numpy.cov): variables in rows by default, divided by n - ddof, or with frequency and
 * reliability weights by numpy's weighted formula.
 *
 * @param x - A variable `[N]` or variables `[K, N]` (`[N, K]` with rowvar false).
 * @param y - More variables, stacked after x.
 * @param options - rowvar, ddof, fweights, aweights.
 * @returns The K x K covariance matrix.
 * @example
 * ```ts
 * import { cov } from 'whitebox-engine/stats'
 * cov([[0, 1, 2], [2, 1, 0]]) // => [[1, -1], [-1, 1]]
 * ```
 */
export function cov(x: TensorLike, y?: TensorLike, options: CovOptions = {}): Tensor {
  const m = variables('cov', x, y, options.rowvar ?? true)
  const k = m.rows
  const n = m.cols
  let w: Float64Array | undefined
  const fw = options.fweights === undefined ? undefined : Float64Array.from(toFlat(asTensor(options.fweights)))
  const aw = options.aweights === undefined ? undefined : Float64Array.from(toFlat(asTensor(options.aweights)))
  for (const v of [fw, aw])
    if (v !== undefined && v.length !== n) throw new WhiteboxError('shape', 'cov: weights must match the observations')
  if (fw !== undefined || aw !== undefined)
    w = Float64Array.from({ length: n }, (_, i) => (fw?.[i] ?? 1) * (aw?.[i] ?? 1))
  const ddof = options.ddof ?? 1
  let wsum = n
  if (w !== undefined) {
    wsum = 0
    for (const v of w) wsum += v
  }
  const avg = new Float64Array(k)
  for (let r = 0; r < k; r++) {
    let s = 0
    for (let j = 0; j < n; j++) s += (w?.[j] ?? 1) * (m.data[r * n + j] ?? 0)
    avg[r] = s / wsum
  }
  let fact: number
  if (w === undefined) fact = n - ddof
  else if (ddof === 0) fact = wsum
  else if (aw === undefined) fact = wsum - ddof
  else {
    let s = 0
    for (let j = 0; j < n; j++) s += (w[j] ?? 0) * (aw[j] ?? 0)
    fact = wsum - (ddof * s) / wsum
  }
  const out = new Float64Array(k * k)
  for (let a = 0; a < k; a++)
    for (let b = a; b < k; b++) {
      let s = 0
      for (let j = 0; j < n; j++)
        s += (w?.[j] ?? 1) * ((m.data[a * n + j] ?? 0) - (avg[a] ?? 0)) * ((m.data[b * n + j] ?? 0) - (avg[b] ?? 0))
      out[a * k + b] = s / fact
      out[b * k + a] = s / fact
    }
  return mat(out, k, k)
}

/**
 * The correlation matrix (numpy.corrcoef), the covariance scaled to unit diagonal (weights as in {@link cov}).
 *
 * @param x - Variables `[K, N]` (or `[N]`).
 * @param y - More variables.
 * @param options - rowvar and weights.
 * @returns The K x K correlation matrix, clipped to [-1, 1].
 * @example
 * ```ts
 * import { corrcoef } from 'whitebox-engine/stats'
 * corrcoef([[1, 2, 3], [2, 4, 7]]) // => [[1, 0.993399], [0.993399, 1]]
 * ```
 */
export function corrcoef(x: TensorLike, y?: TensorLike, options: Omit<CovOptions, 'ddof'> = {}): Tensor {
  const c = toFlat(cov(x, y, options))
  const k = Math.round(Math.sqrt(c.length))
  const d = Array.from({ length: k }, (_, i) => Math.sqrt(c[i * k + i] ?? 0))
  const out = c.map((v, idx) => {
    const i = Math.floor(idx / k)
    const j = idx % k
    return Math.min(1, Math.max(-1, v / (d[i] ?? 1) / (d[j] ?? 1)))
  })
  return mat(out, k, k)
}

/**
 * The partial correlation of x and y given covariates (pingouin.partial_corr): from the precision matrix of (x, y,
 * covariates), with the t test on n - k - 2 degrees of freedom and a Fisher z interval on n - k observations; ranks
 * first for the Spearman version.
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param covariates - Covariates `[N, K]` (or `[N]` for one).
 * @param options - method (`'pearson'`, default, or `'spearman'`), alternative and level.
 * @returns The partial correlation with its interval and p-value.
 * @example
 * ```ts
 * import { partialCorrelation } from 'whitebox-engine/stats'
 * partialCorrelation([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], [1, 3, 2, 5, 4, 6]).statistic // => 0.981703
 * ```
 */
export function partialCorrelation(
  x: TensorLike,
  y: TensorLike,
  covariates: TensorLike,
  options: TestOptions & { readonly method?: 'pearson' | 'spearman' } = {},
): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const [a, b] = paired('partialCorrelation', x, y, 3)
  const n = a.length
  const z = rowsOf('partialCorrelation', covariates)
  if (z.rows !== n) throw new WhiteboxError('shape', 'partialCorrelation: covariates must have one row per observation')
  const k = z.cols
  const p = k + 2
  const cols: Float64Array[] = [a, b]
  for (let j = 0; j < k; j++) cols.push(Float64Array.from({ length: n }, (_, i) => z.data[i * k + j] ?? 0))
  const used = options.method === 'spearman' ? cols.map((c) => ranksOf(c)) : cols
  const data = new Float64Array(n * p)
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) data[i * p + j] = used[j]?.[i] ?? 0
  const C = toFlat(cov(mat(data, n, p), undefined, { rowvar: false }))
  const VI = pinvSym(C, p)
  const r = -(VI[1] ?? 0) / Math.sqrt((VI[0] ?? 0) * (VI[p + 1] ?? 0))
  const dof = n - k - 2
  const t = r * Math.sqrt(dof / (1 - r * r))
  const pValue = pValueFromTails(alternative, tCdf(dof, t), tSf(dof, t))
  const m = n - k
  const interval: [number, number] = m > 3 ? fisherInterval(r, 1 / Math.sqrt(m - 3), alternative, level) : [-1, 1]
  return testResult({
    test: 'partial correlation',
    statistic: r,
    pValue,
    alternative,
    df: dof,
    estimate: intervalEstimate(r, interval, level, { method: "Fisher's z" }),
    method: options.method ?? 'pearson',
  })
}

/**
 * The point-biserial correlation of a binary and a continuous variable (scipy.stats.pointbiserialr): Pearson's r with
 * its test.
 *
 * @param binary - The 0/1 variable `[N]`.
 * @param x - The continuous variable `[N]`.
 * @param options - alternative and level.
 * @returns r with its interval and p-value.
 * @example
 * ```ts
 * import { pointBiserial } from 'whitebox-engine/stats'
 * pointBiserial([0, 0, 1, 1], [1, 2, 3, 5]).statistic // => 0.845154
 * ```
 */
export function pointBiserial(binary: TensorLike, x: TensorLike, options: TestOptions = {}): TestResult {
  for (const v of toFlat(asTensor(binary)))
    if (v !== 0 && v !== 1) throw new WhiteboxError('value', 'pointBiserial: the first variable must be 0 or 1')
  return { ...pearson(binary, x, options), test: 'point-biserial correlation' }
}

/** Double-centred distance matrix of the rows of an n x p matrix. */
export function centredDistances(m: Rows): Float64Array {
  const n = m.rows
  const p = m.cols
  const d = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0
      for (let c = 0; c < p; c++) s += ((m.data[i * p + c] ?? 0) - (m.data[j * p + c] ?? 0)) ** 2
      const v = Math.sqrt(s)
      d[i * n + j] = v
      d[j * n + i] = v
    }
  const row = new Float64Array(n)
  let all = 0
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += d[i * n + j] ?? 0
    row[i] = s / n
    all += s
  }
  all /= n * n
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) d[i * n + j] = (d[i * n + j] ?? 0) - (row[i] ?? 0) - (row[j] ?? 0) + all
  return d
}

/** The distance correlation of two double-centred distance matrices, optionally with y permuted. */
export function distanceCorrelationOf(A: Float64Array, B: Float64Array, n: number, perm?: ArrayLike<number>): number {
  let xy = 0
  let xx = 0
  let yy = 0
  for (let i = 0; i < n; i++) {
    const pi = perm === undefined ? i : (perm[i] ?? i)
    for (let j = 0; j < n; j++) {
      const pj = perm === undefined ? j : (perm[j] ?? j)
      const a = A[i * n + j] ?? 0
      const b = B[pi * n + pj] ?? 0
      xy += a * b
      xx += a * a
      yy += b * b
    }
  }
  const nn = n * n
  if (xx === 0 || yy === 0) return 0
  return Math.sqrt(Math.max(0, xy / nn)) / Math.sqrt(Math.sqrt(xx / nn) * Math.sqrt(yy / nn))
}

/**
 * The distance correlation of Szekely, Rizzo and Bakirov (2007), the biased V-statistic pingouin.distance_corr
 * computes; zero exactly when the variables are independent (in the population). With a key or seed, a bootstrap
 * percentile interval is added. Its permutation test is `tests.distanceCovarianceTest`.
 *
 * @param x - First variable(s) `[N]` or `[N, P]`.
 * @param y - Second variable(s) `[N]` or `[N, Q]`.
 * @param options - key or seed for the interval, nResamples (default 1000) and level.
 * @returns The distance correlation in [0, 1].
 * @example
 * ```ts
 * import { distanceCorrelation } from 'whitebox-engine/stats'
 * distanceCorrelation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]).value // => 0.986916
 * ```
 */
export function distanceCorrelation(
  x: TensorLike,
  y: TensorLike,
  options: { readonly key?: Key; readonly seed?: number; readonly nResamples?: number; readonly level?: number } = {},
): Estimate {
  const a = rowsOf('distanceCorrelation', x)
  const b = rowsOf('distanceCorrelation', y)
  if (a.rows !== b.rows) throw new WhiteboxError('shape', 'distanceCorrelation: x and y differ in length')
  const n = a.rows
  const value = distanceCorrelationOf(centredDistances(a), centredDistances(b), n)
  if (options.key === undefined && options.seed === undefined) return { value, method: 'V-statistic' }
  const level = resolveTestOptions({ level: options.level ?? 0.95 }).level
  const root = options.key ?? makeKey(options.seed ?? 0)
  const reps = options.nResamples ?? 1000
  const stats = new Float64Array(reps)
  for (let r = 0; r < reps; r++) {
    const idx = bootstrapIndices(foldIn(root, r), n).data
    const pick = (m: Rows): Rows => {
      const d = new Float64Array(n * m.cols)
      for (let i = 0; i < n; i++)
        for (let c = 0; c < m.cols; c++) d[i * m.cols + c] = m.data[(idx[i] ?? 0) * m.cols + c] ?? 0
      return { rows: n, cols: m.cols, data: d }
    }
    stats[r] = distanceCorrelationOf(centredDistances(pick(a)), centredDistances(pick(b)), n)
  }
  const alpha = (1 - level) / 2
  return {
    value,
    interval: [quantile(stats, alpha), quantile(stats, 1 - alpha)],
    level,
    method: 'bootstrap percentile',
  }
}
