import { type TestResult, WhiteboxError } from '../../base/index.ts'
import { gammaincc } from '../../numerics/special/index.ts'
import { foldIn, key as makeKey, permutation } from '../../random/index.ts'
import { item, type TensorLike } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { type Rows, rowsOf } from '../_util.ts'
import { median } from '../quantiles.ts'
import { testResult } from './result.ts'

/** A positive-definite kernel on points (rows of the data). */
export type Kernel = (a: ArrayLike<number>, b: ArrayLike<number>) => number

/**
 * The Gaussian (RBF) kernel exp(-|a - b|^2 / (2 sigma^2)).
 *
 * @param sigma - The bandwidth.
 * @returns The kernel.
 * @example
 * ```ts
 * import { gaussianKernel } from 'whitebox-engine/stats/tests'
 * gaussianKernel(1)([0, 0], [1, 1]) // => 0.367879
 * ```
 */
export function gaussianKernel(sigma: number): Kernel {
  if (!(sigma > 0)) throw new WhiteboxError('value', 'gaussianKernel: sigma must be positive')
  const c = 1 / (2 * sigma * sigma)
  return (a, b) => {
    let s = 0
    for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
    return Math.exp(-s * c)
  }
}

function rowAt(m: Rows, i: number): Float64Array {
  return m.data.subarray(i * m.cols, (i + 1) * m.cols)
}

/**
 * The median heuristic for the Gaussian kernel bandwidth (Gretton et al.): sigma = sqrt(median / 2) of the non-zero
 * squared distances between the points, so the kernel is exp(-d^2 / median(d^2)).
 *
 * @param x - Points `[N]` or `[N, D]`.
 * @returns The bandwidth.
 * @example
 * ```ts
 * import { medianHeuristic } from 'whitebox-engine/stats/tests'
 * medianHeuristic([0, 1, 2]) // => 0.707107
 * ```
 */
export function medianHeuristic(x: TensorLike): number {
  const m = rowsOf('medianHeuristic', x)
  const d: number[] = []
  for (let i = 0; i < m.rows; i++)
    for (let j = i + 1; j < m.rows; j++) {
      let s = 0
      for (let c = 0; c < m.cols; c++) s += ((m.data[i * m.cols + c] ?? 0) - (m.data[j * m.cols + c] ?? 0)) ** 2
      if (s > 0) d.push(s)
    }
  if (d.length === 0) return 1
  return Math.sqrt(0.5 * median(d))
}

function gram(a: Rows, b: Rows, k: Kernel): Float64Array {
  const g = new Float64Array(a.rows * b.rows)
  for (let i = 0; i < a.rows; i++) for (let j = 0; j < b.rows; j++) g[i * b.rows + j] = k(rowAt(a, i), rowAt(b, j))
  return g
}

function stack(a: Rows, b: Rows): Rows {
  if (a.cols !== b.cols) throw new WhiteboxError('shape', 'kernel test: samples differ in dimension')
  return { rows: a.rows + b.rows, cols: a.cols, data: Float64Array.from([...a.data, ...b.data]) }
}

/** The squared MMD from the Gram matrix of pooled points, with the first `m` rows as the first sample. */
function mmd2Of(K: Float64Array, N: number, idx: ArrayLike<number>, m: number, unbiased: boolean): number {
  const n = N - m
  let xx = 0
  let yy = 0
  let xy = 0
  for (let a = 0; a < N; a++) {
    const i = idx[a] ?? a
    for (let b = 0; b < N; b++) {
      if (unbiased && a === b) continue
      const v = K[i * N + (idx[b] ?? b)] ?? 0
      if (a < m && b < m) xx += v
      else if (a >= m && b >= m) yy += v
      else if (a < m) xy += v
    }
  }
  if (unbiased) return xx / (m * (m - 1)) + yy / (n * (n - 1)) - (2 * xy) / (m * n)
  return xx / (m * m) + yy / (n * n) - (2 * xy) / (m * n)
}

/** Options of the kernel tests. */
export interface KernelTestOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The kernel (default Gaussian with the median-heuristic bandwidth of the pooled data). */
  readonly kernel?: Kernel
  /** The null distribution: permutations (default) or Gretton et al.'s gamma approximation. */
  readonly null?: 'permutation' | 'gamma'
  /** Permutations (default 999). */
  readonly nResamples?: number
}

/**
 * The squared maximum mean discrepancy between two samples under a kernel: the biased V-statistic (default) or
 * Gretton et al.'s unbiased U-statistic.
 *
 * @param x - First sample `[N]` or `[N, D]`.
 * @param y - Second sample `[M]` or `[M, D]`.
 * @param options - kernel (default Gaussian, median heuristic) and unbiased.
 * @returns MMD^2.
 * @example
 * ```ts
 * import { gaussianKernel, mmd2 } from 'whitebox-engine/stats/tests'
 * mmd2([0], [0], { kernel: gaussianKernel(1) }) // => 0
 * ```
 */
export function mmd2(
  x: TensorLike,
  y: TensorLike,
  options: { readonly kernel?: Kernel; readonly unbiased?: boolean } = {},
): number {
  const a = rowsOf('mmd2', x)
  const b = rowsOf('mmd2', y)
  const pooled = stack(a, b)
  const k = options.kernel ?? gaussianKernel(medianHeuristic(toNested(pooled)))
  const K = gram(pooled, pooled, k)
  return mmd2Of(
    K,
    pooled.rows,
    Int32Array.from({ length: pooled.rows }, (_, i) => i),
    a.rows,
    options.unbiased ?? false,
  )
}

function toNested(m: Rows): number[][] {
  return Array.from({ length: m.rows }, (_, i) => Array.from(rowAt(m, i)))
}

/** The state of a kernel permutation test: null statistics so far. */
export interface KernelTestState {
  readonly done: number
  readonly nullDistribution: readonly number[]
}

function permutationAlgorithm(
  name: string,
  total: number,
  N: number,
  statistic: (perm: ArrayLike<number>) => number,
): Algorithm<KernelTestState> {
  return {
    name,
    config: { nResamples: total },
    schema: channels({ metrics: { resamples: { description: 'permutations drawn' } } }),
    init: () => ({ done: 0, nullDistribution: [] }),
    step: (s, ctx) => {
      const next = [...s.nullDistribution]
      const end = Math.min(total, s.done + 100)
      for (let r = s.done; r < end; r++) next.push(statistic(permutation(foldIn(ctx.key, r), N).data))
      return { done: end, nullDistribution: next }
    },
    done: (s) => (s.done >= total ? 'converged' : undefined),
    observe: (s) => ({ metrics: { resamples: s.done } }),
  }
}

function permutationP(observed: number, nulls: readonly number[]): number {
  let c = 0
  for (const v of nulls) if (v >= observed - Math.abs(observed) * 1e-14) c++
  return (c + 1) / (nulls.length + 1)
}

const gammaSf = (x: number, shape: number, scale: number): number => item(gammaincc(shape, x / scale))

/**
 * The kernel two-sample test by the maximum mean discrepancy (Gretton et al. 2012): the biased MMD^2 with a permutation
 * null (traced, keyed) or the gamma approximation of m MMD^2_b for equal sample sizes (Gretton et al. 2009); any kernel,
 * by default Gaussian with the median heuristic.
 *
 * @param x - First sample `[N]` or `[N, D]`.
 * @param y - Second sample.
 * @param options - kernel, null, nResamples, key or seed, tracer.
 * @returns MMD^2 and its p-value.
 * @reference A. Gretton, K. Borgwardt, M. Rasch, B. Scholkopf and A. Smola, A kernel two-sample test, Journal of
 * Machine Learning Research 13 (2012).
 * @example
 * ```ts
 * import { mmdTest } from 'whitebox-engine/stats/tests'
 * mmdTest([0, 0.1, 0.2, 0.3, 0.4], [5, 5.1, 5.2, 5.3, 5.4], { seed: 0, nResamples: 199 }).pValue // => 0.025
 * ```
 */
export function mmdTest(x: TensorLike, y: TensorLike, options: KernelTestOptions = {}): TestResult {
  const a = rowsOf('mmdTest', x)
  const b = rowsOf('mmdTest', y)
  const pooled = stack(a, b)
  const N = pooled.rows
  const m = a.rows
  const k = options.kernel ?? gaussianKernel(medianHeuristic(toNested(pooled)))
  const K = gram(pooled, pooled, k)
  const identity = Int32Array.from({ length: N }, (_, i) => i)
  const stat = mmd2Of(K, N, identity, m, false)
  if (options.null === 'gamma') {
    if (a.rows !== b.rows) throw new WhiteboxError('value', 'mmdTest: the gamma approximation needs equal sample sizes')
    // Gretton et al. (2009): m MMD^2_b against a gamma with matched mean and variance (kernels with k(x, x) = 1).
    let diagXY = 0
    for (let i = 0; i < m; i++) diagXY += K[i * N + (m + i)] ?? 0
    const meanMMD = (2 / m) * (1 - diagXY / m)
    let s = 0
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) {
        if (i === j) continue
        const h =
          (K[i * N + j] ?? 0) + (K[(m + i) * N + (m + j)] ?? 0) - (K[i * N + (m + j)] ?? 0) - (K[(m + i) * N + j] ?? 0)
        s += h * h
      }
    const varMMD = (2 / (m * (m - 1))) * (1 / (m * (m - 1))) * s
    const al = (meanMMD * meanMMD) / varMMD
    const bet = (varMMD * m) / meanMMD
    return testResult({
      test: 'kernel two-sample test (MMD)',
      statistic: stat,
      pValue: gammaSf(m * stat, al, bet),
      alternative: 'greater',
      method: 'gamma approximation',
    })
  }
  const total = options.nResamples ?? 999
  const { kernel: _k, null: _n, nResamples: _r, ...run } = options
  const result = runAlgorithm(
    permutationAlgorithm('mmdTest', total, N, (perm) => mmd2Of(K, N, perm, m, false)),
    {
      ...run,
      ...(run.key === undefined && run.seed === undefined ? { key: makeKey(0) } : {}),
      maxSteps: Number.MAX_SAFE_INTEGER,
    },
  )
  return testResult({
    test: 'kernel two-sample test (MMD)',
    statistic: stat,
    pValue: permutationP(stat, result.state.nullDistribution),
    alternative: 'greater',
    method: `${total} permutations`,
  })
}

/** Centre a Gram matrix: H K H with H = I - 1/n. */
export function centreGram(K: Float64Array, n: number): Float64Array {
  const row = new Float64Array(n)
  let all = 0
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += K[i * n + j] ?? 0
    row[i] = s / n
    all += s
  }
  all /= n * n
  return Float64Array.from(K, (v, idx) => v - (row[Math.floor(idx / n)] ?? 0) - (row[idx % n] ?? 0) + all)
}

/** Options of {@link hsic} and {@link hsicTest}. */
export interface HsicOptions extends KernelTestOptions {
  /** The kernel on y (default Gaussian with y's median heuristic); `kernel` is the kernel on x. */
  readonly kernelY?: Kernel
}

function hsicGrams(
  x: TensorLike,
  y: TensorLike,
  options: HsicOptions,
): { Kc: Float64Array; Lc: Float64Array; K: Float64Array; L: Float64Array; n: number } {
  const a = rowsOf('hsic', x)
  const b = rowsOf('hsic', y)
  if (a.rows !== b.rows) throw new WhiteboxError('shape', 'hsic: x and y differ in length')
  const n = a.rows
  const kx = options.kernel ?? gaussianKernel(medianHeuristic(toNested(a)))
  const ky = options.kernelY ?? gaussianKernel(medianHeuristic(toNested(b)))
  const K = gram(a, a, kx)
  const L = gram(b, b, ky)
  return { Kc: centreGram(K, n), Lc: centreGram(L, n), K, L, n }
}

/**
 * The Hilbert-Schmidt independence criterion, the biased statistic tr(K H L H) / n^2 (Gretton et al. 2005).
 *
 * @param x - First variable(s) `[N]` or `[N, D]`.
 * @param y - Second variable(s).
 * @param options - kernel (on x) and kernelY.
 * @returns HSIC.
 * @example
 * ```ts
 * import { hsic } from 'whitebox-engine/stats/tests'
 * hsic([1, 2, 3, 4], [1, 2, 3, 4]) > hsic([1, 2, 3, 4], [2, 4, 1, 3]) // => true
 * ```
 */
export function hsic(x: TensorLike, y: TensorLike, options: HsicOptions = {}): number {
  const { Kc, Lc, n } = hsicGrams(x, y, options)
  let s = 0
  for (let i = 0; i < n * n; i++) s += (Kc[i] ?? 0) * (Lc[i] ?? 0)
  return s / (n * n)
}

/**
 * The HSIC test of independence (Gretton et al. 2008): permutations of y (traced, keyed) or the gamma approximation of
 * n HSIC_b; any kernels, by default Gaussian with the median heuristic.
 *
 * @param x - First variable(s).
 * @param y - Second variable(s).
 * @param options - kernel, kernelY, null, nResamples, key or seed.
 * @returns HSIC and its p-value.
 * @example
 * ```ts
 * import { hsicTest } from 'whitebox-engine/stats/tests'
 * const x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
 * hsicTest(x, x.map((v) => v * v), { null: 'gamma' }).pValue < 0.01 // => true
 * ```
 */
export function hsicTest(x: TensorLike, y: TensorLike, options: HsicOptions = {}): TestResult {
  const { Kc, Lc, K, L, n } = hsicGrams(x, y, options)
  let s = 0
  for (let i = 0; i < n * n; i++) s += (Kc[i] ?? 0) * (Lc[i] ?? 0)
  const stat = s / (n * n)
  if (options.null === 'gamma') {
    let v = 0
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue
        v += (((Kc[i * n + j] ?? 0) * (Lc[i * n + j] ?? 0)) / 6) ** 2
      }
    let varHSIC = v / (n * (n - 1))
    varHSIC = ((72 * (n - 4) * (n - 5)) / (n * (n - 1) * (n - 2) * (n - 3))) * varHSIC
    let sk = 0
    let sl = 0
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue
        sk += K[i * n + j] ?? 0
        sl += L[i * n + j] ?? 0
      }
    const muX = sk / (n * (n - 1))
    const muY = sl / (n * (n - 1))
    const mHSIC = (1 + muX * muY - muX - muY) / n
    const al = (mHSIC * mHSIC) / varHSIC
    const bet = (varHSIC * n) / mHSIC
    return testResult({
      test: 'HSIC independence test',
      statistic: stat,
      pValue: gammaSf(n * stat, al, bet),
      alternative: 'greater',
      method: 'gamma approximation',
    })
  }
  const total = options.nResamples ?? 999
  const { kernel: _k, kernelY: _ky, null: _n, nResamples: _r, ...run } = options
  const statistic = (perm: ArrayLike<number>): number => {
    let t = 0
    for (let i = 0; i < n; i++) {
      const pi = perm[i] ?? i
      for (let j = 0; j < n; j++) t += (Kc[i * n + j] ?? 0) * (Lc[pi * n + (perm[j] ?? j)] ?? 0)
    }
    return t / (n * n)
  }
  const result = runAlgorithm(permutationAlgorithm('hsicTest', total, n, statistic), {
    ...run,
    ...(run.key === undefined && run.seed === undefined ? { key: makeKey(0) } : {}),
    maxSteps: Number.MAX_SAFE_INTEGER,
  })
  return testResult({
    test: 'HSIC independence test',
    statistic: stat,
    pValue: permutationP(stat, result.state.nullDistribution),
    alternative: 'greater',
    method: `${total} permutations`,
  })
}
