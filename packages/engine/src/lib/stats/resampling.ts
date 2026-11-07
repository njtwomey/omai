/**
 * Resampling inference, scipy.stats style: permutation tests of any statistic (exact by enumeration when feasible,
 * Monte Carlo otherwise), bootstrap confidence intervals by the percentile, basic, BCa and studentised methods, the
 * jackknife estimate of bias and standard error, and Monte Carlo tests against a simulated null. The permutation test
 * and bootstrap are also traced algorithms with explicit keys.
 */

import {
  type Alternative,
  type Estimate,
  factorialOf,
  MdmlError,
  meanOf,
  type TestResult,
  varianceOf,
} from '../base/index.ts'
import { bootstrapIndices, foldIn, type Key, permutation, randint } from '../random/index.ts'
import { type Tensor, type TensorLike, toVector } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { normCdf, normPpf, sample, vec } from './_util.ts'
import { quantile } from './quantiles.ts'
import { resolveAlternative, resolveTestOptions, type TestOptions, testResult } from './tests/result.ts'

/** A statistic of one or more samples. */
export type SampleStatistic = (...samples: Float64Array[]) => number

/** Run options shared by the resampling procedures (seed or key, tracer, policy, parent run). */
export type ResamplingRunOptions = Partial<Omit<RunOptions, 'maxSteps' | 'start'>>

/** How {@link permutationTest} permutes the data (scipy.stats.permutation_test). */
export type PermutationType = 'independent' | 'samples' | 'pairings'

/** Options of {@link permutationTest}. */
export interface PermutationOptions extends ResamplingRunOptions {
  /**
   * `'independent'` (default) reassigns observations between samples; `'samples'` permutes the paired observations
   * across samples (flips signs for one sample); `'pairings'` permutes the order within each sample.
   */
  readonly permutationType?: PermutationType
  readonly alternative?: Alternative
  /** Random resamples when exhaustive enumeration is larger (default 9999). */
  readonly nResamples?: number
  /** Resamples per algorithm step (default 200). */
  readonly batch?: number
}

/** The result of a permutation or Monte Carlo test. */
export interface ResamplingTestResult extends TestResult {
  readonly nullDistribution: Tensor
  /** Whether every distinct arrangement was enumerated (an exact test). */
  readonly exact: boolean
}

/** The state of {@link permutationTestAlgorithm}: the null statistics computed so far. */
export interface ResamplingState {
  readonly done: number
  readonly total: number
  readonly nullDistribution: readonly number[]
}

function binomialCoefficient(n: number, k: number): number {
  if (k < 0 || k > n) return 0
  let r = 1
  for (let i = 1; i <= Math.min(k, n - k); i++) r = (r * (n - Math.min(k, n - k) + i)) / i
  return Math.round(r)
}

// The lexicographic combination of `m` of the items 0..N-1 with rank r.
function unrankCombination(N: number, m: number, rank: number): number[] {
  const out: number[] = []
  let r = rank
  let left = m
  for (let i = 0; i < N && left > 0; i++) {
    const count = binomialCoefficient(N - i - 1, left - 1)
    if (r < count) {
      out.push(i)
      left--
    } else r -= count
  }
  return out
}

// The permutation of 0..n-1 with Lehmer rank r.
function unrankPermutation(n: number, rank: number): number[] {
  const items = Array.from({ length: n }, (_, i) => i)
  const out: number[] = []
  let r = rank
  for (let i = n; i >= 1; i--) {
    const f = factorialOf(i - 1)
    const j = Math.floor(r / f)
    r -= j * f
    out.push(items.splice(j, 1)[0] ?? 0)
  }
  return out
}

/** The number of distinct arrangements a permutation scheme visits. */
function arrangements(type: PermutationType, data: readonly Float64Array[]): number {
  const sizes = data.map((d) => d.length)
  if (type === 'independent') {
    let left = sizes.reduce((s, n) => s + n, 0)
    let total = 1
    for (const n of sizes) {
      total *= binomialCoefficient(left, n)
      left -= n
    }
    return total
  }
  if (type === 'samples') {
    const n = sizes[0] ?? 0
    return data.length === 1 ? 2 ** n : factorialOf(data.length) ** n
  }
  return sizes.reduce((s, n) => s * factorialOf(n), 1)
}

/** Build the samples of arrangement `rank` (exact) or of a random arrangement (key). */
function arrange(type: PermutationType, data: readonly Float64Array[], rank: number | Key): Float64Array[] {
  const sizes = data.map((d) => d.length)
  const exact = typeof rank === 'number'
  if (type === 'independent') {
    const pooled = data.flatMap((d) => Array.from(d))
    const N = pooled.length
    let order: number[]
    if (exact) {
      order = []
      let remaining = Array.from({ length: N }, (_, i) => i)
      let r = rank
      for (let g = 0; g < sizes.length - 1; g++) {
        const n = sizes[g] ?? 0
        const count = binomialCoefficient(remaining.length, n)
        const pick = unrankCombination(remaining.length, n, r % count)
        r = Math.floor(r / count)
        const chosen = new Set(pick)
        order.push(...pick.map((i) => remaining[i] ?? 0))
        remaining = remaining.filter((_, i) => !chosen.has(i))
      }
      order.push(...remaining)
    } else order = Array.from(permutation(rank, N).data)
    const out: Float64Array[] = []
    let offset = 0
    for (const n of sizes) {
      out.push(Float64Array.from(order.slice(offset, offset + n), (i) => pooled[i] ?? 0))
      offset += n
    }
    return out
  }
  if (type === 'samples') {
    const n = sizes[0] ?? 0
    if (data.length === 1) {
      const d = data[0] ?? new Float64Array()
      const flips = exact ? null : randint(rank, [n], 0, 2).data
      return [Float64Array.from(d, (v, i) => ((exact ? Math.floor(rank / 2 ** i) % 2 : (flips?.[i] ?? 0)) ? -v : v))]
    }
    const k = data.length
    const out = data.map(() => new Float64Array(n))
    let r = exact ? rank : 0
    const kf = factorialOf(k)
    for (let i = 0; i < n; i++) {
      const p = exact ? unrankPermutation(k, r % kf) : Array.from(permutation(foldIn(rank, i), k).data)
      if (exact) r = Math.floor(r / kf)
      for (let g = 0; g < k; g++) {
        const col = out[g]
        if (col !== undefined) col[i] = data[p[g] ?? 0]?.[i] ?? 0
      }
    }
    return out
  }
  let r = exact ? rank : 0
  return data.map((d, g) => {
    const n = d.length
    const p = exact ? unrankPermutation(n, r % factorialOf(n)) : Array.from(permutation(foldIn(rank, g), n).data)
    if (exact) r = Math.floor(r / factorialOf(n))
    return Float64Array.from(p, (i) => d[i] ?? 0)
  })
}

/** scipy's p-value of an observed statistic against a null distribution (relative tolerance 100 eps). */
function resamplingPValue(
  alternative: Alternative,
  observed: number,
  nulls: readonly number[],
  adjustment: number,
): number {
  const gamma = Math.abs(100 * Number.EPSILON * observed)
  let le = 0
  let ge = 0
  for (const v of nulls) {
    if (v <= observed + gamma) le++
    if (v >= observed - gamma) ge++
  }
  const n = nulls.length
  const less = (le + adjustment) / (n + adjustment)
  const greater = (ge + adjustment) / (n + adjustment)
  const p = alternative === 'less' ? less : alternative === 'greater' ? greater : 2 * Math.min(less, greater)
  return Math.min(1, Math.max(0, p))
}

const RESAMPLING_SCHEMA = channels({
  metrics: {
    resamples: { description: 'null statistics computed so far' },
    pValue: { description: 'the p-value of the null distribution so far' },
  },
})

/**
 * A permutation test as a traced algorithm: each step computes a batch of null statistics, by exhaustive enumeration
 * when the number of distinct arrangements is at most `nResamples` (an exact test) and otherwise from keyed random
 * permutations.
 *
 * @param samples - The samples.
 * @param statistic - The statistic of the samples.
 * @param options - permutationType, alternative, nResamples and batch.
 * @returns The algorithm; run it with `runAlgorithm`, or call {@link permutationTest}.
 * @example
 * ```ts
 * import { permutationTestAlgorithm, meanDifference } from 'mdml-engine/stats'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = permutationTestAlgorithm([[1, 2], [3, 4]], meanDifference)
 * runAlgorithm(alg, { maxSteps: 10 }).state.nullDistribution.length // => 6
 * ```
 */
export function permutationTestAlgorithm(
  samples: readonly TensorLike[],
  statistic: SampleStatistic,
  options: PermutationOptions = {},
): Algorithm<ResamplingState> {
  const data = samples.map((s) => sample('permutationTest', s, 1))
  const type = options.permutationType ?? 'independent'
  if (type !== 'independent' && data.some((d) => d.length !== data[0]?.length))
    throw new MdmlError('shape', `permutationTest: '${type}' permutations need samples of equal length`)
  if (type === 'independent' && data.length < 2) throw new MdmlError('value', 'permutationTest: needs two samples')
  const nResamples = options.nResamples ?? 9999
  const count = arrangements(type, data)
  const exact = count <= nResamples
  const total = exact ? count : nResamples
  const batch = options.batch ?? 200
  const alternative = resolveAlternative(options.alternative)
  const observed = statistic(...data)
  return {
    name: 'permutationTest',
    config: { permutationType: type, nResamples, exact, alternative },
    schema: RESAMPLING_SCHEMA,
    init: () => ({ done: 0, total, nullDistribution: [] }),
    step: (s, ctx) => {
      const next = [...s.nullDistribution]
      const end = Math.min(s.total, s.done + batch)
      for (let r = s.done; r < end; r++) next.push(statistic(...arrange(type, data, exact ? r : foldIn(ctx.key, r))))
      return { done: end, total: s.total, nullDistribution: next }
    },
    done: (s) => (s.done >= s.total ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        resamples: s.done,
        pValue: s.done === 0 ? Number.NaN : resamplingPValue(alternative, observed, s.nullDistribution, exact ? 0 : 1),
      },
    }),
  }
}

/**
 * A permutation test of any statistic (scipy.stats.permutation_test): exact by enumeration when feasible, otherwise
 * Monte Carlo with the +1 adjustment; p-values compare with a relative tolerance as scipy's do.
 *
 * @param samples - The samples.
 * @param statistic - The statistic of the samples.
 * @param options - permutationType (default independent), alternative, nResamples (default 9999), key or seed, tracer.
 * @returns The statistic, its p-value and the null distribution.
 * @index function Permutation test of any statistic, exact when feasible.
 * @example
 * ```ts
 * import { meanDifference, permutationTest } from 'mdml-engine/stats'
 * permutationTest([[1, 2, 3], [4, 5, 6]], meanDifference).pValue // => 0.1
 * ```
 */
export function permutationTest(
  samples: readonly TensorLike[],
  statistic: SampleStatistic,
  options: PermutationOptions = {},
): ResamplingTestResult {
  const alg = permutationTestAlgorithm(samples, statistic, options)
  const { nResamples: _n, batch: _b, permutationType: _p, alternative: _a, ...run } = options
  const r = runAlgorithm(alg, { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  const data = samples.map((s) => sample('permutationTest', s, 1))
  const observed = statistic(...data)
  const exact = (alg.config as { exact: boolean }).exact
  const alternative = resolveAlternative(options.alternative)
  return {
    ...testResult({
      test: 'permutation test',
      statistic: observed,
      pValue: resamplingPValue(alternative, observed, r.state.nullDistribution, exact ? 0 : 1),
      alternative,
      method: exact ? 'exact enumeration' : `${r.state.total} random permutations`,
    }),
    nullDistribution: vec(r.state.nullDistribution),
    exact,
  }
}

/**
 * The difference of sample means, mean(x) - mean(y): the usual permutation-test statistic. Takes tensors (migration,
 * D-009: `x` and `y` were `Float64Array`; the permutation loops still hand it typed arrays, which are read in place).
 *
 * @param x - First sample: a vector (a tensor, a typed array or a flat array).
 * @param y - Second sample.
 * @returns mean(x) - mean(y).
 * @throws MdmlError with code `shape` unless both are vectors.
 * @example
 * ```ts
 * import { meanDifference } from 'mdml-engine/stats'
 * meanDifference(Float64Array.of(3, 5), Float64Array.of(1, 2)) // => 2.5
 * ```
 */
export function meanDifference(x: TensorLike, y: TensorLike): number {
  const a = x instanceof Float64Array ? x : toVector(x, 'meanDifference')
  const b = y instanceof Float64Array ? y : toVector(y, 'meanDifference')
  return meanOf(a) - meanOf(b)
}

/** A method for {@link bootstrap} intervals. */
export type BootstrapMethod = 'percentile' | 'basic' | 'bca' | 'studentized'

/** Options of {@link bootstrap}. */
export interface BootstrapOptions extends ResamplingRunOptions {
  /** Interval method (default `'bca'`, as scipy). */
  readonly method?: BootstrapMethod
  /** Number of bootstrap resamples (default 9999). */
  readonly nResamples?: number
  readonly level?: number
  /** Resample the samples jointly by observation (default false: each sample independently). */
  readonly paired?: boolean
  /**
   * For `'studentized'`: the standard error of the statistic on a resample; default the jackknife standard error of
   * each resample.
   */
  readonly standardError?: (...samples: Float64Array[]) => number
  /** Resamples per algorithm step (default 200). */
  readonly batch?: number
}

/** The result of {@link bootstrap}. */
export interface BootstrapResult {
  /** The statistic with the bootstrap standard error and the interval. */
  readonly estimate: Estimate
  /** The mean of the bootstrap distribution minus the statistic. */
  readonly bias: number
  readonly distribution: Tensor
}

/** The state of {@link bootstrapAlgorithm}. */
export interface BootstrapState {
  readonly done: number
  readonly distribution: readonly number[]
  /** Studentised statistics (theta* - theta) / se* for the studentised interval. */
  readonly studentized: readonly number[]
}

function resample(data: readonly Float64Array[], key: Key, paired: boolean): Float64Array[] {
  if (paired) {
    const idx = bootstrapIndices(key, data[0]?.length ?? 0).data
    return data.map((d) => Float64Array.from(idx, (i) => d[i] ?? 0))
  }
  return data.map((d, g) => {
    const idx = bootstrapIndices(foldIn(key, g), d.length).data
    return Float64Array.from(idx, (i) => d[i] ?? 0)
  })
}

// Leave-one-out values of a statistic for each sample (paired: leave out an observation of every sample).
function jackknifeValues(data: readonly Float64Array[], statistic: SampleStatistic, paired: boolean): number[][] {
  if (paired) {
    const n = data[0]?.length ?? 0
    return [
      Array.from({ length: n }, (_, i) =>
        statistic(...data.map((d) => Float64Array.from([...d.subarray(0, i), ...d.subarray(i + 1)]))),
      ),
    ]
  }
  return data.map((d, j) =>
    Array.from({ length: d.length }, (_, i) => {
      const left = data.map((s, k) => (k === j ? Float64Array.from([...s.subarray(0, i), ...s.subarray(i + 1)]) : s))
      return statistic(...left)
    }),
  )
}

function jackknifeSe(values: readonly number[]): number {
  const n = values.length
  const m = meanOf(values)
  let s = 0
  for (const v of values) s += (v - m) ** 2
  return Math.sqrt(((n - 1) / n) * s)
}

/**
 * The bootstrap as a traced algorithm: each step draws a batch of resamples with keyed indices and records the
 * statistic (and, for the studentised interval, its studentised value).
 *
 * @param samples - The samples.
 * @param statistic - The statistic.
 * @param options - nResamples, paired, method and batch.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { bootstrapAlgorithm } from 'mdml-engine/stats'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = bootstrapAlgorithm([[1, 2, 3, 4]], (x) => x[0] ?? 0, { nResamples: 50 })
 * runAlgorithm(alg, { seed: 1, maxSteps: 10 }).state.distribution.length // => 50
 * ```
 */
export function bootstrapAlgorithm(
  samples: readonly TensorLike[],
  statistic: SampleStatistic,
  options: BootstrapOptions = {},
): Algorithm<BootstrapState> {
  const data = samples.map((s) => sample('bootstrap', s, 2))
  const paired = options.paired ?? false
  if (paired && data.some((d) => d.length !== data[0]?.length))
    throw new MdmlError('shape', 'bootstrap: paired samples must have equal lengths')
  const total = options.nResamples ?? 9999
  const batch = options.batch ?? 200
  const studentize = options.method === 'studentized'
  const theta = statistic(...data)
  const seOf =
    options.standardError ??
    ((...s: Float64Array[]): number => jackknifeSe(jackknifeValues(s, statistic, paired).flat()))
  return {
    name: 'bootstrap',
    config: { nResamples: total, paired, method: options.method ?? 'bca' },
    schema: channels({ metrics: { resamples: { description: 'bootstrap resamples drawn' } } }),
    init: () => ({ done: 0, distribution: [], studentized: [] }),
    step: (s, ctx) => {
      const dist = [...s.distribution]
      const stud = [...s.studentized]
      const end = Math.min(total, s.done + batch)
      for (let b = s.done; b < end; b++) {
        const r = resample(data, foldIn(ctx.key, b), paired)
        const v = statistic(...r)
        dist.push(v)
        if (studentize) stud.push((v - theta) / seOf(...r))
      }
      return { done: end, distribution: dist, studentized: stud }
    },
    done: (s) => (s.done >= total ? 'converged' : undefined),
    observe: (s) => ({ metrics: { resamples: s.done } }),
  }
}

/**
 * A bootstrap confidence interval (scipy.stats.bootstrap): percentile, basic (reverse percentile), bias-corrected and
 * accelerated (BCa, with the jackknife acceleration) or studentised (bootstrap-t, with a jackknife or supplied standard
 * error per resample); randomness comes from the key.
 *
 * @param samples - One or more samples.
 * @param statistic - The statistic.
 * @param options - method (default `'bca'`), nResamples (default 9999), level, paired, key or seed, tracer.
 * @returns The estimate with its bootstrap standard error and interval, the bias and the bootstrap distribution.
 * @index function Bootstrap confidence intervals for any statistic.
 * @example
 * ```ts
 * import { bootstrap, mean } from 'mdml-engine/stats'
 * const r = bootstrap([[2, 4, 3, 5, 4, 6, 5, 3]], (x) => mean(x), { seed: 0, nResamples: 500, method: 'percentile' })
 * r.estimate.value // => 4
 * ```
 */
export function bootstrap(
  samples: readonly TensorLike[],
  statistic: SampleStatistic,
  options: BootstrapOptions = {},
): BootstrapResult {
  const { level } = resolveTestOptions({ level: options.level ?? 0.95 })
  const method = options.method ?? 'bca'
  const data = samples.map((s) => sample('bootstrap', s, 2))
  const paired = options.paired ?? false
  const theta = statistic(...data)
  const { method: _m, nResamples: _n, level: _l, paired: _p, standardError: _s, batch: _b, ...run } = options
  const r = runAlgorithm(bootstrapAlgorithm(samples, statistic, options), { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  const dist = Float64Array.from(r.state.distribution)
  const alpha = (1 - level) / 2
  const se = Math.sqrt(varianceOf(dist, 1))
  let interval: [number, number]
  if (method === 'percentile') interval = [quantile(dist, alpha), quantile(dist, 1 - alpha)]
  else if (method === 'basic') interval = [2 * theta - quantile(dist, 1 - alpha), 2 * theta - quantile(dist, alpha)]
  else if (method === 'studentized') {
    const t = Float64Array.from(r.state.studentized)
    interval = [theta - quantile(t, 1 - alpha) * se, theta - quantile(t, alpha) * se]
  } else {
    let below = 0
    let atOrBelow = 0
    for (const v of dist) {
      if (v < theta) below++
      if (v <= theta) atOrBelow++
    }
    const z0 = normPpf((below + atOrBelow) / (2 * dist.length))
    let num = 0
    let den = 0
    for (const values of jackknifeValues(data, statistic, paired)) {
      const n = values.length
      const m = meanOf(values)
      let u3 = 0
      let u2 = 0
      for (const v of values) {
        const u = (n - 1) * (m - v)
        u3 += u ** 3
        u2 += u ** 2
      }
      num += u3 / n ** 3
      den += u2 / n ** 2
    }
    const a = num / (6 * den ** 1.5)
    const za = normPpf(alpha)
    const a1 = normCdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
    const a2 = normCdf(z0 + (z0 - za) / (1 - a * (z0 - za)))
    interval = [quantile(dist, a1), quantile(dist, a2)]
  }
  return {
    estimate: { value: theta, standardError: se, interval, level, method: `bootstrap ${method}` },
    bias: meanOf(dist) - theta,
    distribution: vec(dist),
  }
}

/** The result of {@link jackknife}. */
export interface JackknifeResult {
  /** The bias-corrected estimate n theta - (n - 1) mean(theta_(i)), with the jackknife standard error. */
  readonly estimate: Estimate
  readonly statistic: number
  readonly bias: number
  /** The leave-one-out values. */
  readonly values: Tensor
}

/**
 * The jackknife (Quenouille, Tukey): leave-one-out values of a statistic, the bias estimate (n - 1)(mean - theta), the
 * bias-corrected estimate and the standard error sqrt((n - 1) / n sum (theta_(i) - mean)^2).
 *
 * @param x - The sample.
 * @param statistic - The statistic.
 * @param options - level of the normal interval (default 0.95).
 * @returns The corrected estimate with its standard error, the plain statistic, the bias and the values.
 * @example
 * ```ts
 * import { jackknife, mean } from 'mdml-engine/stats'
 * jackknife([1, 2, 3, 4], (x) => mean(x)).estimate.standardError // => 0.6454972243679028
 * ```
 */
export function jackknife(
  x: TensorLike,
  statistic: (x: Float64Array) => number,
  options: { readonly level?: number } = {},
): JackknifeResult {
  const { level } = resolveTestOptions({ level: options.level ?? 0.95 })
  const a = sample('jackknife', x, 2)
  const n = a.length
  const theta = statistic(a)
  const values = jackknifeValues([a], (s) => statistic(s), false)[0] ?? []
  const m = meanOf(values)
  const bias = (n - 1) * (m - theta)
  const se = jackknifeSe(values)
  const corrected = theta - bias
  const z = normPpf(0.5 + level / 2)
  return {
    estimate: {
      value: corrected,
      standardError: se,
      interval: [corrected - z * se, corrected + z * se],
      level,
      method: 'jackknife',
    },
    statistic: theta,
    bias,
    values: vec(values),
  }
}

/** A sampler of one sample of size n under the null hypothesis. */
export type NullSampler = (key: Key, n: number) => ArrayLike<number>

/** Options of {@link monteCarloTest}. */
export interface MonteCarloOptions extends ResamplingRunOptions {
  readonly alternative?: Alternative
  /** Simulated datasets (default 9999). */
  readonly nResamples?: number
  readonly batch?: number
}

/**
 * A Monte Carlo test (scipy.stats.monte_carlo_test): the null distribution of a statistic of samples simulated from
 * the null hypothesis with the observed sizes, and the p-value (count + 1) / (nResamples + 1); traced like the
 * permutation test.
 *
 * @param samples - The observed samples.
 * @param rvs - One null sampler per sample (or one for all).
 * @param statistic - The statistic.
 * @param options - alternative, nResamples, key or seed, tracer.
 * @returns The statistic, its p-value and the null distribution.
 * @example
 * ```ts
 * import { monteCarloTest, mean } from 'mdml-engine/stats'
 * import { normal as gaussian } from 'mdml-engine/random'
 * const r = monteCarloTest([[0.1, -0.3, 0.2, 0.4]], (k, n) => gaussian(k, [n]).data, (x) => mean(x), { seed: 3, nResamples: 99 })
 * r.pValue > 0.05 // => true
 * ```
 */
export function monteCarloTest(
  samples: readonly TensorLike[],
  rvs: NullSampler | readonly NullSampler[],
  statistic: SampleStatistic,
  options: MonteCarloOptions & TestOptions = {},
): ResamplingTestResult {
  const data = samples.map((s) => sample('monteCarloTest', s, 1))
  const samplers = typeof rvs === 'function' ? data.map(() => rvs) : rvs
  if (samplers.length !== data.length)
    throw new MdmlError('value', 'monteCarloTest: give one sampler per sample (or one for all)')
  const alternative = resolveAlternative(options.alternative)
  const total = options.nResamples ?? 9999
  const batch = options.batch ?? 200
  const observed = statistic(...data)
  const alg: Algorithm<ResamplingState> = {
    name: 'monteCarloTest',
    config: { nResamples: total, alternative },
    schema: RESAMPLING_SCHEMA,
    init: () => ({ done: 0, total, nullDistribution: [] }),
    step: (s, ctx) => {
      const next = [...s.nullDistribution]
      const end = Math.min(total, s.done + batch)
      for (let r = s.done; r < end; r++) {
        const k = foldIn(ctx.key, r)
        const sim = data.map((d, g) => {
          const f = samplers[g]
          if (f === undefined) throw new MdmlError('value', 'monteCarloTest: missing sampler')
          return Float64Array.from(f(foldIn(k, g), d.length))
        })
        next.push(statistic(...sim))
      }
      return { done: end, total, nullDistribution: next }
    },
    done: (s) => (s.done >= total ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        resamples: s.done,
        pValue: s.done === 0 ? Number.NaN : resamplingPValue(alternative, observed, s.nullDistribution, 1),
      },
    }),
  }
  const { alternative: _a, nResamples: _n, batch: _b, level: _l, ...run } = options
  const r = runAlgorithm(alg, { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  return {
    ...testResult({
      test: 'Monte Carlo test',
      statistic: observed,
      pValue: resamplingPValue(alternative, observed, r.state.nullDistribution, 1),
      alternative,
      method: `${total} simulated datasets`,
    }),
    nullDistribution: vec(r.state.nullDistribution),
    exact: false,
  }
}
