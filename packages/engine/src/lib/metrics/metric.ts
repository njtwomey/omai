/**
 * The metric contract: a metric is a function of targets and predictions returning a `MetricResult`, with a
 * specification (name, direction, range, inputs, category) attached by `defineMetric`. Also the input readers every
 * metric uses (flattening, shapes, columns, sample weights), per-output averaging, scikit-learn's weighted percentile
 * and `streaming`, which accumulates a metric over batches as torchmetrics does.
 *
 * @index structure The metric contract: a spec plus a function returning a MetricResult.
 */

import { estimate, MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import { asTensor, isValue, type TensorLike, toFlat, toVector } from '../tensor/index.ts'

/** Options every metric accepts. */
export interface MetricOptions {
  /** Per-sample weights (default all ones). */
  readonly sampleWeight?: TensorLike
}

/** What a metric reads: numeric arrays (labels, scores, values) or string labels. */
export type MetricInput = TensorLike | readonly string[]

/** A metric function with its specification attached. */
export type Metric<O extends MetricOptions = MetricOptions, I = TensorLike> = ((
  yTrue: I,
  yPred: I,
  options?: O,
) => MetricResult) & { readonly spec: MetricSpec }

/** Any function returning a {@link MetricResult}, with its specification attached. */
export type MetricFunction<A extends readonly unknown[]> = ((...args: A) => MetricResult) & {
  readonly spec: MetricSpec
}

/**
 * Attach a specification to a metric function.
 *
 * @param spec - The metric's specification.
 * @param fn - The function computing its result.
 * @returns The function with `spec`.
 * @example
 * ```ts
 * import { defineMetric, metricResult } from 'mdml-engine/metrics'
 * const spec = { name: 'one', direction: 'higher', range: [0, 1], inputs: 'values', category: 'demo' } as const
 * const one = defineMetric(spec, () => metricResult('one', 1, 1))
 * one.spec.name // => 'one'
 * ```
 */
export function defineMetric<A extends readonly unknown[]>(
  spec: MetricSpec,
  fn: (...args: A) => MetricResult,
): MetricFunction<A> {
  return Object.assign(fn, { spec })
}

/**
 * Build a {@link MetricResult}.
 *
 * @param name - The metric's name.
 * @param value - The point estimate.
 * @param n - The number of samples it summarises.
 * @param breakdown - Optional per-class, per-output or count details.
 * @returns The result.
 * @example
 * ```ts
 * import { metricResult } from 'mdml-engine/metrics'
 * metricResult('accuracy', 0.75, 4).estimate.value // => 0.75
 * ```
 */
export function metricResult(
  name: string,
  value: number,
  n: number,
  breakdown?: Readonly<Record<string, number>>,
): MetricResult {
  return { metric: name, estimate: estimate(value), n, ...(breakdown === undefined ? {} : { breakdown }) }
}

/** Targets and predictions as columns: `n` samples by `k` outputs, row-major, with weights. */
export interface Columns {
  readonly n: number
  readonly k: number
  readonly yTrue: Float64Array
  readonly yPred: Float64Array
  readonly weight: Float64Array
}

/**
 * Per-sample weights as a Float64Array (all ones when absent), checked against the number of samples.
 *
 * @param name - The metric's name (for errors).
 * @param n - The number of samples.
 * @param sampleWeight - The weights, if any.
 * @returns The weights.
 * @throws MdmlError with code `shape` when the length differs from n, and `value` for a negative or non-finite weight.
 * @example
 * ```ts
 * import { sampleWeights } from 'mdml-engine/metrics'
 * Array.from(sampleWeights('demo', 2)) // => [1, 1]
 * ```
 */
export function sampleWeights(name: string, n: number, sampleWeight?: TensorLike): Float64Array {
  if (sampleWeight === undefined) return new Float64Array(n).fill(1)
  const w = Float64Array.from(toFlat(asTensor(sampleWeight)))
  if (w.length !== n) throw new MdmlError('shape', `${name}: sampleWeight must have one entry per sample`)
  for (const v of w)
    if (!(v >= 0 && v < Number.POSITIVE_INFINITY))
      throw new MdmlError('value', `${name}: sampleWeight must be finite and non-negative (got ${v})`)
  return w
}

/**
 * The values of an array input, flattened, as a Float64Array.
 *
 * @param x - Any numeric array.
 * @returns Its values.
 * @example
 * ```ts
 * import { flat } from 'mdml-engine/metrics'
 * Array.from(flat([[1, 2], [3, 4]])) // => [1, 2, 3, 4]
 * ```
 */
export function flat(x: TensorLike): Float64Array {
  return Float64Array.from(toFlat(asTensor(x)))
}

/** A numeric array input with its shape, read once. */
export interface Matrix {
  readonly shape: readonly number[]
  readonly data: Float64Array
}

/**
 * Read an array input with its shape.
 *
 * @param x - Any numeric array.
 * @returns Its shape and row-major values.
 * @example
 * ```ts
 * import { matrix } from 'mdml-engine/metrics'
 * matrix([[1, 2], [3, 4]]).shape // => [2, 2]
 * ```
 */
export function matrix(x: TensorLike): Matrix {
  const t = asTensor(x)
  return { shape: t.shape, data: Float64Array.from(toFlat(t)) }
}

/** Read 1-D or 2-D targets and predictions (and weights) into columns, checking their shapes. */
export function columns(name: string, yTrue: TensorLike, yPred: TensorLike, options: MetricOptions = {}): Columns {
  const t = asTensor(yTrue)
  const p = asTensor(yPred)
  if (t.shape.length !== p.shape.length || t.shape.some((d, i) => d !== p.shape[i])) {
    throw new MdmlError('shape', `${name}: yTrue [${t.shape.join(', ')}] and yPred [${p.shape.join(', ')}] differ`)
  }
  if (t.shape.length > 2) throw new MdmlError('shape', `${name}: expected 1-D or 2-D targets`)
  const n = t.shape[0] ?? 1
  const k = t.shape.length === 2 ? (t.shape[1] ?? 1) : 1
  if (n === 0) throw new MdmlError('value', `${name}: needs at least one sample`)
  const weight = sampleWeights(name, n, options.sampleWeight)
  return { n, k, yTrue: Float64Array.from(toFlat(t)), yPred: Float64Array.from(toFlat(p)), weight }
}

/** Define a metric from its spec and a per-output score; multi-output scores are averaged uniformly (sklearn). */
export function perOutputMetric<O extends MetricOptions>(
  spec: MetricSpec,
  score: (yTrue: Float64Array, yPred: Float64Array, weight: Float64Array, options: O) => number,
): Metric<O> {
  const metric = (yTrue: TensorLike, yPred: TensorLike, options: O = {} as O): MetricResult => {
    const c = columns(spec.name, yTrue, yPred, options)
    const scores: number[] = []
    for (let j = 0; j < c.k; j++) {
      const t = new Float64Array(c.n)
      const p = new Float64Array(c.n)
      for (let i = 0; i < c.n; i++) {
        t[i] = c.yTrue[i * c.k + j] ?? 0
        p[i] = c.yPred[i * c.k + j] ?? 0
      }
      scores.push(score(t, p, c.weight, options))
    }
    const value = scores.reduce((a, b) => a + b, 0) / scores.length
    return {
      metric: spec.name,
      estimate: estimate(value),
      n: c.n,
      ...(c.k > 1 ? { breakdown: Object.fromEntries(scores.map((s, j) => [`output ${j}`, s])) } : {}),
    }
  }
  return Object.assign(metric, { spec })
}

/** The weighted mean of `values`. */
export function weightedMean(values: ArrayLike<number>, weight: ArrayLike<number>): number {
  let s = 0
  let w = 0
  for (let i = 0; i < values.length; i++) {
    s += (values[i] ?? 0) * (weight[i] ?? 0)
    w += weight[i] ?? 0
  }
  return s / w
}

/**
 * scikit-learn's weighted percentile (sklearn.utils.stats._weighted_percentile): the first sorted value whose
 * cumulative weight reaches the rank, or with `average` the mean of it and the next value when the rank falls exactly
 * on a cumulative weight (so that unit weights give the usual median). Takes tensors (migration, D-009: `values` and
 * `weights` were `ArrayLike<number>`; nested arrays still work).
 *
 * @param values - The values: a vector (a tensor, a typed array or a flat array).
 * @param weights - Their weights, a vector of the same length (default all ones).
 * @param percentile - The rank in [0, 100].
 * @param average - Average at exact ties (default true).
 * @returns The percentile (NaN when every weight is zero).
 * @throws MdmlError with code `shape` unless `values` and `weights` are vectors.
 * @example
 * ```ts
 * import { weightedPercentile } from 'mdml-engine/metrics'
 * weightedPercentile([1, 2, 3, 4], undefined, 50) // => 2.5
 * ```
 */
export function weightedPercentile(
  values: TensorLike,
  weights: TensorLike | undefined,
  percentile: number,
  average = true,
): number {
  const xs = toVector(values, 'weightedPercentile')
  const given = weights === undefined ? undefined : toVector(weights, 'weightedPercentile')
  const n = xs.length
  const w = (i: number): number => (given === undefined ? 1 : (given[i] ?? 0))
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (xs[a] ?? 0) - (xs[b] ?? 0))
  const cdf: number[] = []
  let acc = 0
  for (const i of order) {
    acc += w(i)
    cdf.push(acc)
  }
  if (acc === 0) return Number.NaN
  let rank = (percentile / 100) * acc
  if (rank === 0) rank = Number.MIN_VALUE
  let idx = 0
  while (idx < n - 1 && (cdf[idx] ?? 0) < rank) idx++
  const at = xs[order[idx] ?? 0] ?? 0
  if (!average || (cdf[idx] ?? 0) - rank > Number.EPSILON) return at
  // The next value with positive weight.
  let next = idx + 1
  while (next < n && w(order[next] ?? 0) === 0) next++
  if (next >= n) return at
  return (at + (xs[order[next] ?? 0] ?? 0)) / 2
}

/** The number of samples (rows) of a metric input. */
export function rowCount(x: MetricInput): number {
  if (Array.isArray(x) && x.every((v) => typeof v === 'string')) return x.length
  return asTensor(x as TensorLike).shape[0] ?? 1
}

/** Select rows of a metric input (numeric arrays along their first axis, string arrays by index). */
export function selectRows<I extends MetricInput>(x: I, rows: readonly number[]): I {
  if (Array.isArray(x) && x.every((v) => typeof v === 'string')) return rows.map((i) => x[i] as string) as unknown as I
  const t = asTensor(x as TensorLike)
  const shape = t.shape
  const width = shape.slice(1).reduce((a, b) => a * b, 1)
  const data = toFlat(t)
  const out = new Float64Array(rows.length * width)
  for (const [r, i] of rows.entries()) for (let j = 0; j < width; j++) out[r * width + j] = data[i * width + j] ?? 0
  const rowsShape = [rows.length, ...shape.slice(1)]
  return reshapeNested(out, rowsShape) as unknown as I
}

function reshapeNested(data: Float64Array, shape: readonly number[]): TensorLike {
  if (shape.length <= 1) return Array.from(data)
  const [head = 0, ...rest] = shape
  const size = rest.reduce((a, b) => a * b, 1)
  return Array.from({ length: head }, (_, i) =>
    reshapeNested(data.subarray(i * size, (i + 1) * size), rest),
  ) as TensorLike
}

/** A metric accumulated over batches: `update` with each batch, `compute` the metric on everything seen. */
export interface StreamingMetric<O extends MetricOptions, I> {
  update(yTrue: I, yPred: I, sampleWeight?: TensorLike): void
  compute(options?: Omit<O, 'sampleWeight'>): MetricResult
  reset(): void
  /** The number of samples seen. */
  readonly count: number
}

/**
 * Stream a metric over batches (torchmetrics' update / compute): batches are stored and the metric is computed on
 * their concatenation, so the result equals the metric of the whole data for every metric, decomposable or not.
 *
 * @param metric - Any metric of targets and predictions.
 * @returns The accumulator.
 * @example
 * ```ts
 * import { meanAbsoluteError, streaming } from 'mdml-engine/metrics'
 * const s = streaming(meanAbsoluteError)
 * s.update([1, 2], [2, 2])
 * s.update([3], [5])
 * s.compute().estimate.value // => 1
 * ```
 */
export function streaming<O extends MetricOptions, I extends MetricInput>(metric: Metric<O, I>): StreamingMetric<O, I> {
  let trues: unknown[] = []
  let preds: unknown[] = []
  let weights: number[] = []
  let weighted = false
  const rows = (x: I): unknown[] => {
    if (Array.isArray(x) && x.every((v) => typeof v === 'string')) return [...x]
    const t = asTensor(x as TensorLike)
    return selectRowsList(t)
  }
  return {
    update(yTrue, yPred, sampleWeight) {
      const t = rows(yTrue)
      const p = rows(yPred)
      if (t.length !== p.length) throw new MdmlError('shape', 'streaming: yTrue and yPred differ in length')
      trues = trues.concat(t)
      preds = preds.concat(p)
      if (sampleWeight !== undefined) weighted = true
      weights = weights.concat(sampleWeight === undefined ? t.map(() => 1) : Array.from(flat(sampleWeight)))
    },
    compute(options) {
      if (trues.length === 0) throw new MdmlError('value', 'streaming: no batches seen')
      const opts = { ...(options ?? {}), ...(weighted ? { sampleWeight: weights } : {}) } as unknown as O
      return metric(trues as unknown as I, preds as unknown as I, opts)
    },
    reset() {
      trues = []
      preds = []
      weights = []
      weighted = false
    },
    get count() {
      return trues.length
    },
  }
}

function selectRowsList(t: ReturnType<typeof asTensor>): unknown[] {
  const n = t.shape[0] ?? 0
  const rowsShape = t.shape.slice(1)
  const width = rowsShape.reduce((a, b) => a * b, 1)
  const data = Float64Array.from(toFlat(t))
  if (t.shape.length <= 1) return Array.from(data)
  return Array.from({ length: n }, (_, i) => reshapeNested(data.subarray(i * width, (i + 1) * width), rowsShape))
}

/** Whether a metric input holds strings. */
export function isStringInput(x: MetricInput): x is readonly string[] {
  return Array.isArray(x) && x.length > 0 && x.every((v) => typeof v === 'string') && !isValue(x)
}
