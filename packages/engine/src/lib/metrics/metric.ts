import { estimate, type MetricResult, type MetricSpec, WhiteboxError } from '../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** Options every metric accepts. */
export interface MetricOptions {
  /** Per-sample weights (default all ones). */
  readonly sampleWeight?: TensorLike
}

/** A metric function with its specification attached. */
export type Metric<O extends MetricOptions = MetricOptions> = ((
  yTrue: TensorLike,
  yPred: TensorLike,
  options?: O,
) => MetricResult) & { readonly spec: MetricSpec }

/** Targets and predictions as columns: `n` samples by `k` outputs, row-major, with weights. */
export interface Columns {
  readonly n: number
  readonly k: number
  readonly yTrue: Float64Array
  readonly yPred: Float64Array
  readonly weight: Float64Array
}

/** Read 1-D or 2-D targets and predictions (and weights) into columns, checking their shapes. */
export function columns(name: string, yTrue: TensorLike, yPred: TensorLike, options: MetricOptions = {}): Columns {
  const t = asTensor(yTrue)
  const p = asTensor(yPred)
  if (t.shape.length !== p.shape.length || t.shape.some((d, i) => d !== p.shape[i])) {
    throw new WhiteboxError('shape', `${name}: yTrue [${t.shape.join(', ')}] and yPred [${p.shape.join(', ')}] differ`)
  }
  if (t.shape.length > 2) throw new WhiteboxError('shape', `${name}: expected 1-D or 2-D targets`)
  const n = t.shape[0] ?? 1
  const k = t.shape.length === 2 ? (t.shape[1] ?? 1) : 1
  const weight =
    options.sampleWeight === undefined
      ? new Float64Array(n).fill(1)
      : Float64Array.from(toFlat(asTensor(options.sampleWeight)))
  if (weight.length !== n) throw new WhiteboxError('shape', `${name}: sampleWeight must have one entry per sample`)
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
