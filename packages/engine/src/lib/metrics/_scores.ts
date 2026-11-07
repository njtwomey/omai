import { MdmlError, type MetricSpec } from '../base/index.ts'
import { type Label, labelValues, sortLabels } from './_labels.ts'
import type { MetricInput } from './metric.ts'

/** A row-major matrix (or vector) of values. */
export interface Grid {
  readonly shape: readonly number[]
  readonly data: Float64Array
}

/** The spec of a score-based classification metric. */
export const scoreSpec = (
  name: string,
  direction: 'higher' | 'lower' = 'higher',
  concept = 'roc-curve',
): MetricSpec => ({
  name,
  direction,
  range: [0, 1],
  inputs: 'scores',
  category: 'classification',
  concept,
})

/** Binary targets (1 for the positive class) of label input; the positive class defaults to 1 or the larger label. */
export function positives(yTrue: MetricInput, posLabel: Label | undefined): Float64Array {
  const t = labelValues(yTrue)
  let pos = posLabel
  if (pos === undefined) {
    const classes = sortLabels(t)
    if (classes.length > 2) throw new MdmlError('value', 'binary curves need two classes or a posLabel')
    // sklearn's rule: labels within {0, 1} or {-1, 1} take 1 as the positive class even when it is absent (all negatives
    // then have no positives); otherwise the larger of two labels.
    const onlyNegative = classes.length === 1 && (classes[0] === 0 || classes[0] === -1)
    pos = classes.includes(1) || onlyNegative ? 1 : (classes[classes.length - 1] ?? 1)
  }
  return Float64Array.from(t, (v) => (v === pos ? 1 : 0))
}

/** Column j of a matrix. */
export function column(m: Grid, j: number): Float64Array {
  const n = m.shape[0] ?? 0
  const c = m.shape[1] ?? 1
  return Float64Array.from({ length: n }, (_, i) => m.data[i * c + j] ?? 0)
}

const repeatEach = (w: Float64Array, k: number): Float64Array =>
  Float64Array.from({ length: w.length * k }, (_, i) => w[Math.floor(i / k)] ?? 0)

/** Average a binary score over the columns of indicator targets (sklearn's _average_binary_score). */
export function averageBinary(
  t: Grid,
  s: Grid,
  w: Float64Array | undefined,
  average: 'macro' | 'weighted' | 'micro' | 'samples',
  score: (y: Float64Array, s: Float64Array, w: Float64Array | undefined) => number,
): { value: number; perClass: number[] } {
  const n = t.shape[0] ?? 0
  const c = t.shape[1] ?? 1
  if (average === 'micro') {
    return { value: score(t.data, s.data, w === undefined ? undefined : repeatEach(w, c)), perClass: [] }
  }
  if (average === 'samples') {
    const per = Array.from({ length: n }, (_, i) =>
      score(t.data.subarray(i * c, (i + 1) * c), s.data.subarray(i * c, (i + 1) * c), undefined),
    )
    const ws = w ?? new Float64Array(n).fill(1)
    let num = 0
    let den = 0
    for (const [i, v] of per.entries()) {
      num += v * (ws[i] ?? 0)
      den += ws[i] ?? 0
    }
    return { value: num / den, perClass: [] }
  }
  const perClass = Array.from({ length: c }, (_, j) => score(column(t, j), column(s, j), w))
  if (average === 'weighted') {
    const weights = Array.from({ length: c }, (_, j) => {
      let total = 0
      for (let i = 0; i < n; i++) total += (t.data[i * c + j] ?? 0) * (w?.[i] ?? 1)
      return total
    })
    const total = weights.reduce((a, b) => a + b, 0)
    return { value: perClass.reduce((a, v, j) => a + v * (weights[j] ?? 0), 0) / total, perClass }
  }
  return { value: perClass.reduce((a, b) => a + b, 0) / c, perClass }
}

/** One-hot indicator of labels against a class list. */
export function binarise(labels: readonly Label[], classes: readonly Label[]): Grid {
  const data = new Float64Array(labels.length * classes.length)
  for (const [i, l] of labels.entries()) {
    const j = classes.indexOf(l)
    if (j >= 0) data[i * classes.length + j] = 1
  }
  return { shape: [labels.length, classes.length], data }
}
