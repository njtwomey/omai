import { OmaiError } from '../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { isStringInput, type MetricInput, sampleWeights } from './metric.ts'

/** A class label: a number or a string. */
export type Label = number | string

/** The values of a label input: numbers or strings. */
export function labelValues(x: MetricInput): Label[] {
  const values = isStringInput(x) ? [...x] : toFlat(asTensor(x as TensorLike))
  if (values.length === 0) throw new OmaiError('value', 'a metric of labels needs at least one sample')
  // A NaN is not a label: as a Map key it would equal itself, so a missing label would count as a class and as correct.
  if (values.some((v) => typeof v === 'number' && Number.isNaN(v)))
    throw new OmaiError('value', 'a metric of labels does not accept NaN labels (drop or encode the missing ones)')
  return values
}

/** Reject `sampleWeight` for a metric that has no weighted definition (silently ignoring it would be a wrong answer). */
export function rejectSampleWeight(name: string, options: { readonly sampleWeight?: TensorLike } | undefined): void {
  if (options?.sampleWeight !== undefined)
    throw new OmaiError('value', `${name}: sampleWeight is not supported by this metric`)
}

/** Sort labels as sklearn's unique_labels does: numbers numerically, strings lexicographically. */
export function sortLabels(labels: Iterable<Label>): Label[] {
  const unique = [...new Set(labels)]
  return unique.sort((a, b) => {
    if (typeof a === 'number' && typeof b === 'number') return a - b
    return String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0
  })
}

/** Targets and predictions encoded as indices into a class list, with weights. */
export interface EncodedLabels {
  readonly classes: readonly Label[]
  /** Index of each sample's true class in `classes`, or -1 when it is not listed. */
  readonly t: Int32Array
  readonly p: Int32Array
  readonly w: Float64Array
  readonly n: number
}

/** Encode label targets and predictions against the sorted union of their classes (or the given labels). */
export function encodeLabels(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  options: { readonly labels?: readonly Label[]; readonly sampleWeight?: TensorLike } = {},
): EncodedLabels {
  const t = labelValues(yTrue)
  const p = labelValues(yPred)
  if (t.length !== p.length) throw new OmaiError('shape', `${name}: yTrue and yPred have different lengths`)
  const classes = options.labels === undefined ? sortLabels([...t, ...p]) : [...options.labels]
  const index = new Map<Label, number>(classes.map((c, i) => [c, i]))
  return {
    classes,
    t: Int32Array.from(t, (v) => index.get(v) ?? -1),
    p: Int32Array.from(p, (v) => index.get(v) ?? -1),
    w: sampleWeights(name, t.length, options.sampleWeight),
    n: t.length,
  }
}

/** Per-class one-vs-rest counts (weighted): true positives, false positives, false negatives, true negatives. */
export interface ClassCounts {
  readonly tp: Float64Array
  readonly fp: Float64Array
  readonly fn: Float64Array
  readonly tn: Float64Array
}

/** One-vs-rest counts for each listed class of encoded multi-class labels. */
export function classCounts(e: EncodedLabels): ClassCounts {
  const k = e.classes.length
  const tp = new Float64Array(k)
  const fp = new Float64Array(k)
  const fn = new Float64Array(k)
  let total = 0
  for (let i = 0; i < e.n; i++) {
    const w = e.w[i] ?? 0
    const a = e.t[i] ?? -1
    const b = e.p[i] ?? -1
    total += w
    if (a >= 0 && a === b) tp[a] = (tp[a] ?? 0) + w
    else {
      if (b >= 0) fp[b] = (fp[b] ?? 0) + w
      if (a >= 0) fn[a] = (fn[a] ?? 0) + w
    }
  }
  const tn = Float64Array.from(tp, (v, c) => total - v - (fp[c] ?? 0) - (fn[c] ?? 0))
  return { tp, fp, fn, tn }
}

/** A multilabel indicator matrix `[N, L]` read as rows of 0/1 values. */
export interface Indicator {
  readonly n: number
  readonly l: number
  readonly data: Float64Array
}

/** Read a 2-D 0/1 indicator matrix. */
export function indicator(name: string, x: TensorLike): Indicator {
  const t = asTensor(x)
  if (t.shape.length !== 2) throw new OmaiError('shape', `${name}: expected an [N, L] indicator matrix`)
  return { n: t.shape[0] ?? 0, l: t.shape[1] ?? 0, data: Float64Array.from(toFlat(t)) }
}

/** Whether an input is a 2-D multilabel indicator matrix. */
export function isMultilabel(x: MetricInput): boolean {
  if (isStringInput(x)) return false
  return asTensor(x as TensorLike).shape.length === 2
}

/** Per-label counts of multilabel indicator targets and predictions (weighted). */
export function multilabelCounts(
  t: Indicator,
  p: Indicator,
  w: Float64Array,
  labels?: readonly number[],
): ClassCounts & { readonly labels: readonly number[] } {
  const cols = labels ?? Array.from({ length: t.l }, (_, j) => j)
  const k = cols.length
  const tp = new Float64Array(k)
  const fp = new Float64Array(k)
  const fn = new Float64Array(k)
  const tn = new Float64Array(k)
  for (let i = 0; i < t.n; i++) {
    const wi = w[i] ?? 0
    for (const [c, j] of cols.entries()) {
      const a = (t.data[i * t.l + j] ?? 0) > 0
      const b = (p.data[i * p.l + j] ?? 0) > 0
      if (a && b) tp[c] = (tp[c] ?? 0) + wi
      else if (b) fp[c] = (fp[c] ?? 0) + wi
      else if (a) fn[c] = (fn[c] ?? 0) + wi
      else tn[c] = (tn[c] ?? 0) + wi
    }
  }
  return { tp, fp, fn, tn, labels: cols }
}

/** a / b, or `zeroDivision` when b is zero. */
export function ratio(a: number, b: number, zeroDivision: number): number {
  return b === 0 ? zeroDivision : a / b
}
