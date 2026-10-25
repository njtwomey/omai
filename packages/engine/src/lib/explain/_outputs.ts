// Shared by the perturbation explainers (private to lib/explain): reading a model's target output for many rows, and
// broadcasting a baseline to every row.
import { MdmlError } from '../base/index.ts'
import { asTensor, fromStorage, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Target } from './gradients.ts'
import type { PredictFunction } from './shapley.ts'

/** The model's target output at each of m rows of d features. */
export function targetOutputs(
  predict: PredictFunction,
  data: Float64Array,
  m: number,
  d: number,
  target: Target | undefined,
  rowOf: (i: number) => number = (i) => i,
): Float64Array {
  const out = predict(fromStorage(data, [m, d], 'float64'))
  const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  const v = toFlat(out)
  return Float64Array.from({ length: m }, (_, i) => {
    const t = typeof target === 'number' ? target : (target?.[rowOf(i)] ?? 0)
    if (!(Number.isInteger(t) && t >= 0 && t < k)) throw new MdmlError('value', `target ${t} outside [0, ${k})`)
    return v[i * k + t] ?? 0
  })
}

/** A baseline as one value per (row, feature): a number, a row `[d]` or one row per input `[n, d]`. */
export function baselineRows(
  baseline: TensorLike | number | undefined,
  n: number,
  d: number,
  who: string,
): Float64Array {
  if (baseline === undefined) return new Float64Array(n * d)
  if (typeof baseline === 'number') return new Float64Array(n * d).fill(baseline)
  const b = toFlat(asTensor(baseline))
  if (b.length === n * d) return Float64Array.from(b)
  if (b.length === d) return Float64Array.from({ length: n * d }, (_, i) => b[i % d] ?? 0)
  throw new MdmlError('shape', `${who}: the baseline must be a number, [d] or [n, d]`)
}
