/**
 * Displaying and serialising tensors: `formatTensor` prints a tensor like numpy's `repr` (nested brackets, fixed
 * precision, long axes summarised with `...`), and `tensorToJson` and `tensorFromJson` round-trip tensors as JSON in
 * the format the Python oracle writes.
 */

import { decodeNumber, encodeNumber, WhiteboxError } from '../base/index.ts'
import { cpu } from './cpu.ts'
import { allocate, type DType } from './dtype.ts'
import { sizeOf } from './shape.ts'
import type { Tensor } from './tensor.ts'
import { isTraced, type Value } from './traced.ts'

/** Options for {@link formatTensor}. */
export interface FormatOptions {
  readonly precision?: number
  readonly threshold?: number
  readonly edgeItems?: number
}

function formatNumber(x: number, dtype: DType, precision: number): string {
  if (dtype === 'bool') return x ? 'true' : 'false'
  if (dtype === 'int32') return String(x)
  if (Number.isNaN(x)) return 'nan'
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf'
  if (x === 0) return '0'
  const abs = Math.abs(x)
  if (abs >= 1e6 || abs < 1e-4) return x.toExponential(precision - 1).replace(/\.?0+e/, 'e')
  return String(Number(x.toFixed(precision)))
}

/**
 * Display a tensor like numpy's `repr`: nested brackets, a fixed precision, and long axes summarised with `...`.
 *
 * @param t - The tensor (a traced value is shown by shape and dtype only).
 * @param options - `precision` (default 4), `threshold` (summarise above this many elements, default 1000) and
 *   `edgeItems` (items kept at each end, default 3).
 * @returns The text.
 * @example
 * ```ts
 * import { formatTensor, tensor } from 'whitebox-engine/tensor'
 * formatTensor(tensor([[1, 2.5], [3, 4]])) // => 'tensor([[1, 2.5], [3, 4]])'
 * ```
 */
export function formatTensor(t: Value, options: FormatOptions = {}): string {
  if (isTraced(t)) return `traced(shape=[${t.shape.join(', ')}], dtype=${t.dtype})`
  const precision = options.precision ?? 4
  const summarise = t.data.length > (options.threshold ?? 1000)
  const edge = options.edgeItems ?? 3
  const build = (depth: number, offset: number): string => {
    if (depth === t.shape.length) return formatNumber(t.data[offset] ?? 0, t.dtype, precision)
    const n = t.shape[depth] ?? 0
    const inner = sizeOf(t.shape.slice(depth + 1))
    const items: string[] = []
    for (let i = 0; i < n; i++) {
      if (summarise && n > 2 * edge && i === edge) {
        items.push('...')
        i = n - edge - 1
        continue
      }
      items.push(build(depth + 1, offset + i * inner))
    }
    return `[${items.join(', ')}]`
  }
  const suffix = t.dtype === 'float64' ? '' : `, dtype=${t.dtype}`
  return `tensor(${build(0, 0)}${suffix})`
}

/** A tensor as JSON: dtype, shape and row-major data, with non-finite numbers as strings. */
export interface TensorJson {
  readonly dtype: DType
  readonly shape: readonly number[]
  readonly data: readonly (number | string)[]
}

/**
 * Encode a concrete tensor as JSON (non-finite numbers become `'inf'`, `'-inf'`, `'nan'`).
 *
 * @param t - A concrete tensor.
 * @returns The JSON form.
 * @example
 * ```ts
 * import { tensorToJson, tensor } from 'whitebox-engine/tensor'
 * tensorToJson(tensor([1, Infinity])).data // => [1, 'inf']
 * ```
 */
export function tensorToJson(t: Value): TensorJson {
  if (isTraced(t)) throw new WhiteboxError('not-differentiable', 'tensorToJson needs a concrete tensor')
  return { dtype: t.dtype, shape: [...t.shape], data: Array.from(t.data, encodeNumber) }
}

/**
 * Decode a tensor written by {@link tensorToJson} (or by the Python oracle).
 *
 * @param json - The JSON form.
 * @returns The tensor.
 * @throws WhiteboxError with code `shape` when the data does not fill the shape.
 * @example
 * ```ts
 * import { tensorFromJson, toArray } from 'whitebox-engine/tensor'
 * toArray(tensorFromJson({ dtype: 'float64', shape: [2], data: [1, '-inf'] })) // => [1, -Infinity]
 * ```
 */
export function tensorFromJson(json: TensorJson): Tensor {
  const data = allocate(json.dtype, json.data.length)
  json.data.forEach((v, i) => {
    data[i] = decodeNumber(v)
  })
  return cpu.wrap(json.dtype, json.shape, data)
}
