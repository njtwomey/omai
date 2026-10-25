// Small helpers shared by the information files: normalisation, logarithm bases and label coding. Not exported.
import { MdmlError, pairwiseSum } from '../base/index.ts'
import { asFloat, div, sum, type TensorLike, toFlat, type Value } from '../tensor/index.ts'

/** A float value of the input. */
export function float(x: TensorLike): Value {
  return asFloat(x)
}

/** Divide by the sum along the axes (all axes when undefined). */
export function normaliseAlong(p: Value, axis?: number | readonly number[]): Value {
  return div(p, sum(p, axis, true))
}

/** Convert a quantity in nats to the given base (nats when undefined). */
export function inBase(v: Value, base: number | undefined): Value {
  if (base === undefined) return v
  if (!(base > 0) || base === 1)
    throw new MdmlError('value', `information: base must be positive and not 1, got ${base}`)
  return div(v, Math.log(base))
}

/** The natural logarithm of the base (1 for nats). */
export function logOfBase(base: number | undefined): number {
  if (base === undefined) return 1
  if (!(base > 0) || base === 1)
    throw new MdmlError('value', `information: base must be positive and not 1, got ${base}`)
  return Math.log(base)
}

/** Discrete labels: numbers, strings, or a tensor of numbers. */
export type Labels = readonly (number | string)[] | TensorLike

/** Map labels to codes 0..m-1 in order of first appearance. */
export function labelCodes(labels: Labels, name: string): { readonly codes: Int32Array; readonly m: number } {
  const xs: readonly (number | string)[] =
    Array.isArray(labels) && labels.every((v) => typeof v === 'number' || typeof v === 'string')
      ? (labels as readonly (number | string)[])
      : toFlat(asFloat(labels as TensorLike))
  const index = new Map<number | string, number>()
  const codes = new Int32Array(xs.length)
  xs.forEach((x, i) => {
    let c = index.get(x)
    if (c === undefined) {
      c = index.size
      index.set(x, c)
    }
    codes[i] = c
  })
  if (xs.length === 0) throw new MdmlError('value', `${name}: no samples`)
  return { codes, m: index.size }
}

/** Codes of the joint label (a, b): a * mb + b, recoded densely. */
export function jointCodes(...parts: readonly { readonly codes: Int32Array; readonly m: number }[]): {
  readonly codes: Int32Array
  readonly m: number
} {
  const n = parts[0]?.codes.length ?? 0
  const index = new Map<string, number>()
  const codes = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const key = parts.map((p) => p.codes[i] ?? 0).join(',')
    let c = index.get(key)
    if (c === undefined) {
      c = index.size
      index.set(key, c)
    }
    codes[i] = c
  }
  return { codes, m: index.size }
}

/** Counts of each code. */
export function countsOf(codes: Int32Array, m: number): Float64Array {
  const counts = new Float64Array(m)
  for (const c of codes) counts[c] = (counts[c] ?? 0) + 1
  return counts
}

/** Plug-in entropy in nats of a vector of counts (zero counts ignored). */
export function entropyOfCounts(counts: ArrayLike<number>): number {
  const n = pairwiseSum(counts)
  let h = 0
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i] ?? 0
    if (c > 0) h -= (c / n) * Math.log(c / n)
  }
  return h
}
