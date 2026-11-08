/**
 * Variable-length codes for integers: Elias's universal gamma, delta and omega codes, Golomb and Rice codes with the
 * parameter that is optimal for a geometric distribution, and run-length encoding, each with its decoder.
 */

import { OmaiError } from '../base/index.ts'

/** The Elias universal codes for positive integers. */
export type EliasKind = 'gamma' | 'delta' | 'omega'

const binary = (n: number): number[] => Array.from(n.toString(2), (c) => (c === '1' ? 1 : 0))

function checkInteger(n: number, min: number, name: string): void {
  if (!Number.isSafeInteger(n) || n < min) throw new OmaiError('value', `${name}: ${n} is not an integer >= ${min}`)
}

function eliasOne(n: number, kind: EliasKind): number[] {
  if (kind === 'gamma') {
    const b = binary(n)
    return [...new Array<number>(b.length - 1).fill(0), ...b]
  }
  if (kind === 'delta') {
    const b = binary(n)
    return [...eliasOne(b.length, 'gamma'), ...b.slice(1)]
  }
  const out: number[] = [0]
  let v = n
  while (v > 1) {
    const b = binary(v)
    out.unshift(...b)
    v = b.length - 1
  }
  return out
}

/**
 * Elias's universal codes of positive integers (Elias 1975), each within a constant factor of the optimal length for
 * any decreasing distribution: gamma writes floor(log2 n) zeros and then n in binary; delta writes the binary length
 * of n in gamma and then n without its leading 1; omega writes n in binary preceded, recursively, by its length minus
 * one, and ends with a 0.
 *
 * @param values - Integers >= 1.
 * @param kind - `'gamma'`, `'delta'` or `'omega'`.
 * @returns The concatenated codewords as bits.
 * @concept universal-codes
 * @example
 * ```ts
 * import { eliasEncode } from 'omai-engine/information'
 * eliasEncode([1, 5], 'gamma') // => [1, 0, 0, 1, 0, 1]
 * eliasEncode([10], 'delta') // => [0, 0, 1, 0, 0, 0, 1, 0]
 * eliasEncode([10], 'omega') // => [1, 1, 1, 0, 1, 0, 0]
 * ```
 */
export function eliasEncode(values: readonly number[], kind: EliasKind): number[] {
  const out: number[] = []
  for (const n of values) {
    checkInteger(n, 1, 'eliasEncode')
    out.push(...eliasOne(n, kind))
  }
  return out
}

/**
 * Decode a sequence of Elias codewords.
 *
 * @param bits - The bits.
 * @param kind - The code used to encode them.
 * @returns The integers.
 * @throws OmaiError with code `value` when the bits end inside a codeword.
 * @concept universal-codes
 * @example
 * ```ts
 * import { eliasDecode, eliasEncode } from 'omai-engine/information'
 * eliasDecode(eliasEncode([1, 2, 300], 'omega'), 'omega') // => [1, 2, 300]
 * ```
 */
export function eliasDecode(bits: readonly number[], kind: EliasKind): number[] {
  let i = 0
  const take = (k: number): number => {
    if (i + k > bits.length) throw new OmaiError('value', 'eliasDecode: the bits end inside a codeword')
    let v = 0
    for (let j = 0; j < k; j++) v = 2 * v + ((bits[i++] ?? 0) & 1)
    return v
  }
  const gamma = (): number => {
    let zeros = 0
    while (i < bits.length && bits[i] === 0) {
      zeros++
      i++
    }
    return take(zeros + 1)
  }
  const out: number[] = []
  while (i < bits.length) {
    if (kind === 'gamma') out.push(gamma())
    else if (kind === 'delta') {
      const length = gamma()
      out.push(2 ** (length - 1) + take(length - 1))
    } else {
      let n = 1
      for (;;) {
        if (i >= bits.length) throw new OmaiError('value', 'eliasDecode: the bits end inside a codeword')
        if (bits[i] === 0) {
          i++
          break
        }
        n = take(n + 1)
      }
      out.push(n)
    }
  }
  return out
}

/**
 * Golomb coding of non-negative integers with parameter m (Golomb 1966): the quotient floor(n / m) in unary (ones
 * ended by a zero) and the remainder in truncated binary. Optimal for geometric sources with the matching m; Rice
 * coding is the case of m a power of two, where the remainder is plain binary.
 *
 * @param values - Integers >= 0.
 * @param m - The parameter, an integer >= 1.
 * @returns The bits.
 * @concept universal-codes
 * @example
 * ```ts
 * import { golombEncode } from 'omai-engine/information'
 * golombEncode([7], 3) // => [1, 1, 0, 1, 0]
 * golombEncode([5], 4) // => [1, 0, 0, 1]
 * ```
 */
export function golombEncode(values: readonly number[], m: number): number[] {
  checkInteger(m, 1, 'golombEncode')
  const b = Math.ceil(Math.log2(m))
  const cutoff = 2 ** b - m
  const out: number[] = []
  for (const n of values) {
    checkInteger(n, 0, 'golombEncode')
    const q = Math.floor(n / m)
    const r = n - q * m
    for (let j = 0; j < q; j++) out.push(1)
    out.push(0)
    const [value, width] = r < cutoff ? [r, b - 1] : [r + cutoff, b]
    for (let j = width - 1; j >= 0; j--) out.push(Math.floor(value / 2 ** j) % 2)
  }
  return out
}

/**
 * Decode Golomb (or Rice) codewords.
 *
 * @param bits - The bits.
 * @param m - The parameter used to encode them.
 * @returns The integers.
 * @throws OmaiError with code `value` when the bits end inside a codeword.
 * @concept universal-codes
 * @example
 * ```ts
 * import { golombDecode, golombEncode } from 'omai-engine/information'
 * golombDecode(golombEncode([0, 4, 9, 2], 5), 5) // => [0, 4, 9, 2]
 * ```
 */
export function golombDecode(bits: readonly number[], m: number): number[] {
  checkInteger(m, 1, 'golombDecode')
  const b = Math.ceil(Math.log2(m))
  const cutoff = 2 ** b - m
  let i = 0
  const bit = (): number => {
    if (i >= bits.length) throw new OmaiError('value', 'golombDecode: the bits end inside a codeword')
    return (bits[i++] ?? 0) & 1
  }
  const out: number[] = []
  while (i < bits.length) {
    let q = 0
    while (bit() === 1) q++
    let r = 0
    for (let j = 0; j < b - 1; j++) r = 2 * r + bit()
    if (b > 0 && r >= cutoff) r = 2 * r + bit() - cutoff
    out.push(q * m + r)
  }
  return out
}

/**
 * The Golomb parameter that is optimal for the geometric distribution P(n) = (1 - theta) theta^n (Gallager and van
 * Voorhis 1975): the smallest m with theta^m + theta^(m + 1) <= 1.
 *
 * @param theta - The ratio P(n + 1) / P(n), in (0, 1).
 * @returns The optimal m.
 * @concept universal-codes
 * @example
 * ```ts
 * import { golombParameter } from 'omai-engine/information'
 * golombParameter(0.5) // => 1
 * golombParameter(0.9) // => 7
 * ```
 */
export function golombParameter(theta: number): number {
  if (!(theta > 0 && theta < 1)) throw new OmaiError('value', 'golombParameter: theta must lie in (0, 1)')
  return Math.max(1, Math.ceil(-Math.log(1 + theta) / Math.log(theta) - 1e-12))
}

/** Runs of equal values: run values and their lengths. */
export interface Runs<T> {
  readonly values: readonly T[]
  readonly lengths: readonly number[]
}

/**
 * Run-length encoding: each maximal run of equal values becomes a (value, length) pair.
 *
 * @param xs - The sequence.
 * @returns The runs.
 * @concept run-length-coding
 * @example
 * ```ts
 * import { runLengthEncode } from 'omai-engine/information'
 * runLengthEncode([0, 0, 0, 1, 1, 0]) // => { values: [0, 1, 0], lengths: [3, 2, 1] }
 * ```
 */
export function runLengthEncode<T extends number | string>(xs: readonly T[]): Runs<T> {
  const values: T[] = []
  const lengths: number[] = []
  for (const x of xs) {
    const last = lengths.length - 1
    if (last >= 0 && values[last] === x) lengths[last] = (lengths[last] ?? 0) + 1
    else {
      values.push(x)
      lengths.push(1)
    }
  }
  return { values, lengths }
}

/**
 * Expand runs back into the sequence.
 *
 * @param runs - Run values and lengths.
 * @returns The sequence.
 * @concept run-length-coding
 * @example
 * ```ts
 * import { runLengthDecode } from 'omai-engine/information'
 * runLengthDecode({ values: ['a', 'b'], lengths: [2, 3] }) // => ['a', 'a', 'b', 'b', 'b']
 * ```
 */
export function runLengthDecode<T extends number | string>(runs: Runs<T>): T[] {
  if (runs.values.length !== runs.lengths.length) throw new OmaiError('shape', 'runLengthDecode: lengths differ')
  const out: T[] = []
  runs.values.forEach((v, i) => {
    const n = runs.lengths[i] ?? 0
    checkInteger(n, 0, 'runLengthDecode')
    for (let j = 0; j < n; j++) out.push(v)
  })
  return out
}
