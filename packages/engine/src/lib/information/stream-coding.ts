import { pairwiseSum, WhiteboxError } from '../base/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { float } from './_shared.ts'

/**
 * The probability model of the stream coders: fixed integer frequencies, or adaptive counts that start at one per
 * symbol and grow by `increment` after each coded symbol (halved when their total passes 2^16), so that encoder and
 * decoder learn the source together.
 */
export type FrequencyModel =
  | { readonly kind: 'static'; readonly frequencies: readonly number[] }
  | { readonly kind: 'adaptive'; readonly alphabetSize: number; readonly increment?: number }

/** The largest total frequency: keeps every product of the 32-bit coders exact in doubles. */
const MAX_TOTAL = 1 << 16

interface Model {
  readonly size: number
  total(): number
  low(s: number): number
  freq(s: number): number
  find(count: number): number
  update(s: number): void
}

function model(spec: FrequencyModel, name: string): Model {
  const adaptive = spec.kind === 'adaptive'
  const counts = adaptive
    ? new Array<number>(spec.alphabetSize).fill(1)
    : spec.frequencies.map((f) => {
        if (!Number.isInteger(f) || f < 0)
          throw new WhiteboxError('value', `${name}: frequencies must be integers >= 0`)
        return f
      })
  if (counts.length === 0) throw new WhiteboxError('value', `${name}: empty alphabet`)
  const increment = adaptive ? (spec.increment ?? 1) : 0
  let cumulative: number[] = []
  const rebuild = (): void => {
    cumulative = [0]
    for (const c of counts) cumulative.push((cumulative[cumulative.length - 1] ?? 0) + c)
  }
  rebuild()
  const total = (): number => cumulative[counts.length] ?? 0
  if (total() > MAX_TOTAL || total() === 0)
    throw new WhiteboxError('value', `${name}: the frequencies must total between 1 and ${MAX_TOTAL}`)
  return {
    size: counts.length,
    total,
    low: (s) => cumulative[s] ?? 0,
    freq: (s) => counts[s] ?? 0,
    find: (count) => {
      let lo = 0
      let hi = counts.length - 1
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1
        if ((cumulative[mid] ?? 0) <= count) lo = mid
        else hi = mid - 1
      }
      return lo
    },
    update: (s) => {
      if (!adaptive) return
      counts[s] = (counts[s] ?? 0) + increment
      if (pairwiseSum(counts) > MAX_TOTAL)
        for (let i = 0; i < counts.length; i++) counts[i] = Math.max(1, (counts[i] ?? 0) >> 1)
      rebuild()
    },
  }
}

function checkSymbol(m: Model, s: number, name: string): void {
  if (!Number.isInteger(s) || s < 0 || s >= m.size || m.freq(s) === 0)
    throw new WhiteboxError('value', `${name}: symbol ${s} has zero probability or is out of range`)
}

/**
 * Integer frequencies proportional to probabilities, summing to `total`, with every positive probability getting at
 * least 1 (largest-remainder rounding): the static model of the stream coders.
 *
 * @param probs - Symbol probabilities or counts.
 * @param total - The frequency total (default 2^16, the most the coders accept).
 * @returns The frequencies.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { frequencyTable } from 'whitebox-engine/information'
 * frequencyTable([0.5, 0.3, 0.2], 10) // => [5, 3, 2]
 * ```
 */
export function frequencyTable(probs: TensorLike, total: number = MAX_TOTAL): number[] {
  const p = toFlat(float(probs))
  const sum = pairwiseSum(p)
  if (p.some((v) => !(v >= 0)) || !(sum > 0)) throw new WhiteboxError('value', 'frequencyTable: invalid probabilities')
  const positive = p.filter((v) => v > 0).length
  if (!Number.isInteger(total) || total < positive || total > MAX_TOTAL)
    throw new WhiteboxError('value', `frequencyTable: total must be an integer in [${positive}, ${MAX_TOTAL}]`)
  const spare = total - positive
  const exact = p.map((v) => (v / sum) * spare)
  const freq = p.map((v, i) => (v > 0 ? 1 + Math.floor(exact[i] ?? 0) : 0))
  let left = total - pairwiseSum(freq)
  const order = exact
    .map((e, i) => [e - Math.floor(e), i] as const)
    .filter(([, i]) => (p[i] ?? 0) > 0)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
  for (const [, i] of order) {
    if (left <= 0) break
    freq[i] = (freq[i] ?? 0) + 1
    left--
  }
  return freq
}

const FULL = 2 ** 32
const HALF = 2 ** 31
const QUARTER = 2 ** 30

/**
 * Arithmetic coding (Witten, Neal and Cleary 1987) with 32-bit integer arithmetic and pending (underflow) bits: the
 * interval [low, high] narrows in proportion to each symbol's frequency, and the common leading bits are emitted as
 * they settle. The output length approaches the information content sum -log2 p(s) plus two bits.
 *
 * @param symbols - Symbols 0..m-1.
 * @param spec - The frequency model.
 * @returns The bits (0 or 1).
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { arithmeticDecode, arithmeticEncode } from 'whitebox-engine/information'
 * const spec = { kind: 'static', frequencies: [3, 1] } as const
 * const bits = arithmeticEncode([0, 0, 1, 0, 0, 0], spec)
 * arithmeticDecode(bits, 6, spec) // => [0, 0, 1, 0, 0, 0]
 * ```
 */
export function arithmeticEncode(symbols: readonly number[], spec: FrequencyModel): number[] {
  const m = model(spec, 'arithmeticEncode')
  const out: number[] = []
  let low = 0
  let high = FULL - 1
  let pending = 0
  const emit = (bit: number): void => {
    out.push(bit)
    for (; pending > 0; pending--) out.push(1 - bit)
  }
  for (const s of symbols) {
    checkSymbol(m, s, 'arithmeticEncode')
    const range = high - low + 1
    const total = m.total()
    high = low + Math.floor((range * (m.low(s) + m.freq(s))) / total) - 1
    low = low + Math.floor((range * m.low(s)) / total)
    for (;;) {
      if (high < HALF) emit(0)
      else if (low >= HALF) {
        emit(1)
        low -= HALF
        high -= HALF
      } else if (low >= QUARTER && high < HALF + QUARTER) {
        pending++
        low -= QUARTER
        high -= QUARTER
      } else break
      low = 2 * low
      high = 2 * high + 1
    }
    m.update(s)
  }
  pending++
  emit(low < QUARTER ? 0 : 1)
  return out
}

/**
 * Decode {@link arithmeticEncode}'s bits: the decoder mirrors the encoder's interval and reads bits as the encoder
 * wrote them (missing trailing bits count as 0).
 *
 * @param bits - The encoded bits.
 * @param length - The number of symbols to decode.
 * @param spec - The same frequency model as the encoder.
 * @returns The symbols.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { arithmeticDecode, arithmeticEncode } from 'whitebox-engine/information'
 * const spec = { kind: 'adaptive', alphabetSize: 3 } as const
 * arithmeticDecode(arithmeticEncode([2, 2, 2, 1], spec), 4, spec) // => [2, 2, 2, 1]
 * ```
 */
export function arithmeticDecode(bits: readonly number[], length: number, spec: FrequencyModel): number[] {
  const m = model(spec, 'arithmeticDecode')
  let position = 0
  const next = (): number => (bits[position++] ?? 0) & 1
  let low = 0
  let high = FULL - 1
  let value = 0
  for (let i = 0; i < 32; i++) value = 2 * value + next()
  const out: number[] = []
  for (let i = 0; i < length; i++) {
    const range = high - low + 1
    const total = m.total()
    const count = Math.floor(((value - low + 1) * total - 1) / range)
    const s = m.find(count)
    out.push(s)
    high = low + Math.floor((range * (m.low(s) + m.freq(s))) / total) - 1
    low = low + Math.floor((range * m.low(s)) / total)
    for (;;) {
      // Below one half nothing is removed; above it the top bit goes; straddling the middle the second bit goes.
      const offset = high < HALF ? 0 : low >= HALF ? HALF : low >= QUARTER && high < HALF + QUARTER ? QUARTER : -1
      if (offset < 0) break
      low -= offset
      high -= offset
      value -= offset
      low = 2 * low
      high = 2 * high + 1
      value = 2 * value + next()
    }
    m.update(s)
  }
  return out
}

const TOP = 2 ** 24
const BOTTOM = 2 ** 16
const u32 = (x: number): number => x >>> 0

/**
 * Range coding (Martin 1979) in Subbotin's carry-less form: a 32-bit range is split in proportion to the frequencies
 * and whole bytes are emitted once the top byte of the interval settles; the range is clipped instead of propagating
 * carries. Byte-oriented and fast; as efficient as arithmetic coding to within a few bytes.
 *
 * @param symbols - Symbols 0..m-1.
 * @param spec - The frequency model (totals up to 2^16).
 * @returns The bytes.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { rangeDecode, rangeEncode } from 'whitebox-engine/information'
 * const spec = { kind: 'static', frequencies: [6, 1, 1] } as const
 * rangeDecode(rangeEncode([0, 0, 1, 0, 2], spec), 5, spec) // => [0, 0, 1, 0, 2]
 * ```
 */
export function rangeEncode(symbols: readonly number[], spec: FrequencyModel): Uint8Array {
  const m = model(spec, 'rangeEncode')
  const out: number[] = []
  let low = 0
  let range = FULL - 1
  const normalise = (): void => {
    for (;;) {
      if (u32(low ^ u32(low + range)) >= TOP) {
        if (range >= BOTTOM) break
        range = u32(-low) & (BOTTOM - 1)
      }
      out.push(low >>> 24)
      low = u32(low * 256)
      range = u32(range * 256)
    }
  }
  for (const s of symbols) {
    checkSymbol(m, s, 'rangeEncode')
    range = Math.floor(range / m.total())
    low = u32(low + m.low(s) * range)
    range = range * m.freq(s)
    normalise()
    m.update(s)
  }
  for (let i = 0; i < 4; i++) {
    out.push(low >>> 24)
    low = u32(low * 256)
  }
  return Uint8Array.from(out)
}

/**
 * Decode {@link rangeEncode}'s bytes.
 *
 * @param bytes - The encoded bytes.
 * @param length - The number of symbols.
 * @param spec - The same frequency model as the encoder.
 * @returns The symbols.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { rangeDecode, rangeEncode } from 'whitebox-engine/information'
 * const spec = { kind: 'adaptive', alphabetSize: 2 } as const
 * rangeDecode(rangeEncode([1, 1, 1, 0, 1], spec), 5, spec) // => [1, 1, 1, 0, 1]
 * ```
 */
export function rangeDecode(bytes: ArrayLike<number>, length: number, spec: FrequencyModel): number[] {
  const m = model(spec, 'rangeDecode')
  let position = 0
  const next = (): number => (bytes[position++] ?? 0) & 0xff
  let low = 0
  let range = FULL - 1
  let code = 0
  for (let i = 0; i < 4; i++) code = u32(code * 256 + next())
  const out: number[] = []
  for (let i = 0; i < length; i++) {
    const total = m.total()
    range = Math.floor(range / total)
    const s = m.find(Math.min(total - 1, Math.floor(u32(code - low) / range)))
    out.push(s)
    low = u32(low + m.low(s) * range)
    range = range * m.freq(s)
    for (;;) {
      if (u32(low ^ u32(low + range)) >= TOP) {
        if (range >= BOTTOM) break
        range = u32(-low) & (BOTTOM - 1)
      }
      code = u32(code * 256 + next())
      low = u32(low * 256)
      range = u32(range * 256)
    }
    m.update(s)
  }
  return out
}
