/**
 * Stream (entropy) coders that approach the entropy bound to within a fraction of a bit per message: arithmetic coding
 * with 32-bit integers (Witten, Neal and Cleary) and carry-less range coding, each with its decoder, driven by a fixed
 * frequency table or by adaptive counts. Each coder also exists as a traced algorithm, one symbol per step, whose state
 * (the interval, the bits so far, the counts of an adaptive model) serialises, so a session can scrub through the
 * narrowing of the code interval.
 */

import { MdmlError, pairwiseSum } from '../base/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  event,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
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
        if (!Number.isInteger(f) || f < 0) throw new MdmlError('value', `${name}: frequencies must be integers >= 0`)
        return f
      })
  if (counts.length === 0) throw new MdmlError('value', `${name}: empty alphabet`)
  const increment = adaptive ? (spec.increment ?? 1) : 0
  let cumulative: number[] = []
  const rebuild = (): void => {
    cumulative = [0]
    for (const c of counts) cumulative.push((cumulative[cumulative.length - 1] ?? 0) + c)
  }
  rebuild()
  const total = (): number => cumulative[counts.length] ?? 0
  if (total() > MAX_TOTAL || total() === 0)
    throw new MdmlError('value', `${name}: the frequencies must total between 1 and ${MAX_TOTAL}`)
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
      const next = bump(counts, s, increment)
      for (let i = 0; i < counts.length; i++) counts[i] = next[i] ?? 0
      rebuild()
    },
  }
}

/** The adaptive update of the counts after coding symbol `s`: add `increment`, and halve every count (keeping at least 1) when the total passes the limit. */
function bump(counts: readonly number[], s: number, increment: number): number[] {
  const next = counts.slice()
  next[s] = (next[s] ?? 0) + increment
  if (pairwiseSum(next) > MAX_TOTAL) for (let i = 0; i < next.length; i++) next[i] = Math.max(1, (next[i] ?? 0) >> 1)
  return next
}

/** The cumulative frequency below a symbol, its frequency and the total. */
interface SymbolRange {
  readonly low: number
  readonly freq: number
  readonly total: number
}

function rangeOf(counts: readonly number[], s: number): SymbolRange {
  let low = 0
  for (let i = 0; i < s; i++) low += counts[i] ?? 0
  return { low, freq: counts[s] ?? 0, total: pairwiseSum(counts) }
}

/** The symbol whose cumulative interval contains `count`. */
function symbolAt(counts: readonly number[], count: number): number {
  let acc = 0
  for (let i = 0; i < counts.length; i++) {
    acc += counts[i] ?? 0
    if (count < acc) return i
  }
  return counts.length - 1
}

function startingCounts(spec: FrequencyModel, name: string): number[] {
  const m = model(spec, name)
  return Array.from({ length: m.size }, (_, i) => m.freq(i))
}

function checkSymbolIn(counts: readonly number[], s: number, name: string): void {
  if (!Number.isInteger(s) || s < 0 || s >= counts.length || (counts[s] ?? 0) === 0)
    throw new MdmlError('value', `${name}: symbol ${s} has zero probability or is out of range`)
}

function checkSymbol(m: Model, s: number, name: string): void {
  if (!Number.isInteger(s) || s < 0 || s >= m.size || m.freq(s) === 0)
    throw new MdmlError('value', `${name}: symbol ${s} has zero probability or is out of range`)
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
 * import { frequencyTable } from 'mdml-engine/information'
 * frequencyTable([0.5, 0.3, 0.2], 10) // => [5, 3, 2]
 * ```
 */
export function frequencyTable(probs: TensorLike, total: number = MAX_TOTAL): number[] {
  const p = toFlat(float(probs))
  const sum = pairwiseSum(p)
  if (p.some((v) => !(v >= 0)) || !(sum > 0)) throw new MdmlError('value', 'frequencyTable: invalid probabilities')
  const positive = p.filter((v) => v > 0).length
  if (!Number.isInteger(total) || total < positive || total > MAX_TOTAL)
    throw new MdmlError('value', `frequencyTable: total must be an integer in [${positive}, ${MAX_TOTAL}]`)
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
 * import { arithmeticDecode, arithmeticEncode } from 'mdml-engine/information'
 * const spec = { kind: 'static', frequencies: [3, 1] } as const
 * const bits = arithmeticEncode([0, 0, 1, 0, 0, 0], spec)
 * arithmeticDecode(bits, 6, spec) // => [0, 0, 1, 0, 0, 0]
 * ```
 */
export function arithmeticEncode(symbols: readonly number[], spec: FrequencyModel): number[] {
  const m = model(spec, 'arithmeticEncode')
  const out: number[] = []
  let state: ArithmeticCore = { low: 0, high: FULL - 1, pending: 0 }
  for (const s of symbols) {
    checkSymbol(m, s, 'arithmeticEncode')
    state = arithmeticStep(state, { low: m.low(s), freq: m.freq(s), total: m.total() }, out)
    m.update(s)
  }
  arithmeticFinish(state, out)
  return out
}

/** The registers of the arithmetic coder: the interval [low, high] in 32 bits and the count of undecided bits. */
interface ArithmeticCore {
  readonly low: number
  readonly high: number
  readonly pending: number
}

/** Narrow the interval to a symbol's share and emit the bits that settle; appends to `out`. */
function arithmeticStep(state: ArithmeticCore, r: SymbolRange, out: number[]): ArithmeticCore {
  let { low, high, pending } = state
  const emit = (bit: number): void => {
    out.push(bit)
    for (; pending > 0; pending--) out.push(1 - bit)
  }
  const range = high - low + 1
  high = low + Math.floor((range * (r.low + r.freq)) / r.total) - 1
  low = low + Math.floor((range * r.low) / r.total)
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
  return { low, high, pending }
}

/** The bits that pin the final interval down; appends to `out`. */
function arithmeticFinish(state: ArithmeticCore, out: number[]): void {
  const bit = state.low < QUARTER ? 0 : 1
  out.push(bit)
  for (let p = 0; p < state.pending + 1; p++) out.push(1 - bit)
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
 * import { arithmeticDecode, arithmeticEncode } from 'mdml-engine/information'
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
 * import { rangeDecode, rangeEncode } from 'mdml-engine/information'
 * const spec = { kind: 'static', frequencies: [6, 1, 1] } as const
 * rangeDecode(rangeEncode([0, 0, 1, 0, 2], spec), 5, spec) // => [0, 0, 1, 0, 2]
 * ```
 */
export function rangeEncode(symbols: readonly number[], spec: FrequencyModel): Uint8Array {
  const m = model(spec, 'rangeEncode')
  const out: number[] = []
  let state: RangeCore = { low: 0, range: FULL - 1 }
  for (const s of symbols) {
    checkSymbol(m, s, 'rangeEncode')
    state = rangeStep(state, { low: m.low(s), freq: m.freq(s), total: m.total() }, out)
    m.update(s)
  }
  rangeFinish(state, out)
  return Uint8Array.from(out)
}

/** The registers of the carry-less range coder. */
interface RangeCore {
  readonly low: number
  readonly range: number
}

/** Split the range in proportion to a symbol's frequency and shift out the bytes that settle; appends to `out`. */
function rangeStep(state: RangeCore, r: SymbolRange, out: number[]): RangeCore {
  let range = Math.floor(state.range / r.total)
  let low = u32(state.low + r.low * range)
  range = range * r.freq
  for (;;) {
    if (u32(low ^ u32(low + range)) >= TOP) {
      if (range >= BOTTOM) break
      range = u32(-low) & (BOTTOM - 1)
    }
    out.push(low >>> 24)
    low = u32(low * 256)
    range = u32(range * 256)
  }
  return { low, range }
}

/** Flush the four bytes of `low`; appends to `out`. */
function rangeFinish(state: RangeCore, out: number[]): void {
  let low = state.low
  for (let i = 0; i < 4; i++) {
    out.push(low >>> 24)
    low = u32(low * 256)
  }
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
 * import { rangeDecode, rangeEncode } from 'mdml-engine/information'
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

const log2 = (x: number): number => Math.log2(x)

/** The paths of the stepped encoders. */
export type RangeEncoderPath =
  | 'symbol/index'
  | 'interval/low'
  | 'interval/high'
  | 'interval/start'
  | 'code/bits'
  | 'code/ideal'
  | 'code/output'
  | 'model/frequencies'
  | 'code/done'

/** The paths of the stepped arithmetic encoder: those of the range coder and the count of undecided bits. */
export type ArithmeticEncoderPath = RangeEncoderPath | 'code/pending'

const encoderPaths = {
  'symbol/index': scalar({ description: 'symbols coded so far' }),
  'interval/low': scalar({ description: 'the low end of the working interval, as a share of its 32-bit register' }),
  'interval/high': scalar({ description: 'the high end of the working interval, as a share of its register' }),
  'interval/start': scalar({
    description:
      'the start of the interval of the whole message in [0, 1) (a double: exact only while its width exceeds about 2^-50)',
  }),
  'code/bits': scalar({ description: 'bits written so far (including undecided ones)', unit: 'bits' }),
  'code/ideal': scalar({
    description: 'the information content so far, the sum of -log2 of the symbols probabilities',
    unit: 'bits',
  }),
  'code/output': tensor({
    role: 'output',
    shape: ['bits'],
    description: 'the code so far (bits for the arithmetic coder, bytes for the range coder)',
  }),
  'model/frequencies': tensor({
    role: 'state',
    shape: ['symbols'],
    description: 'the frequency of each symbol now (adaptive models change)',
  }),
  'code/done': event({ context: { bits: 'the length of the finished code in bits' } }),
} as const

/** The log contract of {@link rangeEncoderAlgorithm}. */
export const rangeEncoderContract: Contract<StreamEncoderState, RangeEncoderPath> = defineContract(encoderPaths)

/** The log contract of {@link arithmeticEncoderAlgorithm}. */
export const arithmeticEncoderContract: Contract<StreamEncoderState, ArithmeticEncoderPath> = defineContract({
  ...encoderPaths,
  'code/pending': scalar({ description: 'bits whose value waits on a carry (the underflow count)' }),
})

/** State of the stepped encoders. */
export interface StreamEncoderState {
  /** Symbols coded. */
  readonly index: number
  /** The coder's registers: the interval (arithmetic: low and high; range: low and range) and undecided bits. */
  readonly low: number
  readonly high: number
  readonly pending: number
  /** The code so far: bits (arithmetic) or bytes (range). */
  readonly output: readonly number[]
  /** The frequency of each symbol now. */
  readonly counts: readonly number[]
  /** The information content of the symbols coded, in bits. */
  readonly ideal: number
  /** The start and width of the interval of the whole message in [0, 1). */
  readonly start: number
  readonly width: number
  /** Whether the terminating bits have been written. */
  readonly finished: boolean
}

function encoderObserve(s: StreamEncoderState, bitsPerUnit: number, range: boolean) {
  return {
    metrics: {
      'symbol/index': s.index,
      'interval/low': s.low / FULL,
      'interval/high': range ? (s.low + s.high) / FULL : (s.high + 1) / FULL,
      'interval/start': s.start,
      'code/bits': (s.output.length + (range ? 0 : s.pending)) * bitsPerUnit,
      'code/ideal': s.ideal,
      ...(range ? {} : { 'code/pending': s.pending }),
    },
    tensors: {
      'code/output': { role: 'output' as const, value: s.output, shape: [s.output.length] },
      'model/frequencies': { role: 'state' as const, value: s.counts, shape: [s.counts.length] },
    },
  }
}

function encoderSetup(
  symbols: readonly number[],
  spec: FrequencyModel,
  name: string,
): { counts: number[]; increment: number } {
  const counts = startingCounts(spec, name)
  for (const s of symbols) checkSymbolIn(counts, s, name)
  return { counts, increment: spec.kind === 'adaptive' ? (spec.increment ?? 1) : 0 }
}

function encoderStep(
  s: StreamEncoderState,
  symbols: readonly number[],
  adaptive: boolean,
  increment: number,
  code: (
    state: { low: number; high: number; pending: number },
    r: SymbolRange,
    out: number[],
  ) => { low: number; high: number; pending: number },
  finish: (state: { low: number; high: number; pending: number }, out: number[]) => void,
): StreamEncoderState {
  const out = s.output.slice()
  if (s.index >= symbols.length) {
    finish(s, out)
    return { ...s, output: out, finished: true }
  }
  const symbol = symbols[s.index] as number
  const r = rangeOf(s.counts, symbol)
  const next = code(s, r, out)
  return {
    index: s.index + 1,
    low: next.low,
    high: next.high,
    pending: next.pending,
    output: out,
    counts: adaptive ? bump(s.counts, symbol, increment) : s.counts,
    ideal: s.ideal + log2(r.total / r.freq),
    start: s.start + (s.width * r.low) / r.total,
    width: (s.width * r.freq) / r.total,
    finished: false,
  }
}

/**
 * Arithmetic coding as a traced algorithm, one symbol per step: the interval narrows to the symbol's share, the bits
 * that settle are written (those that depend on a later carry are counted as pending), and one last step writes the
 * bits that pin the final interval. The tape holds the working interval as a share of its 32-bit register, the start of
 * the whole message's interval in [0, 1), the bits written against the information content so far (the code tracks
 * `-sum log2 p` to within two bits) and the frequencies, which an adaptive model changes as it learns. The final output
 * is exactly {@link arithmeticEncode}'s. The state serialises.
 *
 * @param symbols - Symbols 0..m-1.
 * @param spec - The frequency model.
 * @returns The algorithm; it completes after one step per symbol plus the finishing step.
 * @throws MdmlError with code `value` for a symbol with zero probability or out of range, or a bad model.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { arithmeticEncode, arithmeticEncoderAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const spec = { kind: 'static', frequencies: [3, 1] } as const
 * const run = runAlgorithm(arithmeticEncoderAlgorithm([0, 0, 1, 0], spec), { seed: 0, maxSteps: 20 })
 * [run.steps, run.state.output.join('') === arithmeticEncode([0, 0, 1, 0], spec).join('')] // => [5, true]
 * ```
 */
export function arithmeticEncoderAlgorithm(
  symbols: readonly number[],
  spec: FrequencyModel,
): Algorithm<StreamEncoderState, ArithmeticEncoderPath> {
  const { counts, increment } = encoderSetup(symbols, spec, 'arithmeticEncoderAlgorithm')
  const adaptive = spec.kind === 'adaptive'
  return {
    name: 'information/arithmetic-encoder',
    config: { symbols: symbols.length, alphabet: counts.length, model: spec.kind },
    schema: contractSchema(arithmeticEncoderContract),
    init: () => ({
      index: 0,
      low: 0,
      high: FULL - 1,
      pending: 0,
      output: [],
      counts,
      ideal: 0,
      start: 0,
      width: 1,
      finished: false,
    }),
    step: (s, ctx) => {
      const next = encoderStep(s, symbols, adaptive, increment, arithmeticStep, arithmeticFinish)
      if (next.finished) ctx.emit('code/done', { bits: next.output.length })
      return next
    },
    done: (s) => (s.finished ? 'completed' : undefined),
    observe: (s) => encoderObserve(s, 1, false),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StreamEncoderState,
  }
}

/**
 * Range coding as a traced algorithm, one symbol per step: the range is split in proportion to the symbol's frequency
 * and whole bytes are shifted out as the top byte of the interval settles; a last step writes the four bytes of
 * the low register. The final output is exactly {@link rangeEncode}'s. The tape is that of
 * {@link arithmeticEncoderAlgorithm} with bytes for bits (`code/bits` counts eight per byte) and no pending count. The
 * state serialises.
 *
 * @param symbols - Symbols 0..m-1.
 * @param spec - The frequency model (totals up to 2^16).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a symbol with zero probability or out of range, or a bad model.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { rangeEncode, rangeEncoderAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const spec = { kind: 'static', frequencies: [6, 1, 1] } as const
 * const run = runAlgorithm(rangeEncoderAlgorithm([0, 0, 1, 0, 2], spec), { seed: 0, maxSteps: 20 })
 * run.state.output.join(',') === Array.from(rangeEncode([0, 0, 1, 0, 2], spec)).join(',') // => true
 * ```
 */
export function rangeEncoderAlgorithm(
  symbols: readonly number[],
  spec: FrequencyModel,
): Algorithm<StreamEncoderState, RangeEncoderPath> {
  const { counts, increment } = encoderSetup(symbols, spec, 'rangeEncoderAlgorithm')
  const adaptive = spec.kind === 'adaptive'
  // The range coder's registers are (low, range); the state keeps the range in `high`.
  const code = (st: { low: number; high: number; pending: number }, r: SymbolRange, out: number[]) => {
    const next = rangeStep({ low: st.low, range: st.high }, r, out)
    return { low: next.low, high: next.range, pending: 0 }
  }
  const finish = (st: { low: number; high: number; pending: number }, out: number[]): void =>
    rangeFinish({ low: st.low, range: st.high }, out)
  return {
    name: 'information/range-encoder',
    config: { symbols: symbols.length, alphabet: counts.length, model: spec.kind },
    schema: contractSchema(rangeEncoderContract),
    init: () => ({
      index: 0,
      low: 0,
      high: FULL - 1,
      pending: 0,
      output: [],
      counts,
      ideal: 0,
      start: 0,
      width: 1,
      finished: false,
    }),
    step: (s, ctx) => {
      const next = encoderStep(s, symbols, adaptive, increment, code, finish)
      if (next.finished) ctx.emit('code/done', { bits: 8 * next.output.length })
      return next
    },
    done: (s) => (s.finished ? 'completed' : undefined),
    observe: (s) => encoderObserve(s, 8, true),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StreamEncoderState,
  }
}

/** The paths of the stepped decoders. */
export type StreamDecoderPath =
  | 'symbol/index'
  | 'symbol/value'
  | 'interval/low'
  | 'interval/high'
  | 'code/read'
  | 'symbol/output'
  | 'model/frequencies'

/** The log contract of {@link arithmeticDecoderAlgorithm} and {@link rangeDecoderAlgorithm}. */
export const streamDecoderContract: Contract<StreamDecoderState, StreamDecoderPath> = defineContract({
  'symbol/index': scalar({ description: 'symbols decoded so far' }),
  'symbol/value': scalar({ description: 'the symbol decoded in the step' }),
  'interval/low': scalar({ description: 'the low end of the working interval, as a share of its 32-bit register' }),
  'interval/high': scalar({ description: 'the high end of the working interval, as a share of its register' }),
  'code/read': scalar({ description: 'bits (arithmetic) or bytes (range) of the code read so far' }),
  'symbol/output': tensor({ role: 'output', shape: ['symbols'], description: 'the symbols decoded so far' }),
  'model/frequencies': tensor({ role: 'state', shape: ['symbols'], description: 'the frequency of each symbol now' }),
})

/** State of the stepped decoders. */
export interface StreamDecoderState {
  /** Symbols decoded. */
  readonly index: number
  /** Registers: arithmetic (low, high, value), range (low, range in `high`, code in `value`). */
  readonly low: number
  readonly high: number
  readonly value: number
  /** How much of the code has been read. */
  readonly position: number
  readonly counts: readonly number[]
  /** The symbols decoded so far. */
  readonly output: readonly number[]
}

function decoderObserve(s: StreamDecoderState, range: boolean) {
  return {
    metrics: {
      'symbol/index': s.index,
      'symbol/value': s.output[s.output.length - 1] ?? -1,
      'interval/low': s.low / FULL,
      'interval/high': range ? (s.low + s.high) / FULL : (s.high + 1) / FULL,
      'code/read': s.position,
    },
    tensors: {
      'symbol/output': { role: 'output' as const, value: s.output, shape: [s.output.length] },
      'model/frequencies': { role: 'state' as const, value: s.counts, shape: [s.counts.length] },
    },
  }
}

/**
 * The decoder of {@link arithmeticEncode} as a traced algorithm, one symbol per step: it keeps the same interval as the
 * encoder, finds which symbol's share holds the code value, and narrows to it, reading in the bits that scroll in as
 * the encoder wrote them. Missing trailing bits count as 0. The state serialises.
 *
 * @param bits - The encoded bits.
 * @param length - The number of symbols to decode.
 * @param spec - The same frequency model as the encoder.
 * @returns The algorithm; it completes after `length` steps.
 * @throws MdmlError with code `value` for a bad model or length.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { arithmeticDecoderAlgorithm, arithmeticEncode } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const spec = { kind: 'adaptive', alphabetSize: 3 } as const
 * const bits = arithmeticEncode([2, 2, 1, 0], spec)
 * runAlgorithm(arithmeticDecoderAlgorithm(bits, 4, spec), { seed: 0, maxSteps: 10 }).state.output // => [2, 2, 1, 0]
 * ```
 */
export function arithmeticDecoderAlgorithm(
  bits: readonly number[],
  length: number,
  spec: FrequencyModel,
): Algorithm<StreamDecoderState, StreamDecoderPath> {
  if (!(Number.isInteger(length) && length >= 0))
    throw new MdmlError('value', 'arithmeticDecoderAlgorithm: length must be a non-negative integer')
  const counts = startingCounts(spec, 'arithmeticDecoderAlgorithm')
  const increment = spec.kind === 'adaptive' ? (spec.increment ?? 1) : 0
  const adaptive = spec.kind === 'adaptive'
  return {
    name: 'information/arithmetic-decoder',
    config: { length, alphabet: counts.length, model: spec.kind },
    schema: contractSchema(streamDecoderContract),
    init: () => {
      let value = 0
      for (let i = 0; i < 32; i++) value = 2 * value + ((bits[i] ?? 0) & 1)
      return { index: 0, low: 0, high: FULL - 1, value, position: 32, counts, output: [] }
    },
    step: (s) => {
      let { low, high, value, position } = s
      const range = high - low + 1
      const total = pairwiseSum(s.counts)
      const count = Math.floor(((value - low + 1) * total - 1) / range)
      const symbol = symbolAt(s.counts, count)
      const r = rangeOf(s.counts, symbol)
      high = low + Math.floor((range * (r.low + r.freq)) / total) - 1
      low = low + Math.floor((range * r.low) / total)
      for (;;) {
        const offset = high < HALF ? 0 : low >= HALF ? HALF : low >= QUARTER && high < HALF + QUARTER ? QUARTER : -1
        if (offset < 0) break
        low -= offset
        high -= offset
        value -= offset
        low = 2 * low
        high = 2 * high + 1
        value = 2 * value + ((bits[position++] ?? 0) & 1)
      }
      return {
        index: s.index + 1,
        low,
        high,
        value,
        position,
        counts: adaptive ? bump(s.counts, symbol, increment) : s.counts,
        output: [...s.output, symbol],
      }
    },
    done: (s) => (s.index >= length ? 'completed' : undefined),
    observe: (s) => decoderObserve(s, false),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StreamDecoderState,
  }
}

/**
 * The decoder of {@link rangeEncode} as a traced algorithm, one symbol per step: it divides the range by the total,
 * reads off the symbol whose share holds the code, and shifts in bytes as the encoder shifted them out. The state
 * serialises.
 *
 * @param bytes - The encoded bytes.
 * @param length - The number of symbols to decode.
 * @param spec - The same frequency model as the encoder.
 * @returns The algorithm; it completes after `length` steps.
 * @throws MdmlError with code `value` for a bad model or length.
 * @concept arithmetic-coding
 * @example
 * ```ts
 * import { rangeDecoderAlgorithm, rangeEncode } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const spec = { kind: 'static', frequencies: [5, 2, 1] } as const
 * const bytes = rangeEncode([0, 1, 0, 2], spec)
 * runAlgorithm(rangeDecoderAlgorithm(bytes, 4, spec), { seed: 0, maxSteps: 10 }).state.output // => [0, 1, 0, 2]
 * ```
 */
export function rangeDecoderAlgorithm(
  bytes: ArrayLike<number>,
  length: number,
  spec: FrequencyModel,
): Algorithm<StreamDecoderState, StreamDecoderPath> {
  if (!(Number.isInteger(length) && length >= 0))
    throw new MdmlError('value', 'rangeDecoderAlgorithm: length must be a non-negative integer')
  const counts = startingCounts(spec, 'rangeDecoderAlgorithm')
  const increment = spec.kind === 'adaptive' ? (spec.increment ?? 1) : 0
  const adaptive = spec.kind === 'adaptive'
  return {
    name: 'information/range-decoder',
    config: { length, alphabet: counts.length, model: spec.kind },
    schema: contractSchema(streamDecoderContract),
    init: () => {
      let code = 0
      for (let i = 0; i < 4; i++) code = u32(code * 256 + ((bytes[i] ?? 0) & 0xff))
      return { index: 0, low: 0, high: FULL - 1, value: code, position: 4, counts, output: [] }
    },
    step: (s) => {
      let { low, high: range, value: code, position } = s
      const total = pairwiseSum(s.counts)
      range = Math.floor(range / total)
      const symbol = symbolAt(s.counts, Math.min(total - 1, Math.floor(u32(code - low) / range)))
      const r = rangeOf(s.counts, symbol)
      low = u32(low + r.low * range)
      range = range * r.freq
      for (;;) {
        if (u32(low ^ u32(low + range)) >= TOP) {
          if (range >= BOTTOM) break
          range = u32(-low) & (BOTTOM - 1)
        }
        code = u32(code * 256 + ((bytes[position++] ?? 0) & 0xff))
        low = u32(low * 256)
        range = u32(range * 256)
      }
      return {
        index: s.index + 1,
        low,
        high: range,
        value: code,
        position,
        counts: adaptive ? bump(s.counts, symbol, increment) : s.counts,
        output: [...s.output, symbol],
      }
    },
    done: (s) => (s.index >= length ? 'completed' : undefined),
    observe: (s) => decoderObserve(s, true),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as StreamDecoderState,
  }
}
