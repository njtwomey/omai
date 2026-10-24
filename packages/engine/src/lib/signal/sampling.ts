import { WhiteboxError } from '../base/index.ts'
import { type Key, uniform } from '../random/index.ts'
import {
  abs,
  add,
  clip,
  div,
  floor,
  log1p,
  matmul,
  mul,
  pow,
  primalOf,
  repeat,
  sign,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { constant, float, sinc } from './_shared.ts'

// Sampling and reconstruction: band-limited (sinc) interpolation, aliasing, uniform and mu-law quantisation with
// dither, zero-order hold, and signal-to-quantisation-noise ratios.

/**
 * Band-limited reconstruction (Whittaker-Shannon interpolation): x(t) = sum_n x_n sinc(fs (t - t_n)), exact for a
 * signal band-limited below fs / 2 and sampled forever; a finite record is reconstructed well away from its ends.
 * Differentiable in the samples.
 *
 * @param x - Samples `[n]`.
 * @param fs - Sampling rate.
 * @param t - Times to evaluate at.
 * @param start - Time of the first sample (default 0).
 * @returns The interpolated values at t.
 * @concept sampling-theorem
 * @example
 * ```ts
 * import { sincInterpolate } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sincInterpolate([0, 1, 0], 1, [1, 2])) // => [1, 0]
 * ```
 */
export function sincInterpolate(x: TensorLike, fs: number, t: TensorLike, start = 0): Value {
  const v = float(x)
  if (v.shape.length !== 1) throw new WhiteboxError('shape', 'sincInterpolate: expected 1-D samples')
  const n = v.shape[0] ?? 0
  const times = toFlat(primalOf(float(t)))
  const m = new Float64Array(times.length * n)
  for (const [i, ti] of times.entries()) for (let k = 0; k < n; k++) m[i * n + k] = sinc(fs * (ti - start) - k)
  return matmul(constant(m, [times.length, n]), v)
}

/**
 * The apparent (aliased) frequency of a sinusoid at f when sampled at fs: the distance from f to the nearest
 * multiple of fs, in [0, fs / 2].
 *
 * @param f - Frequencies.
 * @param fs - Sampling rate.
 * @returns The aliases.
 * @concept aliasing
 * @example
 * ```ts
 * import { alias } from 'whitebox-engine/signal'
 * alias([30, 70, 120], 100) // => [30, 30, 20]
 * ```
 */
export function alias(f: readonly number[], fs: number): number[] {
  if (!(fs > 0)) throw new WhiteboxError('value', 'alias: fs must be positive')
  return f.map((v) => Math.abs(v - fs * Math.round(v / fs)))
}

/** Options for {@link quantise}. */
export interface QuantiseOptions {
  /** Bits per sample (2^bits levels). */
  readonly bits: number
  /** The full-scale range (default [-1, 1]). */
  readonly range?: readonly [number, number]
  /** Mid-tread (a level at zero, default) or mid-rise (zero is a decision threshold). */
  readonly mode?: 'midtread' | 'midrise'
}

/**
 * Uniform quantisation to 2^bits levels over a range, clipping outside it. The step is (hi - lo) / 2^bits; mid-tread
 * rounds to multiples of the step, mid-rise to odd multiples of half a step. The derivative is zero almost everywhere.
 *
 * @param x - Values.
 * @param options - Bits, range and mode.
 * @returns The quantised values.
 * @concept quantisation
 * @example
 * ```ts
 * import { quantise } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(quantise([0.1, 0.3, -0.8], { bits: 2 })) // => [0, 0.5, -1]
 * toArray(quantise([0.1, 0.3, -0.8], { bits: 2, mode: 'midrise' })) // => [0.25, 0.25, -0.75]
 * ```
 */
export function quantise(x: TensorLike, options: QuantiseOptions): Value {
  const { bits } = options
  if (!Number.isInteger(bits) || bits < 1) throw new WhiteboxError('value', 'quantise: bits must be a positive integer')
  const [lo, hi] = options.range ?? [-1, 1]
  if (!(hi > lo)) throw new WhiteboxError('value', 'quantise: the range must be increasing')
  const levels = 2 ** bits
  const step = (hi - lo) / levels
  const v = float(x)
  if ((options.mode ?? 'midtread') === 'midtread') {
    const mid = (lo + hi) / 2
    const k = floor(add(div(sub(v, mid), step), 0.5))
    return add(mul(clip(k, -levels / 2, levels / 2 - 1), step), mid)
  }
  const k = floor(div(sub(v, lo), step))
  return add(mul(add(clip(k, 0, levels - 1), 0.5), step), lo)
}

/**
 * Mu-law companding (librosa.mu_compress): sign(x) log(1 + mu |x|) / log(1 + mu) for x in [-1, 1], optionally
 * quantised to 1 + mu integer codes centred on zero.
 *
 * @param x - Values in [-1, 1].
 * @param mu - Compression parameter (default 255).
 * @param quantize - Return integer codes (default true).
 * @returns The compressed values or codes.
 * @throws WhiteboxError with code `value` for inputs outside [-1, 1] or mu <= 0.
 * @concept quantisation
 * @example
 * ```ts
 * import { muLawEncode } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(muLawEncode([-1, 0, 1], 15)) // => [-8, 0, 7]
 * ```
 */
export function muLawEncode(x: TensorLike, mu = 255, quantize = true): Value {
  if (!(mu > 0)) throw new WhiteboxError('value', 'muLawEncode: mu must be positive')
  const v = float(x)
  for (const s of primalOf(v).data)
    if (s < -1 || s > 1) throw new WhiteboxError('value', 'muLawEncode: x must be in [-1, 1]')
  const comp = div(mul(sign(v), log1p(mul(abs(v), mu))), Math.log1p(mu))
  if (!quantize) return comp
  const count = Math.floor(1 + mu)
  const bins = Array.from({ length: count }, (_, i) => (count === 1 ? -1 : -1 + (2 * i) / (count - 1)))
  const data = primalOf(comp).data
  const half = Math.floor(Math.floor(mu + 1) / 2)
  const codes = Float64Array.from(data, (c) => {
    let i = 0
    while (i < bins.length && (bins[i] ?? 0) < c) i++
    return i - half
  })
  return constant(codes, v.shape)
}

/**
 * Mu-law expansion (librosa.mu_expand), the inverse of {@link muLawEncode}: sign(y) ((1 + mu)^|y| - 1) / mu, from
 * integer codes when `quantize`.
 *
 * @param y - Compressed values in [-1, 1], or integer codes.
 * @param mu - Compression parameter (default 255).
 * @param quantize - The input is integer codes (default true).
 * @returns The expanded values.
 * @throws WhiteboxError with code `value` for inputs outside the range.
 * @concept quantisation
 * @example
 * ```ts
 * import { muLawDecode } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(muLawDecode([-1, 0, 1], 255, false)) // => [-1, 0, 1]
 * ```
 */
export function muLawDecode(y: TensorLike, mu = 255, quantize = true): Value {
  if (!(mu > 0)) throw new WhiteboxError('value', 'muLawDecode: mu must be positive')
  const v0 = float(y)
  const v = quantize ? mul(v0, 2 / (1 + mu)) : v0
  for (const s of primalOf(v).data)
    if (s < -1 || s > 1) throw new WhiteboxError('value', 'muLawDecode: input out of range')
  return mul(div(sign(v), mu), sub(pow(1 + mu, abs(v)), 1))
}

/** Options for {@link dither}. */
export interface DitherOptions {
  /** The quantisation step (one least-significant bit). */
  readonly step: number
  /** Rectangular (uniform over one step) or triangular (the sum of two, default) probability density. */
  readonly kind?: 'rectangular' | 'triangular'
}

/**
 * Add dither before quantisation: zero-mean noise of one step (rectangular) or the sum of two such (triangular,
 * TPDF), which makes the quantisation error independent of the signal (its first two moments, for TPDF).
 *
 * @param key - Random key.
 * @param x - Values.
 * @param options - The step and the dither density.
 * @returns x plus dither.
 * @concept quantisation
 * @example
 * ```ts
 * import { dither } from 'whitebox-engine/signal'
 * import { key } from 'whitebox-engine/random'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(dither(key(0), [0, 0, 0], { step: 0.1 })).every((v) => Math.abs(v) <= 0.1) // => true
 * ```
 */
export function dither(key: Key, x: TensorLike, options: DitherOptions): Value {
  const v = float(x)
  const s = options.step
  if (!(s > 0)) throw new WhiteboxError('value', 'dither: the step must be positive')
  const u = uniform(key, [...v.shape, 2], { low: -s / 2, high: s / 2 })
  const noise = (options.kind ?? 'triangular') === 'triangular' ? sum(u, -1) : sum(mul(u, constant([1, 0])), -1)
  return add(v, noise)
}

/**
 * Zero-order hold: repeat each sample `factor` times (the staircase a DAC produces, at a rate `factor` times higher).
 * Differentiable.
 *
 * @param x - Samples `[n]`.
 * @param factor - Repetitions per sample.
 * @returns The held signal `[n factor]`.
 * @concept zero-order-hold
 * @example
 * ```ts
 * import { zeroOrderHold } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(zeroOrderHold([1, 2], 3)) // => [1, 1, 1, 2, 2, 2]
 * ```
 */
export function zeroOrderHold(x: TensorLike, factor: number): Value {
  if (!Number.isInteger(factor) || factor < 1) throw new WhiteboxError('value', 'zeroOrderHold: factor must be >= 1')
  return repeat(float(x), factor, 0)
}

/**
 * The magnitude response of a zero-order hold of period 1 / fs: |sinc(f / fs)|, the droop that a reconstruction
 * filter compensates.
 *
 * @param f - Frequencies.
 * @param fs - Sampling rate.
 * @returns |H(f)|.
 * @concept zero-order-hold
 * @example
 * ```ts
 * import { zeroOrderHoldResponse } from 'whitebox-engine/signal'
 * zeroOrderHoldResponse([0, 50], 100).map((v) => Math.round(v * 1e6) / 1e6) // => [1, 0.63662]
 * ```
 */
export function zeroOrderHoldResponse(f: readonly number[], fs: number): number[] {
  return f.map((v) => Math.abs(sinc(v / fs)))
}

/**
 * The signal-to-quantisation-noise ratio 10 log10(sum x^2 / sum (x - xq)^2) in decibels.
 *
 * @param x - The original signal.
 * @param xq - The quantised signal.
 * @returns The SQNR in dB (infinite for no error).
 * @concept quantisation
 * @example
 * ```ts
 * import { sqnr } from 'whitebox-engine/signal'
 * Math.round(sqnr([1, -1], [0.9, -0.9]) * 1e9) / 1e9 // => 20
 * ```
 */
export function sqnr(x: TensorLike, xq: TensorLike): number {
  const a = toFlat(primalOf(float(x)))
  const b = toFlat(primalOf(float(xq)))
  if (a.length !== b.length) throw new WhiteboxError('shape', 'sqnr: signals differ in length')
  let s = 0
  let e = 0
  for (const [i, v] of a.entries()) {
    s += v * v
    e += (v - (b[i] ?? 0)) ** 2
  }
  return 10 * Math.log10(s / e)
}

/**
 * The textbook SQNR of a full-scale sinusoid quantised uniformly with b bits: 6.02 b + 1.76 dB (exactly 20 b log10 2
 * + 10 log10 1.5).
 *
 * @param bits - Bits per sample.
 * @returns The SQNR in dB.
 * @concept quantisation
 * @example
 * ```ts
 * import { idealSqnr } from 'whitebox-engine/signal'
 * Math.round(idealSqnr(16) * 100) / 100 // => 98.09
 * ```
 */
export function idealSqnr(bits: number): number {
  return 20 * bits * Math.log10(2) + 10 * Math.log10(1.5)
}
