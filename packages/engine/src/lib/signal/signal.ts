/**
 * The shared vocabulary of the signal domain: the `Signal` (samples with a sampling rate and start time), `Spectrum`
 * and `TimeFrequency` record types the analyses return, `createSignal` and its sample times and duration, decibel
 * conversion, and phase unwrapping (numpy.unwrap).
 */

import { maxOf, WhiteboxError } from '../base/index.ts'
import {
  abs,
  absComplex,
  add,
  type ComplexValue,
  isComplex,
  log10,
  max,
  maximum,
  mul,
  primalOf,
  reshape,
  sub,
  type Tensor,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { axisOf, constant, float } from './_shared.ts'

/**
 * A sampled signal: samples `[n]` (one channel) or `[n, channels]`, a sampling rate in hertz, the time of the first
 * sample, and optional units and channel names.
 *
 * @concept sampled-signal
 */
export interface Signal {
  readonly samples: Value
  readonly fs: number
  readonly start: number
  readonly units?: string
  readonly channels?: readonly string[]
}

/** What a spectrum's values measure. */
export type SpectrumKind = 'density' | 'spectrum' | 'magnitude' | 'complex' | 'pseudospectrum'

/**
 * A spectrum: frequencies (in the units of `fs`), the values there (along the last axis) and what they measure, such
 * as a power spectral density (`density`, per hertz) or a power spectrum (`spectrum`).
 */
export interface Spectrum<V = Value> {
  readonly frequencies: Tensor
  readonly values: V
  readonly kind: SpectrumKind
  readonly fs: number
}

/**
 * A time-frequency representation: values `[..., frequencies, times]` (scipy's layout), with the frequencies and the
 * times of the columns.
 */
export interface TimeFrequency<V = Value> {
  readonly frequencies: Tensor
  readonly times: Tensor
  readonly values: V
  readonly fs: number
}

/** Options for {@link createSignal}. */
export interface SignalOptions {
  /** Sampling rate in hertz (default 1). */
  readonly fs?: number
  /** Time of the first sample (default 0). */
  readonly start?: number
  readonly units?: string
  readonly channels?: readonly string[]
}

/**
 * A sampled signal from its samples.
 *
 * @param samples - Samples `[n]` or `[n, channels]`.
 * @param options - Sampling rate, start time, units and channel names.
 * @returns The signal.
 * @throws WhiteboxError with code `value` for a non-positive rate, `shape` for other ranks or mismatched names.
 * @concept sampled-signal
 * @example
 * ```ts
 * import { createSignal } from 'whitebox-engine/signal'
 * createSignal([0, 1, 0, -1], { fs: 4 }).fs // => 4
 * ```
 */
export function createSignal(samples: TensorLike, options: SignalOptions = {}): Signal {
  const x = float(samples)
  const fs = options.fs ?? 1
  if (!(fs > 0) || !Number.isFinite(fs))
    throw new WhiteboxError('value', `createSignal: fs must be positive, got ${fs}`)
  if (x.shape.length < 1 || x.shape.length > 2)
    throw new WhiteboxError('shape', 'createSignal: samples are [n] or [n, c]')
  const c = x.shape.length === 2 ? (x.shape[1] ?? 0) : 1
  if (options.channels !== undefined && options.channels.length !== c)
    throw new WhiteboxError('shape', `createSignal: ${c} channels but ${options.channels.length} names`)
  return {
    samples: x,
    fs,
    start: options.start ?? 0,
    ...(options.units === undefined ? {} : { units: options.units }),
    ...(options.channels === undefined ? {} : { channels: options.channels }),
  }
}

/**
 * The times of the samples: start + k / fs for k = 0 .. n - 1.
 *
 * @param n - A sample count, or a signal.
 * @param fs - Sampling rate (default 1; ignored for a signal).
 * @param start - Time of the first sample (default 0; ignored for a signal).
 * @returns The times `[n]`.
 * @concept sampled-signal
 * @example
 * ```ts
 * import { sampleTimes } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sampleTimes(4, 2)) // => [0, 0.5, 1, 1.5]
 * ```
 */
export function sampleTimes(n: number | Signal, fs = 1, start = 0): Tensor {
  if (typeof n !== 'number') return sampleTimes(n.samples.shape[0] ?? 0, n.fs, n.start)
  return constant(Float64Array.from({ length: n }, (_, k) => start + k / fs))
}

/**
 * The duration of a signal: the number of samples divided by the rate.
 *
 * @param s - The signal.
 * @returns n / fs.
 * @example
 * ```ts
 * import { createSignal, duration } from 'whitebox-engine/signal'
 * duration(createSignal([1, 2, 3, 4], { fs: 2 })) // => 2
 * ```
 */
export function duration(s: Signal): number {
  return (s.samples.shape[0] ?? 0) / s.fs
}

/** Options for {@link magnitudeDb}. */
export interface DbOptions {
  /** The input is a power (10 log10) rather than an amplitude (20 log10). Default false. */
  readonly power?: boolean
  /** The reference giving 0 dB: a value, or `'max'` for the largest input (default 1). */
  readonly ref?: number | 'max'
  /** Floor applied before the logarithm (default 1e-5 for amplitudes, 1e-10 for powers, as librosa). */
  readonly amin?: number
  /** Clip to at most this many decibels below the peak (default none; librosa uses 80). */
  readonly topDb?: number
}

/**
 * Decibels of amplitudes or powers (librosa.amplitude_to_db and power_to_db): 10 log10(max(amin, P)) - 10
 * log10(max(amin, ref)) with P = |x|^2 for amplitudes, optionally clipped `topDb` below the peak. Complex inputs use
 * their magnitude. Differentiable away from the floor.
 *
 * @param x - Amplitudes or powers (real or complex).
 * @param options - Power or amplitude, reference, floor, dynamic range.
 * @returns The decibels, the shape of x.
 * @concept decibels
 * @example
 * ```ts
 * import { magnitudeDb } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(magnitudeDb([1, 10, 100])) // => [0, 20, 40]
 * toArray(magnitudeDb([1, 10, 100], { power: true, ref: 'max' })) // => [-20, -10, 0]
 * ```
 */
export function magnitudeDb(x: TensorLike | ComplexValue, options: DbOptions = {}): Value {
  const power = options.power ?? false
  const mag = isComplex(x) ? absComplex(x) : abs(float(x as TensorLike))
  const p = power ? mag : mul(mag, mag)
  const amin = power ? (options.amin ?? 1e-10) : (options.amin ?? 1e-5) ** 2
  if (!(amin > 0)) throw new WhiteboxError('value', 'magnitudeDb: amin must be positive')
  const refIn = options.ref ?? 1
  const peak = maxOf(primalOf(p).data)
  const ref = refIn === 'max' ? peak : power ? Math.abs(refIn) : refIn * refIn
  const db = sub(mul(10, log10(maximum(p, amin))), 10 * Math.log10(Math.max(amin, ref)))
  if (options.topDb === undefined) return db
  if (!(options.topDb >= 0)) throw new WhiteboxError('value', 'magnitudeDb: topDb must be non-negative')
  return maximum(db, sub(max(db), options.topDb))
}

/** Options for {@link unwrap}. */
export interface UnwrapOptions {
  /** Jumps smaller than this are kept (default period / 2). */
  readonly discont?: number
  /** Axis (default -1). */
  readonly axis?: number
  /** The period of the phase (default 2 pi). */
  readonly period?: number
}

/**
 * Unwrap a phase (numpy.unwrap): add multiples of the period so that consecutive differences along the axis are at
 * most half a period. The added offsets are piecewise constant, so the derivative is the identity.
 *
 * @param p - Phases.
 * @param options - Discontinuity threshold, axis and period.
 * @returns The unwrapped phases.
 * @concept phase-unwrapping
 * @example
 * ```ts
 * import { unwrap } from 'whitebox-engine/signal'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(unwrap([0, 1.5, 3.5, -2.5], { period: 4 })) // => [0, 1.5, 3.5, 1.5]
 * ```
 */
export function unwrap(p: TensorLike, options: UnwrapOptions = {}): Value {
  const x = float(p)
  const rank = x.shape.length
  if (rank === 0) return x
  const axis = axisOf(options.axis ?? -1, rank, 'unwrap')
  const period = options.period ?? 2 * Math.PI
  const discont = Math.max(options.discont ?? period / 2, period / 2)
  const data = primalOf(x).data
  const shape = x.shape
  const n = shape[axis] ?? 0
  const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1)
  const outer = shape.slice(0, axis).reduce((a, b) => a * b, 1)
  const correction = new Float64Array(data.length)
  const low = -period / 2
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let acc = 0
      for (let k = 1; k < n; k++) {
        const at = (o * n + k) * inner + i
        const dd = (data[at] ?? 0) - (data[at - inner] ?? 0)
        let ddmod = ((((dd - low) % period) + period) % period) + low
        if (ddmod === low && dd > 0) ddmod = period / 2
        if (Math.abs(dd) >= discont) acc += ddmod - dd
        correction[at] = acc
      }
    }
  }
  return add(x, reshape(constant(correction), [...shape]))
}
