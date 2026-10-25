/**
 * Time-frequency analysis: the short-time Fourier transform and its inverse by weighted overlap-add, and the
 * spectrogram (scipy.signal.stft, istft, spectrogram), instantaneous phase and frequency from the analytic signal, and
 * the RMS time and frequency spreads of a pulse whose product is bounded by the Gabor uncertainty limit.
 */

import { MdmlError } from '../base/index.ts'
import { fft, fftfreq, ifft, irfft, rfft, rfftfreq } from '../numerics/fourier/index.ts'
import {
  absComplex,
  add,
  angle,
  type ComplexValue,
  concat,
  cScale,
  div,
  mean,
  mul,
  primalOf,
  reshape,
  scatterAdd,
  sliceAxis,
  slidingWindowView,
  sub,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { constant, float, indexTensor, rowsOf } from './_shared.ts'
import { hilbert } from './fourier.ts'
import { type TimeFrequency, unwrap } from './signal.ts'
import { getWindow, type WindowSpec } from './windows/index.ts'

// The short-time Fourier transform and its inverse (scipy.signal.stft and istft, with librosa's framing as an
// option), spectrograms, instantaneous phase and frequency, and the time and frequency spreads of the uncertainty
// principle. Layouts follow scipy: `[..., frequencies, frames]`.

/** How the signal is extended by half a segment at each end before framing. */
export type StftBoundary = 'zeros' | 'even' | 'odd' | 'constant' | 'reflect' | 'none'

/** Options for {@link stft}. */
export interface StftOptions {
  readonly fs?: number
  /** Window (default periodic Hann) or its samples. */
  readonly window?: WindowSpec | TensorLike
  /** Segment length (default 256, or the window length). */
  readonly nperseg?: number
  /** Overlap (default nperseg / 2). */
  readonly noverlap?: number
  readonly nfft?: number
  /** Per-segment detrending (default none). */
  readonly detrend?: 'constant' | 'none'
  readonly returnOnesided?: boolean
  /** Extension by nperseg / 2 at both ends (default 'zeros', which centres the frames; librosa's center=True). */
  readonly boundary?: StftBoundary
  /** Zero-pad the end so that the last segment is complete (default true; librosa uses false). */
  readonly padded?: boolean
  /** 'spectrum' (divide by sum w, default), 'psd' (by sqrt(fs sum w^2)) or 'none' (librosa). */
  readonly scaling?: 'spectrum' | 'psd' | 'none'
  readonly axis?: number
}

interface Framing {
  readonly win: Tensor
  readonly nperseg: number
  readonly noverlap: number
  readonly nfft: number
}

function framingOf(options: StftOptions, n: number, defaultLength = 256): Framing {
  const w = options.window ?? 'hann'
  let win: Tensor
  let nperseg: number
  if (typeof w === 'string' || typeof w === 'number' || (Array.isArray(w) && typeof w[0] === 'string')) {
    nperseg = options.nperseg ?? Math.min(defaultLength, n)
    win = getWindow(w as WindowSpec, nperseg)
  } else {
    win = primalOf(reshape(float(w as TensorLike), [-1]))
    nperseg = options.nperseg ?? win.shape[0] ?? 0
    if (nperseg !== win.shape[0]) throw new MdmlError('value', 'stft: nperseg must equal the window length')
  }
  if (nperseg < 1) throw new MdmlError('value', 'stft: nperseg must be positive')
  const nfft = options.nfft ?? nperseg
  if (nfft < nperseg) throw new MdmlError('value', 'stft: nfft must be at least nperseg')
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2)
  // A negative overlap (a hop longer than the segment, as librosa allows) skips samples between frames.
  if (noverlap >= nperseg) throw new MdmlError('value', 'stft: noverlap must be less than nperseg')
  return { win, nperseg, noverlap, nfft }
}

/** Extend rows by e samples at both ends (scipy's even_ext, odd_ext, const_ext and zero_ext; numpy reflect). */
function extend(rows: Value, e: number, type: StftBoundary): Value {
  if (e === 0 || type === 'none') return rows
  const [batch = 1, n = 0] = rows.shape
  if (type === 'zeros') return concat([zeros([batch, e]), rows, zeros([batch, e])], 1)
  const first = sliceAxis(rows, 1, 0, 1)
  const last = sliceAxis(rows, 1, n - 1, n)
  if (type === 'constant') {
    const idx = indexTensor(new Array<number>(e).fill(0))
    return concat([take(first, idx, 1), rows, take(last, idx, 1)], 1)
  }
  if (e >= n) throw new MdmlError('value', 'stft: the extension is longer than the signal')
  const left = take(rows, indexTensor(Array.from({ length: e }, (_, i) => e - i)), 1)
  const right = take(rows, indexTensor(Array.from({ length: e }, (_, i) => n - 2 - i)), 1)
  if (type === 'even' || type === 'reflect') return concat([left, rows, right], 1)
  return concat([sub(mul(first, 2), left), rows, sub(mul(last, 2), right)], 1)
}

/** Frames of rows `[batch, n]` as windowed spectra `[batch, frames, bins]`, with the framing's scale applied. */
function framesOf(rows: Value, f: Framing, options: StftOptions): { spec: ComplexValue; ext: number } {
  const hop = f.nperseg - f.noverlap
  const boundary = options.boundary ?? 'zeros'
  let x = extend(rows, boundary === 'none' ? 0 : Math.floor(f.nperseg / 2), boundary)
  const len = x.shape[1] ?? 0
  if (options.padded ?? true) {
    const nadd = (((-(len - f.nperseg) % hop) + hop) % hop) % f.nperseg
    if (nadd > 0) x = concat([x, zeros([x.shape[0] ?? 1, nadd])], 1)
  }
  if ((x.shape[1] ?? 0) < f.nperseg) throw new MdmlError('value', 'stft: the signal is shorter than a segment')
  const frames = slidingWindowView(x, f.nperseg, { axis: 1, step: hop })
  const [batch = 1, count = 0] = frames.shape
  let seg = reshape(frames, [batch * count, f.nperseg])
  if (options.detrend === 'constant') seg = sub(seg, mean(seg, 1, true))
  const windowed = mul(seg, reshape(f.win, [1, f.nperseg]))
  const onesided = options.returnOnesided ?? true
  const spec = onesided ? rfft(windowed, f.nfft) : fft(windowed, f.nfft)
  const bins = spec.re.shape[1] ?? 0
  const scaling = options.scaling ?? 'spectrum'
  let s1 = 0
  let s2 = 0
  for (const v of f.win.data) {
    s1 += v
    s2 += v * v
  }
  const scale = scaling === 'spectrum' ? 1 / s1 : scaling === 'psd' ? 1 / Math.sqrt((options.fs ?? 1) * s2) : 1
  const shaped = { re: reshape(spec.re, [batch, count, bins]), im: reshape(spec.im, [batch, count, bins]) }
  return {
    spec: scale === 1 ? shaped : cScale(shaped, scale),
    ext: boundary === 'none' ? 0 : Math.floor(f.nperseg / 2),
  }
}

/**
 * The short-time Fourier transform (scipy.signal.stft): windowed FFTs of overlapping segments, by default centred
 * (the signal is zero-extended by half a segment) and scaled by 1 / sum(w). With `boundary: 'zeros'`, `padded:
 * false` and `scaling: 'none'` it is librosa.stft. Differentiable.
 *
 * @param x - Real signal.
 * @param options - Sampling rate, window, segment, overlap, FFT length, detrending, sides, boundary, padding,
 *   scaling and axis.
 * @returns Frequencies, frame times and the complex STFT `[..., bins, frames]`.
 * @concept short-time-fourier-transform
 * @example
 * ```ts
 * import { stft } from 'mdml-engine/signal'
 * stft(new Array(32).fill(1), { nperseg: 8 }).values.re.shape // => [5, 9]
 * ```
 */
export function stft(x: TensorLike, options: StftOptions = {}): TimeFrequency<ComplexValue> {
  const r = rowsOf(float(x), options.axis ?? -1, 'stft')
  const f = framingOf(options, r.n)
  const fs = options.fs ?? 1
  const { spec, ext } = framesOf(r.rows, f, options)
  const count = spec.re.shape[1] ?? 0
  const hop = f.nperseg - f.noverlap
  const onesided = options.returnOnesided ?? true
  // scipy subtracts nperseg / 2 (not its floor) for an extended signal.
  const times = Array.from({ length: count }, (_, i) => (f.nperseg / 2 + i * hop - (ext > 0 ? f.nperseg / 2 : 0)) / fs)
  const toLayout = (v: Value): Value => {
    const t = transpose(v, [0, 2, 1])
    const lead = r.rows.shape[0] ?? 1
    return lead === 1 && float(x).shape.length === 1
      ? reshape(t, t.shape.slice(1))
      : restoreLead(t, float(x).shape, options.axis ?? -1)
  }
  return {
    frequencies: onesided ? (rfftfreq(f.nfft, 1 / fs) as Tensor) : (fftfreq(f.nfft, 1 / fs) as Tensor),
    times: constant(times),
    values: { re: toLayout(spec.re), im: toLayout(spec.im) },
    fs,
  }
}

/** Reshape `[batch, bins, frames]` back to the input's leading shape with the time axis replaced by bins, frames. */
function restoreLead(t: Value, shape: readonly number[], axis: number): Value {
  const a = axis < 0 ? axis + shape.length : axis
  const lead = shape.filter((_, i) => i !== a)
  return reshape(t, [...lead, t.shape[1] ?? 0, t.shape[2] ?? 0])
}

/** Options for {@link istft}. */
export interface IstftOptions {
  readonly fs?: number
  readonly window?: WindowSpec | TensorLike
  /** Segment length (default 2 (bins - 1) for one-sided input). */
  readonly nperseg?: number
  readonly noverlap?: number
  readonly nfft?: number
  /** The input is a one-sided spectrum of a real signal (default true). */
  readonly inputOnesided?: boolean
  /** Remove the half-segment extension of a centred STFT (default true). */
  readonly boundary?: boolean
  readonly scaling?: 'spectrum' | 'psd' | 'none'
}

/**
 * The inverse short-time Fourier transform by weighted overlap-add (scipy.signal.istft): invert each frame, multiply
 * by the window, add, and divide by the summed squared window. Exact for any window satisfying the NOLA condition.
 * Differentiable.
 *
 * @param z - The STFT `[bins, frames]` (complex).
 * @param options - Sampling rate, window, segment, overlap, FFT length, sides, boundary and scaling.
 * @returns The times and the reconstructed signal.
 * @throws MdmlError with code `value` when the window and overlap violate NOLA.
 * @concept short-time-fourier-transform
 * @example
 * ```ts
 * import { istft, stft } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [1, 2, 0, -1, 3, 1, -2, 0, 1, 2, 0, 1, 3, -1, 0, 2]
 * toFlat(istft(stft(x, { nperseg: 4 }).values).x).slice(0, 4).map((v) => Math.round(v * 1e12) / 1e12) // => [1, 2, 0, -1]
 * ```
 */
export function istft(z: ComplexValue, options: IstftOptions = {}): { readonly times: Tensor; readonly x: Value } {
  if (z.re.shape.length !== 2) throw new MdmlError('shape', 'istft: expected a [bins, frames] STFT')
  const [bins = 0, nseg = 0] = z.re.shape
  const onesided = options.inputOnesided ?? true
  const nperseg = options.nperseg ?? (onesided ? 2 * (bins - 1) : bins)
  const nfft = options.nfft ?? nperseg
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2)
  const hop = nperseg - noverlap
  const fs = options.fs ?? 1
  const w = options.window ?? 'hann'
  const win =
    typeof w === 'string' || typeof w === 'number' || (Array.isArray(w) && typeof w[0] === 'string')
      ? toFlat(getWindow(w as WindowSpec, nperseg))
      : toFlat(primalOf(float(w as TensorLike)))
  const frames = { re: transpose(z.re), im: transpose(z.im) }
  const inv = onesided ? irfft(frames, nfft) : ifft(frames, nfft).re
  let s1 = 0
  let s2 = 0
  for (const v of win) {
    s1 += v
    s2 += v * v
  }
  const scaling = options.scaling ?? 'spectrum'
  const scale = scaling === 'spectrum' ? s1 : scaling === 'psd' ? Math.sqrt(fs * s2) : 1
  const segs = mul(
    sliceAxis(inv, 1, 0, nperseg),
    constant(
      win.map((v) => v * scale),
      [1, nperseg],
    ),
  )
  const length = nperseg + (nseg - 1) * hop
  const idx = new Int32Array(nseg * nperseg)
  const norm = new Float64Array(length)
  for (let s = 0; s < nseg; s++) {
    for (let j = 0; j < nperseg; j++) {
      idx[s * nperseg + j] = s * hop + j
      norm[s * hop + j] = (norm[s * hop + j] ?? 0) + (win[j] ?? 0) ** 2
    }
  }
  let x = scatterAdd([length], indexTensor(idx), reshape(segs, [nseg * nperseg]), 0)
  let nv = norm
  if (options.boundary ?? true) {
    const e = Math.floor(nperseg / 2)
    x = sliceAxis(x, 0, e, length - e)
    nv = norm.slice(e, length - e)
  }
  if (nv.some((v) => v <= 1e-10)) throw new MdmlError('value', 'istft: the window and overlap violate NOLA')
  return {
    times: constant(Array.from({ length: nv.length }, (_, i) => i / fs)),
    x: div(x, constant(nv)),
  }
}

/** What {@link spectrogram} returns at each time-frequency point. */
export type SpectrogramMode = 'psd' | 'magnitude' | 'complex' | 'angle' | 'phase'

/** Options for {@link spectrogram}. */
export interface SpectrogramOptions {
  readonly fs?: number
  /** Window (default ['tukey', 0.25]). */
  readonly window?: WindowSpec | TensorLike
  readonly nperseg?: number
  /** Overlap (default nperseg / 8). */
  readonly noverlap?: number
  readonly nfft?: number
  readonly detrend?: 'constant' | 'none'
  readonly returnOnesided?: boolean
  readonly scaling?: 'density' | 'spectrum'
  readonly axis?: number
}

/**
 * A spectrogram (scipy.signal.spectrogram): the periodograms of consecutive (uncentred) segments, `[..., bins,
 * frames]`, as power ('psd', one-sided doubled), magnitude, angle or unwrapped phase. Differentiable.
 *
 * @param x - Real signal.
 * @param options - Sampling rate, window, segment, overlap, FFT length, detrending, sides, scaling and axis.
 * @param mode - 'psd' (default), 'magnitude', 'angle' or 'phase' ('complex' is {@link stft} with these frames).
 * @returns Frequencies, segment-centre times and values.
 * @concept spectrogram
 * @example
 * ```ts
 * import { spectrogram } from 'mdml-engine/signal'
 * spectrogram(new Array(64).fill(0).map((_, i) => Math.sin(i)), { nperseg: 16 }).values.shape // => [9, 4]
 * ```
 */
export function spectrogram(
  x: TensorLike,
  options: SpectrogramOptions = {},
  mode: Exclude<SpectrogramMode, 'complex'> = 'psd',
): TimeFrequency {
  const v = float(x)
  const r = rowsOf(v, options.axis ?? -1, 'spectrogram')
  const f = framingOf(
    {
      window: options.window ?? ['tukey', 0.25],
      ...(options.nperseg === undefined ? {} : { nperseg: options.nperseg }),
      ...(options.nfft === undefined ? {} : { nfft: options.nfft }),
      noverlap: 0,
    },
    r.n,
  )
  const noverlap = options.noverlap ?? Math.floor(f.nperseg / 8)
  const framing = { ...f, noverlap }
  const fs = options.fs ?? 1
  const onesided = options.returnOnesided ?? true
  const { spec } = framesOf(r.rows, framing, {
    boundary: 'none',
    padded: false,
    detrend: options.detrend ?? 'constant',
    returnOnesided: onesided,
    scaling: 'none',
  })
  let s1 = 0
  let s2 = 0
  for (const w of f.win.data) {
    s1 += w
    s2 += w * w
  }
  let out: Value
  if (mode === 'psd') {
    const scale = (options.scaling ?? 'density') === 'density' ? 1 / (fs * s2) : 1 / (s1 * s1)
    const bins = spec.re.shape[2] ?? 0
    const fold = Array.from({ length: bins }, (_, k) =>
      !onesided || k === 0 || (f.nfft % 2 === 0 && k === bins - 1) ? scale : 2 * scale,
    )
    out = mul(add(mul(spec.re, spec.re), mul(spec.im, spec.im)), constant(fold, [1, 1, bins]))
  } else {
    const scaled = cScale(spec, (options.scaling ?? 'density') === 'density' ? 1 / Math.sqrt(fs * s2) : 1 / s1)
    if (mode === 'magnitude') out = absComplex(scaled)
    else {
      out = angle(scaled)
      // scipy unwraps along the frequency axis.
      if (mode === 'phase') out = unwrap(out, { axis: 2 })
    }
  }
  const count = spec.re.shape[1] ?? 0
  const hop = f.nperseg - noverlap
  const t = transpose(out, [0, 2, 1])
  return {
    frequencies: onesided ? (rfftfreq(f.nfft, 1 / fs) as Tensor) : (fftfreq(f.nfft, 1 / fs) as Tensor),
    times: constant(Array.from({ length: count }, (_, i) => (f.nperseg / 2 + i * hop) / fs)),
    values: v.shape.length === 1 ? reshape(t, t.shape.slice(1)) : restoreLead(t, v.shape, options.axis ?? -1),
    fs,
  }
}

/**
 * The instantaneous phase of a real signal: the unwrapped angle of its analytic signal.
 *
 * @param x - Real signal `[n]`.
 * @returns The phase in radians `[n]`.
 * @concept instantaneous-frequency
 * @example
 * ```ts
 * import { instantaneousPhase } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(instantaneousPhase([1, 0, -1, 0])).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1.570796326795, 3.14159265359, 4.712388980385]
 * ```
 */
export function instantaneousPhase(x: TensorLike): Value {
  return unwrap(angle(hilbert(x)))
}

/**
 * The instantaneous frequency fs / (2 pi) d phase / dn of a real signal, from successive differences of the unwrapped
 * analytic phase (n - 1 values, between the samples).
 *
 * @param x - Real signal `[n]`.
 * @param fs - Sampling rate (default 1).
 * @returns The frequencies `[n - 1]`.
 * @concept instantaneous-frequency
 * @example
 * ```ts
 * import { instantaneousFrequency } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(instantaneousFrequency([1, 0, -1, 0, 1, 0, -1, 0], 8)).map((v) => Math.round(v * 1e9) / 1e9) // => [2, 2, 2, 2, 2, 2, 2]
 * ```
 */
export function instantaneousFrequency(x: TensorLike, fs = 1): Value {
  const p = instantaneousPhase(x)
  const n = p.shape[0] ?? 0
  return mul(sub(sliceAxis(p, 0, 1, n), sliceAxis(p, 0, 0, n - 1)), fs / (2 * Math.PI))
}

/**
 * The RMS duration of a pulse or window: sqrt(sum (t - tbar)^2 |w|^2 / sum |w|^2) in units of 1 / fs.
 *
 * @param w - Samples.
 * @param fs - Sampling rate (default 1).
 * @returns The time spread.
 * @concept uncertainty-principle
 * @example
 * ```ts
 * import { timeSpread } from 'mdml-engine/signal'
 * timeSpread([0, 1, 0]) // => 0
 * ```
 */
export function timeSpread(w: TensorLike, fs = 1): number {
  const s = toFlat(primalOf(float(w)))
  let e = 0
  let m = 0
  for (const [i, v] of s.entries()) {
    e += v * v
    m += i * v * v
  }
  if (e === 0) throw new MdmlError('value', 'timeSpread: zero energy')
  m /= e
  let v2 = 0
  for (const [i, v] of s.entries()) v2 += (i - m) ** 2 * v * v
  return Math.sqrt(v2 / e) / fs
}

/**
 * The RMS bandwidth of a pulse or window: the spread of |W(f)|^2 about its mean frequency over the two-sided spectrum,
 * computed on a DFT zero-padded to `nfft` (default 64 times the length).
 *
 * @param w - Samples.
 * @param fs - Sampling rate (default 1).
 * @param nfft - DFT length.
 * @returns The frequency spread in the units of fs.
 * @concept uncertainty-principle
 * @example
 * ```ts
 * import { frequencySpread } from 'mdml-engine/signal'
 * frequencySpread([1]) > 0.28 // => true
 * ```
 */
export function frequencySpread(w: TensorLike, fs = 1, nfft?: number): number {
  const s = toFlat(primalOf(float(w)))
  const n = nfft ?? 64 * s.length
  const spec = fft(constant(s), n)
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  let e = 0
  let m = 0
  const freq = (k: number): number => (k < n / 2 ? k : k - n) / n
  for (let k = 0; k < n; k++) {
    const p = (re[k] ?? 0) ** 2 + (im[k] ?? 0) ** 2
    e += p
    m += freq(k) * p
  }
  m /= e
  let v = 0
  for (let k = 0; k < n; k++) v += (freq(k) - m) ** 2 * ((re[k] ?? 0) ** 2 + (im[k] ?? 0) ** 2)
  return Math.sqrt(v / e) * fs
}

/**
 * The time-bandwidth product sigma_t sigma_f of a pulse, bounded below by 1 / (4 pi) (the Gabor limit, attained by
 * Gaussians).
 *
 * @param w - Samples.
 * @returns sigma_t sigma_f (dimensionless).
 * @concept uncertainty-principle
 * @example
 * ```ts
 * import { gaussian, timeBandwidthProduct } from 'mdml-engine/signal'
 * Math.abs(timeBandwidthProduct(gaussian(201, 12)) - 1 / (4 * Math.PI)) < 1e-6 // => true
 * ```
 */
export function timeBandwidthProduct(w: TensorLike): number {
  return timeSpread(w) * frequencySpread(w)
}
