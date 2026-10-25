/**
 * Frame-level audio features, following librosa's conventions and defaults: magnitude, mel and chroma spectrograms,
 * decibel conversions, mel-frequency cepstral coefficients (MFCCs), the spectral shape descriptors (centroid,
 * bandwidth, roll-off, flatness, flux), and the time-domain RMS energy and zero-crossing rate, with the `frame` helper
 * that cuts a signal into overlapping windows.
 */

import { MdmlError } from '../../base/index.ts'
import { padArray } from '../../numerics/convolution/index.ts'
import { dct } from '../../numerics/fourier/index.ts'
import {
  abs,
  add,
  div,
  exp,
  fromStorage,
  log,
  log10,
  matmul,
  maximum,
  mean,
  mul,
  pow,
  primalOf,
  reshape,
  sliceAxis,
  sqrt,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { constant, float, samples } from '../_shared.ts'
import { zeroCrossings } from '../peaks.ts'
import { type StftBoundary, stft } from '../time-frequency.ts'
import { getWindow, type WindowSpec } from '../windows/index.ts'
import { chromaFilterBank, type MelFilterOptions, melFilterBank } from './filterbanks.ts'
import { fftFrequencies } from './scales.ts'

// Spectral audio features as librosa computes them: framing, spectrograms, mel spectrograms, decibel conversion,
// MFCCs, spectral centroid, bandwidth, roll-off, flatness and flux, chroma, zero-crossing rate and RMS energy. The
// spectrogram-based ones are differentiable in the signal.

/** librosa's padding modes for centred frames. */
export type AudioPadMode = 'constant' | 'edge' | 'reflect'

/** Options of the framed spectrogram (librosa's stft conventions). */
export interface SpectrogramOptions {
  /** FFT length (default 2048). */
  readonly nFft?: number
  /** Hop between frames (default winLength / 4). */
  readonly hopLength?: number
  /** Window length, zero-padded to nFft (default nFft). */
  readonly winLength?: number
  /** Window (default 'hann', periodic). */
  readonly window?: WindowSpec
  /** Centre the frames by padding nFft / 2 at both ends (default true). */
  readonly center?: boolean
  /** Padding of centred frames (default 'constant', zeros). */
  readonly padMode?: AudioPadMode
  /** Exponent of the magnitude (default 2, power). */
  readonly power?: number
}

const boundaryOf = (center: boolean, mode: AudioPadMode): StftBoundary =>
  !center ? 'none' : mode === 'constant' ? 'zeros' : mode === 'edge' ? 'constant' : 'reflect'

/**
 * The magnitude spectrogram |STFT|^power (librosa's _spectrogram over librosa.stft): frames of nFft samples every
 * hopLength, centred by default, a periodic Hann window, no scaling.
 *
 * @param y - Signal.
 * @param options - Framing, window, padding and power.
 * @returns The spectrogram `[nFft / 2 + 1, frames]`.
 * @concept spectrogram
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * audio.spectrogram(new Array(4096).fill(0.1), { nFft: 512 }).shape // => [257, 33]
 * ```
 */
export function spectrogram(y: TensorLike, options: SpectrogramOptions = {}): Value {
  const nFft = options.nFft ?? 2048
  const win = options.winLength ?? nFft
  const hop = options.hopLength ?? Math.floor(win / 4)
  const window = options.window ?? 'hann'
  // A window shorter than nFft is centred in a zero-padded frame (librosa.util.pad_center).
  const z = stft(y, {
    window,
    nperseg: nFft,
    noverlap: nFft - hop,
    boundary: boundaryOf(options.center ?? true, options.padMode ?? 'constant'),
    padded: false,
    scaling: 'none',
    ...(win === nFft ? {} : { window: paddedWindow(window, win, nFft) }),
  })
  const p2 = add(mul(z.values.re, z.values.re), mul(z.values.im, z.values.im))
  const power = options.power ?? 2
  return power === 2 ? p2 : pow(p2, power / 2)
}

/** A window of length win centred in nFft samples of zeros (librosa.util.pad_center). */
function paddedWindow(window: WindowSpec, win: number, nFft: number): Tensor {
  if (win > nFft) throw new MdmlError('value', 'spectrogram: winLength must be at most nFft')
  const w = toFlat(getWindow(window, win))
  const out = new Float64Array(nFft)
  out.set(w, Math.floor((nFft - win) / 2))
  return constant(out)
}

/** Options of {@link melSpectrogram}. */
export interface MelSpectrogramOptions extends SpectrogramOptions, MelFilterOptions {}

/**
 * The mel spectrogram (librosa.feature.melspectrogram): the power spectrogram projected on the mel filter bank.
 *
 * @param y - Signal.
 * @param sr - Sampling rate.
 * @param options - Spectrogram and mel filter options.
 * @returns `[nMels, frames]`.
 * @concept mel-spectrogram
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * audio.melSpectrogram(new Array(4096).fill(0.1), 16000, { nFft: 512, nMels: 40 }).shape // => [40, 33]
 * ```
 */
export function melSpectrogram(y: TensorLike, sr: number, options: MelSpectrogramOptions = {}): Value {
  const s = spectrogram(y, options)
  return matmul(melFilterBank(sr, options.nFft ?? 2048, options), s)
}

/** Options of the decibel conversions. */
export interface DbOptions {
  /** Reference value, or 'max' for the largest input (default 1). */
  readonly ref?: number | 'max'
  /** Floor of the input (default 1e-10 for power, 1e-5 for amplitude). */
  readonly amin?: number
  /** Clip to at most topDb below the peak (default 80; null for none). */
  readonly topDb?: number | null
}

/**
 * Power to decibels (librosa.power_to_db): 10 log10(max(amin, S) / max(amin, ref)), clipped to topDb below the peak.
 *
 * @param s - Power values.
 * @param options - Reference, floor and dynamic range.
 * @returns Decibels.
 * @concept decibels
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.powerToDb([1, 10, 100, 1e-12])) // => [0, 10, 20, -60]
 * ```
 */
export function powerToDb(s: TensorLike, options: DbOptions = {}): Value {
  const x = float(s)
  const amin = options.amin ?? 1e-10
  const data = toFlat(primalOf(x))
  let peak = Number.NEGATIVE_INFINITY
  for (const v of data) peak = Math.max(peak, v)
  const ref = options.ref === 'max' ? peak : (options.ref ?? 1)
  const db = sub(mul(log10(maximum(x, amin)), 10), 10 * Math.log10(Math.max(amin, Math.abs(ref))))
  const topDb = options.topDb === undefined ? 80 : options.topDb
  if (topDb === null) return db
  if (topDb < 0) throw new MdmlError('value', 'powerToDb: topDb must be non-negative')
  let top = Number.NEGATIVE_INFINITY
  for (const v of toFlat(primalOf(db))) top = Math.max(top, v)
  return maximum(db, top - topDb)
}

/**
 * Amplitude to decibels (librosa.amplitude_to_db): powerToDb of |S|^2 with the squared reference and floor.
 *
 * @param s - Amplitudes.
 * @param options - Reference, floor (default 1e-5) and dynamic range.
 * @returns Decibels.
 * @concept decibels
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.amplitudeToDb([1, 10])) // => [0, 20]
 * ```
 */
export function amplitudeToDb(s: TensorLike, options: DbOptions = {}): Value {
  const x = float(s)
  const amin = options.amin ?? 1e-5
  const data = toFlat(primalOf(x))
  let peak = 0
  for (const v of data) peak = Math.max(peak, Math.abs(v))
  const ref = options.ref === 'max' ? peak : (options.ref ?? 1)
  return powerToDb(mul(x, x), {
    ref: ref * ref,
    amin: amin * amin,
    topDb: options.topDb === undefined ? 80 : options.topDb,
  })
}

/**
 * Decibels to power (librosa.db_to_power): ref 10^(dB / 10).
 *
 * @param db - Decibels.
 * @param ref - Reference (default 1).
 * @returns Power.
 * @concept decibels
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.dbToPower([0, 10, 20])).map((v) => Math.round(v * 1e9) / 1e9) // => [1, 10, 100]
 * ```
 */
export function dbToPower(db: TensorLike, ref = 1): Value {
  return mul(exp(mul(float(db), Math.LN10 / 10)), ref)
}

/** Options of {@link mfcc}. */
export interface MfccOptions extends MelSpectrogramOptions {
  /** Coefficients (default 20). */
  readonly nMfcc?: number
  /** DCT type (default 2). */
  readonly dctType?: 1 | 2 | 3 | 4
  /** DCT normalisation (default 'ortho'). */
  readonly dctNorm?: 'ortho' | 'backward'
  /** Sinusoidal liftering coefficient (default 0, none). */
  readonly lifter?: number
}

/**
 * Mel-frequency cepstral coefficients (librosa.feature.mfcc): the orthonormal DCT-II of the log (dB) mel spectrogram,
 * the first nMfcc coefficients, optionally liftered.
 *
 * @param y - Signal.
 * @param sr - Sampling rate.
 * @param options - Coefficients, DCT, liftering and mel spectrogram options.
 * @returns `[nMfcc, frames]`.
 * @concept mfcc
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * const y = Array.from({ length: 4096 }, (_, i) => Math.sin(i / 5))
 * audio.mfcc(y, 16000, { nFft: 512, nMels: 40, nMfcc: 13 }).shape // => [13, 33]
 * ```
 */
export function mfcc(y: TensorLike, sr: number, options: MfccOptions = {}): Value {
  return mfccFromDb(powerToDb(melSpectrogram(y, sr, options)), options)
}

/**
 * MFCCs from a log-power (dB) mel spectrogram (librosa.feature.mfcc with S).
 *
 * @param s - Log mel spectrogram `[nMels, frames]`.
 * @param options - Coefficients, DCT and liftering.
 * @returns `[nMfcc, frames]`.
 * @concept mfcc
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * audio.mfccFromDb([[1, 2], [3, 4], [5, 6]], { nMfcc: 2 }).shape // => [2, 2]
 * ```
 */
export function mfccFromDb(
  s: TensorLike,
  options: Pick<MfccOptions, 'nMfcc' | 'dctType' | 'dctNorm' | 'lifter'> = {},
): Value {
  const n = options.nMfcc ?? 20
  const m = sliceAxis(dct(s, options.dctType ?? 2, { axis: 0, norm: options.dctNorm ?? 'ortho' }), 0, 0, n)
  const lifter = options.lifter ?? 0
  if (lifter < 0) throw new MdmlError('value', 'mfcc: lifter must be non-negative')
  if (lifter === 0) return m
  const li = Float64Array.from(
    { length: m.shape[0] ?? 0 },
    (_, i) => 1 + (lifter / 2) * Math.sin((Math.PI * (i + 1)) / lifter),
  )
  return mul(m, constant(li, [li.length, 1]))
}

/** Columns of S normalised to unit sum (librosa.util.normalize with norm 1 along frequency). */
function unitColumns(s: Value): Value {
  const total = sum(s, 0, true)
  const t = toFlat(primalOf(total))
  const safe = constant(
    Float64Array.from(t, (v) => (v < Number.MIN_VALUE * 2 ** 52 ? 1 : 0)),
    [1, t.length],
  )
  return div(s, add(total, safe))
}

/**
 * The spectral centroid of every frame (librosa.feature.spectral_centroid): the magnitude-weighted mean frequency.
 *
 * @param s - Magnitude (or power) spectrogram `[bins, frames]`.
 * @param sr - Sampling rate.
 * @returns `[1, frames]`.
 * @concept spectral-features
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.spectralCentroid([[0], [1], [1]], 8)) // => [[3]]
 * ```
 */
export function spectralCentroid(s: TensorLike, sr: number): Value {
  const v = float(s)
  const bins = v.shape[0] ?? 0
  const freq = reshape(fftFrequencies(sr, 2 * (bins - 1)), [1, bins])
  return matmul(freq, unitColumns(v))
}

/**
 * The spectral bandwidth of every frame (librosa.feature.spectral_bandwidth): the p-th root of the weighted p-th
 * moment of |f - centroid|.
 *
 * @param s - Magnitude spectrogram `[bins, frames]`.
 * @param sr - Sampling rate.
 * @param p - Moment order (default 2).
 * @returns `[1, frames]`.
 * @concept spectral-features
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.spectralBandwidth([[0], [1], [1]], 8)) // => [[1]]
 * ```
 */
export function spectralBandwidth(s: TensorLike, sr: number, p = 2): Value {
  const v = float(s)
  const bins = v.shape[0] ?? 0
  const freq = constant(toFlat(fftFrequencies(sr, 2 * (bins - 1))), [bins, 1])
  const centroid = spectralCentroid(v, sr)
  const dev = pow(abs(sub(freq, centroid)), p)
  return pow(sum(mul(unitColumns(v), dev), 0, true), 1 / p)
}

/**
 * The spectral roll-off of every frame (librosa.feature.spectral_rolloff): the lowest bin frequency below which
 * rollPercent of the frame's energy lies.
 *
 * @param s - Magnitude spectrogram `[bins, frames]`.
 * @param sr - Sampling rate.
 * @param rollPercent - Energy fraction (default 0.85).
 * @returns `[1, frames]` (not differentiable).
 * @concept spectral-features
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.spectralRolloff([[1], [1], [8]], 8)) // => [[4]]
 * ```
 */
export function spectralRolloff(s: TensorLike, sr: number, rollPercent = 0.85): Tensor {
  if (!(rollPercent > 0 && rollPercent < 1)) throw new MdmlError('value', 'spectralRolloff: rollPercent in (0, 1)')
  const v = primalOf(float(s))
  const [bins = 0, frames = 0] = v.shape
  const freq = toFlat(fftFrequencies(sr, 2 * (bins - 1)))
  const out = new Float64Array(frames)
  for (let t = 0; t < frames; t++) {
    let total = 0
    for (let k = 0; k < bins; k++) total += Number(v.data[k * frames + t] ?? 0)
    const threshold = rollPercent * total
    let acc = 0
    out[t] = Number.NaN
    for (let k = 0; k < bins; k++) {
      acc += Number(v.data[k * frames + t] ?? 0)
      if (acc >= threshold) {
        out[t] = freq[k] ?? 0
        break
      }
    }
  }
  return fromStorage(out, [1, frames], 'float64')
}

/**
 * The spectral flatness of every frame (librosa.feature.spectral_flatness): the geometric over the arithmetic mean of
 * max(amin, |S|^power); 1 for white noise, near 0 for tones.
 *
 * @param s - Magnitude spectrogram `[bins, frames]`.
 * @param options - Floor (default 1e-10) and power (default 2).
 * @returns `[1, frames]`.
 * @concept spectral-features
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.spectralFlatness([[1], [1], [1]])) // => [[1]]
 * ```
 */
export function spectralFlatness(
  s: TensorLike,
  options: { readonly amin?: number; readonly power?: number } = {},
): Value {
  const x = maximum(pow(float(s), options.power ?? 2), options.amin ?? 1e-10)
  return div(exp(mean(log(x), 0, true)), mean(x, 0, true))
}

/**
 * The spectral flux of every frame: the norm of the positive change of the magnitude spectrum from the previous
 * frame (0 for the first), the classic onset feature.
 *
 * @param s - Magnitude spectrogram `[bins, frames]`.
 * @param p - The norm (default 2).
 * @returns `[1, frames]`.
 * @concept spectral-features
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.spectralFlux([[0, 3, 1], [0, 4, 4]])) // => [[0, 5, 0]]
 * ```
 */
export function spectralFlux(s: TensorLike, p = 2): Value {
  const v = float(s)
  const frames = v.shape[1] ?? 0
  if (frames < 2) return mul(sum(v, 0, true), 0)
  const diff = sub(sliceAxis(v, 1, 1, frames), sliceAxis(v, 1, 0, frames - 1))
  const flux = pow(sum(pow(maximum(diff, 0), p), 0, true), 1 / p)
  // The first frame has no predecessor: flux 0.
  return padArray(flux, [
    [0, 0],
    [1, 0],
  ])
}

/** Options of {@link chromaStft}. */
export interface ChromaOptions extends SpectrogramOptions {
  /** Bins per octave (default 12). */
  readonly nChroma?: number
  /** Tuning deviation in fractions of a bin (default 0; librosa estimates it). */
  readonly tuning?: number
  /** Normalisation of each frame (default infinity, the maximum). */
  readonly norm?: number | null
}

/**
 * The chromagram (librosa.feature.chroma_stft): the power spectrogram projected on the chroma filter bank, each frame
 * normalised.
 *
 * @param y - Signal.
 * @param sr - Sampling rate.
 * @param options - Spectrogram, chroma bins, tuning and normalisation.
 * @returns `[nChroma, frames]`.
 * @concept chroma
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * const y = Array.from({ length: 8192 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / 22050))
 * audio.chromaStft(y, 22050).shape // => [12, 17]
 * ```
 */
export function chromaStft(y: TensorLike, sr: number, options: ChromaOptions = {}): Value {
  const nFft = options.nFft ?? 2048
  const s = spectrogram(y, { ...options, power: 2 })
  const fb = chromaFilterBank(sr, nFft, { nChroma: options.nChroma ?? 12, tuning: options.tuning ?? 0 })
  const raw = matmul(fb, s)
  const norm = options.norm === undefined ? Number.POSITIVE_INFINITY : options.norm
  if (norm === null) return raw
  const r = primalOf(raw)
  const [c = 0, frames = 0] = r.shape
  const scale = new Float64Array(frames)
  for (let t = 0; t < frames; t++) {
    let len = 0
    for (let k = 0; k < c; k++) {
      const v = Math.abs(Number(r.data[k * frames + t] ?? 0))
      len = norm === Number.POSITIVE_INFINITY ? Math.max(len, v) : len + v ** norm
    }
    if (norm !== Number.POSITIVE_INFINITY) len **= 1 / norm
    scale[t] = len < Number.MIN_VALUE * 2 ** 52 ? 1 : 1 / len
  }
  return mul(raw, constant(scale, [1, frames]))
}

/**
 * Frames of a signal (librosa.util.frame): column t is y[t hop : t hop + frameLength].
 *
 * @param y - Signal.
 * @param frameLength - Frame length.
 * @param hop - Hop.
 * @returns `[frameLength, frames]`.
 * @concept framing
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.frame([1, 2, 3, 4, 5], 3, 2)) // => [[1, 3], [2, 4], [3, 5]]
 * ```
 */
export function frame(y: TensorLike, frameLength: number, hop: number): Value {
  const v = float(y)
  const n = v.shape[0] ?? 0
  if (v.shape.length !== 1) throw new MdmlError('shape', 'frame: expected a 1-D signal')
  if (n < frameLength) throw new MdmlError('value', 'frame: the signal is shorter than a frame')
  const frames = 1 + Math.floor((n - frameLength) / hop)
  const idx = new Int32Array(frameLength * frames)
  for (let i = 0; i < frameLength; i++) for (let t = 0; t < frames; t++) idx[i * frames + t] = t * hop + i
  return reshape(take(v, fromStorage(idx, [idx.length], 'int32'), 0), [frameLength, frames])
}

/** Options of the frame-based features. */
export interface FrameOptions {
  /** Frame length (default 2048). */
  readonly frameLength?: number
  /** Hop (default 512). */
  readonly hopLength?: number
  /** Centre the frames (default true). */
  readonly center?: boolean
  /** Padding of centred frames (default 'constant'). */
  readonly padMode?: AudioPadMode
}

function centred(y: Value, frameLength: number, center: boolean, mode: AudioPadMode): Value {
  if (!center) return y
  const e = Math.floor(frameLength / 2)
  return padArray(y, [[e, e]], mode === 'constant' ? 'constant' : mode === 'edge' ? 'edge' : 'reflect', 0)
}

/**
 * Root-mean-square energy of every frame (librosa.feature.rms from a signal).
 *
 * @param y - Signal.
 * @param options - Frame length, hop, centring and padding.
 * @returns `[1, frames]`.
 * @concept rms-energy
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.rms([1, -1, 1, -1, 1, -1, 1, -1], { frameLength: 4, hopLength: 4, center: false })) // => [[1, 1]]
 * ```
 */
export function rms(y: TensorLike, options: FrameOptions = {}): Value {
  const frameLength = options.frameLength ?? 2048
  const x = centred(float(y), frameLength, options.center ?? true, options.padMode ?? 'constant')
  const f = frame(x, frameLength, options.hopLength ?? 512)
  return sqrt(mean(mul(f, f), 0, true))
}

/**
 * Root-mean-square energy from a magnitude spectrogram (librosa.feature.rms with S): Parseval with the DC and Nyquist
 * bins counted once.
 *
 * @param s - Magnitude spectrogram `[frameLength / 2 + 1, frames]`.
 * @param frameLength - The frame length the spectrogram was computed with.
 * @returns `[1, frames]`.
 * @concept rms-energy
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.rmsFromSpectrogram([[4], [0], [0]], 4)) // => [[1]]
 * ```
 */
export function rmsFromSpectrogram(s: TensorLike, frameLength: number): Value {
  const v = float(s)
  const bins = v.shape[0] ?? 0
  if (bins !== Math.floor(frameLength / 2) + 1) throw new MdmlError('shape', 'rmsFromSpectrogram: bins do not match')
  const w = Float64Array.from({ length: bins }, (_, k) =>
    k === 0 || (k === bins - 1 && frameLength % 2 === 0) ? 0.5 : 1,
  )
  const power = mul(sum(mul(mul(v, v), constant(w, [bins, 1])), 0, true), 2 / frameLength ** 2)
  return sqrt(power)
}

/**
 * The zero-crossing rate of every frame (librosa.feature.zero_crossing_rate): the fraction of samples where the sign
 * changes, frames centred with edge padding.
 *
 * @param y - Signal.
 * @param options - Frame length, hop and centring.
 * @returns `[1, frames]`.
 * @concept zero-crossing-rate
 * @example
 * ```ts
 * import { audio } from 'mdml-engine/signal'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(audio.zeroCrossingRate([1, -1, 1, -1, 1, 1, 1, 1], { frameLength: 4, hopLength: 4, center: false })) // => [[0.75, 0]]
 * ```
 */
export function zeroCrossingRate(y: TensorLike, options: Omit<FrameOptions, 'padMode'> = {}): Tensor {
  const frameLength = options.frameLength ?? 2048
  const hop = options.hopLength ?? 512
  const x0 = samples(y, 'zeroCrossingRate')
  const center = options.center ?? true
  const e = center ? Math.floor(frameLength / 2) : 0
  const x = Float64Array.from(
    { length: x0.length + 2 * e },
    (_, i) => x0[Math.min(x0.length - 1, Math.max(0, i - e))] ?? 0,
  )
  const frames = 1 + Math.floor((x.length - frameLength) / hop)
  const out = new Float64Array(frames)
  for (let t = 0; t < frames; t++) {
    const z = zeroCrossings(x.subarray(t * hop, t * hop + frameLength), { pad: false })
    out[t] = z.filter(Boolean).length / frameLength
  }
  return fromStorage(out, [1, frames], 'float64')
}
