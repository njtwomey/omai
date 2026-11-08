/**
 * Audio data augmentation, librosa style: SpecAugment's random frequency and time masks on a spectrogram, the phase
 * vocoder (an STFT resampled in time with phases kept coherent) and the time stretching and pitch shifting built on it,
 * and mixing a signal with noise at a chosen signal-to-noise ratio.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, normal, uniform } from '../../random/index.ts'
import { type ComplexValue, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { float, samples } from '../_shared.ts'
import { resample } from '../multirate.ts'
import { librosaIstft, librosaStft } from './_stft.ts'

// Audio augmentation: SpecAugment masking (Park et al. 2019), the phase vocoder, time stretching and pitch shifting
// (as librosa.effects, with FFT resampling), mixing at a signal-to-noise ratio and additive noise. Randomness comes
// from explicit keys.

/** Options of {@link specAugment}. */
export interface SpecAugmentOptions {
  /** Number of frequency masks (default 2) and their largest width in bins (default 8). */
  readonly freqMasks?: number
  readonly freqWidth?: number
  /** Number of time masks (default 2) and their largest width in frames (default 10). */
  readonly timeMasks?: number
  readonly timeWidth?: number
  /** Value written into masked cells (default 0). */
  readonly maskValue?: number
}

/**
 * SpecAugment's frequency and time masking of a spectrogram: each mask a band of uniformly random width (0 .. max)
 * at a uniformly random position, set to maskValue. (Time warping is not applied.)
 *
 * @param s - Spectrogram `[bins, frames]`.
 * @param key - Randomness.
 * @param options - Mask counts, widths and value.
 * @returns The masked spectrogram.
 * @concept data-augmentation
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { key } from 'omai-engine/random'
 * audio.specAugment(Array.from({ length: 20 }, () => new Array(30).fill(1)), key(0)).shape // => [20, 30]
 * ```
 */
export function specAugment(s: TensorLike, key: Key, options: SpecAugmentOptions = {}): Tensor {
  const v = primalOf(float(s))
  if (v.shape.length !== 2) throw new OmaiError('shape', 'specAugment: expected a [bins, frames] spectrogram')
  const [bins = 0, frames = 0] = v.shape
  const out = Float64Array.from(v.data, Number)
  const fm = options.freqMasks ?? 2
  const tm = options.timeMasks ?? 2
  const u = toFlat(uniform(key, [2 * (fm + tm)]))
  const fill = options.maskValue ?? 0
  let k = 0
  for (let m = 0; m < fm; m++) {
    const width = Math.floor((u[k++] ?? 0) * ((options.freqWidth ?? 8) + 1))
    const start = Math.floor((u[k++] ?? 0) * Math.max(1, bins - width + 1))
    for (let b = start; b < Math.min(bins, start + width); b++)
      for (let t = 0; t < frames; t++) out[b * frames + t] = fill
  }
  for (let m = 0; m < tm; m++) {
    const width = Math.floor((u[k++] ?? 0) * ((options.timeWidth ?? 10) + 1))
    const start = Math.floor((u[k++] ?? 0) * Math.max(1, frames - width + 1))
    for (let t = start; t < Math.min(frames, start + width); t++)
      for (let b = 0; b < bins; b++) out[b * frames + t] = fill
  }
  return fromStorage(out, [bins, frames], 'float64')
}

/**
 * The phase vocoder (librosa.phase_vocoder): an STFT resampled in time by `rate`, magnitudes linearly interpolated
 * between frames and phases accumulated from the frame-to-frame phase differences.
 *
 * @param d - Complex STFT `[bins, frames]`.
 * @param rate - Speed-up factor (2 halves the number of frames).
 * @returns The stretched STFT.
 * @concept phase-vocoder
 * @example
 * ```ts
 * import { audio, stft } from 'omai-engine/signal'
 * const z = stft(Array.from({ length: 512 }, (_, i) => Math.sin(i / 4)), { nperseg: 64, noverlap: 48, scaling: 'none' }).values
 * audio.phaseVocoder(z, 2).re.shape // => [33, 17]
 * ```
 */
export function phaseVocoder(d: ComplexValue, rate: number): { readonly re: Tensor; readonly im: Tensor } {
  if (!(rate > 0)) throw new OmaiError('value', 'phaseVocoder: rate must be positive')
  const re = primalOf(d.re)
  const im = primalOf(d.im)
  const [bins = 0, frames = 0] = re.shape
  // numpy.arange(0, frames, rate): k rate for k < ceil(frames / rate).
  const tOut = Array.from({ length: Math.ceil(frames / rate) }, (_, k) => k * rate)
  const n = tOut.length
  const ore = new Float64Array(bins * n)
  const oim = new Float64Array(bins * n)
  for (let b = 0; b < bins; b++) {
    const r = (t: number): number => Number(re.data[b * frames + t] ?? 0)
    const i = (t: number): number => Number(im.data[b * frames + t] ?? 0)
    let phase = Math.atan2(i(Math.floor(tOut[0] ?? 0)), r(Math.floor(tOut[0] ?? 0)))
    for (let k = 0; k < n; k++) {
      const t = tOut[k] ?? 0
      const i0 = Math.floor(t)
      const i1 = Math.min(i0 + 1, frames - 1)
      const frac = t - i0
      // Linear interpolation of magnitudes, extrapolated from the last two frames beyond the last one.
      const m0 = Math.hypot(r(i0), i(i0))
      const mag =
        i0 + 1 < frames
          ? m0 + frac * (Math.hypot(r(i0 + 1), i(i0 + 1)) - m0)
          : m0 + frac * (m0 - (i0 > 0 ? Math.hypot(r(i0 - 1), i(i0 - 1)) : m0))
      ore[b * n + k] = mag * Math.cos(phase)
      oim[b * n + k] = mag * Math.sin(phase)
      phase += Math.atan2(i(i1), r(i1)) - Math.atan2(i(i0), r(i0))
    }
  }
  return { re: fromStorage(ore, [bins, n], 'float64'), im: fromStorage(oim, [bins, n], 'float64') }
}

/** Options of {@link timeStretch} and {@link pitchShift}. */
export interface StretchOptions {
  /** FFT length (default 2048). */
  readonly nFft?: number
  /** Hop (default nFft / 4). */
  readonly hopLength?: number
}

/**
 * Time stretching without changing pitch (librosa.effects.time_stretch): STFT, phase vocoder, inverse STFT to
 * round(n / rate) samples.
 *
 * @param y - Signal.
 * @param rate - Speed-up factor (above 1 is faster and shorter).
 * @param options - FFT length and hop.
 * @returns The stretched signal.
 * @concept phase-vocoder
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * audio.timeStretch(Array.from({ length: 4096 }, (_, i) => Math.sin(i / 7)), 2, { nFft: 256 }).length // => 2048
 * ```
 */
export function timeStretch(y: TensorLike, rate: number, options: StretchOptions = {}): Float64Array {
  const x = samples(y, 'timeStretch')
  const nFft = options.nFft ?? 2048
  const hop = options.hopLength ?? Math.floor(nFft / 4)
  const z = librosaStft(x, nFft, hop)
  const stretched = phaseVocoder(z, rate)
  // Python's round: halves to even.
  const raw = x.length / rate
  const len = raw - Math.floor(raw) === 0.5 ? 2 * Math.round(raw / 2) : Math.round(raw)
  return librosaIstft(stretched, nFft, hop, len)
}

/**
 * Pitch shifting by n steps (librosa.effects.pitch_shift with FFT resampling): time stretching by 2^(-n / b), then
 * resampling back to the original rate and length.
 *
 * @param y - Signal.
 * @param sr - Sampling rate.
 * @param nSteps - Steps (semitones for 12 bins per octave).
 * @param options - Bins per octave (default 12), FFT length and hop.
 * @returns The shifted signal, the length of y.
 * @concept phase-vocoder
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * audio.pitchShift(Array.from({ length: 4096 }, (_, i) => Math.sin(i / 7)), 8000, 12, { nFft: 256 }).length // => 4096
 * ```
 */
export function pitchShift(
  y: TensorLike,
  sr: number,
  nSteps: number,
  options: StretchOptions & { readonly binsPerOctave?: number } = {},
): Float64Array {
  const x = samples(y, 'pitchShift')
  const rate = 2 ** (-nSteps / (options.binsPerOctave ?? 12))
  const stretched = timeStretch(x, rate, options)
  const ratio = sr / (sr / rate)
  const n = Math.ceil(stretched.length * ratio)
  const r = toFlat(primalOf(resample(stretched, n)))
  const out = new Float64Array(x.length)
  out.set(r.slice(0, x.length))
  return out
}

/**
 * Mix a signal with noise at a signal-to-noise ratio: the noise is scaled so that 10 log10(P_signal / P_noise)
 * equals snrDb.
 *
 * @param signal - Signal.
 * @param noise - Noise (at least as long; truncated).
 * @param snrDb - Target SNR in dB.
 * @returns The mixture.
 * @concept data-augmentation
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * Array.from(audio.mixAtSnr([1, -1, 1, -1], [1, 1, 1, 1], 0)) // => [2, 0, 2, 0]
 * ```
 */
export function mixAtSnr(signal: TensorLike, noise: TensorLike, snrDb: number): Float64Array {
  const s = samples(signal, 'mixAtSnr')
  const n = samples(noise, 'mixAtSnr')
  if (n.length < s.length) throw new OmaiError('shape', 'mixAtSnr: the noise is shorter than the signal')
  let ps = 0
  let pn = 0
  for (let i = 0; i < s.length; i++) {
    ps += (s[i] ?? 0) ** 2
    pn += (n[i] ?? 0) ** 2
  }
  const g = pn === 0 ? 0 : Math.sqrt(ps / (pn * 10 ** (snrDb / 10)))
  return Float64Array.from(s, (v, i) => v + g * (n[i] ?? 0))
}

/**
 * Add white Gaussian noise at a signal-to-noise ratio.
 *
 * @param signal - Signal.
 * @param key - Randomness.
 * @param snrDb - Target SNR in dB.
 * @returns The noisy signal.
 * @concept data-augmentation
 * @example
 * ```ts
 * import { audio } from 'omai-engine/signal'
 * import { key } from 'omai-engine/random'
 * audio.addNoise(new Array(100).fill(1), key(0), 20).length // => 100
 * ```
 */
export function addNoise(signal: TensorLike, key: Key, snrDb: number): Float64Array {
  const s = samples(signal, 'addNoise')
  const noise = toFlat(normal(foldIn(key, 0), [s.length]))
  return mixAtSnr(s, noise, snrDb)
}
