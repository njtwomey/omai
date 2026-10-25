/**
 * Sharpened time-frequency representations: the reassigned spectrogram, which moves each STFT value to its local
 * instantaneous frequency and group delay (librosa.reassigned_spectrogram), and the synchrosqueezed STFT, which moves
 * coefficients along frequency only, so the signal can still be reconstructed from it.
 */

import { WhiteboxError } from '../base/index.ts'
import { type ComplexValue, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { constant, samples } from './_shared.ts'
import type { TimeFrequency } from './signal.ts'
import { type StftBoundary, stft } from './time-frequency.ts'
import { getWindow, type WindowSpec } from './windows/index.ts'

// Sharpening of short-time Fourier analysis: the reassigned spectrogram (librosa.reassigned_spectrogram), which moves
// each value to the centre of gravity of its energy in time and frequency, and the synchrosqueezing transform, which
// moves complex coefficients in frequency only and so stays invertible.

/** Options shared by the reassignment methods (librosa's names and defaults). */
export interface ReassignmentOptions {
  /** Sampling rate (default 22050, as librosa). */
  readonly sr?: number
  /** FFT and window length (default 2048). */
  readonly nFft?: number
  /** Hop between frames (default nFft / 4). */
  readonly hop?: number
  /** Window (default periodic Hann). */
  readonly window?: WindowSpec
  /** Centre the frames by padding nFft / 2 at both ends (default true). */
  readonly center?: boolean
  /** Padding when centred: 'constant' (zeros, default) or 'reflect'. */
  readonly padMode?: 'constant' | 'reflect'
}

interface Transforms {
  readonly h: { re: Float64Array; im: Float64Array }
  readonly dh: { re: Float64Array; im: Float64Array }
  readonly th: { re: Float64Array; im: Float64Array }
  readonly bins: number
  readonly frames: number
  readonly nFft: number
  readonly hop: number
  readonly win: Float64Array
}

/** STFTs with the window, its cyclic derivative and the time-weighted window (librosa's three transforms). */
function transforms(y: TensorLike, options: ReassignmentOptions): Transforms {
  const x = samples(y, 'reassignment')
  const nFft = options.nFft ?? 2048
  const hop = options.hop ?? Math.floor(nFft / 4)
  if (x.length < 1) throw new WhiteboxError('value', 'reassignment: empty signal')
  const win = Float64Array.from(toFlat(getWindow(options.window ?? 'hann', nFft)))
  const dwin = Float64Array.from(win, (_, i) => ((win[(i + 1) % nFft] ?? 0) - (win[(i - 1 + nFft) % nFft] ?? 0)) / 2)
  const half = Math.floor(nFft / 2)
  const twin = Float64Array.from(win, (v, i) => v * (nFft % 2 === 1 ? i - half : i + 0.5 - half))
  const center = options.center ?? true
  const boundary: StftBoundary = center ? (options.padMode === 'reflect' ? 'reflect' : 'zeros') : 'none'
  const run = (w: Float64Array): { re: Float64Array; im: Float64Array } => {
    const z = stft(x, {
      window: w,
      nperseg: nFft,
      noverlap: nFft - hop,
      boundary,
      padded: false,
      scaling: 'none',
    }).values
    return { re: Float64Array.from(primalOf(z.re).data), im: Float64Array.from(primalOf(z.im).data) }
  }
  const h = run(win)
  const bins = Math.floor(nFft / 2) + 1
  return { h, dh: run(dwin), th: run(twin), bins, frames: h.re.length / bins, nFft, hop, win }
}

/** A reassigned spectrogram: the reassigned frequency and time of each STFT value, and its magnitude. */
export interface ReassignedSpectrogram {
  readonly frequencies: Tensor
  readonly times: Tensor
  readonly magnitudes: Tensor
}

/**
 * The reassigned spectrogram (librosa.reassigned_spectrogram): each STFT value moves to its instantaneous frequency
 * f_k - Im(S_dh / S_h) sr / (2 pi) and group delay t_n + Re(S_th / S_h) / sr, from transforms with the window's
 * derivative and its time-weighted version. Values quieter than refPower are NaN (or the bin centres with fillNan).
 *
 * @param y - Real signal.
 * @param options - STFT options, a power floor `refPower` (default 1e-6), `fillNan` and `clip` (default true).
 * @returns Reassigned frequencies and times and the magnitudes, each `[bins, frames]`.
 * @concept reassignment
 * @example
 * ```ts
 * import { reassignedSpectrogram } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const y = Array.from({ length: 512 }, (_, n) => Math.cos((2 * Math.PI * 1000 * n) / 8000))
 * const r = reassignedSpectrogram(y, { sr: 8000, nFft: 128 })
 * Math.round(toFlat(r.frequencies)[16 * r.frequencies.shape[1] + 5] ?? 0) // => 1000
 * ```
 */
export function reassignedSpectrogram(
  y: TensorLike,
  options: ReassignmentOptions & {
    readonly refPower?: number
    readonly fillNan?: boolean
    readonly clip?: boolean
  } = {},
): ReassignedSpectrogram {
  const sr = options.sr ?? 22050
  const t = transforms(y, options)
  const n = samples(y, 'reassignedSpectrogram').length
  const ref = options.refPower ?? 1e-6
  if (ref < 0) throw new WhiteboxError('value', 'reassignedSpectrogram: refPower must be non-negative')
  const total = t.bins * t.frames
  const f = new Float64Array(total)
  const tm = new Float64Array(total)
  const mag = new Float64Array(total)
  for (let k = 0; k < t.bins; k++) {
    const binFreq = (k * sr) / t.nFft
    for (let j = 0; j < t.frames; j++) {
      const i = k * t.frames + j
      const hr = t.h.re[i] ?? 0
      const hi = t.h.im[i] ?? 0
      const d = hr * hr + hi * hi
      // S_dh / S_h and S_th / S_h
      const qi = ((t.dh.im[i] ?? 0) * hr - (t.dh.re[i] ?? 0) * hi) / d
      const pr = ((t.th.re[i] ?? 0) * hr + (t.th.im[i] ?? 0) * hi) / d
      // Uncentred frames are timed at their middle (librosa's frames_to_time with n_fft).
      const frameTime = (j * t.hop + ((options.center ?? true) ? 0 : Math.floor(t.nFft / 2))) / sr
      let fi = binFreq - qi * ((0.5 * sr) / Math.PI)
      let ti = frameTime + pr / sr
      const m = Math.sqrt(d)
      mag[i] = m
      if ((ref > 0 && m < Math.sqrt(ref)) || !Number.isFinite(fi)) fi = Number.NaN
      if ((ref > 0 && m < Math.sqrt(ref)) || !Number.isFinite(ti)) ti = Number.NaN
      if (options.fillNan === true) {
        if (Number.isNaN(fi)) fi = binFreq
        if (Number.isNaN(ti)) ti = frameTime
      }
      if (options.clip ?? true) {
        if (!Number.isNaN(fi)) fi = Math.min(Math.max(fi, 0), sr / 2)
        if (!Number.isNaN(ti)) ti = Math.min(Math.max(ti, 0), n / sr)
      }
      f[i] = fi
      tm[i] = ti
    }
  }
  const shape = [t.bins, t.frames]
  return { frequencies: constant(f, shape), times: constant(tm, shape), magnitudes: constant(mag, shape) }
}

/**
 * The synchrosqueezed STFT: every coefficient (phase referred to its window centre, one-sided weights folded in)
 * moves to the bin of its instantaneous frequency, concentrating ridges while keeping the transform invertible:
 * summing a frame's coefficients recovers the signal at the frame centre (see {@link inverseSynchrosqueeze}).
 *
 * @param y - Real signal.
 * @param options - STFT options and a magnitude threshold below which coefficients stay put (default 1e-8).
 * @returns Frequencies, frame times and the complex squeezed transform `[bins, frames]`.
 * @concept synchrosqueezing
 * @example
 * ```ts
 * import { synchrosqueeze } from 'whitebox-engine/signal'
 * synchrosqueeze(new Array(256).fill(0).map((_, n) => Math.cos(0.3 * n)), { sr: 1, nFft: 64, hop: 1 }).values.re.shape // => [33, 257]
 * ```
 */
export function synchrosqueeze(
  y: TensorLike,
  options: ReassignmentOptions & { readonly threshold?: number } = {},
): TimeFrequency<ComplexValue> {
  const sr = options.sr ?? 22050
  const t = transforms(y, { ...options, center: options.center ?? true })
  const total = t.bins * t.frames
  const re = new Float64Array(total)
  const im = new Float64Array(total)
  const thr = options.threshold ?? 1e-8
  const half = Math.floor(t.nFft / 2)
  for (let k = 0; k < t.bins; k++) {
    // Refer the phase to the window centre (e^{2 pi i k half / N}) and fold the one-sided weight in.
    const ang = (2 * Math.PI * k * half) / t.nFft
    const c = Math.cos(ang)
    const s = Math.sin(ang)
    const weight = k === 0 || (t.nFft % 2 === 0 && k === t.bins - 1) ? 1 : 2
    for (let j = 0; j < t.frames; j++) {
      const i = k * t.frames + j
      const hr = t.h.re[i] ?? 0
      const hi = t.h.im[i] ?? 0
      const vr = weight * (hr * c - hi * s)
      const vi = weight * (hr * s + hi * c)
      const d = hr * hr + hi * hi
      let target = k
      if (Math.sqrt(d) > thr) {
        const qi = ((t.dh.im[i] ?? 0) * hr - (t.dh.re[i] ?? 0) * hi) / d
        const freq = (k * sr) / t.nFft - qi * ((0.5 * sr) / Math.PI)
        const b = Math.round((freq * t.nFft) / sr)
        if (b >= 0 && b < t.bins) target = b
      }
      const o = target * t.frames + j
      re[o] = (re[o] ?? 0) + vr
      im[o] = (im[o] ?? 0) + vi
    }
  }
  const shape = [t.bins, t.frames]
  return {
    frequencies: constant(Array.from({ length: t.bins }, (_, k) => (k * sr) / t.nFft)),
    times: constant(Array.from({ length: t.frames }, (_, j) => (j * t.hop) / sr)),
    values: { re: constant(re, shape), im: constant(im, shape) },
    fs: sr,
  }
}

/**
 * Reconstruct the signal at the frame centres from a synchrosqueezed transform: x[t_n] = Re sum_k T[k, n] / (N
 * w[N / 2]), exact because squeezing only moves coefficients within a frame.
 *
 * @param t - The squeezed transform from {@link synchrosqueeze}.
 * @param options - The window and FFT length used (defaults as synchrosqueeze).
 * @returns The signal at each frame centre (every sample when the hop is 1).
 * @concept synchrosqueezing
 * @example
 * ```ts
 * import { inverseSynchrosqueeze, synchrosqueeze } from 'whitebox-engine/signal'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const x = Array.from({ length: 128 }, (_, n) => Math.cos(0.3 * n) + 0.5 * Math.sin(0.05 * n * n / 4))
 * const back = toFlat(inverseSynchrosqueeze(synchrosqueeze(x, { sr: 1, nFft: 32, hop: 1 }), { nFft: 32 }))
 * Math.abs((back[40] ?? 0) - (x[40] ?? 0)) < 1e-12 // => true
 * ```
 */
export function inverseSynchrosqueeze(
  t: TimeFrequency<ComplexValue>,
  options: { readonly nFft?: number; readonly window?: WindowSpec } = {},
): Tensor {
  const [bins = 0, frames = 0] = t.values.re.shape
  const nFft = options.nFft ?? 2 * (bins - 1)
  const win = toFlat(getWindow(options.window ?? 'hann', nFft))
  const centre = win[Math.floor(nFft / 2)] ?? 0
  if (centre === 0) throw new WhiteboxError('value', 'inverseSynchrosqueeze: the window vanishes at its centre')
  const re = toFlat(primalOf(t.values.re))
  const out = new Float64Array(frames)
  for (let k = 0; k < bins; k++) for (let j = 0; j < frames; j++) out[j] = (out[j] ?? 0) + (re[k * frames + j] ?? 0)
  return constant(Float64Array.from(out, (v) => v / (nFft * centre)))
}
