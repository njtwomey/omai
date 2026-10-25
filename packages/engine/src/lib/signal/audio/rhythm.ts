/**
 * Rhythm analysis, librosa style: the onset strength envelope (how sharply the mel spectrum rises at each frame), peak
 * picking and onset detection on it, the autocorrelation tempogram, a global tempo estimate, and Ellis's
 * dynamic-programming beat tracker that places beats on strong onsets at a steady period.
 */

import { WhiteboxError } from '../../base/index.ts'
import { primalOf, type TensorLike } from '../../tensor/index.ts'
import { getWindow } from '../windows/index.ts'
import { melSpectrogram, powerToDb } from './features.ts'
import { autocorrelate } from './pitch.ts'

// Rhythm as librosa computes it: the spectral-flux onset strength of a log mel spectrogram, greedy peak picking for
// onsets, the autocorrelation tempogram and a tempo estimate under a log-normal prior, and Ellis's dynamic-programming
// beat tracker.

/** Options of {@link onsetStrength}. */
export interface OnsetOptions {
  /** Sampling rate (default 22050). */
  readonly sr?: number
  /** FFT length (default 2048). */
  readonly nFft?: number
  /** Hop (default 512). */
  readonly hopLength?: number
  /** Frames between compared spectra (default 1). */
  readonly lag?: number
  /** Aggregation over mel bands (default 'mean'). */
  readonly aggregate?: 'mean' | 'median'
  /** Mel bands (default 128). */
  readonly nMels?: number
}

const median = (a: number[]): number => {
  const s = [...a].sort((p, q) => p - q)
  const n = s.length
  return n % 2 === 1 ? (s[(n - 1) / 2] ?? 0) : ((s[n / 2 - 1] ?? 0) + (s[n / 2] ?? 0)) / 2
}

/**
 * The onset strength envelope (librosa.onset.onset_strength): the mean (or median) over mel bands of the positive
 * increase of the dB mel spectrogram from lag frames earlier, delayed to align with centred frames.
 *
 * @param y - Signal.
 * @param options - Sampling rate, framing, lag, aggregation and mel bands.
 * @returns The envelope, one value per frame.
 * @concept onset-detection
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 22050 }, (_, i) => (i % 5512 < 200 ? Math.sin(i) : 0))
 * audio.onsetStrength(y).length // => 44
 * ```
 */
export function onsetStrength(y: TensorLike, options: OnsetOptions = {}): Float64Array {
  const sr = options.sr ?? 22050
  const nFft = options.nFft ?? 2048
  const hop = options.hopLength ?? 512
  const lag = options.lag ?? 1
  if (!Number.isInteger(lag) || lag < 1)
    throw new WhiteboxError('value', 'onsetStrength: lag must be a positive integer')
  const S = primalOf(
    powerToDb(melSpectrogram(y, sr, { nFft, hopLength: hop, fmax: sr / 2, nMels: options.nMels ?? 128 })),
  )
  const [bands = 0, frames = 0] = S.shape
  const at = (b: number, t: number): number => Number(S.data[b * frames + t] ?? 0)
  const env: number[] = []
  for (let t = lag; t < frames; t++) {
    const d = Array.from({ length: bands }, (_, b) => Math.max(0, at(b, t) - at(b, t - lag)))
    env.push((options.aggregate ?? 'mean') === 'median' ? median(d) : d.reduce((s, v) => s + v, 0) / bands)
  }
  const pad = lag + Math.floor(nFft / (2 * hop))
  const out = new Float64Array(frames)
  for (let t = 0; t < frames; t++) out[t] = t < pad ? 0 : (env[t - pad] ?? 0)
  return out
}

/** Options of {@link peakPick}. */
export interface PeakPickOptions {
  readonly preMax: number
  readonly postMax: number
  readonly preAvg: number
  readonly postAvg: number
  readonly delta: number
  readonly wait: number
}

/**
 * Greedy peak picking (librosa.util.peak_pick): x[n] is a peak when it is the maximum of x[n - preMax .. n + postMax),
 * at least delta above the mean of x[n - preAvg .. n + postAvg), and wait frames after the previous peak.
 *
 * @param x - Envelope.
 * @param options - Window extents, threshold and refractory wait (fractional values are rounded up).
 * @returns Peak indices.
 * @concept onset-detection
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * audio.peakPick([0, 1, 0, 0, 3, 0, 0, 1, 0], { preMax: 1, postMax: 1, preAvg: 1, postAvg: 1, delta: 0.1, wait: 1 }) // => [1, 4, 7]
 * ```
 */
export function peakPick(x: ArrayLike<number>, options: PeakPickOptions): number[] {
  const preMax = Math.ceil(options.preMax)
  const postMax = Math.ceil(options.postMax)
  const preAvg = Math.ceil(options.preAvg)
  const postAvg = Math.ceil(options.postAvg)
  const wait = Math.ceil(options.wait)
  if (postMax <= 0 || postAvg <= 0) throw new WhiteboxError('value', 'peakPick: post windows must be positive')
  const n = x.length
  const max = (a: number, b: number): number => {
    let m = Number.NEGATIVE_INFINITY
    for (let i = a; i < b; i++) m = Math.max(m, x[i] ?? 0)
    return m
  }
  const mean = (a: number, b: number): number => {
    let s = 0
    for (let i = a; i < b; i++) s += x[i] ?? 0
    return s / (b - a)
  }
  const out: number[] = []
  const first =
    (x[0] ?? 0) >= max(0, Math.min(postMax, n)) && (x[0] ?? 0) >= mean(0, Math.min(postAvg, n)) + options.delta
  if (first) out.push(0)
  let i = first ? wait + 1 : 1
  while (i < n) {
    const v = x[i] ?? 0
    if (v !== max(Math.max(0, i - preMax), Math.min(i + postMax, n))) {
      i++
      continue
    }
    if (!(v >= mean(Math.max(0, i - preAvg), Math.min(i + postAvg, n)) + options.delta)) {
      i++
      continue
    }
    out.push(i)
    i += wait + 1
  }
  return out
}

/**
 * Onset detection (librosa.onset.onset_detect): the onset envelope normalised to [0, 1], then peak picking with
 * librosa's default windows (30 ms pre-maximum, 100 ms averages, 30 ms wait, delta 0.07).
 *
 * @param y - Signal.
 * @param options - Onset strength options.
 * @returns Onset frame indices.
 * @concept onset-detection
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 22050 }, (_, i) => (i % 5512 < 400 ? Math.sin(i / 3) : 0))
 * audio.onsetDetect(y).length > 0 // => true
 * ```
 */
export function onsetDetect(y: TensorLike, options: OnsetOptions = {}): number[] {
  const env = onsetStrength(y, options)
  return onsetDetectFromEnvelope(env, options.sr ?? 22050, options.hopLength ?? 512)
}

/**
 * Onset detection from an onset envelope (librosa.onset.onset_detect with onset_envelope).
 *
 * @param envelope - Onset strength.
 * @param sr - Sampling rate (default 22050).
 * @param hopLength - Hop (default 512).
 * @returns Onset frame indices.
 * @concept onset-detection
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * audio.onsetDetectFromEnvelope([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 22050, 512) // => [2, 11]
 * ```
 */
export function onsetDetectFromEnvelope(envelope: ArrayLike<number>, sr = 22050, hopLength = 512): number[] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (let i = 0; i < envelope.length; i++) {
    lo = Math.min(lo, envelope[i] ?? 0)
    hi = Math.max(hi, envelope[i] ?? 0)
  }
  const x = Float64Array.from(envelope, (v) => (v - lo) / (hi - lo + Number.MIN_VALUE * 2 ** 52))
  if (!x.some((v) => v !== 0) || !x.every(Number.isFinite)) return []
  const f = Math.floor
  return peakPick(x, {
    preMax: f((0.03 * sr) / hopLength),
    postMax: f((0 * sr) / hopLength) + 1,
    preAvg: f((0.1 * sr) / hopLength),
    postAvg: f((0.1 * sr) / hopLength) + 1,
    wait: f((0.03 * sr) / hopLength),
    delta: 0.07,
  })
}

/**
 * The autocorrelation tempogram (librosa.feature.tempogram): windowed autocorrelations of the onset envelope around
 * every frame (centred with linear ramps to zero), each normalised to a unit maximum.
 *
 * @param envelope - Onset strength.
 * @param winLength - Window in frames (default 384).
 * @returns `[winLength][frames]` as rows of lags.
 * @concept tempo-estimation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * audio.tempogram(Array.from({ length: 50 }, (_, i) => (i % 10 === 0 ? 1 : 0)), 16).length // => 16
 * ```
 */
export function tempogram(envelope: ArrayLike<number>, winLength = 384): Float64Array[] {
  const n = envelope.length
  const half = Math.floor(winLength / 2)
  const first = envelope[0] ?? 0
  const last = envelope[n - 1] ?? 0
  const padded = new Float64Array(n + 2 * half)
  for (let i = 0; i < half; i++) {
    padded[i] = (first * i) / half
    padded[n + 2 * half - 1 - i] = (last * i) / half
  }
  for (let i = 0; i < n; i++) padded[half + i] = envelope[i] ?? 0
  const win = Float64Array.from(primalOf(getWindow('hann', winLength)).data, Number)
  const out = Array.from({ length: winLength }, () => new Float64Array(n))
  for (let t = 0; t < n; t++) {
    const frame = Float64Array.from({ length: winLength }, (_, k) => (padded[t + k] ?? 0) * (win[k] ?? 0))
    const ac = autocorrelate(frame)
    let m = 0
    for (const v of ac) m = Math.max(m, Math.abs(v))
    const scale = m < Number.MIN_VALUE * 2 ** 52 ? 1 : 1 / m
    for (let k = 0; k < winLength; k++) (out[k] as Float64Array)[t] = (ac[k] ?? 0) * scale
  }
  return out
}

/** Options of {@link tempo} and {@link beatTrack}. */
export interface TempoOptions {
  readonly sr?: number
  readonly hopLength?: number
  /** Prior centre in BPM (default 120). */
  readonly startBpm?: number
  /** Prior width in octaves (default 1). */
  readonly stdBpm?: number
  /** Autocorrelation window in seconds (default 8). */
  readonly acSize?: number
  /** Largest tempo considered (default 320). */
  readonly maxTempo?: number
}

/**
 * A global tempo estimate (librosa.feature.tempo): the lag of the time-averaged tempogram that maximises
 * log(1 + 1e6 tg) plus a log-normal prior around startBpm.
 *
 * @param envelope - Onset strength.
 * @param options - Sampling rate, hop, prior and window.
 * @returns Beats per minute.
 * @concept tempo-estimation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * Math.round(audio.tempo(Array.from({ length: 400 }, (_, i) => (i % 22 === 0 ? 1 : 0)))) // => 117
 * ```
 */
export function tempo(envelope: ArrayLike<number>, options: TempoOptions = {}): number {
  const sr = options.sr ?? 22050
  const hop = options.hopLength ?? 512
  const win = Math.floor(Math.floor((options.acSize ?? 8) * sr) / hop)
  const tg = tempogram(envelope, win)
  const mean = tg.map((row) => row.reduce((s, v) => s + v, 0) / row.length)
  const start = options.startBpm ?? 120
  const std = options.stdBpm ?? 1
  const maxTempo = options.maxTempo ?? 320
  const bpms = Float64Array.from({ length: win }, (_, k) =>
    k === 0 ? Number.POSITIVE_INFINITY : (60 * sr) / (hop * k),
  )
  let maxIdx = 0
  for (let k = 0; k < win; k++)
    if ((bpms[k] ?? 0) < maxTempo) {
      maxIdx = k
      break
    }
  let best = -1
  let bestScore = Number.NEGATIVE_INFINITY
  for (let k = maxIdx; k < win; k++) {
    const prior = -0.5 * ((Math.log2(bpms[k] ?? 1) - Math.log2(start)) / std) ** 2
    const score = Math.log1p(1e6 * (mean[k] ?? 0)) + prior
    if (score > bestScore) {
      bestScore = score
      best = k
    }
  }
  return bpms[best] ?? 0
}

/**
 * Beat tracking by dynamic programming (Ellis 2007, librosa.beat.beat_track): a local score from the envelope
 * smoothed by a Gaussian of the beat period, the best predecessor of every frame under a log-period tightness
 * penalty, backtracking from the last strong local maximum, and trimming of weak leading and trailing beats.
 *
 * @param y - Signal.
 * @param options - Onset and tempo options, tightness (default 100), trimming (default true) and a fixed bpm.
 * @returns The tempo and the beat frames.
 * @concept beat-tracking
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 22050 * 4 }, (_, i) => (i % 11025 < 300 ? Math.sin(i / 2) : 0))
 * audio.beatTrack(y).beats.length > 3 // => true
 * ```
 */
export function beatTrack(
  y: TensorLike,
  options: OnsetOptions &
    TempoOptions & { readonly tightness?: number; readonly trim?: boolean; readonly bpm?: number } = {},
): { readonly bpm: number; readonly beats: number[] } {
  const env = onsetStrength(y, { ...options, aggregate: 'median' })
  return beatTrackFromEnvelope(env, options)
}

/**
 * Beat tracking from an onset envelope (librosa.beat.beat_track with onset_envelope).
 *
 * @param envelope - Onset strength.
 * @param options - Tempo options, tightness, trimming and a fixed bpm.
 * @returns The tempo and the beat frames.
 * @concept beat-tracking
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * audio.beatTrackFromEnvelope(Array.from({ length: 300 }, (_, i) => (i % 20 === 0 ? 1 : 0.01)), { bpm: 129.2 }).beats.slice(0, 3) // => [20, 40, 60]
 * ```
 */
export function beatTrackFromEnvelope(
  envelope: ArrayLike<number>,
  options: TempoOptions & { readonly tightness?: number; readonly trim?: boolean; readonly bpm?: number } = {},
): { readonly bpm: number; readonly beats: number[] } {
  const onsets = Float64Array.from(envelope)
  if (!onsets.some((v) => v !== 0)) return { bpm: 0, beats: [] }
  const sr = options.sr ?? 22050
  const hop = options.hopLength ?? 512
  const bpm = options.bpm ?? tempo(onsets, options)
  const tightness = options.tightness ?? 100
  const raw = ((sr / hop) * 60) / bpm
  // numpy.round: halves to even.
  const fpb = raw - Math.floor(raw) === 0.5 ? 2 * Math.round(raw / 2) : Math.round(raw)
  const n = onsets.length
  let mean = 0
  for (const v of onsets) mean += v / n
  let ss = 0
  for (const v of onsets) ss += (v - mean) ** 2
  const sd = Math.sqrt(ss / (n - 1))
  const norm = Float64Array.from(onsets, (v) => v / (sd + Number.MIN_VALUE * 2 ** 52))
  const window = Float64Array.from({ length: 2 * fpb + 1 }, (_, k) => Math.exp(-0.5 * (((k - fpb) * 32) / fpb) ** 2))
  const K = window.length
  const half = Math.floor(K / 2)
  const local = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let k = Math.max(0, i + half - n + 1); k < Math.min(i + half, K); k++)
      s += (window[k] ?? 0) * (norm[i + half - k] ?? 0)
    local[i] = s
  }
  let localMax = Number.NEGATIVE_INFINITY
  for (const v of local) localMax = Math.max(localMax, v)
  const thresh = 0.01 * localMax
  const back = new Int32Array(n)
  const cum = new Float64Array(n)
  let firstBeat = true
  back[0] = -1
  cum[0] = local[0] ?? 0
  for (let i = 0; i < n; i++) {
    let best = Number.NEGATIVE_INFINITY
    let loc = -1
    // Python's round(fpb / 2) rounds halves to even.
    const halfBeat = fpb % 2 === 0 ? fpb / 2 : (fpb - 1) / 2 + (((fpb - 1) / 2) % 2 === 0 ? 0 : 1)
    for (let l = i - halfBeat; l > Math.trunc(i - 2 * fpb - 1); l--) {
      if (l < 0) break
      const score = (cum[l] ?? 0) - tightness * (Math.log(i - l) - Math.log(fpb)) ** 2
      if (score > best) {
        best = score
        loc = l
      }
    }
    cum[i] = loc >= 0 ? (local[i] ?? 0) + best : (local[i] ?? 0)
    if (firstBeat && (local[i] ?? 0) < thresh) back[i] = -1
    else {
      back[i] = loc
      firstBeat = false
    }
  }
  // The last beat: the latest local maximum of the cumulative score above half their median.
  const isMax = Uint8Array.from({ length: n }, (_, i) =>
    i === 0
      ? 0
      : i === n - 1
        ? (cum[i] ?? 0) > (cum[i - 1] ?? 0)
          ? 1
          : 0
        : (cum[i] ?? 0) > (cum[i - 1] ?? 0) && (cum[i] ?? 0) >= (cum[i + 1] ?? 0)
          ? 1
          : 0,
  )
  const maxima: number[] = []
  for (let i = 0; i < n; i++) if (isMax[i] === 1) maxima.push(cum[i] ?? 0)
  const threshold = 0.5 * median(maxima)
  let tail = n - 1
  for (let i = n - 1; i >= 0; i--)
    if (isMax[i] === 1 && (cum[i] ?? 0) >= threshold) {
      tail = i
      break
    }
  const beats = new Uint8Array(n)
  for (let i = tail; i >= 0; i = back[i] ?? -1) beats[i] = 1
  if (options.trim ?? true) {
    const picked: number[] = []
    for (let i = 0; i < n; i++) if (beats[i] === 1) picked.push(local[i] ?? 0)
    const w = [0, 0.5, 1, 0.5, 0]
    const conv = new Float64Array(picked.length + w.length - 1)
    for (let i = 0; i < picked.length; i++)
      for (let k = 0; k < w.length; k++) conv[i + k] = (conv[i + k] ?? 0) + (picked[i] ?? 0) * (w[k] ?? 0)
    const smooth = conv.slice(2, n + 2)
    let ms = 0
    for (const v of smooth) ms += v * v
    const th = 0.5 * Math.sqrt(ms / Math.max(1, smooth.length))
    for (let i = 0; i < n && (local[i] ?? 0) <= th; i++) beats[i] = 0
    for (let i = n - 1; i >= 0 && (local[i] ?? 0) <= th; i--) beats[i] = 0
  }
  const out: number[] = []
  for (let i = 0; i < n; i++) if (beats[i] === 1) out.push(i)
  return { bpm, beats: out }
}
