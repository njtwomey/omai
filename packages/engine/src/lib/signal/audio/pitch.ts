import { WhiteboxError } from '../../base/index.ts'
import { irfft, nextFastLen, rfft } from '../../numerics/fourier/index.ts'
import { betainc } from '../../numerics/special/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { samples } from '../_shared.ts'
import { getWindow } from '../windows/index.ts'

// Pitch estimation as librosa computes it: YIN (cumulative mean normalised difference, absolute threshold, parabolic
// interpolation) and probabilistic YIN (a distribution over thresholds, trough priors, Viterbi decoding of pitch and
// voicing); autocorrelation pitch and the harmonic product spectrum.

/**
 * The autocorrelation of a signal by FFT (librosa.autocorrelate): r[k] = sum_n y[n] y[n + k] for k < maxSize.
 *
 * @param y - Signal.
 * @param maxSize - Lags to return (default all).
 * @returns The autocorrelation `[maxSize]`.
 * @concept autocorrelation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * Array.from(audio.autocorrelate([1, 2, 3])).map((v) => Math.round(v * 1e9) / 1e9) // => [14, 8, 3]
 * ```
 */
export function autocorrelate(y: TensorLike, maxSize?: number): Float64Array {
  const x = samples(y, 'autocorrelate')
  return autocorrelateRaw(x, maxSize ?? x.length)
}

function autocorrelateRaw(x: Float64Array, maxSize: number): Float64Array {
  const n = x.length
  const size = Math.min(maxSize, n)
  const nPad = nextFastLen(2 * n - 1, true)
  const spec = rfft(x, nPad)
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const power = Float64Array.from(re, (r, k) => r * r + (im[k] ?? 0) ** 2)
  const ac = toFlat(irfft({ re: power, im: new Float64Array(power.length) }, nPad))
  return Float64Array.from(ac.slice(0, size))
}

/** Options of {@link yin} and {@link pyin}. */
export interface YinOptions {
  /** Lowest frequency in Hz. */
  readonly fmin: number
  /** Highest frequency in Hz. */
  readonly fmax: number
  /** Sampling rate (default 22050). */
  readonly sr?: number
  /** Frame length (default 2048). */
  readonly frameLength?: number
  /** Hop (default frameLength / 4). */
  readonly hopLength?: number
  /** Centre the frames with zero padding (default true). */
  readonly center?: boolean
}

interface Framed {
  readonly frames: Float64Array[]
  readonly minPeriod: number
  readonly maxPeriod: number
  readonly sr: number
}

function framesOf(y: TensorLike, options: YinOptions, name: string): Framed {
  const sr = options.sr ?? 22050
  const frameLength = options.frameLength ?? 2048
  const hop = options.hopLength ?? Math.floor(frameLength / 4)
  if (!(options.fmin > 0 && options.fmin < options.fmax && options.fmax <= sr / 2))
    throw new WhiteboxError('value', `${name}: need 0 < fmin < fmax <= sr / 2`)
  const x0 = samples(y, name)
  const e = (options.center ?? true) ? Math.floor(frameLength / 2) : 0
  const x = new Float64Array(x0.length + 2 * e)
  x.set(x0, e)
  if (x.length < frameLength) throw new WhiteboxError('value', `${name}: the signal is shorter than a frame`)
  const count = 1 + Math.floor((x.length - frameLength) / hop)
  const frames = Array.from({ length: count }, (_, t) => x.slice(t * hop, t * hop + frameLength))
  const minPeriod = Math.floor(sr / options.fmax)
  const maxPeriod = Math.min(Math.ceil(sr / options.fmin), frameLength - 1)
  if (minPeriod < 1 || minPeriod >= maxPeriod) throw new WhiteboxError('value', `${name}: the period range is empty`)
  return { frames, minPeriod, maxPeriod, sr }
}

/** librosa's _cumulative_mean_normalized_difference of one frame, for periods minPeriod .. maxPeriod. */
function cmnd(frame: Float64Array, minPeriod: number, maxPeriod: number): Float64Array {
  const acf = autocorrelateRaw(frame, maxPeriod + 1)
  const cum = new Float64Array(frame.length)
  let s = 0
  for (let i = 0; i < frame.length; i++) {
    s += (frame[i] ?? 0) ** 2
    cum[i] = s
  }
  const d = new Float64Array(maxPeriod + 1)
  // As librosa: the subtracted energy term is the cumulative energy with its first entry zeroed (librosa zeroes
  // row 0 of the cumulative sum before subtracting it).
  for (let k = 1; k <= maxPeriod; k++) d[k] = 2 * ((acf[0] ?? 0) - (acf[k] ?? 0)) - (k === 1 ? 0 : (cum[k - 1] ?? 0))
  const out = new Float64Array(maxPeriod - minPeriod + 1)
  let running = 0
  const mean = new Float64Array(maxPeriod + 1)
  for (let k = 1; k <= maxPeriod; k++) {
    running += d[k] ?? 0
    mean[k] = running / k
  }
  for (let k = minPeriod; k <= maxPeriod; k++) out[k - minPeriod] = (d[k] ?? 0) / ((mean[k] ?? 0) + Number.MIN_VALUE * 2 ** 52)
  return out
}

/** librosa's parabolic interpolation shifts (0 at the ends and where the vertex is more than a bin away). */
function parabolicShifts(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length)
  for (let i = 1; i + 1 < x.length; i++) {
    const a = (x[i + 1] ?? 0) + (x[i - 1] ?? 0) - 2 * (x[i] ?? 0)
    const b = ((x[i + 1] ?? 0) - (x[i - 1] ?? 0)) / 2
    out[i] = Math.abs(b) >= Math.abs(a) ? 0 : -b / a
  }
  return out
}

/** librosa.util.localmin along a frame, with yin's rule for the first element. */
function troughs(x: Float64Array): Uint8Array {
  const n = x.length
  const out = new Uint8Array(n)
  for (let i = 1; i + 1 < n; i++) out[i] = (x[i] ?? 0) < (x[i - 1] ?? 0) && (x[i] ?? 0) <= (x[i + 1] ?? 0) ? 1 : 0
  if (n > 1) {
    out[n - 1] = (x[n - 1] ?? 0) < (x[n - 2] ?? 0) ? 1 : 0
    out[0] = (x[0] ?? 0) < (x[1] ?? 0) ? 1 : 0
  }
  return out
}

/**
 * The YIN fundamental-frequency estimator (de Cheveigné and Kawahara 2002, librosa.yin): per frame, the first trough
 * of the cumulative mean normalised difference below the threshold (else its global minimum), refined by parabolic
 * interpolation.
 *
 * @param y - Signal.
 * @param options - Frequency range, sampling rate, framing, threshold (default 0.1).
 * @returns The f0 of every frame in Hz.
 * @concept pitch-estimation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 4096 }, (_, i) => Math.sin((2 * Math.PI * 220 * i) / 8000))
 * Math.round(audio.yin(y, { fmin: 80, fmax: 1000, sr: 8000, frameLength: 512 })[4] ?? 0) // => 220
 * ```
 */
export function yin(y: TensorLike, options: YinOptions & { readonly troughThreshold?: number }): Float64Array {
  const f = framesOf(y, options, 'yin')
  const threshold = options.troughThreshold ?? 0.1
  return Float64Array.from(f.frames, (frame) => {
    const d = cmnd(frame, f.minPeriod, f.maxPeriod)
    const shifts = parabolicShifts(d)
    const trough = troughs(d)
    let period = -1
    for (let i = 0; i < d.length; i++)
      if (trough[i] === 1 && (d[i] ?? 0) < threshold) {
        period = i
        break
      }
    if (period < 0) {
      period = 0
      for (let i = 1; i < d.length; i++) if ((d[i] ?? 0) < (d[period] ?? 0)) period = i
    }
    return f.sr / (f.minPeriod + period + (shifts[period] ?? 0))
  })
}

/** Options of {@link pyin}. */
export interface PyinOptions extends YinOptions {
  /** Thresholds of the threshold distribution (default 100). */
  readonly nThresholds?: number
  /** Beta distribution of the threshold (default [2, 18]). */
  readonly betaParameters?: readonly [number, number]
  /** Boltzmann prior over troughs (default 2). */
  readonly boltzmannParameter?: number
  /** Pitch-bin resolution in semitones (default 0.1). */
  readonly resolution?: number
  /** Largest pitch change in octaves per second (default 35.92). */
  readonly maxTransitionRate?: number
  /** Probability of switching between voiced and unvoiced (default 0.01). */
  readonly switchProb?: number
  /** Probability mass for the global minimum when no trough is below a threshold (default 0.01). */
  readonly noTroughProb?: number
  /** Transitions below this probability are pruned in decoding (default 1e-4). */
  readonly transitionMinProb?: number
}

/** The result of {@link pyin}. */
export interface PyinResult {
  /** f0 per frame in Hz (NaN where unvoiced). */
  readonly f0: Float64Array
  readonly voiced: Uint8Array
  /** The probability of each frame being voiced. */
  readonly voicedProb: Float64Array
}

/** The Viterbi path of an HMM (librosa.sequence.viterbi): log-domain, pruning transitions below a threshold. */
function viterbi(obs: Float64Array[], trans: Float64Array, nStates: number, minProb: number): Int32Array {
  const steps = obs.length
  const eps = Number.MIN_VALUE * 2 ** 52
  const logT = Float64Array.from(trans, (v) => Math.log(v + eps))
  const threshold = minProb > 0 ? Math.log(minProb + eps) : Number.NEGATIVE_INFINITY
  const preds: number[][] = Array.from({ length: nStates }, (_, j) => {
    const p: number[] = []
    for (let k = 0; k < nStates; k++) if ((logT[k * nStates + j] ?? 0) >= threshold) p.push(k)
    return p
  })
  let value = Float64Array.from(obs[0] ?? [], (v) => Math.log(v + eps) + Math.log(1 / nStates + eps))
  const ptr = Array.from({ length: steps }, () => new Int32Array(nStates))
  for (let t = 1; t < steps; t++) {
    const next = new Float64Array(nStates)
    const o = obs[t] ?? new Float64Array(nStates)
    const row = ptr[t] as Int32Array
    for (let j = 0; j < nStates; j++) {
      let best = Number.NEGATIVE_INFINITY
      for (const k of preds[j] ?? []) {
        const c = (value[k] ?? 0) + (logT[k * nStates + j] ?? 0)
        if (c > best) {
          best = c
          row[j] = k
        }
      }
      next[j] = Math.log((o[j] ?? 0) + eps) + best
    }
    value = next
  }
  const states = new Int32Array(steps)
  let last = 0
  for (let j = 1; j < nStates; j++) if ((value[j] ?? 0) > (value[last] ?? 0)) last = j
  states[steps - 1] = last
  for (let t = steps - 2; t >= 0; t--) states[t] = ptr[t + 1]?.[states[t + 1] ?? 0] ?? 0
  return states
}

/** librosa.sequence.transition_local with a triangular window, no wrapping, rows normalised. */
function localTransition(n: number, width: number): Float64Array {
  const win = toFlat(getWindow('triang', width, false))
  const out = new Float64Array(n * n)
  const lpad = Math.floor((n - width) / 2)
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n)
    // pad_center then roll by n // 2 + i + 1.
    const centred = new Float64Array(n)
    centred.set(win, lpad)
    const shift = Math.floor(n / 2) + i + 1
    for (let k = 0; k < n; k++) row[(((k + shift) % n) + n) % n] = centred[k] ?? 0
    row.fill(0, Math.min(n, i + Math.floor(width / 2) + 1))
    row.fill(0, 0, Math.max(0, i - Math.floor(width / 2)))
    let s = 0
    for (const v of row) s += v
    for (let k = 0; k < n; k++) out[i * n + k] = (row[k] ?? 0) / s
  }
  return out
}

/**
 * Probabilistic YIN (Mauch and Dixon 2014, librosa.pyin): YIN troughs weighted under a beta distribution of thresholds
 * and a Boltzmann prior, mapped to pitch bins, and decoded with voicing by a hidden Markov model (local pitch
 * transitions, voiced/unvoiced switching).
 *
 * @param y - Signal.
 * @param options - Frequency range, framing and the probabilistic model.
 * @returns f0 (NaN when unvoiced), voicing and voicing probabilities per frame.
 * @concept pitch-estimation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 4096 }, (_, i) => Math.sin((2 * Math.PI * 220 * i) / 8000))
 * const r = audio.pyin(y, { fmin: 80, fmax: 1000, sr: 8000, frameLength: 512 })
 * Math.round(r.f0[4] ?? 0) // => 220
 * ```
 */
export function pyin(y: TensorLike, options: PyinOptions): PyinResult {
  const f = framesOf(y, options, 'pyin')
  const nThr = options.nThresholds ?? 100
  const [ba, bb] = options.betaParameters ?? [2, 18]
  const thresholds = Float64Array.from({ length: nThr + 1 }, (_, i) => i / nThr)
  const cdf = Float64Array.from(toFlat(betainc(ba, bb, thresholds)))
  const betaProbs = Float64Array.from({ length: nThr }, (_, i) => (cdf[i + 1] ?? 0) - (cdf[i] ?? 0))
  const lambda = options.boltzmannParameter ?? 2
  const resolution = options.resolution ?? 0.1
  const perSemitone = Math.ceil(1 / resolution)
  const nBins = Math.floor(12 * perSemitone * Math.log2(options.fmax / options.fmin)) + 1
  const noTrough = options.noTroughProb ?? 0.01
  const frames = f.frames.length
  const obs: Float64Array[] = []
  const voicedProb = new Float64Array(frames)
  for (const [t, frame] of f.frames.entries()) {
    const d = cmnd(frame, f.minPeriod, f.maxPeriod)
    const shifts = parabolicShifts(d)
    const trough = troughs(d)
    const idx: number[] = []
    for (let i = 0; i < d.length; i++) if (trough[i] === 1) idx.push(i)
    const o = new Float64Array(2 * nBins)
    if (idx.length > 0) {
      const heights = idx.map((i) => d[i] ?? 0)
      const probs = new Float64Array(idx.length)
      for (let th = 0; th < nThr; th++) {
        const thr = thresholds[th + 1] ?? 1
        // Troughs below the threshold, in order: position p among them has the Boltzmann prior.
        const below = heights.map((h) => h < thr)
        const count = below.filter(Boolean).length
        if (count === 0) continue
        const norm = (1 - Math.exp(-lambda)) / (1 - Math.exp(-lambda * count))
        let pos = -1
        for (let k = 0; k < idx.length; k++)
          if (below[k] === true) {
            pos++
            probs[k] = (probs[k] ?? 0) + norm * Math.exp(-lambda * pos) * (betaProbs[th] ?? 0)
          }
      }
      let gmin = 0
      for (let k = 1; k < heights.length; k++) if ((heights[k] ?? 0) < (heights[gmin] ?? 0)) gmin = k
      let belowMin = 0
      for (let th = 0; th < nThr; th++) if (!((heights[gmin] ?? 0) < (thresholds[th + 1] ?? 1))) belowMin++
      let extra = 0
      for (let th = 0; th < belowMin; th++) extra += betaProbs[th] ?? 0
      probs[gmin] = (probs[gmin] ?? 0) + noTrough * extra
      for (const [k, i] of idx.entries()) {
        const p = probs[k] ?? 0
        if (p === 0) continue
        const f0 = f.sr / (f.minPeriod + i + (shifts[i] ?? 0))
        const bin = Math.min(nBins, Math.max(0, Math.round(12 * perSemitone * Math.log2(f0 / options.fmin))))
        o[bin] = p
      }
    }
    let v = 0
    for (let b = 0; b < nBins; b++) v += o[b] ?? 0
    v = Math.min(1, Math.max(0, v))
    voicedProb[t] = v
    for (let b = nBins; b < 2 * nBins; b++) o[b] = (1 - v) / nBins
    obs.push(o)
  }
  const sr = f.sr
  const hop = options.hopLength ?? Math.floor((options.frameLength ?? 2048) / 4)
  const maxSemitones = Math.round(((options.maxTransitionRate ?? 35.92) * 12 * hop) / sr)
  const local = localTransition(nBins, maxSemitones * perSemitone + 1)
  const stay = 1 - (options.switchProb ?? 0.01)
  const n2 = 2 * nBins
  const trans = new Float64Array(n2 * n2)
  for (let a = 0; a < 2; a++)
    for (let b = 0; b < 2; b++) {
      const s = a === b ? stay : 1 - stay
      for (let i = 0; i < nBins; i++)
        for (let j = 0; j < nBins; j++) trans[(a * nBins + i) * n2 + b * nBins + j] = s * (local[i * nBins + j] ?? 0)
    }
  const states = viterbi(obs, trans, n2, options.transitionMinProb ?? 1e-4)
  const f0 = new Float64Array(frames)
  const voiced = new Uint8Array(frames)
  for (let t = 0; t < frames; t++) {
    const s = states[t] ?? 0
    voiced[t] = s < nBins ? 1 : 0
    f0[t] = voiced[t] === 1 ? options.fmin * 2 ** ((s % nBins) / (12 * perSemitone)) : Number.NaN
  }
  return { f0, voiced, voicedProb }
}

/**
 * Autocorrelation pitch of every frame: the lag of the largest autocorrelation between the periods of fmax and fmin,
 * refined by parabolic interpolation.
 *
 * @param y - Signal.
 * @param options - Frequency range, sampling rate and framing.
 * @returns f0 per frame in Hz.
 * @concept pitch-estimation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 4096 }, (_, i) => Math.sin((2 * Math.PI * 200 * i) / 8000))
 * Math.abs((audio.autocorrelationPitch(y, { fmin: 80, fmax: 1000, sr: 8000, frameLength: 512 })[4] ?? 0) - 200) < 2 // => true
 * ```
 */
export function autocorrelationPitch(y: TensorLike, options: YinOptions): Float64Array {
  const f = framesOf(y, options, 'autocorrelationPitch')
  return Float64Array.from(f.frames, (frame) => {
    const ac = autocorrelateRaw(frame, f.maxPeriod + 2)
    let best = f.minPeriod
    for (let k = f.minPeriod + 1; k <= f.maxPeriod; k++) if ((ac[k] ?? 0) > (ac[best] ?? 0)) best = k
    const a = (ac[best + 1] ?? 0) + (ac[best - 1] ?? 0) - 2 * (ac[best] ?? 0)
    const b = ((ac[best + 1] ?? 0) - (ac[best - 1] ?? 0)) / 2
    const shift = a === 0 || Math.abs(b) >= Math.abs(a) ? 0 : -b / a
    return f.sr / (best + shift)
  })
}

/**
 * The harmonic product spectrum (Schroeder 1968, Noll 1969): the product of the magnitude spectrum downsampled by
 * 1 .. harmonics; its peak is the fundamental, even when the fundamental itself is weak.
 *
 * @param y - Signal (one analysis frame).
 * @param sr - Sampling rate.
 * @param options - Harmonics (default 5), FFT length (default next power of two of 4 n), Hann window (default true).
 * @returns The product spectrum `[bins / harmonics]`, its frequencies and the f0 estimate.
 * @concept pitch-estimation
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const y = Array.from({ length: 2048 }, (_, i) => [2, 3, 4].reduce((s, h) => s + Math.sin((2 * Math.PI * 150 * h * i) / 8000), 0))
 * Math.round(audio.harmonicProductSpectrum(y, 8000).f0 / 5) * 5 // => 150
 * ```
 */
export function harmonicProductSpectrum(
  y: TensorLike,
  sr: number,
  options: { readonly harmonics?: number; readonly nfft?: number; readonly window?: boolean } = {},
): { readonly spectrum: Tensor; readonly frequencies: Tensor; readonly f0: number } {
  const x = samples(y, 'harmonicProductSpectrum')
  const harmonics = options.harmonics ?? 5
  const nfft = options.nfft ?? 2 ** Math.ceil(Math.log2(4 * x.length))
  const w = (options.window ?? true) ? toFlat(getWindow('hann', x.length)) : new Array<number>(x.length).fill(1)
  const spec = rfft(Float64Array.from(x, (v, i) => v * (w[i] ?? 1)), nfft)
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  const mag = Float64Array.from(re, (r, k) => Math.hypot(r, im[k] ?? 0))
  const len = Math.floor(mag.length / harmonics)
  const hps = new Float64Array(len)
  for (let k = 0; k < len; k++) {
    let p = 1
    for (let h = 1; h <= harmonics; h++) p *= mag[k * h] ?? 0
    hps[k] = p
  }
  let best = 1
  for (let k = 2; k < len; k++) if ((hps[k] ?? 0) > (hps[best] ?? 0)) best = k
  const freqs = Float64Array.from({ length: len }, (_, k) => (k * sr) / nfft)
  return { spectrum: fromStorage(hps, [len], 'float64'), frequencies: fromStorage(freqs, [len], 'float64'), f0: freqs[best] ?? 0 }
}

