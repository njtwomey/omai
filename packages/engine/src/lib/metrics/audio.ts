/**
 * Audio and speech quality metrics, torchmetrics style: signal-to-noise and signal-to-distortion ratios (BSS Eval's SDR
 * and the scale-invariant SI-SDR and SI-SNR), segmental SNR, permutation-invariant evaluation for source separation,
 * the STOI and ESTOI intelligibility measures, log-spectral distance, mel-cepstral distortion and DNSMOS from a
 * supplied model.
 */

import { argmaxOf, MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import { solveToeplitz } from '../numerics/linalg/index.ts'
import { linearSumAssignment } from '../optim/index.ts'
import { dtw, firwin, hann, resamplePoly, stft } from '../signal/index.ts'
import { primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { defineMetric, type MetricFunction, matrix, metricResult } from './metric.ts'

/** A mono signal as a Float64Array. */
function signal(name: string, x: TensorLike): Float64Array {
  const m = matrix(x)
  if (m.shape.length !== 1) throw new MdmlError('shape', `${name}: signals are 1-D`)
  return m.data
}

/** Signals `[T]` or `[S, T]` as rows. */
function signals(name: string, x: TensorLike): Float64Array[] {
  const m = matrix(x)
  if (m.shape.length === 1) return [m.data]
  if (m.shape.length !== 2) throw new MdmlError('shape', `${name}: signals are [T] or [S, T]`)
  const t = m.shape[1] ?? 0
  return Array.from({ length: m.shape[0] ?? 0 }, (_, i) => m.data.slice(i * t, (i + 1) * t))
}

/** Frame offsets 0, hop, 2 hop, ... with each frame inside the signal (strict: range(0, n - m, hop), as pystoi). */
function frameStarts(n: number, m: number, hop: number, strict = false): number[] {
  const out: number[] = []
  for (let s = 0; strict ? s < n - m : s <= n - m; s += hop) out.push(s)
  return out
}

/** Power spectra |rfft(w * frame, nfft)|^2 * scale of the frames at 0, hop, ... (signal's unscaled, unpadded STFT). */
function framePowers(
  x: Float64Array,
  window: Float64Array,
  nfft: number,
  hop: number,
  strict: boolean,
  scale = 1,
): Float64Array[] {
  const m = window.length
  const frames = frameStarts(x.length, m, hop, strict).length
  if (frames === 0) return []
  const z = stft(x, { window, nperseg: m, noverlap: m - hop, nfft, boundary: 'none', padded: false, scaling: 'none' })
  const re = primalOf(z.values.re).data
  const im = primalOf(z.values.im).data
  const bins = Math.floor(nfft / 2) + 1
  const total = z.values.re.shape[1] ?? 0
  return Array.from({ length: frames }, (_, f) =>
    Float64Array.from(
      { length: bins },
      (_, k) => (Number(re[k * total + f] ?? 0) ** 2 + Number(im[k * total + f] ?? 0) ** 2) * scale,
    ),
  )
}

const spec = (
  name: string,
  direction: 'higher' | 'lower',
  range: readonly [number, number] = [-Infinity, Infinity],
): MetricSpec => ({ name, direction, range, inputs: 'values', category: 'audio', concept: 'audio-evaluation' })

const EPS = Number.EPSILON

const pairs = (name: string, target: TensorLike, preds: TensorLike): [Float64Array, Float64Array][] => {
  const t = signals(name, target)
  const p = signals(name, preds)
  if (t.length !== p.length || t.some((x, i) => x.length !== (p[i]?.length ?? -1)))
    throw new MdmlError('shape', `${name}: target and preds differ in shape`)
  return t.map((x, i) => [x, p[i] as Float64Array])
}

const centred = (x: Float64Array): Float64Array => {
  const m = x.reduce((a, b) => a + b, 0) / x.length
  return Float64Array.from(x, (v) => v - m)
}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0)
  return s
}

/** The mean of a per-signal value over a batch `[T]` or `[B, T]`. */
function batched(
  name: string,
  target: TensorLike,
  preds: TensorLike,
  f: (t: Float64Array, p: Float64Array) => number,
): MetricResult {
  const ps = pairs(name, target, preds)
  const values = ps.map(([t, p]) => f(t, p))
  return metricResult(name, values.reduce((a, b) => a + b, 0) / values.length, values.length)
}

/**
 * The signal-to-noise ratio 10 log10(|t|^2 / |t - p|^2) in decibels (torchmetrics' signal_noise_ratio), averaged over
 * a batch.
 *
 * @param target - Clean signals `[T]` or `[B, T]`.
 * @param preds - Estimated signals, the same shape.
 * @param options - zeroMean: centre both first.
 * @returns The metric result in dB.
 * @concept audio-evaluation
 * @example
 * ```ts
 * import { signalNoiseRatio } from 'mdml-engine/metrics'
 * Number(signalNoiseRatio([1, 2, 3, 4], [1, 2, 3, 3]).estimate.value.toFixed(6)) // => 14.771213
 * ```
 */
export const signalNoiseRatio: MetricFunction<[TensorLike, TensorLike, { readonly zeroMean?: boolean }?]> =
  defineMetric(
    spec('signalNoiseRatio', 'higher'),
    (target: TensorLike, preds: TensorLike, options: { readonly zeroMean?: boolean } = {}): MetricResult =>
      batched('signalNoiseRatio', target, preds, (t0, p0) => {
        const t = options.zeroMean === true ? centred(t0) : t0
        const p = options.zeroMean === true ? centred(p0) : p0
        const noise = Float64Array.from(t, (v, i) => v - (p[i] ?? 0))
        return 10 * Math.log10((dot(t, t) + EPS) / (dot(noise, noise) + EPS))
      }),
  )

/** Options of {@link segmentalSnr}. */
export interface SegmentalSnrOptions {
  /** Sampling rate (Hz). */
  readonly fs: number
  /** Frame length in seconds (default 0.03) and overlap fraction (default 0.75). */
  readonly frameLength?: number
  readonly overlap?: number
  /** Per-frame clipping (default -10 and 35 dB). */
  readonly minSnr?: number
  readonly maxSnr?: number
}

/**
 * The segmental SNR (Hansen and Pellom, 1998, as pysepm's SNRseg): per-frame SNRs of Hann-windowed 30 ms frames with
 * 75 % overlap, clipped to [-10, 35] dB and averaged (the last frame dropped).
 *
 * @param target - Clean signal `[T]`.
 * @param preds - Processed signal `[T]`.
 * @param options - fs, frame length, overlap and clipping range.
 * @returns The mean segmental SNR in dB.
 * @example
 * ```ts
 * import { segmentalSnr } from 'mdml-engine/metrics'
 * const x = Array.from({ length: 400 }, (_, i) => Math.sin(i / 3))
 * segmentalSnr(x, x, { fs: 1000 }).estimate.value // => 35
 * ```
 */
export const segmentalSnr: MetricFunction<[TensorLike, TensorLike, SegmentalSnrOptions]> = defineMetric(
  spec('segmentalSnr', 'higher', [-10, 35]),
  (target: TensorLike, preds: TensorLike, options: SegmentalSnrOptions): MetricResult => {
    const t = signal('segmentalSnr', target)
    const p = signal('segmentalSnr', preds)
    const frame = options.frameLength ?? 0.03
    const overlap = options.overlap ?? 0.75
    const m = Math.round(frame * options.fs)
    const skip = Math.floor((1 - overlap) * frame * options.fs)
    const w = Float64Array.from({ length: m }, (_, i) => 0.5 * (1 - Math.cos((2 * Math.PI * (i + 1)) / (m + 1))))
    const lo = options.minSnr ?? -10
    const hi = options.maxSnr ?? 35
    const values: number[] = []
    for (let s = 0; s + m <= t.length; s += skip) {
      let signalEnergy = 0
      let noiseEnergy = 0
      for (let i = 0; i < m; i++) {
        const a = (w[i] ?? 0) * (t[s + i] ?? 0)
        const b = (w[i] ?? 0) * (p[s + i] ?? 0)
        signalEnergy += a * a
        noiseEnergy += (a - b) ** 2
      }
      const snr = 10 * Math.log10(signalEnergy / (noiseEnergy + EPS) + EPS)
      values.push(Math.min(hi, Math.max(lo, snr)))
    }
    values.pop()
    return metricResult('segmentalSnr', values.reduce((a, b) => a + b, 0) / values.length, values.length)
  },
)

/** Options of {@link signalDistortionRatio}. */
export interface SdrOptions {
  /** The length of the distortion filter (default 512). */
  readonly filterLength?: number
  readonly zeroMean?: boolean
  /** Diagonal loading of the autocorrelation system. */
  readonly loadDiag?: number
}

/**
 * The signal-to-distortion ratio of BSS Eval (Vincent et al., 2006) as torchmetrics and fast_bss_eval compute it: the
 * target is allowed a distortion filter of `filterLength` taps, fitted by least squares from the Toeplitz system of
 * its autocorrelation; SDR = 10 log10(c / (1 - c)) for the coherence c of the normalised signals.
 *
 * @param target - Reference signals `[T]` or `[B, T]`.
 * @param preds - Estimated signals, the same shape.
 * @param options - filterLength, zeroMean, loadDiag.
 * @returns The metric result in dB.
 * @example
 * ```ts
 * import { signalDistortionRatio } from 'mdml-engine/metrics'
 * const t = [1, 0.5, -0.3, 0.8, -1, 0.2, 0.4, 0]
 * // A delayed, scaled copy is within reach of the distortion filter.
 * signalDistortionRatio(t, [0, ...t.slice(0, 7).map((v) => 2 * v)], { filterLength: 4 }).estimate.value > 60 // => true
 * ```
 */
export const signalDistortionRatio: MetricFunction<[TensorLike, TensorLike, SdrOptions?]> = defineMetric(
  spec('signalDistortionRatio', 'higher'),
  (target: TensorLike, preds: TensorLike, options: SdrOptions = {}): MetricResult =>
    batched('signalDistortionRatio', target, preds, (t0, p0) => {
      const length = options.filterLength ?? 512
      const normalise = (x: Float64Array): Float64Array => {
        const n = Math.max(Math.sqrt(dot(x, x)), 1e-6)
        return Float64Array.from(x, (v) => v / n)
      }
      const t = normalise(options.zeroMean === true ? centred(t0) : t0)
      const p = normalise(options.zeroMean === true ? centred(p0) : p0)
      const r = new Float64Array(length)
      const b = new Float64Array(length)
      for (let k = 0; k < length; k++) {
        let rk = 0
        let bk = 0
        for (let i = 0; i + k < t.length; i++) {
          rk += (t[i] ?? 0) * (t[i + k] ?? 0)
          bk += (t[i] ?? 0) * (p[i + k] ?? 0)
        }
        r[k] = rk
        b[k] = bk
      }
      if (options.loadDiag !== undefined) r[0] = (r[0] ?? 0) + options.loadDiag
      const sol = toFlat(solveToeplitz(r, undefined, b))
      const coh = dot(b, sol)
      return 10 * Math.log10(coh / (1 - coh))
    }),
)

function siSdr(t0: Float64Array, p0: Float64Array, zeroMean: boolean): number {
  const t = zeroMean ? centred(t0) : t0
  const p = zeroMean ? centred(p0) : p0
  const alpha = (dot(p, t) + EPS) / (dot(t, t) + EPS)
  const scaled = Float64Array.from(t, (v) => alpha * v)
  const noise = Float64Array.from(scaled, (v, i) => v - (p[i] ?? 0))
  return 10 * Math.log10((dot(scaled, scaled) + EPS) / (dot(noise, noise) + EPS))
}

/**
 * The scale-invariant SDR (Le Roux et al., 2019; torchmetrics): the SNR of the estimate against its projection onto
 * the target, so rescaling the estimate does not change it.
 *
 * @param target - Reference signals `[T]` or `[B, T]`.
 * @param preds - Estimated signals, the same shape.
 * @param options - zeroMean.
 * @returns The metric result in dB.
 * @example
 * ```ts
 * import { scaleInvariantSdr } from 'mdml-engine/metrics'
 * const a = scaleInvariantSdr([1, 2, 3, 4], [1, 2, 3, 3]).estimate.value
 * const b = scaleInvariantSdr([1, 2, 3, 4], [5, 10, 15, 15]).estimate.value
 * Math.abs(a - b) < 1e-9 // => true
 * ```
 */
export const scaleInvariantSdr: MetricFunction<[TensorLike, TensorLike, { readonly zeroMean?: boolean }?]> =
  defineMetric(
    spec('scaleInvariantSdr', 'higher'),
    (target: TensorLike, preds: TensorLike, options: { readonly zeroMean?: boolean } = {}): MetricResult =>
      batched('scaleInvariantSdr', target, preds, (t, p) => siSdr(t, p, options.zeroMean === true)),
  )

/**
 * The scale-invariant SNR (torchmetrics): the scale-invariant SDR of zero-mean signals.
 *
 * @param target - Reference signals `[T]` or `[B, T]`.
 * @param preds - Estimated signals, the same shape.
 * @returns The metric result in dB.
 * @example
 * ```ts
 * import { scaleInvariantSnr } from 'mdml-engine/metrics'
 * scaleInvariantSnr([1, 2, 3, 4], [2, 3, 4, 5]).estimate.value > 100 // => true
 * ```
 */
export const scaleInvariantSnr: MetricFunction<[TensorLike, TensorLike]> = defineMetric(
  spec('scaleInvariantSnr', 'higher'),
  (target: TensorLike, preds: TensorLike): MetricResult =>
    batched('scaleInvariantSnr', target, preds, (t, p) => siSdr(t, p, true)),
)

/** The best assignment of estimated sources to targets. */
export interface PermutationInvariantResult {
  /** The mean metric over sources under the best permutation. */
  readonly value: number
  /** permutation[i] is the estimate assigned to target i. */
  readonly permutation: readonly number[]
}

/**
 * Permutation-invariant evaluation (Yu et al., 2017; torchmetrics' permutation_invariant_training): the metric of
 * each (target, estimate) pair, and the assignment of estimates to targets with the best mean, found by the Hungarian
 * algorithm (`optim` `linearSumAssignment`, as torchmetrics does with scipy for many speakers).
 *
 * @param target - Reference sources `[S, T]`.
 * @param preds - Estimated sources `[S, T]`, in any order.
 * @param metric - A pairwise metric metric(targetSource, estimatedSource), e.g. SI-SDR.
 * @param options - direction: 'max' (default) or 'min'.
 * @returns The best mean value and the permutation.
 * @example
 * ```ts
 * import { permutationInvariant } from 'mdml-engine/metrics'
 * const neg = (a: Float64Array, b: Float64Array) => -a.reduce((s, v, i) => s + (v - (b[i] ?? 0)) ** 2, 0)
 * permutationInvariant([[1, 0], [0, 1]], [[0, 1], [1, 0]], neg).permutation // => [1, 0]
 * ```
 */
export function permutationInvariant(
  target: TensorLike,
  preds: TensorLike,
  metric: (target: Float64Array, estimate: Float64Array) => number,
  options: { readonly direction?: 'max' | 'min' } = {},
): PermutationInvariantResult {
  const t = signals('permutationInvariant', target)
  const p = signals('permutationInvariant', preds)
  if (t.length !== p.length) throw new MdmlError('shape', 'permutationInvariant: as many estimates as targets')
  const s = t.length
  const m = t.map((ti) => p.map((pj) => metric(ti, pj)))
  const r = linearSumAssignment(m, { maximize: options.direction !== 'min' })
  return { value: r.cost / s, permutation: [...r.colInd] }
}

// ---------------------------------------------------------------------------------------------------------------
// STOI (pystoi)

const STOI_FS = 10000
const STOI_FRAME = 256
const STOI_NFFT = 512
const STOI_BANDS = 15
const STOI_MIN_FREQ = 150
const STOI_N = 30
const STOI_BETA = -15
const STOI_DYN_RANGE = 40

/** The one-third octave band matrix of pystoi.utils.thirdoct. */
function thirdOctaveBands(): Float64Array[] {
  const bins = STOI_NFFT / 2 + 1
  const f = Float64Array.from({ length: bins }, (_, i) => (i * STOI_FS) / STOI_NFFT)
  const nearest = (v: number): number => argmaxOf(Float64Array.from(f, (x) => -((x - v) ** 2)))
  return Array.from({ length: STOI_BANDS }, (_, k) => {
    const lo = nearest(STOI_MIN_FREQ * 2 ** ((2 * k - 1) / 6))
    const hi = nearest(STOI_MIN_FREQ * 2 ** ((2 * k + 1) / 6))
    return Float64Array.from({ length: bins }, (_, i) => (i >= lo && i < hi ? 1 : 0))
  })
}

/** pystoi's resample_oct: Matlab's resample with a Kaiser-windowed sinc filter. */
function resampleOct(x: Float64Array, p: number, q: number): Float64Array {
  const g = (() => {
    let a = p
    let b = q
    while (b !== 0) [a, b] = [b, a % b]
    return a
  })()
  const pp = p / g
  const qq = q / g
  const stop = 1 / (2 * Math.max(pp, qq))
  const roll = stop / 10
  const rejection = 60
  const l = Math.ceil((rejection - 8) / (28.714 * roll))
  const beta = 0.1102 * (rejection - 8.7)
  // A Kaiser-windowed sinc low-pass normalised to unit DC gain: firwin with cutoff 2 stop (fractions of Nyquist).
  const h = firwin(2 * l + 1, 2 * stop, { window: ['kaiser', beta] })
  return Float64Array.from(primalOf(resamplePoly(x, p, q, { window: h })).data, Number)
}

/** pystoi's frame window: numpy.hanning(framelen + 2)[1:-1]. */
const stoiWindow = (): Float64Array => Float64Array.from(toFlat(hann(STOI_FRAME + 2)).slice(1, -1))

/** pystoi's remove_silent_frames: drop frames more than 40 dB below the loudest clean frame, then overlap-add. */
function removeSilentFrames(x: Float64Array, y: Float64Array): [Float64Array, Float64Array] {
  const hop = STOI_FRAME / 2
  const w = stoiWindow()
  const starts = frameStarts(x.length, STOI_FRAME, hop, true)
  const energies = starts.map((s) => {
    let e = 0
    for (let i = 0; i < STOI_FRAME; i++) e += ((w[i] ?? 0) * (x[s + i] ?? 0)) ** 2
    return 20 * Math.log10(Math.sqrt(e) + EPS)
  })
  const top = energies.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY)
  const kept = starts.filter((_, i) => top - STOI_DYN_RANGE - (energies[i] ?? 0) < 0)
  const length = kept.length === 0 ? 0 : (kept.length - 1) * hop + STOI_FRAME
  const xs = new Float64Array(length)
  const ys = new Float64Array(length)
  for (const [f, s] of kept.entries()) {
    for (let i = 0; i < STOI_FRAME; i++) {
      xs[f * hop + i] = (xs[f * hop + i] ?? 0) + (w[i] ?? 0) * (x[s + i] ?? 0)
      ys[f * hop + i] = (ys[f * hop + i] ?? 0) + (w[i] ?? 0) * (y[s + i] ?? 0)
    }
  }
  return [xs, ys]
}

/** One-third octave band envelopes sqrt(OBM |X|^2): bands by frames. */
function bandEnvelopes(x: Float64Array, obm: readonly Float64Array[]): Float64Array[] {
  const powers = framePowers(x, stoiWindow(), STOI_NFFT, STOI_FRAME / 2, true)
  return obm.map((band) => Float64Array.from(powers, (pw) => Math.sqrt(dot(band, pw))))
}

const meanOf = (x: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] ?? 0
  return s / x.length
}
const norm = (x: ArrayLike<number>): number => Math.sqrt(dot(x, x))

/**
 * The short-time objective intelligibility (Taal et al., 2011) and its extended form ESTOI (Jensen and Taal, 2016), as
 * pystoi computes them: signals resampled to 10 kHz, silent frames removed, one-third octave band envelopes over
 * 384 ms segments, and the mean correlation of clipped, normalised envelopes (ESTOI: of row- and column-normalised
 * segments).
 *
 * @param clean - Clean speech `[T]`.
 * @param processed - Processed speech `[T]`.
 * @param options - fs (Hz) and extended (ESTOI).
 * @returns The intelligibility in about [0, 1] (1e-5 when too short).
 * @example
 * ```ts
 * import { shortTimeObjectiveIntelligibility } from 'mdml-engine/metrics'
 * const x = Array.from({ length: 12000 }, (_, i) => Math.sin(i / 7) * Math.sin(i / 900))
 * Number(shortTimeObjectiveIntelligibility(x, x, { fs: 10000 }).estimate.value.toFixed(6)) // => 1
 * ```
 */
export const shortTimeObjectiveIntelligibility: MetricFunction<
  [TensorLike, TensorLike, { readonly fs: number; readonly extended?: boolean }]
> = defineMetric(
  spec('shortTimeObjectiveIntelligibility', 'higher', [-1, 1]),
  (
    clean: TensorLike,
    processed: TensorLike,
    options: { readonly fs: number; readonly extended?: boolean },
  ): MetricResult => {
    let x = signal('shortTimeObjectiveIntelligibility', clean)
    let y = signal('shortTimeObjectiveIntelligibility', processed)
    if (x.length !== y.length) throw new MdmlError('shape', 'shortTimeObjectiveIntelligibility: lengths differ')
    if (options.fs !== STOI_FS) {
      x = resampleOct(x, STOI_FS, options.fs)
      y = resampleOct(y, STOI_FS, options.fs)
    }
    ;[x, y] = removeSilentFrames(x, y)
    const obm = thirdOctaveBands()
    const xb = bandEnvelopes(x, obm)
    const yb = bandEnvelopes(y, obm)
    const frames = xb[0]?.length ?? 0
    const name = 'shortTimeObjectiveIntelligibility'
    if (frames < STOI_N) return metricResult(name, 1e-5, 0)
    const segments = frames - STOI_N + 1
    const clip = 10 ** (-STOI_BETA / 20)
    let total = 0
    for (let m = STOI_N; m <= frames; m++) {
      const xs = xb.map((row) => row.slice(m - STOI_N, m))
      const ys = yb.map((row) => row.slice(m - STOI_N, m))
      if (options.extended === true) {
        const normalise = (seg: Float64Array[]): Float64Array[] => {
          const rows = seg.map((r) => {
            const mu = meanOf(r)
            const c = Float64Array.from(r, (v) => v - mu)
            const n = norm(c)
            return Float64Array.from(c, (v) => v / n)
          })
          for (let j = 0; j < STOI_N; j++) {
            const col = rows.map((r) => r[j] ?? 0)
            const mu = meanOf(col)
            const n = Math.sqrt(col.reduce((s, v) => s + (v - mu) ** 2, 0))
            for (const r of rows) r[j] = ((r[j] ?? 0) - mu) / n
          }
          return rows
        }
        const xn = normalise(xs)
        const yn = normalise(ys)
        for (const [b, r] of xn.entries()) total += dot(r, yn[b] ?? []) / STOI_N
      } else {
        for (const [b, xr] of xs.entries()) {
          const yr = ys[b] ?? new Float64Array(STOI_N)
          const c = norm(xr) / (norm(yr) + EPS)
          const yp = Float64Array.from(yr, (v, j) => Math.min(v * c, (xr[j] ?? 0) * (1 + clip)))
          const ym = meanOf(yp)
          const xm = meanOf(xr)
          const yc = Float64Array.from(yp, (v) => v - ym)
          const xc = Float64Array.from(xr, (v) => v - xm)
          total += dot(yc, xc) / ((norm(yc) + EPS) * (norm(xc) + EPS))
        }
      }
    }
    const value = options.extended === true ? total / segments : total / (segments * STOI_BANDS)
    return metricResult(name, value, segments)
  },
)

/** Options of {@link logSpectralDistance}. */
export interface LsdOptions {
  /** Frame (and FFT) length (default 512) and hop (default 128). */
  readonly nfft?: number
  readonly hop?: number
  /** Added to the power spectra before the logarithm (default 1e-10). */
  readonly eps?: number
}

/**
 * The log-spectral distance: the mean over frames of the root-mean-square over frequencies of 10 log10 of the ratio
 * of power spectra, with a periodic Hann STFT scaled as scipy.signal.stft.
 *
 * @param target - Reference signal `[T]`.
 * @param preds - Estimated signal `[T]`.
 * @param options - nfft, hop and eps.
 * @returns The distance in dB.
 * @example
 * ```ts
 * import { logSpectralDistance } from 'mdml-engine/metrics'
 * const x = Array.from({ length: 1024 }, (_, i) => Math.sin(i / 5))
 * logSpectralDistance(x, x).estimate.value // => 0
 * ```
 */
export const logSpectralDistance: MetricFunction<[TensorLike, TensorLike, LsdOptions?]> = defineMetric(
  spec('logSpectralDistance', 'lower', [0, Infinity]),
  (target: TensorLike, preds: TensorLike, options: LsdOptions = {}): MetricResult => {
    const t = signal('logSpectralDistance', target)
    const p = signal('logSpectralDistance', preds)
    const nfft = options.nfft ?? 512
    const hop = options.hop ?? 128
    const eps = options.eps ?? 1e-10
    const w = Float64Array.from(toFlat(hann(nfft, { sym: false })))
    const scale = 1 / w.reduce((a, b) => a + b, 0) ** 2
    const pt = framePowers(t, w, nfft, hop, false, scale)
    const pp = framePowers(p, w, nfft, hop, false, scale)
    let sum = 0
    for (const [f, a] of pt.entries()) {
      const b = pp[f] ?? new Float64Array(a.length)
      let s = 0
      for (let k = 0; k < a.length; k++) s += (10 * Math.log10(((a[k] ?? 0) + eps) / ((b[k] ?? 0) + eps))) ** 2
      sum += Math.sqrt(s / a.length)
    }
    return metricResult('logSpectralDistance', sum / pt.length, pt.length)
  },
)

/**
 * The mel-cepstral distortion (Kubichek, 1993): (10 / ln 10) sqrt(2 sum_d (c_d - c'_d)^2) per frame, excluding the
 * energy coefficient c_0 by default, averaged over frames; with `align: 'dtw'` frames are first aligned by dynamic
 * time warping on the same distance (steps (1, 0), (0, 1), (1, 1)) and the mean is over the path. mdml ships no
 * mel-cepstral analysis; pass the cepstra of each frame.
 *
 * @param targetCepstra - Reference mel-cepstra `[T, D]`.
 * @param predCepstra - Synthesised mel-cepstra `[T', D]` (T' = T unless aligned).
 * @param options - includeC0 and align.
 * @returns The distortion in dB.
 * @example
 * ```ts
 * import { melCepstralDistortion } from 'mdml-engine/metrics'
 * const k = 10 / Math.log(10)
 * melCepstralDistortion([[0, 1, 0]], [[5, 1, 1]]).estimate.value === k * Math.SQRT2 // => true
 * ```
 */
export const melCepstralDistortion: MetricFunction<
  [TensorLike, TensorLike, { readonly includeC0?: boolean; readonly align?: 'none' | 'dtw' }?]
> = defineMetric(
  spec('melCepstralDistortion', 'lower', [0, Infinity]),
  (
    targetCepstra: TensorLike,
    predCepstra: TensorLike,
    options: { readonly includeC0?: boolean; readonly align?: 'none' | 'dtw' } = {},
  ): MetricResult => {
    const a = matrix(targetCepstra)
    const b = matrix(predCepstra)
    const name = 'melCepstralDistortion'
    if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[1])
      throw new MdmlError('shape', `${name}: cepstra are [frames, D] with the same D`)
    const d = a.shape[1] ?? 0
    const ta = a.shape[0] ?? 0
    const tb = b.shape[0] ?? 0
    const first = options.includeC0 === true ? 0 : 1
    const k = (10 / Math.log(10)) * Math.SQRT2
    const dist = (i: number, j: number): number => {
      let s = 0
      for (let c = first; c < d; c++) s += ((a.data[i * d + c] ?? 0) - (b.data[j * d + c] ?? 0)) ** 2
      return k * Math.sqrt(s)
    }
    if (options.align !== 'dtw') {
      if (ta !== tb) throw new MdmlError('shape', `${name}: ${ta} and ${tb} frames; use align: 'dtw'`)
      let s = 0
      for (let i = 0; i < ta; i++) s += dist(i, i)
      return metricResult(name, s / ta, ta)
    }
    // Symmetric1 DTW (unit weights) on the scaled cepstra without c0: the mean distance along the warping path.
    const features = (m: { readonly data: Float64Array }, t: number): number[][] =>
      Array.from({ length: t }, (_, i) =>
        Array.from({ length: d - first }, (_, c) => k * (m.data[i * d + first + c] ?? 0)),
      )
    const r = dtw(features(a, ta), features(b, tb), { stepPattern: 'symmetric1' })
    return metricResult(name, r.distance / r.index1.length, r.index1.length)
  },
)

/** A DNSMOS model: the two networks of Reddy et al. (2021, 2022), supplied by the caller. */
export interface DnsmosModel {
  /** The raw P.835 scores [signal, background, overall] of a 9.01 s segment at 16 kHz. */
  readonly p835: (segment: Float64Array) => readonly [number, number, number]
  /** The P.808 MOS of the segment without its last 160 samples (the network computes its own log-mel input). */
  readonly p808: (segment: Float64Array) => number
}

const polyval = (c: readonly number[], x: number): number => c.reduce((acc, v) => acc * x + v, 0)

/**
 * DNSMOS (Reddy et al., 2021, 2022) from a supplied model, with torchmetrics' segmentation and calibration: the clip
 * (repeated up to 9.01 s) is cut into 9.01 s windows with 1 s hops, the P.835 outputs are mapped through the
 * published polynomial fits (personalised or not), and the scores are averaged over windows. mdml ships no
 * networks; supply the ONNX models (or any stand-in) as functions. Audio must be at 16 kHz.
 *
 * @param audio - A mono clip `[T]` at 16 kHz.
 * @param model - The P.835 and P.808 networks.
 * @param options - personalized (penalise interfering speakers).
 * @returns The overall MOS, with p808, sig, bak and ovrl in the breakdown.
 * @example
 * ```ts
 * import { dnsmos } from 'mdml-engine/metrics'
 * const model = { p835: () => [3, 3, 3] as const, p808: () => 3.5 }
 * dnsmos(new Array(16000).fill(0), model).breakdown?.['p808'] // => 3.5
 * ```
 */
export const dnsmos: MetricFunction<[TensorLike, DnsmosModel, { readonly personalized?: boolean }?]> = defineMetric(
  spec('dnsmos', 'higher', [1, 5]),
  (audio: TensorLike, model: DnsmosModel, options: { readonly personalized?: boolean } = {}): MetricResult => {
    const fs = 16000
    const length = Math.trunc(9.01 * fs)
    let x = signal('dnsmos', audio)
    if (x.length === 0) throw new MdmlError('value', 'dnsmos: empty audio')
    while (x.length < length) {
      const y = new Float64Array(2 * x.length)
      y.set(x)
      y.set(x, x.length)
      x = y
    }
    const [ovr, sig, bak] =
      options.personalized === true
        ? [
            [-0.00533021, 0.005101, 1.18058466, -0.11236046],
            [-0.01019296, 0.02751166, 1.19576786, -0.24348726],
            [-0.04976499, 0.44276479, -0.1644611, 0.96883132],
          ]
        : [
            [-0.06766283, 1.11546468, 0.04602535],
            [-0.08397278, 1.22083953, 0.0052439],
            [-0.13166888, 1.60915514, -0.39604546],
          ]
    const hops = Math.trunc(Math.floor(x.length / fs) - 9.01) + 1
    const sums = [0, 0, 0, 0]
    let count = 0
    for (let h = 0; h < hops; h++) {
      const segment = x.slice(Math.trunc(h * fs), Math.trunc((h + 9.01) * fs))
      if (segment.length < length) continue
      const [s, b, o] = model.p835(segment)
      const scores = [
        model.p808(segment.slice(0, -160)),
        polyval(sig ?? [], s),
        polyval(bak ?? [], b),
        polyval(ovr ?? [], o),
      ]
      for (let k = 0; k < 4; k++) sums[k] = (sums[k] ?? 0) + (scores[k] ?? 0)
      count++
    }
    const [p808, mosSig, mosBak, mosOvr] = sums.map((v) => v / count)
    return metricResult('dnsmos', mosOvr ?? Number.NaN, count, {
      p808: p808 ?? Number.NaN,
      sig: mosSig ?? Number.NaN,
      bak: mosBak ?? Number.NaN,
      ovrl: mosOvr ?? Number.NaN,
    })
  },
)
