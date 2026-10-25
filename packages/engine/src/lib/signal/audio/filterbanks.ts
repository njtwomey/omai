/**
 * Auditory filter banks: the triangular mel filter bank and the chroma filter bank that folds frequencies onto the
 * twelve pitch classes (librosa.filters), and the gammatone filter, a model of the cochlea's frequency selectivity
 * (scipy.signal.gammatone), with a bank of them that turns a signal into a cochleagram.
 */

import { WhiteboxError } from '../../base/index.ts'
import { stack, type Tensor, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { constant } from '../_shared.ts'
import { lfilter } from '../filters/index.ts'
import { fftFrequencies, melFrequencies } from './scales.ts'

// Auditory filter banks: triangular mel filters and chroma filters (as librosa.filters), and gammatone filters
// (scipy.signal.gammatone: the FIR impulse response or Slaney's fourth-order IIR) with banks of them on the ERB scale.

/** Row normalisation of a filter bank: Slaney's area normalisation, a p-norm, or none. */
export type FilterNorm = 'slaney' | number | null

/** Normalise each row (or column) of a matrix to unit p-norm, leaving near-zero rows alone (librosa.util.normalize). */
function normalizeRows(w: Float64Array, rows: number, cols: number, p: number, byColumn: boolean): void {
  const outer = byColumn ? cols : rows
  const inner = byColumn ? rows : cols
  const at = (o: number, i: number): number => (byColumn ? i * cols + o : o * cols + i)
  for (let o = 0; o < outer; o++) {
    let len = 0
    if (p === Number.POSITIVE_INFINITY) for (let i = 0; i < inner; i++) len = Math.max(len, Math.abs(w[at(o, i)] ?? 0))
    else {
      for (let i = 0; i < inner; i++) len += Math.abs(w[at(o, i)] ?? 0) ** p
      len **= 1 / p
    }
    if (len < Number.MIN_VALUE * 2 ** 52) continue
    for (let i = 0; i < inner; i++) w[at(o, i)] = (w[at(o, i)] ?? 0) / len
  }
}

/** Options for {@link melFilterBank}. */
export interface MelFilterOptions {
  /** Number of filters (default 128). */
  readonly nMels?: number
  /** Lowest edge in Hz (default 0). */
  readonly fmin?: number
  /** Highest edge in Hz (default sr / 2). */
  readonly fmax?: number
  /** HTK mel formula (default false: Slaney's). */
  readonly htk?: boolean
  /** Normalisation (default 'slaney': each triangle has unit area in Hz). */
  readonly norm?: FilterNorm
}

/**
 * The mel filter bank (librosa.filters.mel): triangular filters on the rFFT bins between mel-spaced edges.
 *
 * @param sr - Sampling rate.
 * @param nfft - FFT length.
 * @param options - Number of filters, band edges, formula and normalisation.
 * @returns The weights `[nMels, nfft / 2 + 1]`.
 * @concept mel-filter-bank
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * audio.melFilterBank(16000, 512, { nMels: 40 }).shape // => [40, 257]
 * ```
 */
export function melFilterBank(sr: number, nfft: number, options: MelFilterOptions = {}): Tensor {
  const nMels = options.nMels ?? 128
  const fmax = options.fmax ?? sr / 2
  const htk = options.htk ?? false
  const bins = Math.floor(nfft / 2) + 1
  const fft = toFlat(fftFrequencies(sr, nfft))
  const melF = toFlat(melFrequencies(nMels + 2, options.fmin ?? 0, fmax, htk))
  const w = new Float64Array(nMels * bins)
  for (let i = 0; i < nMels; i++) {
    const d0 = (melF[i + 1] ?? 0) - (melF[i] ?? 0)
    const d1 = (melF[i + 2] ?? 0) - (melF[i + 1] ?? 0)
    for (let k = 0; k < bins; k++) {
      const lower = -((melF[i] ?? 0) - (fft[k] ?? 0)) / d0
      const upper = ((melF[i + 2] ?? 0) - (fft[k] ?? 0)) / d1
      w[i * bins + k] = Math.max(0, Math.min(lower, upper))
    }
  }
  const norm = options.norm === undefined ? 'slaney' : options.norm
  if (norm === 'slaney')
    for (let i = 0; i < nMels; i++) {
      const e = 2 / ((melF[i + 2] ?? 0) - (melF[i] ?? 0))
      for (let k = 0; k < bins; k++) w[i * bins + k] = (w[i * bins + k] ?? 0) * e
    }
  else if (typeof norm === 'number') normalizeRows(w, nMels, bins, norm, false)
  return constant(w, [nMels, bins])
}

/** Options for {@link chromaFilterBank}. */
export interface ChromaFilterOptions {
  /** Bins per octave (default 12). */
  readonly nChroma?: number
  /** Tuning deviation in fractions of a bin (default 0). */
  readonly tuning?: number
  /** Centre of the octave-weighting Gaussian in octaves above C0 (default 5). */
  readonly ctroct?: number
  /** Its width in octaves, or null for flat weighting (default 2). */
  readonly octwidth?: number | null
  /** Column normalisation (default 2). */
  readonly norm?: number | null
  /** Start the bins at C rather than A (default true). */
  readonly baseC?: boolean
}

/**
 * The chroma filter bank (librosa.filters.chroma, after Ellis's chromagram): each rFFT bin spreads over the pitch
 * classes as a Gaussian in log frequency, weighted by a Gaussian over octaves.
 *
 * @param sr - Sampling rate.
 * @param nfft - FFT length.
 * @param options - Bins per octave, tuning, octave weighting, normalisation and base.
 * @returns The weights `[nChroma, nfft / 2 + 1]`.
 * @concept chroma
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * audio.chromaFilterBank(22050, 2048).shape // => [12, 1025]
 * ```
 */
export function chromaFilterBank(sr: number, nfft: number, options: ChromaFilterOptions = {}): Tensor {
  const n = options.nChroma ?? 12
  const tuning = options.tuning ?? 0
  const a440 = 440 * 2 ** (tuning / n)
  const frq = new Float64Array(nfft)
  for (let k = 1; k < nfft; k++) frq[k] = n * Math.log2((k * sr) / nfft / (a440 / 16))
  frq[0] = (frq[1] ?? 0) - 1.5 * n
  const width = Float64Array.from({ length: nfft }, (_, k) =>
    k + 1 < nfft ? Math.max((frq[k + 1] ?? 0) - (frq[k] ?? 0), 1) : 1,
  )
  const half = Math.round(n / 2)
  const w = new Float64Array(n * nfft)
  for (let c = 0; c < n; c++)
    for (let k = 0; k < nfft; k++) {
      const raw = (frq[k] ?? 0) - c + half + 10 * n
      const d = raw - Math.floor(raw / n) * n - half
      w[c * nfft + k] = Math.exp(-0.5 * ((2 * d) / (width[k] ?? 1)) ** 2)
    }
  const norm = options.norm === undefined ? 2 : options.norm
  if (norm !== null) normalizeRows(w, n, nfft, norm, true)
  const octwidth = options.octwidth === undefined ? 2 : options.octwidth
  if (octwidth !== null)
    for (let k = 0; k < nfft; k++) {
      const g = Math.exp(-0.5 * (((frq[k] ?? 0) / n - (options.ctroct ?? 5)) / octwidth) ** 2)
      for (let c = 0; c < n; c++) w[c * nfft + k] = (w[c * nfft + k] ?? 0) * g
    }
  const shift = (options.baseC ?? true) ? 3 * Math.floor(n / 12) : 0
  const bins = Math.floor(1 + nfft / 2)
  const out = new Float64Array(n * bins)
  for (let c = 0; c < n; c++) {
    const src = (((c + shift) % n) + n) % n
    for (let k = 0; k < bins; k++) out[c * bins + k] = w[src * nfft + k] ?? 0
  }
  return constant(out, [n, bins])
}

/** Options for {@link gammatone}. */
export interface GammatoneOptions {
  /** Sampling rate (default 2, so the frequency is a fraction of Nyquist). */
  readonly fs?: number
  /** FIR order (default 4; the IIR design is always fourth order). */
  readonly order?: number
  /** FIR length (default max(fs 0.015, 15)). */
  readonly numtaps?: number
}

/**
 * A gammatone filter (scipy.signal.gammatone): the FIR impulse response t^(n-1) e^(-2 pi b t) cos(2 pi f t) with
 * b = 1.019 ERB(f) (Slaney's ERB: f / 9.26449 + 24.7), scaled to unit peak gain, or Slaney's fourth-order IIR
 * approximation.
 *
 * @param freq - Centre frequency (0 < freq < fs / 2).
 * @param ftype - 'fir' or 'iir'.
 * @param options - Sampling rate, order and length.
 * @returns Numerator and denominator.
 * @concept gammatone-filter
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const g = audio.gammatone(1000, 'iir', { fs: 16000 })
 * [g.b.length, g.a.length] // => [5, 9]
 * ```
 */
export function gammatone(
  freq: number,
  ftype: 'fir' | 'iir',
  options: GammatoneOptions = {},
): { readonly b: number[]; readonly a: number[] } {
  const fs = options.fs ?? 2
  if (!(freq > 0 && freq < fs / 2)) throw new WhiteboxError('value', 'gammatone: the frequency must be in (0, fs / 2)')
  const erb = freq / 9.26449 + 24.7
  if (ftype === 'fir') {
    const order = options.order ?? 4
    if (!Number.isInteger(order) || order < 1 || order > 24)
      throw new WhiteboxError('value', 'gammatone: order in 1..24')
    const numtaps = options.numtaps ?? Math.max(Math.trunc(fs * 0.015), 15)
    const bw = 1.019 * erb
    let fact = 1
    for (let i = 2; i < order; i++) fact *= i
    const scale = (2 * (2 * Math.PI * bw) ** order) / fact / fs
    const b = Array.from({ length: numtaps }, (_, i) => {
      const t = i / fs
      return t ** (order - 1) * Math.exp(-2 * Math.PI * bw * t) * Math.cos(2 * Math.PI * freq * t) * scale
    })
    return { b, a: [1] }
  }
  const T = 1 / fs
  const bw = 2 * Math.PI * 1.019 * erb
  const fr = 2 * freq * Math.PI * T
  const bwT = bw * T
  // Complex arithmetic of Slaney's gain computation.
  type C = readonly [number, number]
  const cmul = (p: C, q: C): C => [p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]]
  const cexp = (re: number, im: number): C => [Math.exp(re) * Math.cos(im), Math.exp(re) * Math.sin(im)]
  const cadd = (p: C, q: C): C => [p[0] + q[0], p[1] + q[1]]
  const cscale = (p: C, s: number): C => [p[0] * s, p[1] * s]
  const g1 = cscale(cexp(0, 2 * fr), -2 * T)
  const g2 = cscale(cexp(-bwT, fr), 2 * T)
  const g3 = Math.sqrt(3 + 2 ** 1.5) * Math.sin(fr)
  const g4 = Math.sqrt(3 - 2 ** 1.5) * Math.sin(fr)
  const g5 = cexp(0, 2 * fr)
  let g = cadd(g1, cscale(g2, Math.cos(fr) - g4))
  g = cmul(g, cadd(g1, cscale(g2, Math.cos(fr) + g4)))
  g = cmul(g, cadd(g1, cscale(g2, Math.cos(fr) - g3)))
  g = cmul(g, cadd(g1, cscale(g2, Math.cos(fr) + g3)))
  let den: C = cadd(cadd([-2 / Math.exp(2 * bwT), 0], cscale(g5, -2)), cscale(cadd([1, 0], g5), 2 / Math.exp(bwT)))
  den = cmul(cmul(den, den), cmul(den, den))
  const dd = den[0] * den[0] + den[1] * den[1]
  const q: C = [(g[0] * den[0] + g[1] * den[1]) / dd, (g[1] * den[0] - g[0] * den[1]) / dd]
  const gain = Math.hypot(q[0], q[1])
  const T4 = T ** 4
  const b = [
    T4 / gain,
    (-4 * T4 * Math.cos(fr)) / Math.exp(bwT) / gain,
    (6 * T4 * Math.cos(2 * fr)) / Math.exp(2 * bwT) / gain,
    (-4 * T4 * Math.cos(3 * fr)) / Math.exp(3 * bwT) / gain,
    (T4 * Math.cos(4 * fr)) / Math.exp(4 * bwT) / gain,
  ]
  const a = [
    1,
    (-8 * Math.cos(fr)) / Math.exp(bwT),
    (4 * (4 + 3 * Math.cos(2 * fr))) / Math.exp(2 * bwT),
    (-8 * (6 * Math.cos(fr) + Math.cos(3 * fr))) / Math.exp(3 * bwT),
    (2 * (18 + 16 * Math.cos(2 * fr) + Math.cos(4 * fr))) / Math.exp(4 * bwT),
    (-8 * (6 * Math.cos(fr) + Math.cos(3 * fr))) / Math.exp(5 * bwT),
    (4 * (4 + 3 * Math.cos(2 * fr))) / Math.exp(6 * bwT),
    (-8 * Math.cos(fr)) / Math.exp(7 * bwT),
    Math.exp(-8 * bwT),
  ]
  return { b, a }
}

/**
 * A gammatone filter bank applied to a signal: one gammatone filter per centre frequency (for example
 * {@link erbSpace}), each output a band of the cochleagram. Differentiable in the signal.
 *
 * @param x - Signal.
 * @param centres - Centre frequencies in Hz.
 * @param fs - Sampling rate.
 * @param ftype - 'iir' (default) or 'fir'.
 * @returns The band signals `[bands, n]`.
 * @concept gammatone-filter
 * @example
 * ```ts
 * import { audio } from 'whitebox-engine/signal'
 * const x = Array.from({ length: 64 }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / 8000))
 * audio.gammatoneBank(x, [500, 1000, 2000], 8000).shape // => [3, 64]
 * ```
 */
export function gammatoneBank(
  x: TensorLike,
  centres: ArrayLike<number>,
  fs: number,
  ftype: 'fir' | 'iir' = 'iir',
): Value {
  const bands = Array.from(centres, (f) => {
    const g = gammatone(f, ftype, { fs })
    return lfilter(g.b, g.a, x)
  })
  if (bands.length === 0) throw new WhiteboxError('value', 'gammatoneBank: no centre frequencies')
  return stack(bands)
}
