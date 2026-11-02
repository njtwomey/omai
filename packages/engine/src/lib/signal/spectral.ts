/**
 * Non-parametric power spectral density estimation, scipy.signal style: detrending, the periodogram, Welch's averaged
 * periodogram and Bartlett's method, the cross spectral density and magnitude-squared coherence, the Blackman-Tukey
 * correlogram, Thomson's multitaper estimate with DPSS tapers, and the degrees of freedom and chi-square confidence
 * intervals of an estimate.
 */

import { MdmlError } from '../base/index.ts'
import { chi2 } from '../dists/index.ts'
import { fft, fftfreq, rfft, rfftfreq } from '../numerics/fourier/index.ts'
import {
  add,
  type ComplexValue,
  concat,
  div,
  mean,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  slidingWindowView,
  sort,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { axisOf, constant, float, rowsOf } from './_shared.ts'
import type { Spectrum } from './signal.ts'
import { dpss, dpssRatios, getWindow, type WindowSpec } from './windows/index.ts'

// Non-parametric spectral estimation as scipy.signal: detrending, the periodogram, Welch's and Bartlett's averaged
// periodograms, cross-spectral densities and coherence, the Blackman-Tukey correlogram, Thomson's multitaper method
// and chi-square confidence intervals. Signals are real; every estimate is a composition of differentiable
// operations (windows, FFTs, products and means), so it is differentiable in the signal.

/** Options for {@link detrend}. */
export interface DetrendOptions {
  readonly axis?: number
  /** 'linear' (default: subtract least-squares lines) or 'constant' (subtract the mean). */
  readonly type?: 'linear' | 'constant'
  /** Breakpoints of a piecewise-linear trend (each segment is fitted separately). */
  readonly bp?: readonly number[]
}

/** Detrend rows `[batch, n]` along their last axis. */
function detrendRows(rows: Value, type: 'linear' | 'constant' | 'none', bp: readonly number[] = []): Value {
  if (type === 'none') return rows
  const n = rows.shape[1] ?? 0
  if (type === 'constant') return sub(rows, mean(rows, 1, true))
  const edges = [...new Set([0, ...bp, n])].sort((a, b) => a - b)
  if (edges.some((b) => b > n || b < 0)) throw new MdmlError('value', 'detrend: breakpoints must lie in [0, n]')
  const parts: Value[] = []
  for (let s = 0; s + 1 < edges.length; s++) {
    const lo = edges[s] ?? 0
    const hi = edges[s + 1] ?? n
    const seg = sliceAxis(rows, 1, lo, hi)
    const m = hi - lo
    const tc = Array.from({ length: m }, (_, i) => i - (m - 1) / 2)
    let ss = 0
    for (const v of tc) ss += v * v
    const t = constant(tc, [1, m])
    const centred = sub(seg, mean(seg, 1, true))
    parts.push(ss === 0 ? centred : sub(centred, mul(div(sum(mul(seg, t), 1, true), ss), t)))
  }
  return parts.length === 1 ? (parts[0] as Value) : concat(parts, 1)
}

/**
 * Remove a trend along an axis (scipy.signal.detrend): the mean, or the least-squares line (piecewise between
 * breakpoints). Linear in the data, so differentiable.
 *
 * @param x - Data.
 * @param options - Axis (default -1), trend type and breakpoints.
 * @returns The detrended data.
 * @concept detrending
 * @example
 * ```ts
 * import { detrend } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(detrend([1, 3, 5, 7])).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0, 0, 0]
 * ```
 */
export function detrend(x: TensorLike, options: DetrendOptions = {}): Value {
  const r = rowsOf(float(x), options.axis ?? -1, 'detrend')
  return r.restore(detrendRows(r.rows, options.type ?? 'linear', options.bp ?? []))
}

/** Options shared by the averaged-periodogram estimates. */
export interface WelchOptions {
  /** Sampling rate (default 1). */
  readonly fs?: number
  /** Window (default a periodic Hann window), or its samples. */
  readonly window?: WindowSpec | TensorLike
  /** Segment length (default 256, or the signal length if shorter, or the window length). */
  readonly nperseg?: number
  /** Overlap of segments (default nperseg / 2). */
  readonly noverlap?: number
  /** FFT length (default nperseg). */
  readonly nfft?: number
  /** Per-segment detrending (default 'constant'). */
  readonly detrend?: 'constant' | 'linear' | 'none'
  /** One-sided spectrum of a real signal (default true). */
  readonly returnOnesided?: boolean
  /** 'density' (V^2 / Hz, default) or 'spectrum' (V^2). */
  readonly scaling?: 'density' | 'spectrum'
  readonly axis?: number
  /** Average the segment periodograms by 'mean' (default) or bias-corrected 'median'. */
  readonly average?: 'mean' | 'median'
}

interface Segments {
  readonly win: Tensor
  readonly nperseg: number
  readonly noverlap: number
  readonly nfft: number
}

function isWindowSpec(w: unknown): w is WindowSpec {
  return typeof w === 'string' || typeof w === 'number' || (Array.isArray(w) && typeof w[0] === 'string')
}

function segmentsOf(n: number, options: WelchOptions, name: string): Segments {
  const w = options.window ?? 'hann'
  let win: Tensor
  let nperseg: number
  if (isWindowSpec(w)) {
    nperseg = Math.min(options.nperseg ?? 256, n)
    if (nperseg < 1) throw new MdmlError('value', `${name}: nperseg must be positive`)
    win = getWindow(w, nperseg)
  } else {
    win = primalOf(reshape(float(w), [-1]))
    nperseg = options.nperseg ?? win.shape[0] ?? 0
    if (nperseg !== win.shape[0]) throw new MdmlError('value', `${name}: nperseg must equal the window length`)
  }
  const nfft = options.nfft ?? nperseg
  if (nfft < nperseg) throw new MdmlError('value', `${name}: nfft must be at least nperseg`)
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2)
  if (noverlap >= nperseg) throw new MdmlError('value', `${name}: noverlap must be less than nperseg`)
  return { win, nperseg, noverlap, nfft }
}

/** Windowed, detrended FFTs of the segments of rows: `[batch, segments, bins]`. */
function segmentSpectra(
  rows: Value,
  s: Segments,
  detrendType: 'constant' | 'linear' | 'none',
  onesided: boolean,
): ComplexValue {
  const n = rows.shape[1] ?? 0
  const hop = s.nperseg - s.noverlap
  const count = Math.floor((n - s.noverlap) / hop)
  const frames = sliceAxis(slidingWindowView(rows, s.nperseg, { axis: 1, step: hop }), 1, 0, count)
  const [batch = 1] = rows.shape
  const flat = reshape(frames, [batch * count, s.nperseg])
  const detrended = detrendRows(flat, detrendType)
  const windowed = mul(detrended, reshape(s.win, [1, s.nperseg]))
  const spec = onesided ? rfft(windowed, s.nfft) : fft(windowed, s.nfft)
  const bins = spec.re.shape[1] ?? 0
  return { re: reshape(spec.re, [batch, count, bins]), im: reshape(spec.im, [batch, count, bins]) }
}

function medianBias(n: number): number {
  let s = 0
  for (let i = 1; i <= Math.floor((n - 1) / 2); i++) s += 1 / (2 * i + 1) - 1 / (2 * i)
  return 1 + s
}

/** The median along axis 1 of `[batch, count, bins]` values. */
function medianSegments(v: Value): Value {
  const count = v.shape[1] ?? 0
  const sorted = sort(v, 1)
  const mid = Math.floor(count / 2)
  if (count % 2 === 1) return reshape(sliceAxis(sorted, 1, mid, mid + 1), [v.shape[0] ?? 1, v.shape[2] ?? 0])
  const lo = sliceAxis(sorted, 1, mid - 1, mid)
  const hi = sliceAxis(sorted, 1, mid, mid + 1)
  return reshape(mul(add(lo, hi), 0.5), [v.shape[0] ?? 1, v.shape[2] ?? 0])
}

function averageSegments(v: Value, average: 'mean' | 'median'): Value {
  const count = v.shape[1] ?? 0
  if (count <= 1) return reshape(v, [v.shape[0] ?? 1, v.shape[2] ?? 0])
  if (average === 'mean') return mean(v, 1)
  return div(medianSegments(v), medianBias(count))
}

/** Scale and fold a segment cross-spectrum, `[batch, count, bins]`. */
function scaleFold(v: Value, s: Segments, fs: number, scaling: 'density' | 'spectrum', onesided: boolean): Value {
  const w = s.win.data
  let s1 = 0
  let s2 = 0
  for (const x of w) {
    s1 += x
    s2 += x * x
  }
  const scale = scaling === 'density' ? 1 / (fs * s2) : 1 / (s1 * s1)
  const bins = v.shape[2] ?? 0
  const fold = Array.from({ length: bins }, (_, k) => {
    if (!onesided || k === 0) return scale
    if (s.nfft % 2 === 0 && k === bins - 1) return scale
    return 2 * scale
  })
  return mul(v, constant(fold, [1, 1, bins]))
}

function frequenciesOf(nfft: number, fs: number, onesided: boolean): Tensor {
  return onesided ? (rfftfreq(nfft, 1 / fs) as Tensor) : (fftfreq(nfft, 1 / fs) as Tensor)
}

/** The real and imaginary parts of conj(X) Y for segment spectra. */
function crossProduct(x: ComplexValue, y: ComplexValue): ComplexValue {
  return {
    re: add(mul(x.re, y.re), mul(x.im, y.im)),
    im: sub(mul(x.re, y.im), mul(x.im, y.re)),
  }
}

/**
 * The cross power spectral density of two real signals by Welch's method (scipy.signal.csd): the average over
 * windowed, detrended, overlapping segments of conj(X) Y, scaled and folded to one side.
 *
 * @param x - First signal.
 * @param y - Second signal (the shorter is zero-padded).
 * @param options - Sampling rate, window, segments, FFT length, detrending, sides, scaling, axis and averaging.
 * @returns The frequencies and the complex cross spectrum `[..., bins]`.
 * @concept power-spectral-density
 * @example
 * ```ts
 * import { csd } from 'mdml-engine/signal'
 * csd([1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 1, 0, 1, 0, 1, 0], { nperseg: 4 }).values.re.shape // => [3]
 * ```
 */
export function csd(x: TensorLike, y: TensorLike, options: WelchOptions = {}): Spectrum<ComplexValue> {
  const axis = options.axis ?? -1
  const xv = float(x)
  const yv = float(y)
  const a = axisOf(axis, xv.shape.length, 'csd')
  const n = Math.max(xv.shape[a] ?? 0, yv.shape[axisOf(axis, yv.shape.length, 'csd')] ?? 0)
  const padTo = (v: Value): Value => {
    const r = rowsOf(v, axis, 'csd')
    return r.n === n
      ? r.rows
      : concat(
          [r.rows, constant(new Float64Array((r.rows.shape[0] ?? 1) * (n - r.n)), [r.rows.shape[0] ?? 1, n - r.n])],
          1,
        )
  }
  const rx = rowsOf(xv, axis, 'csd')
  const fs = options.fs ?? 1
  const onesided = options.returnOnesided ?? true
  const s = segmentsOf(n, options, 'csd')
  const det = options.detrend ?? 'constant'
  const sx = segmentSpectra(padTo(xv), s, det, onesided)
  const sy = segmentSpectra(padTo(yv), s, det, onesided)
  const c = crossProduct(sx, sy)
  const scaling = options.scaling ?? 'density'
  const average = options.average ?? 'mean'
  const re = averageSegments(scaleFold(c.re, s, fs, scaling, onesided), average)
  const im = averageSegments(scaleFold(c.im, s, fs, scaling, onesided), average)
  return {
    frequencies: frequenciesOf(s.nfft, fs, onesided),
    values: { re: rx.restore(re), im: rx.restore(im) },
    kind: scaling,
    fs,
  }
}

/**
 * The power spectral density by Welch's method (scipy.signal.welch): average the modified periodograms of
 * overlapping windowed segments, trading resolution for variance.
 *
 * @param x - Real signal.
 * @param options - Sampling rate, window, segments, FFT length, detrending, sides, scaling, axis and averaging.
 * @returns The frequencies and the PSD `[..., bins]`.
 * @concept power-spectral-density
 * @index function Welch power spectral density estimate.
 * @example
 * ```ts
 * import { welch } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(welch([1, -1, 1, -1, 1, -1, 1, -1], { nperseg: 4, window: 'boxcar' }).values) // => [0, 0, 4]
 * ```
 */
export function welch(x: TensorLike, options: WelchOptions = {}): Spectrum {
  const axis = options.axis ?? -1
  const r = rowsOf(float(x), axis, 'welch')
  const fs = options.fs ?? 1
  const onesided = options.returnOnesided ?? true
  const s = segmentsOf(r.n, options, 'welch')
  const spec = segmentSpectra(r.rows, s, options.detrend ?? 'constant', onesided)
  const power = add(mul(spec.re, spec.re), mul(spec.im, spec.im))
  const scaling = options.scaling ?? 'density'
  const values = averageSegments(scaleFold(power, s, fs, scaling, onesided), options.average ?? 'mean')
  return { frequencies: frequenciesOf(s.nfft, fs, onesided), values: r.restore(values), kind: scaling, fs }
}

/** Options for {@link periodogram}. */
export interface PeriodogramOptions {
  readonly fs?: number
  /** Window (default boxcar). */
  readonly window?: WindowSpec | TensorLike
  /** FFT length (default the signal length; shorter crops the signal). */
  readonly nfft?: number
  readonly detrend?: 'constant' | 'linear' | 'none'
  readonly returnOnesided?: boolean
  readonly scaling?: 'density' | 'spectrum'
  readonly axis?: number
}

/**
 * The (modified) periodogram (scipy.signal.periodogram): |FFT(w x)|^2, scaled to a density or a power spectrum.
 *
 * @param x - Real signal.
 * @param options - Sampling rate, window, FFT length, detrending, sides, scaling and axis.
 * @returns The frequencies and the PSD.
 * @concept power-spectral-density
 * @example
 * ```ts
 * import { periodogram } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(periodogram([1, -1, 1, -1]).values) // => [0, 0, 4]
 * ```
 */
export function periodogram(x: TensorLike, options: PeriodogramOptions = {}): Spectrum {
  const v = float(x)
  const axis = options.axis ?? -1
  const n = v.shape[axisOf(axis, v.shape.length, 'periodogram')] ?? 0
  let input = v
  let nperseg = n
  let nfft = options.nfft
  if (nfft !== undefined && nfft < n) {
    input = sliceAxis(v, axisOf(axis, v.shape.length, 'periodogram'), 0, nfft)
    nperseg = nfft
    nfft = undefined
  }
  return welch(input, {
    ...options,
    window: options.window ?? 'boxcar',
    nperseg,
    noverlap: 0,
    ...(nfft === undefined ? {} : { nfft }),
  })
}

/**
 * Bartlett's method: the average of the periodograms of non-overlapping rectangular-windowed segments (Welch with a
 * boxcar window and no overlap).
 *
 * @param x - Real signal.
 * @param nperseg - Segment length.
 * @param options - The other Welch options.
 * @returns The frequencies and the PSD.
 * @concept power-spectral-density
 * @example
 * ```ts
 * import { bartlettSpectrum } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(bartlettSpectrum([1, -1, 1, -1, 1, -1, 1, -1], 4).values) // => [0, 0, 4]
 * ```
 */
export function bartlettSpectrum(x: TensorLike, nperseg: number, options: WelchOptions = {}): Spectrum {
  return welch(x, { ...options, window: 'boxcar', nperseg, noverlap: 0 })
}

/**
 * Magnitude-squared coherence (scipy.signal.coherence): |Pxy|^2 / (Pxx Pyy) from Welch estimates, in [0, 1].
 *
 * @param x - First signal.
 * @param y - Second signal.
 * @param options - The Welch options.
 * @returns The frequencies and the coherence.
 * @concept coherence
 * @example
 * ```ts
 * import { coherence } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [1, 3, -2, 0.5, 2, -1, 0, 1.5, -0.5, 2.5, 1, -2]
 * toFlat(coherence(x, x.map((v) => 2 * v), { nperseg: 4 }).values).every((c) => Math.abs(c - 1) < 1e-12) // => true
 * ```
 */
export function coherence(x: TensorLike, y: TensorLike, options: WelchOptions = {}): Spectrum {
  const o = { ...options, returnOnesided: true }
  const pxx = welch(x, o)
  const pyy = welch(y, o)
  const pxy = csd(x, y, o)
  const mag = add(mul(pxy.values.re, pxy.values.re), mul(pxy.values.im, pxy.values.im))
  return { frequencies: pxx.frequencies, values: div(div(mag, pxx.values), pyy.values), kind: 'spectrum', fs: pxx.fs }
}

/** Options for {@link blackmanTukey}. */
export interface BlackmanTukeyOptions {
  readonly fs?: number
  /** Largest lag kept (default n / 4). */
  readonly maxLag?: number
  /** Lag window over lags -maxLag .. maxLag (default 'bartlett'). */
  readonly window?: WindowSpec
  /** FFT length (default the next power of two of 2 maxLag + 1, at least n). */
  readonly nfft?: number
}

/**
 * The Blackman-Tukey (correlogram) PSD: the Fourier transform of the biased autocorrelation estimate, tapered by a
 * lag window and truncated at maxLag. With maxLag n - 1 and a rectangular lag window it equals the periodogram.
 *
 * @param x - Real signal `[n]` (the mean is removed).
 * @param options - Sampling rate, maximum lag, lag window and FFT length.
 * @returns The one-sided frequencies and PSD.
 * @concept power-spectral-density
 * @example
 * ```ts
 * import { blackmanTukey } from 'mdml-engine/signal'
 * blackmanTukey([1, 2, 0, -1, 3, 1, -2, 0], { maxLag: 3 }).values.shape // => [5]
 * ```
 */
export function blackmanTukey(x: TensorLike, options: BlackmanTukeyOptions = {}): Spectrum {
  const v = reshape(float(x), [1, -1])
  const n = v.shape[1] ?? 0
  const fs = options.fs ?? 1
  const m = Math.min(options.maxLag ?? Math.floor(n / 4), n - 1)
  const nfft = options.nfft ?? Math.max(2 ** Math.ceil(Math.log2(2 * m + 1)), n)
  const xc = sub(v, mean(v, 1, true))
  const lags: Value[] = []
  for (let k = 0; k <= m; k++) lags.push(div(sum(mul(sliceAxis(xc, 1, 0, n - k), sliceAxis(xc, 1, k, n))), n))
  const w = toFlat(getWindow(options.window ?? 'bartlett', 2 * m + 1, false))
  // The windowed autocorrelation laid out circularly: r_0, r_1 .. r_m, zeros, r_m .. r_1.
  const zero = constant([0], [])
  const seq: Value[] = new Array<Value>(nfft).fill(zero)
  for (let k = 0; k <= m; k++) {
    const t = mul(lags[k] as Value, w[m + k] ?? 0)
    seq[k] = t
    if (k > 0) seq[nfft - k] = t
  }
  const r = reshape(stack(seq), [1, nfft])
  const spec = rfft(r)
  const bins = spec.re.shape[1] ?? 0
  const fold = Array.from({ length: bins }, (_, k) => (k === 0 || (nfft % 2 === 0 && k === bins - 1) ? 1 / fs : 2 / fs))
  return {
    frequencies: rfftfreq(nfft, 1 / fs) as Tensor,
    values: reshape(mul(spec.re, constant(fold, [1, bins])), [bins]),
    kind: 'density',
    fs,
  }
}

/** Options for {@link multitaper}. */
export interface MultitaperOptions {
  readonly fs?: number
  /** Time-halfbandwidth product (default 4). */
  readonly nw?: number
  /** Number of tapers (default 2 nw - 1). */
  readonly k?: number
  /** FFT length (default the signal length). */
  readonly nfft?: number
  /** Thomson's adaptive weighting (default false: the plain average of the eigenspectra). */
  readonly adaptive?: boolean
}

/**
 * Thomson's multitaper PSD: periodograms of the signal under K orthogonal DPSS tapers, averaged (or adaptively
 * weighted against broadband leakage), giving a low-variance, low-leakage estimate with 2K degrees of freedom.
 *
 * @param x - Real signal `[n]` (the mean is removed).
 * @param options - Sampling rate, NW, taper count, FFT length and adaptive weighting.
 * @returns The one-sided frequencies and PSD.
 * @concept multitaper
 * @example
 * ```ts
 * import { multitaper } from 'mdml-engine/signal'
 * multitaper(new Array(64).fill(0).map((_, i) => Math.sin(i)), { nw: 3 }).values.shape // => [33]
 * ```
 */
export function multitaper(x: TensorLike, options: MultitaperOptions = {}): Spectrum {
  const v = reshape(float(x), [1, -1])
  const n = v.shape[1] ?? 0
  const fs = options.fs ?? 1
  const nw = options.nw ?? 4
  const k = options.k ?? Math.max(1, Math.round(2 * nw) - 1)
  const nfft = options.nfft ?? n
  const tapers = dpss(n, nw, { kmax: k, norm: 2 })
  const xc = sub(v, mean(v, 1, true))
  const spec = rfft(mul(xc, tapers), nfft)
  const eig = add(mul(spec.re, spec.re), mul(spec.im, spec.im))
  const bins = eig.shape[1] ?? 0
  const fold = Array.from({ length: bins }, (_, j) => (j === 0 || (nfft % 2 === 0 && j === bins - 1) ? 1 / fs : 2 / fs))
  const scaled = mul(eig, constant(fold, [1, bins]))
  if (!(options.adaptive ?? false)) {
    return { frequencies: rfftfreq(nfft, 1 / fs) as Tensor, values: mean(scaled, 0), kind: 'density', fs }
  }
  // Thomson's adaptive weights (Percival and Walden 370a-b): d_k = sqrt(l_k) S / (l_k S + (1 - l_k) sigma^2), the
  // broadband bias of taper k, iterated from the average of the first two eigenspectra.
  const ratios = toFlat(dpssRatios(n, nw, k))
  const e = toFlat(primalOf(scaled))
  let sigma2 = 0
  for (const s of toFlat(primalOf(xc))) sigma2 += s * s
  sigma2 /= n
  const est = Float64Array.from({ length: bins }, (_, j) => ((e[j] ?? 0) + (e[(k > 1 ? bins : 0) + j] ?? 0)) / 2)
  const weights = new Float64Array(k * bins)
  for (let it = 0; it < 100; it++) {
    let change = 0
    for (let j = 0; j < bins; j++) {
      const sj = est[j] ?? 0
      const noise = sigma2 * (fold[j] ?? 0)
      let num = 0
      let den = 0
      for (let t = 0; t < k; t++) {
        const lam = ratios[t] ?? 1
        const d = (Math.sqrt(lam) * sj) / (lam * sj + (1 - lam) * noise)
        weights[t * bins + j] = d * d
        num += d * d * (e[t * bins + j] ?? 0)
        den += d * d
      }
      const next = den === 0 ? sj : num / den
      change = Math.max(change, Math.abs(next - sj) / Math.max(Math.abs(sj), 1e-300))
      est[j] = next
    }
    if (change < 1e-10) break
  }
  const w = constant(weights, [k, bins])
  return {
    frequencies: rfftfreq(nfft, 1 / fs) as Tensor,
    values: div(sum(mul(scaled, w), 0), sum(w, 0)),
    kind: 'density',
    fs,
  }
}

/**
 * Equivalent degrees of freedom of a Welch estimate (Percival and Walden, eq. 292b): 2K / (1 + 2 sum_m (1 - m / K)
 * rho(m)^2), where rho(m) is the correlation between segments m hops apart; 2K for non-overlapping segments.
 *
 * @param n - Signal length.
 * @param nperseg - Segment length.
 * @param noverlap - Overlap.
 * @param window - Window (default periodic Hann).
 * @returns The degrees of freedom.
 * @concept spectral-confidence
 * @example
 * ```ts
 * import { welchDof } from 'mdml-engine/signal'
 * welchDof(1024, 128, 0, 'boxcar') // => 16
 * ```
 */
export function welchDof(n: number, nperseg: number, noverlap: number, window: WindowSpec = 'hann'): number {
  const hop = nperseg - noverlap
  const k = Math.floor((n - noverlap) / hop)
  const w = toFlat(getWindow(window, nperseg))
  let s2 = 0
  for (const v of w) s2 += v * v
  let total = 0
  for (let m = 1; m < k; m++) {
    let c = 0
    for (let t = 0; t + m * hop < nperseg; t++) c += (w[t] ?? 0) * (w[t + m * hop] ?? 0)
    total += (1 - m / k) * (c / s2) ** 2
  }
  return (2 * k) / (1 + 2 * total)
}

/**
 * Chi-square confidence intervals for a PSD estimate with nu degrees of freedom: [nu P / chi2_(1 - a/2), nu P /
 * chi2_(a/2)].
 *
 * @param psd - The estimate.
 * @param dof - Degrees of freedom (2 for a raw periodogram bin, 2K for K averaged segments or tapers).
 * @param level - Confidence level (default 0.95).
 * @returns Lower and upper bounds, the shape of psd.
 * @concept spectral-confidence
 * @example
 * ```ts
 * import { spectralConfidenceInterval } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const { lower, upper } = spectralConfidenceInterval([1], 2)
 * [Math.round((toFlat(lower)[0] ?? 0) * 1e4) / 1e4, Math.round((toFlat(upper)[0] ?? 0) * 1e4) / 1e4] // => [0.2711, 39.4979]
 * ```
 */
export function spectralConfidenceInterval(
  psd: TensorLike,
  dof: number,
  level = 0.95,
): { readonly lower: Value; readonly upper: Value } {
  if (!(dof > 0)) throw new MdmlError('value', 'spectralConfidenceInterval: dof must be positive')
  if (!(level > 0 && level < 1)) throw new MdmlError('value', 'spectralConfidenceInterval: level must be in (0, 1)')
  const p = float(psd)
  const d = chi2(dof)
  const a = 1 - level
  const hi = toFlat(primalOf(d.ppf(1 - a / 2)))[0] ?? 1
  const lo = toFlat(primalOf(d.ppf(a / 2)))[0] ?? 1
  return { lower: mul(p, dof / hi), upper: mul(p, dof / lo) }
}
