/**
 * The Lomb-Scargle periodogram, the least-squares fit of a sinusoid at each frequency, which finds periodicities in
 * unevenly sampled data where the FFT cannot be used: the power and amplitude spectra (scipy.signal.lombscargle), and
 * false-alarm probabilities and levels that say how significant a peak is under Gaussian noise.
 */

import { MdmlError } from '../base/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import {
  add,
  type ComplexValue,
  div,
  item,
  matmul,
  mul,
  reshape,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import { constant, float, samples } from './_shared.ts'

// The Lomb-Scargle periodogram of unevenly sampled data (scipy.signal.lombscargle, with weights and a floating mean)
// and its false-alarm probabilities (the single-frequency, naive, Davies and Baluev bounds, as astropy).

/** Options for {@link lombScargle}. */
export interface LombScargleOptions {
  /** 'power' (default: scaled by n / 4), 'normalize' (in [0, 1], the fraction of variance explained). */
  readonly normalize?: 'power' | 'normalize'
  /** Non-negative weights of the samples (default equal). */
  readonly weights?: TensorLike
  /** Fit a mean at every frequency (the generalised Lomb-Scargle periodogram). Default false. */
  readonly floatingMean?: boolean
}

interface Design {
  readonly cos: Value
  readonly sin: Value
  readonly w: Float64Array
  readonly cc: Float64Array
  readonly ss: Float64Array
  readonly c: Float64Array
  readonly s: Float64Array
  readonly tau: Float64Array
}

/** The time-only parts of the periodogram: tau and the shifted cosines and sines (scipy's algorithm). */
function design(t: Float64Array, freqs: Float64Array, weights: Float64Array, floatingMean: boolean): Design {
  const n = t.length
  const m = freqs.length
  const cosT = new Float64Array(n * m)
  const sinT = new Float64Array(n * m)
  const cc = new Float64Array(m)
  const ss = new Float64Array(m)
  const cMean = new Float64Array(m)
  const sMean = new Float64Array(m)
  const tau = new Float64Array(m)
  const epsneg = 2 ** -53
  for (let j = 0; j < m; j++) {
    const f = freqs[j] ?? 0
    let c2 = 0
    let cs = 0
    let c1 = 0
    let s1 = 0
    for (let i = 0; i < n; i++) {
      const a = f * (t[i] ?? 0)
      const wi = weights[i] ?? 0
      c2 += wi * Math.cos(a) ** 2
      cs += wi * Math.cos(a) * Math.sin(a)
      c1 += wi * Math.cos(a)
      s1 += wi * Math.sin(a)
    }
    let s2 = 1 - c2
    if (floatingMean) {
      c2 -= c1 * c1
      s2 -= s1 * s1
      cs -= c1 * s1
    }
    const tj = 0.5 * Math.atan2(2 * cs, c2 - s2)
    tau[j] = tj
    let cct = 0
    let ct = 0
    let st = 0
    for (let i = 0; i < n; i++) {
      const a = f * (t[i] ?? 0) - tj
      const wi = weights[i] ?? 0
      cosT[i * m + j] = Math.cos(a)
      sinT[i * m + j] = Math.sin(a)
      cct += wi * Math.cos(a) ** 2
      ct += wi * Math.cos(a)
      st += wi * Math.sin(a)
    }
    let sst = 1 - cct
    if (floatingMean) {
      cct -= ct * ct
      sst -= st * st
    }
    cc[j] = Math.max(cct, epsneg)
    ss[j] = Math.max(sst, epsneg)
    cMean[j] = ct
    sMean[j] = st
  }
  return { cos: constant(cosT, [n, m]), sin: constant(sinT, [n, m]), w: weights, cc, ss, c: cMean, s: sMean, tau }
}

function inputs(
  x: TensorLike,
  y: TensorLike,
  freqs: TensorLike,
  options: LombScargleOptions,
): { t: Float64Array; yv: Value; f: Float64Array; w: Float64Array } {
  const t = samples(x, 'lombScargle')
  const f = samples(freqs, 'lombScargle')
  const yv = reshape(float(y), [-1])
  if ((yv.shape[0] ?? 0) !== t.length) throw new MdmlError('shape', 'lombScargle: x and y must have the same length')
  const raw =
    options.weights === undefined ? new Float64Array(t.length).fill(1) : samples(options.weights, 'lombScargle')
  if (raw.length !== t.length) throw new MdmlError('shape', 'lombScargle: one weight per sample')
  let total = 0
  for (const v of raw) {
    if (v < 0) throw new MdmlError('value', 'lombScargle: weights must be non-negative')
    total += v
  }
  if (!(total > 0)) throw new MdmlError('value', 'lombScargle: the weights must sum to a positive value')
  return { t, yv, f, w: Float64Array.from(raw, (v) => v / total) }
}

/** The weighted projections YC, YS (mean-corrected for a floating mean) and the weighted mean of y. */
function projections(d: Design, yv: Value, floatingMean: boolean): { yc: Value; ys: Value; ym: Value } {
  const wy = mul(yv, constant(d.w))
  const ym = sum(wy)
  let yc = matmul(wy, d.cos)
  let ys = matmul(wy, d.sin)
  if (floatingMean) {
    yc = sub(yc, mul(ym, constant(d.c)))
    ys = sub(ys, mul(ym, constant(d.s)))
  }
  return { yc, ys, ym }
}

/**
 * The Lomb-Scargle periodogram of unevenly sampled data (scipy.signal.lombscargle): at each angular frequency, the
 * power of the least-squares sinusoid fit with the phase offset tau that decouples the sine and cosine terms;
 * optionally weighted and with a floating mean. Differentiable in y.
 *
 * @param x - Sample times.
 * @param y - Samples.
 * @param freqs - Angular frequencies (rad per unit time).
 * @param options - Normalisation, weights and floating mean.
 * @returns The periodogram at each frequency.
 * @throws MdmlError with code `value` for invalid weights, `shape` for mismatched lengths.
 * @concept lomb-scargle
 * @example
 * ```ts
 * import { lombScargle } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const t = [0, 0.7, 1.3, 2.2, 3.1, 3.8, 4.6, 5.9]
 * const y = t.map((v) => Math.sin(2 * v))
 * const p = toFlat(lombScargle(t, y, [1, 2, 3], { normalize: 'normalize' }))
 * p.indexOf(Math.max(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0)) // => 1
 * ```
 */
export function lombScargle(x: TensorLike, y: TensorLike, freqs: TensorLike, options: LombScargleOptions = {}): Value {
  const { t, yv, f, w } = inputs(x, y, freqs, options)
  const floating = options.floatingMean ?? false
  const d = design(t, f, w, floating)
  const { yc, ys, ym } = projections(d, yv, floating)
  const p = mul(add(div(mul(yc, yc), constant(d.cc)), div(mul(ys, ys), constant(d.ss))), 2)
  if ((options.normalize ?? 'power') === 'power') return mul(p, t.length / 4)
  let yy = sum(mul(mul(yv, yv), constant(w)))
  if (floating) yy = sub(yy, mul(ym, ym))
  return mul(p, div(0.5, yy))
}

/**
 * The Lomb-Scargle amplitudes (scipy.signal.lombscargle with normalize='amplitude'): the complex amplitude (a + i b)
 * exp(i tau) of the fitted sinusoid at each frequency, whose magnitude is the sinusoid's amplitude.
 *
 * @param x - Sample times.
 * @param y - Samples.
 * @param freqs - Angular frequencies.
 * @param options - Weights and floating mean.
 * @returns The complex amplitudes.
 * @concept lomb-scargle
 * @example
 * ```ts
 * import { lombScargleAmplitude } from 'mdml-engine/signal'
 * import { absComplex, toFlat } from 'mdml-engine/tensor'
 * const t = [0, 0.7, 1.3, 2.2, 3.1, 3.8, 4.6, 5.9, 6.4, 7.7]
 * Math.round((toFlat(absComplex(lombScargleAmplitude(t, t.map((v) => 3 * Math.cos(2 * v)), [2])))[0] ?? 0) * 1e9) / 1e9 // => 3
 * ```
 */
export function lombScargleAmplitude(
  x: TensorLike,
  y: TensorLike,
  freqs: TensorLike,
  options: Omit<LombScargleOptions, 'normalize'> = {},
): ComplexValue {
  const { t, yv, f, w } = inputs(x, y, freqs, options)
  const floating = options.floatingMean ?? false
  const d = design(t, f, w, floating)
  const { yc, ys } = projections(d, yv, floating)
  const a = div(yc, constant(d.cc))
  const b = div(ys, constant(d.ss))
  const c = constant(Float64Array.from(d.tau, Math.cos))
  const s = constant(Float64Array.from(d.tau, Math.sin))
  return { re: sub(mul(a, c), mul(b, s)), im: add(mul(a, s), mul(b, c)) }
}

/** How a false-alarm probability is bounded. */
export type FalseAlarmMethod = 'single' | 'naive' | 'davies' | 'baluev'

/** The sampling a false-alarm probability depends on. */
export interface FalseAlarmOptions {
  /** Sample times. */
  readonly t: TensorLike
  /** The highest frequency searched, in cycles per unit time. */
  readonly fmax: number
  /** The bound (default 'baluev'). */
  readonly method?: FalseAlarmMethod
}

function gammaRatio(n: number): number {
  return Math.sqrt(2 / n) * Math.exp(item(lgamma(n / 2)) - item(lgamma((n - 1) / 2)))
}

function fapOf(z: number, t: Float64Array, fmax: number, method: FalseAlarmMethod): number {
  const n = t.length
  const single = (1 - z) ** (0.5 * (n - 3))
  if (method === 'single') return single
  if (method === 'naive') {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (const v of t) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    return -Math.expm1(fmax * (hi - lo) * Math.log1p(-single))
  }
  let m1 = 0
  let m2 = 0
  for (const v of t) {
    m1 += v / n
    m2 += (v * v) / n
  }
  const w = fmax * Math.sqrt(4 * Math.PI * (m2 - m1 * m1))
  const tau = gammaRatio(n - 1) * w * (1 - z) ** (0.5 * (n - 4)) * Math.sqrt(0.5 * (n - 1) * z)
  if (method === 'davies') return single + tau
  return -Math.expm1(-tau) + single * Math.exp(-tau)
}

/**
 * The false-alarm probability of a peak of normalised Lomb-Scargle power z (in [0, 1], `normalize: 'normalize'` with
 * a floating mean) for Gaussian noise, by the single-frequency formula (1 - z)^((n - 3) / 2), the naive independent
 * frequencies bound, Davies' upper bound or Baluev's extreme-value approximation (as astropy).
 *
 * @param z - Normalised peak powers.
 * @param options - Sample times, maximum frequency (cycles per unit time) and method.
 * @returns The probabilities that noise alone reaches each power.
 * @concept lomb-scargle
 * @example
 * ```ts
 * import { falseAlarmProbability } from 'mdml-engine/signal'
 * const t = Array.from({ length: 50 }, (_, i) => i + 0.3 * Math.sin(i))
 * falseAlarmProbability([0.5], { t, fmax: 0.5 })[0] < 0.01 // => true
 * ```
 */
export function falseAlarmProbability(z: readonly number[], options: FalseAlarmOptions): number[] {
  const t = samples(options.t, 'falseAlarmProbability')
  if (t.length < 4) throw new MdmlError('value', 'falseAlarmProbability: at least four samples are needed')
  return z.map((v) => fapOf(v, t, options.fmax, options.method ?? 'baluev'))
}

/**
 * The normalised Lomb-Scargle power with a given false-alarm probability (the inverse of
 * {@link falseAlarmProbability}, by bisection).
 *
 * @param p - False-alarm probabilities in (0, 1).
 * @param options - Sample times, maximum frequency and method.
 * @returns The power levels.
 * @concept lomb-scargle
 * @example
 * ```ts
 * import { falseAlarmLevel, falseAlarmProbability } from 'mdml-engine/signal'
 * const t = Array.from({ length: 40 }, (_, i) => i)
 * const z = falseAlarmLevel([0.01], { t, fmax: 0.5 })
 * Math.round((falseAlarmProbability(z, { t, fmax: 0.5 })[0] ?? 0) * 1e8) / 1e8 // => 0.01
 * ```
 */
export function falseAlarmLevel(p: readonly number[], options: FalseAlarmOptions): number[] {
  const t = samples(options.t, 'falseAlarmLevel')
  const method = options.method ?? 'baluev'
  return p.map((target) => {
    if (!(target > 0 && target < 1)) throw new MdmlError('value', 'falseAlarmLevel: probabilities must lie in (0, 1)')
    let lo = 0
    let hi = 1
    for (let it = 0; it < 200; it++) {
      const mid = 0.5 * (lo + hi)
      if (fapOf(mid, t, options.fmax, method) > target) lo = mid
      else hi = mid
      if (hi - lo < 1e-15) break
    }
    return 0.5 * (lo + hi)
  })
}
