/**
 * Data-driven decompositions of a signal into oscillatory components that sum back to it: empirical mode decomposition
 * (EMD, sifting out intrinsic mode functions) and its noise-assisted ensemble variants EEMD and CEEMDAN, variational
 * mode decomposition (VMD, band-limited modes by ADMM, step for step as vmdpy, also as a traced algorithm), and
 * singular spectrum analysis (SSA, the SVD of the lagged trajectory matrix).
 */

import { type Json, MdmlError } from '../base/index.ts'
import { fft, ifft } from '../numerics/fourier/index.ts'
import { cubicSpline } from '../numerics/interpolate/index.ts'
import { ddot } from '../numerics/linalg/blas.ts'
import { svd } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { constant, samples } from './_shared.ts'

// Adaptive decompositions of a signal into oscillatory modes: empirical mode decomposition and its noise-assisted
// ensembles (EEMD, CEEMDAN), variational mode decomposition (a traced ADMM, as vmdpy), and singular spectrum
// analysis.

/** Options for {@link emd}. */
export interface EmdOptions {
  /** Most IMFs to extract (default until the residue is monotone). */
  readonly maxImfs?: number
  /** Sifting stops when sum (h_old - h)^2 / sum h_old^2 falls below this (default 0.2, Huang's criterion). */
  readonly sdThreshold?: number
  /** Most sifting iterations per IMF (default 50). */
  readonly maxSifts?: number
}

/** Indices of strict local maxima (sign 1) or minima (sign -1), plateaus counted at their first sample. */
function extrema(x: Float64Array, sign: 1 | -1): number[] {
  const out: number[] = []
  for (let i = 1; i + 1 < x.length; i++) {
    const a = sign * ((x[i] ?? 0) - (x[i - 1] ?? 0))
    if (a <= 0) continue
    let j = i
    while (j + 1 < x.length && x[j + 1] === x[i]) j++
    if (j + 1 < x.length && sign * ((x[i] ?? 0) - (x[j + 1] ?? 0)) > 0) out.push(i)
    i = j
  }
  return out
}

/** A cubic-spline envelope through extrema, mirrored about both ends (two knots each side). */
function envelope(x: Float64Array, idx: readonly number[]): Float64Array {
  const n = x.length
  const left = idx.slice(0, 2).map((i) => -i)
  const right = idx.slice(-2).map((i) => 2 * (n - 1) - i)
  const t: number[] = []
  const y: number[] = []
  const push = (ti: number, i: number): void => {
    if (t.length === 0 || ti > (t[t.length - 1] ?? Number.NEGATIVE_INFINITY)) {
      t.push(ti)
      y.push(x[i] ?? 0)
    }
  }
  for (let k = left.length - 1; k >= 0; k--) push(left[k] ?? 0, idx[k] ?? 0)
  if (idx[0] !== 0) push(0, 0)
  for (const i of idx) push(i, i)
  if (idx[idx.length - 1] !== n - 1) push(n - 1, n - 1)
  for (let k = right.length - 1; k >= 0; k--) push(right[k] ?? 0, idx[idx.length - right.length + k] ?? 0)
  const spline = cubicSpline(t, y)
  return Float64Array.from(toFlat(spline.evaluate(Array.from({ length: n }, (_, i) => i))))
}

/** Whether a residue has too few extrema to sift. */
function isMonotone(x: Float64Array): boolean {
  return extrema(x, 1).length + extrema(x, -1).length < 3
}

/** One intrinsic mode function of x by sifting. */
function sift(x: Float64Array, sdThreshold: number, maxSifts: number): Float64Array {
  let h = Float64Array.from(x)
  for (let it = 0; it < maxSifts; it++) {
    const maxima = extrema(h, 1)
    const minima = extrema(h, -1)
    if (maxima.length < 2 || minima.length < 2) break
    const upper = envelope(h, maxima)
    const lower = envelope(h, minima)
    const next = Float64Array.from(h, (v, i) => v - 0.5 * ((upper[i] ?? 0) + (lower[i] ?? 0)))
    let num = 0
    let den = 0
    for (let i = 0; i < h.length; i++) {
      num += ((h[i] ?? 0) - (next[i] ?? 0)) ** 2
      den += (h[i] ?? 0) ** 2
    }
    h = next
    if (den === 0 || num / den < sdThreshold) break
  }
  return h
}

/** Intrinsic mode functions and the residue of a decomposition. */
export interface ModeDecomposition {
  /** IMFs `[k, n]`, highest frequency first. */
  readonly imfs: Tensor
  /** The residue `[n]` (the IMFs and the residue sum to the signal). */
  readonly residue: Tensor
}

function emdNumbers(x: Float64Array, options: EmdOptions): Float64Array[] {
  const imfs: Float64Array[] = []
  let r = Float64Array.from(x)
  const maxImfs = options.maxImfs ?? Number.POSITIVE_INFINITY
  while (imfs.length < maxImfs && !isMonotone(r)) {
    const imf = sift(r, options.sdThreshold ?? 0.2, options.maxSifts ?? 50)
    imfs.push(imf)
    r = Float64Array.from(r, (v, i) => v - (imf[i] ?? 0))
  }
  return imfs
}

function pack(imfs: readonly Float64Array[], x: Float64Array): ModeDecomposition {
  const n = x.length
  const flat = new Float64Array(imfs.length * n)
  const residue = Float64Array.from(x)
  for (const [k, m] of imfs.entries()) {
    flat.set(m, k * n)
    for (let i = 0; i < n; i++) residue[i] = (residue[i] ?? 0) - (m[i] ?? 0)
  }
  return { imfs: constant(flat, [imfs.length, n]), residue: constant(residue) }
}

/**
 * Empirical mode decomposition (Huang et al. 1998): repeatedly sift out the fastest oscillation by subtracting the
 * mean of the cubic-spline envelopes through the maxima and minima (mirrored at the ends), until Huang's standard
 * deviation criterion holds; stop when the residue is monotone. The IMFs and residue sum to the signal exactly.
 *
 * @param x - Signal.
 * @param options - IMF count, sifting threshold and limit.
 * @returns The IMFs and the residue.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { emd } from 'mdml-engine/signal'
 * const x = Array.from({ length: 200 }, (_, i) => Math.sin(i / 3) + 0.5 * Math.sin(i / 20) + i / 100)
 * emd(x).imfs.shape[0] >= 2 // => true
 * ```
 */
export function emd(x: TensorLike, options: EmdOptions = {}): ModeDecomposition {
  const s = samples(x, 'emd')
  return pack(emdNumbers(s, options), s)
}

/** Options for the noise-assisted decompositions. */
export interface EnsembleEmdOptions extends EmdOptions {
  /** Random key for the noise. */
  readonly key: Key
  /** Ensemble size (default 50). */
  readonly trials?: number
  /** Noise amplitude relative to the signal's standard deviation (default 0.2). */
  readonly noiseWidth?: number
}

function std(x: Float64Array): number {
  let m = 0
  for (const v of x) m += v / x.length
  let s = 0
  for (const v of x) s += (v - m) ** 2
  return Math.sqrt(s / x.length)
}

/**
 * Ensemble EMD (Wu and Huang 2009): the average IMFs of the signal plus independent white noise realisations, which
 * reduces mode mixing; the IMF count is fixed (by the first realisation) and the residue is the remainder.
 *
 * @param x - Signal.
 * @param options - Key, ensemble size, noise width and the EMD options.
 * @returns The averaged IMFs and the residue.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { eemd } from 'mdml-engine/signal'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 128 }, (_, i) => Math.sin(i / 2) + Math.sin(i / 15))
 * eemd(x, { key: key(1), trials: 8, maxImfs: 3 }).imfs.shape // => [3, 128]
 * ```
 */
export function eemd(x: TensorLike, options: EnsembleEmdOptions): ModeDecomposition {
  const s = samples(x, 'eemd')
  const n = s.length
  const trials = options.trials ?? 50
  const width = (options.noiseWidth ?? 0.2) * std(s)
  let count = options.maxImfs ?? -1
  let sum: Float64Array[] = []
  for (let t = 0; t < trials; t++) {
    const w = toFlat(normal(foldIn(options.key, t), [n]))
    const noisy = Float64Array.from(s, (v, i) => v + width * (w[i] ?? 0))
    const imfs = emdNumbers(noisy, count < 0 ? options : { ...options, maxImfs: count })
    if (count < 0) count = imfs.length
    if (sum.length === 0) sum = Array.from({ length: count }, () => new Float64Array(n))
    for (let k = 0; k < count; k++) {
      const m = imfs[k]
      const acc = sum[k] as Float64Array
      if (m !== undefined) for (let i = 0; i < n; i++) acc[i] = (acc[i] ?? 0) + (m[i] ?? 0) / trials
    }
  }
  return pack(sum, s)
}

/**
 * Complete ensemble EMD with adaptive noise (CEEMDAN, Torres et al. 2011): each mode is the ensemble average of the
 * first EMD mode of the current residue plus the matching EMD mode of white noise scaled by beta_k = noiseWidth
 * std(residue), so the decomposition is complete (the modes and residue sum to the signal) with no residual noise.
 *
 * @param x - Signal.
 * @param options - Key, ensemble size, noise width and the EMD options.
 * @returns The modes and the residue.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { ceemdan } from 'mdml-engine/signal'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 128 }, (_, i) => Math.sin(i / 2) + Math.sin(i / 15))
 * ceemdan(x, { key: key(2), trials: 8, maxImfs: 3 }).imfs.shape[0] <= 3 // => true
 * ```
 */
export function ceemdan(x: TensorLike, options: EnsembleEmdOptions): ModeDecomposition {
  const s = samples(x, 'ceemdan')
  const n = s.length
  const trials = options.trials ?? 50
  const eps = options.noiseWidth ?? 0.2
  const maxImfs = options.maxImfs ?? Number.POSITIVE_INFINITY
  // Unit-variance white noise realisations and their EMD modes.
  const noise: Float64Array[] = []
  const noiseModes: Float64Array[][] = []
  for (let t = 0; t < trials; t++) {
    const w = Float64Array.from(toFlat(normal(foldIn(options.key, t), [n])))
    const sd = std(w) || 1
    const unit = Float64Array.from(w, (v) => v / sd)
    noise.push(unit)
    noiseModes.push(emdNumbers(unit, options))
  }
  const firstMode = (r: Float64Array): Float64Array =>
    emdNumbers(r, { ...options, maxImfs: 1 })[0] ?? new Float64Array(n)
  const imfs: Float64Array[] = []
  let residue = Float64Array.from(s)
  while (imfs.length < maxImfs && !isMonotone(residue)) {
    const k = imfs.length
    const beta = eps * std(residue)
    const mode = new Float64Array(n)
    for (const [t, modes] of noiseModes.entries()) {
      // IMF_1 from x + beta w; IMF_(k+1) from r_k + beta E_k(w), the k-th EMD mode of the noise.
      const e = k === 0 ? noise[t] : modes[k - 1]
      const perturbed = e === undefined ? residue : Float64Array.from(residue, (v, i) => v + beta * (e[i] ?? 0))
      const local = firstMode(perturbed)
      for (let i = 0; i < n; i++) mode[i] = (mode[i] ?? 0) + (local[i] ?? 0) / trials
    }
    imfs.push(mode)
    residue = Float64Array.from(residue, (v, i) => v - (mode[i] ?? 0))
  }
  return pack(imfs, s)
}

/** Options for {@link vmd}. */
export interface VmdOptions {
  /** Number of modes K. */
  readonly k: number
  /** Bandwidth penalty alpha (default 2000). */
  readonly alpha?: number
  /** Dual ascent step tau (default 0: noise slack). */
  readonly tau?: number
  /** Keep the first mode at zero frequency (default false). */
  readonly dc?: boolean
  /** Centre-frequency initialisation: 'zero' or 'uniform' (default). */
  readonly init?: 'zero' | 'uniform'
  /** Convergence tolerance on the update (default 1e-7). */
  readonly tol?: number
  /** Most iterations (default 500, as vmdpy). */
  readonly maxIter?: number
}

/** The ADMM state of VMD: mode spectra on the mirrored half-plane, centre frequencies and the multiplier. */
export interface VmdState {
  readonly uRe: Float64Array[]
  readonly uIm: Float64Array[]
  readonly omega: Float64Array
  readonly lambdaRe: Float64Array
  readonly lambdaIm: Float64Array
  readonly sumRe: Float64Array
  readonly sumIm: Float64Array
  readonly diff: number
  /** The previous iterate's spectra and frequencies (what Zosso's code and vmdpy return). */
  readonly prevRe: Float64Array[]
  readonly prevIm: Float64Array[]
  readonly prevOmega: Float64Array
}

interface VmdSetup {
  readonly t: number
  readonly freqs: Float64Array
  readonly fRe: Float64Array
  readonly fIm: Float64Array
  readonly length: number
}

function vmdSetup(x: Float64Array): VmdSetup {
  const f = x.length % 2 === 1 ? x.slice(0, -1) : x
  const half = f.length / 2
  const mirr = new Float64Array(2 * f.length)
  for (let i = 0; i < half; i++) mirr[i] = f[half - 1 - i] ?? 0
  mirr.set(f, half)
  for (let i = 0; i < half; i++) mirr[half + f.length + i] = f[f.length - 1 - i] ?? 0
  const t = mirr.length
  const spec = fft(constant(mirr))
  const re = toFlat(spec.re)
  const im = toFlat(spec.im)
  // fftshift, then keep the positive half.
  const fRe = new Float64Array(t)
  const fIm = new Float64Array(t)
  for (let i = 0; i < t; i++) {
    const src = (i + Math.floor(t / 2)) % t
    if (i >= Math.floor(t / 2)) {
      fRe[i] = re[src] ?? 0
      fIm[i] = im[src] ?? 0
    }
  }
  const freqs = Float64Array.from({ length: t }, (_, i) => (i + 1) / t - 0.5 - 1 / t)
  return { t, freqs, fRe, fIm, length: f.length }
}

/**
 * Variational mode decomposition as a traced algorithm (Dragomiretskiy and Zosso 2014, step for step vmdpy): ADMM
 * updates of K mode spectra (Wiener filters around each centre frequency), their centre frequencies (power-weighted
 * means) and the dual multiplier, on the mirror-extended signal.
 *
 * @param x - Signal.
 * @param options - Modes, penalty, step, DC mode, initialisation, tolerance and iteration limit.
 * @returns The algorithm; its state holds the mode spectra and centre frequencies.
 * @concept variational-mode-decomposition
 * @example
 * ```ts
 * import { vmdAlgorithm } from 'mdml-engine/signal'
 * vmdAlgorithm([1, 2, 3, 4], { k: 2 }).name // => 'vmd'
 * ```
 */
export function vmdAlgorithm(x: TensorLike, options: VmdOptions): Algorithm<VmdState> {
  const s = samples(x, 'vmd')
  const setup = vmdSetup(s)
  const { t, freqs, fRe, fIm } = setup
  const k = options.k
  if (!Number.isInteger(k) || k < 1) throw new MdmlError('value', 'vmd: k must be a positive integer')
  const alpha = options.alpha ?? 2000
  const tau = options.tau ?? 0
  const dc = options.dc ?? false
  const half = Math.floor(t / 2)
  return {
    name: 'vmd',
    config: { k, alpha, tau, dc, init: options.init ?? 'uniform' } as Json,
    init: () => ({
      uRe: Array.from({ length: k }, () => new Float64Array(t)),
      uIm: Array.from({ length: k }, () => new Float64Array(t)),
      omega: Float64Array.from({ length: k }, (_, i) =>
        (options.init ?? 'uniform') === 'uniform' && !(dc && i === 0) ? (0.5 / k) * i : 0,
      ),
      lambdaRe: new Float64Array(t),
      lambdaIm: new Float64Array(t),
      sumRe: new Float64Array(t),
      sumIm: new Float64Array(t),
      diff: Number.POSITIVE_INFINITY,
      prevRe: Array.from({ length: k }, () => new Float64Array(t)),
      prevIm: Array.from({ length: k }, () => new Float64Array(t)),
      prevOmega: new Float64Array(k),
    }),
    step: (st) => {
      const uRe = st.uRe.map((u) => Float64Array.from(u))
      const uIm = st.uIm.map((u) => Float64Array.from(u))
      const omega = Float64Array.from(st.omega)
      const sumRe = Float64Array.from(st.sumRe)
      const sumIm = Float64Array.from(st.sumIm)
      for (let m = 0; m < k; m++) {
        const prevLast = m === 0 ? (st.uRe[k - 1] as Float64Array) : (uRe[m - 1] as Float64Array)
        const prevLastIm = m === 0 ? (st.uIm[k - 1] as Float64Array) : (uIm[m - 1] as Float64Array)
        const old = st.uRe[m] as Float64Array
        const oldIm = st.uIm[m] as Float64Array
        const ur = uRe[m] as Float64Array
        const ui = uIm[m] as Float64Array
        let num = 0
        let den = 0
        for (let i = 0; i < t; i++) {
          sumRe[i] = (prevLast[i] ?? 0) + (sumRe[i] ?? 0) - (old[i] ?? 0)
          sumIm[i] = (prevLastIm[i] ?? 0) + (sumIm[i] ?? 0) - (oldIm[i] ?? 0)
          const d = 1 + alpha * ((freqs[i] ?? 0) - (st.omega[m] ?? 0)) ** 2
          ur[i] = ((fRe[i] ?? 0) - (sumRe[i] ?? 0) - (st.lambdaRe[i] ?? 0) / 2) / d
          ui[i] = ((fIm[i] ?? 0) - (sumIm[i] ?? 0) - (st.lambdaIm[i] ?? 0) / 2) / d
          if (i >= half) {
            const p = (ur[i] ?? 0) ** 2 + (ui[i] ?? 0) ** 2
            num += (freqs[i] ?? 0) * p
            den += p
          }
        }
        if (!(dc && m === 0)) omega[m] = num / den
      }
      const lambdaRe = Float64Array.from(st.lambdaRe)
      const lambdaIm = Float64Array.from(st.lambdaIm)
      let diff = Number.EPSILON
      for (let i = 0; i < t; i++) {
        let sr = 0
        let si = 0
        for (let m = 0; m < k; m++) {
          sr += uRe[m]?.[i] ?? 0
          si += uIm[m]?.[i] ?? 0
        }
        lambdaRe[i] = (lambdaRe[i] ?? 0) + tau * (sr - (fRe[i] ?? 0))
        lambdaIm[i] = (lambdaIm[i] ?? 0) + tau * (si - (fIm[i] ?? 0))
      }
      for (let m = 0; m < k; m++) {
        const ur = uRe[m] as Float64Array
        const ui = uIm[m] as Float64Array
        const dr = Float64Array.from(ur, (v, i) => v - (st.uRe[m]?.[i] ?? 0))
        const di = Float64Array.from(ui, (v, i) => v - (st.uIm[m]?.[i] ?? 0))
        diff += (ddot(dr, dr) + ddot(di, di)) / t
      }
      return {
        uRe,
        uIm,
        omega,
        lambdaRe,
        lambdaIm,
        sumRe,
        sumIm,
        diff: Math.abs(diff),
        prevRe: st.uRe,
        prevIm: st.uIm,
        prevOmega: st.omega,
      }
    },
    done: (st) => (st.diff <= (options.tol ?? 1e-7) ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { update: st.diff },
      tensors: { omega: { role: 'state', value: st.omega } },
    }),
  }
}

/** The modes of a variational mode decomposition and their centre frequencies (cycles per sample). */
export interface VmdResult {
  readonly modes: Tensor
  readonly omega: Tensor
  readonly iterations: number
}

/**
 * Variational mode decomposition (vmdpy.VMD): K band-limited modes whose sum reconstructs the signal, by the traced
 * ADMM of {@link vmdAlgorithm}, transformed back and cropped from the mirror extension. As vmdpy and Zosso's code, the
 * modes and frequencies returned are those of the iterate before the converged one.
 *
 * @param x - Signal (an odd length drops the last sample, as vmdpy).
 * @param options - VMD options and tracing.
 * @returns The modes `[K, n]`, final centre frequencies and iterations.
 * @concept variational-mode-decomposition
 * @example
 * ```ts
 * import { vmd } from 'mdml-engine/signal'
 * const x = Array.from({ length: 200 }, (_, i) => Math.cos(2 * Math.PI * 0.05 * i) + Math.cos(2 * Math.PI * 0.25 * i))
 * const r = vmd(x, { k: 2 })
 * Array.from(r.omega.data).map((w) => Math.round(w * 100) / 100) // => [0.05, 0.25]
 * ```
 */
export function vmd(
  x: TensorLike,
  options: VmdOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {} as VmdOptions,
): VmdResult {
  const s = samples(x, 'vmd')
  const setup = vmdSetup(s)
  const { t } = setup
  const maxIter = options.maxIter ?? 500
  const result = runAlgorithm(vmdAlgorithm(s, options), {
    maxSteps: maxIter - 1,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  const st = result.state
  const k = options.k
  const half = Math.floor(t / 2)
  const out = new Float64Array(k * setup.length)
  for (let m = 0; m < k; m++) {
    const ur = st.prevRe[m] as Float64Array
    const ui = st.prevIm[m] as Float64Array
    // Hermitian completion of the half spectrum (vmdpy's index pattern), then ifftshift and invert.
    const hr = new Float64Array(t)
    const hi = new Float64Array(t)
    for (let i = half; i < t; i++) {
      hr[i] = ur[i] ?? 0
      hi[i] = ui[i] ?? 0
    }
    for (let j = 0; j < t - half; j++) {
      const dst = half - j
      hr[dst] = ur[half + j] ?? 0
      hi[dst] = -(ui[half + j] ?? 0)
    }
    hr[0] = hr[t - 1] ?? 0
    hi[0] = -(hi[t - 1] ?? 0)
    const sr = new Float64Array(t)
    const si = new Float64Array(t)
    for (let i = 0; i < t; i++) {
      const src = (i + half) % t
      sr[i] = hr[src] ?? 0
      si[i] = hi[src] ?? 0
    }
    const time = toFlat(ifft({ re: constant(sr), im: constant(si) }).re)
    out.set(time.slice(Math.floor(t / 4), Math.floor((3 * t) / 4)), m * setup.length)
  }
  return { modes: constant(out, [k, setup.length]), omega: constant(st.prevOmega), iterations: result.steps }
}

/** The components of a singular spectrum analysis. */
export interface SsaResult {
  /** Elementary reconstructed components `[L, n]`, by decreasing singular value; they sum to the signal. */
  readonly components: Tensor
  /** Singular values of the trajectory matrix. */
  readonly singularValues: Tensor
}

/**
 * Singular spectrum analysis: the SVD of the L-lagged trajectory (Hankel) matrix and the diagonal averaging of each
 * rank-one term into an elementary component; the components sum to the signal, and grouping them separates trend,
 * oscillations and noise.
 *
 * @param x - Signal `[n]`.
 * @param window - Window length L (2 <= L <= n - 1; default n / 2).
 * @returns The elementary components and singular values.
 * @concept singular-spectrum-analysis
 * @example
 * ```ts
 * import { ssa } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const r = ssa([1, 2, 3, 4, 5, 6], 3)
 * Math.round(toFlat(r.components).slice(0, 6).reduce((a, v, i) => a + v + (toFlat(r.components)[6 + i] ?? 0) + (toFlat(r.components)[12 + i] ?? 0), 0) * 1e9) / 1e9 // => 21
 * ```
 */
export function ssa(x: TensorLike, window?: number): SsaResult {
  const s = samples(x, 'ssa')
  const n = s.length
  const l = window ?? Math.floor(n / 2)
  if (!(l >= 2 && l <= n - 1)) throw new MdmlError('value', 'ssa: the window must satisfy 2 <= L <= n - 1')
  const k = n - l + 1
  const traj = new Float64Array(l * k)
  for (let i = 0; i < l; i++) for (let j = 0; j < k; j++) traj[i * k + j] = s[i + j] ?? 0
  const { u, s: sv, vh } = svd(constant(traj, [l, k]))
  const uu = toFlat(primalOf(u))
  const vv = toFlat(primalOf(vh))
  const sing = toFlat(primalOf(sv))
  const r = sing.length
  const out = new Float64Array(r * n)
  for (let c = 0; c < r; c++) {
    const sum = new Float64Array(n)
    const cnt = new Float64Array(n)
    for (let i = 0; i < l; i++) {
      for (let j = 0; j < k; j++) {
        sum[i + j] = (sum[i + j] ?? 0) + (sing[c] ?? 0) * (uu[i * r + c] ?? 0) * (vv[c * k + j] ?? 0)
        cnt[i + j] = (cnt[i + j] ?? 0) + 1
      }
    }
    for (let t = 0; t < n; t++) out[c * n + t] = (sum[t] ?? 0) / (cnt[t] ?? 1)
  }
  return { components: constant(out, [r, n]), singularValues: constant(sing) }
}
