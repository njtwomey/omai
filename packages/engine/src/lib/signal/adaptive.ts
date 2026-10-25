/**
 * Adaptive FIR filters whose weights update sample by sample to track a desired signal: least mean squares (LMS), its
 * normalised form (NLMS) and recursive least squares (RLS), as in padasip. The traced `adaptiveFilterAlgorithm`,
 * `adaptiveFilter` to run one over a whole record, and Widrow's adaptive noise cancellation, which predicts the noise
 * from a correlated reference and keeps the error as the cleaned signal.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { ddot } from '../numerics/linalg/blas.ts'
import { primalOf, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { constant, float, samples } from './_shared.ts'

// Adaptive FIR filters as padasip: least mean squares, normalised LMS and recursive least squares, one sample per
// step of a traced algorithm (the tap weights are a tensor channel), and adaptive noise cancellation.

/** The adaptation rule. */
export type AdaptiveMethod = 'lms' | 'nlms' | 'rls'

/** Options of the adaptive filters. */
export interface AdaptiveOptions {
  /** Number of taps n (the width of the regressors; required with a signal input). */
  readonly taps?: number
  /** Step size (LMS default 0.01, NLMS default 1) or forgetting factor (RLS, default 0.99). */
  readonly mu?: number
  /** Regulariser: NLMS adds it to the input power (default 1e-3); RLS starts from P = I / eps (default 0.1). */
  readonly eps?: number
  /** Initial weights (default zeros). */
  readonly w0?: TensorLike
}

/** The running state: weights, the inverse correlation (RLS), the sample index and the outputs so far. */
export interface AdaptiveState {
  readonly w: Float64Array
  readonly p: Float64Array | null
  readonly k: number
  readonly y: Float64Array
  readonly e: Float64Array
}

/** Regressor rows from a matrix `[N, n]`, or a tapped delay line [x_k, x_(k-1), ..., x_(k-n+1)] of a signal. */
function regressors(x: TensorLike, n: number | undefined): { rows: Float64Array; count: number; width: number } {
  const v = float(x)
  if (v.shape.length === 2) {
    const [count = 0, width = 0] = v.shape
    return { rows: Float64Array.from(primalOf(v).data), count, width }
  }
  const s = samples(v, 'adaptive filter')
  if (n === undefined || !Number.isInteger(n) || n < 1)
    throw new MdmlError('value', 'adaptive filter: give the number of taps')
  const rows = new Float64Array(s.length * n)
  for (let k = 0; k < s.length; k++) for (let j = 0; j < n; j++) rows[k * n + j] = k - j >= 0 ? (s[k - j] ?? 0) : 0
  return { rows, count: s.length, width: n }
}

/**
 * An adaptive FIR filter as a traced algorithm (padasip FilterLMS, FilterNLMS, FilterRLS): at step k, y = w . x_k,
 * e = d_k - y, and the weights move by mu e x_k (LMS), mu e x_k / (eps + x_k . x_k) (NLMS), or by the RLS gain
 * with forgetting factor mu.
 *
 * @param method - 'lms', 'nlms' or 'rls'.
 * @param x - Regressors `[N, n]`, or a signal (with `taps`) turned into a tapped delay line.
 * @param d - Desired signal `[N]`.
 * @param options - Taps, step or forgetting factor, regulariser and initial weights.
 * @returns The algorithm (one step per sample).
 * @concept adaptive-filters
 * @example
 * ```ts
 * import { adaptiveFilterAlgorithm } from 'mdml-engine/signal'
 * adaptiveFilterAlgorithm('lms', [1, 2, 3], [1, 2, 3], { taps: 2 }).name // => 'lms'
 * ```
 */
export function adaptiveFilterAlgorithm(
  method: AdaptiveMethod,
  x: TensorLike,
  d: TensorLike,
  options: AdaptiveOptions = {},
): Algorithm<AdaptiveState> {
  const { rows, count, width } = regressors(x, options.taps)
  const dv = samples(d, 'adaptive filter')
  if (dv.length !== count) throw new MdmlError('shape', 'adaptive filter: d must have one value per regressor')
  const mu = options.mu ?? (method === 'lms' ? 0.01 : method === 'nlms' ? 1 : 0.99)
  const eps = options.eps ?? (method === 'rls' ? 0.1 : 1e-3)
  const w0 = options.w0 === undefined ? new Float64Array(width) : samples(options.w0, 'adaptive filter')
  if (w0.length !== width) throw new MdmlError('shape', 'adaptive filter: w0 must have one weight per tap')
  return {
    name: method,
    config: { method, mu, eps, taps: width } as Json,
    init: () => {
      let p: Float64Array | null = null
      if (method === 'rls') {
        p = new Float64Array(width * width)
        for (let i = 0; i < width; i++) p[i * width + i] = 1 / eps
      }
      return { w: Float64Array.from(w0), p, k: 0, y: new Float64Array(count), e: new Float64Array(count) }
    },
    step: (st) => {
      const k = st.k
      const xk = rows.subarray(k * width, (k + 1) * width)
      const yk = ddot(st.w, xk)
      const ek = (dv[k] ?? 0) - yk
      // The output buffers are filled in place (one entry per step) rather than copied at every sample.
      const y = st.y
      const e = st.e
      y[k] = yk
      e[k] = ek
      const w = Float64Array.from(st.w)
      let p = st.p
      if (method === 'lms') for (let i = 0; i < width; i++) w[i] = (w[i] ?? 0) + mu * ek * (xk[i] ?? 0)
      else if (method === 'nlms') {
        const g = mu / (eps + ddot(xk, xk))
        for (let i = 0; i < width; i++) w[i] = (w[i] ?? 0) + g * ek * (xk[i] ?? 0)
      } else if (p !== null) {
        // padasip: R1 = P x x^T P, R2 = mu + x^T P x, P <- (P - R1 / R2) / mu, dw = P x e.
        const px = new Float64Array(width)
        for (let i = 0; i < width; i++) px[i] = ddot(p.subarray(i * width, (i + 1) * width), xk)
        const xp = new Float64Array(width)
        for (let j = 0; j < width; j++) {
          let s = 0
          for (let i = 0; i < width; i++) s += (xk[i] ?? 0) * (p[i * width + j] ?? 0)
          xp[j] = s
        }
        const r2 = mu + ddot(xk, px)
        const next = new Float64Array(width * width)
        for (let i = 0; i < width; i++)
          for (let j = 0; j < width; j++)
            next[i * width + j] = ((p[i * width + j] ?? 0) - ((px[i] ?? 0) * (xp[j] ?? 0)) / r2) / mu
        p = next
        for (let i = 0; i < width; i++) w[i] = (w[i] ?? 0) + ddot(p.subarray(i * width, (i + 1) * width), xk) * ek
      }
      return { w, p, k: k + 1, y, e }
    },
    done: (st) => (st.k >= count ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { error: st.e[st.k - 1] ?? 0 },
      tensors: { weights: { role: 'param', value: st.w } },
    }),
  }
}

/** The outputs of an adaptive filter. */
export interface AdaptiveResult {
  readonly y: Tensor
  readonly e: Tensor
  readonly w: Tensor
}

/**
 * Run an adaptive filter over a whole record (padasip's run): the outputs, errors and final weights.
 *
 * @param method - 'lms', 'nlms' or 'rls'.
 * @param x - Regressors `[N, n]` or a signal.
 * @param d - Desired signal.
 * @param options - Filter options and tracing.
 * @returns Outputs, errors and final weights.
 * @concept adaptive-filters
 * @example
 * ```ts
 * import { adaptiveFilter } from 'mdml-engine/signal'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 400 }, (_, i) => Math.sin(i) + Math.cos(2.3 * i))
 * const d = x.map((v, i) => 0.7 * v - 0.2 * (x[i - 1] ?? 0))
 * toFlat(adaptiveFilter('rls', x, d, { taps: 2 }).w).map((v) => Math.round(v * 100) / 100) // => [0.7, -0.2]
 * ```
 */
export function adaptiveFilter(
  method: AdaptiveMethod,
  x: TensorLike,
  d: TensorLike,
  options: AdaptiveOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent'>> = {},
): AdaptiveResult {
  const algorithm = adaptiveFilterAlgorithm(method, x, d, options)
  const n = samples(d, 'adaptive filter').length
  const { state } = runAlgorithm(algorithm, {
    maxSteps: n,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  return { y: constant(state.y), e: constant(state.e), w: constant(state.w) }
}

/**
 * Adaptive noise cancellation (Widrow): filter a reference correlated with the noise to predict the noise in the
 * primary input; the error signal is the cleaned signal.
 *
 * @param primary - Signal plus noise.
 * @param reference - A reference noise measurement.
 * @param options - Method (default NLMS), taps (default 16), step (default 0.1) and tracing.
 * @returns The cleaned signal and the estimated noise.
 * @concept adaptive-filters
 * @example
 * ```ts
 * import { adaptiveNoiseCancel } from 'mdml-engine/signal'
 * const n = Array.from({ length: 600 }, (_, i) => Math.sin(1.7 * i) + 0.5 * Math.sin(0.4 * i))
 * const s = Array.from({ length: 600 }, (_, i) => 0.3 * Math.sin(0.05 * i))
 * const out = adaptiveNoiseCancel(s.map((v, i) => v + 0.8 * (n[i] ?? 0)), n, { taps: 4 })
 * out.cleaned.shape // => [600]
 * ```
 */
export function adaptiveNoiseCancel(
  primary: TensorLike,
  reference: TensorLike,
  options: AdaptiveOptions & { readonly method?: AdaptiveMethod } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent'>
    > = {},
): { readonly cleaned: Tensor; readonly noise: Tensor } {
  const r = adaptiveFilter(options.method ?? 'nlms', reference, primary, {
    ...options,
    taps: options.taps ?? 16,
    mu: options.mu ?? 0.1,
  })
  return { cleaned: r.e, noise: r.y }
}
