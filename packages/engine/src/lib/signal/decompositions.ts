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
import {
  type Algorithm,
  type Contract,
  defineContract,
  type RunOptions,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
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
  /** The rule that ends the sifting of an IMF (default `sdStop(sdThreshold)`; see {@link SiftStopRule}). */
  readonly stopping?: SiftStopRule
}

/** What a stopping rule says after a sift: whether to stop, the statistic it looked at, and its running count. */
export interface SiftCheck {
  readonly stop: boolean
  /** The statistic the rule looked at (the SD, the S-number count, the sift index). */
  readonly statistic: number
  /** The consecutive sifts that satisfied the rule so far (carried in the state). */
  readonly stable: number
}

/**
 * A rule that ends the sifting of an intrinsic mode function: it sees the candidate before and after a sift and the
 * count it carried, and says whether to stop. `sdStop`, `sNumberStop` and `fixedSiftStop` are the rules of the
 * `siftStops` table.
 */
export interface SiftStopRule {
  readonly name: string
  check(before: Float64Array, after: Float64Array, stable: number, sifts: number): SiftCheck
}

/**
 * Huang's standard-deviation criterion: stop when sum (h_old - h)^2 / sum h_old^2 falls below the threshold.
 *
 * @param threshold - The threshold (default 0.2).
 * @returns The rule.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { sdStop } from 'mdml-engine/signal'
 * sdStop(0.2).check(Float64Array.of(1, -1, 1), Float64Array.of(1, -1, 1), 0, 1).stop // => true
 * ```
 */
export function sdStop(threshold = 0.2): SiftStopRule {
  return {
    name: 'sd',
    check: (before, after) => {
      let num = 0
      let den = 0
      for (let i = 0; i < before.length; i++) {
        num += ((before[i] ?? 0) - (after[i] ?? 0)) ** 2
        den += (before[i] ?? 0) ** 2
      }
      return { stop: den === 0 || num / den < threshold, statistic: den === 0 ? 0 : num / den, stable: 0 }
    },
  }
}

/** Sign changes of x. */
function zeroCrossings(x: Float64Array): number {
  let c = 0
  for (let i = 1; i < x.length; i++) if ((x[i - 1] ?? 0) * (x[i] ?? 0) < 0) c += 1
  return c
}

/**
 * The S-number criterion (Huang, Wu et al. 2003): stop when the number of extrema and the number of zero crossings
 * have differed by at most one for `s` consecutive sifts.
 *
 * @param s - The number of consecutive sifts (default 4).
 * @returns The rule; its statistic is the count of consecutive sifts so far.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { sNumberStop } from 'mdml-engine/signal'
 * const x = Float64Array.from({ length: 40 }, (_, i) => Math.sin(i))
 * sNumberStop(1).check(x, x, 0, 1).stop // => true
 * ```
 */
export function sNumberStop(s = 4): SiftStopRule {
  return {
    name: 's-number',
    check: (_before, after, stable) => {
      const extremaCount = extrema(after, 1).length + extrema(after, -1).length
      const kept = Math.abs(extremaCount - zeroCrossings(after)) <= 1 ? stable + 1 : 0
      return { stop: kept >= s, statistic: kept, stable: kept }
    },
  }
}

/**
 * A fixed number of sifts per mode.
 *
 * @param count - The number of sifts.
 * @returns The rule; its statistic is the sift index.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { fixedSiftStop } from 'mdml-engine/signal'
 * fixedSiftStop(3).check(Float64Array.of(1), Float64Array.of(1), 0, 3).stop // => true
 * ```
 */
export function fixedSiftStop(count: number): SiftStopRule {
  return { name: 'fixed', check: (_b, _a, stable, sifts) => ({ stop: sifts >= count, statistic: sifts, stable }) }
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

/** Intrinsic mode functions and the residue of a decomposition. */
export interface ModeDecomposition {
  /** IMFs `[k, n]`, highest frequency first. */
  readonly imfs: Tensor
  /** The residue `[n]` (the IMFs and the residue sum to the signal). */
  readonly residue: Tensor
}

/** The state of {@link emdAlgorithm}: the modes found, the residue, and the candidate being sifted. */
export interface EmdState {
  /** The IMFs extracted so far, highest frequency first. */
  readonly imfs: readonly (readonly number[])[]
  /** The signal minus the IMFs so far. */
  readonly residue: readonly number[]
  /** The candidate for the next IMF (the residue, sifted `sifts` times). */
  readonly candidate: readonly number[]
  readonly sifts: number
  /** The stopping rule's statistic after the last sift. */
  readonly statistic: number
  /** The rule's running count (the S-number criterion's consecutive sifts). */
  readonly stable: number
  readonly finished: boolean
}

/** The paths the EMD algorithms log, as a type. */
export type EmdPath =
  | 'emd/imf'
  | 'emd/sift'
  | 'emd/sd'
  | 'emd/residue'
  | 'emd/envelope'
  | 'emd/count'
  | 'emd/trial'
  | 'emd/modes'

/**
 * What the EMD algorithms log, by path: the candidate (or the finished) mode, the sifts spent on it, the stopping rule's
 * statistic, the residue, the upper and lower envelopes of the sift, the modes so far; for the ensembles, the trials done
 * and the running average of the modes.
 */
export const emdContract: Contract<EmdState, EmdPath> = defineContract({
  'emd/imf': tensor({
    role: 'output',
    shape: ['n'],
    description: 'the candidate IMF after this sift (the mode, once done)',
  }),
  'emd/sift': scalar({ description: 'sifts spent on the current mode' }),
  'emd/sd': scalar({
    description: 'the stopping rule’s statistic after this sift (Huang’s SD by default)',
    direction: 'lower',
  }),
  'emd/residue': tensor({ role: 'state', shape: ['n'], description: 'the signal minus the modes extracted' }),
  'emd/envelope': tensor({ role: 'state', shape: [2, 'n'], description: 'the upper and lower envelopes of this sift' }),
  'emd/count': scalar({ description: 'modes extracted so far', direction: 'higher' }),
  'emd/trial': scalar({ description: 'ensemble members done (EEMD)', direction: 'higher' }),
  'emd/modes': tensor({
    role: 'output',
    shape: ['modes', 'n'],
    description: 'the running average of the modes (EEMD)',
  }),
})

function logEmd(
  ctx: StepContext<EmdPath>,
  imf: readonly number[],
  sifts: number,
  statistic: number,
  residue: readonly number[],
  count: number,
  envelopes: readonly [Float64Array, Float64Array] | undefined,
): void {
  if (ctx.wants('emd/imf')) ctx.log('emd/imf', Float64Array.from(imf))
  if (ctx.wants('emd/sift')) ctx.log('emd/sift', sifts)
  if (ctx.wants('emd/sd')) ctx.log('emd/sd', statistic)
  if (ctx.wants('emd/residue')) ctx.log('emd/residue', Float64Array.from(residue))
  if (ctx.wants('emd/count')) ctx.log('emd/count', count)
  if (envelopes !== undefined && ctx.wants('emd/envelope'))
    ctx.log('emd/envelope', Float64Array.from([...envelopes[0], ...envelopes[1]]))
}

/**
 * Empirical mode decomposition (Huang et al. 1998) as a steppable algorithm: one step is one sift, which subtracts the
 * mean of the cubic-spline envelopes through the maxima and minima (mirrored at the ends) from the candidate; when the
 * stopping rule holds (Huang's SD criterion by default), the sift limit is reached or no more sifting is possible, the
 * candidate becomes the next IMF and is subtracted from the residue. The run ends when the residue is monotone or
 * `maxImfs` modes are found. The IMFs and the residue sum to the signal exactly.
 *
 * @param x - Signal.
 * @param options - IMF count, stopping rule (or SD threshold) and sift limit.
 * @returns The algorithm; its state is an {@link EmdState}.
 * @concept empirical-mode-decomposition
 * @index algorithm Empirical mode decomposition, one sifting step at a time.
 * @example
 * ```ts
 * import { emdAlgorithm } from 'mdml-engine/signal'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 200 }, (_, i) => Math.sin(i / 3) + 0.5 * Math.sin(i / 20) + i / 100)
 * runAlgorithm(emdAlgorithm(x), { maxSteps: 1000 }).state.imfs.length // => 3
 * ```
 */
export function emdAlgorithm(x: TensorLike, options: EmdOptions = {}): Algorithm<EmdState, EmdPath> {
  const signal = samples(x, 'emd')
  const rule = options.stopping ?? sdStop(options.sdThreshold ?? 0.2)
  const maxSifts = options.maxSifts ?? 50
  const maxImfs = options.maxImfs ?? Number.POSITIVE_INFINITY
  return {
    name: 'emd',
    config: { n: signal.length, stopping: rule.name, maxSifts, maxImfs: Number.isFinite(maxImfs) ? maxImfs : -1 },
    init: () => ({
      imfs: [],
      residue: Array.from(signal),
      candidate: Array.from(signal),
      sifts: 0,
      statistic: 0,
      stable: 0,
      finished: maxImfs <= 0 || isMonotone(signal),
    }),
    step: (s, ctx) => {
      const h = Float64Array.from(s.candidate)
      const maxima = extrema(h, 1)
      const minima = extrema(h, -1)
      let next = h
      let statistic = s.statistic
      let stable = s.stable
      let sifts = s.sifts
      let envelopes: [Float64Array, Float64Array] | undefined
      let stop = true
      if (sifts < maxSifts && maxima.length >= 2 && minima.length >= 2) {
        const upper = envelope(h, maxima)
        const lower = envelope(h, minima)
        envelopes = [upper, lower]
        next = Float64Array.from(h, (v, i) => v - 0.5 * ((upper[i] ?? 0) + (lower[i] ?? 0)))
        sifts += 1
        const c = rule.check(h, next, stable, sifts)
        statistic = c.statistic
        stable = c.stable
        stop = c.stop || sifts >= maxSifts
      }
      if (!stop) {
        logEmd(ctx, Array.from(next), sifts, statistic, s.residue, s.imfs.length, envelopes)
        return { ...s, candidate: Array.from(next), sifts, statistic, stable }
      }
      const imf = Array.from(next)
      const residue = s.residue.map((v, i) => v - (imf[i] ?? 0))
      const imfs = [...s.imfs, imf]
      logEmd(ctx, imf, sifts, statistic, residue, imfs.length, envelopes)
      return {
        imfs,
        residue,
        candidate: residue,
        sifts: 0,
        statistic,
        stable: 0,
        finished: imfs.length >= maxImfs || isMonotone(Float64Array.from(residue)),
      }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EmdState,
  }
}

function emdNumbers(x: Float64Array, options: EmdOptions): Float64Array[] {
  const alg = emdAlgorithm(x, options)
  const s = runAlgorithm(alg, { maxSteps: Number.MAX_SAFE_INTEGER }).state
  return s.imfs.map((m) => Float64Array.from(m))
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

/** The state of {@link eemdAlgorithm}: the ensemble members done and the running sum of their modes. */
export interface EemdState {
  /** Ensemble members done. */
  readonly trial: number
  /** The number of modes kept (fixed by the first member unless `maxImfs` is given; -1 before it). */
  readonly count: number
  /** The modes averaged over the members done (each divided by the total number of trials already). */
  readonly sum: readonly (readonly number[])[]
}

/**
 * Ensemble EMD (Wu and Huang 2009) as a steppable algorithm: one step is one ensemble member, the EMD of the signal plus
 * independent white noise; the state carries the running average of the modes, so the average after t members is the
 * state at step t. The IMF count is fixed by the first member (or `maxImfs`).
 *
 * @param x - Signal.
 * @param options - Key, ensemble size, noise width and the EMD options.
 * @returns The algorithm; its state is an {@link EemdState}.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { eemdAlgorithm } from 'mdml-engine/signal'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 128 }, (_, i) => Math.sin(i / 2) + Math.sin(i / 15))
 * runAlgorithm(eemdAlgorithm(x, { key: key(1), trials: 4, maxImfs: 3 }), { maxSteps: 10 }).state.trial // => 4
 * ```
 */
export function eemdAlgorithm(x: TensorLike, options: EnsembleEmdOptions): Algorithm<EemdState, EmdPath> {
  const s = samples(x, 'eemd')
  const n = s.length
  const trials = options.trials ?? 50
  const width = (options.noiseWidth ?? 0.2) * std(s)
  return {
    name: 'eemd',
    config: { n, trials, noiseWidth: options.noiseWidth ?? 0.2, maxImfs: options.maxImfs ?? -1 },
    init: () => ({ trial: 0, count: options.maxImfs ?? -1, sum: [] }),
    step: (st, ctx) => {
      const w = toFlat(normal(foldIn(options.key, st.trial), [n]))
      const noisy = Float64Array.from(s, (v, i) => v + width * (w[i] ?? 0))
      const imfs = emdNumbers(noisy, st.count < 0 ? options : { ...options, maxImfs: st.count })
      const count = st.count < 0 ? imfs.length : st.count
      const sum =
        st.sum.length === 0
          ? Array.from({ length: count }, () => new Array<number>(n).fill(0))
          : st.sum.map((r) => [...r])
      for (let k = 0; k < count; k++) {
        const m = imfs[k]
        const acc = sum[k] as number[]
        if (m !== undefined) for (let i = 0; i < n; i++) acc[i] = (acc[i] ?? 0) + (m[i] ?? 0) / trials
      }
      if (ctx.wants('emd/trial')) ctx.log('emd/trial', st.trial + 1)
      if (ctx.wants('emd/count')) ctx.log('emd/count', imfs.length)
      if (ctx.wants('emd/modes')) ctx.log('emd/modes', Float64Array.from(sum.flat()))
      return { trial: st.trial + 1, count, sum }
    },
    done: (st) => (st.trial >= trials ? 'converged' : undefined),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as EemdState,
  }
}

/**
 * Ensemble EMD (Wu and Huang 2009): the average IMFs of the signal plus independent white noise realisations, which
 * reduces mode mixing; the IMF count is fixed (by the first realisation) and the residue is the remainder.
 * {@link eemdAlgorithm} run over all the members.
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
  const st = runAlgorithm(eemdAlgorithm(x, options), { maxSteps: options.trials ?? 50 }).state
  return pack(
    st.sum.map((r) => Float64Array.from(r)),
    s,
  )
}

/** The state of {@link ceemdanAlgorithm}: the modes found and the residue. */
export interface CeemdanState {
  readonly imfs: readonly (readonly number[])[]
  readonly residue: readonly number[]
  readonly finished: boolean
}

/**
 * Complete ensemble EMD with adaptive noise (CEEMDAN, Torres et al. 2011) as a steppable algorithm: one step finds one
 * mode, the ensemble average of the first EMD mode of the current residue plus the matching EMD mode of white noise
 * scaled by beta_k = noiseWidth std(residue). The decomposition is complete (the modes and residue sum to the signal)
 * with no residual noise. The noise realisations and their EMD modes are computed on the first step and kept in the
 * algorithm (a cache, not state), so a resumed run recomputes them.
 *
 * @param x - Signal.
 * @param options - Key, ensemble size, noise width and the EMD options.
 * @returns The algorithm; its state is a {@link CeemdanState}.
 * @concept empirical-mode-decomposition
 * @example
 * ```ts
 * import { ceemdanAlgorithm } from 'mdml-engine/signal'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 128 }, (_, i) => Math.sin(i / 2) + Math.sin(i / 15))
 * runAlgorithm(ceemdanAlgorithm(x, { key: key(2), trials: 4, maxImfs: 2 }), { maxSteps: 10 }).state.imfs.length // => 2
 * ```
 */
export function ceemdanAlgorithm(x: TensorLike, options: EnsembleEmdOptions): Algorithm<CeemdanState, EmdPath> {
  const s = samples(x, 'ceemdan')
  const n = s.length
  const trials = options.trials ?? 50
  const eps = options.noiseWidth ?? 0.2
  const maxImfs = options.maxImfs ?? Number.POSITIVE_INFINITY
  let cache: { noise: Float64Array[]; noiseModes: Float64Array[][] } | undefined
  const realisations = (): { noise: Float64Array[]; noiseModes: Float64Array[][] } => {
    if (cache !== undefined) return cache
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
    cache = { noise, noiseModes }
    return cache
  }
  const firstMode = (r: Float64Array): Float64Array =>
    emdNumbers(r, { ...options, maxImfs: 1 })[0] ?? new Float64Array(n)
  return {
    name: 'ceemdan',
    config: { n, trials, noiseWidth: eps, maxImfs: Number.isFinite(maxImfs) ? maxImfs : -1 },
    init: () => ({ imfs: [], residue: Array.from(s), finished: maxImfs <= 0 || isMonotone(s) }),
    step: (st, ctx) => {
      const { noise, noiseModes } = realisations()
      const residue = Float64Array.from(st.residue)
      const k = st.imfs.length
      const beta = eps * std(residue)
      const mode = new Float64Array(n)
      for (const [t, modes] of noiseModes.entries()) {
        // IMF_1 from x + beta w; IMF_(k+1) from r_k + beta E_k(w), the k-th EMD mode of the noise.
        const e = k === 0 ? noise[t] : modes[k - 1]
        const perturbed = e === undefined ? residue : Float64Array.from(residue, (v, i) => v + beta * (e[i] ?? 0))
        const local = firstMode(perturbed)
        for (let i = 0; i < n; i++) mode[i] = (mode[i] ?? 0) + (local[i] ?? 0) / trials
      }
      const next = Float64Array.from(residue, (v, i) => v - (mode[i] ?? 0))
      const imfs = [...st.imfs, Array.from(mode)]
      if (ctx.wants('emd/imf')) ctx.log('emd/imf', mode)
      if (ctx.wants('emd/residue')) ctx.log('emd/residue', next)
      if (ctx.wants('emd/count')) ctx.log('emd/count', imfs.length)
      return { imfs, residue: Array.from(next), finished: imfs.length >= maxImfs || isMonotone(next) }
    },
    done: (st) => (st.finished ? 'converged' : undefined),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as CeemdanState,
  }
}

/**
 * Complete ensemble EMD with adaptive noise (CEEMDAN, Torres et al. 2011): each mode is the ensemble average of the
 * first EMD mode of the current residue plus the matching EMD mode of white noise scaled by beta_k = noiseWidth
 * std(residue), so the decomposition is complete (the modes and residue sum to the signal) with no residual noise.
 * {@link ceemdanAlgorithm} run to the end.
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
  const st = runAlgorithm(ceemdanAlgorithm(x, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state
  return pack(
    st.imfs.map((m) => Float64Array.from(m)),
    s,
  )
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
