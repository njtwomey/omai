/**
 * Detection of signals in noise with a controlled false-alarm rate: the energy detector (radiometer), the
 * matched-filter detector for a known template, constant false-alarm rate (CFAR) detection whose threshold adapts to
 * the noise estimated from neighbouring cells, and the receiver operating characteristics of the first two in closed
 * form.
 */

import { MdmlError } from '../base/index.ts'
import { chi2, noncentralChi2, normal } from '../dists/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { constant, samples } from './_shared.ts'

// Detection of signals in noise: the energy (radiometer) and matched-filter detectors with thresholds for a
// false-alarm rate, constant false-alarm rate detection (cell-averaging, greatest-of, smallest-of and ordered
// statistic CFAR), and the theoretical receiver operating characteristics of the two detectors.

function scalar(v: Value): number {
  return toFlat(primalOf(v))[0] ?? Number.NaN
}

/** A detector's per-window statistics, the threshold and the decisions. */
export interface DetectionResult {
  readonly statistic: Tensor
  readonly threshold: number
  readonly detected: boolean[]
}

/** Options for {@link energyDetector}. */
export interface EnergyDetectorOptions {
  /** Window length N (non-overlapping windows; default the whole signal). */
  readonly window?: number
  /** Noise variance per sample. */
  readonly noiseVariance: number
  /** False-alarm probability (default 1e-3). */
  readonly pfa?: number
}

/**
 * The energy detector (radiometer): T = sum x^2 over each window of N samples, compared with sigma^2
 * chi2_N^-1(1 - Pfa), the threshold that real white Gaussian noise of variance sigma^2 exceeds with probability Pfa.
 *
 * @param x - Real signal.
 * @param options - Window, noise variance and false-alarm probability.
 * @returns The window energies, the threshold and the decisions.
 * @concept detection-theory
 * @example
 * ```ts
 * import { energyDetector } from 'mdml-engine/signal'
 * energyDetector([0.1, -0.2, 3, 2.5], { window: 2, noiseVariance: 0.1 }).detected // => [false, true]
 * ```
 */
export function energyDetector(x: TensorLike, options: EnergyDetectorOptions): DetectionResult {
  const s = samples(x, 'energyDetector')
  const n = options.window ?? s.length
  if (!Number.isInteger(n) || n < 1) throw new MdmlError('value', 'energyDetector: the window must be positive')
  const pfa = options.pfa ?? 1e-3
  const t = options.noiseVariance * scalar(chi2(n).isf(pfa))
  const stats: number[] = []
  for (let start = 0; start + n <= s.length; start += n) {
    let e = 0
    for (let i = start; i < start + n; i++) e += (s[i] ?? 0) ** 2
    stats.push(e)
  }
  return { statistic: constant(stats), threshold: t, detected: stats.map((v) => v > t) }
}

/** Options for {@link matchedFilterDetector}. */
export interface MatchedDetectorOptions {
  readonly noiseVariance: number
  readonly pfa?: number
}

/**
 * The matched-filter detector of a known template in white Gaussian noise: the correlation at each lag normalised
 * to unit noise variance, T = (s . x) / (sigma ||s||), compared with Q^-1(Pfa).
 *
 * @param x - Observation.
 * @param template - Known signal.
 * @param options - Noise variance and false-alarm probability (default 1e-3).
 * @returns The normalised statistic at each valid lag, the threshold and the decisions.
 * @concept detection-theory
 * @example
 * ```ts
 * import { matchedFilterDetector } from 'mdml-engine/signal'
 * matchedFilterDetector([0, 0, 3, 3, 0, 0], [1, 1], { noiseVariance: 0.1 }).detected // => [false, true, true, true, false]
 * ```
 */
export function matchedFilterDetector(
  x: TensorLike,
  template: TensorLike,
  options: MatchedDetectorOptions,
): DetectionResult {
  const s = samples(x, 'matchedFilterDetector')
  const h = samples(template, 'matchedFilterDetector')
  let e = 0
  for (const v of h) e += v * v
  const norm = Math.sqrt(options.noiseVariance * e)
  const stats: number[] = []
  for (let lag = 0; lag + h.length <= s.length; lag++) {
    let c = 0
    for (const [j, v] of h.entries()) c += v * (s[lag + j] ?? 0)
    stats.push(c / norm)
  }
  const t = scalar(normal(0, 1).isf(options.pfa ?? 1e-3))
  return { statistic: constant(stats), threshold: t, detected: stats.map((v) => v > t) }
}

/** CFAR variants: cell averaging, greatest-of and smallest-of the two half-windows, ordered statistic. */
export type CfarMethod = 'ca' | 'go' | 'so' | 'os'

/** Options for {@link cfar}. */
export interface CfarOptions {
  /** Training cells on each side (default 8). */
  readonly train?: number
  /** Guard cells on each side (default 2). */
  readonly guard?: number
  /** False-alarm probability (default 1e-4). */
  readonly pfa?: number
  /** Variant (default 'ca'). */
  readonly method?: CfarMethod
  /** For 'os': the rank k of the ordered training cells (default 3/4 of them). */
  readonly rank?: number
}

/** P_fa of OS-CFAR with N cells, rank k and scale alpha: prod_{i < k} (N - i) / (N - i + alpha). */
function osPfa(n: number, k: number, alpha: number): number {
  let p = 1
  for (let i = 0; i < k; i++) p *= (n - i) / (n - i + alpha)
  return p
}

/**
 * Constant false-alarm rate detection on square-law (exponentially distributed) power samples: each cell's threshold
 * is a scale factor times a noise estimate from the training cells around it (excluding guard cells). The factor gives
 * the requested Pfa exactly for cell averaging (N (Pfa^(-1/N) - 1)) and ordered statistics, and approximately for the
 * greatest-of and smallest-of variants (which use the cell-averaging factor of one half-window).
 *
 * @param power - Power samples.
 * @param options - Training and guard cells, false-alarm probability, variant and rank.
 * @returns Thresholds per cell (NaN where the window does not fit) and the decisions.
 * @concept cfar
 * @example
 * ```ts
 * import { cfar } from 'mdml-engine/signal'
 * const p = [1, 1.2, 0.8, 1, 1.1, 0.9, 30, 1, 0.9, 1.1, 1, 1.2, 0.8]
 * cfar(p, { train: 3, guard: 1, pfa: 1e-3 }).detected.indexOf(true) // => 6
 * ```
 */
export function cfar(
  power: TensorLike,
  options: CfarOptions = {},
): { readonly threshold: Tensor; readonly detected: boolean[] } {
  const p = samples(power, 'cfar')
  const train = options.train ?? 8
  const guard = options.guard ?? 2
  const pfa = options.pfa ?? 1e-4
  const method = options.method ?? 'ca'
  const n = 2 * train
  const half = (t: number): number => t * (pfa ** (-1 / t) - 1)
  let alpha: number
  let rank = options.rank ?? Math.max(1, Math.round(0.75 * n))
  if (method === 'ca') alpha = half(n) / n
  else if (method === 'go' || method === 'so') alpha = half(train) / train
  else {
    if (!(rank >= 1 && rank <= n)) throw new MdmlError('value', 'cfar: the rank must lie in 1 .. 2 train')
    rank = Math.round(rank)
    let lo = 0
    let hi = 1
    while (osPfa(n, rank, hi) > pfa) hi *= 2
    for (let it = 0; it < 200; it++) {
      const mid = 0.5 * (lo + hi)
      if (osPfa(n, rank, mid) > pfa) lo = mid
      else hi = mid
    }
    alpha = 0.5 * (lo + hi)
  }
  const thr = new Float64Array(p.length).fill(Number.NaN)
  const detected = new Array<boolean>(p.length).fill(false)
  for (let i = train + guard; i + train + guard < p.length; i++) {
    const left: number[] = []
    const right: number[] = []
    for (let j = i - guard - train; j < i - guard; j++) left.push(p[j] ?? 0)
    for (let j = i + guard + 1; j <= i + guard + train; j++) right.push(p[j] ?? 0)
    const sumOf = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0)
    let noise: number
    if (method === 'ca') noise = sumOf(left) + sumOf(right)
    else if (method === 'go') noise = Math.max(sumOf(left), sumOf(right))
    else if (method === 'so') noise = Math.min(sumOf(left), sumOf(right))
    else noise = [...left, ...right].sort((a, b) => a - b)[rank - 1] ?? 0
    const t = alpha * noise
    thr[i] = t
    detected[i] = (p[i] ?? 0) > t
  }
  return { threshold: constant(thr), detected }
}

/**
 * The ROC of detecting a known signal in white Gaussian noise (the matched filter): Pd = Q(Q^-1(Pfa) - d) with d^2 =
 * E / sigma^2 the deflection.
 *
 * @param pfa - False-alarm probabilities.
 * @param d2 - Deflection (energy-to-noise ratio).
 * @returns Detection probabilities.
 * @concept detection-theory
 * @example
 * ```ts
 * import { rocKnownSignal } from 'mdml-engine/signal'
 * rocKnownSignal([0.5], 0).map((v) => Math.round(v * 1e12) / 1e12) // => [0.5]
 * ```
 */
export function rocKnownSignal(pfa: readonly number[], d2: number): number[] {
  const z = normal(0, 1)
  return pfa.map((p) => scalar(z.sf(scalar(z.isf(p)) - Math.sqrt(d2))))
}

/**
 * The ROC of the energy detector for a deterministic signal of total energy-to-noise ratio snr over N samples:
 * Pd = Q_chi2'(gamma; N, lambda = N snr) at the threshold gamma = chi2_N^-1(1 - Pfa).
 *
 * @param pfa - False-alarm probabilities.
 * @param n - Samples per decision.
 * @param snr - Per-sample signal-to-noise ratio.
 * @returns Detection probabilities.
 * @concept detection-theory
 * @example
 * ```ts
 * import { rocEnergyDetector } from 'mdml-engine/signal'
 * rocEnergyDetector([0.01], 10, 3)[0] > 0.9 // => true
 * ```
 */
export function rocEnergyDetector(pfa: readonly number[], n: number, snr: number): number[] {
  const h0 = chi2(n)
  const h1 = noncentralChi2(n, n * snr)
  return pfa.map((p) => scalar(h1.sf(scalar(h0.isf(p)))))
}
