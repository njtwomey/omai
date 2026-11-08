/**
 * Diffusion models as generative models (as opposed to the SDE solvers of `sde.ts`): the noise schedules (linear,
 * cosine, sigmoid, the discrete DDPM betas, the variance-exploding and EDM ones) each with its signal scale alpha(t),
 * noise scale sigma(t), log signal-to-noise ratio and the drift and diffusion of the equivalent forward SDE; the
 * variance-preserving and variance-exploding SDEs with their closed-form marginals and reverse-time form; the forward
 * process sampler q(x_t | x_0); the training losses with their weightings (L_simple, SNR-plus-one, truncated and
 * min-SNR, the ELBO, EDM's); classifier and classifier-free guidance; and the exact score of a Gaussian mixture under the
 * forward process, a planted truth against which samplers and trained networks can be verified. The reverse samplers
 * and the trainable network are in `diffusion-samplers.ts`.
 *
 * @index module Diffusion models: noise schedules, forward process, losses, guidance, exact mixture score.
 */

import { OmaiError } from '../base/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import {
  add,
  div,
  log,
  logsumexp,
  mean,
  mul,
  neg,
  promoteFloat,
  reshape,
  softmax,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  type Value,
} from '../tensor/index.ts'
import type { Sde } from './sde.ts'

// --- Schedules -------------------------------------------------------------------------------------------------------

/**
 * A noise schedule: the process x_t = alpha(t) x_0 + sigma(t) epsilon over t in (0, 1], with the log signal-to-noise
 * ratio and the drift f(t) = d log alpha / dt and squared diffusion g^2(t) = d sigma^2/dt - 2 f(t) sigma^2 of the forward
 * SDE dx = f(t) x dt + g(t) dw that has these marginals. `kind` is 'vp' (variance preserving: alpha^2 + sigma^2 = 1) or
 * 've' (variance exploding: alpha = 1).
 *
 * @concept diffusion-models
 *
 * @index structure A diffusion noise schedule: signal and noise scales, log-SNR, forward drift.
 */
export interface NoiseSchedule {
  readonly name: string
  readonly kind: 'vp' | 've'
  /** The smallest time a sampler visits before its final step to zero noise. */
  readonly tMin: number
  /** The time sampling starts from. */
  readonly tMax: number
  alpha(t: number): number
  sigma(t: number): number
  /** log(alpha^2 / sigma^2), decreasing in t (plus infinity where sigma is zero). */
  logSnr(t: number): number
  /** f(t) = d log alpha / dt. */
  drift(t: number): number
  /** g^2(t) = d sigma^2 / dt - 2 f(t) sigma^2. */
  diffusionSquared(t: number): number
}

/** What defines a schedule; the derivatives default to central differences. */
export interface ScheduleSpec {
  readonly name: string
  readonly kind: 'vp' | 've'
  readonly tMin: number
  readonly tMax: number
  alpha(t: number): number
  sigma(t: number): number
  /** d log alpha / dt, when known in closed form. */
  dLogAlpha?(t: number): number
  /** d sigma^2 / dt, when known in closed form. */
  dSigma2?(t: number): number
}

/**
 * Build a {@link NoiseSchedule} from its alpha and sigma, deriving the log-SNR, drift and squared diffusion (the
 * derivatives by central differences unless the spec gives them).
 *
 * @param spec - Name, kind, time range, alpha, sigma and optionally their derivatives.
 * @returns The schedule.
 * @index function Build a custom diffusion noise schedule from its alpha and sigma.
 * @example
 * ```ts
 * import { defineSchedule } from 'omai-engine/stochastic'
 * const s = defineSchedule({ name: 've', kind: 've', tMin: 0.01, tMax: 1, alpha: () => 1, sigma: (t) => t })
 * Math.abs(s.diffusionSquared(0.5) - 1) < 1e-8 // => true
 * ```
 */
export function defineSchedule(spec: ScheduleSpec): NoiseSchedule {
  const h = 1e-6
  const dLogAlpha =
    spec.dLogAlpha ?? ((t: number) => (Math.log(spec.alpha(t + h)) - Math.log(spec.alpha(t - h))) / (2 * h))
  const dSigma2 = spec.dSigma2 ?? ((t: number) => (spec.sigma(t + h) ** 2 - spec.sigma(t - h) ** 2) / (2 * h))
  return {
    name: spec.name,
    kind: spec.kind,
    tMin: spec.tMin,
    tMax: spec.tMax,
    alpha: spec.alpha,
    sigma: spec.sigma,
    logSnr: (t) => 2 * (Math.log(spec.alpha(t)) - Math.log(spec.sigma(t))),
    drift: dLogAlpha,
    diffusionSquared: (t) => dSigma2(t) - 2 * dLogAlpha(t) * spec.sigma(t) ** 2,
  }
}

/**
 * The linear variance-preserving schedule of the VP-SDE (Song et al., 2021): beta(t) = betaMin + t (betaMax - betaMin),
 * alpha(t) = exp(-(t^2 (betaMax - betaMin) / 4 + t betaMin / 2)), sigma^2 = 1 - alpha^2; the continuous limit of DDPM's
 * linear betas.
 *
 * @param options - betaMin (default 0.1) and betaMax (default 20).
 * @returns The schedule, with `drift(t) = -beta(t) / 2` and `diffusionSquared(t) = beta(t)`.
 * @concept diffusion-models
 * @index function Linear-beta variance-preserving noise schedule of the VP-SDE.
 * @example
 * ```ts
 * import { linearSchedule } from 'omai-engine/stochastic'
 * const s = linearSchedule()
 * Math.abs(s.diffusionSquared(0.5) - (0.1 + 0.5 * 19.9)) < 1e-12 // => true
 * ```
 */
export function linearSchedule(options: { readonly betaMin?: number; readonly betaMax?: number } = {}): NoiseSchedule {
  const [b0, b1] = [options.betaMin ?? 0.1, options.betaMax ?? 20]
  const beta = (t: number): number => b0 + t * (b1 - b0)
  const logAlpha = (t: number): number => -(0.25 * t * t * (b1 - b0) + 0.5 * t * b0)
  return defineSchedule({
    name: 'linear',
    kind: 'vp',
    tMin: 1e-3,
    tMax: 1,
    alpha: (t) => Math.exp(logAlpha(t)),
    sigma: (t) => Math.sqrt(-Math.expm1(2 * logAlpha(t))),
    dLogAlpha: (t) => -0.5 * beta(t),
    dSigma2: (t) => 2 * Math.exp(2 * logAlpha(t)) * 0.5 * beta(t),
  })
}

/** The floor on alpha-bar of the cosine and sigmoid schedules, so alpha stays positive and x0 predictions stay finite. */
const ALPHA_BAR_FLOOR = 1e-5

/**
 * The cosine schedule (Nichol and Dhariwal, 2021): alpha-bar(t) = cos^2((t + s) / (1 + s) pi / 2) / cos^2(s / (1 + s)
 * pi / 2), with alpha-bar floored at 1e-5 (so sampling starts at t = 0.998 where alpha is small but positive).
 *
 * @param options - s, the offset that keeps beta small near zero (default 0.008).
 * @returns The schedule.
 * @index function Cosine variance-preserving noise schedule (Nichol and Dhariwal).
 * @example
 * ```ts
 * import { cosineSchedule } from 'omai-engine/stochastic'
 * Math.abs(cosineSchedule().alpha(0) - 1) < 1e-12 // => true
 * ```
 */
export function cosineSchedule(options: { readonly s?: number } = {}): NoiseSchedule {
  const s = options.s ?? 0.008
  const angle = (t: number): number => (((t + s) / (1 + s)) * Math.PI) / 2
  const f0 = Math.cos(angle(0)) ** 2
  const alphaBar = (t: number): number => Math.max(Math.cos(angle(t)) ** 2 / f0, ALPHA_BAR_FLOOR)
  const dAngle = Math.PI / (2 * (1 + s))
  return defineSchedule({
    name: 'cosine',
    kind: 'vp',
    tMin: 1e-3,
    tMax: 0.998,
    alpha: (t) => Math.sqrt(alphaBar(t)),
    sigma: (t) => Math.sqrt(1 - alphaBar(t)),
    dLogAlpha: (t) => -Math.tan(angle(t)) * dAngle,
    dSigma2: (t) => 2 * alphaBar(t) * Math.tan(angle(t)) * dAngle,
  })
}

/**
 * The sigmoid schedule (Jabri et al., 2022): alpha-bar(t) = (sigmoid(end / tau) - sigmoid((t (end - start) + start) /
 * tau)) / (sigmoid(end / tau) - sigmoid(start / tau)), floored at 1e-5.
 *
 * @param options - start (default -3), end (default 3) and tau (default 1).
 * @returns The schedule.
 * @index function Sigmoid variance-preserving noise schedule.
 * @example
 * ```ts
 * import { sigmoidSchedule } from 'omai-engine/stochastic'
 * Math.abs(sigmoidSchedule().alpha(0) - 1) < 1e-12 // => true
 * ```
 */
export function sigmoidSchedule(
  options: { readonly start?: number; readonly end?: number; readonly tau?: number } = {},
): NoiseSchedule {
  const [start, end, tau] = [options.start ?? -3, options.end ?? 3, options.tau ?? 1]
  const sig = (x: number): number => 1 / (1 + Math.exp(-x))
  const [vStart, vEnd] = [sig(start / tau), sig(end / tau)]
  const alphaBar = (t: number): number =>
    Math.max((vEnd - sig((t * (end - start) + start) / tau)) / (vEnd - vStart), ALPHA_BAR_FLOOR)
  return defineSchedule({
    name: 'sigmoid',
    kind: 'vp',
    tMin: 1e-3,
    tMax: 0.999,
    alpha: (t) => Math.sqrt(alphaBar(t)),
    sigma: (t) => Math.sqrt(1 - alphaBar(t)),
  })
}

/**
 * DDPM's discrete linear betas (Ho et al., 2020): `n` values from `start` to `end`.
 *
 * @param n - The number of steps (default 1000).
 * @param options - start (default 1e-4) and end (default 0.02).
 * @returns The betas.
 * @example
 * ```ts
 * import { linearBetas } from 'omai-engine/stochastic'
 * linearBetas(3, { start: 0, end: 1 }) // => [0, 0.5, 1]
 * ```
 */
export function linearBetas(n = 1000, options: { readonly start?: number; readonly end?: number } = {}): number[] {
  const [a, b] = [options.start ?? 1e-4, options.end ?? 0.02]
  return Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)))
}

/**
 * The discrete cosine betas of Nichol and Dhariwal: beta_i = min(1 - alpha-bar(i / n) / alpha-bar((i - 1) / n), 0.999).
 *
 * @param n - The number of steps.
 * @param options - s (default 0.008) and the clip (default 0.999).
 * @returns The betas.
 * @example
 * ```ts
 * import { cosineBetas } from 'omai-engine/stochastic'
 * cosineBetas(1000).length // => 1000
 * ```
 */
export function cosineBetas(n: number, options: { readonly s?: number; readonly clip?: number } = {}): number[] {
  const s = options.s ?? 0.008
  const fn = (t: number): number => Math.cos(((t + s) / (1 + s)) * (Math.PI / 2)) ** 2
  return Array.from({ length: n }, (_, i) => Math.min(1 - fn((i + 1) / n) / fn(i / n), options.clip ?? 0.999))
}

/**
 * A discrete DDPM schedule as a continuous one: step i of n sits at t = i / n, alpha-bar_i = prod_{j <= i} (1 - beta_j),
 * and log alpha-bar is interpolated linearly in between, so the schedule is exact at the grid points and a sampler on
 * those points is DDPM itself.
 *
 * @param betas - The betas of the n steps.
 * @returns The schedule, with tMin = 1 / n and tMax = 1.
 * @throws OmaiError with code `value` for a beta outside [0, 1).
 * @index function Noise schedule from a list of discrete DDPM betas.
 * @example
 * ```ts
 * import { discreteSchedule, linearBetas } from 'omai-engine/stochastic'
 * const betas = linearBetas(10, { start: 0.1, end: 0.1 })
 * Math.abs(discreteSchedule(betas).alpha(0.1) ** 2 - 0.9) < 1e-14 // => true
 * ```
 */
export function discreteSchedule(betas: readonly number[]): NoiseSchedule {
  const n = betas.length
  const logBar = [0]
  for (const b of betas) {
    if (!(b >= 0 && b < 1)) throw new OmaiError('value', 'discreteSchedule: betas must lie in [0, 1)')
    logBar.push((logBar[logBar.length - 1] ?? 0) + Math.log1p(-b))
  }
  const at = (t: number): number => {
    const u = Math.min(Math.max(t, 0), 1) * n
    const i = Math.min(Math.floor(u), n - 1)
    const w = u - i
    return (logBar[i] ?? 0) * (1 - w) + (logBar[i + 1] ?? 0) * w
  }
  const slope = (t: number): number => {
    const i = Math.min(Math.floor(Math.min(Math.max(t, 0), 1) * n), n - 1)
    return ((logBar[i + 1] ?? 0) - (logBar[i] ?? 0)) * n
  }
  return defineSchedule({
    name: 'discrete',
    kind: 'vp',
    tMin: 1 / n,
    tMax: 1,
    alpha: (t) => Math.exp(0.5 * at(t)),
    sigma: (t) => Math.sqrt(-Math.expm1(at(t))),
    dLogAlpha: (t) => 0.5 * slope(t),
    dSigma2: (t) => -Math.exp(at(t)) * slope(t),
  })
}

/**
 * The variance-exploding schedule of the VE-SDE (Song et al., 2021): alpha = 1 and sigma(t) = sigmaMin
 * (sigmaMax / sigmaMin)^t, the noise scales of NCSN.
 *
 * @param options - sigmaMin (default 0.01) and sigmaMax (default 50).
 * @returns The schedule, with zero drift and `diffusionSquared(t) = 2 sigma(t)^2 log(sigmaMax / sigmaMin)`.
 * @index function Variance-exploding noise schedule (NCSN, sigma from min to max).
 * @example
 * ```ts
 * import { veSchedule } from 'omai-engine/stochastic'
 * veSchedule({ sigmaMin: 1, sigmaMax: 100 }).sigma(0.5) // => 10
 * ```
 */
export function veSchedule(options: { readonly sigmaMin?: number; readonly sigmaMax?: number } = {}): NoiseSchedule {
  const [lo, hi] = [options.sigmaMin ?? 0.01, options.sigmaMax ?? 50]
  const rate = Math.log(hi / lo)
  return defineSchedule({
    name: 've',
    kind: 've',
    tMin: 1e-3,
    tMax: 1,
    alpha: () => 1,
    sigma: (t) => lo * (hi / lo) ** t,
    dLogAlpha: () => 0,
    dSigma2: (t) => 2 * rate * (lo * (hi / lo) ** t) ** 2,
  })
}

/**
 * The EDM parameterisation of Karras et al. (2022): the time is the noise scale itself, sigma(t) = t and alpha = 1, over
 * [sigmaMin, sigmaMax].
 *
 * @param options - sigmaMin (default 0.002) and sigmaMax (default 80).
 * @returns The schedule.
 * @index function EDM noise schedule where the time is the noise scale.
 * @example
 * ```ts
 * import { edmSchedule } from 'omai-engine/stochastic'
 * edmSchedule().sigma(3) // => 3
 * ```
 */
export function edmSchedule(options: { readonly sigmaMin?: number; readonly sigmaMax?: number } = {}): NoiseSchedule {
  return defineSchedule({
    name: 'edm',
    kind: 've',
    tMin: options.sigmaMin ?? 0.002,
    tMax: options.sigmaMax ?? 80,
    alpha: () => 1,
    sigma: (t) => t,
    dLogAlpha: () => 0,
    dSigma2: (t) => 2 * t,
  })
}

/**
 * The signal-to-noise ratio alpha^2 / sigma^2 of a schedule at a time: large near clean data, small near pure noise.
 *
 * @param schedule - The noise schedule.
 * @param t - The time.
 * @returns exp of the schedule's log-SNR at `t`.
 * @index function Signal-to-noise ratio of a noise schedule at a time.
 * @example
 * ```ts
 * import { linearSchedule, snr } from 'omai-engine/stochastic'
 * const schedule = linearSchedule()
 * snr(schedule, 0.1) > snr(schedule, 0.9) // => true
 * ```
 */
export function snr(schedule: NoiseSchedule, t: number): number {
  return Math.exp(schedule.logSnr(t))
}

/**
 * The preconditioning of EDM (Karras et al., 2022, table 1) for a network F: the denoiser is D(x, sigma) = c_skip x +
 * c_out F(c_in x, c_noise), with c_skip = sigmaData^2 / (sigma^2 + sigmaData^2), c_out = sigma sigmaData /
 * sqrt(sigma^2 + sigmaData^2), c_in = 1 / sqrt(sigma^2 + sigmaData^2) and c_noise = log(sigma) / 4, which keep the
 * network's inputs and targets at unit variance at every noise level.
 *
 * @param sigma - The noise level.
 * @param sigmaData - The data's standard deviation (default 0.5).
 * @returns The four coefficients.
 * @index function EDM preconditioning of a network: skip, output and input scalings.
 * @example
 * ```ts
 * import { edmPreconditioning } from 'omai-engine/stochastic'
 * edmPreconditioning(0.5, 0.5).cSkip // => 0.5
 * ```
 */
export function edmPreconditioning(
  sigma: number,
  sigmaData = 0.5,
): { readonly cSkip: number; readonly cOut: number; readonly cIn: number; readonly cNoise: number } {
  const total = sigma * sigma + sigmaData * sigmaData
  return {
    cSkip: (sigmaData * sigmaData) / total,
    cOut: (sigma * sigmaData) / Math.sqrt(total),
    cIn: 1 / Math.sqrt(total),
    cNoise: Math.log(sigma) / 4,
  }
}

/**
 * EDM's training noise levels: log sigma ~ Normal(pMean, pStd).
 *
 * @param k - The random key.
 * @param n - The number of levels.
 * @param options - pMean (default -1.2) and pStd (default 1.2).
 * @returns The noise levels `[n]`.
 * @example
 * ```ts
 * import { edmNoiseLevels } from 'omai-engine/stochastic'
 * import { key } from 'omai-engine/random'
 * edmNoiseLevels(key(0), 5).length // => 5
 * ```
 */
export function edmNoiseLevels(
  k: Key,
  n: number,
  options: { readonly pMean?: number; readonly pStd?: number } = {},
): number[] {
  const z = normal(k, [n]).data
  return Array.from(z, (v) => Math.exp((options.pMean ?? -1.2) + (options.pStd ?? 1.2) * v))
}

/** How a sampler spaces its times between tMax and tMin. */
export type TimeSpacing = 'uniform' | 'quadratic' | 'logSnr' | 'karras'

/** Bisection of a monotone function for the argument at which it reaches a target. */
function invert(f: (t: number) => number, target: number, lo: number, hi: number): number {
  const increasing = f(hi) > f(lo)
  let [a, b] = [lo, hi]
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (a + b)
    if (f(mid) < target === increasing) a = mid
    else b = mid
  }
  return 0.5 * (a + b)
}

/**
 * The time at which the schedule has a given log-SNR (by bisection; the log-SNR decreases in t).
 *
 * @param schedule - The schedule.
 * @param logSnr - The target log-SNR.
 * @returns The time in [tMin / 100, tMax].
 * @example
 * ```ts
 * import { linearSchedule, timeAtLogSnr } from 'omai-engine/stochastic'
 * const s = linearSchedule()
 * Math.abs(timeAtLogSnr(s, s.logSnr(0.3)) - 0.3) < 1e-9 // => true
 * ```
 */
export function timeAtLogSnr(schedule: NoiseSchedule, logSnr: number): number {
  return invert((t) => schedule.logSnr(t), logSnr, schedule.tMin / 100, schedule.tMax)
}

/**
 * The times of a sampler: `n` points from tMax down to tMin by the chosen spacing, then 0 (clean data) as the final
 * point, so a sampler takes `n` steps and ends at zero noise. Uniform in t, quadratic (the DDIM paper's, finer near the
 * data), uniform in log-SNR (DPM-Solver's recommendation), or Karras's rho = 7 spacing in sigma.
 *
 * @param schedule - The schedule.
 * @param n - The number of steps.
 * @param options - spacing (default 'uniform').
 * @returns `n + 1` decreasing times ending in 0.
 * @index function Times a sampler visits: uniform, quadratic, log-SNR or Karras spacing.
 * @example
 * ```ts
 * import { linearSchedule, samplerTimes } from 'omai-engine/stochastic'
 * samplerTimes(linearSchedule(), 2).length // => 3
 * ```
 */
export function samplerTimes(
  schedule: NoiseSchedule,
  n: number,
  options: { readonly spacing?: TimeSpacing } = {},
): number[] {
  if (!(n >= 1)) throw new OmaiError('value', 'samplerTimes: need at least one step')
  const [hi, lo] = [schedule.tMax, schedule.tMin]
  const u = (i: number): number => (n === 1 ? 0 : i / (n - 1))
  const points = Array.from({ length: n }, (_, i): number => {
    switch (options.spacing ?? 'uniform') {
      case 'uniform':
        return hi + (lo - hi) * u(i)
      case 'quadratic':
        return lo + (hi - lo) * (1 - u(i)) ** 2
      case 'logSnr':
        return timeAtLogSnr(schedule, schedule.logSnr(hi) + (schedule.logSnr(lo) - schedule.logSnr(hi)) * u(i))
      case 'karras': {
        const [sHi, sLo, rho] = [schedule.sigma(hi), schedule.sigma(lo), 7]
        const sigma = (sHi ** (1 / rho) + u(i) * (sLo ** (1 / rho) - sHi ** (1 / rho))) ** rho
        return invert((t) => schedule.sigma(t), sigma, lo / 100, hi)
      }
      default:
        throw new OmaiError('value', `samplerTimes: unknown spacing ${String(options.spacing)}`)
    }
  })
  return [...points, 0]
}

// --- Forward process -------------------------------------------------------------------------------------------------

/**
 * The mean multiplier alpha and standard deviation sigma of q(x_t | x_0) at a time: x_t is alpha x_0 plus sigma times
 * standard normal noise.
 *
 * @param schedule - The noise schedule.
 * @param t - The time.
 * @returns `{ alpha, sigma }`.
 * @index function Signal and noise scales of the forward process at a time.
 * @example
 * ```ts
 * import { edmSchedule, forwardMarginal } from 'omai-engine/stochastic'
 * forwardMarginal(edmSchedule(), 1) // => { alpha: 1, sigma: 1 }
 * ```
 */
export function forwardMarginal(
  schedule: NoiseSchedule,
  t: number,
): { readonly alpha: number; readonly sigma: number } {
  return { alpha: schedule.alpha(t), sigma: schedule.sigma(t) }
}

/** A per-sample column of numbers shaped to broadcast against `ref` (`[N, 1, ...]`), or a scalar for one number. */
function column(values: number | readonly number[], ref: Value): Value {
  if (typeof values === 'number') return tensor(values)
  return reshape(tensor([...values]), [values.length, ...ref.shape.slice(1).map(() => 1)])
}

/** The result of {@link forwardSample}. */
export interface ForwardSample {
  /** x_t = alpha x_0 + sigma epsilon. */
  readonly xt: Value
  readonly noise: Value
  readonly alpha: Value
  readonly sigma: Value
}

/**
 * Sample the forward process q(x_t | x_0) = N(alpha(t) x_0, sigma(t)^2 I) in one step.
 *
 * @param schedule - The schedule.
 * @param x0 - Clean samples `[N, ...]`.
 * @param t - A time for all samples, or one per sample `[N]`.
 * @param k - The random key.
 * @returns x_t, the noise that made it, and alpha and sigma shaped to broadcast against x0.
 * @concept diffusion-models
 * @index function Noise clean samples to a time in one step: q(x_t | x_0).
 * @example
 * ```ts
 * import { forwardSample, linearSchedule } from 'omai-engine/stochastic'
 * import { key } from 'omai-engine/random'
 * forwardSample(linearSchedule(), [[1, 1]], 0.5, key(0)).xt.shape // => [1, 2]
 * ```
 */
export function forwardSample(
  schedule: NoiseSchedule,
  x0: TensorLike,
  t: number | readonly number[],
  k: Key,
): ForwardSample {
  const [x] = promoteFloat(x0) as [Value]
  const noise = normal(k, [...x.shape])
  const times = typeof t === 'number' ? t : [...t]
  const [alpha, sigma] = [
    column(typeof times === 'number' ? schedule.alpha(times) : times.map((u) => schedule.alpha(u)), x),
    column(typeof times === 'number' ? schedule.sigma(times) : times.map((u) => schedule.sigma(u)), x),
  ]
  return { xt: add(mul(alpha, x), mul(sigma, noise)), noise, alpha, sigma }
}

/**
 * The variance-preserving SDE dx = -beta(t) x / 2 dt + sqrt(beta(t)) dw (Song et al., 2021) with its closed-form
 * marginal, as an {@link Sde} for the library's solvers (`reverseSde` and `probabilityFlow` of `sde.ts` take it with a score) and the
 * schedule it comes from.
 *
 * @param options - betaMin and betaMax of the linear schedule (defaults 0.1 and 20).
 * @returns The forward SDE (`drift`, `diffusion`) and the schedule.
 * @concept diffusion-models
 * @index function Variance-preserving SDE of a diffusion model, with reverse and flow forms.
 * @example
 * ```ts
 * import { vpSde } from 'omai-engine/stochastic'
 * vpSde().schedule.name // => 'linear'
 * ```
 */
export function vpSde(options: { readonly betaMin?: number; readonly betaMax?: number } = {}): DiffusionSde {
  return diffusionSde(linearSchedule(options))
}

/**
 * The variance-exploding SDE dx = sqrt(d sigma^2 / dt) dw with sigma(t) = sigmaMin (sigmaMax / sigmaMin)^t.
 *
 * @param options - sigmaMin and sigmaMax (defaults 0.01 and 50).
 * @returns As {@link vpSde}.
 * @index function Variance-exploding SDE of a diffusion model.
 * @example
 * ```ts
 * import { veSde } from 'omai-engine/stochastic'
 * veSde().schedule.kind // => 've'
 * ```
 */
export function veSde(options: { readonly sigmaMin?: number; readonly sigmaMax?: number } = {}): DiffusionSde {
  return diffusionSde(veSchedule(options))
}

/**
 * A forward diffusion SDE with the schedule whose marginals it has.
 *
 * @index structure A diffusion's forward SDE together with its noise schedule.
 */
export interface DiffusionSde extends Sde {
  readonly schedule: NoiseSchedule
}

function diffusionSde(schedule: NoiseSchedule): DiffusionSde {
  return {
    schedule,
    drift: (t, x) => mul(x, schedule.drift(t)),
    diffusion: (t, x) => mul(add(mul(x, 0), 1), Math.sqrt(schedule.diffusionSquared(t))),
    noise: 'diagonal',
  }
}

// --- Predictions, losses and weightings --------------------------------------------------------------------------------

/**
 * What a diffusion model's output estimates: the noise epsilon, the clean sample x0, the velocity v = alpha epsilon -
 * sigma x0 (variance-preserving schedules), or the score of the noisy marginal, -epsilon / sigma.
 */
export type DiffusionPrediction = 'epsilon' | 'x0' | 'v' | 'score'

/**
 * A model of the reverse process: a prediction for a batch `[N, ...]` of noisy samples at one time.
 *
 * @index structure A network function (x_t, t) giving a diffusion model's prediction.
 */
export type DiffusionModel = (xt: Value, t: number) => Value

/**
 * The clean-sample and noise estimates implied by a prediction at x_t = alpha x0 + sigma epsilon.
 *
 * @param prediction - What the model predicts.
 * @param output - The model output.
 * @param xt - The noisy samples.
 * @param alpha - The signal scale.
 * @param sigma - The noise scale.
 * @returns The estimates of x0 and epsilon.
 * @throws OmaiError with code `value` for the velocity of a variance-exploding schedule.
 * @concept diffusion-models
 * @index function Convert an epsilon, x0, v or score prediction to clean sample and noise.
 * @example
 * ```ts
 * import { toX0AndNoise } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * // x_t = 0.6 * 2 + 0.8 * 1 = 2: the noise estimate 1 gives x0 = 2.
 * toArray(toX0AndNoise('epsilon', [1], [2], 0.6, 0.8).x0) // => [2]
 * ```
 */
export function toX0AndNoise(
  prediction: DiffusionPrediction,
  output: TensorLike,
  xt: TensorLike,
  alpha: number,
  sigma: number,
): { readonly x0: Value; readonly noise: Value } {
  const [o, x] = promoteFloat(output, xt) as [Value, Value]
  switch (prediction) {
    case 'epsilon':
      return { noise: o, x0: div(sub(x, mul(o, sigma)), alpha) }
    case 'x0':
      return { x0: o, noise: div(sub(x, mul(o, alpha)), sigma) }
    case 'v': {
      if (Math.abs(alpha * alpha + sigma * sigma - 1) > 1e-6)
        throw new OmaiError('value', 'toX0AndNoise: the velocity needs a variance-preserving schedule')
      return { x0: sub(mul(x, alpha), mul(o, sigma)), noise: add(mul(x, sigma), mul(o, alpha)) }
    }
    case 'score':
      return { noise: mul(o, -sigma), x0: div(add(x, mul(o, sigma * sigma)), alpha) }
  }
}

/** How training errors are weighted across noise levels (the weight of the x0 error at time t). */
export type DiffusionWeighting = 'simple' | 'uniform' | 'snr-plus-one' | 'truncated-snr' | 'min-snr' | 'elbo' | 'edm'

/** Options of {@link x0ErrorWeight} and {@link weightedDiffusionLoss}. */
export interface WeightingOptions {
  /** The cap of 'min-snr' (default 5). */
  readonly gamma?: number
  /** The data's standard deviation, for 'edm' (default 0.5). */
  readonly sigmaData?: number
}

/**
 * The weight of the clean-sample error ||x0_hat - x0||^2 at time t, in Kingma's formulation of the diffusion objective:
 * 'simple' is the SNR (which makes the loss DDPM's epsilon-MSE, L_simple), 'uniform' is 1, 'snr-plus-one' SNR + 1 (the
 * velocity-MSE, Salimans and Ho), 'truncated-snr' max(SNR, 1), 'min-snr' min(SNR, gamma) (Hang et al., 2023), 'elbo'
 * |d SNR / dt| / 2 (the continuous-time evidence lower bound, Kingma et al., 2021) and 'edm' the weighting of Karras et al.
 * with the effective noise sigma / alpha.
 *
 * @param schedule - The schedule.
 * @param t - The time.
 * @param weighting - The weighting.
 * @param options - gamma and sigmaData.
 * @returns The weight.
 * @concept diffusion-models
 * @example
 * ```ts
 * import { linearSchedule, snr, x0ErrorWeight } from 'omai-engine/stochastic'
 * const s = linearSchedule()
 * Math.abs(x0ErrorWeight(s, 0.4, 'simple') - snr(s, 0.4)) < 1e-12 // => true
 * ```
 */
export function x0ErrorWeight(
  schedule: NoiseSchedule,
  t: number,
  weighting: DiffusionWeighting,
  options: WeightingOptions = {},
): number {
  const ratio = snr(schedule, t)
  switch (weighting) {
    case 'simple':
      return ratio
    case 'uniform':
      return 1
    case 'snr-plus-one':
      return ratio + 1
    case 'truncated-snr':
      return Math.max(ratio, 1)
    case 'min-snr':
      return Math.min(ratio, options.gamma ?? 5)
    case 'elbo': {
      const sigma2 = schedule.sigma(t) ** 2
      // d SNR / dt = SNR (2 f - (d sigma^2 / dt) / sigma^2), with d sigma^2 / dt = g^2 + 2 f sigma^2.
      const dLambda = 2 * schedule.drift(t) - (schedule.diffusionSquared(t) + 2 * schedule.drift(t) * sigma2) / sigma2
      return 0.5 * ratio * Math.abs(dLambda)
    }
    case 'edm': {
      const sd = options.sigmaData ?? 0.5
      const effective = schedule.sigma(t) / schedule.alpha(t)
      return (effective * effective + sd * sd) / (effective * sd) ** 2
    }
  }
}

/** The factor from a clean-sample error to the squared error of the model's own output at x_t. */
function errorFactor(prediction: DiffusionPrediction, alpha: number, sigma: number): number {
  switch (prediction) {
    case 'x0':
      return 1
    case 'epsilon':
      return (sigma * sigma) / (alpha * alpha)
    case 'v':
      return sigma * sigma
    case 'score':
      return sigma ** 4 / (alpha * alpha)
  }
}

/**
 * The target of a prediction at x_t = alpha x0 + sigma epsilon, per sample.
 *
 * @param prediction - What is predicted.
 * @param x0 - Clean samples.
 * @param noise - The noise.
 * @param alpha - The signal scales, shaped to broadcast.
 * @param sigma - The noise scales, shaped to broadcast.
 * @returns The target.
 */
function targetOf(prediction: DiffusionPrediction, x0: Value, noise: Value, alpha: Value, sigma: Value): Value {
  switch (prediction) {
    case 'epsilon':
      return noise
    case 'x0':
      return x0
    case 'v':
      return sub(mul(alpha, noise), mul(sigma, x0))
    case 'score':
      return neg(div(noise, sigma))
  }
}

/** Options of {@link weightedDiffusionLoss}. */
export interface DiffusionLossOptions extends WeightingOptions {
  /** What the model predicts (default 'epsilon'). */
  readonly prediction?: DiffusionPrediction
  /** The weighting across noise levels (default 'simple': L_simple, whatever the prediction). */
  readonly weighting?: DiffusionWeighting
}

/**
 * The diffusion training loss with a weighting across noise levels: the mean over samples of w(t) times the mean
 * squared error of the model's output against its target, where w converts the chosen weighting of the clean-sample
 * error ({@link x0ErrorWeight}) to the output's own units, so every prediction trains the same objective: with
 * 'simple' the epsilon-MSE of DDPM whichever of epsilon, x0, velocity or score the network outputs.
 *
 * @param output - The model output at x_t `[N, ...]`.
 * @param x0 - Clean samples `[N, ...]`.
 * @param noise - The noise that made x_t.
 * @param times - The time of each sample `[N]`.
 * @param schedule - The schedule.
 * @param options - The prediction, the weighting and their parameters.
 * @returns The scalar loss, differentiable in the output.
 * @concept diffusion-models
 * @index function Diffusion training loss: simple, min-SNR, ELBO or EDM weighting.
 * @example
 * ```ts
 * import { linearSchedule, weightedDiffusionLoss } from 'omai-engine/stochastic'
 * import { item } from 'omai-engine/tensor'
 * item(weightedDiffusionLoss([[1, 2]], [[0, 0]], [[1, 2]], [0.5], linearSchedule())) // => 0
 * ```
 */
export function weightedDiffusionLoss(
  output: TensorLike,
  x0: TensorLike,
  noise: TensorLike,
  times: readonly number[],
  schedule: NoiseSchedule,
  options: DiffusionLossOptions = {},
): Value {
  const [o, x, e] = promoteFloat(output, x0, noise) as [Value, Value, Value]
  const prediction = options.prediction ?? 'epsilon'
  const weighting = options.weighting ?? 'simple'
  const alpha = column(
    times.map((t) => schedule.alpha(t)),
    x,
  )
  const sigma = column(
    times.map((t) => schedule.sigma(t)),
    x,
  )
  const weights = times.map(
    (t) =>
      x0ErrorWeight(schedule, t, weighting, options) * errorFactor(prediction, schedule.alpha(t), schedule.sigma(t)),
  )
  const perElement = square(sub(o, targetOf(prediction, x, e, alpha, sigma)))
  const perSample = mean(reshape(perElement, [x.shape[0] ?? 1, -1]), 1)
  return mean(mul(perSample, tensor(weights)))
}

/** A training batch: noisy samples with the times, noise and targets that make the loss. */
export interface DiffusionBatch {
  readonly xt: Value
  readonly noise: Value
  readonly x0: Value
  /** The time of each sample. */
  readonly times: readonly number[]
}

/**
 * A training batch: a time per sample (uniform over [tMin, tMax], or EDM's log-normal noise levels for the `edm`
 * schedule), and x_t from the forward process.
 *
 * @param schedule - The schedule.
 * @param x0 - Clean samples `[N, ...]`.
 * @param k - The random key.
 * @param options - tMin and tMax (default the schedule's) and `logNormal` to draw sigma as EDM does (only for a schedule
 *   whose time is sigma).
 * @returns The batch.
 * @index function A noised training batch for a diffusion model: times and x_t.
 * @example
 * ```ts
 * import { diffusionBatch, linearSchedule } from 'omai-engine/stochastic'
 * import { key } from 'omai-engine/random'
 * diffusionBatch(linearSchedule(), [[0, 0], [1, 1]], key(0)).times.length // => 2
 * ```
 */
export function diffusionBatch(
  schedule: NoiseSchedule,
  x0: TensorLike,
  k: Key,
  options: { readonly tMin?: number; readonly tMax?: number; readonly logNormal?: boolean } = {},
): DiffusionBatch {
  const [x] = promoteFloat(x0) as [Value]
  const n = x.shape[0] ?? 1
  const [kt, kn] = [key0(k, 0), key0(k, 1)]
  const [lo, hi] = [options.tMin ?? schedule.tMin, options.tMax ?? schedule.tMax]
  const times =
    options.logNormal === true ? edmNoiseLevels(kt, n) : Array.from(uniform(kt, [n]).data, (u) => lo + (hi - lo) * u)
  const f = forwardSample(schedule, x, times, kn)
  return { xt: f.xt, noise: f.noise, x0: x, times }
}

const key0 = (k: Key, i: number): Key => foldIn(k, i)

// --- Guidance ---------------------------------------------------------------------------------------------------------

/**
 * Classifier-free guidance (Ho and Salimans, 2021): extrapolate from the unconditional prediction toward the
 * conditional one, u + s (c - u). A scale of 1 is the conditional model; the paper's w is s - 1. Applies to any
 * prediction that is linear in the score (epsilon, score, and velocity).
 *
 * @param conditional - The conditional prediction.
 * @param unconditional - The unconditional prediction.
 * @param scale - The guidance scale s (default 1).
 * @returns The guided prediction.
 * @concept diffusion-models
 * @index function Combine conditional and unconditional predictions with a guidance scale.
 * @example
 * ```ts
 * import { classifierFreeGuidance } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(classifierFreeGuidance([2], [1], 3)) // => [4]
 * ```
 */
export function classifierFreeGuidance(conditional: TensorLike, unconditional: TensorLike, scale = 1): Value {
  const [c, u] = promoteFloat(conditional, unconditional) as [Value, Value]
  return add(u, mul(sub(c, u), scale))
}

/**
 * Classifier guidance in score form (Dhariwal and Nichol, 2021): the score plus the scaled gradient of a classifier's
 * log-probability of the label, grad_x log p(y | x_t), which tilts the marginal toward p(x_t) p(y | x_t)^s.
 *
 * @param score - The unconditional score.
 * @param classifierGradient - grad_x log p(y | x_t) from a classifier trained on noisy samples.
 * @param scale - The guidance scale (default 1).
 * @returns The guided score.
 * @example
 * ```ts
 * import { guidedScore } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(guidedScore([1], [2], 3)) // => [7]
 * ```
 */
export function guidedScore(score: TensorLike, classifierGradient: TensorLike, scale = 1): Value {
  const [s, g] = promoteFloat(score, classifierGradient) as [Value, Value]
  return add(s, mul(g, scale))
}

/**
 * Classifier guidance in noise form: epsilon_hat - s sigma_t grad_x log p(y | x_t) (the score is minus epsilon over sigma).
 *
 * @param noise - The unconditional noise prediction.
 * @param classifierGradient - grad_x log p(y | x_t).
 * @param sigma - The noise scale at the time.
 * @param scale - The guidance scale (default 1).
 * @returns The guided noise prediction.
 * @example
 * ```ts
 * import { guidedNoise } from 'omai-engine/stochastic'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(guidedNoise([1], [2], 0.5, 3)) // => [-2]
 * ```
 */
export function guidedNoise(noise: TensorLike, classifierGradient: TensorLike, sigma: number, scale = 1): Value {
  const [e, g] = promoteFloat(noise, classifierGradient) as [Value, Value]
  return sub(e, mul(g, sigma * scale))
}

/**
 * A classifier-free guided model from a conditional and an unconditional one (both predicting the same quantity).
 *
 * @param conditional - The conditional model.
 * @param unconditional - The unconditional model.
 * @param scale - The guidance scale.
 * @returns The guided model.
 * @index function Classifier-free guided model from a conditional and an unconditional one.
 * @example
 * ```ts
 * import { guidedModel } from 'omai-engine/stochastic'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(guidedModel(() => tensor([2]), () => tensor([1]), 3)(tensor([0]), 0.5)) // => [4]
 * ```
 */
export function guidedModel(conditional: DiffusionModel, unconditional: DiffusionModel, scale: number): DiffusionModel {
  return (x, t) => classifierFreeGuidance(conditional(x, t), unconditional(x, t), scale)
}

// --- A planted truth: a Gaussian mixture ----------------------------------------------------------------------------------

/** The mixture a {@link GaussianMixtureTruth} is made of: isotropic components. */
export interface GaussianMixtureSpec {
  /** The component means `[K][D]`. */
  readonly means: readonly (readonly number[])[]
  /** The component standard deviations: one for all or one each (default 0.3). */
  readonly stds?: number | readonly number[]
  /** The mixture weights (default equal), normalised. */
  readonly weights?: readonly number[]
}

/**
 * A Gaussian mixture data distribution whose noised marginals are Gaussian mixtures again, so the score of every
 * noise level is exact: under x_t = alpha x0 + sigma epsilon, component k becomes N(alpha mu_k, (alpha^2 s_k^2 +
 * sigma^2) I). A sampler run with this score must reproduce the data, which verifies the sampler independently of
 * any network, and a trained network can be compared with it.
 */
export interface GaussianMixtureTruth {
  readonly dim: number
  readonly components: number
  /** Draws `n` points and their component. */
  sample(k: Key, n: number): { readonly x: Tensor; readonly component: readonly number[] }
  /** log p_t(x), the log density of the noised marginal, per row. */
  logDensity(x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value
  /** grad_x log p_t(x), the exact score, per row. */
  score(x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value
  /** The posterior probability of each component given x_t `[N, K]`. */
  responsibilities(x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value
  /** grad_x log p_t(x | component k): the score of one component alone. */
  componentScore(k: number, x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value
  /** The exact model in a parameterisation, for a sampler. */
  model(schedule: NoiseSchedule, prediction: DiffusionPrediction): DiffusionModel
  /** The exact model conditioned on one component (for guidance). */
  conditionalModel(k: number, schedule: NoiseSchedule, prediction: DiffusionPrediction): DiffusionModel
}

/**
 * The exact noised score of an isotropic Gaussian mixture. See {@link GaussianMixtureTruth}.
 *
 * @param spec - Means, standard deviations and weights.
 * @returns The truth.
 * @throws OmaiError with code `value` for no components or non-positive spreads or weights.
 * @concept diffusion-models
 * @index function Exact noised score of a Gaussian mixture, to verify samplers and networks.
 * @example
 * ```ts
 * import { gaussianMixtureTruth, linearSchedule } from 'omai-engine/stochastic'
 * import { item, sum } from 'omai-engine/tensor'
 * // One standard normal: the score at x is -x at every noise level.
 * const t = gaussianMixtureTruth({ means: [[0]], stds: 1 })
 * Math.abs(item(sum(t.score([[2]], linearSchedule(), 0.5))) + 2) < 1e-12 // => true
 * ```
 */
export function gaussianMixtureTruth(spec: GaussianMixtureSpec): GaussianMixtureTruth {
  const k = spec.means.length
  const dim = spec.means[0]?.length ?? 0
  if (k === 0 || dim === 0) throw new OmaiError('value', 'gaussianMixtureTruth: need at least one component')
  const stds = Array.from({ length: k }, (_, i) =>
    typeof spec.stds === 'number' ? spec.stds : (spec.stds?.[i] ?? 0.3),
  )
  const raw = spec.weights ?? new Array<number>(k).fill(1)
  if (stds.some((s) => !(s > 0)) || raw.some((w) => !(w > 0)))
    throw new OmaiError('value', 'gaussianMixtureTruth: spreads and weights must be positive')
  const total = raw.reduce((a, b) => a + b, 0)
  const logWeights = raw.map((w) => Math.log(w / total))
  const means = tensor(spec.means.map((m) => [...m]))
  const LOG_2PI = Math.log(2 * Math.PI)

  /** Per-component log N(x; alpha mu_k, v_k I) `[N, K]` and the variances `[N, K]`, for alpha and sigma per row. */
  const parts = (
    x: Value,
    schedule: NoiseSchedule,
    t: number | readonly number[],
  ): { logp: Value; v: Value; diff: Value } => {
    const n = x.shape[0] ?? 1
    const times = typeof t === 'number' ? new Array<number>(n).fill(t) : [...t]
    const alpha = reshape(tensor(times.map((u) => schedule.alpha(u))), [n, 1])
    const sig2 = reshape(tensor(times.map((u) => schedule.sigma(u) ** 2)), [n, 1])
    // v[n, k] = alpha^2 s_k^2 + sigma^2
    const v = add(mul(square(alpha), tensor([stds.map((s) => s * s)])), sig2)
    // diff[n, k, d] = x[n, d] - alpha mu[k, d]
    const diff = sub(reshape(x, [n, 1, dim]), mul(reshape(alpha, [n, 1, 1]), reshape(means, [1, k, dim])))
    const sq = sum(square(diff), -1)
    const logp = sub(add(mul(log(v), -0.5 * dim), mul(div(sq, v), -0.5)), 0.5 * dim * LOG_2PI)
    return { logp: add(logp, tensor([logWeights])), v, diff }
  }

  const responsibilities = (x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value =>
    softmax(parts(promoteFloat(x)[0] as Value, schedule, t).logp, -1)

  const scoreOf = (x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value => {
    const { logp, v, diff } = parts(promoteFloat(x)[0] as Value, schedule, t)
    const r = softmax(logp, -1)
    const n = diff.shape[0] ?? 1
    // sum_k r_k * (-(x - alpha mu_k) / v_k)
    return neg(sum(mul(reshape(div(r, v), [n, k, 1]), diff), 1))
  }

  const componentScore = (c: number, x: TensorLike, schedule: NoiseSchedule, t: number | readonly number[]): Value => {
    const { v, diff } = parts(promoteFloat(x)[0] as Value, schedule, t)
    const n = diff.shape[0] ?? 1
    const pick = tensor(Array.from({ length: k }, (_, i) => (i === c ? 1 : 0)))
    return neg(sum(mul(reshape(mul(div(1, v), pick), [n, k, 1]), diff), 1))
  }

  const asModel =
    (
      scoreFn: (x: Value, t: number) => Value,
      schedule: NoiseSchedule,
      prediction: DiffusionPrediction,
    ): DiffusionModel =>
    (xt, t) => {
      const score = scoreFn(xt, t)
      const [a, s] = [schedule.alpha(t), schedule.sigma(t)]
      switch (prediction) {
        case 'score':
          return score
        case 'epsilon':
          return mul(score, -s)
        case 'x0':
          return div(add(xt, mul(score, s * s)), a)
        case 'v':
          return sub(mul(mul(score, -s), a), mul(div(add(xt, mul(score, s * s)), a), s))
      }
    }

  return {
    dim,
    components: k,
    sample: (key, n) => {
      const u = uniform(key0(key, 0), [n]).data
      const cumulative = raw.map((_, i) => raw.slice(0, i + 1).reduce((a, b) => a + b, 0) / total)
      const component = Array.from(u, (v) =>
        Math.min(
          k - 1,
          cumulative.findIndex((c) => v <= c),
        ),
      )
      const z = normal(key0(key, 1), [n, dim]).data
      const data = Float64Array.from(z, (value, i) => {
        const c = component[Math.floor(i / dim)] ?? 0
        return (spec.means[c]?.[i % dim] ?? 0) + (stds[c] ?? 0) * value
      })
      return {
        x: tensor(Array.from({ length: n }, (_, r) => Array.from(data.slice(r * dim, (r + 1) * dim)))),
        component,
      }
    },
    logDensity: (x, schedule, t) => logsumexp(parts(promoteFloat(x)[0] as Value, schedule, t).logp, -1),
    score: scoreOf,
    responsibilities,
    componentScore,
    model: (schedule, prediction) => asModel((x, t) => scoreOf(x, schedule, t), schedule, prediction),
    conditionalModel: (c, schedule, prediction) =>
      asModel((x, t) => componentScore(c, x, schedule, t), schedule, prediction),
  }
}
