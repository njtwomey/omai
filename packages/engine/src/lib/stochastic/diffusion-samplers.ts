/**
 * The reverse process of a diffusion model as stepped algorithms, and the network that is trained to run it: the DDPM
 * ancestral sampler, DDIM (deterministic or with noise eta), the probability-flow ODE with Euler and Heun steps, and
 * DPM-Solver-2, each an `Algorithm` whose state (the sample cloud, the prediction of the clean sample, the noise
 * level) serialises and logs on typed paths; a small network over (x_t, time features) for low-dimensional data, and the
 * `diffusionTrainer` that trains it, so a full train-and-sample loop runs on a toy mixture.
 */

import type { Json } from '../base/index.ts'
import { MdmlError } from '../base/index.ts'
import { pointsSteps } from '../exposition/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { add, div, maximum, minimum, mul, sub, type Tensor, toArray, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  Log,
  type LogSpec,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import {
  type DiffusionModel,
  type DiffusionPrediction,
  type NoiseSchedule,
  samplerTimes,
  type TimeSpacing,
  timeAtLogSnr,
  toX0AndNoise,
} from './diffusion-models.ts'

/** The reverse samplers. */
export type SamplerMethod = 'ddpm' | 'ddim' | 'euler' | 'heun' | 'dpm2'

/** The state of a reverse sampler. */
export interface SamplerState {
  /** The sample cloud x_t `[N, ...]`. */
  readonly x: Tensor
  /** The latest estimate of the clean sample (before the first step: the cloud itself). */
  readonly x0Hat: Tensor
  /** Steps taken. */
  readonly step: number
  /** The time of x. */
  readonly t: number
}

/** The paths a sampler logs. */
export type DiffusionSamplerPath =
  | 'sampler/xt'
  | 'sampler/x0Hat'
  | 'sampler/noiseLevel'
  | 'sampler/time'
  | 'sampler/step'
  | 'sampler/cloud'

/** The contract of a reverse sampler: the cloud, the clean-sample estimate, the noise level, the time and the step. */
export const diffusionSamplerContract: Contract<SamplerState, DiffusionSamplerPath> = defineContract({
  'sampler/xt': tensorSource({ role: 'state', description: 'the sample cloud x_t after the step' }),
  'sampler/x0Hat': tensorSource({
    role: 'output',
    description: "the model's estimate of the clean samples at the step",
  }),
  'sampler/noiseLevel': scalar({ description: 'sigma(t) of the cloud after the step', direction: 'lower' }),
  'sampler/time': scalar({ description: 'the diffusion time t after the step', direction: 'lower' }),
  'sampler/step': scalar({ description: 'the steps taken' }),
  'sampler/cloud': derived<SamplerState>({
    description: 'the cloud as points (the first two coordinates) at the step',
    compute: (s) => {
      const rows = toArray(s.x) as number[][]
      return pointsSteps({
        dims: 2,
        accumulate: false,
        steps: [{ step: s.step, points: rows.map((r) => [r[0] ?? 0, r[1] ?? 0]) }],
      }) as unknown as Json
    },
  }),
})

/** What a sampler's paths keep by default. */
export const diffusionSamplerDefaults: Readonly<Record<DiffusionSamplerPath, LogSpec>> = {
  'sampler/xt': Log(),
  'sampler/x0Hat': Log(),
  'sampler/noiseLevel': Log(),
  'sampler/time': Log(),
  'sampler/step': Log(),
  'sampler/cloud': Log({ every: 10 }),
}

/** Options of {@link diffusionSampler}. */
export interface DiffusionSamplerOptions {
  readonly schedule: NoiseSchedule
  /** The model: a prediction of the given kind for a noisy batch at a time. */
  readonly model: DiffusionModel
  /** What the model predicts (default 'epsilon'). */
  readonly prediction?: DiffusionPrediction
  /** The shape of the cloud `[N, ...]`. */
  readonly shape: readonly number[]
  /** The number of steps (model evaluations: one per step, two for 'heun' and 'dpm2'). */
  readonly steps: number
  /** The sampler (default 'ddim'). */
  readonly method?: SamplerMethod
  /** The noise of 'ddim' as a share of DDPM's posterior noise (default 0: deterministic; 1 is DDPM). */
  readonly eta?: number
  /** The times (default `n + 1` points spaced uniformly, ending at zero noise); see `samplerTimes`. */
  readonly times?: readonly number[]
  readonly spacing?: TimeSpacing
  /** Clip the clean-sample estimate to this range at every step (as DDPM implementations do for images). */
  readonly clipX0?: readonly [number, number]
  readonly name?: string
}

const sq = (x: number): number => x * x

/**
 * A reverse sampler as an {@link Algorithm}. All five start from x_T = sigma(tMax) epsilon and visit the times of
 * `samplerTimes` down to zero noise:
 *
 * - 'ddpm': ancestral sampling from the posterior q(x_s | x_t, x0_hat) of the forward process (Ho et al., 2020), whose
 *   mean is a combination of the model's clean-sample and noise estimates and whose variance is the posterior's.
 * - 'ddim': x_s = alpha_s x0_hat + sqrt(sigma_s^2 - eta^2 v) epsilon_hat + eta sqrt(v) z with v the posterior variance
 *   (Song et al., 2021); eta = 0 is a deterministic map from noise to sample, eta = 1 is DDPM.
 * - 'euler' and 'heun': the probability-flow ODE dx/dt = f(t) x - g(t)^2 score / 2, which has the same marginals as the
 *   SDE, by Euler's method and by Heun's second-order method (Karras et al., 2022; the last step is Euler).
 * - 'dpm2': DPM-Solver-2 (Lu et al., 2022), a second-order exponential integrator in the log-SNR with a midpoint
 *   evaluation (the last step, to zero noise, is DDIM).
 *
 * @param options - Schedule, model and what it predicts, shape, steps, method and its parameters.
 * @returns The algorithm; its paths are {@link DiffusionSamplerPath}.
 * @throws MdmlError with code `value` for an unknown method or times that do not decrease to zero.
 * @concept diffusion-models
 * @example
 * ```ts
 * import { diffusionSampler, gaussianMixtureTruth, linearSchedule } from 'mdml-engine/stochastic'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const schedule = linearSchedule()
 * const truth = gaussianMixtureTruth({ means: [[3]], stds: 0.5 })
 * const sampler = diffusionSampler({ schedule, model: truth.model(schedule, 'epsilon'), shape: [2, 1], steps: 20 })
 * runAlgorithm(sampler, { maxSteps: 20, seed: 1 }).state.step // => 20
 * ```
 */
export function diffusionSampler(options: DiffusionSamplerOptions): Algorithm<SamplerState, DiffusionSamplerPath> {
  const { schedule, model } = options
  const prediction = options.prediction ?? 'epsilon'
  const method = options.method ?? 'ddim'
  const eta = method === 'ddpm' ? 1 : (options.eta ?? 0)
  const times =
    options.times ??
    samplerTimes(schedule, options.steps, options.spacing === undefined ? {} : { spacing: options.spacing })
  const steps = times.length - 1
  if (steps < 1 || times[steps] !== 0 || times.some((t, i) => i > 0 && !(t < (times[i - 1] ?? 0))))
    throw new MdmlError('value', 'diffusionSampler: times must decrease to zero')
  const clip = options.clipX0

  /** The model's clean-sample and noise estimates at (x, t). */
  const estimate = (x: Value, t: number): { x0: Value; noise: Value } => {
    const [a, s] = [schedule.alpha(t), schedule.sigma(t)]
    const e = toX0AndNoise(prediction, model(x, t), x, a, s)
    if (clip === undefined) return e
    const x0 = clipTo(e.x0, clip)
    return { x0, noise: div(sub(x, mul(x0, a)), s) }
  }

  /** The DDIM/DDPM move from t to s given the estimates. */
  const ancestral = (x0: Value, noise: Value, t: number, s: number, key: Key, shape: readonly number[]): Value => {
    const [at, st, as, ss] = [schedule.alpha(t), schedule.sigma(t), schedule.alpha(s), schedule.sigma(s)]
    const transition = sq(at / as)
    const posterior = st > 0 ? ((sq(st) - transition * sq(ss)) * sq(ss)) / sq(st) : 0
    const noiseScale = eta * Math.sqrt(Math.max(posterior, 0))
    const direction = Math.sqrt(Math.max(sq(ss) - sq(noiseScale), 0))
    let out = add(mul(x0, as), mul(noise, direction))
    if (noiseScale > 0) out = add(out, mul(normal(key, [...shape]), noiseScale))
    return out
  }

  /** The probability-flow drift f(t) x - g^2(t) score / 2 with the score -epsilon / sigma. */
  const flow = (x: Value, noise: Value, t: number): Value =>
    add(mul(x, schedule.drift(t)), mul(noise, (0.5 * schedule.diffusionSquared(t)) / schedule.sigma(t)))

  return {
    name: options.name ?? `diffusion-${method}`,
    config: { method, eta, steps, schedule: schedule.name, prediction },
    init: (key) => {
      const x = mul(normal(foldIn(key, 0), [...options.shape]), schedule.sigma(times[0] ?? schedule.tMax)) as Tensor
      return { x, x0Hat: x, step: 0, t: times[0] ?? schedule.tMax }
    },
    step: (state, ctx) => {
      const i = state.step
      const [t, s] = [times[i] ?? 0, times[i + 1] ?? 0]
      const x = state.x
      const first = estimate(x, t)
      let next: Value
      switch (method) {
        case 'ddpm':
        case 'ddim':
          next = ancestral(first.x0, first.noise, t, s, ctx.key, options.shape)
          break
        case 'euler':
          next = add(x, mul(flow(x, first.noise, t), s - t))
          break
        case 'heun': {
          const d1 = flow(x, first.noise, t)
          const predictor = add(x, mul(d1, s - t))
          if (s === 0) next = predictor
          else {
            const d2 = flow(predictor, estimate(predictor, s).noise, s)
            next = add(x, mul(add(d1, d2), 0.5 * (s - t)))
          }
          break
        }
        case 'dpm2': {
          if (s === 0) {
            next = ancestral(first.x0, first.noise, t, s, ctx.key, options.shape)
            break
          }
          const [lt, ls] = [schedule.logSnr(t) / 2, schedule.logSnr(s) / 2]
          const h = ls - lt
          const mid = timeAtLogSnr(schedule, 2 * (lt + 0.5 * h))
          const [at, as, am, ss, sm] = [
            schedule.alpha(t),
            schedule.alpha(s),
            schedule.alpha(mid),
            schedule.sigma(s),
            schedule.sigma(mid),
          ]
          const u = sub(mul(x, am / at), mul(first.noise, sm * Math.expm1(0.5 * h)))
          const noiseMid = estimate(u, mid).noise
          next = sub(
            sub(mul(x, as / at), mul(first.noise, ss * Math.expm1(h))),
            mul(sub(noiseMid, first.noise), ss * Math.expm1(h)),
          )
          break
        }
        default:
          throw new MdmlError('value', `diffusionSampler: unknown method ${String(method)}`)
      }
      const nextState: SamplerState = {
        x: next as Tensor,
        x0Hat: first.x0 as Tensor,
        step: i + 1,
        t: s,
      }
      ctx.log('sampler/xt', next)
      ctx.log('sampler/x0Hat', first.x0)
      ctx.log('sampler/noiseLevel', schedule.sigma(s))
      ctx.log('sampler/time', s)
      ctx.log('sampler/step', i + 1)
      return nextState
    },
    done: (state) => (state.step >= steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SamplerState,
  }
}

/** Clip a value to a range (a smooth-free clamp, with zero gradient outside). */
function clipTo(x: Value, range: readonly [number, number]): Value {
  const lower = mul(add(mul(x, 0), 1), range[0])
  const upper = mul(add(mul(x, 0), 1), range[1])
  return minimum(maximum(x, lower), upper)
}

/**
 * DDPM ancestral sampling: {@link diffusionSampler} with the method 'ddpm'.
 *
 * @param options - As {@link diffusionSampler}, without a method.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { ddpmSampler, linearSchedule } from 'mdml-engine/stochastic'
 * import { zeros } from 'mdml-engine/tensor'
 * ddpmSampler({ schedule: linearSchedule(), model: (x) => zeros(x.shape), shape: [1, 1], steps: 5 }).name // => 'diffusion-ddpm'
 * ```
 */
export function ddpmSampler(
  options: Omit<DiffusionSamplerOptions, 'method' | 'eta'>,
): Algorithm<SamplerState, DiffusionSamplerPath> {
  return diffusionSampler({ ...options, method: 'ddpm' })
}

/**
 * DDIM: {@link diffusionSampler} with the method 'ddim' and the given eta (0, a deterministic map, by default).
 *
 * @param options - As {@link diffusionSampler}, without a method.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { ddimSampler, linearSchedule } from 'mdml-engine/stochastic'
 * import { zeros } from 'mdml-engine/tensor'
 * ddimSampler({ schedule: linearSchedule(), model: (x) => zeros(x.shape), shape: [1, 1], steps: 5 }).name // => 'diffusion-ddim'
 * ```
 */
export function ddimSampler(
  options: Omit<DiffusionSamplerOptions, 'method'>,
): Algorithm<SamplerState, DiffusionSamplerPath> {
  return diffusionSampler({ ...options, method: 'ddim' })
}

/**
 * Run a sampler to the end.
 *
 * @param options - As {@link diffusionSampler}.
 * @param key - The random key (default key 0).
 * @returns The final cloud (at zero noise) and the last clean-sample estimate.
 * @example
 * ```ts
 * import { gaussianMixtureTruth, linearSchedule, sampleDiffusion } from 'mdml-engine/stochastic'
 * const schedule = linearSchedule()
 * const truth = gaussianMixtureTruth({ means: [[3]], stds: 0.1 })
 * const { x } = sampleDiffusion({ schedule, model: truth.model(schedule, 'epsilon'), shape: [50, 1], steps: 50 })
 * x.shape // => [50, 1]
 * ```
 */
export function sampleDiffusion(
  options: DiffusionSamplerOptions,
  key?: Key,
): { readonly x: Tensor; readonly x0Hat: Tensor } {
  const algorithm = diffusionSampler(options)
  const result = runAlgorithm(algorithm, {
    maxSteps: options.steps + 1,
    ...(key === undefined ? { seed: 0 } : { key }),
  })
  return { x: result.state.x, x0Hat: result.state.x0Hat }
}

// --- The samplers as a slot -------------------------------------------------------------------------------------------

/** What a sampler is, for validity rules and selectors. */
export interface SamplerProperties {
  /** Whether it injects noise at every step (a different draw each run from one start). */
  readonly stochastic: boolean
  /** The order of its local error in the step. */
  readonly order: 1 | 2
  /** Model evaluations per step. */
  readonly evaluationsPerStep: 1 | 2
}

/** The samplers' properties by method. */
export const samplerProperties: Readonly<Record<SamplerMethod, SamplerProperties>> = {
  ddpm: { stochastic: true, order: 1, evaluationsPerStep: 1 },
  ddim: { stochastic: false, order: 1, evaluationsPerStep: 1 },
  euler: { stochastic: false, order: 1, evaluationsPerStep: 1 },
  heun: { stochastic: false, order: 2, evaluationsPerStep: 2 },
  dpm2: { stochastic: false, order: 2, evaluationsPerStep: 2 },
}
