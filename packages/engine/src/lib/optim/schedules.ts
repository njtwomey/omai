/**
 * Learning-rate schedules, matching optax and torch.optim.lr_scheduler: constant, step and multi-step decay,
 * exponential, inverse-time, polynomial and cosine decay, warm-up variants, warm restarts (SGDR), piecewise and joined
 * schedules, the one-cycle policy, cyclical rates, and `reduceOnPlateau`, which is driven by a metric rather than the
 * step count.
 */

import { OmaiError } from '../base/index.ts'

/** A learning-rate schedule: the rate at each step count (from 0). */
export type LearningRateSchedule = (step: number) => number

/**
 * A constant schedule (optax.constant_schedule).
 *
 * @param value - The value.
 * @returns The schedule.
 * @example
 * ```ts
 * import { constantSchedule } from 'omai-engine/optim'
 * constantSchedule(0.1)(100) // => 0.1
 * ```
 */
export function constantSchedule(value: number): LearningRateSchedule {
  return () => value
}

/**
 * A constant factor for the first `steps` steps, then the initial value (torch.optim.lr_scheduler.ConstantLR).
 *
 * @param init - The base value.
 * @param factor - The factor applied at first.
 * @param steps - How long the factor applies (torch's total_iters).
 * @returns The schedule.
 * @example
 * ```ts
 * import { constantFactor } from 'omai-engine/optim'
 * [constantFactor(1, 0.5, 2)(1), constantFactor(1, 0.5, 2)(2)] // => [0.5, 1]
 * ```
 */
export function constantFactor(init: number, factor: number, steps: number): LearningRateSchedule {
  return (t) => (t < steps ? init * factor : init)
}

/**
 * Step decay: init * gamma^floor(t / stepSize) (torch.optim.lr_scheduler.StepLR).
 *
 * @param init - Initial value.
 * @param stepSize - Steps between decays.
 * @param gamma - Factor per decay.
 * @returns The schedule.
 * @example
 * ```ts
 * import { stepDecay } from 'omai-engine/optim'
 * stepDecay(1, 10, 0.5)(25) // => 0.25
 * ```
 */
export function stepDecay(init: number, stepSize: number, gamma = 0.1): LearningRateSchedule {
  return (t) => init * gamma ** Math.floor(t / stepSize)
}

/**
 * Multi-step decay: init * gamma^(number of milestones reached) (torch.optim.lr_scheduler.MultiStepLR).
 *
 * @param init - Initial value.
 * @param milestones - Increasing step counts.
 * @param gamma - Factor per milestone.
 * @returns The schedule.
 * @example
 * ```ts
 * import { multiStepDecay } from 'omai-engine/optim'
 * multiStepDecay(1, [5, 8], 0.5)(8) // => 0.25
 * ```
 */
export function multiStepDecay(init: number, milestones: readonly number[], gamma = 0.1): LearningRateSchedule {
  return (t) => init * gamma ** milestones.filter((m) => t >= m).length
}

/** Optional parts of {@link exponentialDecay} (optax.exponential_decay). */
export interface ExponentialDecayOptions {
  /** Hold the initial value for this many steps (default 0). */
  readonly transitionBegin?: number
  /** A floor (decay rate below 1) or ceiling (above 1) on the value. */
  readonly endValue?: number
}

/**
 * Exponential decay: init * rate^((t - begin) / steps), optionally in whole steps (staircase) and clipped at an end
 * value (optax.exponential_decay; torch ExponentialLR is `exponentialDecay(init, gamma, 1)`).
 *
 * @param init - Initial value.
 * @param decayRate - Factor per `steps` steps.
 * @param steps - Steps per factor.
 * @param staircase - Decay in discrete jumps.
 * @param options - Transition start and end value.
 * @returns The schedule.
 * @example
 * ```ts
 * import { exponentialDecay } from 'omai-engine/optim'
 * exponentialDecay(1, 0.5, 10)(20) // => 0.25
 * ```
 */
export function exponentialDecay(
  init: number,
  decayRate: number,
  steps: number,
  staircase = false,
  options: ExponentialDecayOptions = {},
): LearningRateSchedule {
  const begin = options.transitionBegin ?? 0
  const end = options.endValue
  return (t) => {
    const count = t - begin
    const p = staircase ? Math.floor(count / steps) : count / steps
    const value = count <= 0 ? init : init * decayRate ** p
    if (end === undefined) return value
    return decayRate < 1 ? Math.max(value, end) : Math.min(value, end)
  }
}

/**
 * Inverse-time decay: init / (1 + rate t / steps), optionally in whole steps (Keras InverseTimeDecay; the classic
 * Robbins-Monro 1/t rate).
 *
 * @param init - Initial value.
 * @param decayRate - The rate.
 * @param steps - The time scale.
 * @param staircase - Decay in discrete jumps.
 * @returns The schedule.
 * @example
 * ```ts
 * import { inverseTimeDecay } from 'omai-engine/optim'
 * inverseTimeDecay(1, 0.5, 10)(20) // => 0.5
 * ```
 */
export function inverseTimeDecay(init: number, decayRate: number, steps = 1, staircase = false): LearningRateSchedule {
  return (t) => init / (1 + decayRate * (staircase ? Math.floor(t / steps) : t / steps))
}

/**
 * Inverse-square-root decay: init / sqrt(1 + rate t / steps), the slower rate of the transformer schedule after its
 * warmup (Vaswani et al., 2017) and of Adagrad-like methods.
 *
 * @param init - Initial value.
 * @param decayRate - The rate (default 1).
 * @param steps - The time scale (default 1).
 * @returns The schedule.
 * @index function Inverse-square-root decay of a learning rate.
 * @example
 * ```ts
 * import { inverseSqrtDecay } from 'omai-engine/optim'
 * inverseSqrtDecay(1, 1)(3) // => 0.5
 * inverseSqrtDecay(2, 1, 10)(30) // => 1
 * ```
 */
export function inverseSqrtDecay(init: number, decayRate = 1, steps = 1): LearningRateSchedule {
  return (t) => init / Math.sqrt(1 + (decayRate * t) / steps)
}

/**
 * The step-size decay of ADVI (Kucukelbir et al. 2017): init (t + 1)^(-1/2 + epsilon), a little slower than the inverse
 * square root so that the Robbins-Monro conditions hold. With {@link advi}'s adaptive scaling it is the whole sequence
 * eta k^(-1/2 + epsilon) / (tau + sqrt(s_k)) of Stan's ADVI.
 *
 * @param init - The first value (eta).
 * @param epsilon - The small number added to the exponent (default 1e-16, Stan's).
 * @returns The schedule, with step 0 the first.
 * @index function ADVI's step-size decay k^(-1/2 + epsilon).
 * @example
 * ```ts
 * import { adviDecay } from 'omai-engine/optim'
 * adviDecay(2)(3) // => 1
 * ```
 */
export function adviDecay(init: number, epsilon = 1e-16): LearningRateSchedule {
  return (t) => init * (t + 1) ** (-0.5 + epsilon)
}

/**
 * Cosine decay from init to alpha init over `decaySteps`, constant afterwards (optax.cosine_decay_schedule with its
 * exponent; torch CosineAnnealingLR up to T_max).
 *
 * @param init - Initial value.
 * @param decaySteps - Length of the decay.
 * @param alpha - Final value as a fraction of init.
 * @param exponent - Power of the cosine factor (default 1).
 * @returns The schedule.
 * @example
 * ```ts
 * import { cosineDecay } from 'omai-engine/optim'
 * cosineDecay(1, 100)(50) // => 0.5
 * ```
 */
export function cosineDecay(init: number, decaySteps: number, alpha = 0, exponent = 1): LearningRateSchedule {
  return (t) => {
    const p = Math.min(t, decaySteps) / decaySteps
    return init * ((1 - alpha) * (0.5 * (1 + Math.cos(Math.PI * p))) ** exponent + alpha)
  }
}

/**
 * Cosine annealing with warm restarts (Loshchilov and Hutter 2017; torch CosineAnnealingWarmRestarts): cycles of
 * lengths t0, t0 tMult, t0 tMult^2, ..., each annealing from init to etaMin.
 *
 * @param init - The value at the start of each cycle.
 * @param t0 - Length of the first cycle.
 * @param tMult - Growth factor of the cycle length (a positive integer).
 * @param etaMin - The value at the end of each cycle.
 * @returns The schedule.
 * @example
 * ```ts
 * import { cosineWarmRestarts } from 'omai-engine/optim'
 * [cosineWarmRestarts(1, 10)(5), cosineWarmRestarts(1, 10)(10)] // => [0.5, 1]
 * ```
 */
export function cosineWarmRestarts(init: number, t0: number, tMult = 1, etaMin = 0): LearningRateSchedule {
  return (t) => {
    let cur: number
    let length: number
    if (tMult === 1) {
      cur = t % t0
      length = t0
    } else {
      const n = Math.floor(Math.log((t / t0) * (tMult - 1) + 1) / Math.log(tMult))
      cur = t - (t0 * (tMult ** n - 1)) / (tMult - 1)
      length = t0 * tMult ** n
    }
    return etaMin + ((init - etaMin) * (1 + Math.cos((Math.PI * cur) / length))) / 2
  }
}

/** One cycle of {@link sgdr}: a {@link warmupCosineDecay} whose `totalSteps` is the cycle's length. */
export interface SgdrCycle {
  readonly start: number
  readonly peak: number
  readonly end: number
  readonly warmupSteps: number
  readonly totalSteps: number
  readonly exponent?: number
}

/**
 * Stochastic gradient descent with warm restarts as explicit cycles (Loshchilov and Hutter 2017; optax.sgdr_schedule):
 * each cycle warms up and decays by cosine, and the next starts where it ends.
 *
 * @param cycles - The cycles in order.
 * @returns The schedule.
 * @example
 * ```ts
 * import { sgdr } from 'omai-engine/optim'
 * const s = sgdr([{ start: 0, peak: 1, end: 0, warmupSteps: 0, totalSteps: 10 }, { start: 0, peak: 0.5, end: 0, warmupSteps: 0, totalSteps: 10 }])
 * [s(5), s(10)] // => [0.5, 0.5]
 * ```
 */
export function sgdr(cycles: readonly SgdrCycle[]): LearningRateSchedule {
  const boundaries: number[] = []
  let at = 0
  for (const c of cycles) {
    at += c.totalSteps
    boundaries.push(at)
  }
  return joinSchedules(
    cycles.map((c) => warmupCosineDecay(c)),
    boundaries.slice(0, -1),
  )
}

/**
 * Linear interpolation from `init` to `end` over `steps`, constant afterwards (optax.linear_schedule; torch LinearLR
 * is `linearSchedule(lr * start, lr * end, steps)`).
 *
 * @param init - Initial value.
 * @param end - Final value.
 * @param steps - Length of the transition.
 * @param begin - Hold the initial value for this many steps (optax's transition_begin).
 * @returns The schedule.
 * @example
 * ```ts
 * import { linearSchedule } from 'omai-engine/optim'
 * linearSchedule(1, 0, 4)(1) // => 0.75
 * ```
 */
export function linearSchedule(init: number, end: number, steps: number, begin = 0): LearningRateSchedule {
  return (t) => init + (end - init) * Math.min(1, Math.max(0, t - begin) / steps)
}

/**
 * Polynomial decay: (init - end) (1 - t / steps)^power + end, constant afterwards (optax.polynomial_schedule; torch
 * PolynomialLR is `polynomialDecay(lr, 0, power, totalIters)`).
 *
 * @param init - Initial value.
 * @param end - Final value.
 * @param power - The power.
 * @param steps - Length of the decay.
 * @param begin - Hold the initial value for this many steps.
 * @returns The schedule.
 * @example
 * ```ts
 * import { polynomialDecay } from 'omai-engine/optim'
 * polynomialDecay(1, 0, 2, 10)(5) // => 0.25
 * ```
 */
export function polynomialDecay(
  init: number,
  end: number,
  power: number,
  steps: number,
  begin = 0,
): LearningRateSchedule {
  return (t) => {
    const count = Math.min(Math.max(t - begin, 0), steps)
    return (init - end) * (1 - count / steps) ** power + end
  }
}

/**
 * Linear warm-up from `start` to `peak` over `warmupSteps`, then cosine decay to `end` by `totalSteps`
 * (optax.warmup_cosine_decay_schedule).
 *
 * @param options - Start, peak and end values, warm-up and total steps, and the cosine exponent.
 * @returns The schedule.
 * @example
 * ```ts
 * import { warmupCosineDecay } from 'omai-engine/optim'
 * warmupCosineDecay({ start: 0, peak: 1, end: 0, warmupSteps: 10, totalSteps: 110 })(10) // => 1
 * ```
 */
export function warmupCosineDecay(options: {
  readonly start: number
  readonly peak: number
  readonly end: number
  readonly warmupSteps: number
  readonly totalSteps: number
  readonly exponent?: number
}): LearningRateSchedule {
  const { start, peak, end, warmupSteps, totalSteps } = options
  const alpha = peak === 0 ? 0 : end / peak
  return joinSchedules(
    [
      linearSchedule(start, peak, warmupSteps),
      cosineDecay(peak, totalSteps - warmupSteps, alpha, options.exponent ?? 1),
    ],
    [warmupSteps],
  )
}

/**
 * Linear warm-up to `peak`, then exponential decay (optax.warmup_exponential_decay_schedule).
 *
 * @param options - Start and peak values, warm-up length and the {@link exponentialDecay} parameters.
 * @returns The schedule.
 * @example
 * ```ts
 * import { warmupExponentialDecay } from 'omai-engine/optim'
 * warmupExponentialDecay({ start: 0, peak: 1, warmupSteps: 5, decayRate: 0.5, transitionSteps: 10 })(15) // => 0.5
 * ```
 */
export function warmupExponentialDecay(options: {
  readonly start: number
  readonly peak: number
  readonly warmupSteps: number
  readonly decayRate: number
  readonly transitionSteps: number
  readonly transitionBegin?: number
  readonly staircase?: boolean
  readonly endValue?: number
}): LearningRateSchedule {
  const decay = exponentialDecay(options.peak, options.decayRate, options.transitionSteps, options.staircase ?? false, {
    ...(options.transitionBegin === undefined ? {} : { transitionBegin: options.transitionBegin }),
    ...(options.endValue === undefined ? {} : { endValue: options.endValue }),
  })
  return joinSchedules([linearSchedule(options.start, options.peak, options.warmupSteps), decay], [options.warmupSteps])
}

/**
 * Linear warm-up to `peak`, then constant (optax.warmup_constant_schedule).
 *
 * @param start - The initial value.
 * @param peak - The value after warm-up.
 * @param warmupSteps - Length of the warm-up.
 * @returns The schedule.
 * @example
 * ```ts
 * import { warmupConstant } from 'omai-engine/optim'
 * [warmupConstant(0, 1, 4)(2), warmupConstant(0, 1, 4)(9)] // => [0.5, 1]
 * ```
 */
export function warmupConstant(start: number, peak: number, warmupSteps: number): LearningRateSchedule {
  return linearSchedule(start, peak, warmupSteps)
}

/**
 * Piecewise constant: `values[i]` from boundary i - 1 up to boundary i (torch MultiStepLR expressed directly).
 *
 * @param boundaries - Increasing step counts where the value changes.
 * @param values - One more value than boundaries.
 * @returns The schedule.
 * @example
 * ```ts
 * import { piecewiseConstant } from 'omai-engine/optim'
 * piecewiseConstant([10, 20], [1, 0.1, 0.01])(15) // => 0.1
 * ```
 */
export function piecewiseConstant(boundaries: readonly number[], values: readonly number[]): LearningRateSchedule {
  return (t) => {
    let i = 0
    while (i < boundaries.length && t >= (boundaries[i] ?? Infinity)) i += 1
    return values[i] ?? values[values.length - 1] ?? 0
  }
}

/**
 * Piecewise interpolation (optax.piecewise_interpolate_schedule): the values at the boundaries are the running
 * products of the scales, joined linearly or by half cosines.
 *
 * @param kind - `'linear'` or `'cosine'`.
 * @param init - The value at step 0.
 * @param boundariesAndScales - Pairs [boundary, scale], boundaries increasing.
 * @returns The schedule.
 * @example
 * ```ts
 * import { piecewiseInterpolate } from 'omai-engine/optim'
 * piecewiseInterpolate('linear', 1, [[4, 0.5]])(2) // => 0.75
 * ```
 */
export function piecewiseInterpolate(
  kind: 'linear' | 'cosine',
  init: number,
  boundariesAndScales: readonly (readonly [number, number])[],
): LearningRateSchedule {
  const pairs = [...boundariesAndScales].sort((a, b) => a[0] - b[0])
  const bounds = [0, ...pairs.map((p) => p[0])]
  const values: number[] = [init]
  for (const [, s] of pairs) values.push((values.at(-1) ?? init) * s)
  const interp = (a: number, b: number, p: number): number =>
    kind === 'linear' ? (b - a) * p + a : b + ((a - b) / 2) * (Math.cos(Math.PI * p) + 1)
  return (t) => {
    for (let i = 0; i + 1 < bounds.length; i++) {
      const lo = bounds[i] ?? 0
      const hi = bounds[i + 1] ?? 0
      if (lo <= t && t < hi) return interp(values[i] ?? 0, values[i + 1] ?? 0, (t - lo) / Math.max(hi - lo, 1))
    }
    return values.at(-1) ?? init
  }
}

/**
 * Join schedules (optax.join_schedules, torch SequentialLR): schedule i runs from boundary i - 1, counting its steps
 * from there.
 *
 * @param schedules - The schedules.
 * @param boundaries - One fewer increasing boundaries.
 * @returns The joined schedule.
 * @throws OmaiError with code `value` when the counts do not match.
 * @example
 * ```ts
 * import { constantSchedule, joinSchedules, linearSchedule } from 'omai-engine/optim'
 * const s = joinSchedules([linearSchedule(0, 1, 10), constantSchedule(0.5)], [10])
 * [s(5), s(10)] // => [0.5, 0.5]
 * ```
 */
export function joinSchedules(
  schedules: readonly LearningRateSchedule[],
  boundaries: readonly number[],
): LearningRateSchedule {
  if (schedules.length !== boundaries.length + 1)
    throw new OmaiError('value', 'joinSchedules needs one more schedule than boundaries')
  return (t) => {
    let i = 0
    while (i < boundaries.length && t >= (boundaries[i] ?? Infinity)) i += 1
    const offset = i === 0 ? 0 : (boundaries[i - 1] ?? 0)
    return (schedules[i] as LearningRateSchedule)(t - offset)
  }
}

/** Options of {@link oneCycle} (torch.optim.lr_scheduler.OneCycleLR). */
export interface OneCycleOptions {
  readonly maxLr: number
  readonly totalSteps: number
  /** Fraction of the cycle spent increasing (default 0.3). */
  readonly pctStart?: number
  readonly anneal?: 'cos' | 'linear'
  /** The initial value is maxLr / divFactor (default 25). */
  readonly divFactor?: number
  /** The final value is the initial value / finalDivFactor (default 1e4). */
  readonly finalDivFactor?: number
  /** Up, back down to the initial value, then down to the final value. */
  readonly threePhase?: boolean
}

/**
 * The one-cycle policy (Smith and Topin 2019; torch OneCycleLR): warm up from maxLr / divFactor to maxLr, then
 * anneal to the initial value / finalDivFactor, by cosine or linearly, optionally in three phases.
 *
 * @param options - Peak, length, warm-up fraction, annealing, divisors and phases.
 * @returns The schedule.
 * @example
 * ```ts
 * import { oneCycle } from 'omai-engine/optim'
 * const s = oneCycle({ maxLr: 1, totalSteps: 11, pctStart: 0.5, anneal: 'linear', divFactor: 10 })
 * [s(0), Math.round(s(4.5) * 1e12) / 1e12] // => [0.1, 1]
 * ```
 */
export function oneCycle(options: OneCycleOptions): LearningRateSchedule {
  const total = options.totalSteps
  const pct = options.pctStart ?? 0.3
  const initial = options.maxLr / (options.divFactor ?? 25)
  const min = initial / (options.finalDivFactor ?? 1e4)
  const phases: readonly { readonly end: number; readonly from: number; readonly to: number }[] =
    options.threePhase === true
      ? [
          { end: pct * total - 1, from: initial, to: options.maxLr },
          { end: 2 * pct * total - 2, from: options.maxLr, to: initial },
          { end: total - 1, from: initial, to: min },
        ]
      : [
          { end: pct * total - 1, from: initial, to: options.maxLr },
          { end: total - 1, from: options.maxLr, to: min },
        ]
  const anneal =
    (options.anneal ?? 'cos') === 'cos'
      ? (a: number, b: number, p: number): number => b + ((a - b) / 2) * (Math.cos(Math.PI * p) + 1)
      : (a: number, b: number, p: number): number => (b - a) * p + a
  return (t) => {
    let start = 0
    for (const [i, ph] of phases.entries()) {
      if (t <= ph.end || i === phases.length - 1) return anneal(ph.from, ph.to, (t - start) / (ph.end - start))
      start = ph.end
    }
    return min
  }
}

/** Options of {@link cyclical} (torch.optim.lr_scheduler.CyclicLR). */
export interface CyclicalOptions {
  readonly baseLr: number
  readonly maxLr: number
  /** Steps in the increasing half (default 2000). */
  readonly stepSizeUp?: number
  /** Steps in the decreasing half (default stepSizeUp). */
  readonly stepSizeDown?: number
  /** `triangular`, `triangular2` (amplitude halves each cycle) or `exp_range` (amplitude times gamma^t). */
  readonly mode?: 'triangular' | 'triangular2' | 'exp_range'
  readonly gamma?: number
}

/**
 * Cyclical learning rates (Smith 2017; torch CyclicLR): triangular waves between baseLr and maxLr, with the amplitude
 * constant, halved every cycle, or decayed exponentially.
 *
 * @param options - Bounds, half-cycle lengths, mode and gamma.
 * @returns The schedule.
 * @example
 * ```ts
 * import { cyclical } from 'omai-engine/optim'
 * const s = cyclical({ baseLr: 0, maxLr: 1, stepSizeUp: 4 })
 * [s(2), s(4), s(6), s(8)] // => [0.5, 1, 0.5, 0]
 * ```
 */
export function cyclical(options: CyclicalOptions): LearningRateSchedule {
  const up = options.stepSizeUp ?? 2000
  const down = options.stepSizeDown ?? up
  const totalSize = up + down
  const ratio = up / totalSize
  const mode = options.mode ?? 'triangular'
  const gamma = options.gamma ?? 1
  return (t) => {
    const cycle = Math.floor(1 + t / totalSize)
    const x = 1 + t / totalSize - cycle
    const factor = x <= ratio ? x / ratio : (x - 1) / (ratio - 1)
    const height = (options.maxLr - options.baseLr) * factor
    const scale = mode === 'triangular' ? 1 : mode === 'triangular2' ? 1 / 2 ** (cycle - 1) : gamma ** t
    return options.baseLr + height * scale
  }
}

/** Options of {@link reduceOnPlateau} (torch.optim.lr_scheduler.ReduceLROnPlateau). */
export interface PlateauOptions {
  readonly mode?: 'min' | 'max'
  /** Multiply the rate by this on a plateau (default 0.1). */
  readonly factor?: number
  /** Epochs without improvement before reducing (default 10). */
  readonly patience?: number
  /** Minimum relative (or absolute) improvement (default 1e-4). */
  readonly threshold?: number
  readonly thresholdMode?: 'rel' | 'abs'
  /** Epochs to wait after a reduction (default 0). */
  readonly cooldown?: number
  readonly minLr?: number
  /** Reductions smaller than this are skipped (default 1e-8). */
  readonly eps?: number
}

/** State of {@link reduceOnPlateau}. */
export interface PlateauState {
  readonly lr: number
  readonly best: number
  readonly badEpochs: number
  readonly cooldown: number
}

/**
 * Reduce the learning rate when a metric stops improving (torch ReduceLROnPlateau): a stateful schedule driven by
 * the metric rather than the step count. `init` starts from a rate; `step` takes the epoch's metric.
 *
 * @param options - Mode, factor, patience, threshold, cooldown and floors.
 * @returns `{ init, step }`.
 * @example
 * ```ts
 * import { reduceOnPlateau } from 'omai-engine/optim'
 * const p = reduceOnPlateau({ patience: 1, factor: 0.5 })
 * let s = p.init(1)
 * for (const loss of [3, 2, 2, 2]) s = p.step(s, loss)
 * s.lr // => 0.5
 * ```
 */
export function reduceOnPlateau(options: PlateauOptions = {}): {
  readonly init: (lr: number) => PlateauState
  readonly step: (state: PlateauState, metric: number) => PlateauState
} {
  const mode = options.mode ?? 'min'
  const factor = options.factor ?? 0.1
  const patience = options.patience ?? 10
  const threshold = options.threshold ?? 1e-4
  const relative = (options.thresholdMode ?? 'rel') === 'rel'
  const cooldownSteps = options.cooldown ?? 0
  const minLr = options.minLr ?? 0
  const eps = options.eps ?? 1e-8
  const better = (a: number, best: number): boolean => {
    if (mode === 'min') return relative ? a < best * (1 - threshold) : a < best - threshold
    return relative ? a > best * (threshold + 1) : a > best + threshold
  }
  return {
    init: (lr) => ({ lr, best: mode === 'min' ? Infinity : -Infinity, badEpochs: 0, cooldown: 0 }),
    step: (s, metric) => {
      let { lr, best, badEpochs, cooldown } = s
      if (better(metric, best)) {
        best = metric
        badEpochs = 0
      } else badEpochs += 1
      if (cooldown > 0) {
        cooldown -= 1
        badEpochs = 0
      }
      if (badEpochs > patience) {
        const next = Math.max(lr * factor, minLr)
        if (lr - next > eps) lr = next
        cooldown = cooldownSteps
        badEpochs = 0
      }
      return { lr, best, badEpochs, cooldown }
    },
  }
}
