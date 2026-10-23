/** A learning-rate schedule: the rate at each step count (from 0). */
export type Schedule = (step: number) => number

/**
 * Exponential decay: init * rate^(step / steps), optionally in whole steps (staircase), as optax.exponential_decay.
 *
 * @param init - Initial value.
 * @param decayRate - Factor per `steps` steps.
 * @param steps - Steps per factor.
 * @param staircase - Decay in discrete jumps.
 * @returns The schedule.
 * @example
 * ```ts
 * import { exponentialDecay } from 'whitebox-engine/optim'
 * exponentialDecay(1, 0.5, 10)(20) // => 0.25
 * ```
 */
export function exponentialDecay(init: number, decayRate: number, steps: number, staircase = false): Schedule {
  return (t) => init * decayRate ** (staircase ? Math.floor(t / steps) : t / steps)
}

/**
 * Cosine decay from init to alpha init over `decaySteps`, constant afterwards (optax.cosine_decay_schedule,
 * torch CosineAnnealingLR without restarts).
 *
 * @param init - Initial value.
 * @param decaySteps - Length of the decay.
 * @param alpha - Final value as a fraction of init.
 * @returns The schedule.
 * @example
 * ```ts
 * import { cosineDecay } from 'whitebox-engine/optim'
 * cosineDecay(1, 100)(50) // => 0.5
 * ```
 */
export function cosineDecay(init: number, decaySteps: number, alpha = 0): Schedule {
  return (t) => {
    const p = Math.min(t, decaySteps) / decaySteps
    return init * ((1 - alpha) * 0.5 * (1 + Math.cos(Math.PI * p)) + alpha)
  }
}

/**
 * Linear warm-up from `start` to `peak` over `warmupSteps`, then cosine decay to `end` by `totalSteps`
 * (optax.warmup_cosine_decay_schedule).
 *
 * @param options - Start, peak and end values, warm-up and total steps.
 * @returns The schedule.
 * @example
 * ```ts
 * import { warmupCosineDecay } from 'whitebox-engine/optim'
 * warmupCosineDecay({ start: 0, peak: 1, end: 0, warmupSteps: 10, totalSteps: 110 })(10) // => 1
 * ```
 */
export function warmupCosineDecay(options: {
  readonly start: number
  readonly peak: number
  readonly end: number
  readonly warmupSteps: number
  readonly totalSteps: number
}): Schedule {
  const { start, peak, end, warmupSteps, totalSteps } = options
  return (t) => {
    if (t < warmupSteps) return start + ((peak - start) * t) / warmupSteps
    const p = Math.min(1, (t - warmupSteps) / Math.max(1, totalSteps - warmupSteps))
    return end + (peak - end) * 0.5 * (1 + Math.cos(Math.PI * p))
  }
}

/**
 * Linear interpolation from `init` to `end` over `steps`, constant afterwards (optax.linear_schedule).
 *
 * @param init - Initial value.
 * @param end - Final value.
 * @param steps - Length of the transition.
 * @returns The schedule.
 * @example
 * ```ts
 * import { linearSchedule } from 'whitebox-engine/optim'
 * linearSchedule(1, 0, 4)(1) // => 0.75
 * ```
 */
export function linearSchedule(init: number, end: number, steps: number): Schedule {
  return (t) => init + (end - init) * Math.min(1, t / steps)
}

/**
 * Piecewise constant: `values[i]` from boundary i - 1 up to boundary i (torch MultiStepLR expressed directly).
 *
 * @param boundaries - Increasing step counts where the value changes.
 * @param values - One more value than boundaries.
 * @returns The schedule.
 * @example
 * ```ts
 * import { piecewiseConstant } from 'whitebox-engine/optim'
 * piecewiseConstant([10, 20], [1, 0.1, 0.01])(15) // => 0.1
 * ```
 */
export function piecewiseConstant(boundaries: readonly number[], values: readonly number[]): Schedule {
  return (t) => {
    let i = 0
    while (i < boundaries.length && t >= (boundaries[i] ?? Infinity)) i += 1
    return values[i] ?? values[values.length - 1] ?? 0
  }
}
