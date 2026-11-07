/**
 * The learning-rate schedules as an option set: each the library's schedule from `optim` as a shape over steps that
 * takes the optimiser's own learning rate as its peak (constant, step decay, exponential, inverse time, cosine,
 * cosine with warm restarts, linear, polynomial, warm-up into cosine or exponential decay), with its knobs and its
 * properties (whether it warms up, decays, restarts). A gradient mode takes a schedule from this table beside its
 * optimiser; the step budget of the page is the schedule's horizon where one is needed.
 */

import { bool, dict, int, type Option, type OptionTable, optionTable, real } from '../space/index.ts'
import {
  constantSchedule,
  cosineDecay,
  cosineWarmRestarts,
  exponentialDecay,
  inverseTimeDecay,
  type LearningRateSchedule,
  linearSchedule,
  polynomialDecay,
  stepDecay,
  warmupCosineDecay,
  warmupExponentialDecay,
} from './schedules.ts'

/** A schedule shape: given the peak (the optimiser's learning rate), the schedule over steps. */
export type ScheduleShape = (peak: number) => LearningRateSchedule

/** What a schedule does, for validity rules and selectors. */
export interface ScheduleProperties {
  readonly warmup: boolean
  readonly decays: boolean
  readonly restarts: boolean
}

const shape = (
  properties: ScheduleProperties,
  description: string,
  space: Option<ScheduleShape, ScheduleProperties>['space'],
  make: Option<ScheduleShape, ScheduleProperties>['make'],
): Option<ScheduleShape, ScheduleProperties> => ({ description, space, properties, make })

const steps = int(1, 100_000, { log: true, default: 1000, label: 'horizon (steps)' })
const warmupSteps = int(1, 10_000, { log: true, default: 100, label: 'warm-up steps' })
const endRatio = real(0, 1, { default: 0.01, label: 'end, as a share of the peak' })

/**
 * The schedules as one slot: constant (default), step, exponential, inverse-time, cosine, restarts, linear, polynomial, warm-ups.
 *
 * @index table Learning-rate schedules as one slot.
 * @example
 * ```ts
 * import { schedules } from 'mdml-engine/optim'
 * schedules.names.slice(0, 3) // => ['constant', 'step', 'exponential']
 * schedules.optionOf({ kind: 'cosine' }).name // => 'cosine'
 * ```
 */
export const schedules: OptionTable<ScheduleShape, ScheduleProperties> = optionTable(
  {
    constant: shape(
      { warmup: false, decays: false, restarts: false },
      'the peak throughout',
      dict({}),
      () => constantSchedule,
    ),
    step: shape(
      { warmup: false, decays: true, restarts: false },
      'the peak times gamma every stepSize steps',
      dict({
        stepSize: int(1, 100_000, { log: true, default: 100, label: 'step size' }),
        gamma: real(0.01, 0.99, { default: 0.1, label: 'factor γ' }),
      }),
      (k) => (peak) => stepDecay(peak, k['stepSize'] as number, k['gamma'] as number),
    ),
    exponential: shape(
      { warmup: false, decays: true, restarts: false },
      'the peak times rate to the power of step over decaySteps',
      dict({
        decayRate: real(0.01, 0.999, { default: 0.9, label: 'decay rate' }),
        decaySteps: int(1, 100_000, { log: true, default: 100, label: 'decay steps' }),
        staircase: bool({ default: false, label: 'staircase' }),
      }),
      (k) => (peak) =>
        exponentialDecay(peak, k['decayRate'] as number, k['decaySteps'] as number, k['staircase'] as boolean),
    ),
    inverseTime: shape(
      { warmup: false, decays: true, restarts: false },
      'the peak over one plus rate times step',
      dict({ decayRate: real(1e-4, 10, { log: true, default: 0.01, label: 'decay rate' }) }),
      (k) => (peak) => inverseTimeDecay(peak, k['decayRate'] as number),
    ),
    cosine: shape(
      { warmup: false, decays: true, restarts: false },
      'half a cosine from the peak to its end share over the horizon',
      dict({ steps, endRatio }),
      (k) => (peak) => cosineDecay(peak, k['steps'] as number, k['endRatio'] as number),
    ),
    cosineRestarts: shape(
      { warmup: false, decays: true, restarts: true },
      'cosine decay restarted every period, the period growing by a factor',
      dict({
        period: int(1, 100_000, { log: true, default: 100, label: 'first period' }),
        growth: real(1, 4, { default: 2, label: 'period growth' }),
        endRatio,
      }),
      (k) => (peak) =>
        cosineWarmRestarts(peak, k['period'] as number, k['growth'] as number, peak * (k['endRatio'] as number)),
    ),
    linear: shape(
      { warmup: false, decays: true, restarts: false },
      'a straight line from the peak to its end share over the horizon',
      dict({ steps, endRatio }),
      (k) => (peak) => linearSchedule(peak, peak * (k['endRatio'] as number), k['steps'] as number),
    ),
    polynomial: shape(
      { warmup: false, decays: true, restarts: false },
      'a polynomial from the peak to its end share over the horizon',
      dict({ steps, endRatio, power: real(0.5, 4, { default: 2, label: 'power' }) }),
      (k) => (peak) =>
        polynomialDecay(peak, peak * (k['endRatio'] as number), k['power'] as number, k['steps'] as number),
    ),
    warmupCosine: shape(
      { warmup: true, decays: true, restarts: false },
      'a linear warm-up from zero to the peak, then cosine decay to its end share',
      dict({ warmupSteps, steps, endRatio }),
      (k) => (peak) =>
        warmupCosineDecay({
          start: 0,
          peak,
          end: peak * (k['endRatio'] as number),
          warmupSteps: k['warmupSteps'] as number,
          totalSteps: Math.max(k['steps'] as number, (k['warmupSteps'] as number) + 1),
        }),
    ),
    warmupExponential: shape(
      { warmup: true, decays: true, restarts: false },
      'a linear warm-up from zero to the peak, then exponential decay',
      dict({
        warmupSteps,
        decayRate: real(0.01, 0.999, { default: 0.9, label: 'decay rate' }),
        decaySteps: int(1, 100_000, { log: true, default: 100, label: 'decay steps' }),
      }),
      (k) => (peak) =>
        warmupExponentialDecay({
          start: 0,
          peak,
          warmupSteps: k['warmupSteps'] as number,
          decayRate: k['decayRate'] as number,
          transitionSteps: k['decaySteps'] as number,
        }),
    ),
  },
  { label: 'schedule', description: 'how the learning rate changes over the steps' },
)
