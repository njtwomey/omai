/**
 * Adaptive control: model-reference adaptive control of a first-order plant whose gain (and pole) is unknown, the plant's
 * output forced towards a reference model's by two parameters adapted online, by the MIT rule (gradient descent on the
 * squared error, with the sensitivity approximated) or by the Lyapunov rule (adaptation derived so that a Lyapunov function
 * of the error and the parameter errors decreases), as a stepped algorithm that logs the loop, the error and the distance of
 * the parameters from the ones that would match the model; and gain scheduling, the other way to cope with a plant that
 * changes: gains designed at operating points and interpolated.
 */

import { OmaiError } from '../../base/index.ts'
import { add, mul, sliceAxis, stack, sub, sum, type Value } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { type Contract, defineContract, scalar } from '../../trace/index.ts'
import { type FlowState, flowAlgorithm } from '../ode/flow.ts'
import type { OdeVectorField } from '../ode/problem.ts'

/** The first-order plant y' = -a y + k u, with the gain k (and perhaps the pole a) unknown to the controller. */
export interface FirstOrderPlant {
  /** The pole: the plant decays at rate `a` (negative for an unstable plant). */
  readonly a: number
  /** The gain k (its sign must be known and positive here). */
  readonly k: number
}

/** The reference model ym' = -am ym + km r that the loop should follow. */
export interface ReferenceModel {
  /** The model's pole (positive: a stable model). */
  readonly am: number
  /** The model's gain. */
  readonly km: number
}

/** How the parameters are adapted. */
export type AdaptationLaw = 'mit' | 'lyapunov'

/** The control u = theta1 r - theta2 y, and the parameters that make the loop match the model. */
export interface MatchingParameters {
  /** The feedforward gain km / k. */
  readonly theta1: number
  /** The feedback gain (am - a) / k. */
  readonly theta2: number
}

/**
 * The parameters of u = theta1 r - theta2 y that make y' = -a y + k u identical to the reference model: the closed loop is
 * y' = -(a + k theta2) y + k theta1 r, so theta1 = km / k and theta2 = (am - a) / k.
 *
 * @param plant - The plant.
 * @param model - The reference model.
 * @returns The matching parameters (the truth the adaptation is measured against).
 * @concept model-reference-adaptive-control
 * @example
 * ```ts
 * import { matchingParameters } from 'omai-engine/dynamics/control'
 * matchingParameters({ a: 1, k: 2 }, { am: 3, km: 6 }) // => { theta1: 3, theta2: 1 }
 * ```
 */
export function matchingParameters(plant: FirstOrderPlant, model: ReferenceModel): MatchingParameters {
  return { theta1: model.km / plant.k, theta2: (model.am - plant.a) / plant.k }
}

/** Options of {@link mracAlgorithm}. */
export interface MracOptions {
  readonly plant: FirstOrderPlant
  readonly model: ReferenceModel
  /** The adaptation law (default Lyapunov). */
  readonly law?: AdaptationLaw
  /** The adaptation gain (rate). */
  readonly gamma: number
  /** The reference signal r(t). */
  readonly reference: (t: number) => number
  /** The step of the Runge-Kutta integration of the loop and the adaptation. */
  readonly dt: number
  /** The steps to take. */
  readonly steps: number
  /** The starting parameters (default zero: no control at first). */
  readonly theta0?: readonly [number, number]
  /** The plant's and the model's starting output (default 0 and 0). */
  readonly y0?: readonly [number, number]
}

/** The state of {@link mracAlgorithm}: `[y, ym, theta1, theta2]` as a flow. */
export type MracState = FlowState

/** The paths of {@link mracAlgorithm}. */
export type MracPath =
  | 'loop/output'
  | 'loop/reference'
  | 'adaptation/parameter'
  | 'adaptation/error'
  | 'adaptation/gain'
  | 'adaptation/parameterError'

/** What a model-reference adaptive control run logs. */
export const mracContract: Contract<FlowState, MracPath> = defineContract({
  'loop/output': scalar({ description: 'the plant’s output y' }),
  'loop/reference': scalar({ description: 'the reference model’s output ym, which y should follow' }),
  'adaptation/parameter': scalar({ description: 'the feedforward parameter theta1 (the feedback one is theta2)' }),
  'adaptation/error': scalar({ description: 'the tracking error e = y - ym', direction: 'lower' }),
  'adaptation/gain': scalar({ description: 'the adaptation gain gamma' }),
  'adaptation/parameterError': scalar({
    description: 'the distance of (theta1, theta2) from the parameters that match the model (the truth)',
    direction: 'lower',
  }),
})

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))

/**
 * The model-reference adaptive loop as a stepped algorithm. The plant y' = -a y + k u is driven by u = theta1 r - theta2 y,
 * the model by ym' = -am ym + km r, and the error e = y - ym tells the parameters how to move: the Lyapunov law
 * theta1' = -gamma e r, theta2' = gamma e y comes from V = e^2 / 2 + k ((theta1 - theta1*)^2 + (theta2 - theta2*)^2) / (2 gamma)
 * (whose derivative is -am e^2) and so is stable for any gamma. The MIT rule theta1' = -gamma e ym follows the gradient of
 * e^2 / 2 with the sensitivity d e / d theta1 = k r / (p + a + k theta2) approximated by (k / km) ym (the model's output
 * stands for the filtered reference), which holds when the plant's pole is the model's; it adapts the feedforward gain
 * alone (theta2 stays at its starting value), and goes unstable when gamma times the reference's power is large. The loop is integrated by fixed-step RK4 on plain numbers, so the state serialises. Logs the output, the model's, the
 * parameter, the error, the gain and the distance of the parameters from the matching ones.
 *
 * @param options - Plant, model, law, gain, reference signal and resolution.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a non-positive gain, plant gain or model pole.
 * @concept model-reference-adaptive-control
 * @example
 * ```ts
 * import { mracAlgorithm } from 'omai-engine/dynamics/control'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = mracAlgorithm({
 *   plant: { a: 1, k: 2 }, model: { am: 3, km: 3 }, gamma: 2, dt: 0.01, steps: 3000,
 *   reference: (t) => Math.sin(t) + Math.sin(3.7 * t),
 * })
 * Math.abs(runAlgorithm(alg, { maxSteps: 4000 }).state.y[2] - 1.5) < 0.05 // => true
 * ```
 */
export function mracAlgorithm(options: MracOptions): Algorithm<FlowState, MracPath> {
  const { plant, model, gamma } = options
  if (!(gamma > 0) || !(plant.k > 0) || !(model.am > 0)) {
    throw new OmaiError(
      'value',
      'mracAlgorithm: the adaptation gain, the plant gain and the model pole must be positive',
    )
  }
  const law = options.law ?? 'lyapunov'
  const truth = matchingParameters(plant, model)
  const field: OdeVectorField = (t, s) => {
    const y = at(s, 0)
    const ym = at(s, 1)
    const t1 = at(s, 2)
    const t2 = at(s, 3)
    const r = options.reference(t)
    const e = sub(y, ym)
    const u = sub(mul(t1, r), mul(t2, y))
    const dy = add(mul(y, -plant.a), mul(u, plant.k))
    const dym = add(mul(ym, -model.am), r * model.km)
    const dt1 = law === 'lyapunov' ? mul(mul(e, r), -gamma) : mul(mul(e, ym), -gamma)
    const dt2 = law === 'lyapunov' ? mul(mul(e, y), gamma) : mul(e, 0)
    return stack([dy, dym, dt1, dt2])
  }
  const [y0, ym0] = options.y0 ?? [0, 0]
  const [t10, t20] = options.theta0 ?? [0, 0]
  return flowAlgorithm<MracPath>(field, [y0, ym0, t10, t20], {
    dt: options.dt,
    steps: options.steps,
    name: 'mrac',
    config: { law, gamma, a: plant.a, k: plant.k, am: model.am, km: model.km },
    observe: (_t, s, ctx) => {
      ctx.log('loop/output', s[0] as number)
      ctx.log('loop/reference', s[1] as number)
      ctx.log('adaptation/parameter', s[2] as number)
      ctx.log('adaptation/error', (s[0] as number) - (s[1] as number))
      ctx.log('adaptation/gain', gamma)
      ctx.log('adaptation/parameterError', Math.hypot((s[2] as number) - truth.theta1, (s[3] as number) - truth.theta2))
    },
  })
}

/** One operating point of a gain schedule. */
export interface SchedulePoint {
  /** The value of the scheduling variable (an operating speed, altitude, load). */
  readonly at: number
  /** The gains designed there. */
  readonly gain: readonly number[]
}

/**
 * Gain scheduling: controller gains designed at a few operating points and interpolated linearly in the scheduling variable
 * between them (held at the end values beyond). The schedule is exact at the design points and, for a plant that varies
 * smoothly, close between them; it does not adapt, so it is only as good as the plant's dependence on the variable is known.
 *
 * @param points - The operating points, at least one, with gains of equal size.
 * @returns A function from the scheduling variable to the gains.
 * @throws OmaiError with code `value` for no points, unequal gain sizes or repeated scheduling values.
 * @concept gain-scheduling
 * @example
 * ```ts
 * import { gainSchedule } from 'omai-engine/dynamics/control'
 * const k = gainSchedule([{ at: 0, gain: [1, 10] }, { at: 2, gain: [3, 20] }])
 * [k(1), k(-5), k(9)] // => [[2, 15], [1, 10], [3, 20]]
 * ```
 */
export function gainSchedule(points: readonly SchedulePoint[]): (variable: number) => number[] {
  if (points.length === 0) throw new OmaiError('value', 'gainSchedule: give at least one operating point')
  const sorted = [...points].sort((p, q) => p.at - q.at)
  const size = sorted[0]?.gain.length ?? 0
  if (sorted.some((p) => p.gain.length !== size))
    throw new OmaiError('value', 'gainSchedule: every point needs the same number of gains')
  for (let i = 1; i < sorted.length; i++) {
    if ((sorted[i] as SchedulePoint).at === (sorted[i - 1] as SchedulePoint).at) {
      throw new OmaiError('value', 'gainSchedule: the scheduling values must differ')
    }
  }
  return (v) => {
    const first = sorted[0] as SchedulePoint
    const last = sorted[sorted.length - 1] as SchedulePoint
    if (v <= first.at) return [...first.gain]
    if (v >= last.at) return [...last.gain]
    const j = sorted.findIndex((p) => p.at >= v)
    const lo = sorted[j - 1] as SchedulePoint
    const hi = sorted[j] as SchedulePoint
    const w = (v - lo.at) / (hi.at - lo.at)
    return lo.gain.map((g, i) => g + w * ((hi.gain[i] as number) - g))
  }
}
