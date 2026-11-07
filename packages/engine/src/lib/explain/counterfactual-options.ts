/**
 * The slot of Wachter's counterfactual search that varies as an option table: the schedule of lambda, the weight of the
 * prediction loss against the distance, which grows from one round of the search to the next until the target is met
 * (doubling as in the paper, additive steps, or held constant), each with the knobs of its own. The table is one `union`
 * space for controls, grids and tuners, and builds the schedule function from a configuration.
 */

import { dict, knobOption, type Option, type OptionTable, optionTable, real } from '../space/index.ts'

/** The weight of the prediction loss for the next round, from the current weight and the round just finished (from 0). */
export type LambdaSchedule = (lambda: number, round: number) => number

/** What a lambda schedule does, for validity rules and selectors. */
export interface LambdaScheduleProperties {
  /** Whether lambda never changes (the search is then one subproblem solved repeatedly). */
  readonly constant: boolean
  /** Whether lambda grows geometrically (so the prediction loss comes to dominate quickly). */
  readonly geometric: boolean
}

function schedule(
  description: string,
  properties: LambdaScheduleProperties,
  space: ReturnType<typeof dict>,
  build: (k: (name: string) => number) => LambdaSchedule,
): Option<LambdaSchedule, LambdaScheduleProperties> {
  return knobOption(description, properties, space, build)
}

/**
 * The lambda schedules as one slot: `doubling` (the weight is multiplied by a factor each round, 2 in the paper),
 * `additive` (a fixed increment each round) and `constant` (the weight stays).
 *
 * @example
 * ```ts
 * import { lambdaSchedules } from 'mdml-engine/explain'
 * lambdaSchedules.make({ kind: 'doubling', factor: 3 })(0.1, 0) // => 0.30000000000000004
 * ```
 */
export const lambdaSchedules: OptionTable<LambdaSchedule, LambdaScheduleProperties> = optionTable(
  {
    doubling: schedule(
      'multiply the weight of the prediction loss by a factor each round (the paper doubles it)',
      { constant: false, geometric: true },
      dict({ factor: real(1.01, 100, { log: true, default: 2, label: 'growth factor per round' }) }),
      (k) => (lambda) => lambda * k('factor'),
    ),
    additive: schedule(
      'add a fixed increment to the weight of the prediction loss each round',
      { constant: false, geometric: false },
      dict({ increment: real(0.001, 1000, { log: true, default: 1, label: 'increment per round' }) }),
      (k) => (lambda) => lambda + k('increment'),
    ),
    constant: schedule(
      'keep the weight of the prediction loss: the same subproblem is solved again each round',
      { constant: true, geometric: false },
      dict({}),
      () => (lambda) => lambda,
    ),
  },
  {
    default: 'doubling',
    label: 'lambda schedule',
    description: 'How the weight of the prediction loss against the distance grows from one round to the next.',
  },
)
