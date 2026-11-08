/**
 * What the permutation-sampling estimators of this domain share: the running mean and standard error of one marginal
 * contribution per feature (or training point) over random orders, as a state that serialises, and the paths they log.
 * Sampling Shapley values, SAGE and data Shapley all add one vector of marginal contributions per step.
 */

import { type Contract, defineContract, type StepContext, scalar, tensor } from '../trace/index.ts'

/** The paths the permutation-sampling estimators log, by step (one random order per step). */
export type ShapleyPath = 'shapley/phi' | 'shapley/stderr' | 'shapley/permutation' | 'shapley/marginal'

/** What a stepped permutation-sampling estimator logs: the estimates, their errors, the order count and the step's marginals. */
export const shapleyContract: Contract<unknown, ShapleyPath> = defineContract({
  'shapley/phi': tensor({
    role: 'output',
    shape: ['players'],
    description: 'the running mean marginal contribution of each player',
  }),
  'shapley/stderr': tensor({
    role: 'output',
    shape: ['players'],
    description: 'the standard error of each running mean (NaN before two orders)',
  }),
  'shapley/permutation': scalar({ description: 'random orders sampled so far' }),
  'shapley/marginal': tensor({
    role: 'output',
    shape: ['players'],
    description: 'the marginal contributions of the order just sampled',
  }),
})

/** The running sums of marginal contributions over the orders sampled so far. */
export interface ShapleyState {
  readonly sum: Float64Array
  readonly sumSq: Float64Array
  readonly count: number
}

/**
 * An empty running estimate.
 *
 * @param players - The number of features or training points.
 * @returns The state with no orders yet.
 * @example
 * ```ts
 * import { shapleyStart } from 'omai-engine/explain'
 * shapleyStart(3).count // => 0
 * ```
 */
export function shapleyStart(players: number): ShapleyState {
  return { sum: new Float64Array(players), sumSq: new Float64Array(players), count: 0 }
}

/**
 * Add the marginal contributions of one more random order to a running estimate.
 *
 * @param state - The running estimate.
 * @param marginal - One contribution per player.
 * @returns The new state; the old one is unchanged.
 * @example
 * ```ts
 * import { shapleyAccumulate, shapleyEstimates, shapleyStart } from 'omai-engine/explain'
 * shapleyEstimates(shapleyAccumulate(shapleyAccumulate(shapleyStart(1), [1]), [3])).values // => [2]
 * ```
 */
export function shapleyAccumulate(state: ShapleyState, marginal: ArrayLike<number>): ShapleyState {
  return {
    sum: Float64Array.from(state.sum, (v, j) => v + (marginal[j] ?? 0)),
    sumSq: Float64Array.from(state.sumSq, (v, j) => v + (marginal[j] ?? 0) ** 2),
    count: state.count + 1,
  }
}

/**
 * The mean and standard error of each player's contribution so far.
 *
 * @param state - The running estimate.
 * @returns The values and their standard errors (NaN before two orders).
 * @example
 * ```ts
 * import { shapleyAccumulate, shapleyEstimates, shapleyStart } from 'omai-engine/explain'
 * const s = [1, 3].reduce((acc, c) => shapleyAccumulate(acc, [c]), shapleyStart(1))
 * shapleyEstimates(s) // => { values: [2], standardErrors: [1] }
 * ```
 */
export function shapleyEstimates(state: ShapleyState): {
  readonly values: number[]
  readonly standardErrors: number[]
} {
  const values = Array.from(state.sum, (v) => v / state.count)
  const standardErrors = Array.from(state.sumSq, (v, j) => {
    const m = values[j] ?? 0
    return state.count < 2 ? Number.NaN : Math.sqrt(Math.max(0, v / state.count - m * m) / (state.count - 1))
  })
  return { values, standardErrors }
}

/** Log a step of a permutation-sampling estimator to the paths of {@link shapleyContract}. */
export function logShapley(
  ctx: Pick<StepContext<ShapleyPath>, 'log' | 'wants'>,
  state: ShapleyState,
  marginal: ArrayLike<number>,
): void {
  ctx.log('shapley/permutation', state.count)
  if (ctx.wants('shapley/marginal')) ctx.log('shapley/marginal', Float64Array.from(marginal))
  if (ctx.wants('shapley/phi') || ctx.wants('shapley/stderr')) {
    const { values, standardErrors } = shapleyEstimates(state)
    ctx.log('shapley/phi', Float64Array.from(values))
    ctx.log('shapley/stderr', Float64Array.from(standardErrors))
  }
}
