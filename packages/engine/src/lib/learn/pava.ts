/**
 * Pool adjacent violators as a steppable algorithm. The block stack of the classic formulation: each step either pools
 * the top two blocks when their means violate the order, or pushes the next observation as a block of its own, so the
 * stack, the pooled means and the violations left can be watched while the fit is built. `poolAdjacentViolators` in
 * `calibration.ts` is this run to the end; the direction of the fit is a slot (`isotonicDirections`).
 */

import { OmaiError } from '../base/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { isotonicDirections } from './pava-options.ts'

/** The paths {@link pavaAlgorithm} logs. */
export type PavaPath = 'pava/blocks' | 'pava/means' | 'pava/merge' | 'pava/violations' | 'pava/error'

/** What a stepped PAVA run logs: the block count, their means, whether the step pooled, violations left and the error. */
export const pavaContract: Contract<unknown, PavaPath> = defineContract({
  'pava/blocks': scalar({ description: 'blocks on the stack' }),
  'pava/means': tensor({ role: 'state', shape: ['blocks'], description: 'the mean of each block on the stack' }),
  'pava/merge': scalar({ description: '1 when the step pooled the top two blocks, 0 when it pushed an observation' }),
  'pava/violations': scalar({ description: '1 while the top two blocks of the stack still violate the order, else 0' }),
  'pava/error': scalar({ description: 'weighted squared error of the fit so far', direction: 'lower' }),
})

/**
 * The state of {@link pavaAlgorithm}: the stack of blocks (where each starts, its total weight, weighted sum and
 * weighted sum of squares, in the sign the algorithm works in) and how many observations have been pushed. Steps push
 * and pop the arrays in place (so a step costs O(1)); keep a copy of a state you want to look at later.
 */
export interface PavaState {
  /** Observations pushed so far. */
  pushed: number
  /** The index of the first observation of each block. */
  readonly starts: number[]
  /** The total weight of each block. */
  readonly weight: number[]
  /** The weighted sum of the (signed) values in each block. */
  readonly sum: number[]
  /** The weighted sum of squares of the (signed) values in each block. */
  readonly sumSquares: number[]
  /** Pools done so far. */
  merges: number
  /** Whether the last step pooled. */
  merged: boolean
  /** The weighted squared error of the blocks about their means. */
  error: number
}

/** What {@link pavaAlgorithm} fits and how. */
export interface PavaSettings {
  /** The weights of the observations (default ones). */
  readonly weights?: ArrayLike<number>
  /** The direction of the fit, a value of `isotonicDirections.space` (default increasing). */
  readonly direction?: SpaceValue
}

const blockError = (w: number, s: number, ss: number): number => Math.max(0, ss - (w === 0 ? 0 : (s * s) / w))

/** Whether the top two blocks of the stack violate the order. */
function topViolates(s: PavaState): boolean {
  const k = s.starts.length
  return k >= 2 && (s.sum[k - 2] ?? 0) / (s.weight[k - 2] ?? 1) >= (s.sum[k - 1] ?? 0) / (s.weight[k - 1] ?? 1)
}

/**
 * The block means of a state, in the direction of the data.
 *
 * @param state - A state of {@link pavaAlgorithm}.
 * @param sign - 1 for an increasing fit, -1 for a decreasing one.
 * @returns One mean per block.
 * @example
 * ```ts
 * import { pavaAlgorithm, pavaMeans } from 'omai-engine/learn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * pavaMeans(runAlgorithm(pavaAlgorithm([1, 3, 2]), { maxSteps: 10 }).state) // => [1, 2.5]
 * ```
 */
export function pavaMeans(state: PavaState, sign: 1 | -1 = 1): number[] {
  return state.starts.map((_, j) => (sign * (state.sum[j] ?? 0)) / (state.weight[j] ?? 1))
}

/**
 * The fitted values of a state: every observation gets the mean of its block (the observations not yet pushed get
 * `NaN`).
 *
 * @param state - A state of {@link pavaAlgorithm}.
 * @param n - The number of observations.
 * @param sign - 1 for an increasing fit, -1 for a decreasing one.
 * @returns The fitted value of each observation.
 * @example
 * ```ts
 * import { pavaAlgorithm, pavaFitted } from 'omai-engine/learn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * pavaFitted(runAlgorithm(pavaAlgorithm([1, 3, 2, 4]), { maxSteps: 20 }).state, 4) // => [1, 2.5, 2.5, 4]
 * ```
 */
export function pavaFitted(state: PavaState, n: number, sign: 1 | -1 = 1): number[] {
  const out = new Array<number>(n).fill(Number.NaN)
  const means = pavaMeans(state, sign)
  state.starts.forEach((start, j) => {
    const end = j + 1 < state.starts.length ? (state.starts[j + 1] ?? n) : state.pushed
    for (let i = start; i < end; i++) out[i] = means[j] ?? Number.NaN
  })
  return out
}

/**
 * Pool adjacent violators (Ayer et al. 1955) as a stepped algorithm on a stack of blocks: when the top two blocks
 * violate the order (the lower block's mean is not below the upper's) a step pools them into their weighted mean;
 * otherwise a step pushes the next observation as a block. It is done when every observation is pushed and nothing
 * violates, and the blocks are then the non-decreasing (or, with `direction`, non-increasing) least-squares fit. The
 * state serialises.
 *
 * @param y - The values, in order.
 * @param settings - weights and the direction of the fit.
 * @returns The algorithm.
 * @throws OmaiError with code `shape` for weights of the wrong length and `value` for a negative or non-finite weight.
 * @concept isotonic-regression
 * @example
 * ```ts
 * import { pavaAlgorithm, pavaFitted } from 'omai-engine/learn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(pavaAlgorithm([1, 3, 2, 4]), { maxSteps: 20 })
 * [run.steps, run.state.merges, pavaFitted(run.state, 4)] // => [5, 1, [1, 2.5, 2.5, 4]]
 * ```
 */
export function pavaAlgorithm(y: ArrayLike<number>, settings: PavaSettings = {}): Algorithm<PavaState, PavaPath> {
  const n = y.length
  const sign = isotonicDirections.make(settings.direction)
  const v = Float64Array.from(y, (t) => sign * t)
  const w = settings.weights === undefined ? new Float64Array(n).fill(1) : Float64Array.from(settings.weights)
  if (w.length !== n) throw new OmaiError('shape', `pavaAlgorithm: ${w.length} weights for ${n} values`)
  for (const t of w)
    if (!(Number.isFinite(t) && t >= 0)) throw new OmaiError('value', 'pavaAlgorithm: weights must be finite and >= 0')
  return {
    name: 'learn/pava',
    config: { n, direction: isotonicDirections.optionOf(settings.direction).name },
    schema: channels({
      metrics: {
        blocks: { description: 'blocks on the stack' },
        error: { description: 'weighted squared error of the fit so far', direction: 'lower' },
      },
    }),
    init: () => ({ pushed: 0, starts: [], weight: [], sum: [], sumSquares: [], merges: 0, merged: false, error: 0 }),
    step: (s, ctx) => {
      const k = s.starts.length
      if (topViolates(s)) {
        const [wa, sa, qa] = [s.weight[k - 2] ?? 0, s.sum[k - 2] ?? 0, s.sumSquares[k - 2] ?? 0]
        const [wb, sb, qb] = [s.weight[k - 1] ?? 0, s.sum[k - 1] ?? 0, s.sumSquares[k - 1] ?? 0]
        s.starts.pop()
        s.weight.length = k - 1
        s.sum.length = k - 1
        s.sumSquares.length = k - 1
        s.weight[k - 2] = wa + wb
        s.sum[k - 2] = sa + sb
        s.sumSquares[k - 2] = qa + qb
        s.error += blockError(wa + wb, sa + sb, qa + qb) - blockError(wa, sa, qa) - blockError(wb, sb, qb)
        s.merges += 1
        s.merged = true
      } else {
        const i = s.pushed
        s.starts.push(i)
        s.weight.push(w[i] ?? 0)
        s.sum.push((w[i] ?? 0) * (v[i] ?? 0))
        s.sumSquares.push((w[i] ?? 0) * (v[i] ?? 0) ** 2)
        s.pushed = i + 1
        s.merged = false
      }
      ctx.log('pava/blocks', s.starts.length)
      if (ctx.wants('pava/means')) ctx.log('pava/means', pavaMeans(s, sign))
      ctx.log('pava/merge', s.merged ? 1 : 0)
      ctx.log('pava/violations', topViolates(s) ? 1 : 0)
      ctx.log('pava/error', Math.max(0, s.error))
      return s
    },
    done: (s) => (s.pushed === n && !topViolates(s) ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PavaState,
    observe: (s) => ({ metrics: { blocks: s.starts.length, error: Math.max(0, s.error) } }),
  }
}
