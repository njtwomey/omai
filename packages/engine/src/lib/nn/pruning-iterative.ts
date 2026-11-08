/**
 * Iterative magnitude pruning with rewinding (Frankle and Carbin 2019) as a stepped algorithm: each step is one
 * train-prune-rewind round, with where the smallest weights are ranked (`pruneScopes`) and how the sparsity rises
 * (`pruneRoundSchedules`) as slots. `iterativeMagnitudePruning` in this file is that run to the end.
 */

import { OmaiError } from '../base/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor, toFlat, treeMap, treePaths, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import type { Params } from './module/module.ts'
import { applyMasks, type Masks, rewind } from './pruning.ts'
import { pruneRoundSchedules, pruneScopes } from './pruning-options.ts'

const checkSparsity = (s: number, name: string): void => {
  if (!(s >= 0 && s <= 1)) throw new OmaiError('value', `${name}: sparsity must be in [0, 1], got ${s}`)
}

const onesOf = (v: Value): Tensor =>
  fromStorage(new Float64Array(v.shape.reduce((a, b) => a * b, 1)).fill(1), [...v.shape], 'float64')

/** Options of {@link iterativeMagnitudePruning}. */
export interface IterativePruningOptions {
  /** Pruning rounds (default 3). */
  readonly rounds?: number
  /** The sparsity after the last round. */
  readonly finalSparsity: number
  /** Rewind to these parameters after each round (default the initial ones). */
  readonly rewindTo?: Params
  /** Which leaves to prune (default: at least two axes). */
  readonly filter?: (path: string, value: Value) => boolean
  /** Where the smallest weights are ranked, a value of `pruneScopes.space` (default the whole network together). */
  readonly scope?: SpaceValue
  /** How the sparsity rises, a value of `pruneRoundSchedules.space` (default geometric, the lottery-ticket schedule). */
  readonly schedule?: SpaceValue
  /** Scores the round's trained, masked parameters (a loss to log as `prune/loss`). */
  readonly evaluate?: (trained: Params, masks: Masks, round: number) => number
}

/** Result of {@link iterativeMagnitudePruning}. */
export interface IterativePruningResult {
  /** The final masks. */
  readonly masks: Masks
  /** The winning ticket: the rewound parameters under the final masks, ready to retrain. */
  readonly ticket: Params
  /** Per round: the target sparsity and the trained (masked) parameters. */
  readonly rounds: readonly { readonly sparsity: number; readonly trained: Params }[]
}

/** The paths {@link iterativeMagnitudePruningAlgorithm} logs. */
export type PrunePath = 'prune/round' | 'prune/sparsity' | 'prune/loss' | 'prune/mask'

/** What a stepped pruning logs: the round, the sparsity reached, the round's loss and the masks as one vector. */
export const pruneContract: Contract<unknown, PrunePath> = defineContract({
  'prune/round': scalar({ description: 'the prune-rewind-retrain round just done (from 1)' }),
  'prune/sparsity': scalar({ description: 'the fraction of eligible weights pruned after the round' }),
  'prune/loss': scalar({
    description: 'the loss of the round’s trained, masked parameters (when `evaluate` is given)',
    direction: 'lower',
  }),
  'prune/mask': tensorSource({
    role: 'state',
    shape: ['weights'],
    description: 'the masks after the round, all leaves flattened in order (1 kept, 0 pruned)',
  }),
})

/**
 * The state of {@link iterativeMagnitudePruningAlgorithm}: the rounds done, the parameters to train next (the rewound
 * survivors), the masks so far, and each round's target sparsity and trained parameters.
 */
export interface PruningState {
  readonly round: number
  readonly params: Params
  readonly masks: Masks
  readonly history: readonly { readonly sparsity: number; readonly trained: Params }[]
}

/**
 * Iterative magnitude pruning with rewinding (the lottery-ticket procedure) as a stepped algorithm: each step trains
 * from the current parameters under the current masks (the caller's `train`), prunes a further share of the survivors
 * by magnitude (the scope and the schedule are slots), and rewinds the survivors. The state, parameter trees included,
 * serialises; `train` is not part of it, so a resumed run needs the same function.
 *
 * @param initial - The initial parameters.
 * @param train - Trains from parameters under masks (it should keep pruned weights at zero) and returns the result.
 * @param options - Rounds, final sparsity, rewinding point, leaf filter, scope, schedule and an evaluator.
 * @returns The algorithm, one step per round.
 * @throws OmaiError with code `value` for a final sparsity outside [0, 1].
 * @concept pruning
 * @example
 * ```ts
 * import { iterativeMagnitudePruningAlgorithm } from 'omai-engine/nn'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { tensor } from 'omai-engine/tensor'
 * const alg = iterativeMagnitudePruningAlgorithm({ w: tensor([[1, 2, 3, 4]]) }, (p) => p, { finalSparsity: 0.5, rounds: 2 })
 * runAlgorithm(alg, { maxSteps: 5 }).state.round // => 2
 * ```
 */
export function iterativeMagnitudePruningAlgorithm(
  initial: Params,
  train: (params: Params, masks: Masks, round: number) => Params,
  options: IterativePruningOptions,
): Algorithm<PruningState, PrunePath> {
  checkSparsity(options.finalSparsity, 'iterativeMagnitudePruning')
  const rounds = options.rounds ?? 3
  const target = options.rewindTo ?? initial
  const prune = pruneScopes.make(options.scope)
  const sparsityAt = pruneRoundSchedules.make(options.schedule)
  return {
    name: 'iterative-magnitude-pruning',
    config: {
      rounds,
      finalSparsity: options.finalSparsity,
      scope: pruneScopes.optionOf(options.scope).name,
      schedule: pruneRoundSchedules.optionOf(options.schedule).name,
    },
    init: () => ({ round: 0, params: initial, masks: treeMap((v) => onesOf(v), initial), history: [] }),
    step: (s, ctx) => {
      const r = s.round + 1
      const trained = applyMasks(train(s.params, s.masks, r - 1), s.masks)
      const sparsity = sparsityAt(r, rounds, options.finalSparsity)
      checkSparsity(sparsity, 'iterativeMagnitudePruning')
      const masks = prune(trained, sparsity, {
        masks: s.masks,
        ...(options.filter === undefined ? {} : { filter: options.filter }),
      })
      ctx.log('prune/round', r)
      ctx.log('prune/sparsity', sparsity)
      if (options.evaluate !== undefined) ctx.log('prune/loss', options.evaluate(trained, masks, r))
      if (ctx.wants('prune/mask'))
        ctx.log(
          'prune/mask',
          Float64Array.from(Object.values(treePaths(masks)).flatMap((m) => Array.from(toFlat(m as Value)))),
        )
      return {
        round: r,
        params: rewind(target, masks),
        masks,
        history: [...s.history, { sparsity, trained }],
      }
    },
    done: (s) => (s.round >= rounds ? 'completed' : undefined),
    observe: (s) => ({
      metrics: s.history.length === 0 ? {} : { sparsity: s.history[s.history.length - 1]?.sparsity ?? 0 },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PruningState,
  }
}

/**
 * Iterative magnitude pruning with rewinding (the lottery-ticket procedure): train, prune a further fraction of the
 * surviving weights globally (geometric targets 1 - (1 - s)^(r / R)), rewind the survivors, repeat. This is
 * {@link iterativeMagnitudePruningAlgorithm} run to the end.
 *
 * @param initial - The initial parameters.
 * @param train - Trains from parameters under masks (it should keep pruned weights at zero) and returns the result.
 * @param options - Rounds, final sparsity, rewinding point, leaf filter, scope and schedule.
 * @returns The final masks, the ticket and each round.
 * @example
 * ```ts
 * import { iterativeMagnitudePruning } from 'omai-engine/nn'
 * import { tensor } from 'omai-engine/tensor'
 * iterativeMagnitudePruning({ w: tensor([[1, 2, 3, 4]]) }, (p) => p, { finalSparsity: 0.5, rounds: 2 }).rounds.length // => 2
 * ```
 */
export function iterativeMagnitudePruning(
  initial: Params,
  train: (params: Params, masks: Masks, round: number) => Params,
  options: IterativePruningOptions,
): IterativePruningResult {
  const algorithm = iterativeMagnitudePruningAlgorithm(initial, train, options)
  const { state } = runAlgorithm(algorithm, { maxSteps: options.rounds ?? 3 })
  return { masks: state.masks, ticket: state.params, rounds: [...state.history] }
}
