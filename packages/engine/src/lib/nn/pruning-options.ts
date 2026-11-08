/**
 * The slots of iterative magnitude pruning as option tables: where the smallest weights are ranked (`pruneScopes`: the
 * whole network together, so layers lose different fractions, or each tensor on its own) and how the sparsity rises
 * round by round to its final value (`pruneRoundSchedules`: the lottery-ticket geometric schedule, linear, or the
 * polynomial schedule of Zhu and Gupta). Each table is one `union` space for controls, grids and tuners.
 */

import { dict, int, type Option, type OptionTable, optionTable } from '../space/index.ts'
import {
  fromStorage,
  primalOf,
  type Tensor,
  treeFlatten,
  treePaths,
  treeUnflatten,
  type Value,
} from '../tensor/index.ts'
import type { Params } from './module/module.ts'
import { globalMagnitudeMasks, type Masks, magnitudeMask } from './pruning.ts'

/** Builds the next masks from the trained parameters, a sparsity and the masks so far. */
export type PruneScope = (
  params: Params,
  sparsity: number,
  options: { readonly filter?: (path: string, value: Value) => boolean; readonly masks?: Masks },
) => Masks

/** What a scope does, for validity rules and selectors. */
export interface PruneScopeProperties {
  /** Whether layers can end up with different sparsities. */
  readonly layersDiffer: boolean
}

const onesOf = (v: Value): Tensor =>
  fromStorage(new Float64Array(v.shape.reduce((a, b) => a * b, 1)).fill(1), [...v.shape], 'float64')

/**
 * The pruning scopes as one slot: `global` (rank every eligible weight of the network together, as
 * torch.nn.utils.prune.global_unstructured) and `layerwise` (prune the same fraction of each eligible tensor).
 *
 * @concept pruning
 * @example
 * ```ts
 * import { pruneScopes } from 'omai-engine/nn'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const masks = pruneScopes.make({ kind: 'layerwise' })({ a: tensor([[1, 2]]), b: tensor([[3, 0.5]]) }, 0.5, {})
 * [toArray(masks['a'] as never), toArray(masks['b'] as never)] // => [[[0, 1]], [[1, 0]]]
 * ```
 */
export const pruneScopes: OptionTable<PruneScope, PruneScopeProperties> = optionTable(
  {
    global: {
      description: 'rank all eligible weights of the network together',
      properties: { layersDiffer: true },
      space: dict({}),
      make: (): PruneScope => (params, sparsity, options) => globalMagnitudeMasks(params, sparsity, options),
    } satisfies Option<PruneScope, PruneScopeProperties>,
    layerwise: {
      description: 'prune the same fraction of every eligible tensor',
      properties: { layersDiffer: false },
      space: dict({}),
      make: (): PruneScope => (params, sparsity, options) => {
        const filter = options.filter ?? ((_p: string, v: Value) => v.shape.length >= 2)
        const entries = Object.entries(treePaths(params))
        const { def } = treeFlatten(params)
        const leaves = entries.map(([path, v]) => (filter(path, v) ? magnitudeMask(primalOf(v), sparsity) : onesOf(v)))
        return treeUnflatten(def, leaves) as Masks
      },
    } satisfies Option<PruneScope, PruneScopeProperties>,
  },
  { default: 'global', label: 'pruning scope', description: 'Where the smallest weights are ranked.' },
)

/** The sparsity after round `round` (from 1) of `rounds`, given the final sparsity. */
export type RoundSchedule = (round: number, rounds: number, finalSparsity: number) => number

/** What a round schedule does, for validity rules and selectors. */
export interface RoundScheduleProperties {
  /** Whether early rounds prune the most (a fraction of what is left, or a front-loaded curve). */
  readonly frontLoaded: boolean
}

/**
 * The round schedules as one slot: `geometric` (each round prunes the same fraction of the survivors, the lottery-ticket
 * procedure: 1 - (1 - s)^(r / R)), `linear` (s r / R) and `polynomial` (Zhu and Gupta: s (1 - (1 - r / R)^power), fast
 * early and slow near the end).
 *
 * @concept pruning
 * @example
 * ```ts
 * import { pruneRoundSchedules } from 'omai-engine/nn'
 * pruneRoundSchedules.make({ kind: 'linear' })(1, 4, 0.8) // => 0.2
 * pruneRoundSchedules.make({ kind: 'geometric' })(2, 2, 0.75) // => 0.75
 * ```
 */
export const pruneRoundSchedules: OptionTable<RoundSchedule, RoundScheduleProperties> = optionTable(
  {
    geometric: {
      description: 'prune the same fraction of the survivors every round (the lottery-ticket procedure)',
      properties: { frontLoaded: true },
      space: dict({}),
      make: (): RoundSchedule => (r, rounds, s) => 1 - (1 - s) ** (r / rounds),
    } satisfies Option<RoundSchedule, RoundScheduleProperties>,
    linear: {
      description: 'raise the sparsity by the same amount every round',
      properties: { frontLoaded: false },
      space: dict({}),
      make: (): RoundSchedule => (r, rounds, s) => (s * r) / rounds,
    } satisfies Option<RoundSchedule, RoundScheduleProperties>,
    polynomial: {
      description: 'Zhu and Gupta: s (1 - (1 - r / R)^power), fast early and slow near the end',
      properties: { frontLoaded: true },
      space: dict({ power: int(1, 6, { default: 3, label: 'power' }) }),
      make: (config): RoundSchedule => {
        const power = typeof config['power'] === 'number' ? config['power'] : 3
        return (r, rounds, s) => s * (1 - (1 - r / rounds) ** power)
      },
    } satisfies Option<RoundSchedule, RoundScheduleProperties>,
  },
  { default: 'geometric', label: 'round schedule', description: 'How the sparsity rises from round to round.' },
)
