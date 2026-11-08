/**
 * The slot of recursive feature elimination that varies as an option table: how many features go in a round, either a
 * fixed number (sklearn's integer `step`) or a fraction of those still in (a geometric schedule that takes big bites
 * early and small ones near the target). The table is one `union` space for controls, grids and tuners, and builds the
 * schedule from a configuration.
 */

import { dict, int, type Option, type OptionTable, optionTable, real } from '../space/index.ts'

/** How many features to drop in a round, given the features left and the number wanted at the end (at least 1). */
export type EliminationSchedule = (remaining: number, target: number) => number

/** What a schedule does, for validity rules and selectors. */
export interface EliminationScheduleProperties {
  /** Whether the bite shrinks as the features are used up (a fraction of what is left). */
  readonly geometric: boolean
}

/**
 * The elimination schedules of RFE as one slot: `fixed` (the same number of features per round, the default) and
 * `fraction` (a fraction of the features still in, at least one).
 *
 * @concept feature-selection
 * @example
 * ```ts
 * import { eliminationSchedules } from 'omai-engine/learn'
 * eliminationSchedules.make({ kind: 'fraction', fraction: 0.5 })(10, 2) // => 5
 * eliminationSchedules.make({ kind: 'fixed', step: 3 })(10, 9) // => 1
 * ```
 */
export const eliminationSchedules: OptionTable<EliminationSchedule, EliminationScheduleProperties> = optionTable(
  {
    fixed: {
      description: 'drop the same number of features every round',
      properties: { geometric: false },
      space: dict({ step: int(1, 100, { default: 1, label: 'features per round' }) }),
      make: (config): EliminationSchedule => {
        const step = typeof config['step'] === 'number' ? config['step'] : 1
        return (remaining, target) => Math.max(1, Math.min(step, remaining - target))
      },
    } satisfies Option<EliminationSchedule, EliminationScheduleProperties>,
    fraction: {
      description: 'drop a fraction of the features still in, at least one',
      properties: { geometric: true },
      space: dict({ fraction: real(0.05, 0.95, { default: 0.5, label: 'fraction per round' }) }),
      make: (config): EliminationSchedule => {
        const f = typeof config['fraction'] === 'number' ? config['fraction'] : 0.5
        return (remaining, target) => Math.max(1, Math.min(Math.floor(f * remaining), remaining - target))
      },
    } satisfies Option<EliminationSchedule, EliminationScheduleProperties>,
  },
  { default: 'fixed', label: 'elimination schedule', description: 'How many features are dropped in each round.' },
)
