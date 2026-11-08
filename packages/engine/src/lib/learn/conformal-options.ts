/**
 * The slot of adaptive conformal inference that varies as an option table: how much of the past residuals the interval
 * is calibrated on, either all of them or the most recent window (the window lets the intervals follow a drifting
 * residual scale). The table is one `union` space for controls, grids and tuners, and builds the selection from a
 * configuration.
 */

import { dict, int, type Option, type OptionTable, optionTable } from '../space/index.ts'

/** Picks the scores a quantile is taken over from all the scores so far, oldest first. */
export type ScoreMemory = (scores: readonly number[]) => readonly number[]

/** What a memory does, for validity rules and selectors. */
export interface ScoreMemoryProperties {
  /** Whether old scores are forgotten (so the intervals can follow drift). */
  readonly forgets: boolean
}

/**
 * The score memories of adaptive conformal inference as one slot: `all` (every past score, the default) and `window`
 * (the last `size` scores).
 *
 * @concept conformal-prediction
 * @example
 * ```ts
 * import { scoreMemories } from 'omai-engine/learn'
 * scoreMemories.make({ kind: 'window', size: 2 })([1, 2, 3, 4]) // => [3, 4]
 * ```
 */
export const scoreMemories: OptionTable<ScoreMemory, ScoreMemoryProperties> = optionTable(
  {
    all: {
      description: 'calibrate on every score seen so far',
      properties: { forgets: false },
      space: dict({}),
      make: (): ScoreMemory => (scores) => scores,
    } satisfies Option<ScoreMemory, ScoreMemoryProperties>,
    window: {
      description: 'calibrate on the most recent scores only',
      properties: { forgets: true },
      space: dict({ size: int(5, 5000, { default: 100, label: 'window' }) }),
      make: (config): ScoreMemory => {
        const size = typeof config['size'] === 'number' ? config['size'] : 100
        return (scores) => scores.slice(Math.max(0, scores.length - size))
      },
    } satisfies Option<ScoreMemory, ScoreMemoryProperties>,
  },
  { default: 'all', label: 'score memory', description: 'Which past residuals the interval is calibrated on.' },
)
