/**
 * Hub entry for value iteration planning a little after every episode on FrozenLake, from the reinforcement-learning
 * playground.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `frozen-lake/value-iteration`. */
export const frozenLakeValueIteration: HubEntry = playgroundEntry(
  'frozen-lake/value-iteration',
  'gridworlds',
  'frozen-lake-value-iteration',
  {
    tags: ['dynamic programming', 'planning', 'model-based'],
    agent: { file: 'methods/rl/agents/planning.ts', name: 'sweepingPlannerAgent' },
    references: ['Bellman (1957), Dynamic Programming.'],
  },
)
