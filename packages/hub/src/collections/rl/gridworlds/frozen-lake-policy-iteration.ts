/**
 * Hub entry for policy iteration planning a little after every episode on FrozenLake, from the reinforcement-learning
 * playground.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `frozen-lake/policy-iteration`. */
export const frozenLakePolicyIteration: HubEntry = playgroundEntry(
  'frozen-lake/policy-iteration',
  'gridworlds',
  'frozen-lake-policy-iteration',
  {
    tags: ['dynamic programming', 'planning', 'model-based'],
    agent: { file: 'methods/rl/agents/planning.ts', name: 'sweepingPlannerAgent' },
    references: ['Howard (1960), Dynamic Programming and Markov Processes.'],
  },
)
