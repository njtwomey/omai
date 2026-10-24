import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `frozen-lake/q-learning`. */
export const frozenLakeQLearning: HubEntry = playgroundEntry(
  'frozen-lake/q-learning',
  'gridworlds',
  'frozen-lake-q-learning',
  {
    tags: ['tabular', 'temporal difference', 'off-policy'],
    agent: { file: 'methods/rl/agents/tabular.ts', name: 'qLearning' },
  },
)
