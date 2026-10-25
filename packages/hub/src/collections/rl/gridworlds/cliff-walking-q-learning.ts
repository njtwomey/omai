/**
 * Hub entry for Q-learning on cliff walking, where it learns the risky path along the cliff edge (Sutton and Barto,
 * example 6.6), from the reinforcement-learning playground.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `cliff-walking/q-learning`. */
export const cliffWalkingQLearning: HubEntry = playgroundEntry(
  'cliff-walking/q-learning',
  'gridworlds',
  'cliff-walking-q-learning',
  {
    tags: ['tabular', 'temporal difference', 'off-policy'],
    agent: { file: 'methods/rl/agents/tabular.ts', name: 'qLearning' },
    references: ['Sutton and Barto (2018), Reinforcement Learning: An Introduction, example 6.6.'],
  },
)
