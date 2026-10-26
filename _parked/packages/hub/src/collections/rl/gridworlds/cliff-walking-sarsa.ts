/**
 * Hub entry for SARSA on cliff walking, where it learns the safer path away from the edge, from the
 * reinforcement-learning playground.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `cliff-walking/sarsa`. */
export const cliffWalkingSarsa: HubEntry = playgroundEntry('cliff-walking/sarsa', 'gridworlds', 'cliff-walking-sarsa', {
  tags: ['tabular', 'temporal difference', 'on-policy'],
  agent: { file: 'methods/rl/agents/tabular.ts', name: 'sarsa' },
  references: ['Sutton and Barto (2018), Reinforcement Learning: An Introduction, example 6.6.'],
})
