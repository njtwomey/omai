/**
 * Hub entry for the epsilon-greedy bandit agent learning which arm of a multi-armed bandit pays best, from the
 * reinforcement-learning playground.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `bandit/epsilon-greedy`. */
export const epsilonGreedy: HubEntry = playgroundEntry('bandit/epsilon-greedy', 'bandits', 'epsilon-greedy', {
  tags: ['bandits', 'exploration'],
  agent: { file: 'methods/rl/agents/bandits.ts', name: 'epsilonGreedyBandit' },
  references: ['Sutton and Barto (2018), Reinforcement Learning: An Introduction, chapter 2.'],
})
