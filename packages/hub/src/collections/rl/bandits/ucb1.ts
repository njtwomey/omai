import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `bandit/ucb1`. */
export const ucb1: HubEntry = playgroundEntry('bandit/ucb1', 'bandits', 'ucb1', {
  tags: ['bandits', 'exploration', 'upper confidence bounds'],
  agent: { file: 'methods/rl/agents/bandits.ts', name: 'ucb1' },
  references: ['Auer, Cesa-Bianchi and Fischer (2002), Finite-time analysis of the multiarmed bandit problem.'],
})
