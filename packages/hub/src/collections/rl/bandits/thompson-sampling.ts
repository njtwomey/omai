import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `bandit/thompson`. */
export const thompsonSampling: HubEntry = playgroundEntry('bandit/thompson', 'bandits', 'thompson-sampling', {
  tags: ['bandits', 'exploration', 'Bayesian'],
  agent: { file: 'methods/rl/agents/bandits.ts', name: 'thompsonBernoulli' },
  references: ['Agrawal and Goyal (2012), Analysis of Thompson sampling for the multi-armed bandit problem.'],
})
