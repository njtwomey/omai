/**
 * Hub entry for Thompson sampling, which pulls each arm with the posterior probability that it is the best, on a
 * multi-armed bandit from the reinforcement-learning playground.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `bandit/thompson`. */
export const thompsonSampling: HubEntry = playgroundEntry('bandit/thompson', 'bandits', 'thompson-sampling', {
  tags: ['bandits', 'exploration', 'Bayesian'],
  agent: { file: 'methods/rl/agents/bandits.ts', name: 'thompsonBernoulli' },
  references: ['Agrawal and Goyal (2012), Analysis of Thompson sampling for the multi-armed bandit problem.'],
})
