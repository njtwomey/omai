import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `cliff-walking/expected-sarsa`. */
export const cliffWalkingExpectedSarsa: HubEntry = playgroundEntry(
  'cliff-walking/expected-sarsa',
  'gridworlds',
  'cliff-walking-expected-sarsa',
  {
    tags: ['tabular', 'temporal difference'],
    agent: { file: 'methods/rl/agents/tabular.ts', name: 'expectedSarsa' },
    references: ['van Seijen et al. (2009), A theoretical and empirical analysis of Expected Sarsa.'],
  },
)
