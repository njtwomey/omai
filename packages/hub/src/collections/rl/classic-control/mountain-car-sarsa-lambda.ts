import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `mountain-car/sarsa-lambda`. */
export const mountainCarSarsaLambda: HubEntry = playgroundEntry(
  'mountain-car/sarsa-lambda',
  'classic-control',
  'mountain-car-sarsa-lambda',
  {
    tags: ['tabular', 'eligibility traces', 'discretisation'],
    agent: { file: 'methods/rl/agents/tabular.ts', name: 'sarsaLambda' },
    references: ['Sutton and Barto (2018), Reinforcement Learning: An Introduction, section 12.7 and example 10.1.'],
  },
)
