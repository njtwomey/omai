import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `cart-pole/reinforce`. */
export const cartPoleReinforce: HubEntry = playgroundEntry(
  'cart-pole/reinforce',
  'classic-control',
  'cart-pole-reinforce',
  {
    tags: ['policy gradient', 'neural networks'],
    agent: { file: 'methods/rl/agents/policy-gradient.ts', name: 'reinforce' },
    references: [
      'Williams (1992), Simple statistical gradient-following algorithms for connectionist reinforcement learning.',
    ],
  },
)
