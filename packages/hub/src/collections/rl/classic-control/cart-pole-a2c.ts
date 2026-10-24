import type { HubEntry } from '../../../entry/entry.ts'
import { playgroundEntry } from '../_playground.ts'

/** The playground pairing `cart-pole/a2c`. */
export const cartPoleA2c: HubEntry = playgroundEntry('cart-pole/a2c', 'classic-control', 'cart-pole-a2c', {
  tags: ['actor-critic', 'policy gradient', 'neural networks'],
  agent: { file: 'methods/rl/agents/policy-gradient.ts', name: 'a2c' },
  references: ['Mnih et al. (2016), Asynchronous methods for deep reinforcement learning.'],
})
