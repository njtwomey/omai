/**
 * Baseline agents that do not learn: `randomAgent` draws every action uniformly from the action space (discrete, box,
 * multi-discrete or composite) and `constantAgent` always takes the same action, for reference scores and for probing
 * an environment's dynamics.
 */

import type { Json } from '../base/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { Agent } from '../rl/index.ts'
import { type Space, sample } from '../space/index.ts'

/** What {@link randomAgent} needs from an environment: its action space (and, for type inference, its actions). */
export interface RandomAgentEnv<O, A> {
  readonly actionSpace: Space
  reset?(key: Key): { readonly observation: O }
  step?(state: never, action: A, key: Key): unknown
}

/**
 * The uniformly random agent: every action is a fresh draw from the action space (`sample` of `lib/space` with the
 * step key), so it works for discrete, box, multi-discrete and composite spaces. It never learns; its greedy action is
 * a draw with a fixed key. The baseline every learning agent is compared with.
 *
 * @param env - The environment (or anything with an action space).
 * @returns The agent; its state is `null`.
 * @concept agents
 * @example
 * ```ts
 * import { randomAgent } from 'omai-engine/agents'
 * import { cartPole } from 'omai-engine/environments'
 * import { key } from 'omai-engine/random'
 * const agent = randomAgent(cartPole.create())
 * [0, 1].includes(agent.act(null, [0, 0, 0, 0], key(3))) // => true
 * ```
 */
export function randomAgent<O, A>(env: RandomAgentEnv<O, A>): Agent<null, O, A> {
  const space = env.actionSpace
  return {
    name: 'random',
    config: { actionSpace: space.kind } as Json,
    init: () => null,
    act: (_, __, k) => sample(space, k) as A,
    learn: (s) => s,
    greedy: () => sample(space, GREEDY) as A,
  }
}

/**
 * The constant agent: always takes the same action (a baseline, and a way to probe an environment's dynamics).
 *
 * @param action - The action.
 * @returns The agent; its state is `null`.
 * @concept agents
 * @example
 * ```ts
 * import { constantAgent } from 'omai-engine/agents'
 * import { key } from 'omai-engine/random'
 * const agent = constantAgent(1)
 * agent.act(null, 0, key(0)) // => 1
 * ```
 */
export function constantAgent<A, O = unknown>(action: A): Agent<null, O, A> {
  return {
    name: 'constant',
    config: { action: action as unknown as Json },
    init: () => null,
    act: () => action,
    learn: (s) => s,
    greedy: () => action,
  }
}

// A fixed key keeps the greedy action of the random agent deterministic.
const GREEDY: Key = makeKey(0x67726565)
