import type { Json } from '../base/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { MetricChannel } from '../trace/index.ts'

/** A policy: an action for an observation (and a key, for stochastic policies). */
export type Policy<O, A> = (observation: O, key: Key) => A

/** One step of experience, as agents learn from it. */
export interface Transition<O, A> {
  readonly observation: O
  readonly action: A
  readonly reward: number
  readonly nextObservation: O
  /** The episode reached a terminal state (no bootstrapping from `nextObservation`). */
  readonly terminated: boolean
  /** The episode was cut short (bootstrap from `nextObservation`, then reset). */
  readonly truncated: boolean
  readonly info?: Readonly<Record<string, Json>>
}

/**
 * A learning agent as pure functions over an explicit state `G` (parameters, optimiser state, buffers, counters): `act`
 * chooses an exploratory action, `learn` folds one transition into the state, and `greedy` is the action the agent
 * would take when evaluated. Agents never mutate their state, so a run can be checkpointed and replayed from any step.
 * `stats` reports what the agent wants traced (epsilon, losses, TD errors), declared in `metrics`.
 *
 * @concept agents
 */
export interface Agent<G, O, A> {
  readonly name: string
  readonly config: Json
  /** The metric channels `stats` reports. */
  readonly metrics?: Readonly<Record<string, MetricChannel>>
  init(key: Key): G
  act(state: G, observation: O, key: Key): A
  learn(state: G, transition: Transition<O, A>, key: Key): G
  greedy(state: G, observation: O): A
  stats?(state: G): Readonly<Record<string, number>>
}

/**
 * The exploratory policy of an agent in a fixed state (its behaviour policy, frozen).
 *
 * @param agent - The agent.
 * @param state - The agent state.
 * @returns The policy `(observation, key) => agent.act(state, observation, key)`.
 * @concept agents
 * @example
 * ```ts
 * import { behaviourPolicy, type Agent } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * const always: Agent<null, number, number> = {
 *   name: 'always', config: {}, init: () => null, act: () => 1, learn: (s) => s, greedy: () => 1,
 * }
 * behaviourPolicy(always, null)(0, key(0)) // => 1
 * ```
 */
export function behaviourPolicy<G, O, A>(agent: Agent<G, O, A>, state: G): Policy<O, A> {
  return (observation, k) => agent.act(state, observation, k)
}

/**
 * The greedy (evaluation) policy of an agent in a fixed state.
 *
 * @param agent - The agent.
 * @param state - The agent state.
 * @returns The policy `(observation) => agent.greedy(state, observation)`.
 * @concept agents
 * @example
 * ```ts
 * import { greedyPolicyOf, type Agent } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * const echo: Agent<null, number, number> = {
 *   name: 'echo', config: {}, init: () => null, act: (_, o) => o, learn: (s) => s, greedy: (_, o) => o,
 * }
 * greedyPolicyOf(echo, null)(3, key(0)) // => 3
 * ```
 */
export function greedyPolicyOf<G, O, A>(agent: Agent<G, O, A>, state: G): Policy<O, A> {
  return (observation) => agent.greedy(state, observation)
}

/**
 * An agent that follows a fixed policy and never learns (for baselines and for evaluating hand-written controllers
 * with `trainAgent`).
 *
 * @param policy - The policy.
 * @param name - The agent's name (default `'policy'`).
 * @returns The agent; its state is `null`.
 * @concept agents
 * @example
 * ```ts
 * import { policyAgent } from 'whitebox-engine/rl'
 * import { key } from 'whitebox-engine/random'
 * const a = policyAgent((o: number) => o + 1)
 * a.act(a.init(key(0)), 1, key(1)) // => 2
 * ```
 */
export function policyAgent<O, A>(policy: Policy<O, A>, name = 'policy'): Agent<null, O, A> {
  return {
    name,
    config: {},
    init: () => null,
    act: (_, o, k) => policy(o, k),
    learn: (s) => s,
    // A fixed policy evaluated greedily is the same policy; a fixed key keeps it deterministic.
    greedy: (_, o) => policy(o, GREEDY_KEY),
  }
}

const GREEDY_KEY: Key = makeKey(0x67726565)
