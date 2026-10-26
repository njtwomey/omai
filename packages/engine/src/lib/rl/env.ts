/**
 * The environment contract, gymnax style: environments as pure functions whose `reset` and `step` take keys and return
 * new states, so episodes are reproducible and can be replayed from any state. Also step and reset results, explicit
 * tabular models (gymnasium's `P` table), oracles for regret, environment specs and named definitions, and unwrapping.
 */

import type { Json } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import type { DictSpace, Space } from '../space/index.ts'

/** Static facts about an environment (gymnasium's EnvSpec). */
export interface EnvSpec {
  readonly id: string
  readonly description: string
  /** Episodes are truncated after this many steps by `make` (gymnasium's TimeLimit). */
  readonly maxEpisodeSteps?: number
  /** The average return that counts as solving the task. */
  readonly rewardThreshold?: number
  /** Whether `step` ignores its key (the dynamics are deterministic given the state). */
  readonly deterministic?: boolean
}

/** The result of a step. */
export interface Step<S, O> {
  readonly state: S
  readonly observation: O
  readonly reward: number
  /** The episode reached a terminal state of the task. */
  readonly terminated: boolean
  /** The episode was cut short (a time limit) without reaching a terminal state. */
  readonly truncated: boolean
  readonly info: Readonly<Record<string, Json>>
}

/** The result of a reset. */
export interface Reset<S, O> {
  readonly state: S
  readonly observation: O
  readonly info: Readonly<Record<string, Json>>
}

/** One possible outcome of an action in a tabular model (gymnasium's `P[s][a]` entries). */
export interface Outcome<S> {
  readonly probability: number
  readonly state: S
  readonly reward: number
  readonly terminated: boolean
}

/**
 * An explicit model of a finite environment, for planning and exact solutions: the enumerated states and actions, the
 * initial-state distribution and every outcome of every action (gymnasium's `P` table).
 */
export interface EnvModel<S, A> {
  readonly states: readonly S[]
  readonly actions: readonly A[]
  /** The index of a state in `states`. */
  stateIndex(state: S): number
  /** The index of an action in `actions`. */
  actionIndex(action: A): number
  readonly initial: readonly { readonly probability: number; readonly state: S }[]
  transitions(state: S, action: A): readonly Outcome<S>[]
}

/** Ground truth an environment can reveal for computing regret (bandits and other problems with known values). */
export interface EnvOracle<S, A> {
  /** The expected immediate reward of an action in a state. */
  expectedReward(state: S, action: A): number
  /** The best expected immediate reward available in a state. */
  optimalExpectedReward(state: S): number
  /** The best action in a state. */
  optimalAction?(state: S): A
}

/**
 * An environment as pure functions over an explicit state (the gymnax design): `reset` and `step` take keys and return
 * new states, so episodes are reproducible, can be replayed from any state, and run side by side. Every environment
 * renders its state as a {@link Scene}. Finite environments may expose a `model` (planning, exact solutions) and
 * problems with known values an `oracle` (regret); wrappers record the environment they wrap in `wrapped`.
 *
 * @concept environments
 */
export interface Env<S, O, A> {
  readonly spec: EnvSpec
  readonly observationSpace: Space
  readonly actionSpace: Space
  reset(key: Key): Reset<S, O>
  /**
   * Start a new episode after `previous` ended, carrying what persists across episodes (running statistics of
   * normalising wrappers, episode counters). Environments without such state omit it; use {@link resetAfter}.
   */
  resetFrom?(previous: S, key: Key): Reset<S, O>
  step(state: S, action: A, key: Key): Step<S, O>
  render(state: S): Scene
  /** The observation of a state (what `reset` and `step` return for it). */
  observe?(state: S): O
  readonly model?: EnvModel<S, A>
  readonly oracle?: EnvOracle<S, A>
  /** The environment this one wraps, if it is a wrapper. */
  readonly wrapped?: Env<unknown, unknown, unknown>
}

/** A named environment factory with typed options (what `make` looks up). */
export interface EnvDefinition<S = unknown, O = unknown, A = unknown> {
  readonly id: string
  readonly description: string
  readonly options?: DictSpace
  create(options?: Readonly<Record<string, unknown>>): Env<S, O, A>
}

/**
 * The innermost environment under a stack of wrappers (gymnasium's `env.unwrapped`).
 *
 * @param env - A possibly wrapped environment.
 * @returns The environment with no `wrapped` field.
 * @concept environments
 * @example
 * ```ts
 * import { make, unwrapped } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * unwrapped(make(cartPole)).spec.maxEpisodeSteps // => 500
 * ```
 */
export function unwrapped(env: Env<unknown, unknown, unknown>): Env<unknown, unknown, unknown> {
  let e = env
  while (e.wrapped !== undefined) e = e.wrapped
  return e
}

/**
 * Start the next episode: `env.resetFrom(previous, key)` when the environment carries state across episodes and a
 * previous state is given, else `env.reset(key)`. Every training loop, rollout and vector environment resets this way.
 *
 * @param env - The environment.
 * @param previous - The final state of the previous episode, if any.
 * @param key - The reset key.
 * @returns The reset.
 * @concept environments
 * @example
 * ```ts
 * import { resetAfter } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * resetAfter(cartPole.create(), undefined, key(0)).observation.length // => 4
 * ```
 */
export function resetAfter<S, O, A>(env: Env<S, O, A>, previous: S | undefined, key: Key): Reset<S, O> {
  return previous !== undefined && env.resetFrom !== undefined ? env.resetFrom(previous, key) : env.reset(key)
}

/**
 * Whether an object is a single-agent environment definition: it has an id, a description and a `create` function, and
 * is not a multi-agent definition (those carry a `multiAgent` field naming their form, `'aec'` or `'parallel'`).
 *
 * @param value - Anything.
 * @returns Whether it is an {@link EnvDefinition}.
 * @example
 * ```ts
 * import { isEnvDefinition } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * isEnvDefinition(cartPole) // => true
 * ```
 */
export function isEnvDefinition(value: unknown): value is EnvDefinition {
  if (typeof value !== 'object' || value === null) return false
  const v = value as { id?: unknown; description?: unknown; create?: unknown; multiAgent?: unknown }
  return (
    typeof v.id === 'string' &&
    typeof v.description === 'string' &&
    typeof v.create === 'function' &&
    typeof v.multiAgent !== 'string'
  )
}
