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

/**
 * An environment as pure functions over an explicit state (the gymnax design): `reset` and `step` take keys and return
 * new states, so episodes are reproducible, can be replayed from any state, and run side by side. Every environment
 * renders its state as a {@link Scene}.
 *
 * @concept environments
 */
export interface Env<S, O, A> {
  readonly spec: EnvSpec
  readonly observationSpace: Space
  readonly actionSpace: Space
  reset(key: Key): { readonly state: S; readonly observation: O; readonly info: Readonly<Record<string, Json>> }
  step(state: S, action: A, key: Key): Step<S, O>
  render(state: S): Scene
}

/** A named environment factory with typed options (what `make` looks up). */
export interface EnvDefinition<S = unknown, O = unknown, A = unknown> {
  readonly id: string
  readonly description: string
  readonly options?: DictSpace
  create(options?: Readonly<Record<string, unknown>>): Env<S, O, A>
}
