/**
 * A gymnasium-like front door to the environments: listing the definitions a module exports, `make` (time limits and
 * wrappers applied, as `gymnasium.make`), and `GymEnv`, a stateful wrapper whose `reset` and `step` return gymnasium's
 * tuples for code written in that style.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import { type Space, sample as sampleSpace } from '../space/index.ts'
import { type Env, type EnvDefinition, type EnvSpec, isEnvDefinition, resetAfter } from './env.ts'
import { timeLimit } from './wrappers.ts'

/** Options of {@link make} besides the definition's own options. */
export interface MakeConfig {
  /** Override the spec's time limit; `null` disables it (gymnasium's `max_episode_steps`). */
  readonly maxEpisodeSteps?: number | null
  /** Wrappers applied in order after the time limit. */
  readonly wrappers?: readonly ((env: Env<unknown, unknown, unknown>) => Env<unknown, unknown, unknown>)[]
  /** Definitions to look an id up in, when `make` is given an id. */
  readonly registry?: readonly EnvDefinition[]
}

/**
 * Every environment definition exported by a module, sorted by id (gymnasium's registry, without registration side
 * effects: pass the imported module).
 *
 * @param module - An imported module (or any record of exports).
 * @returns The definitions it exports.
 * @concept environments
 * @example
 * ```ts
 * import { listEnvironments } from 'mdml-engine/rl'
 * import * as rl from 'mdml-engine/environments'
 * listEnvironments(rl).some((d) => d.id === 'CartPole-v1') // => true
 * ```
 */
export function listEnvironments(module: Readonly<Record<string, unknown>>): EnvDefinition[] {
  const seen = new Map<string, EnvDefinition>()
  for (const value of Object.values(module)) if (isEnvDefinition(value)) seen.set(value.id, value)
  return [...seen.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
}

function lookup(id: string, registry: readonly EnvDefinition[] | undefined): EnvDefinition {
  const found = registry?.find((d) => d.id === id)
  if (found === undefined) {
    throw new MdmlError(
      'value',
      `make: no environment '${id}' in the registry (pass registry: listEnvironments(module))`,
    )
  }
  return found
}

/**
 * Create an environment from its definition (or its id in a registry), applying the spec's time limit and then any
 * wrappers (gymnasium.make).
 *
 * @param definition - The environment definition, or its id when `config.registry` is given.
 * @param options - Options for the definition.
 * @param config - Time-limit override, wrappers and the registry.
 * @returns The environment.
 * @throws MdmlError with code `value` for an id missing from the registry.
 * @concept environments
 * @index function Make an environment by name with wrappers, like gymnasium.make.
 * @example
 * ```ts
 * import { make, listEnvironments } from 'mdml-engine/rl'
 * import * as rl from 'mdml-engine/environments'
 * make(rl.cartPole).spec.id // => 'CartPole-v1'
 * make('MountainCar-v0', {}, { registry: listEnvironments(rl) }).spec.maxEpisodeSteps // => 200
 * ```
 */
export function make<S, O, A>(
  definition: EnvDefinition<S, O, A> | string,
  options?: Readonly<Record<string, unknown>>,
  config: MakeConfig = {},
): Env<unknown, O, A> {
  const def = (typeof definition === 'string' ? lookup(definition, config.registry) : definition) as EnvDefinition<
    S,
    O,
    A
  >
  const env = def.create(options)
  const limit = config.maxEpisodeSteps === undefined ? env.spec.maxEpisodeSteps : config.maxEpisodeSteps
  let out = (limit === null || limit === undefined ? env : timeLimit(env, limit)) as Env<unknown, unknown, unknown>
  for (const w of config.wrappers ?? []) out = w(out)
  return out as Env<unknown, O, A>
}

/**
 * A gymnasium-style stateful wrapper around a pure environment: `reset({ seed })` and `step(action)` return
 * gymnasium's tuples, and the environment keeps its own state and key stream (for code written against gymnasium).
 * Statistics that persist across episodes (normalising wrappers) carry over between resets.
 *
 * @concept environments
 * @example
 * ```ts
 * import { GymEnv } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * const env = new GymEnv(cartPole)
 * env.reset({ seed: 0 })
 * env.step(1)[1] // => 1
 * ```
 */
export class GymEnv<O, A> {
  readonly env: Env<unknown, O, A>
  private current: unknown
  private key: Key = makeKey(0)
  private resets = 0
  private t = 0
  private done = true

  /**
   * @param source - An environment definition (made with {@link make}) or an already built environment.
   * @param options - Options for the definition.
   * @param config - Options for {@link make}.
   */
  constructor(
    source: EnvDefinition<unknown, O, A> | Env<unknown, O, A>,
    options?: Readonly<Record<string, unknown>>,
    config: MakeConfig = {},
  ) {
    this.env = isEnvDefinition(source) ? make(source, options, config) : source
  }

  get spec(): EnvSpec {
    return this.env.spec
  }

  get observationSpace(): Space {
    return this.env.observationSpace
  }

  get actionSpace(): Space {
    return this.env.actionSpace
  }

  /** The current state of the pure environment (for planning and replay). */
  get state(): unknown {
    return this.current
  }

  /** The number of steps taken in the current episode. */
  get elapsedSteps(): number {
    return this.t
  }

  /** Start an episode; `seed` restarts the key stream (as gymnasium's reset(seed=...)). */
  reset(options: { readonly seed?: number } = {}): [O, Readonly<Record<string, Json>>] {
    if (options.seed !== undefined) {
      this.key = makeKey(options.seed)
      this.resets = 0
      this.current = undefined
    }
    this.resets += 1
    const r = resetAfter(this.env, this.current, foldIn(this.key, 2 * this.resets))
    this.current = r.state
    this.t = 0
    this.done = false
    return [r.observation, r.info]
  }

  /** Advance one step: `[observation, reward, terminated, truncated, info]`. */
  step(action: A): [O, number, boolean, boolean, Readonly<Record<string, Json>>] {
    if (this.done) throw new MdmlError('value', 'GymEnv.step: call reset() first (the episode has ended)')
    this.t += 1
    const r = this.env.step(this.current, action, foldIn(foldIn(this.key, 2 * this.resets + 1), this.t))
    this.current = r.state
    this.done = r.terminated || r.truncated
    return [r.observation, r.reward, r.terminated, r.truncated, r.info]
  }

  /** A uniformly random action (gymnasium's `action_space.sample()`), drawn from the environment's key stream. */
  sampleAction(): A {
    return sampleSpace(this.env.actionSpace, foldIn(foldIn(this.key, 0x5a), this.resets * 100_003 + this.t)) as A
  }

  /** The current state as a scene. */
  render(): Scene {
    return this.env.render(this.current)
  }
}
