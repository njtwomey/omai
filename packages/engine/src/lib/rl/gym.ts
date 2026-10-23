import { type Json, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import type { Space } from '../space/index.ts'
import type { Env, EnvDefinition, EnvSpec } from './env.ts'

/**
 * Wrap a pure environment so that it truncates episodes after `maxSteps` (gymnasium's TimeLimit). The step count lives
 * in the state.
 *
 * @param env - The environment.
 * @param maxSteps - The time limit.
 * @returns The limited environment.
 * @example
 * ```ts
 * import { timeLimit, make } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * timeLimit(cartPole.create(), 10).spec.maxEpisodeSteps // => 10
 * ```
 */
export function timeLimit<S, O, A>(
  env: Env<S, O, A>,
  maxSteps: number,
): Env<{ readonly inner: S; readonly t: number }, O, A> {
  return {
    spec: { ...env.spec, maxEpisodeSteps: maxSteps },
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (k) => {
      const r = env.reset(k)
      return { state: { inner: r.state, t: 0 }, observation: r.observation, info: r.info }
    },
    step: (s, action, k) => {
      const r = env.step(s.inner, action, k)
      const t = s.t + 1
      return { ...r, state: { inner: r.state, t }, truncated: r.truncated || (t >= maxSteps && !r.terminated) }
    },
    render: (s) => env.render(s.inner),
  }
}

/**
 * Create an environment from its definition, applying the spec's time limit (gymnasium.make).
 *
 * @param definition - The environment definition.
 * @param options - Options for the definition.
 * @returns The environment.
 * @concept environments
 * @example
 * ```ts
 * import { make } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * make(cartPole).spec.id // => 'CartPole-v1'
 * ```
 */
export function make<S, O, A>(
  definition: EnvDefinition<S, O, A>,
  options?: Readonly<Record<string, unknown>>,
): Env<unknown, O, A> {
  const env = definition.create(options)
  const limit = env.spec.maxEpisodeSteps
  return (limit === undefined ? env : timeLimit(env, limit)) as Env<unknown, O, A>
}

/**
 * A gymnasium-style stateful wrapper around a pure environment: `reset({ seed })` and `step(action)` return
 * gymnasium's tuples, and the environment keeps its own state and key stream (for code written against gymnasium).
 *
 * @concept environments
 * @example
 * ```ts
 * import { GymEnv } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * const env = new GymEnv(cartPole)
 * env.reset({ seed: 0 })
 * env.step(1)[4]['truncated'] // => undefined
 * ```
 */
export class GymEnv<O, A> {
  readonly env: Env<unknown, O, A>
  private state: unknown
  private key: Key = makeKey(0)
  private t = 0
  private done = true

  constructor(definition: EnvDefinition<unknown, O, A>, options?: Readonly<Record<string, unknown>>) {
    this.env = make(definition, options)
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

  /** Start an episode; `seed` restarts the key stream (as gymnasium's reset(seed=...)). */
  reset(options: { readonly seed?: number } = {}): [O, Readonly<Record<string, Json>>] {
    if (options.seed !== undefined) this.key = makeKey(options.seed)
    this.key = foldIn(this.key, 0x7265)
    const r = this.env.reset(foldIn(this.key, 0))
    this.state = r.state
    this.t = 0
    this.done = false
    return [r.observation, r.info]
  }

  /** Advance one step: `[observation, reward, terminated, truncated, info]`. */
  step(action: A): [O, number, boolean, boolean, Readonly<Record<string, Json>>] {
    if (this.done) throw new WhiteboxError('value', 'GymEnv.step: call reset() first (the episode has ended)')
    this.t += 1
    const r = this.env.step(this.state, action, foldIn(this.key, this.t))
    this.state = r.state
    this.done = r.terminated || r.truncated
    return [r.observation, r.reward, r.terminated, r.truncated, r.info]
  }

  /** The current state as a scene. */
  render(): Scene {
    return this.env.render(this.state)
  }
}
