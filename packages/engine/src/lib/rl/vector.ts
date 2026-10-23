import { type Json, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Space } from '../space/index.ts'
import { type Env, resetAfter } from './env.ts'

/** How a vector environment restarts finished sub-environments (gymnasium's AutoresetMode). */
export type AutoResetMode = 'same-step' | 'next-step' | 'disabled'

/** The state of a vector environment: the sub-environment states and which of them must reset before stepping. */
export interface VectorState<S> {
  readonly envs: readonly S[]
  readonly needsReset: readonly boolean[]
}

/** The result of a vector step. */
export interface VectorStep<S, O> {
  readonly state: VectorState<S>
  readonly observations: readonly O[]
  readonly rewards: readonly number[]
  readonly terminated: readonly boolean[]
  readonly truncated: readonly boolean[]
  readonly infos: readonly Readonly<Record<string, Json>>[]
  /** With same-step auto-reset: the last observation of each episode that ended this step (stable-baselines3's
   * `terminal_observation`), for bootstrapping truncated episodes. */
  readonly finalObservations: readonly (O | undefined)[]
}

/**
 * Several copies of an environment stepped together with independent keys (gymnasium's SyncVectorEnv, gymnax's vmap):
 * `reset(key)` and `step(state, actions, key)` act on all copies, with keys `foldIn(key, i)` for copy `i`.
 *
 * @concept environments
 */
export interface VectorEnv<S, O, A> {
  readonly env: Env<S, O, A>
  readonly numEnvs: number
  readonly autoReset: AutoResetMode
  readonly observationSpace: Space
  readonly actionSpace: Space
  reset(key: Key): {
    readonly state: VectorState<S>
    readonly observations: readonly O[]
    readonly infos: readonly Readonly<Record<string, Json>>[]
  }
  step(state: VectorState<S>, actions: readonly A[], key: Key): VectorStep<S, O>
}

/**
 * Vectorise an environment. With `'same-step'` auto-reset (stable-baselines3's convention, the default) a finished copy
 * is reset within the same step and its final observation is reported in `finalObservations`; with `'next-step'`
 * (gymnasium's default) the step after the end resets the copy and ignores its action; `'disabled'` leaves resets to
 * the caller.
 *
 * @param env - The environment.
 * @param numEnvs - The number of copies.
 * @param options - The auto-reset mode.
 * @returns The vector environment.
 * @concept environments
 * @example
 * ```ts
 * import { vectorEnv } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const venv = vectorEnv(cartPole.create(), 3)
 * venv.step(venv.reset(key(0)).state, [0, 1, 0], key(1)).rewards // => [1, 1, 1]
 * ```
 */
export function vectorEnv<S, O, A>(
  env: Env<S, O, A>,
  numEnvs: number,
  options: { readonly autoReset?: AutoResetMode } = {},
): VectorEnv<S, O, A> {
  if (!(Number.isInteger(numEnvs) && numEnvs >= 1)) {
    throw new WhiteboxError('value', 'vectorEnv: numEnvs must be a positive integer')
  }
  const mode = options.autoReset ?? 'same-step'
  return {
    env,
    numEnvs,
    autoReset: mode,
    observationSpace: env.observationSpace,
    actionSpace: env.actionSpace,
    reset: (k) => {
      const rs = Array.from({ length: numEnvs }, (_, i) => env.reset(foldIn(k, i)))
      return {
        state: { envs: rs.map((r) => r.state), needsReset: rs.map(() => false) },
        observations: rs.map((r) => r.observation),
        infos: rs.map((r) => r.info),
      }
    },
    step: (state, actions, k) => {
      if (actions.length !== numEnvs) {
        throw new WhiteboxError('shape', `vectorEnv.step: expected ${numEnvs} actions, got ${actions.length}`)
      }
      const envs: S[] = []
      const needsReset: boolean[] = []
      const observations: O[] = []
      const rewards: number[] = []
      const terminated: boolean[] = []
      const truncated: boolean[] = []
      const infos: Readonly<Record<string, Json>>[] = []
      const finals: (O | undefined)[] = []
      for (let i = 0; i < numEnvs; i++) {
        const ki = foldIn(k, i)
        const s = state.envs[i] as S
        if (mode === 'next-step' && state.needsReset[i] === true) {
          const r = resetAfter(env, s, foldIn(ki, 1))
          envs.push(r.state)
          needsReset.push(false)
          observations.push(r.observation)
          rewards.push(0)
          terminated.push(false)
          truncated.push(false)
          infos.push(r.info)
          finals.push(undefined)
          continue
        }
        const r = env.step(s, actions[i] as A, foldIn(ki, 0))
        const done = r.terminated || r.truncated
        rewards.push(r.reward)
        terminated.push(r.terminated)
        truncated.push(r.truncated)
        if (done && mode === 'same-step') {
          const next = resetAfter(env, r.state, foldIn(ki, 1))
          envs.push(next.state)
          needsReset.push(false)
          observations.push(next.observation)
          infos.push({ ...r.info, finalInfo: r.info })
          finals.push(r.observation)
        } else {
          envs.push(r.state)
          needsReset.push(done)
          observations.push(r.observation)
          infos.push(r.info)
          finals.push(undefined)
        }
      }
      return {
        state: { envs, needsReset },
        observations,
        rewards,
        terminated,
        truncated,
        infos,
        finalObservations: finals,
      }
    },
  }
}

/**
 * A gymnasium-style stateful vector environment (gymnasium.vector.SyncVectorEnv): `reset({ seed })` and
 * `step(actions)` return batched tuples and the class keeps the states and the key stream.
 *
 * @concept environments
 * @example
 * ```ts
 * import { GymVectorEnv } from 'whitebox-engine/rl'
 * import { cartPole } from 'whitebox-engine/methods/rl'
 * const venv = new GymVectorEnv(cartPole.create(), 2)
 * venv.reset({ seed: 0 })[0].length // => 2
 * ```
 */
export class GymVectorEnv<S, O, A> {
  readonly vector: VectorEnv<S, O, A>
  private state: VectorState<S> | undefined
  private key: Key = makeKey(0)
  private t = 0

  /**
   * @param env - The environment to copy.
   * @param numEnvs - The number of copies.
   * @param options - The auto-reset mode (default `'next-step'`, as gymnasium).
   */
  constructor(env: Env<S, O, A>, numEnvs: number, options: { readonly autoReset?: AutoResetMode } = {}) {
    this.vector = vectorEnv(env, numEnvs, { autoReset: options.autoReset ?? 'next-step' })
  }

  get numEnvs(): number {
    return this.vector.numEnvs
  }

  /** Reset every copy; `seed` restarts the key stream. */
  reset(options: { readonly seed?: number } = {}): [readonly O[], readonly Readonly<Record<string, Json>>[]] {
    if (options.seed !== undefined) this.key = makeKey(options.seed)
    this.t = 0
    this.key = foldIn(this.key, 0x5645)
    const r = this.vector.reset(foldIn(this.key, 0))
    this.state = r.state
    return [r.observations, r.infos]
  }

  /** Step every copy: `[observations, rewards, terminated, truncated, infos]`. */
  step(
    actions: readonly A[],
  ): [
    readonly O[],
    readonly number[],
    readonly boolean[],
    readonly boolean[],
    readonly Readonly<Record<string, Json>>[],
  ] {
    if (this.state === undefined) throw new WhiteboxError('value', 'GymVectorEnv.step: call reset() first')
    this.t += 1
    const r = this.vector.step(this.state, actions, foldIn(this.key, this.t))
    this.state = r.state
    return [r.observations, r.rewards, r.terminated, r.truncated, r.infos]
  }
}
