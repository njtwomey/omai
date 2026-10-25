/**
 * Learned dynamics models for model-based reinforcement learning: ensembles of models that predict the change in
 * observation and the reward, with the split of uncertainty into epistemic and aleatoric parts, trajectory sampling as
 * in PETS, supervised targets from transitions, and a model ensemble wrapped as an environment so planners and agents
 * can act in imagination (PETS, MBPO, Dyna).
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { scene } from '../scene/index.ts'
import type { Space } from '../space/index.ts'
import { exp, fromStorage, toFlat, type Value } from '../tensor/index.ts'
import type { Transition } from './agent.ts'
import type { Env } from './env.ts'

/**
 * A learned dynamics model as a function of explicit parameters: from a batch of observations `[B, dObs]` and actions
 * `[B, dAct]` it predicts the change in observation and the reward, `[B, dObs + 1]`, as a mean and, for probabilistic
 * models (PETS), a log-variance. Any network (or any differentiable function) can be a model; the model-based agents
 * train them by maximum likelihood.
 *
 * @concept model-based-rl
 */
export interface DynamicsModel<P> {
  readonly observationSize: number
  readonly actionSize: number
  predict(params: P, observations: Value, actions: Value): { readonly mean: Value; readonly logVar?: Value }
}

/** An ensemble of independently trained members of one model (the bootstrap ensembles of PETS and MBPO). */
export interface Ensemble<P> {
  readonly model: DynamicsModel<P>
  readonly members: readonly P[]
}

const rows = (xs: readonly (readonly number[])[], width: number): Value => {
  const out = new Float64Array(xs.length * width)
  for (const [i, x] of xs.entries()) for (let j = 0; j < width; j++) out[i * width + j] = x[j] ?? 0
  return fromStorage(out, [xs.length, width], 'float64')
}

/**
 * The prediction of every member of an ensemble for a batch, with the decomposition of uncertainty: the mean over
 * members, the epistemic variance (between members) and the aleatoric variance (the members' own predicted variances,
 * averaged; zero for deterministic models).
 *
 * @param ensemble - The ensemble.
 * @param observations - Observations, one row each.
 * @param actions - Actions, one row each.
 * @returns Per row: the mean of `[delta observation, reward]` and both variances.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { ensemblePredict } from 'mdml-engine/rl'
 * import { add, mul, concat } from 'mdml-engine/tensor'
 * const model = { observationSize: 1, actionSize: 1, predict: (w: number, o: never, a: never) => ({ mean: concat([mul(a, w), o], -1) }) }
 * ensemblePredict({ model, members: [1, 3] }, [[0]], [[1]]).mean // => [[2, 0]]
 * ```
 */
export function ensemblePredict<P>(
  ensemble: Ensemble<P>,
  observations: readonly (readonly number[])[],
  actions: readonly (readonly number[])[],
): { readonly mean: number[][]; readonly epistemic: number[][]; readonly aleatoric: number[][] } {
  const { model, members } = ensemble
  if (members.length === 0) throw new MdmlError('value', 'ensemblePredict: the ensemble has no members')
  const o = rows(observations, model.observationSize)
  const a = rows(actions, model.actionSize)
  const width = model.observationSize + 1
  const preds = members.map((p) => {
    const r = model.predict(p, o, a)
    return { mean: toFlat(r.mean), variance: r.logVar === undefined ? undefined : toFlat(exp(r.logVar)) }
  })
  const n = observations.length
  const m = members.length
  const mean: number[][] = []
  const epistemic: number[][] = []
  const aleatoric: number[][] = []
  for (let i = 0; i < n; i++) {
    const mu: number[] = []
    const ep: number[] = []
    const al: number[] = []
    for (let j = 0; j < width; j++) {
      let s = 0
      let v = 0
      for (const p of preds) {
        s += p.mean[i * width + j] ?? 0
        v += p.variance?.[i * width + j] ?? 0
      }
      const avg = s / m
      let d = 0
      for (const p of preds) d += ((p.mean[i * width + j] ?? 0) - avg) ** 2
      mu.push(avg)
      ep.push(d / m)
      al.push(v / m)
    }
    mean.push(mu)
    epistemic.push(ep)
    aleatoric.push(al)
  }
  return { mean, epistemic, aleatoric }
}

/**
 * Sample next observations and rewards from an ensemble by trajectory sampling (PETS's TS-1): each row uses a member
 * chosen uniformly at random and, for probabilistic members, a draw from its predicted Gaussian.
 *
 * @param ensemble - The ensemble.
 * @param observations - Observations, one row each.
 * @param actions - Actions, one row each.
 * @param key - The key.
 * @param options - `member` fixes the member for every row (TS-infinity); `deterministic` uses the members' means.
 * @returns Per row: the next observation and the reward.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { ensembleSample } from 'mdml-engine/rl'
 * import { concat, mul } from 'mdml-engine/tensor'
 * import { key } from 'mdml-engine/random'
 * const model = { observationSize: 1, actionSize: 1, predict: (w: number, o: never, a: never) => ({ mean: concat([mul(a, w), o], -1) }) }
 * ensembleSample({ model, members: [2] }, [[1]], [[1]], key(0))[0] // => { observation: [3], reward: 1 }
 * ```
 */
export function ensembleSample<P>(
  ensemble: Ensemble<P>,
  observations: readonly (readonly number[])[],
  actions: readonly (readonly number[])[],
  key: Key,
  options: { readonly member?: number; readonly deterministic?: boolean } = {},
): { readonly observation: number[]; readonly reward: number }[] {
  const { model, members } = ensemble
  const m = members.length
  if (m === 0) throw new MdmlError('value', 'ensembleSample: the ensemble has no members')
  const n = observations.length
  const width = model.observationSize + 1
  const o = rows(observations, model.observationSize)
  const a = rows(actions, model.actionSize)
  const choice = uniform(foldIn(key, 0), [n]).data
  const noise = normal(foldIn(key, 1), [n, width]).data
  const cache = new Map<number, { mean: number[]; std?: number[] }>()
  const predictionOf = (j: number): { mean: number[]; std?: number[] } => {
    let p = cache.get(j)
    if (p === undefined) {
      const r = model.predict(members[j] as P, o, a)
      p =
        r.logVar === undefined || options.deterministic === true
          ? { mean: toFlat(r.mean) }
          : { mean: toFlat(r.mean), std: toFlat(exp(r.logVar)).map(Math.sqrt) }
      cache.set(j, p)
    }
    return p
  }
  return Array.from({ length: n }, (_, i) => {
    const j = options.member ?? Math.min(m - 1, Math.floor((choice[i] ?? 0) * m))
    const p = predictionOf(j)
    const out = Array.from({ length: width }, (_, c) => {
      const mu = p.mean[i * width + c] ?? 0
      return p.std === undefined ? mu : mu + (p.std[i * width + c] ?? 0) * (noise[i * width + c] ?? 0)
    })
    const obs = observations[i] ?? []
    return {
      observation: Array.from({ length: model.observationSize }, (_, c) => (obs[c] ?? 0) + (out[c] ?? 0)),
      reward: out[model.observationSize] ?? 0,
    }
  })
}

/**
 * Supervised data for a dynamics model from transitions: inputs are observations and actions, targets the observation
 * change and the reward (the standard parametrisation of PETS and MBPO).
 *
 * @param transitions - Transitions with vector observations and actions.
 * @returns `observations`, `actions` and `targets` (`[delta observation, reward]`), one row per transition.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { modelTargets } from 'mdml-engine/rl'
 * const t = { observation: [1], action: [0], reward: 2, nextObservation: [1.5], terminated: false, truncated: false }
 * modelTargets([t]).targets // => [[0.5, 2]]
 * ```
 */
export function modelTargets(transitions: readonly Transition<readonly number[], readonly number[] | number>[]): {
  readonly observations: number[][]
  readonly actions: number[][]
  readonly targets: number[][]
} {
  return {
    observations: transitions.map((t) => [...t.observation]),
    actions: transitions.map((t) => (typeof t.action === 'number' ? [t.action] : [...t.action])),
    targets: transitions.map((t) => [...t.nextObservation.map((x, i) => x - (t.observation[i] ?? 0)), t.reward]),
  }
}

/** The state of a {@link modelEnv}: the current observation and the step count. */
export interface ModelEnvState {
  readonly observation: readonly number[]
  readonly t: number
}

/** Options of {@link modelEnv}. */
export interface ModelEnvOptions {
  readonly id?: string
  readonly observationSpace: Space
  readonly actionSpace: Space
  /** Start states (sampled at reset). */
  readonly initial: (key: Key) => readonly number[]
  /** A known termination function (learned models rarely predict it well). */
  readonly terminated?: (observation: readonly number[]) => boolean
  /** A known reward function of observation, action and next observation, replacing the predicted reward. */
  readonly reward?: (observation: readonly number[], action: readonly number[], next: readonly number[]) => number
  /** Fix the ensemble member (default: TS-1, a fresh member each step). */
  readonly member?: number
  readonly deterministic?: boolean
  /** The horizon of imagined episodes. */
  readonly maxEpisodeSteps?: number
}

/**
 * A learned ensemble as an environment: `step` samples the next observation and reward from the model, so planners
 * (CEM, MPPI, MCTS) and agents can act in imagination (PETS, MBPO's short rollouts, Dyna).
 *
 * @param ensemble - The ensemble.
 * @param options - Spaces, start states, optional known reward and termination functions, member choice and horizon.
 * @returns The environment.
 * @concept model-based-rl
 * @example
 * ```ts
 * import { modelEnv } from 'mdml-engine/rl'
 * import { box } from 'mdml-engine/space'
 * import { concat, mul } from 'mdml-engine/tensor'
 * import { key } from 'mdml-engine/random'
 * const model = { observationSize: 1, actionSize: 1, predict: (w: number, o: never, a: never) => ({ mean: concat([mul(a, w), o], -1) }) }
 * const env = modelEnv({ model, members: [1] }, { observationSpace: box(-9, 9, [1]), actionSpace: box(-1, 1, [1]), initial: () => [0] })
 * env.step(env.reset(key(0)).state, [0.5], key(1)).observation // => [0.5]
 * ```
 */
export function modelEnv<P>(
  ensemble: Ensemble<P>,
  options: ModelEnvOptions,
): Env<ModelEnvState, readonly number[], readonly number[]> {
  const id = options.id ?? 'LearnedModel-v0'
  return {
    spec: {
      id,
      description: 'a learned dynamics model',
      ...(options.maxEpisodeSteps === undefined ? {} : { maxEpisodeSteps: options.maxEpisodeSteps }),
    },
    observationSpace: options.observationSpace,
    actionSpace: options.actionSpace,
    reset: (k) => {
      const o = options.initial(k)
      return { state: { observation: o, t: 0 }, observation: o, info: {} }
    },
    step: (s, action, k) => {
      const [p] = ensembleSample(ensemble, [s.observation], [action], k, {
        ...(options.member === undefined ? {} : { member: options.member }),
        ...(options.deterministic === undefined ? {} : { deterministic: options.deterministic }),
      })
      const next = p?.observation ?? [...s.observation]
      const reward = options.reward !== undefined ? options.reward(s.observation, action, next) : (p?.reward ?? 0)
      const t = s.t + 1
      return {
        state: { observation: next, t },
        observation: next,
        reward,
        terminated: options.terminated?.(next) ?? false,
        truncated: options.maxEpisodeSteps !== undefined && t >= options.maxEpisodeSteps,
        info: {},
      }
    },
    render: (s) => scene(200, 40, [], { title: `${id} at step ${s.t}` }),
    observe: (s) => s.observation,
  }
}
