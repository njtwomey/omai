/**
 * The training loop of reinforcement learning (stable-baselines3's `learn`): an agent acts, the environment steps and
 * the agent learns, as a traced algorithm with periodic greedy evaluations and checkpoints. Any episode of a run can be
 * rebuilt with its rendered frames by deterministic replay from the nearest checkpoint.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import {
  type Algorithm,
  channels,
  nullTracer,
  type RunOptions,
  runAlgorithm,
  type StepContext,
  startRun,
  stateFromJson,
  stateToJson,
  stepContext,
} from '../trace/index.ts'
import { type Agent, greedyPolicyOf } from './agent.ts'
import { type Env, resetAfter } from './env.ts'
import { type Episode, evaluatePolicy } from './rollout.ts'

/** A greedy evaluation made during training. */
export interface TrainEvaluation {
  readonly episode: number
  readonly step: number
  readonly meanReturn: number
  readonly stdReturn: number
}

/** The state of a training run. */
export interface TrainState<G, S, O> {
  readonly agent: G
  readonly envState: S
  readonly observation: O
  /** Episodes finished so far. */
  readonly episode: number
  /** Steps taken in the current episode. */
  readonly t: number
  /** Environment steps taken in total. */
  readonly totalSteps: number
  readonly episodeReturn: number
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  /** The total step count at which each episode (including the current one) started. */
  readonly episodeStarts: readonly number[]
  readonly evaluations: readonly TrainEvaluation[]
  /** Snapshots at total steps 0, 1, 2, 4, 8, ... (doubling spacing), for {@link replayEpisode}. */
  readonly checkpoints: readonly TrainCheckpoint<G, S, O>[]
}

/** A snapshot of a training run (its state without the checkpoint list). */
export interface TrainCheckpoint<G, S, O> {
  readonly step: number
  readonly state: Omit<TrainState<G, S, O>, 'checkpoints'>
}

/** Options of {@link trainingAlgorithm} and {@link trainAgent}. */
export interface TrainOptions {
  /** Stop after this many episodes (default: run until the step budget). */
  readonly episodes?: number
  /** Evaluate the greedy policy every this many episodes (default never). */
  readonly evalEvery?: number
  /** Episodes per greedy evaluation (default 5). */
  readonly evalEpisodes?: number
  /** A separate environment for evaluation (default the training environment). */
  readonly evalEnv?: Env<unknown, unknown, unknown>
  /** Keep doubling-spaced checkpoints for replay (default true). */
  readonly checkpoints?: boolean
}

const isPowerOfTwoOrZero = (n: number): boolean => n === 0 || (n & (n - 1)) === 0

function snapshot<G, S, O>(s: TrainState<G, S, O>): TrainCheckpoint<G, S, O> {
  const { checkpoints: _c, ...rest } = s
  return { step: s.totalSteps, state: rest }
}

function initialState<G, S, O, A>(
  env: Env<S, O, A>,
  agent: Agent<G, O, A>,
  k: Key,
  keep: boolean,
): TrainState<G, S, O> {
  const r = env.reset(foldIn(k, 1))
  const s: TrainState<G, S, O> = {
    agent: agent.init(foldIn(k, 0)),
    envState: r.state,
    observation: r.observation,
    episode: 0,
    t: 0,
    totalSteps: 0,
    episodeReturn: 0,
    returns: [],
    lengths: [],
    episodeStarts: [0],
    evaluations: [],
    checkpoints: [],
  }
  return keep ? { ...s, checkpoints: [snapshot(s)] } : s
}

function trainStep<G, S, O, A>(
  env: Env<S, O, A>,
  agent: Agent<G, O, A>,
  options: TrainOptions,
  s: TrainState<G, S, O>,
  ctx: StepContext<TrainingPath>,
): TrainState<G, S, O> {
  const keep = options.checkpoints ?? true
  const action = agent.act(s.agent, s.observation, foldIn(ctx.key, 0))
  const r = env.step(s.envState, action, foldIn(ctx.key, 1))
  const g = agent.learn(
    s.agent,
    {
      observation: s.observation,
      action,
      reward: r.reward,
      nextObservation: r.observation,
      terminated: r.terminated,
      truncated: r.truncated,
      info: r.info,
    },
    foldIn(ctx.key, 2),
  )
  const step = s.totalSteps + 1
  if (ctx.run.wants('metric', 'reward')) ctx.run.metric(step, 'reward', r.reward)
  if (agent.stats !== undefined && ctx.run.enabled) {
    for (const [name, value] of Object.entries(agent.stats(g))) {
      if (ctx.run.wants('metric', name)) ctx.run.metric(step, name, value)
    }
  }
  const ret = s.episodeReturn + r.reward
  let next: TrainState<G, S, O>
  if (!(r.terminated || r.truncated)) {
    next = {
      ...s,
      agent: g,
      envState: r.state,
      observation: r.observation,
      t: s.t + 1,
      totalSteps: step,
      episodeReturn: ret,
    }
  } else {
    ctx.run.episode(s.episode, ret, s.t + 1, r.terminated)
    ctx.emit('rl/episode/end', { episode: s.episode, return: ret, length: s.t + 1, terminated: r.terminated })
    const episode = s.episode + 1
    let evaluations = s.evaluations
    if (options.evalEvery !== undefined && episode % options.evalEvery === 0) {
      const evalEnv = (options.evalEnv ?? env) as Env<unknown, O, A>
      const ev = evaluatePolicy(evalEnv, greedyPolicyOf(agent, g), {
        episodes: options.evalEpisodes ?? 5,
        key: foldIn(ctx.key, 9),
        parent: ctx.run,
      })
      if (ctx.run.wants('metric', 'evalReturn')) ctx.run.metric(step, 'evalReturn', ev.meanReturn)
      evaluations = [...evaluations, { episode, step, meanReturn: ev.meanReturn, stdReturn: ev.stdReturn }]
    }
    const reset = resetAfter(env, r.state, foldIn(ctx.key, 3))
    next = {
      ...s,
      agent: g,
      envState: reset.state,
      observation: reset.observation,
      episode,
      t: 0,
      totalSteps: step,
      episodeReturn: 0,
      returns: [...s.returns, ret],
      lengths: [...s.lengths, s.t + 1],
      episodeStarts: [...s.episodeStarts, step],
      evaluations,
    }
  }
  return keep && isPowerOfTwoOrZero(step) ? { ...next, checkpoints: [...next.checkpoints, snapshot(next)] } : next
}

/**
 * The paths the training algorithm emits by path: the `rl/episode/end` event with the episode’s index, return and length (the per-step reward and the agent’s statistics are channels of its schema).
 *
 * @example
 * ```ts
 * import type { TrainingPath } from 'mdml-engine/rl'
 * const path: TrainingPath = 'rl/episode/end'
 * path // => 'rl/episode/end'
 * ```
 */
export type TrainingPath = 'rl/episode/end'

/**
 * Training an agent in an environment as an {@link Algorithm}: one environment step per algorithm step (act, step,
 * learn), resetting between episodes. Records the reward each step, the agent's own statistics (its `metrics`: epsilon,
 * losses, TD errors), an `episode` event per episode, the greedy evaluations as child runs with an `evalReturn`
 * metric, and, by policy, a rendered frame per step. Snapshots at doubling step counts let {@link replayEpisode}
 * rebuild any episode cheaply.
 *
 * @param env - The environment (with its time limit, as from `make`).
 * @param agent - The agent.
 * @param options - Episodes, evaluation schedule and checkpoints.
 * @returns The algorithm.
 * @concept agents
 * @index algorithm The RL training loop: act, step, learn, evaluate.
 * @example
 * ```ts
 * import { make, policyAgent, trainingAlgorithm } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * trainingAlgorithm(make(cartPole), policyAgent(() => 1)).name // => 'train:policy'
 * ```
 */
export function trainingAlgorithm<G, S, O, A>(
  env: Env<S, O, A>,
  agent: Agent<G, O, A>,
  options: TrainOptions = {},
): Algorithm<TrainState<G, S, O>, TrainingPath> {
  return {
    name: `train:${agent.name}`,
    config: {
      env: env.spec.id,
      agent: agent.config,
      episodes: options.episodes ?? null,
      evalEvery: options.evalEvery ?? null,
    } as Json,
    schema: channels({
      metrics: {
        reward: { description: 'reward of the step' },
        return: { description: 'return of the current episode so far' },
        evalReturn: { description: 'mean return of the greedy policy', direction: 'higher' },
        ...(agent.metrics ?? {}),
      },
      episodes: true,
      frames: true,
    }),
    init: (k) => initialState(env, agent, k, options.checkpoints ?? true),
    step: (s, ctx) => trainStep(env, agent, options, s, ctx),
    done: (s) => (options.episodes !== undefined && s.episode >= options.episodes ? 'completed' : undefined),
    // Checkpoints drop the in-state snapshot list: a seekable tape makes it redundant.
    serialise: (s) => stateToJson({ ...s, checkpoints: [] }),
    deserialise: (j) => stateFromJson(j) as TrainState<G, S, O>,
    observe: (s) => ({ metrics: { return: s.episodeReturn }, frame: env.render(s.envState) as unknown as Json }),
  }
}

/** Result of {@link trainAgent}. */
export interface TrainResult<G, S, O> {
  /** The trained agent state. */
  readonly agent: G
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  readonly evaluations: readonly TrainEvaluation[]
  readonly episodes: number
  readonly steps: number
  readonly status: string
  readonly runId: string
  /** The final training state (with its checkpoints). */
  readonly state: TrainState<G, S, O>
  /** The root key of the run, for {@link replayEpisode}. */
  readonly key: Key
}

/**
 * Train an agent for a number of episodes or environment steps (the stable-baselines3 `learn` loop), traced.
 *
 * @param env - The environment (with its time limit).
 * @param agent - The agent.
 * @param options - Episodes or a step budget (`maxSteps`, default 100000), evaluation, checkpoints, and the run options
 *   (key, seed, tracer, policy).
 * @returns The trained agent state, returns per episode, evaluations and the final state.
 * @throws MdmlError with code `value` when neither `episodes` nor `maxSteps` is given.
 * @concept agents
 * @index function Train an agent in an environment, with periodic evaluation.
 * @example
 * ```ts
 * import { make, policyAgent, trainAgent } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * trainAgent(make(cartPole), policyAgent(() => 1), { episodes: 2 }).returns.length // => 2
 * ```
 */
export function trainAgent<G, S, O, A>(
  env: Env<S, O, A>,
  agent: Agent<G, O, A>,
  options: TrainOptions & Partial<RunOptions> = {},
): TrainResult<G, S, O> {
  const { episodes, evalEvery, evalEpisodes, evalEnv, checkpoints, maxSteps, ...run } = options
  if (episodes === undefined && maxSteps === undefined) {
    throw new MdmlError('value', 'trainAgent: give episodes or maxSteps')
  }
  const key = run.key ?? makeKey(run.seed ?? 0)
  const alg = trainingAlgorithm(env, agent, {
    ...(episodes === undefined ? {} : { episodes }),
    ...(evalEvery === undefined ? {} : { evalEvery }),
    ...(evalEpisodes === undefined ? {} : { evalEpisodes }),
    ...(evalEnv === undefined ? {} : { evalEnv }),
    ...(checkpoints === undefined ? {} : { checkpoints }),
  })
  const result = runAlgorithm(alg, { ...run, key, maxSteps: maxSteps ?? 100_000_000 })
  const s = result.state
  return {
    agent: s.agent,
    returns: s.returns,
    lengths: s.lengths,
    evaluations: s.evaluations,
    episodes: s.episode,
    steps: s.totalSteps,
    status: result.status,
    runId: result.runId,
    state: s,
    key,
  }
}

/**
 * Rebuild any episode of a training run, with its rendered frames, by deterministic replay from the nearest earlier
 * checkpoint: every key of a training step is derived from the run key and the step number, so re-running the steps
 * reproduces the actions, states and agent updates exactly.
 *
 * @param env - The training environment.
 * @param agent - The agent.
 * @param result - The result of {@link trainAgent} (or anything with its final `state` and `key`).
 * @param episode - The episode index (0-based).
 * @returns The episode with states, observations, actions, rewards and frames.
 * @throws MdmlError with code `value` when the episode was not finished in the run.
 * @concept agents
 * @example
 * ```ts
 * import { make, policyAgent, replayEpisode, trainAgent } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/environments'
 * const env = make(cartPole)
 * const agent = policyAgent(() => 1)
 * const result = trainAgent(env, agent, { episodes: 3 })
 * replayEpisode(env, agent, result, 2).return === result.returns[2] // => true
 * ```
 */
export function replayEpisode<G, S, O, A>(
  env: Env<S, O, A>,
  agent: Agent<G, O, A>,
  result: Pick<TrainResult<G, S, O>, 'state' | 'key'>,
  episode: number,
): Episode<S, O, A> & { readonly frames: readonly Scene[] } {
  const final = result.state
  const start = final.episodeStarts[episode]
  if (start === undefined || episode >= final.episode) {
    throw new MdmlError('value', `replayEpisode: episode ${episode} did not finish in this run`)
  }
  let cp: TrainCheckpoint<G, S, O> | undefined
  for (const c of final.checkpoints) if (c.step <= start) cp = c
  let s: TrainState<G, S, O> =
    cp === undefined ? initialState(env, agent, foldIn(result.key, 0), false) : { ...cp.state, checkpoints: [] }
  const stepsKey = foldIn(result.key, 1)
  const run = startRun(nullTracer, { name: 'replay', seed: null, config: {} })
  // Evaluations never change the agent or the environment, so the replay skips them.
  const quiet: TrainOptions = { checkpoints: false }
  const step = (state: TrainState<G, S, O>): TrainState<G, S, O> =>
    trainStep(env, agent, quiet, state, stepContext(state.totalSteps, foldIn(stepsKey, state.totalSteps), run))
  while (s.totalSteps < start) s = step(s)
  const observations: O[] = [s.observation]
  const states: S[] = [s.envState]
  const frames: Scene[] = [env.render(s.envState)]
  const actions: A[] = []
  const rewards: number[] = []
  const infos: Readonly<Record<string, Json>>[] = []
  let terminated = false
  let truncated = false
  for (;;) {
    const ctxKey = foldIn(stepsKey, s.totalSteps)
    const action = agent.act(s.agent, s.observation, foldIn(ctxKey, 0))
    const r = env.step(s.envState, action, foldIn(ctxKey, 1))
    actions.push(action)
    rewards.push(r.reward)
    infos.push(r.info)
    observations.push(r.observation)
    states.push(r.state)
    frames.push(env.render(r.state))
    if (r.terminated || r.truncated) {
      terminated = r.terminated
      truncated = r.truncated
      break
    }
    s = step(s)
  }
  let ret = 0
  for (const x of rewards) ret += x
  return {
    observations,
    states,
    actions,
    rewards,
    infos,
    return: ret,
    length: actions.length,
    terminated,
    truncated,
    frames,
  }
}
