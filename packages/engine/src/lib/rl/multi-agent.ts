/**
 * Multi-agent environments in PettingZoo's two forms as pure functions: simultaneous-move parallel environments and
 * turn-based agent-environment cycles. Playing episodes with a policy per agent, conversion between the two forms, time
 * limits, and single-agent views (one agent against fixed policies, a centralised joint controller) so single-agent
 * learners can train in them.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import type { Scene } from '../scene/index.ts'
import { dict, type Space } from '../space/index.ts'
import type { Policy } from './agent.ts'
import type { Env, EnvSpec } from './env.ts'

/** Values keyed by agent name. */
export type PerAgent<T> = Readonly<Record<string, T>>

/** The result of a simultaneous step. */
export interface ParallelStep<S, O> {
  readonly state: S
  readonly observations: PerAgent<O>
  readonly rewards: PerAgent<number>
  readonly terminations: PerAgent<boolean>
  readonly truncations: PerAgent<boolean>
  readonly infos: PerAgent<Readonly<Record<string, Json>>>
}

/**
 * A simultaneous-move multi-agent environment (PettingZoo's ParallelEnv) as pure functions: every live agent acts at
 * once, and each gets its own observation, reward and termination flags. Agents that have terminated leave
 * `agents(state)`.
 *
 * @concept multi-agent
 */
export interface ParallelEnv<S, O, A> {
  readonly spec: EnvSpec
  readonly possibleAgents: readonly string[]
  readonly observationSpaces: PerAgent<Space>
  readonly actionSpaces: PerAgent<Space>
  /** The agents still acting in a state. */
  agents(state: S): readonly string[]
  reset(key: Key): {
    readonly state: S
    readonly observations: PerAgent<O>
    readonly infos: PerAgent<Readonly<Record<string, Json>>>
  }
  step(state: S, actions: PerAgent<A>, key: Key): ParallelStep<S, O>
  render(state: S): Scene
}

/** The result of a turn in a turn-based environment. */
export interface AecStep<S> {
  readonly state: S
  /** The reward every agent received from this turn. */
  readonly rewards: PerAgent<number>
}

/**
 * A turn-based multi-agent environment (PettingZoo's agent-environment cycle, AECEnv) as pure functions: one agent acts
 * at a time (`currentAgent`), observations are taken per agent, and legal actions can be masked (board games). The
 * episode is over when `currentAgent` returns `undefined`.
 *
 * @concept multi-agent
 */
export interface AecEnv<S, O, A> {
  readonly spec: EnvSpec
  readonly possibleAgents: readonly string[]
  readonly observationSpaces: PerAgent<Space>
  readonly actionSpaces: PerAgent<Space>
  reset(key: Key): S
  /** The agent to move, or `undefined` when every agent is done. */
  currentAgent(state: S): string | undefined
  observe(state: S, agent: string): O
  step(state: S, action: A, key: Key): AecStep<S>
  terminations(state: S): PerAgent<boolean>
  truncations(state: S): PerAgent<boolean>
  /** The legal actions of an agent (the action mask of board games), when not every action is legal. */
  legalActions?(state: S, agent: string): readonly A[]
  render(state: S): Scene
}

/** The cumulative rewards of agents over a turn-based episode. */
export interface AecEpisode<S, A> {
  readonly states: readonly S[]
  readonly agents: readonly string[]
  readonly actions: readonly A[]
  readonly returns: PerAgent<number>
  readonly length: number
}

/**
 * Play one turn-based episode with a policy per agent (pure in the key; at most `maxTurns` turns).
 *
 * @param env - The environment.
 * @param policies - A policy per agent, given that agent's observation.
 * @param key - The key.
 * @param options - The turn cap (default 10000).
 * @returns The states, movers, actions and each agent's total reward.
 * @concept multi-agent
 * @example
 * ```ts
 * import { playAec } from 'mdml-engine/rl'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const take1 = () => 0
 * playAec(nim.create({ heaps: [3] }), { player_0: take1, player_1: take1 }, key(0)).returns // => { player_0: 1, player_1: -1 }
 * ```
 */
export function playAec<S, O, A>(
  env: AecEnv<S, O, A>,
  policies: PerAgent<Policy<O, A>>,
  key: Key,
  options: { readonly maxTurns?: number } = {},
): AecEpisode<S, A> {
  let state = env.reset(foldIn(key, 0))
  const states = [state]
  const movers: string[] = []
  const actions: A[] = []
  const returns: Record<string, number> = Object.fromEntries(env.possibleAgents.map((a) => [a, 0]))
  const cap = options.maxTurns ?? 10_000
  for (let t = 0; t < cap; t++) {
    const agent = env.currentAgent(state)
    if (agent === undefined) break
    const policy = policies[agent]
    if (policy === undefined) throw new MdmlError('value', `playAec: no policy for ${agent}`)
    const action = policy(env.observe(state, agent), foldIn(foldIn(key, 1), t))
    const r = env.step(state, action, foldIn(foldIn(key, 2), t))
    for (const [a, x] of Object.entries(r.rewards)) returns[a] = (returns[a] ?? 0) + x
    state = r.state
    states.push(state)
    movers.push(agent)
    actions.push(action)
  }
  return { states, agents: movers, actions, returns, length: actions.length }
}

/** The state of {@link aecToParallel}: the turn-based state. */
export interface AecParallelState<S> {
  readonly inner: S
}

/**
 * View a turn-based environment as a simultaneous one (PettingZoo's `aec_to_parallel`): one parallel step lets every
 * live agent move once, in turn order, and sums each agent's rewards over the cycle.
 *
 * @param env - The turn-based environment.
 * @returns The parallel environment.
 * @concept multi-agent
 * @example
 * ```ts
 * import { aecToParallel } from 'mdml-engine/rl'
 * import { nim } from 'mdml-engine/environments'
 * aecToParallel(nim.create()).possibleAgents // => ['player_0', 'player_1']
 * ```
 */
export function aecToParallel<S, O, A>(env: AecEnv<S, O, A>): ParallelEnv<AecParallelState<S>, O, A> {
  const live = (s: S): string[] => {
    const term = env.terminations(s)
    const trunc = env.truncations(s)
    return env.possibleAgents.filter((a) => term[a] !== true && trunc[a] !== true)
  }
  const observeAll = (s: S): Record<string, O> => Object.fromEntries(live(s).map((a) => [a, env.observe(s, a)]))
  return {
    spec: env.spec,
    possibleAgents: env.possibleAgents,
    observationSpaces: env.observationSpaces,
    actionSpaces: env.actionSpaces,
    agents: (s) => live(s.inner),
    reset: (k) => {
      const inner = env.reset(k)
      return { state: { inner }, observations: observeAll(inner), infos: {} }
    },
    step: (s, actions, k) => {
      let inner = s.inner
      const rewards: Record<string, number> = Object.fromEntries(env.possibleAgents.map((a) => [a, 0]))
      const acted = new Set<string>()
      for (let i = 0; i < env.possibleAgents.length; i++) {
        const agent = env.currentAgent(inner)
        if (agent === undefined || acted.has(agent)) break
        const action = actions[agent]
        if (action === undefined) throw new MdmlError('value', `aecToParallel: no action for ${agent}`)
        const r = env.step(inner, action, foldIn(k, i))
        for (const [a, x] of Object.entries(r.rewards)) rewards[a] = (rewards[a] ?? 0) + x
        inner = r.state
        acted.add(agent)
      }
      return {
        state: { inner },
        observations: observeAll(inner),
        rewards,
        terminations: env.terminations(inner),
        truncations: env.truncations(inner),
        infos: {},
      }
    },
    render: (s) => env.render(s.inner),
  }
}

/** The state of {@link parallelToAec}: the simultaneous state, the actions chosen so far and the last results. */
export interface ParallelAecState<S, O> {
  readonly inner: S
  readonly pending: PerAgent<unknown>
  readonly order: readonly string[]
  readonly observations: PerAgent<O>
  readonly terminations: PerAgent<boolean>
  readonly truncations: PerAgent<boolean>
}

/**
 * View a simultaneous environment as a turn-based one (PettingZoo's `parallel_to_aec`): live agents choose actions one
 * at a time without seeing each other's choices, and the underlying step happens when the last of them has chosen.
 *
 * @param env - The parallel environment.
 * @returns The turn-based environment.
 * @concept multi-agent
 * @example
 * ```ts
 * import { parallelToAec } from 'mdml-engine/rl'
 * import { matrixGame } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = parallelToAec(matrixGame.create({ game: 'prisonersDilemma' }))
 * env.currentAgent(env.reset(key(0))) // => 'player_0'
 * ```
 */
export function parallelToAec<S, O, A>(env: ParallelEnv<S, O, A>): AecEnv<ParallelAecState<S, O>, O, A> {
  const done = (s: ParallelAecState<S, O>, a: string): boolean =>
    s.terminations[a] === true || s.truncations[a] === true
  return {
    spec: env.spec,
    possibleAgents: env.possibleAgents,
    observationSpaces: env.observationSpaces,
    actionSpaces: env.actionSpaces,
    reset: (k) => {
      const r = env.reset(k)
      const order = env.agents(r.state)
      return { inner: r.state, pending: {}, order, observations: r.observations, terminations: {}, truncations: {} }
    },
    currentAgent: (s) => s.order.find((a) => !(a in s.pending) && !done(s, a)),
    observe: (s, agent) => s.observations[agent] as O,
    step: (s, action, k) => {
      const agent = s.order.find((a) => !(a in s.pending) && !done(s, a))
      if (agent === undefined) throw new MdmlError('value', 'parallelToAec: the episode is over')
      const pending = { ...s.pending, [agent]: action }
      const waiting = s.order.filter((a) => !(a in pending) && !done(s, a))
      if (waiting.length > 0) return { state: { ...s, pending }, rewards: {} }
      const r = env.step(s.inner, pending as PerAgent<A>, k)
      return {
        state: {
          inner: r.state,
          pending: {},
          order: env.agents(r.state),
          observations: { ...s.observations, ...r.observations },
          terminations: { ...s.terminations, ...r.terminations },
          truncations: { ...s.truncations, ...r.truncations },
        },
        rewards: r.rewards,
      }
    },
    terminations: (s) => s.terminations,
    truncations: (s) => s.truncations,
    render: (s) => env.render(s.inner),
  }
}

/** The state of {@link parallelTimeLimit}. */
export interface ParallelTimeLimitState<S> {
  readonly inner: S
  readonly t: number
}

/**
 * Truncate a simultaneous environment after `maxSteps` steps (every live agent is truncated).
 *
 * @param env - The environment.
 * @param maxSteps - The limit.
 * @returns The limited environment.
 * @concept multi-agent
 * @example
 * ```ts
 * import { parallelTimeLimit } from 'mdml-engine/rl'
 * import { matrixGame } from 'mdml-engine/environments'
 * parallelTimeLimit(matrixGame.create(), 5).spec.maxEpisodeSteps // => 5
 * ```
 */
export function parallelTimeLimit<S, O, A>(
  env: ParallelEnv<S, O, A>,
  maxSteps: number,
): ParallelEnv<ParallelTimeLimitState<S>, O, A> {
  return {
    ...env,
    spec: { ...env.spec, maxEpisodeSteps: maxSteps },
    agents: (s) => env.agents(s.inner),
    reset: (k) => {
      const r = env.reset(k)
      return { ...r, state: { inner: r.state, t: 0 } }
    },
    step: (s, actions, k) => {
      const r = env.step(s.inner, actions, k)
      const t = s.t + 1
      const truncations =
        t >= maxSteps
          ? Object.fromEntries(Object.keys(r.observations).map((a) => [a, r.terminations[a] !== true]))
          : r.truncations
      return { ...r, state: { inner: r.state, t }, truncations }
    },
    render: (s) => env.render(s.inner),
  }
}

/** The state of {@link singleAgentView}. */
export interface SingleAgentState<S, O> {
  readonly inner: S
  readonly observations: PerAgent<O>
}

/**
 * One agent's view of a simultaneous environment, with every other agent following a fixed policy: a single-agent
 * {@link Env}, so any single-agent learner can train against fixed opponents or teammates (and against snapshots of
 * itself, for self-play). The episode ends when the chosen agent terminates or is truncated.
 *
 * @param env - The parallel environment.
 * @param agent - The agent to control.
 * @param others - Policies for the other agents (missing agents throw).
 * @returns The single-agent environment.
 * @concept multi-agent
 * @example
 * ```ts
 * import { singleAgentView } from 'mdml-engine/rl'
 * import { matrixGame } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = singleAgentView(matrixGame.create({ game: 'prisonersDilemma' }), 'player_0', { player_1: () => 0 })
 * env.step(env.reset(key(0)).state, 1, key(1)).reward // => 5
 * ```
 */
export function singleAgentView<S, O, A>(
  env: ParallelEnv<S, O, A>,
  agent: string,
  others: PerAgent<Policy<O, A>>,
): Env<SingleAgentState<S, O>, O, A> {
  if (!env.possibleAgents.includes(agent)) throw new MdmlError('value', `singleAgentView: no agent ${agent}`)
  return {
    spec: { ...env.spec, id: `${env.spec.id}/${agent}` },
    observationSpace: env.observationSpaces[agent] as Space,
    actionSpace: env.actionSpaces[agent] as Space,
    reset: (k) => {
      const r = env.reset(k)
      return {
        state: { inner: r.state, observations: r.observations },
        observation: r.observations[agent] as O,
        info: (r.infos[agent] ?? {}) as Readonly<Record<string, Json>>,
      }
    },
    step: (s, action, k) => {
      const actions: Record<string, A> = { [agent]: action }
      for (const [i, a] of env.agents(s.inner).entries()) {
        if (a === agent) continue
        const policy = others[a]
        if (policy === undefined) throw new MdmlError('value', `singleAgentView: no policy for ${a}`)
        actions[a] = policy(s.observations[a] as O, foldIn(k, i + 1))
      }
      const r = env.step(s.inner, actions, foldIn(k, 0))
      return {
        state: { inner: r.state, observations: { ...s.observations, ...r.observations } },
        observation: (r.observations[agent] ?? s.observations[agent]) as O,
        reward: r.rewards[agent] ?? 0,
        terminated: r.terminations[agent] === true,
        truncated: r.truncations[agent] === true,
        info: (r.infos[agent] ?? {}) as Readonly<Record<string, Json>>,
      }
    },
    render: (s) => env.render(s.inner),
    observe: (s) => s.observations[agent] as O,
  }
}

/** One simultaneous-move episode. */
export interface ParallelEpisode<S> {
  readonly states: readonly S[]
  readonly returns: PerAgent<number>
  readonly length: number
}

/**
 * Play one simultaneous-move episode with a policy per agent (pure in the key).
 *
 * @param env - The environment (with a time limit, or pass `maxSteps`).
 * @param policies - A policy per agent.
 * @param key - The key.
 * @param options - The step cap (default 10000).
 * @returns The states and each agent's total reward.
 * @concept multi-agent
 * @example
 * ```ts
 * import { playParallel, parallelTimeLimit } from 'mdml-engine/rl'
 * import { matrixGame } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = parallelTimeLimit(matrixGame.create({ game: 'prisonersDilemma' }), 3)
 * playParallel(env, { player_0: () => 0, player_1: () => 0 }, key(0)).returns // => { player_0: 9, player_1: 9 }
 * ```
 */
export function playParallel<S, O, A>(
  env: ParallelEnv<S, O, A>,
  policies: PerAgent<Policy<O, A>>,
  key: Key,
  options: { readonly maxSteps?: number } = {},
): ParallelEpisode<S> {
  const r0 = env.reset(foldIn(key, 0))
  let state = r0.state
  let observations = r0.observations
  const states = [state]
  const returns: Record<string, number> = Object.fromEntries(env.possibleAgents.map((a) => [a, 0]))
  const cap = options.maxSteps ?? 10_000
  let t = 0
  for (; t < cap; t++) {
    const live = env.agents(state)
    if (live.length === 0) break
    const actions: Record<string, A> = {}
    for (const [i, a] of live.entries()) {
      const policy = policies[a]
      if (policy === undefined) throw new MdmlError('value', `playParallel: no policy for ${a}`)
      actions[a] = policy(observations[a] as O, foldIn(foldIn(key, 1 + i), t))
    }
    const r = env.step(state, actions, foldIn(foldIn(key, 0x7e), t))
    for (const [a, x] of Object.entries(r.rewards)) returns[a] = (returns[a] ?? 0) + x
    state = r.state
    observations = { ...observations, ...r.observations }
    states.push(state)
    const over = live.every((a) => r.terminations[a] === true || r.truncations[a] === true)
    if (over) {
      t += 1
      break
    }
  }
  return { states, returns, length: t }
}

/** The state of {@link jointEnv}: the simultaneous state and every agent's latest observation. */
export interface JointState<S, O> {
  readonly inner: S
  readonly observations: PerAgent<O>
}

/** How {@link jointEnv} reduces the agents' rewards to one scalar. */
export type JointReward = 'sum' | 'mean' | { readonly agent: string }

/**
 * A simultaneous environment as one single-agent {@link Env} over joint observations and joint actions (the view of a
 * centralised controller, and the form multi-agent learners take so that `trainAgent` can run them): observations
 * and actions are records keyed by agent, the scalar reward is the team's (the sum or the mean of the agents'
 * rewards) or one chosen agent's, and `info` carries every live agent's own `rewards`, `terminations` and
 * `truncations`. The episode terminates when every live agent has terminated and is truncated when every live agent
 * is done otherwise (or none is left).
 *
 * @param env - The parallel environment.
 * @param options - `reward`: `'sum'` (default), `'mean'` or `{ agent }`.
 * @returns The joint environment.
 * @throws MdmlError with code `value` for an unknown reward agent, or when a live agent has no action.
 * @concept multi-agent
 * @example
 * ```ts
 * import { jointEnv } from 'mdml-engine/rl'
 * import { matrixGame } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = jointEnv(matrixGame.create({ game: 'prisonersDilemma' }))
 * const r = env.step(env.reset(key(0)).state, { player_0: 1, player_1: 0 }, key(1))
 * [r.reward, r.info['rewards']] // => [5, { player_0: 5, player_1: 0 }]
 * ```
 */
export function jointEnv<S, O, A>(
  env: ParallelEnv<S, O, A>,
  options: { readonly reward?: JointReward } = {},
): Env<JointState<S, O>, PerAgent<O>, PerAgent<A>> {
  const mode = options.reward ?? 'sum'
  if (typeof mode === 'object' && !env.possibleAgents.includes(mode.agent))
    throw new MdmlError('value', `jointEnv: no agent ${mode.agent}`)
  const scalar = (rewards: PerAgent<number>): number => {
    if (typeof mode === 'object') return rewards[mode.agent] ?? 0
    let total = 0
    for (const a of env.possibleAgents) total += rewards[a] ?? 0
    return mode === 'mean' ? total / env.possibleAgents.length : total
  }
  return {
    spec: { ...env.spec, id: `${env.spec.id}/joint` },
    observationSpace: dict(env.observationSpaces),
    actionSpace: dict(env.actionSpaces),
    reset: (k) => {
      const r = env.reset(k)
      return { state: { inner: r.state, observations: r.observations }, observation: r.observations, info: {} }
    },
    step: (s, actions, k) => {
      const live = env.agents(s.inner)
      const acting: Record<string, A> = {}
      for (const a of live) {
        const action = actions[a]
        if (action === undefined) throw new MdmlError('value', `jointEnv: no action for ${a}`)
        acting[a] = action
      }
      const r = env.step(s.inner, acting, k)
      const observations = { ...s.observations, ...r.observations }
      const done = (a: string): boolean => r.terminations[a] === true || r.truncations[a] === true
      const terminated = live.every((a) => r.terminations[a] === true)
      const finished = live.every(done) || env.agents(r.state).length === 0
      const pick = (x: PerAgent<boolean>): Record<string, boolean> =>
        Object.fromEntries(live.map((a) => [a, x[a] === true]))
      return {
        state: { inner: r.state, observations },
        observation: observations,
        reward: scalar(r.rewards),
        terminated,
        truncated: !terminated && finished,
        info: {
          rewards: Object.fromEntries(live.map((a) => [a, r.rewards[a] ?? 0])),
          terminations: pick(r.terminations),
          truncations: pick(r.truncations),
        },
      }
    },
    render: (s) => env.render(s.inner),
    observe: (s) => s.observations,
  }
}

/** The state of {@link aecSingleAgentView}: the turn-based state and reward earned before the agent's next turn. */
export interface AecViewState<S> {
  readonly inner: S
  readonly carry: number
}

/**
 * One agent's view of a turn-based game as a single-agent {@link Env}, with every other agent following a fixed
 * policy that sees its own, current observation: a step plays the agent's move and then the others' moves until the
 * agent is to move again or the game ends, summing the agent's rewards. Reset plays the others' opening moves (if they end the game, the next step, whatever the action, is the final one and pays the agent's reward). The
 * `info.legalActions` of a reset or step lists the agent's legal moves for its next turn (for masking). This is how
 * a single-agent learner trains against fixed opponents or snapshots of itself (self-play) in board games and poker.
 *
 * @param env - The turn-based game.
 * @param agent - The agent to control.
 * @param others - Policies for the other agents.
 * @returns The single-agent environment.
 * @throws MdmlError with code `value` for an unknown agent or a missing policy.
 * @concept multi-agent
 * @example
 * ```ts
 * import { aecSingleAgentView } from 'mdml-engine/rl'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = aecSingleAgentView(nim.create({ heaps: [3] }), 'player_0', { player_1: () => 0 })
 * const r = env.step(env.reset(key(0)).state, 0, key(1))
 * env.step(r.state, 0, key(2)).reward // => 1
 * ```
 */
export function aecSingleAgentView<S, O, A>(
  env: AecEnv<S, O, A>,
  agent: string,
  others: PerAgent<Policy<O, A>>,
): Env<AecViewState<S>, O, A> {
  if (!env.possibleAgents.includes(agent)) throw new MdmlError('value', `aecSingleAgentView: no agent ${agent}`)
  // Play the other agents until it is the agent's turn or the game ends, summing the agent's rewards.
  const advance = (start: S, k: Key): { readonly state: S; readonly reward: number } => {
    let s = start
    let reward = 0
    for (let t = 0; t < 100_000; t++) {
      const mover = env.currentAgent(s)
      if (mover === undefined || mover === agent) break
      const policy = others[mover]
      if (policy === undefined) throw new MdmlError('value', `aecSingleAgentView: no policy for ${mover}`)
      const r = env.step(s, policy(env.observe(s, mover), foldIn(k, 2 * t)), foldIn(k, 2 * t + 1))
      reward += r.rewards[agent] ?? 0
      s = r.state
    }
    return { state: s, reward }
  }
  const info = (s: S): Readonly<Record<string, Json>> =>
    env.legalActions === undefined || env.currentAgent(s) !== agent
      ? {}
      : { legalActions: env.legalActions(s, agent) as unknown as Json }
  return {
    spec: { ...env.spec, id: `${env.spec.id}/${agent}` },
    observationSpace: env.observationSpaces[agent] as Space,
    actionSpace: env.actionSpaces[agent] as Space,
    reset: (k) => {
      const r = advance(env.reset(foldIn(k, 0)), foldIn(k, 1))
      return {
        state: { inner: r.state, carry: r.reward },
        observation: env.observe(r.state, agent),
        info: info(r.state),
      }
    },
    step: (s, action, k) => {
      const mover = env.currentAgent(s.inner)
      // The others' opening moves ended the game at reset: one final step delivers its reward.
      if (mover === undefined) {
        const terminated = env.terminations(s.inner)[agent] === true
        return {
          state: { inner: s.inner, carry: 0 },
          observation: env.observe(s.inner, agent),
          reward: s.carry,
          terminated,
          truncated: !terminated,
          info: {},
        }
      }
      if (mover !== agent) throw new MdmlError('value', `aecSingleAgentView: it is not ${agent}'s turn`)
      const r = env.step(s.inner, action, foldIn(k, 0))
      const next = advance(r.state, foldIn(k, 1))
      const over = env.currentAgent(next.state) === undefined
      const terminated = over && env.terminations(next.state)[agent] === true
      return {
        state: { inner: next.state, carry: 0 },
        observation: env.observe(next.state, agent),
        reward: s.carry + (r.rewards[agent] ?? 0) + next.reward,
        terminated,
        truncated: over && !terminated,
        info: info(next.state),
      }
    },
    render: (s) => env.render(s.inner),
    observe: (s) => env.observe(s.inner, agent),
  }
}

/**
 * A turn-based game whose observations are its full states, for planners and search agents that need the true
 * position (minimax, MCTS and AlphaZero-style agents) when playing with {@link playAec}.
 *
 * @param env - The game.
 * @returns The same game with `observe(state, agent) = state`.
 * @concept multi-agent
 * @example
 * ```ts
 * import { stateObservations } from 'mdml-engine/rl'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = stateObservations(nim.create({ heaps: [2, 2] }))
 * env.observe(env.reset(key(0)), 'player_1').heaps // => [2, 2]
 * ```
 */
export function stateObservations<S, O, A>(env: AecEnv<S, O, A>): AecEnv<S, S, A> {
  return { ...env, observe: (s) => s }
}
