import { argmaxOf, type Json, meanOf, WhiteboxError } from '../../../lib/base/index.ts'
import { solveZeroSum } from '../../../lib/games/index.ts'
import { crossEntropy } from '../../../lib/losses/index.ts'
import { mlp } from '../../../lib/nn/index.ts'
import { adam, type Tree } from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../../lib/random/index.ts'
import {
  type AecEnv,
  type Agent,
  aecSingleAgentView,
  behaviourPolicy,
  type Env,
  greedyPolicyOf,
  type ParallelEnv,
  type PerAgent,
  type Policy,
  singleAgentView,
  type Transition,
  trainAgent,
} from '../../../lib/rl/index.ts'
import type { Space } from '../../../lib/space/index.ts'
import { fromStorage, softmax, toFlat, type Value } from '../../../lib/tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../../lib/trace/index.ts'
import { actionCount, gradientStep, rowsTensor } from './_common.ts'
import {
  encodedSize,
  encodeObservation,
  observationKey,
  type Reservoir,
  randomIndex,
  reservoir,
  reservoirAdd,
  reservoirSample,
  sampleIndex,
  type TabularQOptions,
  tabularQ,
} from './_multi-agent.ts'
import { type DqnOptions, type DqnState, dqn } from './value-based.ts'

/** What a multi-agent learner needs from an environment: its agents and their spaces (any `ParallelEnv`). */
export interface MultiAgentSpaces {
  readonly possibleAgents: readonly string[]
  readonly observationSpaces: PerAgent<Space>
  readonly actionSpaces: PerAgent<Space>
}

/** A joint transition of `jointEnv`, seen by one agent: its observation, action, own reward and flags. */
function agentTransition<O, A>(t: Transition<PerAgent<O>, PerAgent<A>>, agent: string): Transition<O, A> | undefined {
  const rewards = t.info?.['rewards'] as Readonly<Record<string, number>> | undefined
  if (rewards !== undefined && !(agent in rewards)) return undefined
  const terminations = t.info?.['terminations'] as Readonly<Record<string, boolean>> | undefined
  const truncations = t.info?.['truncations'] as Readonly<Record<string, boolean>> | undefined
  const observation = t.observation[agent]
  const action = t.action[agent]
  if (observation === undefined || action === undefined) return undefined
  return {
    observation,
    action,
    reward: rewards?.[agent] ?? t.reward,
    nextObservation: t.nextObservation[agent] ?? observation,
    terminated: terminations?.[agent] ?? t.terminated,
    truncated: truncations?.[agent] ?? t.truncated,
  }
}

const runOptions = (o: Partial<Omit<RunOptions, 'maxSteps'>>): Partial<RunOptions> => ({
  ...(o.seed === undefined ? {} : { seed: o.seed }),
  ...(o.key === undefined ? {} : { key: o.key }),
  ...(o.tracer === undefined ? {} : { tracer: o.tracer }),
  ...(o.policy === undefined ? {} : { policy: o.policy }),
  ...(o.runId === undefined ? {} : { runId: o.runId }),
  ...(o.parent === undefined ? {} : { parent: o.parent }),
})

// ---------------------------------------------------------------------------------------- independent learners

/** Options of {@link independentLearners}. */
export interface IndependentLearnersOptions<O, A> extends TabularQOptions {
  /** A learner per agent (default tabular Q-learning with the tabular options). */
  readonly learner?: (agent: string) => Agent<unknown, O, A>
}

/**
 * Independent learners (Tan 1993): one single-agent learner per agent, each treating the others as part of the
 * environment and learning from its own reward. Runs on `jointEnv(parallelEnv)` with `trainAgent`; each agent's
 * transition comes from the joint one (its observation, action and the reward in `info.rewards`). The default
 * learner is tabular Q-learning on the observations used as table keys; pass `learner` for DQN or anything else.
 *
 * @param env - The parallel environment (its agents and spaces).
 * @param options - The learner factory, or the tabular Q-learning options.
 * @returns The joint agent; its state holds every agent's learner state.
 * @concept multi-agent-learning
 * @example
 * ```ts
 * import { independentLearners } from 'whitebox-engine/methods/rl/agents'
 * import { matrixGame } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = independentLearners(matrixGame.create())
 * Object.keys(agent.init(key(0))) // => ['player_0', 'player_1']
 * ```
 */
export function independentLearners<O = unknown, A = number>(
  env: MultiAgentSpaces,
  options: IndependentLearnersOptions<O, A> = {},
): Agent<PerAgent<unknown>, PerAgent<O>, PerAgent<A>> {
  const { learner, ...tabular } = options
  const learners: Record<string, Agent<unknown, O, A>> = {}
  for (const a of env.possibleAgents) {
    learners[a] =
      learner?.(a) ??
      (tabularQ(actionCount(env.actionSpaces[a] as Space, 'independentLearners'), tabular) as unknown as Agent<
        unknown,
        O,
        A
      >)
  }
  const each = <T>(
    obs: PerAgent<O>,
    f: (l: Agent<unknown, O, A>, a: string, o: O, i: number) => T,
  ): Record<string, T> => {
    const out: Record<string, T> = {}
    for (const [i, a] of env.possibleAgents.entries()) {
      const o = obs[a]
      const l = learners[a]
      if (o !== undefined && l !== undefined) out[a] = f(l, a, o, i)
    }
    return out
  }
  return {
    name: 'independentLearners',
    config: { agents: [...env.possibleAgents], learner: learner === undefined ? 'tabularQ' : 'custom' } as Json,
    init: (k) => Object.fromEntries(env.possibleAgents.map((a, i) => [a, learners[a]?.init(foldIn(k, i))])),
    act: (s, obs, k) => each(obs, (l, a, o, i) => l.act(s[a], o, foldIn(k, i))),
    greedy: (s, obs) => each(obs, (l, a, o) => l.greedy(s[a], o)),
    learn: (s, t, k) => {
      const next: Record<string, unknown> = { ...s }
      for (const [i, a] of env.possibleAgents.entries()) {
        const own = agentTransition(t, a)
        const l = learners[a]
        if (own !== undefined && l !== undefined) next[a] = l.learn(s[a], own, foldIn(k, i))
      }
      return next
    },
  }
}

// -------------------------------------------------------------------------------------------------- minimax-Q

/** One agent's minimax-Q tables: `Q[s][a][o]`, the state values and the equilibrium strategies. */
export interface MinimaxQTables {
  readonly Q: Readonly<Record<string, readonly (readonly number[])[]>>
  readonly V: Readonly<Record<string, number>>
  readonly pi: Readonly<Record<string, readonly number[]>>
  readonly alpha: number
}

/** The state of {@link minimaxQ}: each agent's tables and the step count. */
export interface MinimaxQState {
  readonly tables: PerAgent<MinimaxQTables>
  readonly steps: number
}

/** Options of {@link minimaxQ}. */
export interface MinimaxQOptions {
  /** The initial step size (default 1) and its decay per update (default 0.999), down to `alphaMin` (default 0.01). */
  readonly alpha?: number
  readonly alphaDecay?: number
  readonly alphaMin?: number
  /** Discount (default 0.9). */
  readonly gamma?: number
  /** Probability of a uniformly random action (default 0.2, Littman's explor). */
  readonly explore?: number
}

/**
 * Minimax-Q (Littman 1994) for two-player zero-sum Markov games: each agent learns `Q(s, a, o)` over its own action
 * `a` and the opponent's `o` from its own reward, `Q(s, a, o) <- (1 - alpha) Q + alpha (r + gamma V(s'))`, and at
 * every update solves the matrix game `max_pi min_o sum_a pi(a) Q(s, a, o)` for its strategy `pi(s)` and value
 * `V(s)`. It acts by `pi(s)`, with probability `explore` uniformly; `greedy` plays the most probable action of
 * `pi(s)` (read the mixed strategy with {@link minimaxQStrategy}). The matrix games are solved exactly by linear
 * programming (`games.solveZeroSum`). Runs on `jointEnv` of a two-agent
 * `ParallelEnv` with discrete actions; observations are table keys.
 *
 * @param env - The two-agent parallel environment.
 * @param options - Step sizes, discount and exploration.
 * @returns The joint agent (both players learn by minimax-Q).
 * @throws WhiteboxError with code `value` unless there are exactly two agents with discrete actions.
 * @concept multi-agent-learning
 * @example
 * ```ts
 * import { minimaxQ } from 'whitebox-engine/methods/rl/agents'
 * import { matrixGame } from 'whitebox-engine/methods/rl'
 * minimaxQ(matrixGame.create({ game: 'matchingPennies' })).name // => 'minimaxQ'
 * ```
 */
export function minimaxQ(
  env: MultiAgentSpaces,
  options: MinimaxQOptions = {},
): Agent<MinimaxQState, PerAgent<unknown>, PerAgent<number>> {
  if (env.possibleAgents.length !== 2) throw new WhiteboxError('value', 'minimaxQ: needs exactly two agents')
  const [p0, p1] = env.possibleAgents as [string, string]
  const n: Record<string, number> = {
    [p0]: actionCount(env.actionSpaces[p0] as Space, 'minimaxQ'),
    [p1]: actionCount(env.actionSpaces[p1] as Space, 'minimaxQ'),
  }
  const opponent = (a: string): string => (a === p0 ? p1 : p0)
  const gamma = options.gamma ?? 0.9
  const explore = options.explore ?? 0.2
  const decay = options.alphaDecay ?? 0.999
  const alphaMin = options.alphaMin ?? 0.01
  const strategy = (t: MinimaxQTables | undefined, agent: string, o: unknown): readonly number[] => {
    const m = n[agent] ?? 1
    return t?.pi[observationKey(o)] ?? new Array<number>(m).fill(1 / m)
  }
  return {
    name: 'minimaxQ',
    config: { gamma, explore, alphaDecay: decay } as Json,
    init: () => ({
      tables: Object.fromEntries(
        env.possibleAgents.map((a) => [a, { Q: {}, V: {}, pi: {}, alpha: options.alpha ?? 1 }]),
      ),
      steps: 0,
    }),
    act: (s, obs, k) => {
      const out: Record<string, number> = {}
      for (const [i, a] of env.possibleAgents.entries()) {
        const ki = foldIn(k, i)
        out[a] =
          (uniform(foldIn(ki, 0), []).data[0] ?? 0) < explore
            ? randomIndex(foldIn(ki, 1), n[a] ?? 1)
            : sampleIndex(foldIn(ki, 2), strategy(s.tables[a], a, obs[a]))
      }
      return out
    },
    greedy: (s, obs) =>
      Object.fromEntries(env.possibleAgents.map((a) => [a, argmaxOf(strategy(s.tables[a], a, obs[a]))])),
    learn: (s, t) => {
      const tables: Record<string, MinimaxQTables> = {}
      for (const a of env.possibleAgents) {
        const tab = s.tables[a] as MinimaxQTables
        const own = agentTransition(t, a)
        const o = t.action[opponent(a)]
        if (own === undefined || o === undefined) {
          tables[a] = tab
          continue
        }
        const key = observationKey(own.observation)
        const m = n[a] ?? 1
        const k = n[opponent(a)] ?? 1
        const q = tab.Q[key] ?? Array.from({ length: m }, () => new Array<number>(k).fill(0))
        const target = own.reward + (own.terminated ? 0 : gamma * (tab.V[observationKey(own.nextObservation)] ?? 0))
        const updated = q.map((row, i) =>
          i === own.action ? row.map((v, j) => (j === o ? v + tab.alpha * (target - v) : v)) : row,
        )
        const solution = solveZeroSum(updated)
        tables[a] = {
          Q: { ...tab.Q, [key]: updated },
          V: { ...tab.V, [key]: solution.value },
          pi: { ...tab.pi, [key]: solution.row },
          alpha: Math.max(alphaMin, tab.alpha * decay),
        }
      }
      return { tables, steps: s.steps + 1 }
    },
  }
}

/**
 * The mixed strategy a minimax-Q agent has learned for an observation (uniform where it has not learned).
 *
 * @param state - The minimax-Q state.
 * @param agent - The agent.
 * @param observation - The observation.
 * @returns Action probabilities.
 * @concept multi-agent-learning
 * @example
 * ```ts
 * import { minimaxQ, minimaxQStrategy } from 'whitebox-engine/methods/rl/agents'
 * import { matrixGame } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const agent = minimaxQ(matrixGame.create({ game: 'rockPaperScissors' }))
 * minimaxQStrategy(agent.init(key(0)), 'player_0', 0).length // => 0
 * ```
 */
export function minimaxQStrategy(state: MinimaxQState, agent: string, observation: unknown): readonly number[] {
  return state.tables[agent]?.pi[observationKey(observation)] ?? []
}

// --------------------------------------------------------------------------------------------------- self-play

/** Options of {@link selfPlayAlgorithm} and {@link selfPlay}. */
export interface SelfPlayOptions {
  /** The seat the learner trains in (default the first agent). */
  readonly seat?: string
  /** Swap between the first two seats each generation (default false), for games where one network plays both. */
  readonly alternateSeats?: boolean
  /** Environment steps of training per generation (default 1000). */
  readonly stepsPerGeneration?: number
  /** Snapshots kept as opponents (default 10). */
  readonly poolSize?: number
  /** Probability of facing the latest snapshot rather than a uniform draw from the pool (default 0.5). */
  readonly latestProbability?: number
  /** How snapshots play: their greedy policy (default) or their exploratory one. */
  readonly opponent?: 'greedy' | 'behaviour'
}

/** The state of {@link selfPlayAlgorithm}. */
export interface SelfPlayState<G> {
  readonly learner: G
  /** Frozen snapshots of the learner, oldest first. */
  readonly pool: readonly G[]
  readonly generation: number
  /** Mean training return of the last generation, and the pool index of its opponent. */
  readonly meanReturn: number
  readonly opponent: number
}

const isAec = <S, O, A>(env: AecEnv<S, O, A> | ParallelEnv<S, O, A>): env is AecEnv<S, O, A> => 'currentAgent' in env

/**
 * Self-play against a pool of past selves (as in fictitious-play-style league training): each generation picks an
 * opponent snapshot (the latest with probability `latestProbability`, otherwise uniformly from the pool), trains the
 * single-agent learner for `stepsPerGeneration` steps in the one-seat view of the game with every other seat played
 * by that snapshot (`aecSingleAgentView` for turn-based games, `singleAgentView` for simultaneous ones), and then
 * adds the learner to the pool. Records the mean training return per generation.
 *
 * @param env - The game (turn-based or simultaneous).
 * @param learner - A single-agent learner for the seat's observations and actions (DQN, tabular, ...).
 * @param options - Seats, steps per generation and the snapshot pool.
 * @returns The algorithm.
 * @concept self-play
 * @example
 * ```ts
 * import { dqn, selfPlayAlgorithm } from 'whitebox-engine/methods/rl/agents'
 * import { nim } from 'whitebox-engine/methods/rl'
 * const env = nim.create({ heaps: [1, 2] })
 * const spaces = { observationSpace: env.observationSpaces['player_0'], actionSpace: env.actionSpaces['player_0'] }
 * selfPlayAlgorithm(env, dqn(spaces, { hidden: [8] })).name // => 'selfPlay:dqn'
 * ```
 */
export function selfPlayAlgorithm<G, S, O, A>(
  env: AecEnv<S, O, A> | ParallelEnv<S, O, A>,
  learner: Agent<G, O, A>,
  options: SelfPlayOptions = {},
): Algorithm<SelfPlayState<G>> {
  const first = options.seat ?? env.possibleAgents[0] ?? 'player_0'
  const seats = options.alternateSeats === true ? env.possibleAgents.slice(0, 2) : [first]
  const steps = options.stepsPerGeneration ?? 1000
  const poolSize = options.poolSize ?? 10
  const latest = options.latestProbability ?? 0.5
  const policyOf = (g: G): Policy<O, A> =>
    options.opponent === 'behaviour' ? behaviourPolicy(learner, g) : greedyPolicyOf(learner, g)
  const viewFor = (seat: string, opponent: G): Env<unknown, O, A> => {
    const others = Object.fromEntries(env.possibleAgents.filter((a) => a !== seat).map((a) => [a, policyOf(opponent)]))
    return (isAec(env) ? aecSingleAgentView(env, seat, others) : singleAgentView(env, seat, others)) as Env<
      unknown,
      O,
      A
    >
  }
  return {
    name: `selfPlay:${learner.name}`,
    config: { env: env.spec.id, learner: learner.config, stepsPerGeneration: steps, poolSize, latest } as Json,
    schema: channels({
      metrics: {
        meanReturn: { description: 'mean training return of the generation', direction: 'higher' },
        opponent: { description: 'pool index of the generation opponent' },
      },
    }),
    init: (k) => {
      const g = learner.init(foldIn(k, 0))
      return { learner: g, pool: [g], generation: 0, meanReturn: 0, opponent: 0 }
    },
    step: (s, ctx) => {
      const u = uniform(foldIn(ctx.key, 0), [2]).data
      const index =
        (u[0] ?? 0) < latest ? s.pool.length - 1 : Math.min(s.pool.length - 1, Math.floor((u[1] ?? 0) * s.pool.length))
      const seat = seats[s.generation % seats.length] ?? first
      const opponent = s.pool[index] as G
      const current = s.learner
      const trained = trainAgent(
        viewFor(seat, opponent),
        { ...learner, init: () => current },
        {
          maxSteps: steps,
          key: foldIn(ctx.key, 1),
          checkpoints: false,
          parent: ctx.run,
        },
      )
      return {
        learner: trained.agent,
        pool: [...s.pool, trained.agent].slice(-poolSize),
        generation: s.generation + 1,
        meanReturn: trained.returns.length === 0 ? 0 : meanOf(trained.returns),
        opponent: index,
      }
    },
    observe: (s) => ({ metrics: { meanReturn: s.meanReturn, opponent: s.opponent } }),
  }
}

/** The result of {@link selfPlay}. */
export interface SelfPlayResult<G> {
  readonly learner: G
  readonly pool: readonly G[]
  readonly state: SelfPlayState<G>
  readonly runId: string
}

/**
 * Train a learner by self-play against snapshots of itself (see {@link selfPlayAlgorithm}).
 *
 * @param env - The game.
 * @param learner - The single-agent learner.
 * @param options - Self-play options, `generations` (default 10) and run options.
 * @returns The trained learner state and the snapshot pool.
 * @concept self-play
 * @example
 * ```ts
 * import { selfPlay } from 'whitebox-engine/methods/rl/agents'
 * import { matrixGame } from 'whitebox-engine/methods/rl'
 * import { policyAgent } from 'whitebox-engine/rl'
 * selfPlay(matrixGame.create({ rounds: 5 }), policyAgent(() => 1), { generations: 2, stepsPerGeneration: 5 }).state.meanReturn // => 5
 * ```
 */
export function selfPlay<G, S, O, A>(
  env: AecEnv<S, O, A> | ParallelEnv<S, O, A>,
  learner: Agent<G, O, A>,
  options: SelfPlayOptions & { readonly generations?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): SelfPlayResult<G> {
  const {
    generations,
    seed: _seed,
    key: _key,
    tracer: _tracer,
    policy: _policy,
    runId: _runId,
    parent: _parent,
    start: _s,
    ...rest
  } = options
  const result = runAlgorithm(selfPlayAlgorithm(env, learner, rest), {
    ...runOptions(options),
    maxSteps: generations ?? 10,
  })
  return { learner: result.state.learner, pool: result.state.pool, state: result.state, runId: result.runId }
}

// ---------------------------------------------------------------------------------------- fictitious self-play

/** Options of {@link fictitiousSelfPlayAlgorithm}. */
export interface FictitiousSelfPlayOptions extends DqnOptions {
  /** Anticipatory parameter: probability of playing the best response in an episode (default 0.1). */
  readonly eta?: number
  /** Hidden widths of the average-policy network (default `[64]`), its learning rate (default 1e-3). */
  readonly averageHidden?: readonly number[]
  readonly averageLearningRate?: number
  /** Reservoir of the best response's (observation, action) pairs (default 2000), batch and updates per episode. */
  readonly reservoirSize?: number
  readonly averageBatchSize?: number
  readonly averageUpdates?: number
}

/** One player's state in neural fictitious self-play. */
export interface FspPlayer {
  readonly br: DqnState
  readonly average: Tree
  readonly averageOpt: unknown
  readonly reservoir: Reservoir<{ readonly observation: readonly number[]; readonly action: number }>
  readonly averageLoss: number
}

/** The state of {@link fictitiousSelfPlayAlgorithm}. */
export interface FspState {
  readonly players: PerAgent<FspPlayer>
  readonly episodes: number
  /** The returns of the last episode. */
  readonly returns: PerAgent<number>
}

interface FspModel {
  readonly best: Agent<DqnState, readonly number[], number>
  readonly averageApply: (p: Tree, x: Value) => Value
  readonly averageParams: Tree
  readonly size: number
  readonly nActions: number
}

function fspModel<S>(env: AecEnv<S, readonly number[], number>, options: FictitiousSelfPlayOptions): FspModel {
  const first = env.possibleAgents[0] ?? 'player_0'
  const observationSpace = env.observationSpaces[first] as Space
  const actionSpace = env.actionSpaces[first] as Space
  const size = encodedSize(observationSpace)
  const nActions = actionCount(actionSpace, 'fictitiousSelfPlay')
  const net = mlp([size, ...(options.averageHidden ?? [64]), nActions], {
    key: foldIn(options.key ?? makeKey(0), 77),
  })
  return {
    best: dqn({ observationSpace, actionSpace }, options),
    averageApply: (p, x) => net.apply(p as typeof net.params, x),
    averageParams: net.params,
    size,
    nActions,
  }
}

/** The average policy's action probabilities, restricted to the legal actions when given. */
function averageProbabilities(
  model: FspModel,
  params: Tree,
  observation: readonly number[],
  legal?: readonly number[],
): number[] {
  const logits = toFlat(model.averageApply(params, rowsTensor([observation], model.size)))
  const allowed = legal ?? Array.from({ length: model.nActions }, (_, a) => a)
  const p = toFlat(
    softmax(fromStorage(Float64Array.from(allowed.map((a) => logits[a] ?? 0)), [allowed.length], 'float64'), -1),
  )
  const out = new Array<number>(model.nActions).fill(0)
  for (const [i, a] of allowed.entries()) out[a] = p[i] ?? 0
  return out
}

/**
 * Neural fictitious self-play (Heinrich and Silver 2016; fictitious self-play, Heinrich, Lanctot and Silver 2015)
 * for turn-based games, as an {@link Algorithm} over episodes. Every player has a best-response learner (DQN on the
 * transitions between its own turns) and an average-policy network trained by supervised learning (cross-entropy) on
 * a reservoir of its best response's own actions, so it tracks the average of past best responses: fictitious play
 * in sample-based form. In each episode a player plays its best response (epsilon-greedy) with the anticipatory
 * probability `eta` and its average policy otherwise; the average policy approaches a Nash equilibrium in two-player
 * zero-sum games. Records the players' returns and average-policy losses.
 *
 * @param env - A turn-based game with number-array observations (Kuhn and Leduc poker's information states).
 * @param options - DQN options for the best responses, `eta`, and the average network and reservoir.
 * @returns The algorithm.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { fictitiousSelfPlayAlgorithm } from 'whitebox-engine/methods/rl/agents'
 * import { kuhnPoker } from 'whitebox-engine/methods/rl'
 * fictitiousSelfPlayAlgorithm(kuhnPoker.create(), { hidden: [16] }).name // => 'fictitiousSelfPlay'
 * ```
 */
export function fictitiousSelfPlayAlgorithm<S>(
  env: AecEnv<S, readonly number[], number>,
  options: FictitiousSelfPlayOptions = {},
): Algorithm<FspState> {
  const model = fspModel(env, options)
  const eta = options.eta ?? 0.1
  const opt = adam({ learningRate: options.averageLearningRate ?? 1e-3 })
  const capacity = options.reservoirSize ?? 2000
  const batch = options.averageBatchSize ?? 64
  const updates = options.averageUpdates ?? 1
  const agents = env.possibleAgents
  const pending = (): Record<
    string,
    { observation: readonly number[]; action: number; reward: number } | undefined
  > => ({})
  return {
    name: 'fictitiousSelfPlay',
    config: { env: env.spec.id, eta, reservoirSize: capacity, averageBatchSize: batch } as Json,
    schema: channels({
      metrics: {
        averageLoss: { description: 'cross-entropy of the average policy on the reservoir', direction: 'lower' },
        firstPlayerReturn: { description: 'return of player_0 in the episode' },
      },
    }),
    init: (k) => ({
      players: Object.fromEntries(
        agents.map((a, i) => [
          a,
          {
            br: model.best.init(foldIn(k, i)),
            average: model.averageParams,
            averageOpt: opt.init(model.averageParams),
            reservoir: reservoir(capacity),
            averageLoss: 0,
          },
        ]),
      ),
      episodes: 0,
      returns: {},
    }),
    step: (s, ctx) => {
      const players: Record<string, FspPlayer> = { ...s.players } as Record<string, FspPlayer>
      const modes = Object.fromEntries(
        agents.map((a, i) => [a, (uniform(foldIn(foldIn(ctx.key, 0), i), []).data[0] ?? 0) < eta ? 'best' : 'average']),
      )
      const last = pending()
      const returns: Record<string, number> = Object.fromEntries(agents.map((a) => [a, 0]))
      let state = env.reset(foldIn(ctx.key, 1))
      const learnFrom = (a: string, next: readonly number[], terminated: boolean, k: Key): void => {
        const p = last[a]
        const pl = players[a]
        if (p === undefined || pl === undefined) return
        const t: Transition<readonly number[], number> = {
          observation: p.observation,
          action: p.action,
          reward: p.reward,
          nextObservation: next,
          terminated,
          truncated: false,
        }
        players[a] = { ...pl, br: model.best.learn(pl.br, t, k) }
        last[a] = undefined
      }
      for (let t = 0; t < 10_000; t++) {
        const mover = env.currentAgent(state)
        if (mover === undefined) break
        const kt = foldIn(foldIn(ctx.key, 2), t)
        const observation = encodeObservation(env.observationSpaces[mover], env.observe(state, mover))
        learnFrom(mover, observation, false, foldIn(kt, 0))
        const pl = players[mover] as FspPlayer
        let action: number
        if (modes[mover] === 'best') {
          action = model.best.act(pl.br, observation, foldIn(kt, 1))
          players[mover] = { ...pl, reservoir: reservoirAdd(pl.reservoir, { observation, action }, foldIn(kt, 2)) }
        } else {
          const legal = env.legalActions?.(state, mover)
          action = sampleIndex(foldIn(kt, 3), averageProbabilities(model, pl.average, observation, legal))
        }
        const r = env.step(state, action, foldIn(kt, 4))
        for (const [a, x] of Object.entries(r.rewards)) {
          returns[a] = (returns[a] ?? 0) + x
          const p = last[a]
          if (p !== undefined) last[a] = { ...p, reward: p.reward + x }
        }
        last[mover] = { observation, action, reward: r.rewards[mover] ?? 0 }
        state = r.state
      }
      for (const [i, a] of agents.entries()) {
        learnFrom(
          a,
          encodeObservation(env.observationSpaces[a], env.observe(state, a)),
          true,
          foldIn(foldIn(ctx.key, 3), i),
        )
        let pl = players[a] as FspPlayer
        for (let u = 0; u < updates && pl.reservoir.size > 0; u++) {
          const items = reservoirSample(
            pl.reservoir,
            foldIn(foldIn(foldIn(ctx.key, 4), i), u),
            Math.min(batch, pl.reservoir.size),
          )
          const x = rowsTensor(
            items.map((it) => it.observation),
            model.size,
          )
          const y = fromStorage(Float64Array.from(items.map((it) => it.action)), [items.length], 'float64')
          const step = gradientStep((p) => crossEntropy(model.averageApply(p, x), y), pl.average, opt, pl.averageOpt)
          pl = { ...pl, average: step.params, averageOpt: step.optState, averageLoss: step.loss }
        }
        players[a] = pl
      }
      return { players, episodes: s.episodes + 1, returns }
    },
    observe: (s) => ({
      metrics: {
        averageLoss: meanOf(agents.map((a) => s.players[a]?.averageLoss ?? 0)),
        firstPlayerReturn: s.returns[agents[0] ?? 'player_0'] ?? 0,
      },
    }),
  }
}

/** The result of {@link fictitiousSelfPlay}. */
export interface FictitiousSelfPlayResult {
  readonly state: FspState
  /** The average policy of an agent as action probabilities (restricted to legal actions when given). */
  averagePolicy(agent: string, observation: readonly number[], legal?: readonly number[]): number[]
  /** The average policy of an agent as a sampling policy. */
  policy(agent: string): Policy<readonly number[], number>
  readonly runId: string
}

/**
 * Train by neural fictitious self-play (see {@link fictitiousSelfPlayAlgorithm}) for a number of episodes; the
 * result exposes each player's average policy, the strategy that approaches equilibrium.
 *
 * @param env - The turn-based game.
 * @param options - Algorithm options, `episodes` (default 1000) and run options.
 * @returns The final state and the average policies.
 * @concept fictitious-play
 * @example
 * ```ts
 * import { fictitiousSelfPlay } from 'whitebox-engine/methods/rl/agents'
 * import { kuhnPoker } from 'whitebox-engine/methods/rl'
 * const r = fictitiousSelfPlay(kuhnPoker.create(), { hidden: [8], averageHidden: [8], episodes: 3 })
 * r.averagePolicy('player_0', [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).length // => 2
 * ```
 */
export function fictitiousSelfPlay<S>(
  env: AecEnv<S, readonly number[], number>,
  options: FictitiousSelfPlayOptions & { readonly episodes?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): FictitiousSelfPlayResult {
  const {
    episodes,
    seed: _seed,
    key: _key,
    tracer: _tracer,
    policy: _policy,
    runId: _runId,
    parent: _parent,
    start: _s,
    ...rest
  } = options
  const model = fspModel(env, rest)
  const result = runAlgorithm(fictitiousSelfPlayAlgorithm(env, rest), {
    ...runOptions(options),
    maxSteps: episodes ?? 1000,
  })
  const params = (agent: string): Tree => {
    const p = result.state.players[agent]
    if (p === undefined) throw new WhiteboxError('value', `fictitiousSelfPlay: no agent ${agent}`)
    return p.average
  }
  return {
    state: result.state,
    averagePolicy: (agent, observation, legal) => averageProbabilities(model, params(agent), observation, legal),
    policy: (agent) => (o, k) => sampleIndex(k, averageProbabilities(model, params(agent), o)),
    runId: result.runId,
  }
}
