/**
 * Planning agents that act from a model rather than learning from reward: dynamic programming (value or policy
 * iteration solved once, or a few sweeps at a time), Monte Carlo tree search over an environment used as its own model,
 * and players of two-player zero-sum games. For games: MCTS with negamax backups, alpha-beta minimax search,
 * AlphaZero-style self-play training of a policy and value network that guides PUCT search, and a random legal-move
 * baseline.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { crossEntropy, mse } from '../losses/index.ts'
import { mlp } from '../nn/index.ts'
import { adamw, type Tree } from '../optim/index.ts'
import { categorical, dirichlet, foldIn, type Key, key as makeKey } from '../random/index.ts'
import {
  type AecEnv,
  type Agent,
  bufferAdd,
  bufferSample,
  type Env,
  gameMcts,
  greedyPolicy,
  type MctsOptions,
  mcts,
  mdpFromEnv,
  type Policy,
  type PolicyIterationState,
  policyIteration,
  policyIterationAlgorithm,
  qFromV,
  type ReplayBuffer,
  replayBuffer,
  type ValueIterationState,
  valueIteration,
  valueIterationAlgorithm,
} from '../rl/index.ts'
import type { Space } from '../space/index.ts'
import { add, fromStorage, mul, sliceAxis, softmax, tanh, toFlat, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  channels,
  nullTracer,
  type RunOptions,
  runAlgorithm,
  type StepContext,
  startRun,
  stepContext,
} from '../trace/index.ts'
import { gradientStep, rowsTensor } from './_common.ts'

const identity = <O, S>(o: O): S => o as unknown as S
const GREEDY_KEY = makeKey(0x6d637473)

function discreteCount(space: Space | undefined, name: string): number {
  if (space?.kind !== 'discrete') throw new MdmlError('value', `${name}: needs discrete action spaces`)
  return space.n
}

// ------------------------------------------------------------------------------------------- dynamic programming

/** The state of a planning agent built on a tabular model: the optimal values and the greedy policy. */
export interface TabularPlanState {
  readonly V: readonly number[]
  readonly policy: readonly number[]
  readonly iterations: number
}

/** Options of {@link valueIterationAgent} and {@link policyIterationAgent}. */
export interface DynamicProgrammingOptions<O, S> {
  /** The discount (default 0.99). */
  readonly gamma?: number
  /** The model state of an observation (default the observation itself). */
  readonly toState?: (observation: O) => S
  /** Value-iteration tolerance (default 1e-10) and iteration cap. */
  readonly tol?: number
  readonly maxIter?: number
}

function tabularPlanner<S, O, A>(
  name: 'valueIteration' | 'policyIteration',
  env: Env<S, O, A>,
  options: DynamicProgrammingOptions<O, S>,
): Agent<TabularPlanState, O, A> {
  const model = env.model
  if (model === undefined) throw new MdmlError('value', `${name}Agent: ${env.spec.id} has no model`)
  const gamma = options.gamma ?? 0.99
  const toState = options.toState ?? identity<O, S>
  const act = (s: TabularPlanState, o: O): A => model.actions[s.policy[model.stateIndex(toState(o))] ?? 0] as A
  return {
    name: `${name}Agent`,
    config: { env: env.spec.id, gamma, planner: name } as Json,
    metrics: { startValue: { description: 'optimal value of the initial-state distribution', direction: 'higher' } },
    init: () => {
      const mdp = mdpFromEnv(env, gamma)
      const solution =
        name === 'valueIteration'
          ? valueIteration(mdp, {
              tol: options.tol ?? 1e-10,
              ...(options.maxIter === undefined ? {} : { maxIter: options.maxIter }),
            })
          : policyIteration(mdp, options.maxIter === undefined ? {} : { maxIter: options.maxIter })
      return { V: solution.V, policy: solution.policy, iterations: solution.iterations }
    },
    act: (s, o) => act(s, o),
    greedy: (s, o) => act(s, o),
    // The plan is exact for the model; experience adds nothing.
    learn: (s) => s,
    stats: (s) => {
      let v = 0
      for (const { probability, state } of model.initial) v += probability * (s.V[model.stateIndex(state)] ?? 0)
      return { startValue: v }
    },
  }
}

/**
 * An agent that solves the environment's model by value iteration once (in `init`) and then acts greedily on the
 * optimal action values; `learn` keeps the plan, which is already exact for the model.
 *
 * @param env - An environment with a tabular model (`env.model`).
 * @param options - Discount, tolerance and the map from observations to model states.
 * @returns The agent.
 * @throws MdmlError with code `value` when the environment has no model.
 * @concept value-iteration
 * @example
 * ```ts
 * import { valueIterationAgent } from 'mdml-engine/agents'
 * import { cliffWalking } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = valueIterationAgent(cliffWalking.create(), { gamma: 1 })
 * agent.greedy(agent.init(key(0)), 36) // => 0
 * ```
 */
export function valueIterationAgent<S, O, A>(
  env: Env<S, O, A>,
  options: DynamicProgrammingOptions<O, S> = {},
): Agent<TabularPlanState, O, A> {
  return tabularPlanner('valueIteration', env, options)
}

/**
 * An agent that solves the environment's model by policy iteration once (in `init`) and then follows the optimal
 * policy; `learn` keeps the plan.
 *
 * @param env - An environment with a tabular model.
 * @param options - Discount, iteration cap and the map from observations to model states.
 * @returns The agent.
 * @throws MdmlError with code `value` when the environment has no model.
 * @concept policy-iteration
 * @example
 * ```ts
 * import { policyIterationAgent } from 'mdml-engine/agents'
 * import { cliffWalking } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = policyIterationAgent(cliffWalking.create(), { gamma: 0.9 })
 * agent.greedy(agent.init(key(0)), 36) // => 0
 * ```
 */
export function policyIterationAgent<S, O, A>(
  env: Env<S, O, A>,
  options: DynamicProgrammingOptions<O, S> = {},
): Agent<TabularPlanState, O, A> {
  return tabularPlanner('policyIteration', env, options)
}

/** Options of {@link sweepingPlannerAgent}. */
export interface SweepingPlannerOptions<O, S> {
  /** The planner: value iteration (default) or policy iteration. */
  readonly planner?: 'valueIteration' | 'policyIteration'
  /** The discount (default 0.99). */
  readonly gamma?: number
  /** Planner iterations (Bellman sweeps, or evaluate-and-improve steps) after each block of episodes (default 1). */
  readonly sweeps?: number
  /** Episodes per block (default 1). */
  readonly every?: number
  /** The model state of an observation (default the observation itself). */
  readonly toState?: (observation: O) => S
}

/** The state of {@link sweepingPlannerAgent}: the planner's values and greedy policy after some iterations. */
export interface SweepingPlanState {
  readonly V: readonly number[]
  readonly policy: readonly number[]
  readonly iterations: number
  readonly episodes: number
  /** Value iteration's last sup-norm change, or the actions policy iteration last changed. */
  readonly residual: number
}

/**
 * A planning agent that plans a little at a time: it starts from zero values (value iteration) or a fixed policy
 * (policy iteration) and runs `sweeps` iterations of the planner on the environment's model after every `every`
 * episodes, acting greedily on its current plan in between. Its returns therefore improve episode by episode as the
 * values propagate from the rewards, which makes dynamic programming visible on a learning curve (the
 * {@link valueIterationAgent} solves the model in full before the first step). The sweeps are
 * `valueIterationAlgorithm` and `policyIterationAlgorithm` of `lib/rl`.
 *
 * @param env - An environment with a tabular model (`env.model`).
 * @param options - The planner, discount, sweeps per block, episodes per block and the map to model states.
 * @returns The agent.
 * @throws MdmlError with code `value` when the environment has no model.
 * @concept value-iteration
 * @example
 * ```ts
 * import { sweepingPlannerAgent } from 'mdml-engine/agents'
 * import { cliffWalking } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = sweepingPlannerAgent(cliffWalking.create(), { gamma: 1 })
 * agent.init(key(0)).iterations // => 0
 * ```
 */
export function sweepingPlannerAgent<S, O, A>(
  env: Env<S, O, A>,
  options: SweepingPlannerOptions<O, S> = {},
): Agent<SweepingPlanState, O, A> {
  const model = env.model
  if (model === undefined) throw new MdmlError('value', `sweepingPlannerAgent: ${env.spec.id} has no model`)
  const planner = options.planner ?? 'valueIteration'
  const gamma = options.gamma ?? 0.99
  const sweeps = options.sweeps ?? 1
  const every = options.every ?? 1
  const toState = options.toState ?? identity<O, S>
  const mdp = mdpFromEnv(env, gamma)
  const run = startRun(nullTracer, { name: 'sweep', seed: null, config: {} })
  const ctx = (t: number): StepContext => stepContext(t, SWEEP_KEY, run)
  const greedyOf = (V: readonly number[]): number[] => greedyPolicy(qFromV(mdp, V, gamma))
  const vi = valueIterationAlgorithm(mdp, { gamma })
  const pi = policyIterationAlgorithm(mdp, { gamma })
  const advance = (s: SweepingPlanState): SweepingPlanState => {
    if (planner === 'valueIteration') {
      let state: ValueIterationState = { V: Float64Array.from(s.V), delta: s.residual, iteration: s.iterations }
      for (let i = 0; i < sweeps; i++) state = vi.step(state, ctx(state.iteration))
      const V = Array.from(state.V)
      return { ...s, V, policy: greedyOf(V), iterations: state.iteration, residual: state.delta }
    }
    let state: PolicyIterationState = { policy: s.policy, V: s.V, changed: s.residual, iteration: s.iterations }
    for (let i = 0; i < sweeps; i++) state = pi.step(state, ctx(state.iteration))
    return { ...s, V: state.V, policy: state.policy, iterations: state.iteration, residual: state.changed }
  }
  const act = (s: SweepingPlanState, o: O): A => model.actions[s.policy[model.stateIndex(toState(o))] ?? 0] as A
  return {
    name: 'sweepingPlanner',
    config: { env: env.spec.id, planner, gamma, sweeps, every } as Json,
    metrics: {
      iterations: { description: 'planner iterations so far' },
      residual: { description: 'last value change (value iteration) or actions changed (policy iteration)' },
    },
    init: () => {
      if (planner === 'valueIteration') {
        const V = new Array<number>(mdp.nStates).fill(0)
        return { V, policy: greedyOf(V), iterations: 0, episodes: 0, residual: Infinity }
      }
      const s = pi.init(SWEEP_KEY)
      return { V: s.V, policy: s.policy, iterations: 0, episodes: 0, residual: s.changed }
    },
    act,
    greedy: act,
    learn: (s, t) => {
      if (!(t.terminated || t.truncated)) return s
      const episodes = s.episodes + 1
      return episodes % every === 0 ? advance({ ...s, episodes }) : { ...s, episodes }
    },
    stats: (s) => ({ iterations: s.iterations, residual: Number.isFinite(s.residual) ? s.residual : Number.MAX_VALUE }),
  }
}

const SWEEP_KEY = makeKey(0x73776570)

// --------------------------------------------------------------------------------------------------------- MCTS

/** Options of {@link mctsAgent}. */
export interface MctsAgentOptions<S, O, A> extends MctsOptions<S, O, A> {
  /** Simulations per decision (default 100). */
  readonly simulations?: number
  /** The true environment state behind an observation (default the observation itself). */
  readonly toState?: (observation: O) => S
}

/**
 * An agent that plans every action by Monte Carlo tree search over the environment used as its own model (UCT, or
 * PUCT with a prior and a value function), from the true state behind the observation. Planning needs no learning:
 * `learn` keeps the (empty) state.
 *
 * @param env - The environment (its pure `step` is the model).
 * @param options - Search options, simulations per decision and `toState`.
 * @returns The agent.
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { mctsAgent } from 'mdml-engine/agents'
 * import { cartPole } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const agent = mctsAgent(cartPole.create(), { simulations: 300, depth: 25, c: 5 })
 * agent.greedy(agent.init(key(0)), [0, 0, 0.12, 0]) // => 1
 * ```
 */
export function mctsAgent<S, O, A>(env: Env<S, O, A>, options: MctsAgentOptions<S, O, A> = {}): Agent<null, O, A> {
  const { toState, simulations, ...search } = options
  const state = toState ?? identity<O, S>
  const plan = (o: O, k: Key): A => mcts(env, state(o), { ...search, simulations: simulations ?? 100, key: k }).action
  return {
    name: 'mcts',
    config: { env: env.spec.id, simulations: simulations ?? 100, c: options.c ?? Math.SQRT2 } as Json,
    init: () => null,
    act: (_, o, k) => plan(o, k),
    greedy: (_, o) => plan(o, GREEDY_KEY),
    learn: (s) => s,
  }
}

/** Options of {@link gameMctsAgent}. */
export interface GameMctsAgentOptions<S, O> {
  readonly simulations?: number
  readonly c?: number
  readonly prior?: (state: S, actions: readonly number[]) => readonly number[]
  readonly value?: (state: S, player: string) => number
  readonly maxTurns?: number
  /** The game state behind an observation (default the observation itself: use `stateObservations(env)`). */
  readonly toState?: (observation: O) => S
}

/**
 * A game-playing agent for two-player zero-sum turn-based games that chooses each move by Monte Carlo tree search
 * with negamax backups (`gameMcts`: UCT with random playouts, or PUCT with a prior and value function).
 *
 * @param env - The game.
 * @param options - Simulations, exploration constant, prior and value, and `toState`.
 * @returns The agent (stateless).
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { gameMctsAgent } from 'mdml-engine/agents'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = nim.create({ heaps: [2] })
 * gameMctsAgent(env, { simulations: 100 }).greedy(null, env.reset(key(0))) // => 1
 * ```
 */
export function gameMctsAgent<S, O>(
  env: AecEnv<S, O, number>,
  options: GameMctsAgentOptions<S, O> = {},
): Agent<null, O | S, number> {
  const { toState, ...search } = options
  const state = (o: O | S): S => (toState === undefined ? (o as S) : toState(o as O))
  const plan = (o: O | S, k: Key): number => gameMcts(env, state(o), { ...search, key: k }).action
  return {
    name: 'gameMcts',
    config: { env: env.spec.id, simulations: options.simulations ?? 200 } as Json,
    init: () => null,
    act: (_, o, k) => plan(o, k),
    greedy: (_, o) => plan(o, GREEDY_KEY),
    learn: (s) => s,
  }
}

// -------------------------------------------------------------------------------------------------- alpha-beta

/** Options of {@link alphaBeta}. */
export interface AlphaBetaOptions<S> {
  /** Search depth in plies (default unlimited). */
  readonly depth?: number
  /** The value of a non-terminal position at the depth limit for the player to move there (default 0). */
  readonly evaluate?: (state: S, player: string) => number
  /** Move ordering: the legal actions in the order to try them (default as given; the table's best move first). */
  readonly order?: (state: S, actions: readonly number[]) => readonly number[]
  /** A key identifying positions for the transposition table (default JSON); `null` disables the table. */
  readonly stateKey?: ((state: S) => string) | null
}

/** The result of an alpha-beta search. */
export interface AlphaBetaResult {
  /** The best move at the root (the first of equal ones in search order). */
  readonly action: number
  /** The negamax value of the root for the player to move. */
  readonly value: number
  /** Positions searched. */
  readonly nodes: number
}

interface TableEntry {
  readonly depth: number
  readonly value: number
  readonly flag: 'exact' | 'lower' | 'upper'
  readonly best: number | undefined
}

/**
 * Alpha-beta search in negamax form (Knuth and Moore 1975) for two-player zero-sum turn-based games: the value of a
 * position for the player to move is the best over moves of the move's reward plus the next position's value
 * (negated when the opponent moves next), searched with alpha-beta pruning, a transposition table with bound flags,
 * move ordering (the table's best move first, then `order`) and an optional depth limit with an evaluation function.
 * It searches an environment (`AecEnv`: rewards on every move, the transposition table above); `alphaBeta` of
 * `search` searches a `Game` with utilities at the terminal states and traces the pruning, and `alphaBetaAlgorithm`
 * steps it. Their node counts differ because the environment's search also keeps a transposition table.
 *
 * @param env - The game (deterministic steps).
 * @param state - The position.
 * @param options - Depth, evaluation, move ordering and the transposition key.
 * @returns The best move, its value and the number of positions searched.
 * @throws MdmlError with code `value` when the game is over.
 * @concept game-tree-search
 * @example
 * ```ts
 * import { alphaBeta } from 'mdml-engine/agents'
 * import { ticTacToe } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = ticTacToe.create()
 * alphaBeta(env, env.reset(key(0))).value // => 0
 * ```
 */
export function alphaBeta<S, O>(
  env: AecEnv<S, O, number>,
  state: S,
  options: AlphaBetaOptions<S> = {},
): AlphaBetaResult {
  const maxDepth = options.depth ?? Infinity
  const evaluate = options.evaluate ?? (() => 0)
  const keyOf = options.stateKey === null ? undefined : (options.stateKey ?? ((s: S) => JSON.stringify(s)))
  const table = new Map<string, TableEntry>()
  const k0 = makeKey(0)
  let nodes = 0
  const search = (s: S, depth: number, alpha0: number, beta0: number): { value: number; best: number | undefined } => {
    nodes += 1
    const mover = env.currentAgent(s)
    if (mover === undefined) return { value: 0, best: undefined }
    if (depth <= 0) return { value: evaluate(s, mover), best: undefined }
    let alpha = alpha0
    let beta = beta0
    const id = keyOf?.(s)
    const entry = id === undefined ? undefined : table.get(id)
    if (entry !== undefined && entry.depth >= depth) {
      if (entry.flag === 'exact') return { value: entry.value, best: entry.best }
      if (entry.flag === 'lower') alpha = Math.max(alpha, entry.value)
      else beta = Math.min(beta, entry.value)
      if (alpha >= beta) return { value: entry.value, best: entry.best }
    }
    const legal = env.legalActions?.(s, mover) ?? []
    let moves = options.order === undefined ? [...legal] : [...options.order(s, legal)]
    if (entry?.best !== undefined && moves.includes(entry.best))
      moves = [entry.best, ...moves.filter((m) => m !== entry.best)]
    let value = -Infinity
    let best: number | undefined
    for (const a of moves) {
      const r = env.step(s, a, k0)
      const immediate = r.rewards[mover] ?? 0
      const next = env.currentAgent(r.state)
      let v: number
      if (next === undefined) v = immediate
      else if (next === mover) v = immediate + search(r.state, depth - 1, alpha - immediate, beta - immediate).value
      else v = immediate - search(r.state, depth - 1, immediate - beta, immediate - alpha).value
      if (v > value) {
        value = v
        best = a
      }
      alpha = Math.max(alpha, v)
      if (alpha >= beta) break
    }
    if (id !== undefined) {
      const flag = value <= alpha0 ? 'upper' : value >= beta ? 'lower' : 'exact'
      table.set(id, { depth, value, flag, best })
    }
    return { value, best }
  }
  if (env.currentAgent(state) === undefined) throw new MdmlError('value', 'alphaBeta: the game is over')
  const root = search(state, maxDepth, -Infinity, Infinity)
  return { action: root.best ?? 0, value: root.value, nodes }
}

/** Options of {@link minimaxAgent}. */
export interface MinimaxAgentOptions<S, O> extends AlphaBetaOptions<S> {
  /** The game state behind an observation (default the observation itself: use `stateObservations(env)`). */
  readonly toState?: (observation: O) => S
}

/**
 * A minimax agent: every move is the best move of an {@link alphaBeta} search (exact with unlimited depth, or with a
 * depth limit and an evaluation function). Deterministic and stateless.
 *
 * @param env - The game.
 * @param options - Search options and `toState`.
 * @returns The agent.
 * @concept game-tree-search
 * @example
 * ```ts
 * import { minimaxAgent } from 'mdml-engine/agents'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = nim.create({ heaps: [1, 2] })
 * minimaxAgent(env).greedy(null, env.reset(key(0))) // => 2
 * ```
 */
export function minimaxAgent<S, O>(
  env: AecEnv<S, O, number>,
  options: MinimaxAgentOptions<S, O> = {},
): Agent<null, O | S, number> {
  const { toState, ...search } = options
  const state = (o: O | S): S => (toState === undefined ? (o as S) : toState(o as O))
  const move = (o: O | S): number => alphaBeta(env, state(o), search).action
  return {
    name: 'minimax',
    config: { env: env.spec.id, depth: options.depth ?? null } as Json,
    init: () => null,
    act: (_, o) => move(o),
    greedy: (_, o) => move(o),
    learn: (s) => s,
  }
}

// --------------------------------------------------------------------------------------------------- AlphaZero

/** A policy and value network for {@link alphaZero}: logits over every action and a value in [-1, 1]. */
export interface PolicyValueNetwork {
  readonly params: Tree
  readonly inputSize: number
  readonly nActions: number
  /** Logits `[B, nActions]` and values `[B, 1]` (tanh) for a batch of encoded positions `[B, inputSize]`. */
  apply(params: Tree, x: Value): { readonly logits: Value; readonly value: Value }
}

/**
 * A small policy and value network (an MLP trunk with ReLUs and a linear layer giving the action logits and the
 * value, squashed by tanh).
 *
 * @param inputSize - The size of an encoded position.
 * @param nActions - The number of actions.
 * @param hidden - Hidden widths.
 * @param key - The initialisation key.
 * @returns The network.
 * @concept alphazero
 * @example
 * ```ts
 * import { policyValueNetwork } from 'mdml-engine/agents'
 * import { key } from 'mdml-engine/random'
 * import { tensor } from 'mdml-engine/tensor'
 * const net = policyValueNetwork(4, 3, [8], key(0))
 * net.apply(net.params, tensor([[0, 1, 0, 1]])).logits.shape // => [1, 3]
 * ```
 */
export function policyValueNetwork(
  inputSize: number,
  nActions: number,
  hidden: readonly number[],
  key: Key,
): PolicyValueNetwork {
  // A linear policy head and a linear value head on one trunk are one linear layer with nActions + 1 outputs.
  const net = mlp([inputSize, ...hidden, nActions + 1], { key })
  return {
    params: net.params,
    inputSize,
    nActions,
    apply: (p, x) => {
      const out = net.apply(p as typeof net.params, x)
      return { logits: sliceAxis(out, -1, 0, nActions), value: tanh(sliceAxis(out, -1, nActions, nActions + 1)) }
    },
  }
}

/** One self-play training example: an encoded position, the search's visit distribution and the game's outcome. */
export interface AlphaZeroSample {
  readonly input: readonly number[]
  readonly policy: readonly number[]
  readonly outcome: number
}

/** Options of {@link alphaZeroAlgorithm} and {@link alphaZero}. */
export interface AlphaZeroOptions<S> {
  /** Hidden widths of the network (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  /** Search simulations per move (default 50). */
  readonly simulations?: number
  /** The PUCT exploration constant (default 1.5). */
  readonly c?: number
  /** Dirichlet root-noise concentration (default 0.3) and mixing fraction (default 0.25). */
  readonly dirichletAlpha?: number
  readonly noiseFraction?: number
  /** Moves are sampled from visits^(1/temperature) for the first `temperatureMoves` moves (default 1 and 8). */
  readonly temperature?: number
  readonly temperatureMoves?: number
  /** Self-play games per iteration (default 8). */
  readonly gamesPerIteration?: number
  /** Minibatch updates per iteration (default 16), batch size (default 64) and replay size (default 4096). */
  readonly trainSteps?: number
  readonly batchSize?: number
  readonly bufferSize?: number
  readonly learningRate?: number
  /** Decoupled weight decay (the L2 term of the AlphaZero loss; default 1e-4). */
  readonly weightDecay?: number
  /** The weight of the value loss (default 1). */
  readonly valueWeight?: number
  /** The network input of a position for the player to move (default `env.observe(state, player)`). */
  readonly encode?: (state: S, player: string) => readonly number[]
  /** The initialisation key (default key(0)). */
  readonly key?: Key
}

/** The state of {@link alphaZeroAlgorithm}. */
export interface AlphaZeroState {
  readonly params: Tree
  readonly optState: unknown
  readonly buffer: ReplayBuffer<AlphaZeroSample>
  readonly iteration: number
  readonly games: number
  readonly policyLoss: number
  readonly valueLoss: number
  /** Mean outcome for `player_0` over the last iteration's self-play games. */
  readonly firstPlayerOutcome: number
}

interface Resolved<S> {
  readonly net: PolicyValueNetwork
  readonly encode: (state: S, player: string) => readonly number[]
  readonly simulations: number
  readonly c: number
}

function resolve<S, O>(env: AecEnv<S, O, number>, options: AlphaZeroOptions<S>): Resolved<S> {
  const first = env.possibleAgents[0] ?? 'player_0'
  const nA = discreteCount(env.actionSpaces[first], 'alphaZero')
  const encode =
    options.encode ??
    ((s: S, p: string): readonly number[] => {
      const o = env.observe(s, p)
      if (!Array.isArray(o))
        throw new MdmlError('value', 'alphaZero: observations must be number arrays (or pass encode)')
      return o as readonly number[]
    })
  const space = env.observationSpaces[first]
  const d = space?.kind === 'box' ? space.shape.reduce((a, b) => a * b, 1) : encode(env.reset(makeKey(0)), first).length
  return {
    net: policyValueNetwork(d, nA, options.hidden ?? [64, 64], options.key ?? makeKey(0)),
    encode,
    simulations: options.simulations ?? 50,
    c: options.c ?? 1.5,
  }
}

/** The network's prior over the legal actions and value of a position, cached per position during one search. */
function evaluator<S, O>(
  env: AecEnv<S, O, number>,
  r: Resolved<S>,
  params: Tree,
  noise?: { readonly root: S; readonly key: Key; readonly alpha: number; readonly fraction: number },
): {
  prior: (state: S, actions: readonly number[]) => readonly number[]
  value: (state: S, player: string) => number
} {
  const cache = new Map<unknown, { logits: number[]; value: number }>()
  const run = (s: S, player: string): { logits: number[]; value: number } => {
    const hit = cache.get(s)
    if (hit !== undefined) return hit
    const out = r.net.apply(params, rowsTensor([r.encode(s, player)], r.net.inputSize))
    const e = { logits: toFlat(out.logits), value: toFlat(out.value)[0] ?? 0 }
    cache.set(s, e)
    return e
  }
  return {
    prior: (s, actions) => {
      const logits = run(s, env.currentAgent(s) ?? 'player_0').logits
      const legal = toFlat(
        softmax(fromStorage(Float64Array.from(actions.map((a) => logits[a] ?? 0)), [actions.length], 'float64'), -1),
      )
      if (noise === undefined || s !== noise.root) return legal
      const eta = toFlat(
        dirichlet(
          noise.key,
          actions.map(() => noise.alpha),
          1,
        ),
      )
      return legal.map((p, i) => (1 - noise.fraction) * p + noise.fraction * (eta[i] ?? 0))
    },
    value: (s, player) => run(s, player).value,
  }
}

/** Play one self-play game with the network-guided search; returns the training samples and the outcomes. */
function selfPlayGame<S, O>(
  env: AecEnv<S, O, number>,
  r: Resolved<S>,
  params: Tree,
  options: AlphaZeroOptions<S>,
  k: Key,
): { readonly samples: AlphaZeroSample[]; readonly returns: Readonly<Record<string, number>> } {
  const nA = r.net.nActions
  const temperature = options.temperature ?? 1
  const tMoves = options.temperatureMoves ?? 8
  let s = env.reset(foldIn(k, 0))
  const records: { input: readonly number[]; policy: number[]; player: string }[] = []
  const returns: Record<string, number> = Object.fromEntries(env.possibleAgents.map((a) => [a, 0]))
  for (let t = 0; t < 10_000; t++) {
    const player = env.currentAgent(s)
    if (player === undefined) break
    const ev = evaluator(env, r, params, {
      root: s,
      key: foldIn(k, 3 * t + 1),
      alpha: options.dirichletAlpha ?? 0.3,
      fraction: options.noiseFraction ?? 0.25,
    })
    const search = gameMcts(env, s, { simulations: r.simulations, c: r.c, ...ev, key: foldIn(k, 3 * t + 2) })
    const policy = new Array<number>(nA).fill(0)
    let total = 0
    for (const v of search.visits) total += v
    for (const [i, a] of search.actions.entries()) policy[a] = (search.visits[i] ?? 0) / Math.max(1, total)
    records.push({ input: r.encode(s, player), policy, player })
    let action: number
    if (t < tMoves && temperature > 0) {
      const w = search.visits.map((v) => v ** (1 / temperature))
      action = search.actions[categorical(foldIn(k, 3 * t + 3), w, []).data[0] ?? 0] ?? search.action
    } else action = search.action
    const step = env.step(s, action, foldIn(k, 3 * t + 4))
    for (const [a, x] of Object.entries(step.rewards)) returns[a] = (returns[a] ?? 0) + x
    s = step.state
  }
  return {
    samples: records.map((rec) => ({
      input: rec.input,
      policy: rec.policy,
      outcome: Math.max(-1, Math.min(1, returns[rec.player] ?? 0)),
    })),
    returns,
  }
}

/**
 * AlphaZero-style self-play (Silver et al. 2018) as an {@link Algorithm} over iterations: each iteration plays
 * `gamesPerIteration` games of the network against itself, choosing moves by PUCT search guided by the network's
 * prior and value with Dirichlet noise at the root, sampled from the visit counts with a temperature for the opening
 * moves; every position yields a target (the search's visit distribution and the final outcome for the player to
 * move), stored in a replay buffer; then `trainSteps` minibatch updates minimise cross-entropy to the visit
 * distribution plus the squared value error (AdamW supplies the weight decay). Records the losses and the first
 * player's mean outcome.
 *
 * @param env - A two-player zero-sum game with legal actions and number-array observations (or `encode`).
 * @param options - Network, search, self-play and training options.
 * @returns The algorithm.
 * @concept alphazero
 * @example
 * ```ts
 * import { alphaZeroAlgorithm } from 'mdml-engine/agents'
 * import { nim } from 'mdml-engine/environments'
 * alphaZeroAlgorithm(nim.create({ heaps: [2, 2] }), { hidden: [8] }).name // => 'alphaZero'
 * ```
 */
export function alphaZeroAlgorithm<S, O>(
  env: AecEnv<S, O, number>,
  options: AlphaZeroOptions<S> = {},
): Algorithm<AlphaZeroState> {
  const r = resolve(env, options)
  const opt = adamw({ learningRate: options.learningRate ?? 1e-3, weightDecay: options.weightDecay ?? 1e-4 })
  const games = options.gamesPerIteration ?? 8
  const trainSteps = options.trainSteps ?? 16
  const batchSize = options.batchSize ?? 64
  const valueWeight = options.valueWeight ?? 1
  const nA = r.net.nActions
  return {
    name: 'alphaZero',
    config: {
      env: env.spec.id,
      hidden: [...(options.hidden ?? [64, 64])],
      simulations: r.simulations,
      c: r.c,
      gamesPerIteration: games,
      trainSteps,
      batchSize,
    } as Json,
    schema: channels({
      metrics: {
        policyLoss: { description: 'cross-entropy to the search visit distributions', direction: 'lower' },
        valueLoss: { description: 'squared error of the value head against game outcomes', direction: 'lower' },
        firstPlayerOutcome: { description: 'mean self-play outcome of player_0' },
      },
    }),
    init: () => ({
      params: r.net.params,
      optState: opt.init(r.net.params),
      buffer: replayBuffer<AlphaZeroSample>(options.bufferSize ?? 4096),
      iteration: 0,
      games: 0,
      policyLoss: 0,
      valueLoss: 0,
      firstPlayerOutcome: 0,
    }),
    step: (s, ctx) => {
      let buffer = s.buffer
      let first = 0
      for (let g = 0; g < games; g++) {
        const played = selfPlayGame(env, r, s.params, options, foldIn(foldIn(ctx.key, 0), g))
        for (const sample of played.samples) buffer = bufferAdd(buffer, sample)
        first += (played.returns[env.possibleAgents[0] ?? 'player_0'] ?? 0) / games
      }
      let params = s.params
      let optState = s.optState
      let policyLoss = 0
      let valueLoss = 0
      for (let i = 0; i < trainSteps; i++) {
        const batch = bufferSample(buffer, foldIn(foldIn(ctx.key, 1), i), Math.min(batchSize, buffer.size)).items
        const x = rowsTensor(
          batch.map((b) => b.input),
          r.net.inputSize,
        )
        const pi = rowsTensor(
          batch.map((b) => b.policy),
          nA,
        )
        const z = fromStorage(Float64Array.from(batch.map((b) => b.outcome)), [batch.length, 1], 'float64')
        const step = gradientStep(
          (p) => {
            const out = r.net.apply(p, x)
            return add(crossEntropy(out.logits, pi), mul(mse(out.value, z), valueWeight))
          },
          params,
          opt,
          optState,
        )
        params = step.params
        optState = step.optState
        // The two terms of the loss on this batch, after the update.
        const out = r.net.apply(params, x)
        policyLoss = toFlat(crossEntropy(out.logits, pi))[0] ?? 0
        valueLoss = toFlat(mse(out.value, z))[0] ?? 0
      }
      return {
        params,
        optState,
        buffer,
        iteration: s.iteration + 1,
        games: s.games + games,
        policyLoss,
        valueLoss,
        firstPlayerOutcome: first,
      }
    },
    observe: (s) => ({
      metrics: { policyLoss: s.policyLoss, valueLoss: s.valueLoss, firstPlayerOutcome: s.firstPlayerOutcome },
    }),
  }
}

/** The result of {@link alphaZero}. */
export interface AlphaZeroResult {
  readonly params: Tree
  readonly network: PolicyValueNetwork
  readonly state: AlphaZeroState
  readonly runId: string
}

/**
 * Train a policy and value network by AlphaZero-style self-play (see {@link alphaZeroAlgorithm}) for a number of
 * iterations; play with it through {@link alphaZeroAgent}.
 *
 * @param env - The game.
 * @param options - Algorithm options, `iterations` (default 10) and run options (key, seed, tracer).
 * @returns The trained parameters, the network and the final state.
 * @concept alphazero
 * @example
 * ```ts
 * import { alphaZero } from 'mdml-engine/agents'
 * import { nim } from 'mdml-engine/environments'
 * const r = alphaZero(nim.create({ heaps: [1, 2] }), { hidden: [8], iterations: 1, gamesPerIteration: 1, simulations: 4, trainSteps: 1 })
 * r.state.games // => 1
 * ```
 */
export function alphaZero<S, O>(
  env: AecEnv<S, O, number>,
  options: AlphaZeroOptions<S> & { readonly iterations?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): AlphaZeroResult {
  const { iterations, seed, key, tracer, policy, runId, parent, start, ...algorithmOptions } = options
  const run: Partial<RunOptions> = {
    ...(seed === undefined ? {} : { seed }),
    ...(key === undefined ? {} : { key }),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(runId === undefined ? {} : { runId }),
    ...(parent === undefined ? {} : { parent }),
    ...(start === undefined ? {} : { start }),
  }
  const result = runAlgorithm(alphaZeroAlgorithm(env, algorithmOptions), { ...run, maxSteps: iterations ?? 10 })
  return {
    params: result.state.params,
    network: resolve(env, algorithmOptions).net,
    state: result.state,
    runId: result.runId,
  }
}

/**
 * A game-playing agent from trained AlphaZero parameters: PUCT search guided by the network (no root noise), playing
 * the most visited move. With `simulations: 0` it plays the network's most probable legal move directly.
 *
 * @param env - The game (as trained).
 * @param params - The trained parameters (from {@link alphaZero}).
 * @param options - The training options (network shape, encoding) plus `simulations`, `c` and `toState`.
 * @returns The agent (its state is the parameters).
 * @concept alphazero
 * @example
 * ```ts
 * import { alphaZero, alphaZeroAgent } from 'mdml-engine/agents'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = nim.create({ heaps: [2] })
 * const opts = { hidden: [8], iterations: 1, gamesPerIteration: 1, simulations: 4, trainSteps: 1 }
 * const agent = alphaZeroAgent(env, alphaZero(env, opts).params, { ...opts, simulations: 50 })
 * agent.greedy(agent.init(key(0)), env.reset(key(0))) // => 1
 * ```
 */
export function alphaZeroAgent<S, O>(
  env: AecEnv<S, O, number>,
  params: Tree,
  options: AlphaZeroOptions<S> & { readonly toState?: (observation: O) => S } = {},
): Agent<Tree, O | S, number> {
  const r = resolve(env, options)
  const toState = (o: O | S): S => (options.toState === undefined ? (o as S) : options.toState(o as O))
  const choose = (p: Tree, o: O | S, k: Key): number => {
    const s = toState(o)
    const player = env.currentAgent(s)
    if (player === undefined) throw new MdmlError('value', 'alphaZeroAgent: the game is over')
    const ev = evaluator(env, r, p)
    if (r.simulations === 0) {
      const legal = env.legalActions?.(s, player) ?? []
      return legal[argmaxOf(ev.prior(s, legal))] ?? 0
    }
    return gameMcts(env, s, { simulations: r.simulations, c: r.c, ...ev, key: k }).action
  }
  return {
    name: 'alphaZero',
    config: { env: env.spec.id, simulations: r.simulations } as Json,
    init: () => params,
    act: (p, o, k) => choose(p, o, k),
    greedy: (p, o) => choose(p, o, GREEDY_KEY),
    learn: (p) => p,
  }
}

/**
 * A policy for `playAec` that plays uniformly at random among the legal moves of a game whose observations are
 * states (see `stateObservations`): the usual baseline opponent.
 *
 * @param env - The game.
 * @param agent - The agent the policy plays for.
 * @returns The policy.
 * @concept agents
 * @example
 * ```ts
 * import { randomLegalPolicy } from 'mdml-engine/agents'
 * import { nim } from 'mdml-engine/environments'
 * import { key } from 'mdml-engine/random'
 * const env = nim.create({ heaps: [1] })
 * randomLegalPolicy(env, 'player_0')(env.reset(key(0)), key(1)) // => 0
 * ```
 */
export function randomLegalPolicy<S, O>(env: AecEnv<S, O, number>, agent: string): Policy<S, number> {
  return (s, k) => {
    const legal = env.legalActions?.(s, agent) ?? []
    if (legal.length === 0) throw new MdmlError('value', `randomLegalPolicy: ${agent} has no legal move`)
    return (
      legal[
        categorical(
          k,
          legal.map(() => 1),
          [],
        ).data[0] ?? 0
      ] ?? 0
    )
  }
}
