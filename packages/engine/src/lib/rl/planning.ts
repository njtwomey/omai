/**
 * Planning with an environment used as its own model: Monte Carlo tree search (UCT, or PUCT with priors and a value
 * function as in AlphaZero) as a traced algorithm, its version for turn-based two-player games, Monte Carlo rollout
 * planning, and the action-sequence planners used in model-based control, the cross-entropy method and model predictive
 * path integral control (MPPI).
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../random/index.ts'
import type { Space } from '../space/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import type { Policy } from './agent.ts'
import type { Env } from './env.ts'
import type { AecEnv } from './multi-agent.ts'

const u1 = (k: Key): number => uniform(k, []).data[0] ?? 0

/** The actions of a discrete action space, `start .. start + n - 1`. */
function discreteActions(space: Space, name: string): number[] {
  if (space.kind !== 'discrete')
    throw new MdmlError('value', `${name}: needs a discrete action space or explicit actions`)
  return Array.from({ length: space.n }, (_, i) => (space.start ?? 0) + i)
}

// ------------------------------------------------------------------------------------------------------ single agent

/** A node of the single-agent search tree (immutable: a simulation copies the nodes on its path). */
interface Node<S> {
  readonly state: S
  readonly terminal: boolean
  readonly visits: number
  /** Per action: visits, total return, and children keyed by the sampled next state. */
  readonly n: readonly number[]
  readonly w: readonly number[]
  readonly children: readonly ReadonlyMap<string, Node<S>>[]
}

/** Options of {@link mctsAlgorithm} and {@link mcts}. */
export interface MctsOptions<S, O, A> {
  /** The actions to consider (default the discrete action space). */
  readonly actions?: readonly A[]
  /** The exploration constant of UCT (default sqrt 2) or PUCT. */
  readonly c?: number
  readonly gamma?: number
  /** The search depth (default 50). */
  readonly depth?: number
  /** The rollout policy at leaves (default uniformly random); ignored when `value` is given. */
  readonly rolloutPolicy?: Policy<O, A>
  /** A leaf evaluator replacing rollouts (AlphaZero-style). */
  readonly value?: (state: S) => number
  /** Prior action probabilities for PUCT (default none: UCT). */
  readonly prior?: (state: S) => readonly number[]
  /** A key identifying a state, so stochastic outcomes branch (default JSON). */
  readonly stateKey?: (state: S) => string
}

/** The state of an MCTS run. */
export interface MctsState<S> {
  readonly root: Node<S>
  readonly simulations: number
}

/** The result of a search. */
export interface MctsResult<A> {
  /** The most visited action at the root. */
  readonly action: A
  readonly actions: readonly A[]
  readonly visits: readonly number[]
  /** Mean returns of the root actions. */
  readonly values: readonly number[]
  readonly simulations: number
}

/**
 * Monte Carlo tree search over an environment used as its own model (UCT, Kocsis and Szepesvari 2006; PUCT with
 * priors and a value function as in AlphaZero), as an {@link Algorithm}: each step runs one simulation from the root
 * (selection by upper confidence, expansion, a rollout or value estimate, backup of the discounted return). Stochastic
 * transitions branch on the sampled next state. Records the root's visit counts and value estimates.
 *
 * @param env - The environment (pure, so steps can be simulated from any state).
 * @param state - The root state.
 * @param options - Actions, exploration constant, discount, depth, rollout policy or value, priors.
 * @returns The algorithm.
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { mctsAlgorithm } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * mctsAlgorithm(cartPole.create(), [0, 0, 0, 0]).name // => 'mcts'
 * ```
 */
export function mctsAlgorithm<S, O, A>(
  env: Env<S, O, A>,
  state: S,
  options: MctsOptions<S, O, A> = {},
): Algorithm<MctsState<S>> {
  const actions = options.actions ?? (discreteActions(env.actionSpace, 'mcts') as unknown as A[])
  const c = options.c ?? Math.SQRT2
  const gamma = options.gamma ?? 1
  const depth = options.depth ?? 50
  const stateKey = options.stateKey ?? ((s: S) => JSON.stringify(s))
  const observe = env.observe
  const node = (s: S, terminal: boolean): Node<S> => ({
    state: s,
    terminal,
    visits: 0,
    n: actions.map(() => 0),
    w: actions.map(() => 0),
    children: actions.map(() => new Map<string, Node<S>>()),
  })
  const rollout = (s: S, k: Key, d: number): number => {
    if (options.value !== undefined) return options.value(s)
    let total = 0
    let discount = 1
    let cur = s
    for (let t = 0; t < d; t++) {
      const a =
        options.rolloutPolicy !== undefined && observe !== undefined
          ? options.rolloutPolicy(observe(cur), foldIn(k, 2 * t))
          : (actions[Math.min(actions.length - 1, Math.floor(u1(foldIn(k, 2 * t)) * actions.length))] as A)
      const r = env.step(cur, a, foldIn(k, 2 * t + 1))
      total += discount * r.reward
      discount *= gamma
      cur = r.state
      if (r.terminated || r.truncated) break
    }
    return total
  }
  const select = (nd: Node<S>): number => {
    const prior = options.prior?.(nd.state)
    const scores = actions.map((_, i) => {
      const n = nd.n[i] ?? 0
      const q = n > 0 ? (nd.w[i] ?? 0) / n : 0
      if (prior !== undefined) return q + c * (prior[i] ?? 0) * (Math.sqrt(nd.visits) / (1 + n))
      return n === 0 ? Infinity : q + c * Math.sqrt(Math.log(Math.max(nd.visits, 1)) / n)
    })
    return argmaxOf(scores)
  }
  // One simulation: returns the sampled return and the updated copy of the node.
  const simulate = (nd: Node<S>, k: Key, d: number): { readonly g: number; readonly node: Node<S> } => {
    if (nd.terminal || d === 0) return { g: 0, node: nd }
    const i = select(nd)
    const unvisited = (nd.n[i] ?? 0) === 0
    const r = env.step(nd.state, actions[i] as A, foldIn(k, 0))
    const sk = stateKey(r.state)
    const existing = nd.children[i]?.get(sk)
    const end = r.terminated || r.truncated
    let child = existing ?? node(r.state, r.terminated)
    let future = 0
    if (!end) {
      if (unvisited || existing === undefined) future = rollout(r.state, foldIn(k, 1), d - 1)
      else {
        const sub = simulate(existing, foldIn(k, 2), d - 1)
        future = sub.g
        child = sub.node
      }
    }
    const g = r.reward + gamma * future
    const children = nd.children.map((m, j) => (j === i ? new Map(m).set(sk, child) : m))
    return {
      g,
      node: {
        ...nd,
        visits: nd.visits + 1,
        n: nd.n.map((x, j) => (j === i ? x + 1 : x)),
        w: nd.w.map((x, j) => (j === i ? x + g : x)),
        children,
      },
    }
  }
  return {
    name: 'mcts',
    config: { env: env.spec.id, actions: actions.length, c, gamma, depth } as Json,
    schema: channels({
      metrics: { rootValue: { description: 'value estimate of the most visited root action', direction: 'higher' } },
      tensors: {
        visits: { role: 'state', description: 'visit counts of the root actions' },
        values: { role: 'state', description: 'mean returns of the root actions' },
      },
    }),
    init: () => ({ root: node(state, false), simulations: 0 }),
    step: (s, ctx) => ({ root: simulate(s.root, ctx.key, depth).node, simulations: s.simulations + 1 }),
    observe: (s) => {
      const values = s.root.n.map((n, i) => (n > 0 ? (s.root.w[i] ?? 0) / n : 0))
      return {
        metrics: { rootValue: values[argmaxOf(s.root.n)] ?? 0 },
        tensors: { visits: { role: 'state', value: [...s.root.n] }, values: { role: 'state', value: values } },
      }
    },
  }
}

function summarise<S, A>(root: Node<S>, actions: readonly A[], simulations: number): MctsResult<A> {
  const best = argmaxOf(root.n)
  return {
    action: actions[best] as A,
    actions,
    visits: [...root.n],
    values: root.n.map((n, i) => (n > 0 ? (root.w[i] ?? 0) / n : 0)),
    simulations,
  }
}

/**
 * Plan one action by Monte Carlo tree search (see {@link mctsAlgorithm}).
 *
 * @param env - The environment.
 * @param state - The current state.
 * @param options - Search options, the number of simulations (default 200) and the run options (key, tracer).
 * @returns The chosen action with the root statistics.
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { mcts } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * mcts(cartPole.create(), [0, 0, 0.12, 0], { simulations: 300, depth: 25, c: 5 }).action // => 1
 * ```
 */
export function mcts<S, O, A>(
  env: Env<S, O, A>,
  state: S,
  options: MctsOptions<S, O, A> & { readonly simulations?: number } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): MctsResult<A> {
  const { simulations, actions, c, gamma, depth, rolloutPolicy, value, prior, stateKey, ...run } = options
  const opts: MctsOptions<S, O, A> = {
    ...(actions === undefined ? {} : { actions }),
    ...(c === undefined ? {} : { c }),
    ...(gamma === undefined ? {} : { gamma }),
    ...(depth === undefined ? {} : { depth }),
    ...(rolloutPolicy === undefined ? {} : { rolloutPolicy }),
    ...(value === undefined ? {} : { value }),
    ...(prior === undefined ? {} : { prior }),
    ...(stateKey === undefined ? {} : { stateKey }),
  }
  const result = runAlgorithm(mctsAlgorithm(env, state, opts), { ...run, maxSteps: simulations ?? 200 })
  const acts = actions ?? (discreteActions(env.actionSpace, 'mcts') as unknown as A[])
  return summarise(result.state.root, acts, result.state.simulations)
}

// ----------------------------------------------------------------------------------------------------- two players

/** A node of a two-player game tree (values from the perspective of the player to move at the parent). */
interface GameNode<S> {
  readonly state: S
  readonly player: string | undefined
  readonly actions: readonly number[]
  visits: number
  readonly n: number[]
  readonly w: number[]
  readonly children: (GameNode<S> | undefined)[]
  readonly prior: readonly number[] | undefined
}

/** Options of {@link gameMcts}. */
export interface GameMctsOptions<S> {
  /** Simulations (default 200). */
  readonly simulations?: number
  readonly c?: number
  /** Prior move probabilities over the legal actions (PUCT), in their order. */
  readonly prior?: (state: S, actions: readonly number[]) => readonly number[]
  /** A value of a state for the player to move there, in [-1, 1] (replaces random playouts). */
  readonly value?: (state: S, player: string) => number
  /** Dirichlet-style root noise is not applied; pass a key for the random playouts and tie breaks. */
  readonly key?: Key
  /** Cap on the length of a random playout (default 1000 turns). */
  readonly maxTurns?: number
}

/**
 * Monte Carlo tree search for turn-based two-player zero-sum games (an {@link AecEnv} with legal actions): UCT with
 * random playouts, or PUCT with a prior and a value function (AlphaZero), backing values up with alternating signs
 * (negamax). The value at a node is from the perspective of the player who moved into it.
 *
 * @param env - The game.
 * @param state - The position.
 * @param options - Simulations, exploration constant, prior, value function and key.
 * @returns The most visited move with the root statistics (actions are the legal moves).
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { gameMcts } from 'mdml-engine/rl'
 * import { nim } from 'mdml-engine/methods/rl'
 * import { key } from 'mdml-engine/random'
 * const env = nim.create({ heaps: [2] })
 * gameMcts(env, env.reset(key(0)), { simulations: 100 }).action // => 1
 * ```
 */
export function gameMcts<S, O>(
  env: AecEnv<S, O, number>,
  state: S,
  options: GameMctsOptions<S> = {},
): MctsResult<number> {
  const c = options.c ?? (options.prior === undefined ? Math.SQRT2 : 1.5)
  const k0 = options.key ?? makeKey(0)
  const legal = (s: S, p: string | undefined): readonly number[] => {
    if (p === undefined) return []
    return env.legalActions?.(s, p) ?? discreteActions(env.actionSpaces[p] as Space, 'gameMcts')
  }
  const make = (s: S): GameNode<S> => {
    const player = env.currentAgent(s)
    const actions = legal(s, player)
    return {
      state: s,
      player,
      actions,
      visits: 0,
      n: actions.map(() => 0),
      w: actions.map(() => 0),
      children: actions.map(() => undefined),
      prior: options.prior === undefined || player === undefined ? undefined : options.prior(s, actions),
    }
  }
  // The reward to `player` from playing out the game at random (or the value function).
  const evaluate = (s: S, player: string, k: Key): number => {
    const mover = env.currentAgent(s)
    if (mover === undefined) return 0
    if (options.value !== undefined) {
      const v = options.value(s, mover)
      return mover === player ? v : -v
    }
    let cur = s
    let total = 0
    for (let t = 0; t < (options.maxTurns ?? 1000); t++) {
      const p = env.currentAgent(cur)
      if (p === undefined) break
      const acts = legal(cur, p)
      const a = acts[Math.min(acts.length - 1, Math.floor(u1(foldIn(k, 2 * t)) * acts.length))] as number
      const r = env.step(cur, a, foldIn(k, 2 * t + 1))
      total += r.rewards[player] ?? 0
      cur = r.state
    }
    return total
  }
  // Returns the value of the node's state for the player who moves at the node.
  const simulate = (nd: GameNode<S>, k: Key): number => {
    const player = nd.player
    if (player === undefined) return 0
    const scores = nd.actions.map((_, i) => {
      const n = nd.n[i] ?? 0
      const q = n > 0 ? (nd.w[i] ?? 0) / n : 0
      if (nd.prior !== undefined) return q + c * (nd.prior[i] ?? 0) * (Math.sqrt(nd.visits) / (1 + n))
      return n === 0 ? Infinity : q + c * Math.sqrt(Math.log(Math.max(nd.visits, 1)) / n)
    })
    const i = argmaxOf(scores)
    const r = env.step(nd.state, nd.actions[i] as number, foldIn(k, 0))
    const immediate = r.rewards[player] ?? 0
    let child = nd.children[i]
    let future: number
    if (child === undefined) {
      child = make(r.state)
      nd.children[i] = child
      future = evaluate(r.state, player, foldIn(k, 1))
    } else {
      const next = child.player
      const v = simulate(child, foldIn(k, 2))
      // Two-player zero-sum: the opponent's value is the negative of ours; the same player keeps the sign.
      future = next === undefined ? 0 : next === player ? v : -v
    }
    const g = immediate + future
    nd.n[i] = (nd.n[i] ?? 0) + 1
    nd.w[i] = (nd.w[i] ?? 0) + g
    nd.visits += 1
    return g
  }
  const root = make(state)
  if (root.player === undefined) throw new MdmlError('value', 'gameMcts: the game is over')
  const sims = options.simulations ?? 200
  for (let i = 0; i < sims; i++) simulate(root, foldIn(k0, i))
  const best = argmaxOf(root.n)
  return {
    action: root.actions[best] as number,
    actions: root.actions,
    visits: [...root.n],
    values: root.n.map((n, i) => (n > 0 ? (root.w[i] ?? 0) / n : 0)),
    simulations: sims,
  }
}

// ---------------------------------------------------------------------------------------------- rollout planning

/**
 * A Monte Carlo rollout planner (Tesauro and Galperin 1997): estimate each action by the mean discounted return of
 * `rollouts` simulated episodes that start with it and continue with a base policy, and pick the best. One step of
 * policy improvement over the base policy.
 *
 * @param env - The environment.
 * @param state - The current state.
 * @param options - Actions, base policy (default random), rollouts per action (default 16), horizon (default 100),
 *   discount and key.
 * @returns The best action and the estimates.
 * @concept planning
 * @example
 * ```ts
 * import { rolloutPlanner } from 'mdml-engine/rl'
 * import { cartPole } from 'mdml-engine/methods/rl'
 * rolloutPlanner(cartPole.create(), [0, 0, 0.1, 0], { rollouts: 8, horizon: 30 }).action // => 1
 * ```
 */
export function rolloutPlanner<S, O, A>(
  env: Env<S, O, A>,
  state: S,
  options: {
    readonly actions?: readonly A[]
    readonly policy?: Policy<O, A>
    readonly rollouts?: number
    readonly horizon?: number
    readonly gamma?: number
    readonly key?: Key
  } = {},
): { readonly action: A; readonly values: readonly number[] } {
  const actions = options.actions ?? (discreteActions(env.actionSpace, 'rolloutPlanner') as unknown as A[])
  const n = options.rollouts ?? 16
  const horizon = options.horizon ?? 100
  const gamma = options.gamma ?? 1
  const k = options.key ?? makeKey(0)
  const observe = env.observe
  const values = actions.map((first) => {
    let total = 0
    for (let j = 0; j < n; j++) {
      // Common random numbers: rollout j uses the same keys for every first action.
      const kj = foldIn(k, j)
      let cur = state
      let discount = 1
      let ret = 0
      for (let t = 0; t < horizon; t++) {
        const a =
          t === 0
            ? first
            : options.policy !== undefined && observe !== undefined
              ? options.policy(observe(cur), foldIn(kj, 2 * t))
              : (actions[Math.min(actions.length - 1, Math.floor(u1(foldIn(kj, 2 * t)) * actions.length))] as A)
        const r = env.step(cur, a, foldIn(kj, 2 * t + 1))
        ret += discount * r.reward
        discount *= gamma
        cur = r.state
        if (r.terminated || r.truncated) break
      }
      total += ret
    }
    return total / n
  })
  return { action: actions[argmaxOf(values)] as A, values }
}

// ------------------------------------------------------------------------------------------------ action sequences

function boxBounds(space: Space): { readonly low: number[]; readonly high: number[]; readonly dim: number } {
  if (space.kind !== 'box') throw new MdmlError('value', 'planner: needs a box action space')
  const dim = space.shape.reduce((a, b) => a * b, 1)
  const at = (b: number | readonly number[], i: number): number => (typeof b === 'number' ? b : (b[i] ?? 0))
  return {
    low: Array.from({ length: dim }, (_, i) => at(space.low, i)),
    high: Array.from({ length: dim }, (_, i) => at(space.high, i)),
    dim,
  }
}

function evaluateSequence<S, O>(
  env: Env<S, O, readonly number[]>,
  state: S,
  plan: readonly (readonly number[])[],
  gamma: number,
  k: Key,
): number {
  let cur = state
  let ret = 0
  let discount = 1
  for (const [t, a] of plan.entries()) {
    const r = env.step(cur, a, foldIn(k, t))
    ret += discount * r.reward
    discount *= gamma
    cur = r.state
    if (r.terminated || r.truncated) break
  }
  return ret
}

/** Options of the action-sequence planners. */
export interface SequencePlannerOptions {
  /** Planning horizon in steps (default 15). */
  readonly horizon?: number
  /** Candidate sequences per iteration (default 64). */
  readonly population?: number
  readonly gamma?: number
  /** A warm start: the previous plan (shifted by one step by the caller). */
  readonly initial?: readonly (readonly number[])[]
  /** Initial standard deviation of the sampling distribution (default a quarter of the action range). */
  readonly sigma?: number
  /**
   * Score all candidate plans at once (their returns, higher is better), replacing one simulation per plan in `env`:
   * batched rollouts of a learned or differentiable model. `env` then only supplies the action bounds.
   */
  readonly score?: (plans: readonly (readonly (readonly number[])[])[], key: Key) => readonly number[]
}

/** The state of {@link cemPlannerAlgorithm}. */
export interface CemPlannerState {
  readonly mean: readonly (readonly number[])[]
  readonly std: readonly (readonly number[])[]
  readonly bestReturn: number
  readonly eliteReturn: number
}

/**
 * The cross-entropy method over action sequences (Rubinstein 1999; the planner of PETS) as an {@link Algorithm}:
 * each iteration samples sequences from a Gaussian per step and dimension (clipped to the action box), scores them by
 * simulated return, and refits the Gaussian to the best `elites`.
 *
 * @param env - The environment (or a learned model as an environment) with a box action space.
 * @param state - The current state.
 * @param options - Horizon, population, elites (default 10%), smoothing `alpha` (default 0) and warm start.
 * @returns The algorithm; the plan is `state.mean`.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { cemPlannerAlgorithm } from 'mdml-engine/rl'
 * import { pendulum } from 'mdml-engine/methods/rl'
 * cemPlannerAlgorithm(pendulum.create(), [0, 0]).name // => 'cem'
 * ```
 */
export function cemPlannerAlgorithm<S, O>(
  env: Env<S, O, readonly number[]>,
  state: S,
  options: SequencePlannerOptions & { readonly elites?: number; readonly alpha?: number } = {},
): Algorithm<CemPlannerState> {
  const { low, high, dim } = boxBounds(env.actionSpace)
  const horizon = options.horizon ?? 15
  const population = options.population ?? 64
  const elites = options.elites ?? Math.max(2, Math.round(population / 10))
  const gamma = options.gamma ?? 1
  const alpha = options.alpha ?? 0
  const sigma0 = options.sigma
  return {
    name: 'cem',
    config: { env: env.spec.id, horizon, population, elites, gamma, alpha } as Json,
    schema: channels({
      metrics: {
        eliteReturn: { description: 'mean simulated return of the elite sequences', direction: 'higher' },
        bestReturn: { description: 'best simulated return so far', direction: 'higher' },
      },
    }),
    init: () => ({
      mean: Array.from({ length: horizon }, (_, t) =>
        Array.from({ length: dim }, (_, d) => options.initial?.[t]?.[d] ?? ((low[d] ?? -1) + (high[d] ?? 1)) / 2),
      ),
      std: Array.from({ length: horizon }, () =>
        Array.from({ length: dim }, (_, d) => sigma0 ?? ((high[d] ?? 1) - (low[d] ?? -1)) / 4),
      ),
      bestReturn: -Infinity,
      eliteReturn: -Infinity,
    }),
    step: (s, ctx) => {
      const z = normal(foldIn(ctx.key, 0), [population, horizon, dim]).data
      const candidates = Array.from({ length: population }, (_, p) =>
        Array.from({ length: horizon }, (_, t) =>
          Array.from({ length: dim }, (_, d) => {
            const v = (s.mean[t]?.[d] ?? 0) + (s.std[t]?.[d] ?? 0) * (z[(p * horizon + t) * dim + d] ?? 0)
            return Math.min(high[d] ?? 1, Math.max(low[d] ?? -1, v))
          }),
        ),
      )
      const scores =
        options.score !== undefined
          ? options.score(candidates, foldIn(ctx.key, 1))
          : candidates.map((plan) => evaluateSequence(env, state, plan, gamma, foldIn(ctx.key, 1)))
      const order = scores.map((_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0))
      const elite = order.slice(0, elites).map((i) => candidates[i] as number[][])
      const mean = s.mean.map((row, t) =>
        row.map((m, d) => {
          let acc = 0
          for (const e of elite) acc += e[t]?.[d] ?? 0
          return alpha * m + (1 - alpha) * (acc / elite.length)
        }),
      )
      const std = s.std.map((row, t) =>
        row.map((sd, d) => {
          const mu = mean[t]?.[d] ?? 0
          let acc = 0
          for (const e of elite) acc += ((e[t]?.[d] ?? 0) - mu) ** 2
          return alpha * sd + (1 - alpha) * Math.sqrt(acc / elite.length)
        }),
      )
      let eliteReturn = 0
      for (const i of order.slice(0, elites)) eliteReturn += scores[i] ?? 0
      return {
        mean,
        std,
        eliteReturn: eliteReturn / elites,
        bestReturn: Math.max(s.bestReturn, scores[order[0] ?? 0] ?? -Infinity),
      }
    },
    observe: (s) => ({ metrics: { eliteReturn: s.eliteReturn, bestReturn: s.bestReturn } }),
  }
}

/** The result of an action-sequence planner. */
export interface PlanResult {
  /** The first action of the plan. */
  readonly action: number[]
  /** The whole plan (one action per step of the horizon). */
  readonly plan: number[][]
  /** The planner's estimate of the plan's return. */
  readonly value: number
}

/**
 * Plan with the cross-entropy method (see {@link cemPlannerAlgorithm}).
 *
 * @param env - The environment or model.
 * @param state - The current state.
 * @param options - Planner options, iterations (default 5) and run options (key, tracer).
 * @returns The first action, the plan and its simulated return.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { cem } from 'mdml-engine/rl'
 * import { mountainCarContinuous } from 'mdml-engine/methods/rl'
 * cem(mountainCarContinuous.create(), [-0.5, 0], { horizon: 5, population: 16 }).plan.length // => 5
 * ```
 */
export function cem<S, O>(
  env: Env<S, O, readonly number[]>,
  state: S,
  options: SequencePlannerOptions & {
    readonly elites?: number
    readonly alpha?: number
    readonly iterations?: number
  } & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): PlanResult {
  const { iterations, horizon, population, gamma, initial, sigma, elites, alpha, score, ...run } = options
  const opts = {
    ...(horizon === undefined ? {} : { horizon }),
    ...(population === undefined ? {} : { population }),
    ...(gamma === undefined ? {} : { gamma }),
    ...(initial === undefined ? {} : { initial }),
    ...(sigma === undefined ? {} : { sigma }),
    ...(elites === undefined ? {} : { elites }),
    ...(alpha === undefined ? {} : { alpha }),
    ...(score === undefined ? {} : { score }),
  }
  const result = runAlgorithm(cemPlannerAlgorithm(env, state, opts), { ...run, maxSteps: iterations ?? 5 })
  const plan = result.state.mean.map((r) => [...r])
  return { action: [...(plan[0] ?? [])], plan, value: result.state.eliteReturn }
}

/**
 * Model predictive path integral control (Williams et al. 2017): sample `population` perturbations of the current
 * plan, weight them by `exp(return / lambda)` and move the plan to the weighted average. One call is one update; in a
 * receding-horizon loop the caller shifts the plan and calls again.
 *
 * @param env - The environment or model with a box action space.
 * @param state - The current state.
 * @param options - Horizon, population, temperature `lambda` (default 1), noise `sigma`, warm start, iterations
 *   (default 1) and key.
 * @returns The first action, the updated plan and the weighted return.
 * @concept model-predictive-control
 * @example
 * ```ts
 * import { mppi } from 'mdml-engine/rl'
 * import { pendulum } from 'mdml-engine/methods/rl'
 * mppi(pendulum.create(), [Math.PI, 0], { horizon: 4, population: 8 }).action.length // => 1
 * ```
 */
export function mppi<S, O>(
  env: Env<S, O, readonly number[]>,
  state: S,
  options: SequencePlannerOptions & {
    readonly lambda?: number
    readonly iterations?: number
    readonly key?: Key
  } = {},
): PlanResult {
  const { low, high, dim } = boxBounds(env.actionSpace)
  const horizon = options.horizon ?? 15
  const population = options.population ?? 64
  const lambda = options.lambda ?? 1
  const gamma = options.gamma ?? 1
  const k = options.key ?? makeKey(0)
  const sigma = Array.from({ length: dim }, (_, d) => options.sigma ?? ((high[d] ?? 1) - (low[d] ?? -1)) / 4)
  let plan = Array.from({ length: horizon }, (_, t) =>
    Array.from({ length: dim }, (_, d) => options.initial?.[t]?.[d] ?? ((low[d] ?? -1) + (high[d] ?? 1)) / 2),
  )
  let value = -Infinity
  for (let it = 0; it < (options.iterations ?? 1); it++) {
    const z = normal(foldIn(k, 2 * it), [population, horizon, dim]).data
    const candidates = Array.from({ length: population }, (_, p) =>
      plan.map((row, t) =>
        row.map((m, d) =>
          Math.min(high[d] ?? 1, Math.max(low[d] ?? -1, m + (sigma[d] ?? 0) * (z[(p * horizon + t) * dim + d] ?? 0))),
        ),
      ),
    )
    const scores =
      options.score !== undefined
        ? options.score(candidates, foldIn(k, 2 * it + 1))
        : candidates.map((c) => evaluateSequence(env, state, c, gamma, foldIn(k, 2 * it + 1)))
    let best = -Infinity
    for (const s of scores) best = Math.max(best, s)
    const weights = scores.map((s) => Math.exp((s - best) / lambda))
    let total = 0
    for (const w of weights) total += w
    plan = plan.map((row, t) =>
      row.map((_, d) => {
        let acc = 0
        for (const [p, c] of candidates.entries()) acc += (weights[p] ?? 0) * (c[t]?.[d] ?? 0)
        return acc / total
      }),
    )
    let v = 0
    for (const [p, s] of scores.entries()) v += ((weights[p] ?? 0) / total) * s
    value = v
  }
  return { action: [...(plan[0] ?? [])], plan, value }
}
