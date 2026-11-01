/**
 * Adversarial search for two-player and chance games, AIMA style: the game contract (and a stochastic variant with
 * chance nodes), minimax, alpha-beta pruning, negamax and expectimax with optional depth limits, and Monte Carlo tree
 * search (UCT, or PUCT with priors as in AlphaZero), each as a traced algorithm reporting the best move, its value and
 * the principal variation.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { Generator, type Key } from '../random/index.ts'
import { type Algorithm, channels, runAlgorithm, type TraceSchema } from '../trace/index.ts'
import { type SearchRunOptions, type StateKey, stateKeyOf } from './problem.ts'

/**
 * A game (AIMA's `Game`): the player to move, the legal actions, the transition model, a terminal test and the utility
 * of a terminal state for a player. Players are named by strings (`'MAX'`, `'X'`, ...).
 *
 * @concept adversarial-search
 */
export interface Game<S, A> {
  readonly initial: S
  player(state: S): string
  actions(state: S): readonly A[]
  result(state: S, action: A): S
  isTerminal(state: S): boolean
  utility(state: S, player: string): number
  key?(state: S): StateKey
  label?(state: S): string
}

/** A game with chance nodes (dice, shuffles): at a chance state the outcomes and their probabilities replace actions. */
export interface StochasticGame<S, A> extends Game<S, A> {
  isChance(state: S): boolean
  outcomes(state: S): readonly { readonly state: S; readonly probability: number }[]
}

/** Options of the game-tree searches. */
export interface GameSearchOptions<S, A> extends SearchRunOptions {
  /** The search depth in plies (default unlimited); non-terminal states at the limit are evaluated. */
  readonly depth?: number
  /** The value of a non-terminal state at the depth limit for a player (default 0). */
  readonly evaluate?: (state: S, player: string) => number
  /**
   * Move ordering: the actions in their given order (default), sorted by `evaluate` of the resulting states (best first
   * for the player to move), or a function returning the actions in the order to try.
   */
  readonly order?: 'given' | 'evaluate' | ((state: S, actions: readonly A[]) => readonly A[])
  /** Stop after visiting this many nodes (default 1000000); the result then has status `budget`. */
  readonly maxNodes?: number
}

/** The result of a game-tree search: the best root action, its value, the principal variation and the effort. */
export interface GameSearchResult<A> {
  /** The best action at the root (the first of equal ones in search order); undefined at a terminal root. */
  readonly action: A | undefined
  /** The value of the root for the player to move there. */
  readonly value: number
  /** The best line of play from the root. */
  readonly principalVariation: readonly A[]
  /** The nodes visited, the root included. */
  readonly nodes: number
  /** The cutoffs made by alpha-beta pruning. */
  readonly cutoffs: number
  readonly status: 'solved' | 'budget'
}

/** The state of a traced game-tree search, as seen by observers. */
export interface GameSearchState<A> {
  readonly status: 'running' | 'solved'
  readonly nodes: number
  readonly cutoffs: number
  readonly value: number
  readonly principalVariation: readonly A[]
  /** The node visited in the last step. */
  readonly visited: { readonly node: string; readonly depth: number; readonly kind: NodeKind } | null
  /** Values backed up in the last step, deepest first. */
  readonly backedUp: readonly { readonly node: string; readonly value: number }[]
  /** Cutoffs made in the last step. */
  readonly pruned: readonly { readonly node: string; readonly skipped: number }[]
}

type NodeKind = 'max' | 'min' | 'chance' | 'leaf'
type Mode = 'minimax' | 'alpha-beta' | 'negamax' | 'negamax-alpha-beta' | 'expectimax'

interface Frame<S, A> {
  readonly state: S
  readonly label: string
  readonly depth: number
  readonly kind: 'max' | 'min' | 'chance'
  /** Negamax: the player whose value this frame holds. */
  readonly player: string
  readonly moves: readonly A[]
  readonly chances: readonly { readonly state: S; readonly probability: number }[]
  index: number
  alpha: number
  beta: number
  value: number
  best: A | undefined
  line: A[]
  /** Negamax: whether the child being searched holds its value for the other player. */
  flip: boolean
}

interface EngineState<S, A> {
  status: 'running' | 'solved'
  nodes: number
  cutoffs: number
  value: number
  principalVariation: A[]
  action: A | undefined
  visited: GameSearchState<A>['visited']
  backedUp: { node: string; value: number }[]
  pruned: { node: string; skipped: number }[]
  readonly stack: Frame<S, A>[]
  /** A value waiting to be returned to the top frame (from a leaf or a finished child), with its line. */
  pending: { value: number; line: A[] } | null
}

const GAME_SCHEMA: TraceSchema = channels({
  metrics: {
    nodes: { description: 'nodes visited so far' },
    cutoffs: { description: 'alpha-beta cutoffs so far' },
    value: { description: 'the root value so far' },
  },
  messages: {
    visit: {
      description: 'the node visited',
      payload: { node: 'state label', depth: 'ply', kind: 'max, min, chance or leaf' },
    },
    value: { description: 'values backed up', payload: { nodes: 'labels and values' } },
    prune: { description: 'cutoffs', payload: { nodes: 'labels and the number of moves skipped' } },
  },
})

function isStochastic<S, A>(game: Game<S, A>): game is StochasticGame<S, A> {
  return typeof (game as Partial<StochasticGame<S, A>>).isChance === 'function'
}

function gameEngine<S, A>(
  name: string,
  mode: Mode,
  game: Game<S, A>,
  root: S,
  options: GameSearchOptions<S, A>,
): Algorithm<GameSearchState<A> & EngineState<S, A>> {
  const key = stateKeyOf(game)
  const label = (s: S): string => game.label?.(s) ?? String(key(s))
  const limit = options.depth ?? Number.POSITIVE_INFINITY
  const evaluate = options.evaluate ?? (() => 0)
  const maxNodes = options.maxNodes ?? 1_000_000
  const prunes = mode === 'alpha-beta' || mode === 'negamax-alpha-beta'
  const nega = mode === 'negamax' || mode === 'negamax-alpha-beta'
  if (mode === 'expectimax' && !isStochastic(game)) {
    throw new MdmlError('value', 'expectimax: the game needs isChance and outcomes')
  }
  if (!(limit >= 0)) throw new MdmlError('value', `${name}: the depth ${limit} must be non-negative`)
  const rootPlayer = game.player(root)
  const chance = (s: S): boolean => mode === 'expectimax' && isStochastic(game) && game.isChance(s)
  const ordered = (s: S, mover: string, maximise: boolean): readonly A[] => {
    const moves = game.actions(s)
    const order = options.order ?? 'given'
    if (order === 'given') return moves
    if (typeof order === 'function') return order(s, moves)
    const scored = moves.map((a, i) => {
      const next = game.result(s, a)
      const v = game.isTerminal(next) ? game.utility(next, mover) : evaluate(next, mover)
      return { a, i, v: maximise ? v : -v }
    })
    return scored.sort((x, y) => y.v - x.v || x.i - y.i).map((x) => x.a)
  }
  // Enter a node: a leaf value, or a new frame. Values are for the root player (or, in negamax, for `perspective`).
  const enter = (s: EngineState<S, A>, state: S, depth: number, alpha: number, beta: number, perspective: string) => {
    s.nodes += 1
    const terminal = game.isTerminal(state)
    if (terminal || depth >= limit) {
      const who = nega ? perspective : rootPlayer
      const value = terminal ? game.utility(state, who) : evaluate(state, who)
      s.visited = { node: label(state), depth, kind: 'leaf' }
      s.pending = { value, line: [] }
      s.backedUp.push({ node: label(state), value })
      return
    }
    const isChance = chance(state)
    const mover = isChance ? perspective : game.player(state)
    const kind: Frame<S, A>['kind'] = isChance ? 'chance' : nega || mover === rootPlayer ? 'max' : 'min'
    s.visited = { node: label(state), depth, kind }
    s.stack.push({
      state,
      label: label(state),
      depth,
      kind,
      player: nega ? (isChance ? perspective : mover) : rootPlayer,
      moves: isChance ? [] : ordered(state, nega ? mover : rootPlayer, kind === 'max'),
      chances: isChance && isStochastic(game) ? game.outcomes(state) : [],
      index: 0,
      alpha,
      beta,
      value: kind === 'max' ? Number.NEGATIVE_INFINITY : kind === 'min' ? Number.POSITIVE_INFINITY : 0,
      best: undefined,
      line: [],
      flip: false,
    })
  }
  const finish = (s: EngineState<S, A>, f: Frame<S, A>): void => {
    s.stack.pop()
    s.pending = { value: f.value, line: f.best === undefined ? f.line : [f.best, ...f.line] }
    s.backedUp.push({ node: f.label, value: f.value })
    if (s.stack.length === 0) {
      s.status = 'solved'
      s.value = f.value
      s.action = f.best
      s.principalVariation = s.pending.line
    }
  }
  return {
    name,
    config: { mode, depth: Number.isFinite(limit) ? limit : null } as Json,
    schema: GAME_SCHEMA,
    init: () => {
      const s: EngineState<S, A> = {
        status: 'running',
        nodes: 0,
        cutoffs: 0,
        value: Number.NaN,
        principalVariation: [],
        action: undefined,
        visited: null,
        backedUp: [],
        pruned: [],
        stack: [],
        pending: null,
      }
      enter(s, root, 0, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, rootPlayer)
      if (s.pending !== null && s.stack.length === 0) {
        s.status = 'solved'
        s.value = s.pending.value
      }
      return s
    },
    step: (s) => {
      s.visited = null
      s.backedUp = []
      s.pruned = []
      for (;;) {
        const f = s.stack[s.stack.length - 1]
        if (f === undefined) return s
        if (s.pending !== null) {
          // Back up a child's value into its parent frame.
          const { value: raw, line } = s.pending
          s.pending = null
          const child = f.index - 1
          if (f.kind === 'chance') {
            f.value += (f.chances[child]?.probability ?? 0) * raw
            continue
          }
          const move = f.moves[child] as A
          const v = f.flip ? 0 - raw : raw
          if (f.kind === 'max' ? v > f.value : v < f.value) {
            f.value = v
            f.best = move
            f.line = line
          } else if (f.best === undefined) {
            f.best = move
            f.line = line
          }
          if (prunes) {
            if (f.kind === 'max') {
              if (f.value >= f.beta) {
                s.cutoffs += 1
                s.pruned.push({ node: f.label, skipped: f.moves.length - f.index })
                finish(s, f)
                continue
              }
              f.alpha = Math.max(f.alpha, f.value)
            } else {
              if (f.value <= f.alpha) {
                s.cutoffs += 1
                s.pruned.push({ node: f.label, skipped: f.moves.length - f.index })
                finish(s, f)
                continue
              }
              f.beta = Math.min(f.beta, f.value)
            }
          }
          continue
        }
        if (s.nodes >= maxNodes) return s
        if (f.kind === 'chance') {
          const outcome = f.chances[f.index]
          if (outcome === undefined) {
            finish(s, f)
            continue
          }
          f.index += 1
          enter(s, outcome.state, f.depth + 1, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, f.player)
          return s
        }
        const move = f.moves[f.index]
        if (move === undefined) {
          finish(s, f)
          continue
        }
        f.index += 1
        const next = game.result(f.state, move)
        if (nega) {
          // A leaf is valued for this frame's player directly; an inner node for its own mover, negated on backup.
          const q = game.isTerminal(next) || f.depth + 1 >= limit ? f.player : game.player(next)
          const same = q === f.player
          f.flip = !same
          enter(s, next, f.depth + 1, same ? f.alpha : -f.beta, same ? f.beta : -f.alpha, q)
        } else {
          enter(s, next, f.depth + 1, f.alpha, f.beta, rootPlayer)
        }
        return s
      }
    },
    done: (s) => (s.status === 'solved' ? 'converged' : s.nodes >= maxNodes ? 'stopped' : undefined),
    observe: (s) => ({
      metrics: {
        nodes: s.nodes,
        cutoffs: s.cutoffs,
        value: s.status === 'solved' ? s.value : (s.stack[0]?.value ?? 0),
      },
      messages: [
        ...(s.visited === null ? [] : [{ kind: 'visit', payload: s.visited as Json }]),
        ...(s.backedUp.length === 0
          ? []
          : [{ kind: 'value', payload: { nodes: s.backedUp.map((b) => ({ node: b.node, value: b.value })) } }]),
        ...(s.pruned.length === 0 ? [] : [{ kind: 'prune', payload: { nodes: s.pruned } as Json }]),
      ],
    }),
  }
}

function runGame<S, A>(algorithm: Algorithm<GameSearchState<A> & EngineState<S, A>>, options: GameSearchOptions<S, A>) {
  const { depth: _d, evaluate: _e, order: _o, maxNodes = 1_000_000, ...run } = options
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: maxNodes + 1 })
  const root = state.stack[0]
  const solvedRun = state.status === 'solved'
  return {
    action: solvedRun ? state.action : root?.best,
    value: solvedRun ? state.value : (root?.value ?? Number.NaN),
    principalVariation: solvedRun ? state.principalVariation : root?.best === undefined ? [] : [root.best],
    nodes: state.nodes,
    cutoffs: state.cutoffs,
    status: solvedRun ? ('solved' as const) : ('budget' as const),
  }
}

/**
 * Minimax search as a traced algorithm (aima-python's `minimax_decision`, with an optional depth limit): values are
 * utilities for the player to move at the root, maximised where that player moves and minimised elsewhere. Each step
 * visits one node.
 *
 * @param game - The game.
 * @param state - The root position.
 * @param options - Depth, evaluation, move ordering and node budget.
 * @returns The algorithm.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { minimaxAlgorithm } from 'mdml-engine/search'
 * const game = { initial: 0, player: () => 'MAX', actions: () => [], result: (s: number) => s, isTerminal: () => true, utility: () => 0 }
 * minimaxAlgorithm(game, 0).name // => 'minimax'
 * ```
 */
export function minimaxAlgorithm<S, A>(
  game: Game<S, A>,
  state: S,
  options: GameSearchOptions<S, A> = {},
): Algorithm<GameSearchState<A>> {
  return gameEngine('minimax', 'minimax', game, state, options)
}

/**
 * Minimax search: the value of a position under optimal play, and the move that achieves it.
 *
 * @param game - The game.
 * @param state - The root position.
 * @param options - Depth, evaluation, move ordering, node budget and tracing.
 * @returns The best action, the value for the player to move, the principal variation and the nodes visited.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { minimax } from 'mdml-engine/search'
 * // AIMA figure 5.2: MAX moves a1, a2 or a3; MIN replies; leaves hold MAX's utility.
 * const tree: Record<string, string[]> = { A: ['B', 'C', 'D'], B: ['B1', 'B2', 'B3'], C: ['C1', 'C2', 'C3'], D: ['D1', 'D2', 'D3'] }
 * const utils: Record<string, number> = { B1: 3, B2: 12, B3: 8, C1: 2, C2: 4, C3: 6, D1: 14, D2: 5, D3: 2 }
 * const game = { initial: 'A', player: (s: string) => (s === 'A' ? 'MAX' : 'MIN'), actions: (s: string) => tree[s] ?? [], result: (_s: string, a: string) => a, isTerminal: (s: string) => !(s in tree), utility: (s: string, p: string) => (p === 'MAX' ? 1 : -1) * (utils[s] ?? 0) }
 * const r = minimax(game, 'A')
 * [r.action, r.value, r.nodes] // => ['B', 3, 13]
 * ```
 */
export function minimax<S, A>(game: Game<S, A>, state: S, options: GameSearchOptions<S, A> = {}): GameSearchResult<A> {
  return runGame(gameEngine('minimax', 'minimax', game, state, options), options)
}

/**
 * Alpha-beta search as a traced algorithm (aima-python's `alpha_beta_search` and `alpha_beta_cutoff_search`): minimax
 * that stops searching a node's moves once its value can no longer affect the root (a cutoff), with optional move
 * ordering to find cutoffs sooner. Each step visits one node; `prune` messages record the cutoffs.
 *
 * @param game - The game.
 * @param state - The root position.
 * @param options - Depth, evaluation, move ordering and node budget.
 * @returns The algorithm.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { alphaBetaAlgorithm } from 'mdml-engine/search'
 * const game = { initial: 0, player: () => 'MAX', actions: () => [], result: (s: number) => s, isTerminal: () => true, utility: () => 0 }
 * alphaBetaAlgorithm(game, 0).name // => 'alpha-beta'
 * ```
 */
export function alphaBetaAlgorithm<S, A>(
  game: Game<S, A>,
  state: S,
  options: GameSearchOptions<S, A> = {},
): Algorithm<GameSearchState<A>> {
  return gameEngine('alpha-beta', 'alpha-beta', game, state, options)
}

/**
 * Alpha-beta search: the minimax value and move, visiting fewer nodes. It searches a `Game` (utilities at the terminal
 * states); `alphaBeta` of `agents` searches an `AecEnv` with rewards on every move and a transposition table.
 *
 * @param game - The game.
 * @param state - The root position.
 * @param options - Depth, evaluation, move ordering, node budget and tracing.
 * @returns The best action, its value, the principal variation, the nodes visited and the cutoffs made.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { alphaBeta } from 'mdml-engine/search'
 * const tree: Record<string, string[]> = { A: ['B', 'C', 'D'], B: ['B1', 'B2', 'B3'], C: ['C1', 'C2', 'C3'], D: ['D1', 'D2', 'D3'] }
 * const utils: Record<string, number> = { B1: 3, B2: 12, B3: 8, C1: 2, C2: 4, C3: 6, D1: 14, D2: 5, D3: 2 }
 * const game = { initial: 'A', player: (s: string) => (s === 'A' ? 'MAX' : 'MIN'), actions: (s: string) => tree[s] ?? [], result: (_s: string, a: string) => a, isTerminal: (s: string) => !(s in tree), utility: (s: string, p: string) => (p === 'MAX' ? 1 : -1) * (utils[s] ?? 0) }
 * const r = alphaBeta(game, 'A')
 * [r.action, r.value, r.nodes] // => ['B', 3, 11]
 * ```
 */
export function alphaBeta<S, A>(
  game: Game<S, A>,
  state: S,
  options: GameSearchOptions<S, A> = {},
): GameSearchResult<A> {
  return runGame(gameEngine('alpha-beta', 'alpha-beta', game, state, options), options)
}

/**
 * Negamax search as a traced algorithm (Knuth and Moore 1975): for two-player zero-sum games, the value of a position
 * for the player to move is the best over moves of the next position's value, negated when the opponent moves there,
 * optionally with alpha-beta pruning. Gives the same values as minimax from the mover's side.
 *
 * @param game - The game (two-player, zero-sum).
 * @param state - The root position.
 * @param options - Alpha-beta pruning (default true), depth, evaluation, move ordering and node budget.
 * @returns The algorithm.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { negamaxAlgorithm } from 'mdml-engine/search'
 * const game = { initial: 0, player: () => 'MAX', actions: () => [], result: (s: number) => s, isTerminal: () => true, utility: () => 0 }
 * negamaxAlgorithm(game, 0).name // => 'negamax'
 * ```
 */
export function negamaxAlgorithm<S, A>(
  game: Game<S, A>,
  state: S,
  options: GameSearchOptions<S, A> & { readonly alphaBeta?: boolean } = {},
): Algorithm<GameSearchState<A>> {
  return gameEngine('negamax', options.alphaBeta === false ? 'negamax' : 'negamax-alpha-beta', game, state, options)
}

/**
 * Negamax search.
 *
 * @param game - The game (two-player, zero-sum).
 * @param state - The root position.
 * @param options - Alpha-beta pruning (default true), depth, evaluation, move ordering, node budget and tracing.
 * @returns The best action, its value for the player to move, the principal variation and the effort.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { negamax } from 'mdml-engine/search'
 * const tree: Record<string, string[]> = { A: ['B', 'C', 'D'], B: ['B1', 'B2', 'B3'], C: ['C1', 'C2', 'C3'], D: ['D1', 'D2', 'D3'] }
 * const utils: Record<string, number> = { B1: 3, B2: 12, B3: 8, C1: 2, C2: 4, C3: 6, D1: 14, D2: 5, D3: 2 }
 * const game = { initial: 'A', player: (s: string) => (s === 'A' ? 'MAX' : 'MIN'), actions: (s: string) => tree[s] ?? [], result: (_s: string, a: string) => a, isTerminal: (s: string) => !(s in tree), utility: (s: string, p: string) => (p === 'MAX' ? 1 : -1) * (utils[s] ?? 0) }
 * const r = negamax(game, 'A')
 * [r.action, r.value] // => ['B', 3]
 * ```
 */
export function negamax<S, A>(
  game: Game<S, A>,
  state: S,
  options: GameSearchOptions<S, A> & { readonly alphaBeta?: boolean } = {},
): GameSearchResult<A> {
  const { alphaBeta: ab, ...rest } = options
  return runGame(gameEngine('negamax', ab === false ? 'negamax' : 'negamax-alpha-beta', game, state, rest), rest)
}

/**
 * Expectimax search as a traced algorithm (AIMA 5.5, "expectiminimax"): minimax in which chance states take the
 * probability-weighted mean of their outcomes' values.
 *
 * @param game - The stochastic game.
 * @param state - The root position.
 * @param options - Depth (chance levels count as plies), evaluation, move ordering and node budget.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when the game has no chance nodes.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { expectimaxAlgorithm } from 'mdml-engine/search'
 * const game = { initial: 0, player: () => 'MAX', actions: () => [], result: (s: number) => s, isTerminal: () => true, utility: () => 0, isChance: () => false, outcomes: () => [] }
 * expectimaxAlgorithm(game, 0).name // => 'expectimax'
 * ```
 */
export function expectimaxAlgorithm<S, A>(
  game: StochasticGame<S, A>,
  state: S,
  options: GameSearchOptions<S, A> = {},
): Algorithm<GameSearchState<A>> {
  return gameEngine('expectimax', 'expectimax', game, state, options)
}

/**
 * Expectimax search: the expected value of a position under optimal play against an optimal opponent and chance.
 *
 * @param game - The stochastic game.
 * @param state - The root position.
 * @param options - Depth, evaluation, move ordering, node budget and tracing.
 * @returns The best action, the expected value for the player to move, and the effort.
 * @throws MdmlError with code `value` when the game has no chance nodes.
 * @concept adversarial-search
 * @example
 * ```ts
 * import { expectimax } from 'mdml-engine/search'
 * // MAX bets or passes; a bet is settled by a fair coin: +3 or -2. Passing is worth 0.
 * const game = {
 *   initial: 'start', player: () => 'MAX', actions: (s: string) => (s === 'start' ? ['bet', 'pass'] : []),
 *   result: (_s: string, a: string) => a, isTerminal: (s: string) => s === 'pass' || s === 'win' || s === 'lose',
 *   utility: (s: string) => (s === 'win' ? 3 : s === 'lose' ? -2 : 0), isChance: (s: string) => s === 'bet',
 *   outcomes: () => [{ state: 'win', probability: 0.5 }, { state: 'lose', probability: 0.5 }],
 * }
 * const r = expectimax(game, 'start')
 * [r.action, r.value] // => ['bet', 0.5]
 * ```
 */
export function expectimax<S, A>(
  game: StochasticGame<S, A>,
  state: S,
  options: GameSearchOptions<S, A> = {},
): GameSearchResult<A> {
  return runGame(gameEngine('expectimax', 'expectimax', game, state, options), options)
}

/** Options of Monte Carlo tree search. */
export interface MctsOptions<S, A> extends SearchRunOptions {
  /** The number of simulations (default 1000). */
  readonly simulations?: number
  /** The exploration constant (default sqrt 2 for UCT, 1.5 for PUCT). */
  readonly c?: number
  /** Prior move probabilities for PUCT (AlphaZero's selection rule); without them selection is UCT. */
  readonly prior?: (state: S, actions: readonly A[]) => readonly number[]
  /** A leaf evaluation for a player replacing rollouts (AlphaZero style), in utility units. */
  readonly value?: (state: S, player: string) => number
  /** The rollout policy (default uniformly random). */
  readonly rollout?: (state: S, actions: readonly A[], key: Key) => A
  /** The longest rollout (default 1000 moves); a rollout stopped early scores by `value` or 0. */
  readonly rolloutDepth?: number
}

interface TreeNode<S, A> {
  readonly state: S
  readonly terminal: boolean
  readonly mover: string | undefined
  readonly actions: readonly A[]
  readonly children: (TreeNode<S, A> | undefined)[]
  readonly prior: readonly number[]
  visits: number
  /** The total return for the player who moved into this node. */
  total: number
}

/** The state of a Monte Carlo tree search: the tree's root statistics after each simulation. */
export interface MctsState<S, A> {
  readonly root: {
    readonly state: S
    readonly visits: number
    readonly children: readonly (
      | { readonly visits: number; readonly total: number; readonly children: unknown[] }
      | undefined
    )[]
  }
  readonly actions: readonly A[]
  readonly simulations: number
  /** The states along the last simulation's tree path, as labels. */
  readonly path: readonly string[]
}

/** The result of Monte Carlo tree search. */
export interface MctsResult<A> {
  /** The most visited root action (the first of equal ones). */
  readonly action: A | undefined
  readonly actions: readonly A[]
  readonly visits: readonly number[]
  /** Mean returns of the root actions for the player to move at the root. */
  readonly values: readonly number[]
  readonly simulations: number
}

function mctsEngine<S, A>(
  game: Game<S, A>,
  state: S,
  options: MctsOptions<S, A>,
): Algorithm<MctsState<S, A> & { readonly tree: TreeNode<S, A>; path: string[]; simulations: number }> {
  const key = stateKeyOf(game)
  const label = (s: S): string => game.label?.(s) ?? String(key(s))
  const puct = options.prior !== undefined
  const c = options.c ?? (puct ? 1.5 : Math.SQRT2)
  const rolloutDepth = options.rolloutDepth ?? 1000
  if (game.isTerminal(state)) throw new MdmlError('value', 'mcts: the root position is terminal')
  const node = (s: S): TreeNode<S, A> => {
    const terminal = game.isTerminal(s)
    const actions = terminal ? [] : game.actions(s)
    const p = puct && !terminal ? (options.prior?.(s, actions) ?? []) : []
    return {
      state: s,
      terminal,
      mover: terminal ? undefined : game.player(s),
      actions,
      children: actions.map(() => undefined),
      prior: p,
      visits: 0,
      total: 0,
    }
  }
  const select = (n: TreeNode<S, A>): number => {
    if (!puct) {
      const fresh = n.children.indexOf(undefined)
      if (fresh >= 0) return fresh
    }
    const scores = n.children.map((ch, i) => {
      const visits = ch?.visits ?? 0
      const q = visits === 0 ? 0 : (ch?.total ?? 0) / visits
      if (puct) return q + c * (n.prior[i] ?? 1 / n.actions.length) * (Math.sqrt(n.visits) / (1 + visits))
      return q + c * Math.sqrt(Math.log(n.visits) / visits)
    })
    return argmaxOf(scores)
  }
  return {
    name: 'mcts',
    config: { c, puct, rolloutDepth },
    schema: channels({
      metrics: { simulations: { description: 'simulations so far' }, best: { description: 'visits of the best move' } },
      messages: { simulation: { description: 'the tree path of a simulation', payload: { path: 'state labels' } } },
    }),
    init: () => {
      const tree = node(state)
      return { root: tree, tree, actions: tree.actions, simulations: 0, path: [] }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const path: TreeNode<S, A>[] = [s.tree]
      let n = s.tree
      // Selection and expansion: descend by the upper confidence rule, creating the first unvisited node.
      while (!n.terminal) {
        const i = select(n)
        const existing = n.children[i]
        if (existing === undefined) {
          const child = node(game.result(n.state, n.actions[i] as A))
          n.children[i] = child
          path.push(child)
          n = child
          break
        }
        path.push(existing)
        n = existing
      }
      // Evaluation: the leaf's utility, a value estimate, or a rollout's final utility (per player, lazily).
      let end = n.state
      let scored: ((player: string) => number) | undefined
      if (n.terminal) scored = (p) => game.utility(end, p)
      else if (options.value !== undefined) {
        const v = options.value
        scored = (p) => v(n.state, p)
      } else {
        for (let t = 0; t < rolloutDepth && !game.isTerminal(end); t++) {
          const moves = game.actions(end)
          const a =
            options.rollout === undefined
              ? (moves[Math.min(moves.length - 1, Math.floor((rng.uniform([1]).data[0] ?? 0) * moves.length))] as A)
              : options.rollout(end, moves, rng.nextKey())
          end = game.result(end, a)
        }
        const finalState = end
        scored = game.isTerminal(finalState) ? (p) => game.utility(finalState, p) : () => 0
      }
      // Backup: each node accumulates the return of the player who moved into it.
      const memo = new Map<string, number>()
      const value = (p: string): number => {
        const known = memo.get(p)
        if (known !== undefined) return known
        const v = scored === undefined ? 0 : scored(p)
        memo.set(p, v)
        return v
      }
      for (let k = 0; k < path.length; k++) {
        const x = path[k] as TreeNode<S, A>
        x.visits += 1
        const mover = path[k - 1]?.mover
        if (mover !== undefined) x.total += value(mover)
      }
      s.simulations += 1
      s.path = path.map((x) => label(x.state))
      return s
    },
    observe: (s) => ({
      metrics: {
        simulations: s.simulations,
        best: s.tree.children.reduce((m, ch) => Math.max(m, ch?.visits ?? 0), 0),
      },
      messages: [{ kind: 'simulation', payload: { path: [...s.path] } }],
    }),
  }
}

/**
 * Monte Carlo tree search for games as a traced algorithm (UCT, Kocsis and Szepesvari 2006; PUCT with priors as in
 * AlphaZero): each step runs one simulation (selection by upper confidence bounds from the mover's side, expansion of
 * one node, a random rollout or a value estimate, and backup of each player's return). Works for any number of
 * players; returns are utilities.
 *
 * @param game - The game.
 * @param state - The root position (not terminal).
 * @param options - Exploration constant, priors, value function, rollout policy and depth (the simulation count is the
 * run's step budget).
 * @returns The algorithm; the state's `root` holds the visit statistics.
 * @throws MdmlError with code `value` when the root is terminal.
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { mctsAlgorithm } from 'mdml-engine/search'
 * const game = { initial: 0, player: () => 'A', actions: () => [1, 2], result: (s: number, a: number) => s + a, isTerminal: (s: number) => s >= 3, utility: (s: number) => (s === 3 ? 1 : 0) }
 * mctsAlgorithm(game, 0).name // => 'mcts'
 * ```
 */
export function mctsAlgorithm<S, A>(
  game: Game<S, A>,
  state: S,
  options: MctsOptions<S, A> = {},
): Algorithm<MctsState<S, A>> {
  return mctsEngine(game, state, options)
}

/**
 * Monte Carlo tree search: the most visited move after a number of simulations, with the root statistics.
 *
 * @param game - The game.
 * @param state - The root position (not terminal).
 * @param options - Simulations (default 1000), exploration constant, priors, value function, rollout policy, a seed or
 * key, and tracing.
 * @returns The chosen action, the root actions with their visits and mean returns, and the simulations run.
 * @throws MdmlError with code `value` when the root is terminal.
 * @concept monte-carlo-tree-search
 * @example
 * ```ts
 * import { mcts } from 'mdml-engine/search'
 * // A one-player race to exactly 3: adding 1 then 2 (or 2 then 1) wins; overshooting scores 0.
 * const game = { initial: 0, player: () => 'A', actions: () => [1, 2, 3], result: (s: number, a: number) => s + a, isTerminal: (s: number) => s >= 3, utility: (s: number) => (s === 3 ? 1 : 0) }
 * mcts(game, 0, { simulations: 200, seed: 1 }).values.every((v) => v > 0.5) // => true
 * ```
 */
export function mcts<S, A>(game: Game<S, A>, state: S, options: MctsOptions<S, A> = {}): MctsResult<A> {
  const { simulations = 1000, c: _c, prior: _p, value: _v, rollout: _r, rolloutDepth: _d, ...run } = options
  if (!(Number.isInteger(simulations) && simulations >= 1)) {
    throw new MdmlError('value', `mcts: simulations ${simulations} must be a positive integer`)
  }
  const { state: s } = runAlgorithm(mctsEngine(game, state, options), { ...run, maxSteps: simulations })
  const visits = s.tree.children.map((ch) => ch?.visits ?? 0)
  const values = s.tree.children.map((ch) => (ch === undefined || ch.visits === 0 ? 0 : ch.total / ch.visits))
  return {
    action: s.tree.actions.length === 0 ? undefined : s.tree.actions[argmaxOf(visits)],
    actions: s.tree.actions,
    visits,
    values,
    simulations: s.simulations,
  }
}
