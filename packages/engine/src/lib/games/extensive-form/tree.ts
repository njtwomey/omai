/**
 * The extensive-form game contract and its tree, in OpenSpiel's style: games with players, chance and information sets
 * as immutable states, materialised trees with their information sets, tabular behaviour policies and expected returns.
 * Also backward induction for perfect-information games, games from a literal tree, and conversion of a small game to
 * normal form.
 */

import { WhiteboxError } from '../../base/index.ts'
import { type NormalFormGame, normalFormGame } from '../normal-form/index.ts'
import { CHANCE, nodeValues, policyTable, TERMINAL, tablePolicy } from './_tree.ts'

export { CHANCE, TERMINAL } from './_tree.ts'

/**
 * A finite game in extensive form, in OpenSpiel's style: immutable states, a current player (or {@link CHANCE}, or
 * {@link TERMINAL}), legal actions as integers, chance outcomes with probabilities, information-state keys (equal for
 * states the player cannot tell apart) and returns at terminal states. Players are numbered from 0.
 *
 * @concept extensive-form-game
 */
export interface ExtensiveGame<S> {
  readonly name: string
  readonly numPlayers: number
  initial(): S
  currentPlayer(state: S): number
  legalActions(state: S): readonly number[]
  chanceOutcomes(state: S): readonly (readonly [number, number])[]
  apply(state: S, action: number): S
  returns(state: S): readonly number[]
  informationState(state: S, player: number): string
  actionName?(state: S, action: number): string
}

/** One node of a materialised game tree. */
export interface GameNode {
  /** The player to move, {@link CHANCE} or {@link TERMINAL}. */
  readonly player: number
  /** The information set (index into `infosets`), or -1 at chance and terminal nodes. */
  readonly infoset: number
  readonly actions: readonly number[]
  readonly children: readonly number[]
  /** Chance probabilities, aligned with `actions` (empty elsewhere). */
  readonly chance: readonly number[]
  /** Terminal returns (empty elsewhere). */
  readonly returns: readonly number[]
  readonly parent: number
  readonly depth: number
}

/** An information set: its key, its player, its legal actions and the nodes it contains. */
export interface InfoSet {
  readonly key: string
  readonly player: number
  readonly actions: readonly number[]
  readonly actionNames: readonly string[]
  readonly nodes: readonly number[]
}

/** A game tree with every node in depth-first preorder (children after their parent) and its information sets. */
export interface GameTree {
  readonly name: string
  readonly numPlayers: number
  readonly nodes: readonly GameNode[]
  readonly infosets: readonly InfoSet[]
  /** The index of every information-set key. */
  readonly index: ReadonlyMap<string, number>
}

/**
 * A behaviour policy for any players: for each information-state key, the probabilities of its legal actions (in
 * their order). Missing keys play uniformly.
 */
export type TabularPolicy = Readonly<Record<string, readonly number[]>>

/**
 * Materialise a game: every node in depth-first preorder, with information sets discovered in the same order (as
 * OpenSpiel's CFR solvers do).
 *
 * @param game - The game.
 * @param options - A cap on nodes (default 1000000).
 * @returns The tree.
 * @throws WhiteboxError with code `value` when the tree is larger than the cap or an information set's legal actions
 * differ between its states.
 * @concept extensive-form-game
 * @example
 * ```ts
 * import { gameTree, kuhnPoker } from 'whitebox-engine/games'
 * const t = gameTree(kuhnPoker())
 * [t.nodes.length, t.infosets.length] // => [58, 12]
 * ```
 */
export function gameTree<S>(game: ExtensiveGame<S>, options: { readonly maxNodes?: number } = {}): GameTree {
  const maxNodes = options.maxNodes ?? 1_000_000
  const nodes: GameNode[] = []
  const infosets: { key: string; player: number; actions: number[]; actionNames: string[]; nodes: number[] }[] = []
  const index = new Map<string, number>()
  const visit = (state: S, parent: number, depth: number): number => {
    if (nodes.length >= maxNodes) throw new WhiteboxError('value', `${game.name}: more than ${maxNodes} nodes`)
    const id = nodes.length
    const player = game.currentPlayer(state)
    const placeholder: GameNode = {
      player,
      infoset: -1,
      actions: [],
      children: [],
      chance: [],
      returns: [],
      parent,
      depth,
    }
    nodes.push(placeholder)
    if (player === TERMINAL) {
      nodes[id] = { ...placeholder, returns: [...game.returns(state)] }
      return id
    }
    if (player === CHANCE) {
      const outcomes = game.chanceOutcomes(state)
      const children = outcomes.map(([a]) => visit(game.apply(state, a), id, depth + 1))
      nodes[id] = { ...placeholder, actions: outcomes.map(([a]) => a), chance: outcomes.map(([, p]) => p), children }
      return id
    }
    const actions = [...game.legalActions(state)]
    const key = game.informationState(state, player)
    let k = index.get(key)
    if (k === undefined) {
      k = infosets.length
      index.set(key, k)
      infosets.push({
        key,
        player,
        actions,
        actionNames: actions.map((a) => game.actionName?.(state, a) ?? String(a)),
        nodes: [],
      })
    }
    const info = infosets[k]
    if (
      info === undefined ||
      info.player !== player ||
      info.actions.length !== actions.length ||
      info.actions.some((a, i) => a !== actions[i])
    )
      throw new WhiteboxError(
        'value',
        `${game.name}: the states of information set '${key}' disagree on player or actions`,
      )
    info.nodes.push(id)
    const children = actions.map((a) => visit(game.apply(state, a), id, depth + 1))
    nodes[id] = { ...placeholder, infoset: k, actions, children }
    return id
  }
  visit(game.initial(), -1, 0)
  return { name: game.name, numPlayers: game.numPlayers, nodes, infosets, index }
}

/**
 * The materialised tree of a game, or the tree itself when given one (every solver accepts either).
 *
 * @param game - A game or its tree.
 * @returns The tree.
 * @example
 * ```ts
 * import { gameTree, kuhnPoker, treeOf } from 'whitebox-engine/games'
 * const t = gameTree(kuhnPoker())
 * treeOf(t) === t // => true
 * ```
 */
export function treeOf<S>(game: ExtensiveGame<S> | GameTree): GameTree {
  return 'nodes' in game ? game : gameTree(game)
}

/**
 * The uniform policy of every player.
 *
 * @param tree - The game tree.
 * @returns Uniform probabilities at every information set.
 * @example
 * ```ts
 * import { gameTree, kuhnPoker, uniformPolicy } from 'whitebox-engine/games'
 * uniformPolicy(gameTree(kuhnPoker()))['0'] // => [0.5, 0.5]
 * ```
 */
export function uniformPolicy(tree: GameTree): TabularPolicy {
  return tablePolicy(
    tree,
    tree.infosets.map((i) => new Float64Array(i.actions.length).fill(1 / i.actions.length)),
  )
}

/**
 * Every player's expected return when all follow a policy.
 *
 * @param game - The game or its tree.
 * @param policy - Action probabilities by information state (uniform where missing).
 * @returns The expected returns.
 * @concept extensive-form-game
 * @example
 * ```ts
 * import { expectedReturns, kuhnPoker, uniformPolicy, gameTree } from 'whitebox-engine/games'
 * const t = gameTree(kuhnPoker())
 * expectedReturns(t, uniformPolicy(t)).map((v) => Math.round(v * 1e9) / 1e9) // => [0.125, -0.125]
 * ```
 */
export function expectedReturns<S>(game: ExtensiveGame<S> | GameTree, policy: TabularPolicy): number[] {
  const tree = treeOf(game)
  return Array.from(nodeValues(tree, policyTable(tree, policy))[0] ?? [])
}

/** The outcome of {@link backwardInduction}. */
export interface BackwardInductionResult {
  /** The expected returns from the root under subgame-perfect play. */
  readonly values: readonly number[]
  /** The pure subgame-perfect policy: probability 1 on the chosen action at every information set. */
  readonly policy: TabularPolicy
  /** The value of every node (in tree order). */
  readonly nodeValues: readonly (readonly number[])[]
}

/**
 * Backward induction (Zermelo; Kuhn 1953) for a perfect-information game with any number of players: from the leaves
 * up, every player picks the action maximising its own value (ties to the first action) and chance takes
 * expectations. The result is a subgame-perfect equilibrium.
 *
 * @param game - A perfect-information game (every information set a single node) or its tree.
 * @returns The root values, the pure policy and every node's value.
 * @throws WhiteboxError with code `value` when some information set holds more than one node.
 * @concept backward-induction
 * @example
 * ```ts
 * import { backwardInduction, treeGame } from 'whitebox-engine/games'
 * // The ultimatum game with a pie of 2: offer 0 or 1 to the responder, who accepts or rejects.
 * const g = treeGame('ultimatum', 2, { player: 0, actions: [
 *   { label: 'offer 0', child: { player: 1, actions: [{ label: 'accept', child: { returns: [2, 0] } }, { label: 'reject', child: { returns: [0, 0] } }] } },
 *   { label: 'offer 1', child: { player: 1, actions: [{ label: 'accept', child: { returns: [1, 1] } }, { label: 'reject', child: { returns: [0, 0] } }] } },
 * ] })
 * backwardInduction(g).values // => [2, 0]
 * ```
 */
export function backwardInduction<S>(game: ExtensiveGame<S> | GameTree): BackwardInductionResult {
  const tree = treeOf(game)
  for (const info of tree.infosets)
    if (info.nodes.length > 1)
      throw new WhiteboxError(
        'value',
        `${tree.name}: information set '${info.key}' has several nodes; backward induction needs perfect information`,
      )
  const n = tree.numPlayers
  const values = tree.nodes.map(() => new Array<number>(n).fill(0))
  const choice = new Array<number>(tree.infosets.length).fill(0)
  for (let id = tree.nodes.length - 1; id >= 0; id--) {
    const node = tree.nodes[id]
    if (node === undefined) continue
    if (node.player === TERMINAL) {
      values[id] = [...node.returns]
      continue
    }
    if (node.player === CHANCE) {
      const v = new Array<number>(n).fill(0)
      for (const [a, c] of node.children.entries())
        for (let p = 0; p < n; p++) v[p] = (v[p] ?? 0) + (node.chance[a] ?? 0) * (values[c]?.[p] ?? 0)
      values[id] = v
      continue
    }
    let best = 0
    for (const [a, c] of node.children.entries())
      if ((values[c]?.[node.player] ?? -Infinity) > (values[node.children[best] ?? 0]?.[node.player] ?? -Infinity))
        best = a
    choice[node.infoset] = best
    values[id] = [...(values[node.children[best] ?? 0] ?? [])]
  }
  return {
    values: values[0] ?? [],
    policy: tablePolicy(
      tree,
      tree.infosets.map((info, k) => info.actions.map((_, a) => (a === choice[k] ? 1 : 0))),
    ),
    nodeValues: values,
  }
}

/** A node of a literal game tree for {@link treeGame}. */
export type TreeSpec =
  | { readonly returns: readonly number[] }
  | { readonly chance: readonly { readonly p: number; readonly label?: string; readonly child: TreeSpec }[] }
  | {
      readonly player: number
      /** The information set's key; nodes sharing a key are indistinguishable to the player (default: the node). */
      readonly infoset?: string
      readonly actions: readonly { readonly label: string; readonly child: TreeSpec }[]
    }

/**
 * An extensive-form game from a literal tree: decision nodes (a player, an optional information-set key and labelled
 * actions), chance nodes (outcome probabilities) and leaves (returns). States are paths of action indices from the
 * root.
 *
 * @param name - The game's name.
 * @param numPlayers - The number of players.
 * @param root - The tree.
 * @returns The game.
 * @concept extensive-form-game
 * @example
 * ```ts
 * import { gameTree, treeGame } from 'whitebox-engine/games'
 * const g = treeGame('coin', 1, { chance: [{ p: 0.5, child: { returns: [1] } }, { p: 0.5, child: { returns: [-1] } }] })
 * gameTree(g).nodes.length // => 3
 * ```
 */
export function treeGame(name: string, numPlayers: number, root: TreeSpec): ExtensiveGame<readonly number[]> {
  const at = (path: readonly number[]): TreeSpec => {
    let node = root
    for (const i of path) {
      const next = 'chance' in node ? node.chance[i]?.child : 'actions' in node ? node.actions[i]?.child : undefined
      if (next === undefined) throw new WhiteboxError('value', `${name}: no action ${i} at [${path.join(', ')}]`)
      node = next
    }
    return node
  }
  return {
    name,
    numPlayers,
    initial: () => [],
    currentPlayer: (s) => {
      const node = at(s)
      return 'returns' in node ? TERMINAL : 'chance' in node ? CHANCE : node.player
    },
    legalActions: (s) => {
      const node = at(s)
      return 'actions' in node ? node.actions.map((_, i) => i) : []
    },
    chanceOutcomes: (s) => {
      const node = at(s)
      return 'chance' in node ? node.chance.map((o, i) => [i, o.p] as const) : []
    },
    apply: (s, a) => [...s, a],
    returns: (s) => {
      const node = at(s)
      return 'returns' in node ? node.returns : new Array<number>(numPlayers).fill(0)
    },
    informationState: (s, player) => {
      const node = at(s)
      return 'actions' in node && node.infoset !== undefined ? node.infoset : `${player}@${s.join('.')}`
    },
    actionName: (s, a) => {
      const node = at(s)
      return 'actions' in node
        ? (node.actions[a]?.label ?? String(a))
        : 'chance' in node
          ? (node.chance[a]?.label ?? String(a))
          : String(a)
    },
  }
}

/**
 * The normal form of a small game: every player's pure strategies (one action at each of its information sets) and
 * the expected returns of every profile. The number of strategies grows exponentially with the information sets.
 *
 * @param game - The game or its tree.
 * @param options - A cap on each player's strategies (default 4096).
 * @returns The normal-form game; strategy names list `key=action` for every information set.
 * @throws WhiteboxError with code `value` when a player has more strategies than the cap.
 * @concept extensive-form-game
 * @example
 * ```ts
 * import { kuhnPoker, toNormalForm } from 'whitebox-engine/games'
 * toNormalForm(kuhnPoker()).shape // => [64, 64]
 * ```
 */
export function toNormalForm<S>(
  game: ExtensiveGame<S> | GameTree,
  options: { readonly maxStrategies?: number } = {},
): NormalFormGame {
  const tree = treeOf(game)
  const cap = options.maxStrategies ?? 4096
  const own = Array.from({ length: tree.numPlayers }, (_, p) =>
    tree.infosets.flatMap((info, k) => (info.player === p ? [k] : [])),
  )
  const counts = own.map((ks, p) => {
    let c = 1
    for (const k of ks) c *= tree.infosets[k]?.actions.length ?? 1
    if (c > cap) throw new WhiteboxError('value', `${tree.name}: player ${p} has ${c} pure strategies (cap ${cap})`)
    return c
  })
  // Pure strategy s of player p: mixed-radix digits over its information sets.
  const digits = (p: number, s: number): number[] => {
    const out: number[] = []
    let r = s
    for (const k of own[p] ?? []) {
      const m = tree.infosets[k]?.actions.length ?? 1
      out.push(r % m)
      r = Math.floor(r / m)
    }
    return out
  }
  let total = 1
  for (const c of counts) total *= c
  const payoffs = counts.map(() => new Float64Array(total))
  const table = tree.infosets.map((info) => new Float64Array(info.actions.length))
  for (let idx = 0; idx < total; idx++) {
    let r = idx
    const profile = new Array<number>(counts.length).fill(0)
    for (let p = counts.length - 1; p >= 0; p--) {
      profile[p] = r % (counts[p] ?? 1)
      r = Math.floor(r / (counts[p] ?? 1))
    }
    for (const t of table) t.fill(0)
    for (const [p, s] of profile.entries())
      for (const [j, d] of digits(p, s).entries()) table[own[p]?.[j] ?? 0]?.fill(1, d, d + 1)
    const v = nodeValues(tree, table)[0] ?? new Float64Array(0)
    for (const [p, u] of payoffs.entries()) u[idx] = v[p] ?? 0
  }
  const names = counts.map((c, p) =>
    Array.from({ length: c }, (_, s) =>
      digits(p, s)
        .map((d, j) => {
          const info = tree.infosets[own[p]?.[j] ?? 0]
          return `${info?.key ?? ''}=${info?.actionNames[d] ?? d}`
        })
        .join(';'),
    ),
  )
  const nest = (flat: Float64Array): unknown => {
    const build = (depth: number, offset: number, stride: number): unknown => {
      const k = counts[depth] ?? 1
      const inner = stride / k
      return depth === counts.length - 1
        ? Array.from(flat.subarray(offset, offset + k))
        : Array.from({ length: k }, (_, i) => build(depth + 1, offset + i * inner, inner))
    }
    return build(0, 0, total)
  }
  return normalFormGame(
    payoffs.map((u) => nest(u) as number[][]),
    { name: `${tree.name} (normal form)`, actionNames: names },
  )
}
