/**
 * Best responses and exploitability in extensive-form games, as in OpenSpiel: the best response of one player to the
 * others' policy, computed from counterfactual values at its information sets, and NashConv and exploitability, which
 * measure how far a policy profile is from a Nash equilibrium.
 */

import { OmaiError } from '../../base/index.ts'
import { CHANCE, nodeValues, policyTable, TERMINAL, tablePolicy } from './_tree.ts'
import { type ExtensiveGame, type GameTree, type TabularPolicy, treeOf } from './tree.ts'

/** The outcome of {@link bestResponse}. */
export interface BestResponseResult {
  /** The responder's expected return when it best responds and the others follow the policy. */
  readonly value: number
  /** The responder's expected return under the policy itself. */
  readonly onPolicyValue: number
  /** A pure best response: probability 1 on the chosen action at each of the responder's information sets. */
  readonly policy: TabularPolicy
}

/** The reach probability of every node from the players other than `player`, and chance. */
function othersReach(tree: GameTree, table: readonly Float64Array[], player: number): Float64Array {
  const reach = new Float64Array(tree.nodes.length)
  reach[0] = 1
  for (const [id, node] of tree.nodes.entries()) {
    const r = reach[id] ?? 0
    if (node.player === TERMINAL) continue
    const probs = node.player === CHANCE ? node.chance : node.player === player ? null : table[node.infoset]
    for (const [a, c] of node.children.entries()) reach[c] = probs === null ? r : r * (probs?.[a] ?? 0)
  }
  return reach
}

/**
 * A best response of one player to a policy of the others (OpenSpiel's `BestResponsePolicy`): at every information
 * set of the responder, the action maximising the counterfactual value, the sum over its states of the others' reach
 * probability times the value of the action (ties to the first action), evaluated from the leaves up so deeper
 * choices are already best responses. Needs perfect recall.
 *
 * @param game - The game or its tree.
 * @param policy - The policy of every player (the responder's own entries are ignored).
 * @param player - The responder.
 * @returns The best-response value, the on-policy value and the pure best response.
 * @throws OmaiError with code `value` for a missing player.
 * @concept best-response
 * @example
 * ```ts
 * import { bestResponse, gameTree, kuhnPoker, uniformPolicy } from 'omai-engine/games'
 * const t = gameTree(kuhnPoker())
 * Math.round(bestResponse(t, uniformPolicy(t), 0).value * 1e9) / 1e9 // => 0.5
 * ```
 */
export function bestResponse<S>(
  game: ExtensiveGame<S> | GameTree,
  policy: TabularPolicy,
  player: number,
): BestResponseResult {
  const tree = treeOf(game)
  if (!Number.isInteger(player) || player < 0 || player >= tree.numPlayers)
    throw new OmaiError('value', `${tree.name}: no player ${player}`)
  const table = policyTable(tree, policy)
  const reach = othersReach(tree, table, player)
  const choice = new Int32Array(tree.infosets.length).fill(-1)
  const memo = new Float64Array(tree.nodes.length).fill(Number.NaN)
  const value = (id: number): number => {
    const cached = memo[id] ?? Number.NaN
    if (!Number.isNaN(cached)) return cached
    const node = tree.nodes[id]
    let v = 0
    if (node === undefined) v = 0
    else if (node.player === TERMINAL) v = node.returns[player] ?? 0
    else if (node.player === CHANCE) for (const [a, c] of node.children.entries()) v += (node.chance[a] ?? 0) * value(c)
    else if (node.player !== player) {
      const probs = table[node.infoset]
      for (const [a, c] of node.children.entries()) {
        const p = probs?.[a] ?? 0
        if (p > 0) v += p * value(c)
      }
    } else v = value(node.children[best(node.infoset)] ?? 0)
    memo[id] = v
    return v
  }
  const best = (k: number): number => {
    const known = choice[k] ?? -1
    if (known >= 0) return known
    const info = tree.infosets[k]
    let arg = 0
    let top = -Infinity
    for (let a = 0; a < (info?.actions.length ?? 0); a++) {
      let total = 0
      for (const h of info?.nodes ?? []) {
        const w = reach[h] ?? 0
        if (w !== 0) total += w * value(tree.nodes[h]?.children[a] ?? 0)
      }
      if (total > top) {
        top = total
        arg = a
      }
    }
    choice[k] = arg
    return arg
  }
  const brValue = value(0)
  for (const [k, info] of tree.infosets.entries()) if (info.player === player) best(k)
  const onPolicy = nodeValues(tree, table)[0]?.[player] ?? 0
  const brTable = tree.infosets.map((info, k) =>
    info.player === player
      ? Float64Array.from(info.actions, (_, a) => (a === choice[k] ? 1 : 0))
      : Float64Array.from(table[k] ?? []),
  )
  const full = tablePolicy(tree, brTable)
  const own: Record<string, readonly number[]> = {}
  for (const info of tree.infosets) if (info.player === player) own[info.key] = full[info.key] ?? []
  return { value: brValue, onPolicyValue: onPolicy, policy: own }
}

/**
 * NashConv of a policy (OpenSpiel's `nash_conv`): the sum over players of what each gains by best responding to the
 * others. Zero exactly at a Nash equilibrium.
 *
 * @param game - The game or its tree.
 * @param policy - The policy of every player.
 * @returns The NashConv.
 * @concept nash-equilibrium
 * @example
 * ```ts
 * import { extensiveNashConv, gameTree, kuhnPoker, uniformPolicy } from 'omai-engine/games'
 * const t = gameTree(kuhnPoker())
 * Math.round(extensiveNashConv(t, uniformPolicy(t)) * 1e9) / 1e9 // => 0.916666667
 * ```
 */
export function extensiveNashConv<S>(game: ExtensiveGame<S> | GameTree, policy: TabularPolicy): number {
  const tree = treeOf(game)
  let s = 0
  for (let p = 0; p < tree.numPlayers; p++) {
    const r = bestResponse(tree, policy, p)
    s += r.value - r.onPolicyValue
  }
  return s
}

/**
 * The exploitability of a policy (OpenSpiel's `exploitability`): NashConv divided by the number of players; in a
 * two-player zero-sum game, the average of what the two best responses win against it.
 *
 * @param game - The game or its tree.
 * @param policy - The policy of every player.
 * @returns The exploitability.
 * @concept exploitability
 * @example
 * ```ts
 * import { exploitability, gameTree, kuhnEquilibrium, kuhnPoker } from 'omai-engine/games'
 * Math.abs(exploitability(gameTree(kuhnPoker()), kuhnEquilibrium(0.2))) < 1e-12 // => true
 * ```
 */
export function exploitability<S>(game: ExtensiveGame<S> | GameTree, policy: TabularPolicy): number {
  const tree = treeOf(game)
  return extensiveNashConv(tree, policy) / tree.numPlayers
}
