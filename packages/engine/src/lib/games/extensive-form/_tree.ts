/**
 * Policies and node values over a game tree, private to the extensive-form solvers: the chance and terminal player
 * markers, conversion between tabular policies and per-information-set probability arrays, and the value of every node
 * under such a table.
 */

import { OmaiError } from '../../base/index.ts'
import type { GameTree, TabularPolicy } from './tree.ts'

/** The `currentPlayer` of a chance node. */
export const CHANCE = -1
/** The `currentPlayer` of a terminal state. */
export const TERMINAL = -4

/** The probabilities of every information set under a policy (uniform where it is silent). */
export function policyTable(tree: GameTree, policy: TabularPolicy): Float64Array[] {
  return tree.infosets.map((info) => {
    const p = policy[info.key]
    const k = info.actions.length
    if (p === undefined) return new Float64Array(k).fill(1 / k)
    if (p.length !== k) throw new OmaiError('shape', `policy for '${info.key}' needs ${k} probabilities`)
    return Float64Array.from(p)
  })
}

/** A tabular policy from per-information-set probabilities. */
export function tablePolicy(tree: GameTree, table: readonly ArrayLike<number>[]): TabularPolicy {
  const out: Record<string, number[]> = {}
  for (const [k, info] of tree.infosets.entries()) out[info.key] = Array.from(table[k] ?? [])
  return out
}

/** The value vector of every node under per-infoset probabilities (bottom-up over the preorder). */
export function nodeValues(tree: GameTree, table: readonly ArrayLike<number>[]): Float64Array[] {
  const n = tree.numPlayers
  const values = tree.nodes.map(() => new Float64Array(n))
  for (let id = tree.nodes.length - 1; id >= 0; id--) {
    const node = tree.nodes[id]
    const v = values[id]
    if (node === undefined || v === undefined) continue
    if (node.player === TERMINAL) {
      for (let p = 0; p < n; p++) v[p] = node.returns[p] ?? 0
      continue
    }
    const probs = node.player === CHANCE ? node.chance : (table[node.infoset] ?? [])
    for (const [a, c] of node.children.entries()) {
      const w = probs[a] ?? 0
      if (w === 0) continue
      const cv = values[c]
      for (let p = 0; p < n; p++) v[p] = (v[p] ?? 0) + w * (cv?.[p] ?? 0)
    }
  }
  return values
}
