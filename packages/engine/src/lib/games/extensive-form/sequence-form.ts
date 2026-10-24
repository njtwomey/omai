import { WhiteboxError } from '../../base/index.ts'
import type { LinprogOptions } from '../../optim/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { solveLp } from '../_lp.ts'
import { CHANCE, TERMINAL, tablePolicy } from './_tree.ts'
import { type ExtensiveGame, type GameTree, type TabularPolicy, treeOf } from './tree.ts'

/** A sequence of one player: the empty sequence (index 0) or an action at one of its information sets. */
export interface Sequence {
  /** The information set (index into the tree's), or -1 for the empty sequence. */
  readonly infoset: number
  /** The action's position at the information set, or -1. */
  readonly action: number
  /** The sequence leading to the information set. */
  readonly parent: number
}

/** The sequence form of a two-player game (von Stengel 1996). */
export interface SequenceForm {
  /** Each player's sequences. */
  readonly sequences: readonly [readonly Sequence[], readonly Sequence[]]
  /** Each player's realisation-plan constraints `E x = e` (one row for the empty sequence, one per information set). */
  readonly constraints: readonly [
    { readonly matrix: readonly (readonly number[])[]; readonly rhs: readonly number[] },
    { readonly matrix: readonly (readonly number[])[]; readonly rhs: readonly number[] },
  ]
  /** Each player's payoffs by pair of sequences (chance probabilities folded in): `x^T A y` is player 0's return. */
  readonly payoffs: readonly [readonly (readonly number[])[], readonly (readonly number[])[]]
}

/**
 * The sequence form of a two-player game with perfect recall: each player's sequences (actions at its information
 * sets, with the sequence leading there), the realisation-plan constraints, and the payoff matrices over pairs of
 * sequences, with chance probabilities folded in. Its size is linear in the game tree, unlike the normal form.
 *
 * @param game - A two-player game or its tree.
 * @returns The sequences, constraints and payoffs.
 * @throws WhiteboxError with code `value` for other than two players or without perfect recall.
 * @concept sequence-form
 * @reference von Stengel (1996), Efficient computation of behavior strategies.
 * @example
 * ```ts
 * import { kuhnPoker, sequenceForm } from 'whitebox-engine/games'
 * sequenceForm(kuhnPoker()).sequences.map((s) => s.length) // => [13, 13]
 * ```
 */
export function sequenceForm<S>(game: ExtensiveGame<S> | GameTree): SequenceForm {
  const tree = treeOf(game)
  if (tree.numPlayers !== 2) throw new WhiteboxError('value', `${tree.name}: the sequence form needs two players`)
  const sequences: [Sequence[], Sequence[]] = [
    [{ infoset: -1, action: -1, parent: -1 }],
    [{ infoset: -1, action: -1, parent: -1 }],
  ]
  const first = new Int32Array(tree.infosets.length).fill(-1)
  const parentOf = new Int32Array(tree.infosets.length).fill(-1)
  // Preorder walk carrying each player's last sequence and the chance reach.
  const last = tree.nodes.map(() => [0, 0])
  const reach = new Float64Array(tree.nodes.length)
  reach[0] = 1
  const leaves: { id: number; seqs: number[] }[] = []
  for (const [id, node] of tree.nodes.entries()) {
    const seqs = last[id] ?? [0, 0]
    if (node.player === TERMINAL) {
      leaves.push({ id, seqs })
      continue
    }
    if (node.player === CHANCE) {
      for (const [a, c] of node.children.entries()) {
        last[c] = [...seqs]
        reach[c] = (reach[id] ?? 0) * (node.chance[a] ?? 0)
      }
      continue
    }
    const p = node.player
    const k = node.infoset
    const own = seqs[p] ?? 0
    if ((first[k] ?? -1) < 0) {
      parentOf[k] = own
      first[k] = sequences[p as 0 | 1].length
      for (const [a] of node.actions.entries()) sequences[p as 0 | 1].push({ infoset: k, action: a, parent: own })
    } else if (parentOf[k] !== own)
      throw new WhiteboxError('value', `${tree.name}: the game lacks perfect recall at '${tree.infosets[k]?.key}'`)
    for (const [a, c] of node.children.entries()) {
      const next = [...seqs]
      next[p] = (first[k] ?? 0) + a
      last[c] = next
      reach[c] = reach[id] ?? 0
    }
  }
  const constraintsOf = (p: 0 | 1): { matrix: number[][]; rhs: number[] } => {
    const seqs = sequences[p]
    const own = tree.infosets.flatMap((info, k) => (info.player === p ? [k] : []))
    const matrix: number[][] = [seqs.map((_, j) => (j === 0 ? 1 : 0))]
    for (const k of own) {
      const row = new Array<number>(seqs.length).fill(0)
      row[parentOf[k] ?? 0] = -1
      for (let a = 0; a < (tree.infosets[k]?.actions.length ?? 0); a++) row[(first[k] ?? 0) + a] = 1
      matrix.push(row)
    }
    return { matrix, rhs: matrix.map((_, r) => (r === 0 ? 1 : 0)) }
  }
  const constraints: SequenceForm['constraints'] = [constraintsOf(0), constraintsOf(1)]
  const payoffs = [0, 1].map(() => sequences[0].map(() => new Array<number>(sequences[1].length).fill(0))) as [
    number[][],
    number[][],
  ]
  for (const { id, seqs } of leaves) {
    const node = tree.nodes[id]
    const w = reach[id] ?? 0
    for (const p of [0, 1] as const) {
      const row = payoffs[p][seqs[0] ?? 0]
      if (row !== undefined) row[seqs[1] ?? 0] = (row[seqs[1] ?? 0] ?? 0) + w * (node?.returns[p] ?? 0)
    }
  }
  return { sequences, constraints, payoffs }
}

/** The outcome of {@link solveSequenceForm}. */
export interface SequenceFormSolution {
  /** Each player's security value (the game value for player 0 and its negation, in a zero-sum game). */
  readonly values: readonly [number, number]
  /** Each player's optimal realisation plan, one weight per sequence. */
  readonly plans: readonly [readonly number[], readonly number[]]
  /** The behaviour policy of both players derived from the plans (uniform where a plan gives no weight). */
  readonly policy: TabularPolicy
}

/**
 * Solve a two-player zero-sum game in sequence form by linear programming (Koller, Megiddo and von Stengel 1994):
 * each player maximises its worst-case return, `max_x min_y x^T A y` over realisation plans, as one LP by duality of
 * the inner problem; the plans are turned into behaviour policies. In a general-sum game the result is each player's
 * security strategy.
 *
 * @param game - A two-player game with perfect recall, or its tree.
 * @param options - Options for the linear programs.
 * @returns The values, realisation plans and behaviour policy.
 * @throws WhiteboxError with code `convergence` when a linear program fails.
 * @concept sequence-form
 * @example
 * ```ts
 * import { kuhnPoker, solveSequenceForm } from 'whitebox-engine/games'
 * Math.round(solveSequenceForm(kuhnPoker()).values[0] * 18 * 1e9) / 1e9 // => -1
 * ```
 */
export function solveSequenceForm<S>(
  game: ExtensiveGame<S> | GameTree,
  options: LinprogOptions = {},
): SequenceFormSolution {
  const tree = treeOf(game)
  const form = sequenceForm(tree)
  const solve = (p: 0 | 1): { plan: number[]; value: number } => {
    const o = (1 - p) as 0 | 1
    const own = form.constraints[p]
    const opp = form.constraints[o]
    const nOwn = form.sequences[p].length
    const nOpp = form.sequences[o].length
    const rows = opp.matrix.length
    // u(i, j): player p's payoff for its sequence i against the opponent's sequence j.
    const u = (i: number, j: number): number => (p === 0 ? form.payoffs[0][i]?.[j] : form.payoffs[1][j]?.[i]) ?? 0
    // max opp.rhs . q subject to opp.matrix^T q <= U^T x (one row per opponent sequence), own.matrix x = own.rhs.
    const r = solveLp(
      {
        c: [...new Array<number>(nOwn).fill(0), ...opp.rhs.map((v) => -v)],
        aUb: Array.from({ length: nOpp }, (_, j) => [
          ...Array.from({ length: nOwn }, (_, i) => -u(i, j)),
          ...Array.from({ length: rows }, (_, r2) => opp.matrix[r2]?.[j] ?? 0),
        ]),
        bUb: new Array<number>(nOpp).fill(0),
        aEq: own.matrix.map((row) => [...row, ...new Array<number>(rows).fill(0)]),
        bEq: [...own.rhs],
        bounds: [
          ...new Array<readonly [number, null]>(nOwn).fill([0, null]),
          ...new Array<readonly [null, null]>(rows).fill([null, null]),
        ],
      },
      options,
    )
    if (!r.success)
      throw new WhiteboxError('convergence', `solveSequenceForm: the linear program ended with '${r.status}'`)
    return {
      plan: toFlat(r.x)
        .slice(0, nOwn)
        .map((v) => Math.max(0, v)),
      value: -r.fun,
    }
  }
  const a = solve(0)
  const b = solve(1)
  const plans: [number[], number[]] = [a.plan, b.plan]
  const table = tree.infosets.map((info, k) => {
    const p = info.player as 0 | 1
    const seqs = form.sequences[p]
    const start = seqs.findIndex((s) => s.infoset === k)
    const parent = seqs[start]?.parent ?? 0
    const mass = plans[p][parent] ?? 0
    const m = info.actions.length
    if (mass <= 1e-12) return new Float64Array(m).fill(1 / m)
    const w = Float64Array.from({ length: m }, (_, i) => Math.max(0, plans[p][start + i] ?? 0))
    let total = 0
    for (const v of w) total += v
    return total > 0 ? w.map((v) => v / total) : new Float64Array(m).fill(1 / m)
  })
  return { values: [a.value, b.value], plans, policy: tablePolicy(tree, table) }
}
