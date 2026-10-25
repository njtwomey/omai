/**
 * Dominated strategies: actions that some other pure or mixed strategy beats against everything the others might do.
 * Tests for strict and weak dominance, iterated elimination of dominated strategies, restriction of a game to a subset
 * of actions, and pure-strategy security levels.
 */

import { MdmlError, maxOf, minOf } from '../../base/index.ts'
import { linprog } from '../../optim/index.ts'
import { actionRows, decodeProfile, profileCount, stridesOf } from './_profiles.ts'
import type { NormalFormGame } from './game.ts'

/** Options for the dominance tests. */
export interface DominanceOptions {
  /** Strict dominance (better against every profile of the others; the default) or weak (never worse, once better). */
  readonly strict?: boolean
  /** Allow dominance by mixed strategies, decided by a linear program (default false: pure strategies only). */
  readonly mixed?: boolean
  /** Margin a difference must exceed to count as better (default 1e-9). */
  readonly tol?: number
}

/** The payoffs of each of `player`'s actions against every profile of the others: `rows[a][s]`. */
function payoffRows(game: NormalFormGame, player: number): number[][] {
  const u = game.payoffs[player]
  if (u === undefined) throw new MdmlError('value', `${game.name}: no player ${player}`)
  return actionRows(game.shape, u, player)
}

function dominatedBy(rows: number[][], a: number, strict: boolean, mixed: boolean, tol: number): boolean {
  const target = rows[a] ?? []
  const others = rows.map((_, b) => b).filter((b) => b !== a)
  if (others.length === 0) return false
  if (!mixed) {
    return others.some((b) => {
      const r = rows[b] ?? []
      let better = false
      for (let s = 0; s < target.length; s++) {
        const d = (r[s] ?? 0) - (target[s] ?? 0)
        if (strict ? d <= tol : d < -tol) return false
        if (d > tol) better = true
      }
      return better
    })
  }
  const m = target.length
  const k = others.length
  if (strict) {
    // max e subject to sum_b sigma_b u(b, s) - e >= u(a, s) for every s, sigma in the simplex.
    const r = linprog({
      c: [...new Array<number>(k).fill(0), -1],
      aUb: Array.from({ length: m }, (_, s) => [...others.map((b) => -(rows[b]?.[s] ?? 0)), 1]),
      bUb: target.map((v) => -v),
      aEq: [[...new Array<number>(k).fill(1), 0]],
      bEq: [1],
      bounds: [...new Array<readonly [number, null]>(k).fill([0, null]), [null, null]],
    })
    return r.success && -r.fun > tol
  }
  // max sum_s (sum_b sigma_b u(b, s) - u(a, s)) subject to every difference >= 0.
  const r = linprog({
    c: others.map((b) => {
      let s = 0
      for (const v of rows[b] ?? []) s -= v
      return s
    }),
    aUb: Array.from({ length: m }, (_, s) => others.map((b) => -(rows[b]?.[s] ?? 0))),
    bUb: target.map((v) => -v),
    aEq: [new Array<number>(k).fill(1)],
    bEq: [1],
  })
  if (!r.success) return false
  let base = 0
  for (const v of target) base += v
  return -r.fun - base > tol
}

/**
 * Whether one of `player`'s actions is dominated: some other pure (or, with `mixed`, mixed) strategy does strictly
 * better against every profile of the others (or, for weak dominance, never worse and once better).
 *
 * @param game - The game.
 * @param player - The player.
 * @param action - The action tested.
 * @param options - Strict or weak, pure or mixed, and the margin.
 * @returns Whether the action is dominated.
 * @concept dominance
 * @example
 * ```ts
 * import { bimatrixGame, isDominated } from 'mdml-engine/games'
 * // The middle row is beaten by no pure row, but by the mix of the other two.
 * const g = bimatrixGame([[3, 0], [1, 1], [0, 3]], [[0, 0], [0, 0], [0, 0]])
 * [isDominated(g, 0, 1), isDominated(g, 0, 1, { mixed: true })] // => [false, true]
 * ```
 */
export function isDominated(
  game: NormalFormGame,
  player: number,
  action: number,
  options: DominanceOptions = {},
): boolean {
  const rows = payoffRows(game, player)
  if (action < 0 || action >= rows.length) throw new MdmlError('value', `${game.name}: no action ${action}`)
  return dominatedBy(rows, action, options.strict ?? true, options.mixed ?? false, options.tol ?? 1e-9)
}

/**
 * Every dominated action of `player`.
 *
 * @param game - The game.
 * @param player - The player.
 * @param options - Strict or weak, pure or mixed, and the margin.
 * @returns The dominated actions, ascending.
 * @concept dominance
 * @example
 * ```ts
 * import { dominatedActions, prisonersDilemma } from 'mdml-engine/games'
 * dominatedActions(prisonersDilemma(), 0) // => [0]
 * ```
 */
export function dominatedActions(game: NormalFormGame, player: number, options: DominanceOptions = {}): number[] {
  const rows = payoffRows(game, player)
  const out: number[] = []
  for (let a = 0; a < rows.length; a++)
    if (dominatedBy(rows, a, options.strict ?? true, options.mixed ?? false, options.tol ?? 1e-9)) out.push(a)
  return out
}

/**
 * The subgame where each player keeps only the listed actions (in the order given), with their names.
 *
 * @param game - The game.
 * @param actions - The actions kept for each player.
 * @returns The restricted game.
 * @throws MdmlError with code `value` when a player keeps no action or an action does not exist.
 * @example
 * ```ts
 * import { restrictGame, rockPaperScissors } from 'mdml-engine/games'
 * restrictGame(rockPaperScissors(), [[0, 1], [2]]).actionNames // => [['rock', 'paper'], ['scissors']]
 * ```
 */
export function restrictGame(game: NormalFormGame, actions: readonly (readonly number[])[]): NormalFormGame {
  if (actions.length !== game.shape.length) throw new MdmlError('value', `${game.name}: one action list per player`)
  for (const [p, list] of actions.entries())
    if (list.length === 0 || list.some((a) => a < 0 || a >= (game.shape[p] ?? 0)))
      throw new MdmlError('value', `${game.name}: player ${p} must keep existing actions`)
  const shape = actions.map((l) => l.length)
  const strides = stridesOf(game.shape)
  const total = profileCount(shape)
  const payoffs = game.payoffs.map(() => new Float64Array(total))
  for (let s = 0; s < total; s++) {
    const profile = decodeProfile(s, shape)
    let idx = 0
    for (let p = 0; p < shape.length; p++) idx += (actions[p]?.[profile[p] ?? 0] ?? 0) * (strides[p] ?? 0)
    for (const [p, u] of payoffs.entries()) u[s] = game.payoffs[p]?.[idx] ?? 0
  }
  return {
    name: game.name,
    shape,
    payoffs,
    actionNames: actions.map((l, p) => l.map((a) => game.actionNames[p]?.[a] ?? String(a))),
  }
}

/** The outcome of {@link iteratedElimination}. */
export interface EliminationResult {
  /** The game that remains. */
  readonly game: NormalFormGame
  /** The surviving actions of each player, as indices into the original game. */
  readonly actions: readonly (readonly number[])[]
  /** Every removal in order: the round, the player and the original action index. */
  readonly eliminated: readonly { readonly round: number; readonly player: number; readonly action: number }[]
}

/**
 * Iterated elimination of dominated strategies: in each round, remove every dominated action of every player (judged
 * in the game that remains at the start of the round), until none is left. With strict dominance the result does not
 * depend on the order; with weak dominance it can.
 *
 * @param game - The game.
 * @param options - Strict or weak, pure or mixed, and the margin.
 * @returns The reduced game, the surviving actions and the removals.
 * @concept dominance
 * @example
 * ```ts
 * import { iteratedElimination, prisonersDilemma } from 'mdml-engine/games'
 * iteratedElimination(prisonersDilemma()).actions // => [[1], [1]]
 * ```
 */
export function iteratedElimination(game: NormalFormGame, options: DominanceOptions = {}): EliminationResult {
  let actions = game.shape.map((k) => Array.from({ length: k }, (_, a) => a))
  const eliminated: { round: number; player: number; action: number }[] = []
  let current = game
  for (let round = 1; round <= profileCount(game.shape) + 1; round++) {
    const drop = current.shape.map((_, p) => dominatedActions(current, p, options))
    if (drop.every((d) => d.length === 0)) break
    for (const [p, d] of drop.entries())
      for (const a of d) eliminated.push({ round, player: p, action: actions[p]?.[a] ?? a })
    const keep = current.shape.map((k, p) =>
      Array.from({ length: k }, (_, a) => a).filter((a) => !(drop[p] ?? []).includes(a)),
    )
    actions = keep.map((l, p) => l.map((a) => actions[p]?.[a] ?? a))
    current = restrictGame(current, keep)
  }
  return { game: current, actions, eliminated }
}

/**
 * The pure-strategy security (maxmin) level of a player: the best payoff it can guarantee with a pure action whatever
 * the others do, and the actions achieving it.
 *
 * @param game - The game.
 * @param player - The player.
 * @returns The level and the maxmin actions.
 * @example
 * ```ts
 * import { pureMaxmin, stagHunt } from 'mdml-engine/games'
 * pureMaxmin(stagHunt(), 0) // => { value: 2, actions: [1] }
 * ```
 */
export function pureMaxmin(game: NormalFormGame, player: number): { value: number; actions: number[] } {
  const worst = payoffRows(game, player).map((r) => minOf(r))
  const value = maxOf(worst)
  return { value, actions: worst.flatMap((w, a) => (w === value ? [a] : [])) }
}
