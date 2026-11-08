/**
 * Two-player zero-sum games solved by linear programming through von Neumann's minimax theorem: the row and column
 * players' optimal mixed strategies and the game value, the equilibrium of a constant-sum game, and any player's
 * maximin security strategy in a general finite game.
 */

import { maxOf, minOf, OmaiError } from '../../base/index.ts'
import { type LinprogOptions, linprog } from '../../optim/index.ts'
import { asTensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { actionRows } from './_profiles.ts'
import { type Equilibrium, isConstantSum, type NormalFormGame, payoffMatrices } from './game.ts'

/** The solution of a zero-sum matrix game. */
export interface ZeroSumSolution {
  /** The maximising row player's optimal mixed strategy. */
  readonly row: readonly number[]
  /** The minimising column player's optimal mixed strategy. */
  readonly column: readonly number[]
  /** The value of the game, `row^T M column`. */
  readonly value: number
  /** The duality gap `max_i (M column)_i - min_j (row^T M)_j` (zero up to rounding). */
  readonly gap: number
  /** Simplex pivots taken by the two linear programs. */
  readonly iterations: number
}

/**
 * The maximin strategy and value of a player facing every column of `rows[a][s]` (its payoff for action a against
 * opponent choice s): `max_x min_s sum_a x_a rows[a][s]`, as the linear program `max v` subject to `rows^T x >= v`,
 * `x` in the simplex.
 */
function maximinLp(
  rows: readonly (readonly number[])[],
  options: LinprogOptions,
): { x: number[]; v: number; iterations: number } {
  const k = rows.length
  const cols = rows[0]?.length ?? 0
  const r = linprog(
    {
      c: [...new Array<number>(k).fill(0), -1],
      aUb: Array.from({ length: cols }, (_, s) => [...rows.map((row) => -(row[s] ?? 0)), 1]),
      bUb: new Array<number>(cols).fill(0),
      aEq: [[...new Array<number>(k).fill(1), 0]],
      bEq: [1],
      bounds: [...new Array<readonly [number, null]>(k).fill([0, null]), [null, null]],
    },
    options,
  )
  if (!r.success) throw new OmaiError('convergence', `maximin: the linear program ended with '${r.status}'`)
  const z = toFlat(r.x)
  const x = z.slice(0, k).map((p) => Math.max(0, p))
  let s = 0
  for (const p of x) s += p
  return { x: x.map((p) => p / s), v: z[k] ?? 0, iterations: r.iterations }
}

/**
 * Solve a two-player zero-sum matrix game `max_x min_y x^T M y` by linear programming (von Neumann's minimax theorem):
 * one program for the row player's maximin strategy and one for the column player's minimax strategy, solved with
 * `optim.linprog` (as `scipy.optimize.linprog` would be).
 *
 * @param matrix - The row player's payoffs, m x n (the column player gets the negation).
 * @param options - Options for the linear programs.
 * @returns Optimal strategies, the value and the duality gap.
 * @throws OmaiError with code `shape` for an empty or non-matrix input.
 * @concept minimax-theorem
 * @example
 * ```ts
 * import { solveZeroSum } from 'omai-engine/games'
 * const r = solveZeroSum([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
 * [r.row.map((p) => Math.round(p * 1e9) / 1e9), Math.round(r.value * 1e9) / 1e9] // => [[0.333333333, 0.333333333, 0.333333333], 0]
 * ```
 */
export function solveZeroSum(matrix: TensorLike, options: LinprogOptions = {}): ZeroSumSolution {
  const t = asTensor(matrix)
  if (t.shape.length !== 2 || (t.shape[0] ?? 0) === 0 || (t.shape[1] ?? 0) === 0)
    throw new OmaiError('shape', 'solveZeroSum: the payoffs must be a non-empty matrix')
  const [m, n] = t.shape as [number, number]
  const flat = toFlat(t)
  const a = Array.from({ length: m }, (_, i) => flat.slice(i * n, (i + 1) * n))
  const row = maximinLp(a, options)
  // The column player maximises -M^T y.
  const col = maximinLp(
    Array.from({ length: n }, (_, j) => a.map((r) => -(r[j] ?? 0))),
    options,
  )
  const rowValues = a.map((r) => r.reduce((s, v, j) => s + v * (col.x[j] ?? 0), 0))
  const colValues = Array.from({ length: n }, (_, j) => a.reduce((s, r, i) => s + (r[j] ?? 0) * (row.x[i] ?? 0), 0))
  let value = 0
  for (let i = 0; i < m; i++) value += (row.x[i] ?? 0) * (rowValues[i] ?? 0)
  return {
    row: row.x,
    column: col.x,
    value,
    gap: maxOf(rowValues) - minOf(colValues),
    iterations: row.iterations + col.iterations,
  }
}

/**
 * A Nash equilibrium of a two-player constant-sum game, from {@link solveZeroSum} on the row player's payoffs. In
 * such games every equilibrium gives the same payoffs and the equilibrium strategies are interchangeable.
 *
 * @param game - A two-player constant-sum game.
 * @param options - Options for the linear programs.
 * @returns The equilibrium (payoffs `[value, constant - value]`).
 * @throws OmaiError with code `value` when the game is not two-player constant-sum.
 * @concept minimax-theorem
 * @example
 * ```ts
 * import { matchingPennies, zeroSumEquilibrium } from 'omai-engine/games'
 * zeroSumEquilibrium(matchingPennies()).strategies // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function zeroSumEquilibrium(game: NormalFormGame, options: LinprogOptions = {}): Equilibrium {
  if (game.shape.length !== 2 || !isConstantSum(game, 1e-9))
    throw new OmaiError('value', `${game.name}: zeroSumEquilibrium needs a two-player constant-sum game`)
  const [a] = payoffMatrices(game)
  const r = solveZeroSum(a, options)
  const constant = (game.payoffs[0]?.[0] ?? 0) + (game.payoffs[1]?.[0] ?? 0)
  return { strategies: [r.row, r.column], payoffs: [r.value, constant - r.value] }
}

/**
 * A player's security (maxmin) strategy in any finite game: the mixed strategy maximising its worst-case expected
 * payoff against every joint pure profile of the others, by linear programming. In a two-player zero-sum game this is
 * the equilibrium strategy and the level is the value.
 *
 * @param game - The game.
 * @param player - The player.
 * @param options - Options for the linear program.
 * @returns The strategy and the guaranteed level.
 * @throws OmaiError with code `value` for a missing player.
 * @concept minimax-theorem
 * @example
 * ```ts
 * import { battleOfTheSexes, maximin } from 'omai-engine/games'
 * const r = maximin(battleOfTheSexes(), 0)
 * [r.strategy.map((p) => Math.round(p * 1e9) / 1e9), Math.round(r.value * 1e9) / 1e9] // => [[0.4, 0.6], 1.2]
 * ```
 */
export function maximin(
  game: NormalFormGame,
  player: number,
  options: LinprogOptions = {},
): { strategy: number[]; value: number } {
  const u = game.payoffs[player]
  if (u === undefined) throw new OmaiError('value', `${game.name}: no player ${player}`)
  const r = maximinLp(actionRows(game.shape, u, player), options)
  return { strategy: r.x, value: r.v }
}
