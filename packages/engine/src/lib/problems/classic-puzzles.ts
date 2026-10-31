/**
 * Classic state-space puzzles with the facts known about them: the sliding-tile puzzles (the 8-puzzle and 15-puzzle,
 * with the parity test that says which boards are solvable, scrambles that are solvable by construction and the
 * Manhattan-distance heuristic) and the Towers of Hanoi (whose shortest solution has exactly `2^n - 1` moves). Each is a
 * `SearchProblem` the algorithms of `search` run on.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { aStarSearch, type SearchProblem } from '../search/index.ts'

/** A sliding-tile move: the direction the blank moves. */
export type TileMove = 'up' | 'down' | 'left' | 'right'

/** A board of an `n x n` sliding puzzle, row by row, with `0` the blank. */
export type TileBoard = readonly number[]

/**
 * The solved board of the `n x n` sliding puzzle: the tiles 1 to `n^2 - 1` in order and the blank last.
 *
 * @param n - The side (3 for the 8-puzzle, 4 for the 15-puzzle).
 * @returns The board.
 * @example
 * ```ts
 * import { solvedBoard } from 'mdml-engine/problems'
 * solvedBoard(3) // => [1, 2, 3, 4, 5, 6, 7, 8, 0]
 * ```
 */
export function solvedBoard(n: number): TileBoard {
  if (!(Number.isInteger(n) && n >= 2)) throw new MdmlError('value', `solvedBoard: side ${n} must be an integer >= 2`)
  return Array.from({ length: n * n }, (_, i) => (i + 1) % (n * n))
}

/**
 * Whether a board can be solved. Half of the `(n^2)!` arrangements can: the parity of the permutation (inversions among
 * the tiles) decides on an odd side, and on an even side the parity of the blank's row from the bottom joins it, because
 * every move changes both together.
 *
 * @param board - The board.
 * @returns Whether some sequence of moves reaches the solved board.
 * @example
 * ```ts
 * import { isSolvable } from 'mdml-engine/problems'
 * [isSolvable([1, 2, 3, 4, 5, 6, 8, 7, 0]), isSolvable([1, 2, 3, 4, 5, 6, 7, 8, 0])] // => [false, true]
 * ```
 */
export function isSolvable(board: TileBoard): boolean {
  const n = Math.round(Math.sqrt(board.length))
  if (n * n !== board.length) throw new MdmlError('shape', 'isSolvable: the board is not square')
  const tiles = board.filter((t) => t !== 0)
  let inversions = 0
  for (let i = 0; i < tiles.length; i++)
    for (let j = i + 1; j < tiles.length; j++) if ((tiles[i] ?? 0) > (tiles[j] ?? 0)) inversions += 1
  if (n % 2 === 1) return inversions % 2 === 0
  const blankRowFromBottom = n - Math.floor(board.indexOf(0) / n)
  return (inversions + blankRowFromBottom) % 2 === 1
}

/** The Manhattan distance of every tile from its place: an admissible and consistent estimate of the moves left. */
function manhattan(board: TileBoard, n: number): number {
  let total = 0
  for (const [i, t] of board.entries()) {
    if (t === 0) continue
    const goal = t - 1
    total += Math.abs(Math.floor(i / n) - Math.floor(goal / n)) + Math.abs((i % n) - (goal % n))
  }
  return total
}

/**
 * The sliding-tile puzzle as a search problem: states are boards, actions move the blank, every move costs 1, and the
 * heuristic is the Manhattan distance. The 8-puzzle has 181,440 reachable boards and every solvable one needs at most
 * 31 moves.
 *
 * @param board - The start board (default solved 8-puzzle).
 * @returns The problem.
 * @throws MdmlError with code `value` for a board that is not a permutation of `0 .. n^2 - 1`.
 * @example
 * ```ts
 * import { slidingPuzzle } from 'mdml-engine/problems'
 * import { aStarSearch } from 'mdml-engine/search'
 * aStarSearch(slidingPuzzle([1, 2, 3, 4, 5, 6, 0, 7, 8])).cost // => 2
 * ```
 */
export function slidingPuzzle(board: TileBoard = solvedBoard(3)): SearchProblem<TileBoard, TileMove> {
  const n = Math.round(Math.sqrt(board.length))
  const goal = solvedBoard(n)
  if (n * n !== board.length || [...board].sort((a, b) => a - b).some((t, i) => t !== i))
    throw new MdmlError('value', 'slidingPuzzle: the board must be a permutation of 0 .. n^2 - 1')
  const deltas: Readonly<Record<TileMove, readonly [number, number]>> = {
    up: [-1, 0],
    down: [1, 0],
    left: [0, -1],
    right: [0, 1],
  }
  return {
    initial: board,
    actions: (b) => {
      const at = b.indexOf(0)
      const r = Math.floor(at / n)
      const c = at % n
      return (['up', 'down', 'left', 'right'] as const).filter((m) => {
        const [dr, dc] = deltas[m]
        return r + dr >= 0 && r + dr < n && c + dc >= 0 && c + dc < n
      })
    },
    result: (b, m) => {
      const at = b.indexOf(0)
      const [dr, dc] = deltas[m]
      const to = at + dr * n + dc
      const next = b.slice()
      next[at] = b[to] ?? 0
      next[to] = 0
      return next
    },
    isGoal: (b) => b.every((t, i) => t === goal[i]),
    heuristic: (b) => manhattan(b, n),
    key: (b) => b.join(','),
    label: (b) => b.join(' '),
  }
}

/** A scrambled board with what is known of it. */
export interface ScrambledPuzzle {
  readonly board: TileBoard
  /** The number of random moves applied (the optimal solution is never longer). */
  readonly moves: number
}

/**
 * A board reached from the solved one by `moves` random legal moves (never undoing the last): solvable by construction,
 * and no more than `moves` moves from solved.
 *
 * @param key - The key.
 * @param n - The side (default 3).
 * @param moves - The number of moves (default 20).
 * @returns The board and the number of moves.
 * @example
 * ```ts
 * import { isSolvable, scrambledBoard } from 'mdml-engine/problems'
 * import { key } from 'mdml-engine/random'
 * isSolvable(scrambledBoard(key(0), 4, 30).board) // => true
 * ```
 */
export function scrambledBoard(key: Key, n = 3, moves = 20): ScrambledPuzzle {
  const problem = slidingPuzzle(solvedBoard(n))
  let board = problem.initial
  let previous: TileBoard | undefined
  for (let t = 0; t < moves; t++) {
    const options = problem.actions(board).map((m) => problem.result(board, m))
    const fresh = options.filter((b) => previous === undefined || b.join() !== previous.join())
    const pool = fresh.length > 0 ? fresh : options
    const next =
      pool[Math.min(pool.length - 1, Math.floor((uniform(foldIn(key, t), []).data[0] ?? 0) * pool.length))] ?? board
    previous = board
    board = next
  }
  return { board, moves }
}

/**
 * The fewest moves that solve a board (A* with the Manhattan heuristic), or `undefined` when the board is not solvable.
 *
 * @param board - The board.
 * @returns The optimal number of moves.
 * @example
 * ```ts
 * import { optimalMoves } from 'mdml-engine/problems'
 * optimalMoves([1, 2, 3, 4, 5, 6, 0, 7, 8]) // => 2
 * ```
 */
export function optimalMoves(board: TileBoard): number | undefined {
  if (!isSolvable(board)) return undefined
  const r = aStarSearch(slidingPuzzle(board))
  return r.status === 'solved' ? r.cost : undefined
}

/** A state of the Towers of Hanoi: the pegs (0, 1, 2) each disc is on, smallest disc first. */
export type HanoiState = readonly number[]

/** A Hanoi move: the disc and the peg it goes to. */
export type HanoiMove = readonly [disc: number, peg: number]

/**
 * The Towers of Hanoi with `discs` discs, all on peg 0 and to be moved to peg 2, one disc at a time and never a larger
 * on a smaller. The shortest solution has exactly `2^discs - 1` moves.
 *
 * @param discs - The number of discs.
 * @returns The problem.
 * @example
 * ```ts
 * import { towersOfHanoi } from 'mdml-engine/problems'
 * import { breadthFirstSearch } from 'mdml-engine/search'
 * breadthFirstSearch(towersOfHanoi(4)).cost // => 15
 * ```
 */
export function towersOfHanoi(discs: number): SearchProblem<HanoiState, HanoiMove> {
  if (!(Number.isInteger(discs) && discs >= 1)) throw new MdmlError('value', `towersOfHanoi: ${discs} discs`)
  return {
    initial: new Array<number>(discs).fill(0),
    actions: (s) => {
      const out: HanoiMove[] = []
      for (let d = 0; d < discs; d++) {
        // A disc moves only if nothing smaller is on its peg, and only onto a peg whose smaller discs are none.
        if (s.slice(0, d).includes(s[d] ?? 0)) continue
        for (let p = 0; p < 3; p++) if (p !== s[d] && !s.slice(0, d).includes(p)) out.push([d, p])
      }
      return out
    },
    result: (s, [d, p]) => s.map((peg, i) => (i === d ? p : peg)),
    isGoal: (s) => s.every((p) => p === 2),
    key: (s) => s.join(''),
    label: (s) => s.join(''),
  }
}
