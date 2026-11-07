// Textbook search problems for the search tests, mirroring the oracle's port of aima-python
// (oracle/mdml_oracle/aima_search.py). Not published.

import { randint } from '../lib/random/index.ts'
import {
  allDifferent,
  binaryConstraint,
  type Csp,
  type CspConstraint,
  constraint,
  csp,
  type Game,
  type LocalSearchProblem,
  type SearchProblem,
} from '../lib/search/index.ts'

/** aima-python's tie-break: compare states as Python compares strings and tuples. */
export function pythonOrder(a: unknown, b: unknown): number {
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0
  if (typeof a === 'number' && typeof b === 'number') return a - b
  if (Array.isArray(a) && Array.isArray(b)) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const c = pythonOrder(a[i], b[i])
      if (c !== 0) return c
    }
    return a.length - b.length
  }
  return 0
}

/** A road map as the oracle stores it: each city with its neighbours in order and their road lengths. */
export type RoadMap = readonly (readonly [string, readonly (readonly [string, number])[]])[]

/** aima-python's GraphProblem on a road map with a heuristic table (default 0). */
export function roadProblem(
  map: RoadMap,
  start: string,
  goal: string,
  h: Readonly<Record<string, number>> = {},
): SearchProblem<string, string> & { readonly goal: string } {
  const roads = new Map(map.map(([c, nbrs]) => [c, new Map(nbrs.map(([b, d]) => [b, d]))]))
  return {
    initial: start,
    goal,
    actions: (s) => [...(roads.get(s)?.keys() ?? [])],
    result: (_s, a) => a,
    isGoal: (s) => s === goal,
    stepCost: (s, _a, t) => roads.get(s)?.get(t) ?? Number.POSITIVE_INFINITY,
    heuristic: (s) => h[s] ?? 0,
  }
}

const GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0]

/** aima-python's EightPuzzle: blank moves UP, DOWN, LEFT, RIGHT; misplaced squares (blank included) or Manhattan. */
export function eightPuzzle(
  tiles: readonly number[],
  heuristic: 'misplaced' | 'manhattan' | 'none' = 'none',
): SearchProblem<readonly number[], string> {
  const delta: Record<string, number> = { UP: -3, DOWN: 3, LEFT: -1, RIGHT: 1 }
  return {
    initial: tiles,
    actions: (s) => {
      const b = s.indexOf(0)
      return ['UP', 'DOWN', 'LEFT', 'RIGHT'].filter(
        (m) =>
          !(
            (m === 'LEFT' && b % 3 === 0) ||
            (m === 'UP' && b < 3) ||
            (m === 'RIGHT' && b % 3 === 2) ||
            (m === 'DOWN' && b > 5)
          ),
      )
    },
    result: (s, a) => {
      const t = [...s]
      const b = t.indexOf(0)
      const n = b + (delta[a] ?? 0)
      t[b] = t[n] ?? 0
      t[n] = 0
      return t
    },
    isGoal: (s) => s.every((x, i) => x === GOAL[i]),
    heuristic: (s) => {
      if (heuristic === 'misplaced') return s.filter((x, i) => x !== GOAL[i]).length
      if (heuristic === 'none') return 0
      let total = 0
      s.forEach((x, i) => {
        if (x === 0) return
        const j = GOAL.indexOf(x)
        total += Math.abs(Math.floor(i / 3) - Math.floor(j / 3)) + Math.abs((i % 3) - (j % 3))
      })
      return total
    },
    key: (s) => s.join(','),
  }
}

/** Whether two queens attack each other. */
export function queensConflict(r1: number, c1: number, r2: number, c2: number): boolean {
  return r1 === r2 || c1 === c2 || r1 - c1 === r2 - c2 || r1 + c1 === r2 + c2
}

/** aima-python's incremental NQueensProblem: fill columns left to right with rows not attacked. */
export function nQueensProblem(n: number): SearchProblem<readonly number[], number> {
  const conflicted = (s: readonly number[], row: number, col: number): boolean => {
    for (let c = 0; c < col; c++) if (queensConflict(row, col, s[c] ?? -1, c)) return true
    return false
  }
  return {
    initial: Array.from({ length: n }, () => -1),
    actions: (s) => {
      if (s[n - 1] !== -1) return []
      const col = s.indexOf(-1)
      return Array.from({ length: n }, (_, r) => r).filter((r) => !conflicted(s, r, col))
    },
    result: (s, row) => {
      const t = [...s]
      t[t.indexOf(-1)] = row
      return t
    },
    isGoal: (s) => s[n - 1] !== -1 && s.every((r, c) => !conflicted(s, r, c)),
    key: (s) => s.join(','),
  }
}

/** A tic-tac-toe or connect-four position as aima-python's GameState: mover, utility for X, board and free squares. */
export interface BoardState {
  readonly toMove: 'X' | 'O'
  readonly utility: number
  readonly board: ReadonlyMap<string, 'X' | 'O'>
  readonly moves: readonly (readonly [number, number])[]
}

/** A square (x, y), counted from 1. */
export type Square = readonly [number, number]

/** aima-python's TicTacToe(h, v, k), or ConnectFour when `gravity` is set: squares (x, y) from 1, X first. */
export function boardGame(
  h = 3,
  v = 3,
  k = 3,
  gravity = false,
): Game<BoardState, Square> & { readonly play: (moves: readonly Square[]) => BoardState } {
  const at = (x: number, y: number): string => `${x},${y}`
  const inRow = (board: ReadonlyMap<string, string>, m: Square, p: string, dx: number, dy: number): boolean => {
    let n = 0
    for (let x = m[0], y = m[1]; board.get(at(x, y)) === p; x += dx, y += dy) n += 1
    for (let x = m[0], y = m[1]; board.get(at(x, y)) === p; x -= dx, y -= dy) n += 1
    return n - 1 >= k
  }
  const moves: Square[] = []
  for (let x = 1; x <= h; x++) for (let y = 1; y <= v; y++) moves.push([x, y])
  const initial: BoardState = { toMove: 'X', utility: 0, board: new Map(), moves }
  const directions: readonly Square[] = [
    [0, 1],
    [1, 0],
    [1, -1],
    [1, 1],
  ]
  const result = (s: BoardState, m: Square): BoardState => {
    const board = new Map(s.board)
    board.set(at(m[0], m[1]), s.toMove)
    const win = directions.some(([dx, dy]) => inRow(board, m, s.toMove, dx, dy))
    return {
      toMove: s.toMove === 'X' ? 'O' : 'X',
      utility: win ? (s.toMove === 'X' ? 1 : -1) : 0,
      board,
      moves: s.moves.filter((x) => x[0] !== m[0] || x[1] !== m[1]),
    }
  }
  return {
    initial,
    player: (s) => s.toMove,
    actions: (s) => (gravity ? s.moves.filter(([x, y]) => y === 1 || s.board.has(at(x, y - 1))) : s.moves),
    result,
    isTerminal: (s) => s.utility !== 0 || s.moves.length === 0,
    utility: (s, p) => (p === 'X' ? s.utility : 0 - s.utility),
    key: (s) => `${[...s.board.entries()].sort().join(';')}${s.toMove}`,
    play: (ms) => ms.reduce(result, initial),
  }
}

/** The number of attacking pairs of queens, one per column, rows given. */
export function attackingPairs(rows: readonly number[]): number {
  let count = 0
  for (let a = 0; a < rows.length; a++) {
    for (let b = a + 1; b < rows.length; b++) if (queensConflict(rows[a] ?? 0, a, rows[b] ?? 0, b)) count += 1
  }
  return count
}

/** Complete-state n-queens for local search (AIMA 4.1.1): move one queen within its column; value is minus the attacks. */
export function queensLocal(n: number, initial?: readonly number[]): LocalSearchProblem<readonly number[]> {
  return {
    ...(initial === undefined ? {} : { initial }),
    random: (k) => Array.from(randint(k, [n], 0, n).data),
    neighbours: (s) => {
      const out: number[][] = []
      for (let c = 0; c < n; c++)
        for (let r = 0; r < n; r++) if (r !== s[c]) out.push(s.map((x, i) => (i === c ? r : x)))
      return out
    },
    value: (s) => -attackingPairs(s),
    isGoal: (s) => attackingPairs(s) === 0,
    key: (s) => s.join(','),
  }
}

/** n-queens as a CSP:variables '0' .. 'n-1' (columns) valued by row, with an optional fixed first queen. */
export function queensCsp(n: number, firstRow?: number): Csp<number> {
  const cols = Array.from({ length: n }, (_, i) => String(i))
  const rows = Array.from({ length: n }, (_, i) => i)
  const domains = Object.fromEntries(cols.map((c) => [c, c === '0' && firstRow !== undefined ? [firstRow] : rows]))
  const constraints = cols.flatMap((a, i) =>
    cols.slice(i + 1).map((b, k) => binaryConstraint<number>(a, b, (r, s) => r !== s && Math.abs(r - s) !== k + 1)),
  )
  return csp(cols, domains, constraints)
}

/** Map colouring as a CSP. */
export function mapColouring(
  regions: readonly string[],
  borders: readonly (readonly [string, string])[],
  colours: readonly string[],
  fixed: Readonly<Record<string, string>> = {},
): Csp<string> {
  const domains = Object.fromEntries(regions.map((r) => [r, r in fixed ? [fixed[r] as string] : [...colours]]))
  return csp(
    regions,
    domains,
    borders.map(([a, b]) => binaryConstraint<string>(a, b, (x, y) => x !== y)),
  )
}

/** SEND + MORE = MONEY in column form with carries C1 .. C3, all letters different, S and M non-zero. */
export function sendMoreMoney(): Csp<number> {
  const letters = ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']
  const digits = Array.from({ length: 10 }, (_, i) => i)
  const domains: Record<string, number[]> = Object.fromEntries(letters.map((l) => [l, digits]))
  for (const c of ['C1', 'C2', 'C3']) domains[c] = [0, 1]
  const sum = (scope: string[], f: (v: readonly number[]) => boolean) => constraint<number>(scope, f)
  return csp([...letters, 'C1', 'C2', 'C3'], domains, [
    ...allDifferent<number>(letters),
    constraint<number>(['S'], (v) => v[0] !== 0),
    constraint<number>(['M'], (v) => v[0] !== 0),
    sum(['D', 'E', 'Y', 'C1'], ([d = 0, e = 0, y = 0, c = 0]) => d + e === y + 10 * c),
    sum(['N', 'R', 'C1', 'E', 'C2'], ([n = 0, r = 0, c1 = 0, e = 0, c2 = 0]) => n + r + c1 === e + 10 * c2),
    sum(['E', 'O', 'C2', 'N', 'C3'], ([e = 0, o = 0, c2 = 0, n = 0, c3 = 0]) => e + o + c2 === n + 10 * c3),
    sum(['S', 'M', 'C3', 'O'], ([s = 0, m = 0, c3 = 0, o = 0]) => s + m + c3 === o + 10 * m),
  ])
}

/** A Sudoku of side 4 or 9 as a CSP over cells '0' .. (side^2 - 1), with all-different rows, columns and boxes. */
export function sudokuCsp(grid: string, side: 4 | 9 = 9): Csp<number> {
  const box = side === 4 ? 2 : 3
  const cells = Array.from({ length: side * side }, (_, i) => String(i))
  const digits = Array.from({ length: side }, (_, i) => i + 1)
  const domains = Object.fromEntries(
    cells.map((c, i) => {
      const ch = grid[i] ?? '.'
      return [c, ch === '.' ? digits : [Number(ch)]]
    }),
  )
  const units: string[][] = []
  for (let r = 0; r < side; r++) units.push(digits.map((_, c) => String(side * r + c)))
  for (let c = 0; c < side; c++) units.push(digits.map((_, r) => String(side * r + c)))
  for (let br = 0; br < box; br++) {
    for (let bc = 0; bc < box; bc++) {
      const unit: string[] = []
      for (let r = 0; r < box; r++)
        for (let c = 0; c < box; c++) unit.push(String(side * (box * br + r) + box * bc + c))
      units.push(unit)
    }
  }
  const seen = new Set<string>()
  const pairs: CspConstraint<number>[] = []
  for (const unit of units) {
    for (const p of allDifferent<number>(unit)) {
      const id = [...p.scope].sort().join(',')
      if (seen.has(id)) continue
      seen.add(id)
      pairs.push(p)
    }
  }
  return csp(cells, domains, pairs)
}
