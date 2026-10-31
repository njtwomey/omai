/**
 * The tagged registry of search problems: every puzzle, tour and satisfiability family of this domain as one entry with
 * its kind, whether its instances are satisfiable, the size knob, and a `make` that builds an instance together with the
 * truth known about it (the optimal depth of a puzzle, the optimal tour length, the planted assignment, the verdict).
 * `whereProblems` answers a selector's queries.
 */

import { find, type Json, matchesText, type Registry, registry } from '../base/index.ts'
import { foldIn, type Key, key as makeKey } from '../random/index.ts'
import { optimalMoves, scrambledBoard, slidingPuzzle, solvedBoard, towersOfHanoi } from './classic-puzzles.ts'
import { nQueens, SUDOKUS, sudoku } from './puzzles.ts'
import { pigeonhole, plantedKSat, randomKSat, SAT_THRESHOLDS } from './sat-instances.ts'
import { circleTour, gridTour, plantedHamiltonian, randomTour } from './tours.ts'

/** What kind of problem an entry is. */
export type ProblemKind = 'puzzle' | 'constraint' | 'tour' | 'satisfiability'

/** Whether the instances of an entry have solutions. */
export type Solvability = 'always' | 'never' | 'varies'

/** An instance with the truth known about it. */
export interface ProblemInstance {
  /** The problem value: a `SearchProblem`, a `Csp`, a `Cnf`, a tour instance or a graph. */
  readonly value: unknown
  /** What is known of it: the optimal cost, the planted solution, the verdict. */
  readonly truth: Readonly<Record<string, Json>>
}

/** One registered problem family. */
export interface ProblemEntry {
  readonly name: string
  readonly title: string
  readonly description: string
  readonly kind: ProblemKind
  readonly solvable: Solvability
  /** What `size` means, and its default. */
  readonly sizeMeaning: string
  readonly defaultSize: number
  make(options?: { readonly size?: number; readonly key?: Key }): ProblemInstance
}

/** A registry of problem families. */
export type ProblemRegistry = Registry<ProblemEntry>

const row = (
  name: string,
  title: string,
  kind: ProblemKind,
  solvable: Solvability,
  sizeMeaning: string,
  defaultSize: number,
  description: string,
  make: (size: number, key: Key) => ProblemInstance,
): ProblemEntry => ({
  name,
  title,
  description,
  kind,
  solvable,
  sizeMeaning,
  defaultSize,
  make: (o) => make(o?.size ?? defaultSize, o?.key ?? makeKey(0)),
})

/**
 * The library's registry of search problems, tagged.
 *
 * @example
 * ```ts
 * import { problemRegistry, whereProblems } from 'mdml-engine/problems'
 * whereProblems(problemRegistry, { kind: 'satisfiability' }).map((e) => e.name) // => ['plantedKSat', 'randomKSat', 'pigeonhole']
 * ```
 */
export const problemRegistry: ProblemRegistry = registry<ProblemEntry>([
  row(
    'slidingPuzzle',
    'Sliding-tile puzzle',
    'puzzle',
    'always',
    'the side of the board',
    3,
    'A board scrambled by random moves from solved, with its optimal solution length.',
    (n, k) => {
      const { board, moves } = scrambledBoard(k, n, n === 3 ? 22 : 12)
      return {
        value: slidingPuzzle(board),
        truth: {
          board: [...board],
          scrambleMoves: moves,
          optimalMoves: n === 3 ? (optimalMoves(board) ?? null) : null,
          solvedBoard: [...solvedBoard(n)],
        },
      }
    },
  ),
  row(
    'towersOfHanoi',
    'Towers of Hanoi',
    'puzzle',
    'always',
    'the number of discs',
    4,
    'Move the discs from the first peg to the last; the shortest solution has 2^n - 1 moves.',
    (n) => ({ value: towersOfHanoi(n), truth: { optimalMoves: 2 ** n - 1 } }),
  ),
  row(
    'nQueens',
    'N queens',
    'constraint',
    'varies',
    'the board size',
    8,
    'Place n queens so that none attacks another; solvable for every n except 2 and 3.',
    (n) => ({ value: nQueens(n), truth: { solvable: n !== 2 && n !== 3 } }),
  ),
  row(
    'sudoku',
    'Sudoku',
    'constraint',
    'always',
    'the index of the grid (0 mini, 1 easy, 2 hard)',
    1,
    'A grid with a unique solution; the easy one yields to propagation, the hard one needs search.',
    (i) => {
      const grid = Object.values(SUDOKUS)[Math.min(2, Math.max(0, i))] as string
      return { value: sudoku(grid), truth: { grid } }
    },
  ),
  row(
    'circleTour',
    'Circle tour',
    'tour',
    'always',
    'the number of cities',
    12,
    'Cities on a circle; the optimal tour is the polygon, of length 2 n sin(pi / n).',
    (n) => {
      const t = circleTour(n)
      return { value: t, truth: { optimalLength: t.optimalLength, optimalTour: [...t.optimalTour] } }
    },
  ),
  row(
    'gridTour',
    'Grid tour',
    'tour',
    'always',
    'the side of a square lattice (even)',
    4,
    'Cities on a lattice; the optimal tour has unit steps and length n^2.',
    (n) => {
      const t = gridTour(n, n)
      return { value: t, truth: { optimalLength: t.optimalLength, optimalTour: [...t.optimalTour] } }
    },
  ),
  row(
    'randomTour',
    'Random tour',
    'tour',
    'always',
    'the number of cities (13 or fewer for the exact optimum)',
    10,
    'Cities in the unit square, the optimum by the Held-Karp dynamic programme.',
    (n, k) => {
      const t = randomTour(k, n)
      return { value: t, truth: { optimalLength: t.optimalLength, optimalTour: [...t.optimalTour] } }
    },
  ),
  row(
    'plantedHamiltonian',
    'Planted Hamiltonian cycle',
    'tour',
    'always',
    'the number of vertices',
    12,
    'A graph with a Hamiltonian cycle hidden among decoy edges.',
    (n, k) => {
      const p = plantedHamiltonian(k, n)
      return { value: p, truth: { cycle: [...p.planted] } }
    },
  ),
  row(
    'plantedKSat',
    'Planted 3-SAT',
    'satisfiability',
    'always',
    'the number of variables (ratio 4.3)',
    40,
    'Clauses all satisfied by a hidden assignment.',
    (n, k) => {
      const p = plantedKSat(k, n, SAT_THRESHOLDS[3])
      return { value: p.formula, truth: { satisfiable: true, assignment: [...p.assignment] } }
    },
  ),
  row(
    'randomKSat',
    'Random 3-SAT',
    'satisfiability',
    'varies',
    'the number of variables (ratio at the threshold, 4.27)',
    40,
    'Uniform random clauses; around the threshold ratio the verdict is a coin flip.',
    (n, k) => ({
      value: randomKSat(foldIn(k, 0), n, SAT_THRESHOLDS[3]),
      truth: { thresholdRatio: SAT_THRESHOLDS[3] },
    }),
  ),
  row(
    'pigeonhole',
    'Pigeonhole',
    'satisfiability',
    'never',
    'the number of holes (one more pigeon)',
    5,
    'n + 1 pigeons in n holes: unsatisfiable, and hard for resolution.',
    (n) => ({ value: pigeonhole(n), truth: { satisfiable: false } }),
  ),
])

/** A query over a problem registry: every given filter must hold. */
export interface ProblemQuery {
  readonly kind?: ProblemKind | readonly ProblemKind[]
  readonly solvable?: Solvability | readonly Solvability[]
  /** Search-box text: every word must appear in the name, title or description. */
  readonly text?: string
}

const oneOf = <T>(filter: T | readonly T[] | undefined, value: T): boolean =>
  filter === undefined || (Array.isArray(filter) ? (filter as readonly T[]).includes(value) : filter === value)

/**
 * The entries of a problem registry that satisfy a query, in registration order.
 *
 * @param reg - The registry.
 * @param query - The filters.
 * @returns The matching entries.
 * @example
 * ```ts
 * import { problemRegistry, whereProblems } from 'mdml-engine/problems'
 * whereProblems(problemRegistry, { kind: 'tour' }).length // => 4
 * ```
 */
export function whereProblems(reg: ProblemRegistry, query: ProblemQuery = {}): ProblemEntry[] {
  return reg.entries.filter(
    (e) =>
      oneOf(query.kind, e.kind) &&
      oneOf(query.solvable, e.solvable) &&
      matchesText(query.text, [e.name, e.title, e.description]),
  )
}

/**
 * The entry of a name in a problem registry.
 *
 * @param reg - The registry.
 * @param name - The entry's name.
 * @returns The entry, or `undefined`.
 * @example
 * ```ts
 * import { findProblem, problemRegistry } from 'mdml-engine/problems'
 * findProblem(problemRegistry, 'towersOfHanoi')?.make({ size: 3 }).truth // => { optimalMoves: 7 }
 * ```
 */
export function findProblem(reg: ProblemRegistry, name: string): ProblemEntry | undefined {
  return find(reg, name)
}
