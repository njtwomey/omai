/**
 * Constraint satisfaction played value by value on the puzzle's own board, from the core's traced solvers
 * (`whitebox-engine/search`): backtracking with variable and value ordering and inference, or min-conflicts local
 * search. Ready-made plays put queens on a chess board, digits in a Sudoku and colours on a map.
 */

import { type Json, WhiteboxError } from '../../lib/base/index.ts'
import { type BoardLayout, type BoardSteps, boardSteps, type Explained } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey, randint } from '../../lib/random/index.ts'
import {
  type Assignment,
  type BacktrackingOptions,
  type BacktrackingState,
  backtrackingAlgorithm,
  type Csp,
  type MinConflictsState,
  minConflictsAlgorithm,
  type VariableOrder,
  violations,
} from '../../lib/search/index.ts'
import { iterate } from '../../lib/trace/index.ts'
import { type ColouringMap, mapColouring, nQueens, sudoku } from './puzzles.ts'

/** Options of the played CSP solvers. */
export interface CspPlayOptions {
  /** Backtracking search (default) or min-conflicts local search. */
  readonly solver?: 'backtracking' | 'min-conflicts'
  /** Backtracking: variable ordering (default `mrv`), value ordering (default static) and inference (default forward checking). */
  readonly variableOrder?: VariableOrder
  readonly valueOrder?: 'static' | 'lcv'
  readonly inference?: 'none' | 'forward-checking' | 'mac'
  /** Min-conflicts: start from a greedy assignment or a random one (default random). */
  readonly start?: 'greedy' | 'random'
  /** Stop after this many steps (default 3000). */
  readonly maxSteps?: number
  /** Record at most this many frames, evenly spaced, the last always kept (default 400). */
  readonly maxFrames?: number
  readonly key?: Key
}

/** What a played CSP solver found. */
export interface CspPlay<D> {
  readonly solver: 'backtracking' | 'min-conflicts'
  readonly status: 'solved' | 'failure' | 'budget'
  readonly assignment: Assignment<D> | null
  /** Steps taken: value attempts and backtracks, or reassignments. */
  readonly steps: number
  readonly assignments: number
  readonly backtracks: number
}

/** What a picture of a CSP sees at one step. */
export interface CspView<D> {
  readonly assignment: Readonly<Record<string, D>>
  /** The variable the step acted on, the value it tried, and what happened (`assign`, `conflict`, `wipeout`, ...). */
  readonly variable?: string
  readonly value?: D
  readonly event: string
  /** Variables in violated constraints (min-conflicts). */
  readonly conflicted: ReadonlySet<string>
  /** The values of a variable ruled out by the assignment (inconsistent with every assigned neighbour). */
  ruledOut(variable: string): ReadonlySet<D>
}

/** How to draw a CSP on a board: the layout, the categories, and each step's cells and text. */
export interface CspPicture<D> {
  readonly layout: BoardLayout
  readonly categories: readonly string[]
  draw(view: CspView<D>): { readonly cells: readonly number[]; readonly labels: readonly string[] }
}

function sample<T>(frames: readonly T[], cap: number): T[] {
  if (frames.length <= cap) return [...frames]
  return Array.from({ length: cap }, (_, i) => frames[Math.round((i * (frames.length - 1)) / (cap - 1))] as T)
}

/**
 * Solve a CSP by backtracking (with variable and value ordering and inference) or by min-conflicts, recording every
 * step for playback on the puzzle's board: the assignment so far, the variable acted on and the value tried, conflicts
 * and the values the assignment rules out.
 *
 * @param problem - The CSP.
 * @param picture - How to draw it.
 * @param options - Solver, ordering, inference, start, budget and frames.
 * @returns The outcome, with a `boardSteps` record.
 */
export function playCsp<D>(
  problem: Csp<D>,
  picture: CspPicture<D>,
  options: CspPlayOptions = {},
): Explained<CspPlay<D>> {
  const solver = options.solver ?? 'backtracking'
  const maxSteps = options.maxSteps ?? 3000
  const k = options.key ?? makeKey(0)
  // Constraints by variable, for the values each variable has left.
  const on = new Map<string, (typeof problem.constraints)[number][]>()
  for (const c of problem.constraints) for (const v of new Set(c.scope)) on.set(v, [...(on.get(v) ?? []), c])
  const ruledOutIn =
    (assignment: Readonly<Record<string, D>>) =>
    (u: string): ReadonlySet<D> => {
      const out = new Set<D>()
      for (const b of problem.domains[u] ?? []) {
        const at = { ...assignment, [u]: b }
        for (const c of on.get(u) ?? []) {
          if (!c.scope.every((w) => w in at)) continue
          if (!c.test(c.scope.map((w) => at[w] as D))) {
            out.add(b)
            break
          }
        }
      }
      return out
    }
  const frames: { cells: readonly number[]; labels: readonly string[]; caption: string; m: number[] }[] = []
  let status: CspPlay<D>['status'] = 'budget'
  let final: Record<string, D> | null = null
  let steps = 0
  let assignments = 0
  let backtracks = 0
  if (solver === 'backtracking') {
    const bt: BacktrackingOptions = {
      variableOrder: options.variableOrder ?? 'mrv',
      valueOrder: options.valueOrder ?? 'static',
      inference: options.inference ?? 'forward-checking',
      maxAttempts: maxSteps,
    }
    const assignment: Record<string, D> = {}
    let last: BacktrackingState | undefined
    for (const { t, state: s } of iterate(backtrackingAlgorithm(problem, bt), { key: k, maxSteps: 2 * maxSteps + 2 })) {
      const e = s.event
      const p = (e?.payload ?? {}) as { variable?: string; value?: Json; assignment?: Record<string, D> }
      if (e !== null && p.variable !== undefined) delete assignment[p.variable]
      if (e !== null && (e.kind === 'assign' || e.kind === 'wipeout') && p.variable !== undefined)
        assignment[p.variable] = p.value as D
      if (e?.kind === 'solution' && p.assignment !== undefined) Object.assign(assignment, p.assignment)
      const view: CspView<D> = {
        assignment,
        ...(p.variable === undefined ? {} : { variable: p.variable }),
        ...(p.value === undefined ? {} : { value: p.value as D }),
        event: e?.kind ?? (t === 0 ? 'start' : 'none'),
        conflicted: new Set(),
        ruledOut: ruledOutIn(assignment),
      }
      const drawn = picture.draw(view)
      const caption =
        e === null
          ? 'start'
          : e.kind === 'backtrack'
            ? `backtrack: ${p.variable} has no values left`
            : e.kind === 'solution'
              ? 'solved'
              : `${e.kind === 'assign' ? 'assign' : e.kind === 'conflict' ? 'reject' : 'wipe-out after'} ${p.variable} = ${String(p.value)}`
      frames.push({ ...drawn, caption, m: [s.assignments, s.backtracks, s.depth] })
      last = s
    }
    if (last === undefined) throw new WhiteboxError('value', 'playCsp: the solver did not start')
    status = last.status === 'running' ? 'budget' : last.status
    final = status === 'solved' ? { ...assignment } : null
    steps = frames.length - 1
    assignments = last.assignments
    backtracks = last.backtracks
  } else {
    const initial: Record<string, D> = {}
    if ((options.start ?? 'random') === 'random') {
      const picks = Array.from(randint(k, [problem.variables.length], 0, 1 << 30).data)
      problem.variables.forEach((v, i) => {
        const d = problem.domains[v] ?? []
        if (d.length > 0) initial[v] = d[(picks[i] ?? 0) % d.length] as D
      })
    }
    let last: MinConflictsState<D> | undefined
    for (const { state: s } of iterate(minConflictsAlgorithm(problem, { initial }), { key: k, maxSteps })) {
      const conflicted = new Set<string>()
      for (const c of violations(problem, s.assignment)) for (const v of c.scope) conflicted.add(v)
      const view: CspView<D> = {
        assignment: s.assignment,
        ...(s.changed === null ? {} : { variable: s.changed, value: s.assignment[s.changed] as D }),
        event: s.changed === null ? 'start' : 'change',
        conflicted,
        ruledOut: ruledOutIn(s.assignment),
      }
      const caption =
        s.changed === null
          ? `start: ${s.conflicts} violated constraints`
          : `${s.changed} = ${String(s.assignment[s.changed])}: ${s.conflicts} violated`
      frames.push({ ...picture.draw(view), caption, m: [s.conflicts] })
      last = s
    }
    if (last === undefined) throw new WhiteboxError('value', 'playCsp: the solver did not start')
    status = last.conflicts === 0 ? 'solved' : 'budget'
    final = status === 'solved' ? { ...last.assignment } : null
    steps = frames.length - 1
    assignments = steps
  }
  const kept = sample(frames, options.maxFrames ?? 400)
  const metrics =
    solver === 'backtracking'
      ? [
          { name: 'assignments', values: kept.map((f) => f.m[0] ?? 0) },
          { name: 'backtracks', values: kept.map((f) => f.m[1] ?? 0) },
        ]
      : [{ name: 'violated constraints', values: kept.map((f) => f.m[0] ?? 0) }]
  const record: BoardSteps = boardSteps({
    method: solver,
    layout: picture.layout,
    categories: picture.categories,
    steps: kept.map((f) => ({ cells: f.cells, labels: f.labels, caption: f.caption })),
    metrics,
    stepName: 'step',
    summary: `${status} after ${steps} steps`,
  })
  return { solver, status, assignment: final, steps, assignments, backtracks, exposition: [record] }
}

const QUEEN = '♛'

/**
 * The n-queens puzzle solved and played on its board: queens placed column by column, the squares their attacks rule
 * out for the columns still empty, rejected and conflicting queens.
 *
 * @param n - The board size (default 8).
 * @param options - Solver, ordering, inference, start and budget.
 * @returns The outcome, with a `boardSteps` record on an n by n board.
 */
export function solveQueens(n = 8, options: CspPlayOptions = {}): Explained<CspPlay<number>> {
  const categories = ['empty', 'shade', 'ruled out', 'queen', 'current', 'conflict']
  const picture: CspPicture<number> = {
    layout: { type: 'grid', rows: n, columns: n },
    categories,
    draw: (view) => {
      const cells: number[] = Array.from({ length: n * n }, (_, i) => ((Math.floor(i / n) + (i % n)) % 2 === 0 ? 0 : 1))
      const labels = new Array<string>(n * n).fill('')
      for (let c = 0; c < n; c++) {
        const col = String(c)
        const r = view.assignment[col]
        if (r !== undefined) continue
        for (const ruled of view.ruledOut(col)) cells[ruled * n + c] = 2
      }
      for (const [col, r] of Object.entries(view.assignment)) {
        const i = r * n + Number(col)
        cells[i] = view.conflicted.has(col) ? 5 : 3
        labels[i] = QUEEN
      }
      if (view.variable !== undefined && view.value !== undefined && view.event !== 'backtrack') {
        const i = view.value * n + Number(view.variable)
        cells[i] = view.event === 'conflict' || view.event === 'wipeout' ? 5 : 4
        labels[i] = QUEEN
      }
      return { cells, labels }
    },
  }
  return playCsp(nQueens(n), picture, options)
}

/**
 * A Sudoku solved and played on its grid: givens, digits placed, the cell acted on and rejected digits.
 *
 * @param grid - The puzzle (81 or 16 characters, `.` for blanks).
 * @param options - Solver, ordering, inference, start and budget.
 * @returns The outcome, with a `boardSteps` record on the grid.
 */
export function solveSudoku(grid: string, options: CspPlayOptions = {}): Explained<CspPlay<number>> {
  const problem = sudoku(grid)
  const side = grid.length === 16 ? 4 : 9
  const box = side === 4 ? 2 : 3
  const given = Array.from({ length: side * side }, (_, i) => (problem.domains[String(i)] ?? []).length === 1)
  const shade = (i: number): number =>
    (Math.floor(Math.floor(i / side) / box) + Math.floor((i % side) / box)) % 2 === 0 ? 0 : 1
  const picture: CspPicture<number> = {
    layout: { type: 'grid', rows: side, columns: side },
    categories: ['empty', 'shade', 'given', 'assigned', 'current', 'conflict'],
    draw: (view) => {
      const cells: number[] = Array.from({ length: side * side }, (_, i) => shade(i))
      const labels = new Array<string>(side * side).fill('')
      for (const [v, d] of Object.entries(view.assignment)) {
        const i = Number(v)
        cells[i] = given[i] === true ? 2 : view.conflicted.has(v) ? 5 : 3
        labels[i] = String(d)
      }
      for (let i = 0; i < side * side; i++)
        if (given[i] === true) {
          cells[i] = 2
          labels[i] = String(problem.domains[String(i)]?.[0] ?? '')
        }
      if (view.variable !== undefined && view.value !== undefined && view.event !== 'backtrack') {
        const i = Number(view.variable)
        if (given[i] !== true) {
          cells[i] = view.event === 'conflict' || view.event === 'wipeout' ? 5 : 4
          labels[i] = String(view.value)
        }
      }
      return { cells, labels }
    },
  }
  return playCsp(problem, picture, options)
}

/**
 * A map coloured and played on its regions: each region in its colour once assigned, the region acted on, and
 * rejected colours.
 *
 * @param map - The map (regions, positions and borders).
 * @param colours - The colours (default red, green and blue).
 * @param options - Solver, ordering, inference, start and budget.
 * @returns The outcome, with a `boardSteps` record on the map's graph.
 */
export function solveMapColouring(
  map: ColouringMap,
  colours: readonly string[] = ['red', 'green', 'blue'],
  options: CspPlayOptions = {},
): Explained<CspPlay<string>> {
  const index = new Map(map.regions.map((r, i) => [r, i]))
  const categories = ['unassigned', ...colours, 'conflict']
  const conflict = categories.length - 1
  const picture: CspPicture<string> = {
    layout: {
      type: 'graph',
      nodes: map.regions.map((r, i) => ({ x: map.positions[i]?.[0] ?? 0, y: map.positions[i]?.[1] ?? 0, label: r })),
      edges: map.borders.map(([a, b]) => [index.get(a) ?? 0, index.get(b) ?? 0] as [number, number]),
    },
    categories,
    draw: (view) => {
      const cells = map.regions.map((r) => {
        const c = view.assignment[r]
        if (c === undefined) return 0
        return view.conflicted.has(r) ? conflict : 1 + colours.indexOf(c)
      })
      if (view.variable !== undefined && view.value !== undefined && view.event !== 'backtrack') {
        const i = index.get(view.variable) ?? 0
        cells[i] = view.event === 'conflict' || view.event === 'wipeout' ? conflict : 1 + colours.indexOf(view.value)
      }
      return { cells, labels: [...map.regions] }
    },
  }
  return playCsp(mapColouring(map, colours), picture, options)
}
