/**
 * The exposition record of a search, solver or simulation played out on a board: a grid world, maze, chess board,
 * Sudoku, game tree, map, itemset lattice or cellular automaton. A board is a grid of cells or a graph of positioned
 * nodes, and each step gives every cell's category, text and a caption.
 */

import { WhiteboxError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/**
 * Where a board's cells sit: a grid of `rows` by `columns` cells (cell `r * columns + c`, row 0 at the top), or the
 * nodes of a graph at given positions (y down, as on a page) with edges between them.
 */
export type BoardLayout =
  | { readonly type: 'grid'; readonly rows: number; readonly columns: number }
  | {
      readonly type: 'graph'
      readonly nodes: readonly { readonly x: number; readonly y: number; readonly label?: string }[]
      readonly edges: readonly (readonly [number, number])[]
      /** Draw edges as arrows (default false). */
      readonly directed?: boolean
    }

/** One step on a board: the category of every cell (an index into the record's `categories`), text and a caption. */
export interface BoardStep {
  /** cells[i]: the category of cell (or node) i at this step. */
  readonly cells: readonly number[]
  /** Text drawn in each cell ('' for none): a digit, a queen, a value backed up, a support count. */
  readonly labels?: readonly string[]
  /** What happened at this step, in a line (`expand (3, 4), g = 7`). */
  readonly caption?: string
}

/**
 * A search, solver or simulation played on a board: a grid world, a maze, a chess board, a Sudoku, a game tree, a map,
 * an itemset lattice or a cellular automaton. Each step gives every cell a category (explored, frontier, current,
 * path; a colour; alive or dead), optional text per cell and a caption; numbers that evolve with the steps (nodes
 * expanded, frontier size, conflicts) ride along as metrics, so a view can play the whole run.
 */
export interface BoardSteps {
  readonly kind: 'boardSteps'
  readonly method: string
  readonly layout: BoardLayout
  /**
   * The names of the cell categories, in index order. Views colour well-known names consistently (`empty`, `wall`,
   * `explored`, `frontier`, `current`, `path`, `start`, `goal`, `pruned`, `conflict`, `alive`, `dead`, colour names)
   * and give the rest the categorical palette.
   */
  readonly categories: readonly string[]
  readonly steps: readonly BoardStep[]
  /** Numbers per step (one value per step), e.g. nodes expanded, frontier size, conflicts, best fitness. */
  readonly metrics?: readonly { readonly name: string; readonly values: readonly number[] }[]
  /** What a step is (default 'step'): 'expansion', 'node', 'assignment', 'generation', 'level'. */
  readonly stepName?: string
  /** The outcome in a line (`path cost 23 after 140 expansions`). */
  readonly summary?: string
}

/** The number of cells of a board layout. */
function cellCount(layout: BoardLayout): number {
  return layout.type === 'grid' ? layout.rows * layout.columns : layout.nodes.length
}

/**
 * Board steps, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a step has the wrong number of cells or labels, or a metric the wrong number of
 *   values; `value` when a category index or an edge is out of range.
 * @example
 * ```ts
 * import { boardSteps } from 'whitebox-engine/exposition'
 * const r = boardSteps({
 *   method: 'bfs',
 *   layout: { type: 'grid', rows: 1, columns: 3 },
 *   categories: ['empty', 'explored', 'frontier'],
 *   steps: [{ cells: [2, 0, 0] }, { cells: [1, 2, 0] }],
 *   metrics: [{ name: 'expanded', values: [0, 1] }],
 * })
 * r.steps.length // => 2
 * ```
 */
export function boardSteps(record: Omit<BoardSteps, 'kind'>): BoardSteps {
  const n = cellCount(record.layout)
  const k = record.categories.length
  if (record.layout.type === 'graph') {
    for (const [a, b] of record.layout.edges)
      if (!(a >= 0 && a < n && b >= 0 && b < n))
        throw new WhiteboxError('value', `boardSteps: the edge ${a}-${b} names a node outside 0..${n - 1}`)
  }
  for (const [t, step] of record.steps.entries()) {
    sameLength('boardSteps', `steps[${t}].cells`, step.cells.length, n)
    if (step.labels !== undefined) sameLength('boardSteps', `steps[${t}].labels`, step.labels.length, n)
    for (const c of step.cells)
      if (!(Number.isInteger(c) && c >= 0 && c < k))
        throw new WhiteboxError('value', `boardSteps: steps[${t}] has the category ${c}, outside 0..${k - 1}`)
  }
  for (const m of record.metrics ?? [])
    sameLength('boardSteps', `metrics.${m.name}`, m.values.length, record.steps.length)
  return { kind: 'boardSteps', ...record }
}
