/**
 * A matrix as labelled cells with marks on it: a cell highlighted, a row or a column outlined, a boundary drawn
 * between two rows or two columns (the edge of a fold, the split between users and items, the observed ratings of a
 * ratings matrix). Used by the tensor panel of a session when it is given labels or marks.
 */

import type { EChartsOption } from 'echarts'
import { lines } from '../charts/lines.ts'
import { matrix } from '../charts/matrices.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { type ViewOptions, viewFigure } from './_view.ts'

/**
 * A mark on a matrix: a `cell` `[row, col]` outlined, a whole `row` or `col` outlined, or a boundary drawn above row
 * `rowEdge` (between rows `rowEdge - 1` and `rowEdge`) or left of column `colEdge`.
 */
export type MatrixMark =
  | { readonly cell: readonly [number, number] }
  | { readonly row: number }
  | { readonly col: number }
  | { readonly rowEdge: number }
  | { readonly colEdge: number }

/** Options of {@link matrixMarksView}. */
export interface MatrixMarksOptions extends ViewOptions {
  readonly rowLabels?: readonly string[]
  readonly colLabels?: readonly string[]
  readonly marks?: readonly MatrixMark[]
  /** Write each cell's value in it (default when there are at most 400 cells). */
  readonly annotate?: boolean
}

/**
 * The layers that draw marks over a matrix of `rows` by `cols` cells on category axes (row 0 at the top).
 *
 * @param marks - The marks; those outside the matrix are dropped.
 * @param rows - The number of rows.
 * @param cols - The number of columns.
 * @param mode - The colour mode.
 * @returns The layers.
 * @example
 * ```ts
 * import { matrixMarkLayers } from 'mdml-render/views'
 * matrixMarkLayers([{ cell: [0, 1] }, { rowEdge: 1 }], 3, 3, 'light').length // => 2
 * ```
 */
export function matrixMarkLayers(
  marks: readonly MatrixMark[],
  rows: number,
  cols: number,
  mode: 'light' | 'dark',
): Layer[] {
  // The y axis lists rows bottom-up, so row r sits at category index rows - 1 - r.
  const yOf = (r: number): number => rows - 1 - r
  const outline = { x: [] as number[], y: [] as number[] }
  const edge = { x: [] as number[], y: [] as number[] }
  const box = (target: typeof outline, r0: number, r1: number, c0: number, c1: number): void => {
    target.x.push(c0 - 0.5, c1 + 0.5, c1 + 0.5, c0 - 0.5, c0 - 0.5, Number.NaN)
    target.y.push(yOf(r0) + 0.5, yOf(r0) + 0.5, yOf(r1) - 0.5, yOf(r1) - 0.5, yOf(r0) + 0.5, Number.NaN)
  }
  for (const m of marks) {
    if ('cell' in m && m.cell[0] >= 0 && m.cell[0] < rows && m.cell[1] >= 0 && m.cell[1] < cols)
      box(outline, m.cell[0], m.cell[0], m.cell[1], m.cell[1])
    else if ('row' in m && m.row >= 0 && m.row < rows) box(outline, m.row, m.row, 0, cols - 1)
    else if ('col' in m && m.col >= 0 && m.col < cols) box(outline, 0, rows - 1, m.col, m.col)
    else if ('rowEdge' in m && m.rowEdge > 0 && m.rowEdge < rows) {
      edge.x.push(-0.5, cols - 0.5, Number.NaN)
      edge.y.push(yOf(m.rowEdge) + 0.5, yOf(m.rowEdge) + 0.5, Number.NaN)
    } else if ('colEdge' in m && m.colEdge > 0 && m.colEdge < cols) {
      edge.x.push(m.colEdge - 0.5, m.colEdge - 0.5, Number.NaN)
      edge.y.push(-0.5, rows - 0.5, Number.NaN)
    }
  }
  const out: Layer[] = []
  if (outline.x.length > 0) out.push(lines([{ ...outline, name: '__marks', colour: 'accent', width: 2.6 }], { mode }))
  if (edge.x.length > 0)
    out.push(lines([{ ...edge, name: '__edges', colour: 'fg', width: 2, dash: 'dashed' }], { mode }))
  return out
}

/**
 * A matrix as labelled cells coloured by value, with the marks over it.
 *
 * @param values - The matrix `[rows, cols]`, row 0 at the top.
 * @param options - Row and column labels, marks, annotation, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { matrixMarksView } from 'mdml-render/views'
 * const option = matrixMarksView([[1, 2], [3, 4]], { rowLabels: ['u1', 'u2'], marks: [{ cell: [1, 0] }] })
 * option.series.length // => 2
 * ```
 */
export function matrixMarksView(
  values: readonly (readonly number[])[],
  options: MatrixMarksOptions = {},
): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.4 })
  const mode = fig.mode
  const rows = values.length
  const cols = values[0]?.length ?? 0
  const layers: Layer[] = [
    matrix(values, {
      ...(options.rowLabels === undefined ? {} : { rowLabels: options.rowLabels }),
      ...(options.colLabels === undefined ? {} : { colLabels: options.colLabels }),
      ...(options.annotate === undefined ? {} : { annotate: options.annotate }),
      mode,
    }),
    ...matrixMarkLayers(options.marks ?? [], rows, cols, mode),
  ]
  return { ...cartesian(layers, { mode, figure: fig, legend: false, tooltip: false }), animation: false }
}
