/**
 * A matrix as labelled cells with marks on it: a cell highlighted, a row or a column outlined, a boundary drawn
 * between two rows or two columns (the edge of a fold, the split between users and items, the observed ratings of a
 * ratings matrix). Used by the tensor panel of a session when it is given labels or marks.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { colourFor } from '../charts/_data.ts'
import { matrix } from '../charts/matrices.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { ariaOf, type ViewOptions, viewFigure } from './_view.ts'

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

/** A mark drawn over cells, in continuous coordinates where the centre of column `c` is `c` (x) and of row `r` is `rows - 1 - r` (y). */
export interface CellShape {
  /** An outlined rectangle between two corners, or a line between them. */
  readonly shape: 'box' | 'line'
  readonly x0: number
  readonly y0: number
  readonly x1: number
  readonly y1: number
  /** A colour token or CSS colour. */
  readonly colour: string
  readonly width?: number
  readonly dashed?: boolean
}

/**
 * Outlines and lines over a grid of cells on category axes, as one custom series drawn in pixels (so a box lies on the
 * cells' edges: a category axis would round a half-cell position to a cell centre). The first cell of each shape
 * anchors it on the axes.
 *
 * @param shapes - The shapes, in the coordinates of {@link CellShape}.
 * @param rows - The number of rows of the matrix.
 * @param cols - The number of columns.
 * @param mode - The colour mode.
 * @returns The layer (no series when there are no shapes).
 * @example
 * ```ts
 * import { cellShapes } from 'omai-render/views'
 * cellShapes([{ shape: 'box', x0: -0.5, y0: 1.5, x1: 0.5, y1: 0.5, colour: 'accent' }], 2, 2, 'light').series.length // => 1
 * ```
 */
export function cellShapes(shapes: readonly CellShape[], rows: number, cols: number, mode: ColourMode): Layer {
  if (shapes.length === 0) return layer([])
  const clamp = (v: number, n: number): number => Math.min(n - 1, Math.max(0, Math.round(v)))
  const series = {
    type: 'custom',
    name: '__cellShapes',
    // Each datum is anchored on a cell that exists; the shape's own coordinates follow.
    data: shapes.map((m) => [
      clamp(m.x0, cols),
      clamp(m.y0, rows),
      m.x0,
      m.y0,
      m.x1,
      m.y1,
      m.shape === 'box' ? 0 : 1,
      m.width ?? 2.4,
      m.dashed === true ? 1 : 0,
    ]),
    encode: { x: 0, y: 1 },
    silent: true,
    z: 6,
    renderItem: (
      params: { dataIndex: number },
      api: { value(d: number): number; coord(p: number[]): number[]; size(p: number[]): number[] },
    ): Record<string, unknown> => {
      const m = shapes[params.dataIndex]
      if (m === undefined) return { type: 'group', children: [] }
      const [ox = 0, oy = 0] = api.coord([0, 0])
      const [w = 0, h = 0] = api.size([1, 1])
      // Column c centres at ox + c w; row index y (bottom-up) at oy - y h.
      const px = (x: number): number => ox + x * w
      const py = (y: number): number => oy - y * h
      const style = {
        stroke: colourFor(m.colour, mode),
        lineWidth: m.width ?? 2.4,
        fill: 'none',
        ...(m.dashed === true ? { lineDash: [6, 4] } : {}),
      }
      return m.shape === 'box'
        ? {
            type: 'rect',
            shape: {
              x: px(Math.min(m.x0, m.x1)),
              y: py(Math.max(m.y0, m.y1)),
              width: Math.abs(m.x1 - m.x0) * w,
              height: Math.abs(m.y1 - m.y0) * h,
            },
            style,
          }
        : { type: 'line', shape: { x1: px(m.x0), y1: py(m.y0), x2: px(m.x1), y2: py(m.y1) }, style }
    },
  } as unknown as SeriesOption
  return layer([series])
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
 * import { matrixMarkLayers } from 'omai-render/views'
 * matrixMarkLayers([{ cell: [0, 1] }, { rowEdge: 1 }], 3, 3, 'light').length // => 1
 * ```
 */
export function matrixMarkLayers(marks: readonly MatrixMark[], rows: number, cols: number, mode: ColourMode): Layer[] {
  // The y axis lists rows bottom-up, so row r sits at index rows - 1 - r and its top edge is half a cell above it.
  const yOf = (r: number): number => rows - 1 - r
  const shapes: CellShape[] = []
  const box = (r0: number, r1: number, c0: number, c1: number): void => {
    shapes.push({
      shape: 'box',
      x0: c0 - 0.5,
      y0: yOf(r0) + 0.5,
      x1: c1 + 0.5,
      y1: yOf(r1) - 0.5,
      colour: 'accent',
      width: 2.6,
    })
  }
  for (const m of marks) {
    if ('cell' in m && m.cell[0] >= 0 && m.cell[0] < rows && m.cell[1] >= 0 && m.cell[1] < cols)
      box(m.cell[0], m.cell[0], m.cell[1], m.cell[1])
    else if ('row' in m && m.row >= 0 && m.row < rows) box(m.row, m.row, 0, cols - 1)
    else if ('col' in m && m.col >= 0 && m.col < cols) box(0, rows - 1, m.col, m.col)
    else if ('rowEdge' in m && m.rowEdge > 0 && m.rowEdge < rows)
      shapes.push({
        shape: 'line',
        x0: -0.5,
        y0: yOf(m.rowEdge) + 0.5,
        x1: cols - 0.5,
        y1: yOf(m.rowEdge) + 0.5,
        colour: 'fg',
        dashed: true,
      })
    else if ('colEdge' in m && m.colEdge > 0 && m.colEdge < cols)
      shapes.push({
        shape: 'line',
        x0: m.colEdge - 0.5,
        y0: -0.5,
        x1: m.colEdge - 0.5,
        y1: rows - 0.5,
        colour: 'fg',
        dashed: true,
      })
  }
  return shapes.length === 0 ? [] : [cellShapes(shapes, rows, cols, mode)]
}

/**
 * A matrix as labelled cells coloured by value, with the marks over it.
 *
 * @param values - The matrix `[rows, cols]`, row 0 at the top.
 * @param options - Row and column labels, marks, annotation, mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { matrixMarksView } from 'omai-render/views'
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
  const option = cartesian(layers, { mode, figure: fig, legend: false, tooltip: false }) as {
    xAxis: Record<string, unknown>[]
  }
  // Every column named, not every other one.
  const [x] = option.xAxis
  const named =
    options.colLabels === undefined ? undefined : { ...x, axisLabel: { ...(x?.['axisLabel'] as object), interval: 0 } }
  return {
    ...option,
    ...(named === undefined ? {} : { xAxis: [named] }),
    animation: false,
    ...ariaOf(
      `Matrix of ${rows} rows by ${cols} columns, cells coloured by value${(options.marks ?? []).length === 0 ? '' : `, with ${(options.marks ?? []).length} marks`}`,
    ),
  } as EChartsOption
}
