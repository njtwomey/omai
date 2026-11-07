/**
 * The view of a `MarkedTable` record: a small table drawn as a scene, its cells tinted by their status (ok, warn, bad,
 * info, muted) and the cell, row or column the rule chose outlined. `markedTableModel` is the resolved table (text and
 * status per cell) that the scene is drawn from, for tests and for other renderers.
 */

import { maxOf } from 'mdml-engine/base'
import type { CellStatus, MarkedTable } from 'mdml-engine/exposition'
import { line, type Paint, rect, type Scene, type SceneNode, scene, text } from 'mdml-engine/scene'
import { formatNumber } from '../theme/format.ts'

/** One resolved cell of a marked table. */
export interface ModelCell {
  readonly text: string
  /** Whether the cell is a number (right-aligned). */
  readonly numeric: boolean
  /** The tint's status, the last mark that gave one (`decision` is not a tint). */
  readonly status?: Exclude<CellStatus, 'decision'>
  /** Whether a `decision` mark covers the cell. */
  readonly decision: boolean
  readonly note?: string
}

/** A marked table with its text formatted and its marks resolved per cell. */
export interface TableModel {
  readonly header: readonly string[]
  /** The leading column of row names, when the record has them. */
  readonly rowNames?: readonly string[]
  readonly cells: readonly (readonly ModelCell[])[]
}

/**
 * Resolve a marked table: numbers formatted to its digits (missing as a dash), and each cell's status and decision
 * flag from the marks that cover it (a mark on a row or a column covers its cells; later marks win).
 *
 * @param record - The record.
 * @returns The model.
 * @example
 * ```ts
 * import { markedTable } from 'mdml-engine/exposition'
 * import { markedTableModel } from 'mdml-render/views'
 * const r = markedTable({ columns: ['p'], cells: [[0.0312], [null]], marks: [{ row: 0, status: 'bad' }] })
 * markedTableModel(r).cells.map((row) => row[0]?.text) // => ['0.0312', '—']
 * ```
 */
export function markedTableModel(record: MarkedTable): TableModel {
  const digits = record.digits ?? 3
  return {
    header: record.columns,
    ...(record.rowNames === undefined ? {} : { rowNames: record.rowNames }),
    cells: record.cells.map((row, r) =>
      row.map((v, c): ModelCell => {
        const covering = (record.marks ?? []).filter(
          (m) => (m.row === undefined || m.row === r) && (m.col === undefined || m.col === c),
        )
        const tint = covering.findLast((m) => m.status !== 'decision')
        const note = covering.findLast((m) => m.note !== undefined)?.note
        return {
          text: v === null ? '—' : typeof v === 'number' ? formatNumber(v, { digits }) : v,
          numeric: typeof v === 'number',
          ...(tint === undefined ? {} : { status: tint.status as Exclude<CellStatus, 'decision'> }),
          decision: covering.some((m) => m.status === 'decision'),
          ...(note === undefined ? {} : { note }),
        }
      }),
    ),
  }
}

const TINT: Readonly<Record<Exclude<CellStatus, 'decision'>, Paint>> = {
  ok: 'tab2',
  warn: 'tab1',
  bad: 'tab3',
  info: 'tab0',
  muted: 'grid',
}

const FONT = 12
const ROW = 24
const CHAR = 7.2
const PAD = 8

/**
 * A marked table as a scene: a header row, one row per row of cells, status tints behind the cells and an accent
 * outline on the cells a `decision` mark covers.
 *
 * @param record - The record.
 * @returns The scene (its width follows the longest entries of each column).
 * @example
 * ```ts
 * import { markedTable } from 'mdml-engine/exposition'
 * import { markedTableScene } from 'mdml-render/views'
 * markedTableScene(markedTable({ columns: ['a'], cells: [[1]] })).kind // => 'scene'
 * ```
 */
export function markedTableScene(record: MarkedTable): Scene {
  const model = markedTableModel(record)
  const names = model.rowNames
  const columns = [...(names === undefined ? [] : ['']), ...model.header]
  const body = model.cells.map((row, r) => [
    ...(names === undefined
      ? []
      : [
          {
            text: names[r] ?? '',
            numeric: false,
            // A decision on a whole row outlines its name too.
            decision: (record.marks ?? []).some((m) => m.status === 'decision' && m.row === r && m.col === undefined),
          } as ModelCell,
        ]),
    ...row,
  ])
  const widths = columns.map((h, c) =>
    Math.ceil(Math.max(h.length, maxOf(body.map((row) => row[c]?.text.length ?? 0))) * CHAR + 2 * PAD),
  )
  const left: number[] = []
  widths.reduce((x, w) => {
    left.push(x)
    return x + w
  }, 0)
  const width = widths.reduce((a, b) => a + b, 0)
  const top = record.caption === undefined ? 0 : ROW
  const height = top + ROW * (body.length + 1)
  const children: SceneNode[] = []
  if (record.caption !== undefined)
    children.push(text(0, ROW / 2, record.caption, { size: FONT, baseline: 'middle', fill: 'muted' }))
  // The tints, then the rules and text over them, then the outlines.
  const tints: SceneNode[] = []
  const outlines: SceneNode[] = []
  const words: SceneNode[] = []
  for (const [c, h] of columns.entries())
    words.push(
      text((left[c] ?? 0) + PAD, top + ROW / 2, h, { size: FONT, weight: 'bold', baseline: 'middle', fill: 'fg' }),
    )
  body.forEach((row, r) => {
    const y = top + ROW * (r + 1)
    row.forEach((cell, c) => {
      const x = left[c] ?? 0
      const w = widths[c] ?? 0
      if (cell.status !== undefined)
        tints.push(
          rect(x, y, w, ROW, {
            fill: TINT[cell.status],
            opacity: 0.28,
            ...(cell.note === undefined ? {} : { title: cell.note }),
          }),
        )
      if (cell.decision)
        outlines.push(rect(x + 1, y + 1, w - 2, ROW - 2, { fill: 'none', stroke: 'accent', strokeWidth: 2.2 }))
      words.push(
        text(cell.numeric ? x + w - PAD : x + PAD, y + ROW / 2, cell.text, {
          size: FONT,
          baseline: 'middle',
          anchor: cell.numeric ? 'end' : 'start',
          fill: cell.status === 'muted' ? 'muted' : 'fg',
        }),
      )
    })
  })
  const rules = Array.from({ length: body.length + 2 }, (_, r) =>
    line(0, top + ROW * r, width, top + ROW * r, { stroke: 'grid', strokeWidth: r === 1 ? 1.5 : 0.8 }),
  )
  const title =
    record.caption ??
    `Table of ${body.length} rows and ${record.columns.length} columns${record.cells.length === 0 ? '' : `: ${record.columns.join(', ')}`}`
  return scene(width, height, [...children, ...tints, ...rules, ...words, ...outlines], { title })
}
