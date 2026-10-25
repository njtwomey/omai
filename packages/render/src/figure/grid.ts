/**
 * Grid specifications for multi-panel figures, as matplotlib's gridspec: rows and columns with relative sizes and gaps
 * (down to zero), cells that span them and hold axes, text or TeX, their placement in pixels, scatter plots with
 * touching marginal axes, bento layouts from named areas, and the groups of shared and linked axes.
 */

import { MdmlError } from 'mdml-engine/base'
import type { Rect } from './figure.ts'

/** What a grid cell holds: axes (a chart panel), plain text, a TeX equation, or nothing. */
export type CellKind = 'axes' | 'text' | 'tex' | 'empty'

/**
 * One cell of a {@link GridSpec}: its id, its top-left row and column, its spans, and what it holds. `sharex` and
 * `sharey` name another cell whose axis range this cell shares (the ranges are equal and inner tick labels hide);
 * `linkx` and `linky` link hover and zoom without forcing equal ranges.
 */
export interface CellSpec {
  readonly id: string
  readonly row: number
  readonly col: number
  readonly rowSpan?: number
  readonly colSpan?: number
  readonly kind?: CellKind
  /** The text or TeX of a `text` or `tex` cell. */
  readonly content?: string
  readonly sharex?: string
  readonly sharey?: string
  readonly linkx?: string
  readonly linky?: string
}

/**
 * A grid specification (matplotlib's gridspec): rows and columns with relative sizes, gaps between them in pixels (down
 * to zero, for marginal plots that touch their main axes), and cells that span them.
 */
export interface GridSpec {
  readonly kind: 'grid'
  readonly rows: number
  readonly cols: number
  readonly widthRatios: readonly number[]
  readonly heightRatios: readonly number[]
  /** Horizontal gap between columns, in pixels. */
  readonly wspace: number
  /** Vertical gap between rows, in pixels. */
  readonly hspace: number
  readonly cells: readonly CellSpec[]
}

/** Options of {@link gridSpec}. */
export interface GridOptions {
  readonly widthRatios?: readonly number[]
  readonly heightRatios?: readonly number[]
  readonly wspace?: number
  readonly hspace?: number
  /** The cells; by default one axes cell per position, named `r{row}c{col}`. */
  readonly cells?: readonly CellSpec[]
}

function check(spec: GridSpec): GridSpec {
  const { rows, cols } = spec
  if (!(Number.isInteger(rows) && rows >= 1 && Number.isInteger(cols) && cols >= 1))
    throw new MdmlError('value', 'gridSpec: rows and cols must be positive integers')
  if (spec.widthRatios.length !== cols || spec.heightRatios.length !== rows)
    throw new MdmlError('value', 'gridSpec: give one ratio per column and per row')
  if (![...spec.widthRatios, ...spec.heightRatios].every((r) => r > 0))
    throw new MdmlError('value', 'gridSpec: ratios must be positive')
  const taken = new Map<string, string>()
  const ids = new Set<string>()
  for (const c of spec.cells) {
    if (ids.has(c.id)) throw new MdmlError('value', `gridSpec: duplicate cell id '${c.id}'`)
    ids.add(c.id)
    const rs = c.rowSpan ?? 1
    const cs = c.colSpan ?? 1
    if (c.row < 0 || c.col < 0 || c.row + rs > rows || c.col + cs > cols || rs < 1 || cs < 1)
      throw new MdmlError('value', `gridSpec: cell '${c.id}' lies outside the grid`)
    for (let r = c.row; r < c.row + rs; r++)
      for (let q = c.col; q < c.col + cs; q++) {
        const other = taken.get(`${r},${q}`)
        if (other !== undefined) throw new MdmlError('value', `gridSpec: cells '${other}' and '${c.id}' overlap`)
        taken.set(`${r},${q}`, c.id)
      }
  }
  for (const c of spec.cells)
    for (const ref of [c.sharex, c.sharey, c.linkx, c.linky])
      if (ref !== undefined && !ids.has(ref))
        throw new MdmlError('value', `gridSpec: cell '${c.id}' refers to unknown cell '${ref}'`)
  return spec
}

/**
 * Make a grid specification.
 *
 * @param rows - Number of rows.
 * @param cols - Number of columns.
 * @param options - Ratios, gaps (default 56 px between columns and 52 px between rows: room for tick labels and panel titles) and cells.
 * @returns The grid.
 * @throws MdmlError with code `value` for overlapping or out-of-range cells and unknown shared cells.
 * @example
 * ```ts
 * import { gridSpec } from 'mdml-render/figure'
 * gridSpec(2, 3).cells.length // => 6
 * gridSpec(1, 2, { cells: [{ id: 'wide', row: 0, col: 0, colSpan: 2 }] }).cells[0].colSpan // => 2
 * ```
 */
export function gridSpec(rows: number, cols: number, options: GridOptions = {}): GridSpec {
  const cells =
    options.cells ??
    Array.from({ length: rows * cols }, (_, i) => ({
      id: `r${Math.floor(i / cols)}c${i % cols}`,
      row: Math.floor(i / cols),
      col: i % cols,
    }))
  return check({
    kind: 'grid',
    rows,
    cols,
    widthRatios: options.widthRatios ?? new Array<number>(cols).fill(1),
    heightRatios: options.heightRatios ?? new Array<number>(rows).fill(1),
    wspace: options.wspace ?? 56,
    hspace: options.hspace ?? 52,
    cells,
  })
}

function tracks(total: number, ratios: readonly number[], gap: number): { start: number[]; size: number[] } {
  const free = Math.max(0, total - gap * (ratios.length - 1))
  const sum = ratios.reduce((a, b) => a + b, 0)
  const size = ratios.map((r) => (free * r) / sum)
  const start: number[] = []
  let at = 0
  for (const s of size) {
    start.push(at)
    at += s + gap
  }
  return { start, size }
}

/**
 * Place the cells of a grid in a rectangle: each cell's box in pixels, spans covering the gaps between their tracks.
 *
 * @param spec - The grid.
 * @param area - The rectangle to fill.
 * @returns Each cell's rectangle by id.
 * @example
 * ```ts
 * import { gridSpec, layoutGrid } from 'mdml-render/figure'
 * const boxes = layoutGrid(gridSpec(1, 2, { wspace: 20 }), { x: 0, y: 0, width: 220, height: 100 })
 * boxes['r0c1'] // => { x: 120, y: 0, width: 100, height: 100 }
 * ```
 */
export function layoutGrid(spec: GridSpec, area: Rect): Record<string, Rect> {
  const cx = tracks(area.width, spec.widthRatios, spec.wspace)
  const cy = tracks(area.height, spec.heightRatios, spec.hspace)
  const out: Record<string, Rect> = {}
  for (const c of spec.cells) {
    const c0 = c.col
    const c1 = c.col + (c.colSpan ?? 1) - 1
    const r0 = c.row
    const r1 = c.row + (c.rowSpan ?? 1) - 1
    const x0 = cx.start[c0] ?? 0
    const y0 = cy.start[r0] ?? 0
    out[c.id] = {
      x: area.x + x0,
      y: area.y + y0,
      width: (cx.start[c1] ?? 0) + (cx.size[c1] ?? 0) - x0,
      height: (cy.start[r1] ?? 0) + (cy.size[r1] ?? 0) - y0,
    }
  }
  return out
}

/** Options of {@link withMarginals}. */
export interface MarginalOptions {
  /** The marginal axes' size relative to the main axes (default 0.25). */
  readonly ratio?: number
  /** The gap between the main and marginal axes in pixels (default 0: they touch). */
  readonly gap?: number
  /** Which marginals to draw (default both). */
  readonly sides?: readonly ('top' | 'right')[]
}

/**
 * A grid for a scatter plot with marginal distributions: a `main` axes cell, a `top` cell sharing its x axis and a
 * `right` cell sharing its y axis, touching it by default.
 *
 * @param options - Marginal size, gap and sides.
 * @returns The grid, with cells `main`, `top` and `right` (and an empty `corner`).
 * @example
 * ```ts
 * import { withMarginals } from 'mdml-render/figure'
 * const g = withMarginals()
 * g.cells.find((c) => c.id === 'top')?.sharex // => 'main'
 * g.wspace // => 0
 * ```
 */
export function withMarginals(options: MarginalOptions = {}): GridSpec {
  const { ratio = 0.25, gap = 0, sides = ['top', 'right'] } = options
  const top = sides.includes('top')
  const right = sides.includes('right')
  const rows = top ? 2 : 1
  const cols = right ? 2 : 1
  const mainRow = top ? 1 : 0
  const cells: CellSpec[] = [{ id: 'main', row: mainRow, col: 0, kind: 'axes' }]
  if (top) cells.push({ id: 'top', row: 0, col: 0, kind: 'axes', sharex: 'main' })
  if (right) cells.push({ id: 'right', row: mainRow, col: 1, kind: 'axes', sharey: 'main' })
  if (top && right) cells.push({ id: 'corner', row: 0, col: 1, kind: 'empty' })
  return gridSpec(rows, cols, {
    widthRatios: right ? [1, ratio] : [1],
    heightRatios: top ? [ratio, 1] : [1],
    wspace: gap,
    hspace: gap,
    cells,
  })
}

/**
 * A bento layout from named areas, as CSS `grid-template-areas`: each string is a row of space-separated names, a name
 * repeated over a rectangle spans it, and `.` leaves a position empty.
 *
 * @param areas - The rows.
 * @param options - Ratios and gaps as {@link gridSpec}, and the kind of each named cell (default axes).
 * @returns The grid.
 * @throws MdmlError with code `value` when rows differ in length or a name does not cover a rectangle.
 * @example
 * ```ts
 * import { bento } from 'mdml-render/figure'
 * const g = bento(['a a b', 'c d b'])
 * g.cells.find((c) => c.id === 'b')?.rowSpan // => 2
 * g.cells.find((c) => c.id === 'a')?.colSpan // => 2
 * ```
 */
export function bento(
  areas: readonly string[],
  options: Omit<GridOptions, 'cells'> & { readonly kinds?: Readonly<Record<string, CellKind>> } = {},
): GridSpec {
  const grid = areas.map((row) => row.trim().split(/\s+/))
  const rows = grid.length
  const cols = grid[0]?.length ?? 0
  if (rows === 0 || grid.some((r) => r.length !== cols))
    throw new MdmlError('value', 'bento: every row needs the same number of names')
  const boxes = new Map<string, { r0: number; r1: number; c0: number; c1: number }>()
  grid.forEach((row, r) => {
    row.forEach((name, c) => {
      if (name === '.') return
      const b = boxes.get(name)
      if (b === undefined) boxes.set(name, { r0: r, r1: r, c0: c, c1: c })
      else
        boxes.set(name, { r0: Math.min(b.r0, r), r1: Math.max(b.r1, r), c0: Math.min(b.c0, c), c1: Math.max(b.c1, c) })
    })
  })
  const cells: CellSpec[] = []
  for (const [name, b] of boxes) {
    for (let r = b.r0; r <= b.r1; r++)
      for (let c = b.c0; c <= b.c1; c++)
        if (grid[r]?.[c] !== name) throw new MdmlError('value', `bento: area '${name}' is not a rectangle`)
    cells.push({
      id: name,
      row: b.r0,
      col: b.c0,
      rowSpan: b.r1 - b.r0 + 1,
      colSpan: b.c1 - b.c0 + 1,
      kind: options.kinds?.[name] ?? 'axes',
    })
  }
  const { kinds: _kinds, ...rest } = options
  return gridSpec(rows, cols, { ...rest, cells })
}

/**
 * The groups of cells whose x (or y) ranges are shared, following `sharex` (or `sharey`) chains; cells that share
 * with nothing are left out.
 *
 * @param spec - The grid.
 * @param axis - `x` or `y`.
 * @returns Groups of cell ids, each in grid order.
 * @example
 * ```ts
 * import { sharedGroups, withMarginals } from 'mdml-render/figure'
 * sharedGroups(withMarginals(), 'x') // => [['main', 'top']]
 * ```
 */
export function sharedGroups(spec: GridSpec, axis: 'x' | 'y'): string[][] {
  return groups(spec, (c) => [axis === 'x' ? c.sharex : c.sharey])
}

/**
 * The groups of cells whose x (or y) axes are linked for hover and zoom: shared cells and `linkx` (or `linky`) chains.
 *
 * @param spec - The grid.
 * @param axis - `x` or `y`.
 * @returns Groups of cell ids.
 * @example
 * ```ts
 * import { gridSpec, linkedGroups } from 'mdml-render/figure'
 * const g = gridSpec(2, 1, { cells: [{ id: 'a', row: 0, col: 0 }, { id: 'b', row: 1, col: 0, linkx: 'a' }] })
 * linkedGroups(g, 'x') // => [['a', 'b']]
 * ```
 */
export function linkedGroups(spec: GridSpec, axis: 'x' | 'y'): string[][] {
  return groups(spec, (c) => (axis === 'x' ? [c.sharex, c.linkx] : [c.sharey, c.linky]))
}

function groups(spec: GridSpec, refs: (c: CellSpec) => readonly (string | undefined)[]): string[][] {
  const parent = new Map(spec.cells.map((c) => [c.id, c.id]))
  const find = (id: string): string => {
    let r = id
    while (parent.get(r) !== r) r = parent.get(r) ?? r
    return r
  }
  for (const c of spec.cells) {
    for (const other of refs(c)) if (other !== undefined) parent.set(find(c.id), find(other))
  }
  const out = new Map<string, string[]>()
  for (const c of spec.cells) {
    const root = find(c.id)
    out.set(root, [...(out.get(root) ?? []), c.id])
  }
  return [...out.values()].filter((g) => g.length > 1)
}
