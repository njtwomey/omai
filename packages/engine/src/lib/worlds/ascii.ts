import { WhiteboxError } from '../base/index.ts'
import { hasWall, type ItemKind, type ItemSpec, TERRAINS, type TerrainSpec, type World, world } from './world.ts'

/** How an ASCII character reads: a terrain, and whether the cell is a start, a goal, holds an item or is a landmark. */
export interface Glyph {
  readonly terrain: string
  readonly start?: boolean
  readonly goal?: boolean
  readonly item?: { readonly kind: ItemKind; readonly colour?: string; readonly reward?: number }
  /** Record the cell under this landmark name. */
  readonly landmark?: string
}

/** Characters and what they mean. */
export type Legend = Readonly<Record<string, Glyph>>

/**
 * The usual ASCII conventions: every terrain character of {@link TERRAINS}, a space for floor, `S` start, `G` goal,
 * `K` a key, `D` a door and `$` a pickup (all on floor).
 */
export const ASCII_LEGEND: Legend = {
  ...Object.fromEntries(TERRAINS.map((t) => [t.char, { terrain: t.name }])),
  ' ': { terrain: 'floor' },
  S: { terrain: 'floor', start: true },
  G: { terrain: 'floor', goal: true },
  K: { terrain: 'floor', item: { kind: 'key' } },
  D: { terrain: 'floor', item: { kind: 'door' } },
  $: { terrain: 'floor', item: { kind: 'pickup', reward: 1 } },
}

/** Options of {@link fromAscii} and {@link toAscii}. */
export interface AsciiOptions {
  /** Extra characters, which take priority over {@link ASCII_LEGEND}. */
  readonly legend?: Legend
  /** Extra or replacement terrains (matched by name) added to {@link TERRAINS}. */
  readonly terrains?: readonly TerrainSpec[]
  /**
   * `'cells'` (default): one character per cell. `'walls'`: a `(2 rows + 1) x (2 cols + 1)` drawing with cells at odd
   * rows and columns and thin walls between them (`#`, `|`, `-`, `+` or `=` closed; anything else open).
   */
  readonly format?: 'cells' | 'walls'
  /** Read the rows as a square (default) or hexagonal (odd rows shifted right) grid. */
  readonly topology?: 'square' | 'hex'
  readonly name?: string
}

const WALL_CHARS = '#|-+='

function palette(extra: readonly TerrainSpec[] | undefined): TerrainSpec[] {
  const out: TerrainSpec[] = [...TERRAINS]
  for (const t of extra ?? []) {
    const i = out.findIndex((x) => x.name === t.name)
    if (i >= 0) out[i] = t
    else out.push(t)
  }
  return out
}

/**
 * Read a world from an ASCII map: one character per cell (FrozenLake, MiniGrid and roguelike style), or thin walls
 * drawn between cells (`format: 'walls'`, the usual maze drawing). Starts, goals, items and landmarks are collected in
 * row-major order.
 *
 * @param rows - The map, one string per row (all the same length).
 * @param options - Extra legend entries and terrains, the format, square or hexagonal cells, and a name.
 * @returns The world.
 * @throws WhiteboxError with code `value` for ragged rows, unknown characters or a malformed wall drawing.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { fromAscii } from 'whitebox-engine/worlds'
 * const w = fromAscii(['S.#', '..G'])
 * [w.n, w.starts, w.goals] // => [6, [0], [5]]
 * fromAscii(['#####', '#. .#', '#####'], { format: 'walls' }).walls[0] // => 0
 * ```
 */
export function fromAscii(rows: readonly string[], options: AsciiOptions = {}): World {
  const legend: Legend = { ...ASCII_LEGEND, ...options.legend }
  const width = rows[0]?.length ?? 0
  if (rows.length === 0 || width === 0 || rows.some((r) => r.length !== width)) {
    throw new WhiteboxError('value', 'fromAscii: rows must be non-empty and of equal length')
  }
  const walled = options.format === 'walls'
  if (walled && (rows.length % 2 === 0 || width % 2 === 0 || rows.length < 3 || width < 3)) {
    throw new WhiteboxError('value', 'fromAscii: a wall drawing has 2 rows + 1 lines of 2 cols + 1 characters')
  }
  if (walled && options.topology === 'hex') throw new WhiteboxError('value', 'fromAscii: wall drawings are square')
  const nr = walled ? (rows.length - 1) / 2 : rows.length
  const nc = walled ? (width - 1) / 2 : width
  const terrain: string[] = []
  const starts: number[] = []
  const goals: number[] = []
  const items: ItemSpec[] = []
  const landmarks: Record<string, number[]> = {}
  for (let r = 0; r < nr; r++) {
    for (let c = 0; c < nc; c++) {
      const ch = walled ? (rows[2 * r + 1]?.[2 * c + 1] ?? '') : (rows[r]?.[c] ?? '')
      const g = legend[ch]
      if (g === undefined) throw new WhiteboxError('value', `fromAscii: unknown character '${ch}' at row ${r}`)
      const cell = r * nc + c
      terrain.push(g.terrain)
      if (g.start === true) starts.push(cell)
      if (g.goal === true) goals.push(cell)
      if (g.item !== undefined) items.push({ ...g.item, cell })
      if (g.landmark !== undefined) {
        const list = landmarks[g.landmark] ?? []
        list.push(cell)
        landmarks[g.landmark] = list
      }
    }
  }
  const walls: [number, number][] = []
  if (walled) {
    for (let r = 0; r < nr; r++) {
      for (let c = 0; c < nc; c++) {
        const cell = r * nc + c
        if (c + 1 < nc && WALL_CHARS.includes(rows[2 * r + 1]?.[2 * c + 2] ?? '#')) walls.push([cell, cell + 1])
        if (r + 1 < nr && WALL_CHARS.includes(rows[2 * r + 2]?.[2 * c + 1] ?? '#')) walls.push([cell, cell + nc])
      }
    }
  }
  return world({
    topology: { kind: options.topology === 'hex' ? 'hex' : 'square', rows: nr, cols: nc },
    terrains: palette(options.terrains),
    terrain,
    walls,
    items,
    starts,
    goals,
    landmarks,
    ...(options.name === undefined ? {} : { name: options.name }),
  })
}

function glyphMatches(w: World, cell: number, g: Glyph, starts: Set<number>, goals: Set<number>): boolean {
  if (w.terrains[w.terrain[cell] ?? 0]?.name !== g.terrain) return false
  if ((g.start === true) !== starts.has(cell) || (g.goal === true) !== goals.has(cell)) return false
  const item = w.items.find((it) => it.cell === cell)
  if ((g.item === undefined) !== (item === undefined)) return false
  if (g.item !== undefined && item !== undefined) {
    if (g.item.kind !== item.kind || (g.item.colour ?? 'yellow') !== item.colour) return false
  }
  const marks = Object.entries(w.landmarks).filter(([, cells]) => cells.includes(cell))
  if (g.landmark === undefined) return marks.length === 0
  return marks.length === 1 && marks[0]?.[0] === g.landmark
}

/**
 * Write a world as an ASCII map, choosing for each cell the first legend character (the options' legend first, then
 * {@link ASCII_LEGEND}) that describes it exactly, else its terrain's character. `fromAscii(toAscii(w))` rebuilds `w`
 * whenever every cell has an exact character.
 *
 * @param w - A square-grid world (hexagonal worlds are written row by row, without the half-cell shift).
 * @param options - Extra legend entries and the format (`'cells'` or `'walls'`).
 * @returns The rows.
 * @throws WhiteboxError with code `value` for graph worlds, or a wall drawing of a hexagonal world.
 * @example
 * ```ts
 * import { fromAscii, toAscii } from 'whitebox-engine/worlds'
 * toAscii(fromAscii(['S #', '..G'])) // => ['S.#', '..G']
 * toAscii(fromAscii(['S.', '.G']), { format: 'walls' }) // => ['#####', '#S..#', '#.#.#', '#..G#', '#####']
 * ```
 */
export function toAscii(w: World, options: Pick<AsciiOptions, 'legend' | 'format'> = {}): string[] {
  const t = w.topology
  if (t.kind === 'graph') throw new WhiteboxError('value', 'toAscii: graph worlds have no grid')
  if (options.format === 'walls' && t.kind !== 'square') throw new WhiteboxError('value', 'toAscii: walls need squares')
  const legend = [...Object.entries(options.legend ?? {}), ...Object.entries(ASCII_LEGEND)]
  const starts = new Set(w.starts)
  const goals = new Set(w.goals)
  const charOf = (cell: number): string => {
    for (const [ch, g] of legend) if (glyphMatches(w, cell, g, starts, goals)) return ch
    return w.terrains[w.terrain[cell] ?? 0]?.char ?? '?'
  }
  if (options.format !== 'walls') {
    return Array.from({ length: t.rows }, (_, r) =>
      Array.from({ length: t.cols }, (_, c) => charOf(r * t.cols + c)).join(''),
    )
  }
  const out: string[][] = Array.from({ length: 2 * t.rows + 1 }, () => new Array<string>(2 * t.cols + 1).fill('#'))
  for (let r = 0; r < t.rows; r++) {
    for (let c = 0; c < t.cols; c++) {
      const cell = r * t.cols + c
      const row = out[2 * r + 1] as string[]
      row[2 * c + 1] = charOf(cell)
      if (c + 1 < t.cols && !hasWall(w, cell, 1)) row[2 * c + 2] = '.'
      if (r + 1 < t.rows && !hasWall(w, cell, 2)) (out[2 * r + 2] as string[])[2 * c + 1] = '.'
    }
  }
  return out.map((row) => row.join(''))
}

/**
 * Build a world from rows of terrain indices or names (an occupancy grid: `0` floor and `1` wall with the default
 * terrains).
 *
 * @param rows - The cells, row by row (all rows the same length).
 * @param options - Terrains, starts, goals, square or hexagonal cells and a name.
 * @returns The world.
 * @throws WhiteboxError with code `value` for ragged rows or unknown terrains.
 * @example
 * ```ts
 * import { fromRows, toAscii } from 'whitebox-engine/worlds'
 * toAscii(fromRows([[0, 1], ['water', 0]], { goals: [3] })) // => ['.#', '~G']
 * ```
 */
export function fromRows(
  rows: readonly (readonly (number | string)[])[],
  options: {
    readonly terrains?: readonly TerrainSpec[]
    readonly starts?: readonly number[]
    readonly goals?: readonly number[]
    readonly topology?: 'square' | 'hex'
    readonly name?: string
  } = {},
): World {
  const cols = rows[0]?.length ?? 0
  if (rows.length === 0 || cols === 0 || rows.some((r) => r.length !== cols)) {
    throw new WhiteboxError('value', 'fromRows: rows must be non-empty and of equal length')
  }
  const names = (options.terrains ?? TERRAINS).map((x) => x.name)
  const terrain = rows.flatMap((row) =>
    row.map((v) => {
      const k = typeof v === 'number' ? v : names.indexOf(v)
      if (!(k >= 0 && k < names.length)) throw new WhiteboxError('value', `fromRows: unknown terrain ${String(v)}`)
      return k
    }),
  )
  return world({
    topology: { kind: options.topology === 'hex' ? 'hex' : 'square', rows: rows.length, cols },
    terrain,
    ...(options.terrains === undefined ? {} : { terrains: options.terrains }),
    ...(options.starts === undefined ? {} : { starts: options.starts }),
    ...(options.goals === undefined ? {} : { goals: options.goals }),
    ...(options.name === undefined ? {} : { name: options.name }),
  })
}
