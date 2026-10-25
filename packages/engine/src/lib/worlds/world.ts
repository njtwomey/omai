/**
 * The world data type: cells on a square grid, a hexagonal grid or the nodes of a graph, with walls between cells,
 * terrains (passability, cost, sight), items (keys, doors, pickups), starts and goals. Construction and editing, cell
 * geometry in world units, and movement queries (passability, walls, move destinations, legal exits and neighbours
 * under 4- or 8-connectivity).
 */

import { MdmlError } from '../base/index.ts'
import type { Paint } from '../scene/index.ts'
import {
  cellAt,
  cellCount,
  centre,
  corners,
  directionCount,
  extent,
  oppositeSide,
  SQUARE_OFFSETS,
  sideBetween,
  sideCount,
  step,
} from './_topology.ts'

/**
 * The shape of a world's cells: a square grid (`rows x cols`, cell `r * cols + c`), a hexagonal grid of pointy-top
 * hexagons in odd-r offset coordinates (odd rows sit half a cell to the right; cell `r * cols + c`), or the nodes of a
 * graph given by neighbour lists (side `k` of node `v` faces `neighbours[v][k]`; the lists must be symmetric).
 *
 * @concept grid-worlds
 */
export type Topology =
  | { readonly kind: 'square'; readonly rows: number; readonly cols: number }
  | { readonly kind: 'hex'; readonly rows: number; readonly cols: number }
  | {
      readonly kind: 'graph'
      readonly n: number
      readonly neighbours: readonly (readonly number[])[]
      /** Drawing positions in world units, one `[x, y]` per node (default: a circle). */
      readonly positions?: readonly (readonly [number, number])[]
    }

/** What a cell is made of: whether it can be entered, what entering costs, and how it looks and behaves. */
export interface Terrain {
  readonly name: string
  /** Its character in ASCII maps. */
  readonly char: string
  readonly passable: boolean
  /** The cost of entering (default 1). */
  readonly cost: number
  /** Whether it blocks sight (default: when impassable). */
  readonly opaque: boolean
  /** Entering it is dangerous: holes, lava, cliffs and traps (rules decide what that means). */
  readonly hazard: boolean
  readonly paint: Paint
  /** A one-way cell: it may only be entered moving in this direction (a side index, as {@link DIRECTIONS}). */
  readonly direction?: number
}

/** A terrain as given to {@link world}: a name and character, everything else defaulted. */
export type TerrainSpec = Pick<Terrain, 'name' | 'char'> & Partial<Omit<Terrain, 'name' | 'char'>>

/** The kinds of item: keys (open doors of their colour), doors (need a key of their colour) and pickups. */
export type ItemKind = 'key' | 'door' | 'pickup'

/** An item lying in a cell. */
export interface Item {
  readonly kind: ItemKind
  readonly cell: number
  /** Pairs keys with doors (default `'yellow'`). */
  readonly colour: string
  /** The reward for collecting a pickup (rules decide; default 0). */
  readonly reward: number
}

/**
 * A world: cells on a square or hexagonal grid or the nodes of a graph; walls between cells (`walls[cell]` has bit `k`
 * set when side `k` is closed) or blocking cells (impassable terrain); per-cell terrain with costs; items (keys,
 * doors, pickups); start and goal sets; and named landmark cells. It is plain data: functions take it and return
 * new worlds, and every application (search problems, reinforcement learning environments, ray casting, figures)
 * reads the same value.
 *
 * @concept grid-worlds
 */
export interface World {
  readonly kind: 'world'
  readonly topology: Topology
  /** The number of cells. */
  readonly n: number
  readonly terrains: readonly Terrain[]
  /** The terrain index of each cell. */
  readonly terrain: Uint8Array
  /** Closed sides of each cell, one bit per side; always set on both cells of a wall. */
  readonly walls: Uint32Array
  readonly items: readonly Item[]
  readonly starts: readonly number[]
  readonly goals: readonly number[]
  /** Named cells (Taxi's R, G, Y and B; MiniGrid's rooms), in the order they were added. */
  readonly landmarks: Readonly<Record<string, readonly number[]>>
  readonly name?: string
}

/** The names of the move directions: square grids have four sides then four diagonals, hexagons six sides. */
export const DIRECTIONS: { readonly square: readonly string[]; readonly hex: readonly string[] } = {
  square: ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW'],
  hex: ['E', 'SE', 'SW', 'W', 'NW', 'NE'],
}

const terrain = (spec: TerrainSpec): Terrain => {
  const passable = spec.passable ?? true
  return {
    name: spec.name,
    char: spec.char,
    passable,
    cost: spec.cost ?? 1,
    opaque: spec.opaque ?? !passable,
    hazard: spec.hazard ?? false,
    paint: spec.paint ?? (passable ? 'surface' : 'fg'),
    ...(spec.direction === undefined ? {} : { direction: spec.direction }),
  }
}

/**
 * The default terrains: floor `.`, wall `#`, water `~` (cost 5), grass `,`, forest `%` (cost 2, blocks sight),
 * mountain `M` (impassable), the hazards hole `H`, lava `L`, cliff `C`, pit `X` and trap `T`, and one-way cells
 * `^ > v <` (entered only moving north, east, south and west).
 */
export const TERRAINS: readonly Terrain[] = [
  terrain({ name: 'floor', char: '.' }),
  terrain({ name: 'wall', char: '#', passable: false }),
  terrain({ name: 'water', char: '~', cost: 5, paint: 'tab0' }),
  terrain({ name: 'grass', char: ',', paint: 'tab8' }),
  terrain({ name: 'forest', char: '%', cost: 2, opaque: true, paint: 'tab2' }),
  terrain({ name: 'mountain', char: 'M', passable: false, paint: 'muted' }),
  terrain({ name: 'hole', char: 'H', hazard: true, paint: 'tab4' }),
  terrain({ name: 'lava', char: 'L', hazard: true, paint: 'tab1' }),
  terrain({ name: 'cliff', char: 'C', hazard: true, paint: 'tab3' }),
  terrain({ name: 'pit', char: 'X', hazard: true, paint: 'tab3' }),
  terrain({ name: 'trap', char: 'T', hazard: true, paint: 'tab6' }),
  terrain({ name: 'oneway-n', char: '^', direction: 0 }),
  terrain({ name: 'oneway-e', char: '>', direction: 1 }),
  terrain({ name: 'oneway-s', char: 'v', direction: 2 }),
  terrain({ name: 'oneway-w', char: '<', direction: 3 }),
]

/** An item as given to {@link world}. */
export interface ItemSpec {
  readonly kind: ItemKind
  readonly cell: number
  readonly colour?: string
  readonly reward?: number
}

/** Everything {@link world} needs: the topology and, optionally, terrains, cells, walls, items, starts and goals. */
export interface WorldSpec {
  readonly topology: Topology
  /** The terrain palette (default {@link TERRAINS}; index 0 is the default cell). */
  readonly terrains?: readonly TerrainSpec[]
  /** The terrain of each cell, by index or name (default all index 0). */
  readonly terrain?: ArrayLike<number> | readonly string[]
  /** Closed sides, as pairs of side-adjacent cells. */
  readonly walls?: readonly (readonly [number, number])[]
  readonly items?: readonly ItemSpec[]
  readonly starts?: readonly number[]
  readonly goals?: readonly number[]
  readonly landmarks?: Readonly<Record<string, readonly number[]>>
  readonly name?: string
}

function checkTopology(t: Topology): void {
  if (t.kind === 'graph') {
    if (!Number.isInteger(t.n) || t.n < 1 || t.neighbours.length !== t.n) {
      throw new MdmlError('value', 'world: a graph topology needs n >= 1 nodes and one neighbour list per node')
    }
    for (const [v, list] of t.neighbours.entries()) {
      if (list.length > 32) throw new MdmlError('value', `world: node ${v} has more than 32 neighbours`)
      for (const u of list) {
        if (!Number.isInteger(u) || u < 0 || u >= t.n || u === v || !(t.neighbours[u] ?? []).includes(v)) {
          throw new MdmlError('value', `world: neighbour lists must be symmetric, loop-free indices (node ${v})`)
        }
      }
    }
    return
  }
  if (!(Number.isInteger(t.rows) && Number.isInteger(t.cols) && t.rows >= 1 && t.cols >= 1)) {
    throw new MdmlError('value', 'world: rows and cols must be positive integers')
  }
}

function checkCell(n: number, cell: number, what: string): number {
  if (!Number.isInteger(cell) || cell < 0 || cell >= n) throw new MdmlError('value', `world: ${what} ${cell}`)
  return cell
}

/**
 * Build a world from a topology and its contents, checking every index.
 *
 * @param spec - The topology, terrains, cell terrain, walls, items, starts, goals, landmarks and name.
 * @returns The world.
 * @throws MdmlError with code `value` for bad sizes, unknown terrains, out-of-range cells or walls between cells
 *   that are not side neighbours.
 * @concept grid-worlds
 * @example
 * ```ts
 * import { world } from 'mdml-engine/worlds'
 * const w = world({ topology: { kind: 'square', rows: 2, cols: 3 }, walls: [[0, 1]], goals: [5] })
 * w.n // => 6
 * w.walls[0] // => 2
 * ```
 */
export function world(spec: WorldSpec): World {
  const t = spec.topology
  checkTopology(t)
  const n = cellCount(t)
  const terrains = (spec.terrains ?? TERRAINS).map(terrain)
  if (terrains.length === 0 || terrains.length > 256) throw new MdmlError('value', 'world: 1 to 256 terrains')
  const byName = new Map(terrains.map((x, i) => [x.name, i]))
  const cells = new Uint8Array(n)
  if (spec.terrain !== undefined) {
    if (spec.terrain.length !== n) throw new MdmlError('value', `world: terrain needs ${n} entries`)
    for (let i = 0; i < n; i++) {
      const v = spec.terrain[i]
      const k = typeof v === 'string' ? byName.get(v) : v
      if (k === undefined || !Number.isInteger(k) || k < 0 || k >= terrains.length) {
        throw new MdmlError('value', `world: unknown terrain ${String(v)} at cell ${i}`)
      }
      cells[i] = k
    }
  }
  const walls = new Uint32Array(n)
  for (const [a, b] of spec.walls ?? []) closeSide(t, walls, checkCell(n, a, 'wall cell'), checkCell(n, b, 'wall cell'))
  const items = (spec.items ?? []).map((it) => ({
    kind: it.kind,
    cell: checkCell(n, it.cell, 'item cell'),
    colour: it.colour ?? 'yellow',
    reward: it.reward ?? 0,
  }))
  const landmarks: Record<string, readonly number[]> = {}
  for (const [name, list] of Object.entries(spec.landmarks ?? {})) {
    landmarks[name] = list.map((c) => checkCell(n, c, 'landmark cell'))
  }
  return {
    kind: 'world',
    topology: t,
    n,
    terrains,
    terrain: cells,
    walls,
    items,
    starts: (spec.starts ?? []).map((c) => checkCell(n, c, 'start cell')),
    goals: (spec.goals ?? []).map((c) => checkCell(n, c, 'goal cell')),
    landmarks,
    ...(spec.name === undefined ? {} : { name: spec.name }),
  }
}

function closeSide(t: Topology, walls: Uint32Array, a: number, b: number, closed = true): void {
  const side = sideBetween(t, a, b)
  if (side < 0 || side >= sideCount(t, a)) {
    throw new MdmlError('value', `world: cells ${a} and ${b} do not share a side`)
  }
  const back = oppositeSide(t, a, side, b)
  if (closed) {
    walls[a] = (walls[a] ?? 0) | (1 << side)
    walls[b] = (walls[b] ?? 0) | (1 << back)
  } else {
    walls[a] = (walls[a] ?? 0) & ~(1 << side)
    walls[b] = (walls[b] ?? 0) & ~(1 << back)
  }
}

/**
 * Close or open the walls between pairs of side-adjacent cells.
 *
 * @param w - The world.
 * @param pairs - Pairs of cells sharing a side.
 * @param closed - Close (default) or open them.
 * @returns A new world.
 * @throws MdmlError with code `value` when two cells do not share a side.
 * @example
 * ```ts
 * import { hasWall, setWalls, world } from 'mdml-engine/worlds'
 * const w = setWalls(world({ topology: { kind: 'square', rows: 1, cols: 2 } }), [[0, 1]])
 * hasWall(w, 1, 3) // => true
 * ```
 */
export function setWalls(w: World, pairs: readonly (readonly [number, number])[], closed = true): World {
  const walls = Uint32Array.from(w.walls)
  for (const [a, b] of pairs)
    closeSide(w.topology, walls, checkCell(w.n, a, 'wall cell'), checkCell(w.n, b, 'wall cell'), closed)
  return { ...w, walls }
}

/**
 * Set the terrain of some cells.
 *
 * @param w - The world.
 * @param cells - The cells.
 * @param name - A terrain name from the world's palette.
 * @returns A new world.
 * @throws MdmlError with code `value` for an unknown terrain or cell.
 * @example
 * ```ts
 * import { passable, setTerrain, world } from 'mdml-engine/worlds'
 * passable(setTerrain(world({ topology: { kind: 'square', rows: 1, cols: 2 } }), [1], 'wall'), 1) // => false
 * ```
 */
export function setTerrain(w: World, cells: readonly number[], name: string): World {
  const k = w.terrains.findIndex((x) => x.name === name)
  if (k < 0) throw new MdmlError('value', `setTerrain: unknown terrain '${name}'`)
  const out = Uint8Array.from(w.terrain)
  for (const c of cells) out[checkCell(w.n, c, 'cell')] = k
  return { ...w, terrain: out }
}

// ------------------------------------------------------------------------------------------ cells and coordinates

/**
 * The cell at a row and column of a grid world.
 *
 * @param w - A square or hexagonal world.
 * @param row - The row.
 * @param col - The column.
 * @returns The cell index `row * cols + col`, or -1 off the grid.
 * @throws MdmlError with code `value` for graph worlds.
 * @example
 * ```ts
 * import { cellIndex, fromAscii } from 'mdml-engine/worlds'
 * cellIndex(fromAscii(['...', '...']), 1, 2) // => 5
 * ```
 */
export function cellIndex(w: World, row: number, col: number): number {
  const t = w.topology
  if (t.kind === 'graph') throw new MdmlError('value', 'cellIndex: graph worlds have no rows')
  return row < 0 || row >= t.rows || col < 0 || col >= t.cols ? -1 : row * t.cols + col
}

/**
 * The row and column of a cell of a grid world.
 *
 * @param w - A square or hexagonal world.
 * @param cell - The cell.
 * @returns `[row, col]`.
 * @throws MdmlError with code `value` for graph worlds.
 * @example
 * ```ts
 * import { cellCoords, fromAscii } from 'mdml-engine/worlds'
 * cellCoords(fromAscii(['...', '...']), 5) // => [1, 2]
 * ```
 */
export function cellCoords(w: World, cell: number): [number, number] {
  const t = w.topology
  if (t.kind === 'graph') throw new MdmlError('value', 'cellCoords: graph worlds have no rows')
  return [Math.floor(cell / t.cols), cell % t.cols]
}

/**
 * The centre of a cell in world units: square cells are unit squares (`x` the column, `y` the row, growing down as in
 * scenes); hexagons are one unit wide; graph nodes sit at their positions.
 *
 * @param w - The world.
 * @param cell - The cell.
 * @returns `[x, y]`.
 * @example
 * ```ts
 * import { cellCentre, fromAscii } from 'mdml-engine/worlds'
 * cellCentre(fromAscii(['...', '...']), 5) // => [2.5, 1.5]
 * ```
 */
export function cellCentre(w: World, cell: number): [number, number] {
  return centre(w.topology, cell)
}

/**
 * The cell containing a point in world units.
 *
 * @param w - The world.
 * @param x - The x coordinate.
 * @param y - The y coordinate.
 * @returns The cell, or -1 off the map (always -1 for graph worlds).
 * @example
 * ```ts
 * import { cellAtPoint, fromAscii } from 'mdml-engine/worlds'
 * cellAtPoint(fromAscii(['...', '...']), 2.2, 1.9) // => 5
 * cellAtPoint(fromAscii(['...', '...'], { topology: 'hex' }), 1, 1.4) // => 3
 * ```
 */
export function cellAtPoint(w: World, x: number, y: number): number {
  return cellAt(w.topology, x, y)
}

/**
 * The corners of a cell in world units, in order, so side `k` runs from corner `k` to corner `k + 1`.
 *
 * @param w - A square or hexagonal world.
 * @param cell - The cell.
 * @returns The corners (four or six; none for graph worlds).
 * @example
 * ```ts
 * import { cellCorners, fromAscii } from 'mdml-engine/worlds'
 * cellCorners(fromAscii(['..']), 1) // => [[1, 0], [2, 0], [2, 1], [1, 1]]
 * ```
 */
export function cellCorners(w: World, cell: number): [number, number][] {
  return corners(w.topology, cell)
}

/**
 * The width and height of a world in world units.
 *
 * @param w - The world.
 * @returns `[width, height]`.
 * @example
 * ```ts
 * import { fromAscii, worldSize } from 'mdml-engine/worlds'
 * worldSize(fromAscii(['...', '...'])) // => [3, 2]
 * ```
 */
export function worldSize(w: World): [number, number] {
  return extent(w.topology)
}

// ---------------------------------------------------------------------------------- neighbours and passability

/** How cells connect: 4 or 8 neighbours on square grids, how diagonals treat corners, and extra blocked cells. */
export interface MoveOptions {
  /** Square grids only: 4 (default) or 8 neighbours. Hexagonal grids always have 6; graphs their edges. */
  readonly connectivity?: 4 | 8
  /**
   * Diagonal moves on square grids: `'strict'` (default) needs both orthogonal routes around the corner open, `'loose'`
   * needs one.
   */
  readonly corners?: 'strict' | 'loose'
  /** Cells to treat as blocked as well (closed doors, other agents). */
  readonly blocked?: (cell: number) => boolean
}

/**
 * Whether a cell can be entered at all: on the map and of passable terrain.
 *
 * @param w - The world.
 * @param cell - The cell.
 * @returns Whether it is passable.
 * @example
 * ```ts
 * import { fromAscii, passable } from 'mdml-engine/worlds'
 * [passable(fromAscii(['.#']), 0), passable(fromAscii(['.#']), 1)] // => [true, false]
 * ```
 */
export function passable(w: World, cell: number): boolean {
  return cell >= 0 && cell < w.n && (w.terrains[w.terrain[cell] ?? 0]?.passable ?? false)
}

/**
 * The terrain of a cell.
 *
 * @param w - The world.
 * @param cell - The cell.
 * @returns Its terrain.
 * @throws MdmlError with code `value` for a cell off the map.
 * @example
 * ```ts
 * import { fromAscii, terrainOf } from 'mdml-engine/worlds'
 * terrainOf(fromAscii(['.~']), 1).cost // => 5
 * ```
 */
export function terrainOf(w: World, cell: number): Terrain {
  const t = w.terrains[w.terrain[checkCell(w.n, cell, 'cell')] ?? 0]
  if (t === undefined) throw new MdmlError('value', `terrainOf: cell ${cell} has no terrain`)
  return t
}

/**
 * Whether side `side` of a cell is walled (the map edge is not a wall; moving off the map is simply impossible).
 *
 * @param w - The world.
 * @param cell - The cell.
 * @param side - The side (`0..3` N, E, S, W on squares; `0..5` on hexagons; the neighbour slot on graphs).
 * @returns Whether that side is closed.
 * @example
 * ```ts
 * import { fromAscii, hasWall } from 'mdml-engine/worlds'
 * hasWall(fromAscii(['#####', '#.#.#', '#####'], { format: 'walls' }), 0, 1) // => true
 * ```
 */
export function hasWall(w: World, cell: number, side: number): boolean {
  return side >= 0 && side < 32 && (((w.walls[cell] ?? 0) >>> side) & 1) === 1
}

function enterable(w: World, cell: number, direction: number, options: MoveOptions): boolean {
  if (!passable(w, cell) || options.blocked?.(cell) === true) return false
  const dir = w.terrains[w.terrain[cell] ?? 0]?.direction
  return dir === undefined || dir === direction
}

function sideMove(w: World, cell: number, side: number, options: MoveOptions): number {
  const to = step(w.topology, cell, side)
  if (to < 0 || hasWall(w, cell, side) || !enterable(w, to, side, options)) return -1
  return to
}

/**
 * Where a move goes: the neighbouring cell in `direction`, or -1 when the move is impossible (off the map, through a
 * wall, into impassable or blocked terrain, against a one-way cell, or a diagonal cutting a blocked corner).
 *
 * @param w - The world.
 * @param cell - The cell moved from.
 * @param direction - The direction (as {@link DIRECTIONS}; on graphs, the neighbour slot).
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns The cell entered, or -1.
 * @example
 * ```ts
 * import { destination, fromAscii } from 'mdml-engine/worlds'
 * const w = fromAscii(['..', '#.'])
 * [destination(w, 0, 1), destination(w, 0, 2), destination(w, 0, 5, { connectivity: 8 })] // => [1, -1, -1]
 * destination(w, 0, 5, { connectivity: 8, corners: 'loose' }) // => 3
 * ```
 */
export function destination(w: World, cell: number, direction: number, options: MoveOptions = {}): number {
  const t = w.topology
  if (cell < 0 || cell >= w.n) return -1
  if (t.kind !== 'square' || direction < 4) {
    return direction < 0 || direction >= sideCount(t, cell) ? -1 : sideMove(w, cell, direction, options)
  }
  if ((options.connectivity ?? 4) !== 8 || direction >= 8) return -1
  const to = step(t, cell, direction)
  if (to < 0 || !enterable(w, to, direction, options)) return -1
  const [dr, dc] = SQUARE_OFFSETS[direction] ?? [0, 0]
  const vertical = dr < 0 ? 0 : 2
  const horizontal = dc > 0 ? 1 : 3
  const via = (first: number, second: number): boolean => {
    const mid = sideMove(w, cell, first, options)
    return mid >= 0 && !hasWall(w, mid, second) && step(t, mid, second) === to
  }
  const a = via(vertical, horizontal)
  const b = via(horizontal, vertical)
  return (options.corners === 'loose' ? a || b : a && b) ? to : -1
}

/**
 * The legal moves from a cell: each direction that leads somewhere, with the cell it reaches, in direction order.
 *
 * @param w - The world.
 * @param cell - The cell.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns `[direction, cell]` pairs.
 * @example
 * ```ts
 * import { exits, fromAscii } from 'mdml-engine/worlds'
 * exits(fromAscii(['...', '.#.']), 1) // => [[1, 2], [3, 0]]
 * ```
 */
export function exits(w: World, cell: number, options: MoveOptions = {}): [number, number][] {
  const out: [number, number][] = []
  const k =
    w.topology.kind === 'square' ? ((options.connectivity ?? 4) === 8 ? 8 : 4) : directionCount(w.topology, cell)
  for (let d = 0; d < k; d++) {
    const to = destination(w, cell, d, options)
    if (to >= 0) out.push([d, to])
  }
  return out
}

/**
 * The cells reachable in one move from a cell, under 4- or 8-connectivity on square grids (6 on hexagons, the edges
 * of a graph), respecting walls, terrain, one-way cells and corners.
 *
 * @param w - The world.
 * @param cell - The cell.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns The neighbours, in direction order.
 * @example
 * ```ts
 * import { fromAscii, neighbours } from 'mdml-engine/worlds'
 * const w = fromAscii(['...', '...', '...'])
 * [neighbours(w, 4).length, neighbours(w, 4, { connectivity: 8 }).length] // => [4, 8]
 * ```
 */
export function neighbours(w: World, cell: number, options: MoveOptions = {}): number[] {
  return exits(w, cell, options).map(([, to]) => to)
}

/**
 * Turn walls between cells into blocking cells: a square world of `R x C` cells becomes a `(2R + 1) x (2C + 1)` world
 * whose odd-odd cells are the originals, whose cells between them are wall where a wall stood (floor otherwise), and
 * whose border and corners are wall. Ray casters and occupancy-grid methods want this form.
 *
 * @param w - A square-grid world.
 * @returns The blocking-cell world (starts, goals, items and landmarks moved with their cells).
 * @throws MdmlError with code `value` for other topologies, or a palette without `floor` and `wall`.
 * @example
 * ```ts
 * import { fromAscii, toAscii, wallsToBlocks } from 'mdml-engine/worlds'
 * toAscii(wallsToBlocks(fromAscii(['#####', '#S#G#', '#####'], { format: 'walls' }))) // => ['#####', '#S#G#', '#####']
 * ```
 */
export function wallsToBlocks(w: World): World {
  const t = w.topology
  if (t.kind !== 'square') throw new MdmlError('value', 'wallsToBlocks: square worlds only')
  const floor = w.terrains.findIndex((x) => x.name === 'floor')
  const wall = w.terrains.findIndex((x) => x.name === 'wall')
  if (floor < 0 || wall < 0) throw new MdmlError('value', 'wallsToBlocks: the palette needs floor and wall')
  const cols = 2 * t.cols + 1
  const map = (cell: number): number => (2 * Math.floor(cell / t.cols) + 1) * cols + 2 * (cell % t.cols) + 1
  const terrain = new Array<number>((2 * t.rows + 1) * cols).fill(wall)
  for (let cell = 0; cell < w.n; cell++) {
    const m = map(cell)
    terrain[m] = w.terrain[cell] ?? floor
    const [r, c] = [Math.floor(cell / t.cols), cell % t.cols]
    if (c + 1 < t.cols) terrain[m + 1] = hasWall(w, cell, 1) ? wall : floor
    if (r + 1 < t.rows) terrain[m + cols] = hasWall(w, cell, 2) ? wall : floor
  }
  return world({
    topology: { kind: 'square', rows: 2 * t.rows + 1, cols },
    terrains: w.terrains,
    terrain,
    items: w.items.map((it) => ({ ...it, cell: map(it.cell) })),
    starts: w.starts.map(map),
    goals: w.goals.map(map),
    landmarks: Object.fromEntries(Object.entries(w.landmarks).map(([k, cells]) => [k, cells.map(map)])),
    ...(w.name === undefined ? {} : { name: w.name }),
  })
}
