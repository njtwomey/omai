/**
 * Procedural level generation from keys: dungeons of rooms and corridors by binary space partitioning, caves grown by a
 * cellular automaton (as a traced algorithm) and reduced to their largest region, value and Perlin noise, and terrain
 * maps cut from noise into bands such as water, grass, forest and mountain.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { foldIn, type Key, uniform, uniformStream } from '../random/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { distanceField, regions } from './analysis.ts'
import { TERRAINS, type TerrainSpec, type World, world } from './world.ts'

const positive = (n: number, what: string): number => {
  if (!Number.isInteger(n) || n < 1) throw new MdmlError('value', `worlds: ${what} must be a positive integer`)
  return n
}

// ------------------------------------------------------------------------------------------------- BSP dungeons

/** A room of a dungeon: its top-left cell and size. */
export interface Room {
  readonly row: number
  readonly col: number
  readonly height: number
  readonly width: number
}

/** A dungeon: the world, its rooms in BSP leaf order, and the corridors as the cells they carve. */
export interface Dungeon {
  readonly world: World
  readonly rooms: readonly Room[]
  readonly corridors: readonly (readonly number[])[]
}

/** Options of {@link bspDungeon}. */
export interface DungeonOptions {
  /** Map size in cells, walls included (default 32 x 48). */
  readonly rows?: number
  readonly cols?: number
  /** The smallest leaf side a split may leave (default 6). */
  readonly minLeaf?: number
  /** Split depth cap (default 5, at most 32 leaves). */
  readonly maxDepth?: number
}

/**
 * Rooms and corridors by binary space partitioning: split the map recursively (along its longer side, at a random
 * position leaving leaves at least `minLeaf` across), carve a random room inside each leaf, and join sibling subtrees
 * by an L-shaped corridor between random rooms of each, so every room is connected. The start is the centre of the
 * first room and the goal the centre of the last.
 *
 * @param key - The key.
 * @param options - Size, leaf size and depth.
 * @returns The dungeon.
 * @throws MdmlError with code `value` for a map smaller than a leaf.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { bspDungeon, regions } from 'mdml-engine/worlds'
 * import { key } from 'mdml-engine/random'
 * const d = bspDungeon(key(0), { rows: 20, cols: 30 })
 * [d.rooms.length > 1, regions(d.world).count] // => [true, 1]
 * ```
 */
export function bspDungeon(key: Key, options: DungeonOptions = {}): Dungeon {
  const rows = positive(options.rows ?? 32, 'rows')
  const cols = positive(options.cols ?? 48, 'cols')
  const minLeaf = Math.max(3, options.minLeaf ?? 6)
  const maxDepth = Math.min(10, Math.max(0, options.maxDepth ?? 5))
  if (rows < minLeaf || cols < minLeaf) throw new MdmlError('value', 'bspDungeon: the map is smaller than a leaf')
  const next = uniformStream(key)
  const int = (lo: number, hi: number): number => lo + Math.min(hi - lo, Math.floor(next() * (hi - lo + 1)))
  const floor = new Uint8Array(rows * cols)
  const rooms: Room[] = []
  const corridors: number[][] = []
  const carve = (r: number, c: number): void => {
    if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) floor[r * cols + c] = 1
  }
  const centre = (room: Room): [number, number] => [
    room.row + Math.floor(room.height / 2),
    room.col + Math.floor(room.width / 2),
  ]
  // Returns the indices of the rooms made inside the leaf.
  const build = (r0: number, c0: number, h: number, w: number, depth: number): number[] => {
    const canRows = h >= 2 * minLeaf
    const canCols = w >= 2 * minLeaf
    if (depth < maxDepth && (canRows || canCols)) {
      const alongRows = canRows && (!canCols || h > w || (h === w && next() < 0.5))
      const at = alongRows ? int(minLeaf, h - minLeaf) : int(minLeaf, w - minLeaf)
      const a = alongRows ? build(r0, c0, at, w, depth + 1) : build(r0, c0, h, at, depth + 1)
      const b = alongRows ? build(r0 + at, c0, h - at, w, depth + 1) : build(r0, c0 + at, h, w - at, depth + 1)
      const ra = rooms[a[int(0, a.length - 1)] ?? 0]
      const rb = rooms[b[int(0, b.length - 1)] ?? 0]
      if (ra !== undefined && rb !== undefined) {
        const [y1, x1] = centre(ra)
        const [y2, x2] = centre(rb)
        const cells: number[] = []
        // An L: along the first room's row then the second's column, or the first's column then the second's row.
        const [elbowY, elbowX] = next() < 0.5 ? [y1, x2] : [y2, x1]
        for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) cells.push(elbowY * cols + x)
        for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) cells.push(y * cols + elbowX)
        for (const cell of cells) carve(Math.floor(cell / cols), cell % cols)
        corridors.push([...new Set(cells)])
      }
      return [...a, ...b]
    }
    // A room inside the leaf, with at least one wall cell to every side.
    const height = int(Math.max(1, Math.floor((h - 2) / 2)), h - 2)
    const width = int(Math.max(1, Math.floor((w - 2) / 2)), w - 2)
    const row = r0 + int(1, h - 1 - height)
    const col = c0 + int(1, w - 1 - width)
    const room = { row, col, height, width }
    for (let y = row; y < row + height; y++) for (let x = col; x < col + width; x++) carve(y, x)
    rooms.push(room)
    return [rooms.length - 1]
  }
  build(0, 0, rows, cols, 0)
  const [sr, sc] = centre(rooms[0] ?? { row: 1, col: 1, height: 1, width: 1 })
  const [gr, gc] = centre(rooms[rooms.length - 1] ?? { row: 1, col: 1, height: 1, width: 1 })
  const w = world({
    topology: { kind: 'square', rows, cols },
    terrain: Array.from(floor, (f) => (f === 1 ? 0 : 1)),
    starts: [sr * cols + sc],
    goals: [gr * cols + gc],
    name: 'BSP dungeon',
  })
  return { world: w, rooms, corridors }
}

// ------------------------------------------------------------------------------------------- cellular-automaton caves

/** Options of {@link caveAlgorithm} and {@link cave}. */
export interface CaveOptions {
  readonly rows?: number
  readonly cols?: number
  /** The initial probability of wall (default 0.45). */
  readonly fill?: number
  /** Generations (default 5). */
  readonly generations?: number
  /** A cell becomes wall with at least this many walls among its 8 neighbours (default 5)... */
  readonly birth?: number
  /** ...and stays wall with at least this many (default 4). Cells off the map count as walls. */
  readonly survival?: number
}

/** The state of {@link caveAlgorithm}: wall flags per cell and the generation. */
export interface CaveState {
  readonly wall: Uint8Array
  readonly generation: number
}

/**
 * Cellular-automaton caves as a traced {@link Algorithm} (the "4-5 rule"): start from random walls with probability
 * `fill` (the border always wall), then each step applies one generation: a floor cell becomes wall when at least
 * `birth` of its 8 neighbours are walls, a wall stays wall with at least `survival`, counting cells off the map as
 * walls. The run stops after `generations` steps.
 *
 * @param options - Size, fill, generations and the rule.
 * @returns The algorithm.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { caveAlgorithm } from 'mdml-engine/worlds'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(caveAlgorithm({ rows: 10, cols: 10 }), { seed: 0, maxSteps: 100 }).steps // => 5
 * ```
 */
export function caveAlgorithm(options: CaveOptions = {}): Algorithm<CaveState> {
  const rows = positive(options.rows ?? 24, 'rows')
  const cols = positive(options.cols ?? 40, 'cols')
  const fill = options.fill ?? 0.45
  const generations = Math.max(0, Math.floor(options.generations ?? 5))
  const birth = options.birth ?? 5
  const survival = options.survival ?? 4
  const border = (r: number, c: number): boolean => r === 0 || c === 0 || r === rows - 1 || c === cols - 1
  return {
    name: 'worlds/cave',
    config: { rows, cols, fill, generations, birth, survival } as Json,
    schema: channels({
      metrics: { wallFraction: { description: 'fraction of cells that are wall' } },
      tensors: { wall: { role: 'state', description: 'wall flags', shape: [rows, cols] } },
    }),
    init: (key) => {
      const u = uniform(key, [rows * cols]).data
      const wall = new Uint8Array(rows * cols)
      for (let i = 0; i < rows * cols; i++) {
        wall[i] = border(Math.floor(i / cols), i % cols) || (u[i] ?? 1) < fill ? 1 : 0
      }
      return { wall, generation: 0 }
    },
    step: (s) => {
      const wall = new Uint8Array(rows * cols)
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          let n = 0
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              if (dr === 0 && dc === 0) continue
              const r2 = r + dr
              const c2 = c + dc
              n += r2 < 0 || c2 < 0 || r2 >= rows || c2 >= cols ? 1 : (s.wall[r2 * cols + c2] ?? 1)
            }
          }
          const i = r * cols + c
          wall[i] = border(r, c) || n >= (s.wall[i] === 1 ? survival : birth) ? 1 : 0
        }
      }
      return { wall, generation: s.generation + 1 }
    },
    done: (s) => (s.generation >= generations ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { wallFraction: s.wall.reduce((a, b) => a + b, 0) / (rows * cols) },
      tensors: { wall: { role: 'state', value: s.wall, shape: [rows, cols] } },
    }),
  }
}

/**
 * Generate a cave: run {@link caveAlgorithm}, then (by default) keep only the largest open region, filling the others
 * with wall, and put the start and goal at the two ends of that region's longest breadth-first path from its first
 * cell (approximately its diameter).
 *
 * @param key - The key.
 * @param options - The cave options, `keepLargest` (default true) and run options.
 * @returns The cave world.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { cave, regions } from 'mdml-engine/worlds'
 * import { key } from 'mdml-engine/random'
 * regions(cave(key(2), { rows: 20, cols: 30 })).count // => 1
 * ```
 */
export function cave(
  key: Key,
  options: CaveOptions & { readonly keepLargest?: boolean } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent'>
    > = {},
): World {
  const { keepLargest, tracer, policy, parent, ...rest } = options
  const algorithm = caveAlgorithm(rest)
  const run = runAlgorithm(algorithm, {
    key,
    maxSteps: (rest.generations ?? 5) + 1,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
  })
  const rows = rest.rows ?? 24
  const cols = rest.cols ?? 40
  let w = world({ topology: { kind: 'square', rows, cols }, terrain: Array.from(run.state.wall), name: 'cave' })
  const reg = regions(w)
  if (reg.count === 0) return w
  let largest = 0
  for (let i = 1; i < reg.count; i++) if ((reg.sizes[i] ?? 0) > (reg.sizes[largest] ?? 0)) largest = i
  if (keepLargest !== false) {
    w = world({
      topology: w.topology,
      terrain: Array.from(reg.labels, (l) => (l === largest ? 0 : 1)),
      name: 'cave',
    })
  }
  const first = reg.labels.indexOf(largest)
  const far = (from: number): number => {
    const d = distanceField(w, from)
    let best = from
    for (let c = 0; c < w.n; c++) if (Number.isFinite(d[c] ?? 0) && (d[c] ?? 0) > (d[best] ?? 0)) best = c
    return best
  }
  const a = far(first)
  const b = far(a)
  return { ...w, starts: [a], goals: [b] }
}

// ------------------------------------------------------------------------------------------------------------ noise

/** Options of {@link valueNoise} and {@link perlinNoise}. */
export interface NoiseOptions {
  /** Cells per lattice period of the first octave (default 8). */
  readonly scale?: number
  /** Octaves summed, each `lacunarity` times finer and `persistence` times weaker (defaults 1, 2, 0.5). */
  readonly octaves?: number
  readonly lacunarity?: number
  readonly persistence?: number
}

const fade = (t: number): number => t * t * t * (t * (t * 6 - 15) + 10)

/**
 * Sum octaves of lattice noise: octave `o` has period `scale / lacunarity^o` cells and amplitude `persistence^o`, and
 * draws its lattice (one uniform per lattice point, row-major) from `foldIn(key, o)`; `sample` reads one point.
 */
function octaves(
  key: Key,
  rows: number,
  cols: number,
  options: NoiseOptions,
  sample: (lattice: Float64Array, width: number, x: number, y: number) => number,
): Float64Array {
  positive(rows, 'rows')
  positive(cols, 'cols')
  const n = Math.max(1, Math.floor(options.octaves ?? 1))
  const lac = options.lacunarity ?? 2
  const per = options.persistence ?? 0.5
  const out = new Float64Array(rows * cols)
  let amplitude = 1
  let total = 0
  let period = options.scale ?? 8
  if (!(period > 0)) throw new MdmlError('value', 'worlds: the noise scale must be positive')
  for (let o = 0; o < n; o++) {
    const width = Math.floor((cols - 1) / period) + 2
    const height = Math.floor((rows - 1) / period) + 2
    const lattice = uniform(foldIn(key, o), [width * height]).data as Float64Array
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out[r * cols + c] = (out[r * cols + c] ?? 0) + amplitude * sample(lattice, width, c / period, r / period)
      }
    }
    total += amplitude
    amplitude *= per
    period /= lac
  }
  // Adding 0 turns any -0 into 0.
  for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / total + 0
  return out
}

/**
 * Value noise: random values at the points of a lattice (one per `scale` cells), smoothly interpolated with the
 * quintic fade `6t^5 - 15t^4 + 10t^3`; with octaves, the normalised sum of finer, weaker copies (fractal Brownian
 * motion). Values lie in `[0, 1]`, and a single octave equals its lattice value at every lattice point.
 *
 * @param key - The key.
 * @param rows - Rows of the field.
 * @param cols - Columns of the field.
 * @param options - Scale and octaves.
 * @returns The field, row-major.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { valueNoise } from 'mdml-engine/worlds'
 * import { key } from 'mdml-engine/random'
 * const f = valueNoise(key(0), 16, 16, { octaves: 3 })
 * [f.length, f.every((v) => v >= 0 && v <= 1)] // => [256, true]
 * ```
 */
export function valueNoise(key: Key, rows: number, cols: number, options: NoiseOptions = {}): Float64Array {
  return octaves(key, rows, cols, options, (v, lw, x, y) => {
    const at = (i: number, j: number): number => v[j * lw + i] ?? 0
    const i = Math.floor(x)
    const j = Math.floor(y)
    const sx = fade(x - i)
    const sy = fade(y - j)
    const top = at(i, j) + sx * (at(i + 1, j) - at(i, j))
    const bottom = at(i, j + 1) + sx * (at(i + 1, j + 1) - at(i, j + 1))
    return top + sy * (bottom - top)
  })
}

/**
 * Perlin's gradient noise (improved, 2002): random unit gradients at lattice points, the dot products with the offsets
 * to each corner interpolated with the quintic fade; zero at every lattice point and within `[-1, 1]` (a single
 * octave stays within `sqrt(2) / 2`). Octaves sum finer, weaker copies, normalised.
 *
 * @param key - The key.
 * @param rows - Rows of the field.
 * @param cols - Columns of the field.
 * @param options - Scale and octaves.
 * @returns The field, row-major.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { perlinNoise } from 'mdml-engine/worlds'
 * import { key } from 'mdml-engine/random'
 * perlinNoise(key(0), 9, 9, { scale: 4 })[4 * 9 + 4] // => 0
 * ```
 */
export function perlinNoise(key: Key, rows: number, cols: number, options: NoiseOptions = {}): Float64Array {
  return octaves(key, rows, cols, options, (angles, lw, x, y) => {
    const grad = (i: number, j: number, dx: number, dy: number): number => {
      const a = 2 * Math.PI * (angles[j * lw + i] ?? 0)
      return Math.cos(a) * dx + Math.sin(a) * dy
    }
    const i = Math.floor(x)
    const j = Math.floor(y)
    const fx = x - i
    const fy = y - j
    const sx = fade(fx)
    const sy = fade(fy)
    const n00 = grad(i, j, fx, fy)
    const n10 = grad(i + 1, j, fx - 1, fy)
    const n01 = grad(i, j + 1, fx, fy - 1)
    const n11 = grad(i + 1, j + 1, fx - 1, fy - 1)
    const top = n00 + sx * (n10 - n00)
    const bottom = n01 + sx * (n11 - n01)
    return top + sy * (bottom - top)
  })
}

/** A band of a noise terrain: cells with normalised noise below `below` get `terrain`. */
export interface TerrainBand {
  readonly below: number
  readonly terrain: string
}

/**
 * Terrain from noise: a value or Perlin noise field, rescaled to `[0, 1]`, cut into bands (by default water below
 * 0.35, grass to 0.6, forest to 0.8, then impassable mountain), so movement costs and sight vary across a natural
 * looking map. The start and goal are the passable cells of lowest and highest noise among the non-water bands.
 *
 * @param key - The key.
 * @param options - Size, the noise kind and options, the bands and extra terrains.
 * @returns The world.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { noiseTerrain, terrainOf } from 'mdml-engine/worlds'
 * import { key } from 'mdml-engine/random'
 * const w = noiseTerrain(key(0), { rows: 12, cols: 12 })
 * new Set(Array.from({ length: w.n }, (_, c) => terrainOf(w, c).name)).size > 1 // => true
 * ```
 */
export function noiseTerrain(
  key: Key,
  options: NoiseOptions & {
    readonly rows?: number
    readonly cols?: number
    readonly noise?: 'perlin' | 'value'
    readonly bands?: readonly TerrainBand[]
    readonly terrains?: readonly TerrainSpec[]
  } = {},
): World {
  const rows = options.rows ?? 32
  const cols = options.cols ?? 32
  const field = (options.noise === 'value' ? valueNoise : perlinNoise)(key, rows, cols, { octaves: 3, ...options })
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of field) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const bands = options.bands ?? [
    { below: 0.35, terrain: 'water' },
    { below: 0.6, terrain: 'grass' },
    { below: 0.8, terrain: 'forest' },
    { below: Number.POSITIVE_INFINITY, terrain: 'mountain' },
  ]
  const terrains = [...TERRAINS, ...(options.terrains ?? [])]
  const passable = new Map(terrains.map((t) => [t.name, t.passable ?? true]))
  const names = Array.from(field, (v) => {
    const x = hi > lo ? (v - lo) / (hi - lo) : 0
    return (bands.find((b) => x < b.below) ?? bands[bands.length - 1])?.terrain ?? 'floor'
  })
  let start = -1
  let goal = -1
  for (let i = 0; i < names.length; i++) {
    const t = names[i] ?? 'floor'
    if (!(passable.get(t) ?? true) || t === 'water') continue
    if (start < 0 || (field[i] ?? 0) < (field[start] ?? 0)) start = i
    if (goal < 0 || (field[i] ?? 0) > (field[goal] ?? 0)) goal = i
  }
  return world({
    topology: { kind: 'square', rows, cols },
    terrains,
    terrain: names,
    starts: start < 0 ? [] : [start],
    goals: goal < 0 ? [] : [goal],
    name: `${options.noise ?? 'perlin'} terrain`,
  })
}
