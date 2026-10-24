import { type Json, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { oppositeSide, sideBetween, sideCount, step } from './_topology.ts'
import { hasWall, type Topology, type World, world } from './world.ts'

/**
 * The maze algorithms. On any topology (square, hexagonal, graph): the recursive backtracker (randomised depth-first
 * search), randomised Prim's and Kruskal's algorithms, Wilson's algorithm (loop-erased random walks) and the
 * Aldous-Broder algorithm (a random walk); the last two draw uniform spanning trees. On square grids only: Eller's
 * algorithm (row by row), the binary tree, the sidewinder and recursive division.
 */
export type MazeAlgorithmName =
  | 'backtracker'
  | 'prim'
  | 'kruskal'
  | 'wilson'
  | 'aldousBroder'
  | 'eller'
  | 'binaryTree'
  | 'sidewinder'
  | 'division'

/** Options of the maze generators. */
export interface MazeOptions {
  /** Cells per column (default 8). */
  readonly rows?: number
  /** Cells per row (default 8). */
  readonly cols?: number
  /** Square (default) or hexagonal cells. */
  readonly topology?: 'square' | 'hex'
  /** Carve over this world's cells instead (its walls are replaced; its terrain, items, starts and goals kept). */
  readonly base?: World
  /** The algorithm (default `'backtracker'`). */
  readonly algorithm?: MazeAlgorithmName
  /** The corner the binary tree carves towards (default `'NE'`). */
  readonly bias?: 'NE' | 'NW' | 'SE' | 'SW'
}

/**
 * The state of a maze generator: the world being carved and its walls, the cells already in the maze, the cell being
 * worked on, and the algorithm's own bookkeeping (a stack or walk, a frontier, an edge order with a union-find, a row
 * or cell index, row sets or chambers).
 */
export interface MazeState {
  readonly base: World
  readonly walls: Uint32Array
  readonly inMaze: Uint8Array
  /** The cell being worked on (-1 when none). */
  readonly current: number
  /** Cells not yet in the maze. */
  readonly remaining: number
  /** Passages opened (walls added, for recursive division). */
  readonly carved: number
  /** The backtracker's stack, or Wilson's current loop-erased walk. */
  readonly stack: readonly number[]
  /** Prim's frontier of `[inside, outside]` cell pairs. */
  readonly frontier: readonly (readonly [number, number])[]
  /** Kruskal's edges, in random order, and its union-find parents. */
  readonly edges: readonly (readonly [number, number])[]
  readonly parent: Int32Array
  /** The next edge (Kruskal), cell (binary tree, sidewinder, Wilson's scan) or row (Eller). */
  readonly index: number
  /** Eller's set of each cell in the current row; the sidewinder's run start in `sets[0]`. */
  readonly sets: Int32Array
  /** Recursive division's chambers still to divide, `[row, col, height, width]`. */
  readonly chambers: readonly (readonly [number, number, number, number])[]
  readonly done: boolean
}

const SQUARE_ONLY: readonly MazeAlgorithmName[] = ['eller', 'binaryTree', 'sidewinder', 'division']

function baseWorld(options: MazeOptions): World {
  if (options.base !== undefined) return { ...options.base, walls: new Uint32Array(options.base.n) }
  const rows = options.rows ?? 8
  const cols = options.cols ?? 8
  const t: Topology = { kind: options.topology === 'hex' ? 'hex' : 'square', rows, cols }
  const w = world({ topology: t })
  return { ...w, starts: [0], goals: [w.n - 1] }
}

/** Every side neighbour of a cell, as `[side, cell]`. */
function sides(t: Topology, cell: number): [number, number][] {
  const out: [number, number][] = []
  const k = sideCount(t, cell)
  for (let s = 0; s < k; s++) {
    const nb = step(t, cell, s)
    if (nb >= 0) out.push([s, nb])
  }
  return out
}

function closedWalls(t: Topology, n: number): Uint32Array {
  const walls = new Uint32Array(n)
  for (let c = 0; c < n; c++) for (const [s] of sides(t, c)) walls[c] = (walls[c] ?? 0) | (1 << s)
  return walls
}

/** A copy of the walls with the wall between a and b opened (or closed). */
function toggled(t: Topology, walls: Uint32Array, a: number, b: number, open: boolean): Uint32Array {
  const out = Uint32Array.from(walls)
  const s = sideBetween(t, a, b)
  const back = oppositeSide(t, a, s, b)
  if (open) {
    out[a] = (out[a] ?? 0) & ~(1 << s)
    out[b] = (out[b] ?? 0) & ~(1 << back)
  } else {
    out[a] = (out[a] ?? 0) | (1 << s)
    out[b] = (out[b] ?? 0) | (1 << back)
  }
  return out
}

const draw = (k: Key, n: number): Float64Array => uniform(k, [n]).data as Float64Array
const pick = (u: number, n: number): number => Math.min(n - 1, Math.floor(u * n))

function initial(base: World, algorithm: MazeAlgorithmName, key: Key): MazeState {
  const t = base.topology
  const n = base.n
  const u = draw(key, 2)
  const start = pick(u[0] ?? 0, n)
  const state: MazeState = {
    base,
    walls: algorithm === 'division' ? new Uint32Array(n) : closedWalls(t, n),
    inMaze: new Uint8Array(n),
    current: -1,
    remaining: n,
    carved: 0,
    stack: [],
    frontier: [],
    edges: [],
    parent: new Int32Array(0),
    index: 0,
    sets: new Int32Array(0),
    chambers: [],
    done: false,
  }
  const mark = (cells: readonly number[]): Uint8Array => {
    const m = new Uint8Array(n)
    for (const c of cells) m[c] = 1
    return m
  }
  switch (algorithm) {
    case 'backtracker':
    case 'aldousBroder':
      return { ...state, inMaze: mark([start]), current: start, remaining: n - 1, stack: [start], done: n === 1 }
    case 'wilson':
      return { ...state, inMaze: mark([start]), remaining: n - 1, done: n === 1 }
    case 'prim':
      return {
        ...state,
        inMaze: mark([start]),
        current: start,
        remaining: n - 1,
        frontier: sides(t, start).map(([, b]) => [start, b] as const),
        done: n === 1,
      }
    case 'kruskal': {
      const edges: [number, number][] = []
      for (let a = 0; a < n; a++) for (const [, b] of sides(t, a)) if (b > a) edges.push([a, b])
      const order = permutation(foldIn(key, 1), edges.length).data
      return {
        ...state,
        edges: Array.from(order, (i) => edges[i] ?? [0, 0]),
        parent: Int32Array.from({ length: n }, (_, i) => i),
        done: n === 1,
      }
    }
    case 'division': {
      if (t.kind !== 'square') break
      return {
        ...state,
        inMaze: new Uint8Array(n).fill(1),
        remaining: 0,
        chambers: [[0, 0, t.rows, t.cols]],
      }
    }
    case 'eller': {
      if (t.kind !== 'square') break
      return { ...state, sets: Int32Array.from({ length: t.cols }, (_, i) => i), index: 0 }
    }
    default:
      return state
  }
  return state
}

const find = (parent: Int32Array, x: number): number => {
  let r = x
  while (parent[r] !== r) r = parent[r] ?? r
  return r
}

function stepBacktracker(s: MazeState, u: Float64Array): MazeState {
  const t = s.base.topology
  const cur = s.stack[s.stack.length - 1]
  if (cur === undefined) return { ...s, done: true }
  const fresh = sides(t, cur).filter(([, b]) => s.inMaze[b] === 0)
  if (fresh.length === 0) {
    const stack = s.stack.slice(0, -1)
    return { ...s, stack, current: stack[stack.length - 1] ?? -1, done: stack.length === 0 }
  }
  const [, next] = fresh[pick(u[0] ?? 0, fresh.length)] ?? [0, cur]
  const inMaze = Uint8Array.from(s.inMaze)
  inMaze[next] = 1
  return {
    ...s,
    walls: toggled(t, s.walls, cur, next, true),
    inMaze,
    stack: [...s.stack, next],
    current: next,
    remaining: s.remaining - 1,
    carved: s.carved + 1,
  }
}

function stepPrim(s: MazeState, u: Float64Array): MazeState {
  const t = s.base.topology
  if (s.frontier.length === 0) return { ...s, done: true }
  const i = pick(u[0] ?? 0, s.frontier.length)
  const [a, b] = s.frontier[i] ?? [0, 0]
  const frontier = [...s.frontier.slice(0, i), ...s.frontier.slice(i + 1)]
  if (s.inMaze[b] === 1) return { ...s, frontier, done: frontier.length === 0 }
  const inMaze = Uint8Array.from(s.inMaze)
  inMaze[b] = 1
  for (const [, c] of sides(t, b)) if (inMaze[c] === 0) frontier.push([b, c])
  return {
    ...s,
    walls: toggled(t, s.walls, a, b, true),
    inMaze,
    frontier,
    current: b,
    remaining: s.remaining - 1,
    carved: s.carved + 1,
    done: frontier.length === 0,
  }
}

function stepKruskal(s: MazeState): MazeState {
  const t = s.base.topology
  const [a, b] = s.edges[s.index] ?? [0, 0]
  const index = s.index + 1
  const parent = Int32Array.from(s.parent)
  const ra = find(parent, a)
  const rb = find(parent, b)
  if (ra === rb) return { ...s, parent, index, current: a, done: index >= s.edges.length }
  parent[ra] = rb
  const carved = s.carved + 1
  return {
    ...s,
    walls: toggled(t, s.walls, a, b, true),
    parent,
    index,
    current: b,
    carved,
    remaining: s.base.n - 1 - carved,
    done: index >= s.edges.length || carved === s.base.n - 1,
  }
}

function stepWilson(s: MazeState, u: Float64Array): MazeState {
  const t = s.base.topology
  if (s.stack.length === 0) {
    let start = s.index
    while (start < s.base.n && s.inMaze[start] === 1) start++
    if (start >= s.base.n) return { ...s, done: true }
    return { ...s, stack: [start], current: start, index: start }
  }
  const cur = s.stack[s.stack.length - 1] ?? 0
  const nbs = sides(t, cur)
  const [, next] = nbs[pick(u[0] ?? 0, nbs.length)] ?? [0, cur]
  if (s.inMaze[next] === 1) {
    const path = [...s.stack, next]
    let walls = s.walls
    const inMaze = Uint8Array.from(s.inMaze)
    for (let i = 0; i + 1 < path.length; i++) {
      walls = toggled(t, walls, path[i] ?? 0, path[i + 1] ?? 0, true)
      inMaze[path[i] ?? 0] = 1
    }
    const remaining = s.remaining - s.stack.length
    return {
      ...s,
      walls,
      inMaze,
      stack: [],
      current: next,
      remaining,
      carved: s.carved + s.stack.length,
      done: remaining === 0,
    }
  }
  const at = s.stack.indexOf(next)
  const stack = at >= 0 ? s.stack.slice(0, at + 1) : [...s.stack, next]
  return { ...s, stack, current: next }
}

function stepAldousBroder(s: MazeState, u: Float64Array): MazeState {
  const t = s.base.topology
  const nbs = sides(t, s.current)
  const [, next] = nbs[pick(u[0] ?? 0, nbs.length)] ?? [0, s.current]
  if (s.inMaze[next] === 1) return { ...s, current: next }
  const inMaze = Uint8Array.from(s.inMaze)
  inMaze[next] = 1
  const remaining = s.remaining - 1
  return {
    ...s,
    walls: toggled(t, s.walls, s.current, next, true),
    inMaze,
    current: next,
    remaining,
    carved: s.carved + 1,
    done: remaining === 0,
  }
}

function stepEller(s: MazeState, u: Float64Array, rows: number, cols: number): MazeState {
  const t = s.base.topology
  const r = s.index
  const last = r === rows - 1
  const sets = Int32Array.from(s.sets)
  let walls = s.walls
  let carved = s.carved
  for (let c = 0; c + 1 < cols; c++) {
    const a = sets[c] ?? 0
    const b = sets[c + 1] ?? 0
    if (a === b || !(last || (u[c] ?? 0) < 0.5)) continue
    walls = toggled(t, walls, r * cols + c, r * cols + c + 1, true)
    carved++
    for (let k = 0; k < cols; k++) if (sets[k] === b) sets[k] = a
  }
  const inMaze = Uint8Array.from(s.inMaze)
  for (let c = 0; c < cols; c++) inMaze[r * cols + c] = 1
  if (last) return { ...s, walls, sets, inMaze, carved, index: r + 1, current: -1, remaining: 0, done: true }
  // Each set sends at least one passage down; a set's members are in row order, and the forced one is drawn at random.
  const next = new Int32Array(cols).fill(-1)
  const members = new Map<number, number[]>()
  for (let c = 0; c < cols; c++) {
    const list = members.get(sets[c] ?? 0) ?? []
    list.push(c)
    members.set(sets[c] ?? 0, list)
  }
  for (const [set, cs] of members) {
    const down = cs.filter((c) => (u[cols + c] ?? 0) < 0.5)
    if (down.length === 0) down.push(cs[pick(u[2 * cols + (cs[0] ?? 0)] ?? 0, cs.length)] ?? 0)
    for (const c of down) {
      walls = toggled(t, walls, r * cols + c, (r + 1) * cols + c, true)
      carved++
      next[c] = set
    }
  }
  let fresh = (r + 1) * cols
  for (let c = 0; c < cols; c++) if (next[c] === -1) next[c] = fresh++
  return {
    ...s,
    walls,
    inMaze,
    carved,
    sets: next,
    index: r + 1,
    current: (r + 1) * cols,
    remaining: (rows - r - 1) * cols,
  }
}

function stepBinaryTree(s: MazeState, u: Float64Array, rows: number, cols: number, bias: string): MazeState {
  const t = s.base.topology
  const cell = s.index
  const r = Math.floor(cell / cols)
  const c = cell % cols
  const dr = bias.startsWith('N') ? -1 : 1
  const dc = bias.endsWith('E') ? 1 : -1
  const options: number[] = []
  if (r + dr >= 0 && r + dr < rows) options.push((r + dr) * cols + c)
  if (c + dc >= 0 && c + dc < cols) options.push(r * cols + c + dc)
  const inMaze = Uint8Array.from(s.inMaze)
  inMaze[cell] = 1
  const index = cell + 1
  const base = { ...s, inMaze, index, current: cell, remaining: s.remaining - 1, done: index >= rows * cols }
  if (options.length === 0) return base
  const to = options[pick(u[0] ?? 0, options.length)] ?? cell
  return { ...base, walls: toggled(t, s.walls, cell, to, true), carved: s.carved + 1 }
}

function stepSidewinder(s: MazeState, u: Float64Array, rows: number, cols: number): MazeState {
  const t = s.base.topology
  const cell = s.index
  const r = Math.floor(cell / cols)
  const c = cell % cols
  const runStart = s.sets[0] ?? 0
  const inMaze = Uint8Array.from(s.inMaze)
  inMaze[cell] = 1
  const index = cell + 1
  const done = index >= rows * cols
  const common = { ...s, inMaze, index, current: cell, remaining: s.remaining - 1, done }
  const east = c + 1 < cols
  if (r === 0) {
    // The first row is one corridor; the runs of the rows below each open one passage up into the row above.
    if (!east) return { ...common, sets: Int32Array.of(0) }
    return { ...common, walls: toggled(t, s.walls, cell, cell + 1, true), carved: s.carved + 1 }
  }
  if (east && (u[0] ?? 0) >= 0.5) {
    return { ...common, walls: toggled(t, s.walls, cell, cell + 1, true), carved: s.carved + 1 }
  }
  const from = runStart + pick(u[1] ?? 0, c - runStart + 1)
  const walls = toggled(t, s.walls, r * cols + from, (r - 1) * cols + from, true)
  return { ...common, walls, carved: s.carved + 1, sets: Int32Array.of(east ? c + 1 : 0) }
}

function stepDivision(s: MazeState, u: Float64Array, cols: number): MazeState {
  const t = s.base.topology
  const top = s.chambers[s.chambers.length - 1]
  if (top === undefined) return { ...s, done: true }
  const rest = s.chambers.slice(0, -1)
  const [r0, c0, h, w] = top
  if (h < 2 || w < 2) return { ...s, chambers: rest, done: rest.length === 0 }
  const horizontal = h > w ? true : w > h ? false : (u[0] ?? 0) < 0.5
  let walls = s.walls
  let carved = s.carved
  if (horizontal) {
    const k = 1 + pick(u[1] ?? 0, h - 1)
    const gap = c0 + pick(u[2] ?? 0, w)
    for (let c = c0; c < c0 + w; c++) {
      if (c === gap) continue
      walls = toggled(t, walls, (r0 + k - 1) * cols + c, (r0 + k) * cols + c, false)
      carved++
    }
    const chambers = [...rest, [r0 + k, c0, h - k, w] as const, [r0, c0, k, w] as const]
    return { ...s, walls, carved, chambers, current: (r0 + k) * cols + gap }
  }
  const k = 1 + pick(u[1] ?? 0, w - 1)
  const gap = r0 + pick(u[2] ?? 0, h)
  for (let r = r0; r < r0 + h; r++) {
    if (r === gap) continue
    walls = toggled(t, walls, r * cols + c0 + k - 1, r * cols + c0 + k, false)
    carved++
  }
  const chambers = [...rest, [r0, c0 + k, h, w - k] as const, [r0, c0, h, k] as const]
  return { ...s, walls, carved, chambers, current: gap * cols + c0 + k }
}

/**
 * A maze generator as a traced {@link Algorithm}: each step is one move of the algorithm (a carve or backtrack, a
 * frontier pick, an edge considered, a random-walk step, a row, a cell or a chamber divided), so figures can animate
 * the construction. Every algorithm produces a perfect maze: its open passages form a spanning tree of the cells, so
 * exactly one path joins any two cells. The state's `walls` record the maze so far ({@link mazeOf}).
 *
 * @param options - Size, topology or base world, algorithm and binary-tree bias.
 * @returns The algorithm (its `init` key chooses the start cell and Kruskal's edge order; step keys the rest).
 * @throws WhiteboxError with code `value` for a square-only algorithm on another topology, or a bad size.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { mazeAlgorithm, mazeOf, bridges } from 'whitebox-engine/worlds'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const run = runAlgorithm(mazeAlgorithm({ rows: 3, cols: 3, algorithm: 'wilson' }), { seed: 1, maxSteps: 10000 })
 * [run.status, bridges(mazeOf(run.state)).length] // => ['converged', 8]
 * ```
 */
export function mazeAlgorithm(options: MazeOptions = {}): Algorithm<MazeState> {
  const algorithm = options.algorithm ?? 'backtracker'
  const base = baseWorld(options)
  const t = base.topology
  if (SQUARE_ONLY.includes(algorithm) && t.kind !== 'square') {
    throw new WhiteboxError('value', `mazeAlgorithm: '${algorithm}' needs a square grid`)
  }
  const rows = t.kind === 'graph' ? 1 : t.rows
  const cols = t.kind === 'graph' ? base.n : t.cols
  const bias = options.bias ?? 'NE'
  return {
    name: `maze/${algorithm}`,
    config: { algorithm, n: base.n, topology: t.kind, ...(algorithm === 'binaryTree' ? { bias } : {}) } as Json,
    schema: channels({
      metrics: {
        carved: { description: 'passages opened (walls added, for recursive division)' },
        remaining: { description: 'cells not yet in the maze', direction: 'lower' },
      },
      tensors: { walls: { role: 'state', description: 'closed sides of each cell (bit masks)', shape: ['n'] } },
      frames: true,
    }),
    init: (key) => initial(base, algorithm, key),
    step: (s, ctx) => {
      if (s.done) return s
      const u = draw(ctx.key, algorithm === 'eller' ? 3 * cols : 3)
      switch (algorithm) {
        case 'backtracker':
          return stepBacktracker(s, u)
        case 'prim':
          return stepPrim(s, u)
        case 'kruskal':
          return stepKruskal(s)
        case 'wilson':
          return stepWilson(s, u)
        case 'aldousBroder':
          return stepAldousBroder(s, u)
        case 'eller':
          return stepEller(s, u, rows, cols)
        case 'binaryTree':
          return stepBinaryTree(s, u, rows, cols, bias)
        case 'sidewinder':
          return stepSidewinder(s, u, rows, cols)
        default:
          return stepDivision(s, u, cols)
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { carved: s.carved, remaining: s.remaining },
      tensors: { walls: { role: 'state', value: s.walls } },
      frame: { current: s.current, stack: [...s.stack] },
    }),
  }
}

/**
 * The world a maze generator's state describes (its base world with the walls carved so far).
 *
 * @param state - A maze state.
 * @returns The world.
 * @example
 * ```ts
 * import { mazeAlgorithm, mazeOf } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * mazeOf(mazeAlgorithm({ rows: 2, cols: 2 }).init(key(0))).walls[0] // => 6
 * ```
 */
export function mazeOf(state: MazeState): World {
  return { ...state.base, walls: state.walls }
}

/**
 * Generate a perfect maze from a key (the same key gives the same maze): runs {@link mazeAlgorithm} to completion,
 * with a step cap generous enough for the random walks of Wilson's and the Aldous-Broder algorithms.
 *
 * @param key - The key.
 * @param options - Size, topology or base world, algorithm and bias, and run options (tracer, policy).
 * @returns The maze, starting in the first cell with its goal in the last (or the base world's starts and goals).
 * @throws WhiteboxError with code `convergence` if the cap is reached (practically never).
 * @concept procedural-generation
 * @example
 * ```ts
 * import { generateMaze, toAscii } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * toAscii(generateMaze(key(0), { rows: 2, cols: 3, algorithm: 'binaryTree' }), { format: 'walls' }).length // => 5
 * ```
 */
export function generateMaze(
  key: Key,
  options: MazeOptions & Partial<Pick<RunOptions, 'tracer' | 'policy' | 'parent' | 'runId'>> = {},
): World {
  const { tracer, policy, parent, runId, ...maze } = options
  const algorithm = mazeAlgorithm(maze)
  const n = baseWorld(maze).n
  const cap = Math.ceil(64 * n * (1 + Math.log(n)) ** 2) + 1000
  const run = runAlgorithm(algorithm, {
    key,
    maxSteps: cap,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  })
  if (!run.state.done) throw new WhiteboxError('convergence', `generateMaze: not finished after ${cap} steps`)
  return mazeOf(run.state)
}

/**
 * Braid a maze (Buck's braiding): visit the dead ends in random order and, with probability `p` each, open a wall to
 * a closed neighbour (preferring neighbours that are dead ends too), turning the tree into a maze with loops. A
 * predicate can forbid openings (key-door tasks keep their doors on the only path).
 *
 * @param w - The maze.
 * @param key - The key.
 * @param p - The probability of removing each dead end (default 1).
 * @param options - Which walls may be opened.
 * @returns The braided maze.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { braid, deadEnds, generateMaze } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * deadEnds(braid(generateMaze(key(0), { rows: 5, cols: 5 }), key(1))).length // => 0
 * ```
 */
export function braid(
  w: World,
  key: Key,
  p = 1,
  options: { readonly allowed?: (a: number, b: number) => boolean } = {},
): World {
  const t = w.topology
  let walls: Uint32Array = Uint32Array.from(w.walls)
  const openSides = (c: number): number => sides(t, c).filter(([s]) => !(((walls[c] ?? 0) >>> s) & 1)).length
  const order = permutation(foldIn(key, 0), w.n).data
  const u = draw(foldIn(key, 1), 2 * w.n)
  for (const [i, c] of Array.from(order).entries()) {
    if (openSides(c) !== 1 || (u[2 * i] ?? 1) >= p) continue
    const closed = sides(t, c).filter(([s, b]) => ((walls[c] ?? 0) >>> s) & 1 && (options.allowed?.(c, b) ?? true))
    if (closed.length === 0) continue
    const ends = closed.filter(([, b]) => openSides(b) === 1)
    const pool = ends.length > 0 ? ends : closed
    const [, b] = pool[pick(u[2 * i + 1] ?? 0, pool.length)] ?? [0, c]
    walls = toggled(t, walls, c, b, true)
  }
  return { ...w, walls }
}

/**
 * Whether a world's open passages form a perfect maze: a spanning tree of its cells (connected, with exactly
 * `n - 1` open adjacencies).
 *
 * @param w - The world.
 * @returns Whether it is perfect.
 * @example
 * ```ts
 * import { generateMaze, isPerfectMaze } from 'whitebox-engine/worlds'
 * import { key } from 'whitebox-engine/random'
 * isPerfectMaze(generateMaze(key(4), { rows: 4, cols: 4, algorithm: 'division' })) // => true
 * ```
 */
export function isPerfectMaze(w: World): boolean {
  const t = w.topology
  let open = 0
  const seen = new Uint8Array(w.n)
  const queue = [0]
  seen[0] = 1
  for (let q = 0; q < queue.length; q++) {
    const c = queue[q] ?? 0
    for (const [s, b] of sides(t, c)) {
      if (hasWall(w, c, s)) continue
      if (b > c) open++
      if (seen[b] === 0) {
        seen[b] = 1
        queue.push(b)
      }
    }
  }
  return queue.length === w.n && open === w.n - 1
}
