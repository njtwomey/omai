import { type Json, WhiteboxError } from '../base/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import {
  baseWorld,
  draw,
  initial,
  pick,
  SQUARE_ONLY,
  sides,
  stepAldousBroder,
  stepBacktracker,
  stepBinaryTree,
  stepDivision,
  stepEller,
  stepKruskal,
  stepPrim,
  stepSidewinder,
  stepWilson,
  toggled,
} from './_carving.ts'
import { hasWall, type World } from './world.ts'

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
