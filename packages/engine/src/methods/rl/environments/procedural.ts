import { WhiteboxError } from '../../../lib/base/index.ts'
import { foldIn, Generator, type Key, key as makeKey } from '../../../lib/random/index.ts'
import type { Env, EnvDefinition, EnvModel, Outcome } from '../../../lib/rl/index.ts'
import { circle, type Paint, rect, type Scene } from '../../../lib/scene/index.ts'
import { box, choice, dict, discrete, int, real, type Space } from '../../../lib/space/index.ts'
import { renderGrid } from './gridworlds.ts'

// ------------------------------------------------------------------------------------------------- Maze generation

/** How a maze is carved: randomised depth-first search, randomised Prim's algorithm or randomised Kruskal's. */
export type MazeAlgorithm = 'backtracker' | 'prim' | 'kruskal'

/**
 * A maze of `rows x cols` cells drawn as a `(2 rows + 1) x (2 cols + 1)` character grid: cell `(r, c)` sits at
 * `(2r + 1, 2c + 1)`, the wall between two neighbouring cells at their midpoint; `#` is wall and `.` open.
 */
export interface Maze {
  readonly rows: number
  readonly cols: number
  readonly grid: readonly string[]
}

/** Options of {@link generateMaze}. */
export interface MazeOptions {
  /** Cells per column (default 5). */
  readonly rows?: number
  /** Cells per row (default 5). */
  readonly cols?: number
  /** The carving algorithm (default `'backtracker'`). */
  readonly algorithm?: MazeAlgorithm
  /** The probability of opening a loop at each dead end (default 0, a perfect maze). */
  readonly braid?: number
}

const NEIGHBOURS: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
]

function cellNeighbours(rows: number, cols: number, i: number): number[] {
  const r = Math.floor(i / cols)
  const c = i % cols
  const out: number[] = []
  for (const [dr, dc] of NEIGHBOURS) {
    const r2 = r + dr
    const c2 = c + dc
    if (r2 >= 0 && r2 < rows && c2 >= 0 && c2 < cols) out.push(r2 * cols + c2)
  }
  return out
}

/** A mutable character grid with carving between cells. */
class Carver {
  readonly chars: string[][]
  readonly rows: number
  readonly cols: number
  constructor(rows: number, cols: number, grid?: readonly string[]) {
    this.rows = rows
    this.cols = cols
    this.chars =
      grid?.map((row) => row.split('')) ??
      Array.from({ length: 2 * rows + 1 }, (_, y) =>
        Array.from({ length: 2 * cols + 1 }, (_, x) => (y % 2 === 1 && x % 2 === 1 ? '.' : '#')),
      )
  }
  wall(a: number, b: number): [number, number] {
    const ra = Math.floor(a / this.cols)
    const ca = a % this.cols
    const rb = Math.floor(b / this.cols)
    const cb = b % this.cols
    return [ra + rb + 1, ca + cb + 1]
  }
  isOpen(a: number, b: number): boolean {
    const [y, x] = this.wall(a, b)
    return this.chars[y]?.[x] !== '#'
  }
  carve(a: number, b: number): void {
    const [y, x] = this.wall(a, b)
    const row = this.chars[y]
    if (row !== undefined) row[x] = '.'
  }
  degree(a: number): number {
    return cellNeighbours(this.rows, this.cols, a).filter((b) => this.isOpen(a, b)).length
  }
  maze(): Maze {
    return { rows: this.rows, cols: this.cols, grid: this.chars.map((row) => row.join('')) }
  }
}

function backtracker(m: Carver, pick: (n: number) => number): void {
  const n = m.rows * m.cols
  const visited = new Uint8Array(n)
  const start = pick(n)
  visited[start] = 1
  const stack = [start]
  while (stack.length > 0) {
    const cur = stack[stack.length - 1] ?? 0
    const fresh = cellNeighbours(m.rows, m.cols, cur).filter((b) => visited[b] === 0)
    if (fresh.length === 0) {
      stack.pop()
      continue
    }
    const next = fresh[pick(fresh.length)] ?? 0
    m.carve(cur, next)
    visited[next] = 1
    stack.push(next)
  }
}

function prim(m: Carver, pick: (n: number) => number): void {
  const n = m.rows * m.cols
  const inMaze = new Uint8Array(n)
  const start = pick(n)
  inMaze[start] = 1
  const frontier: [number, number][] = cellNeighbours(m.rows, m.cols, start).map((b) => [start, b])
  while (frontier.length > 0) {
    const i = pick(frontier.length)
    const [a, b] = frontier[i] ?? [0, 0]
    frontier.splice(i, 1)
    if (inMaze[b] === 1) continue
    m.carve(a, b)
    inMaze[b] = 1
    for (const c of cellNeighbours(m.rows, m.cols, b)) if (inMaze[c] === 0) frontier.push([b, c])
  }
}

function kruskal(m: Carver, rng: Generator): void {
  const n = m.rows * m.cols
  const edges: [number, number][] = []
  for (let a = 0; a < n; a++) for (const b of cellNeighbours(m.rows, m.cols, a)) if (b > a) edges.push([a, b])
  const parent = Int32Array.from({ length: n }, (_, i) => i)
  const find = (x: number): number => {
    let r = x
    while (parent[r] !== r) r = parent[r] ?? r
    let y = x
    while (parent[y] !== r) {
      const next = parent[y] ?? r
      parent[y] = r
      y = next
    }
    return r
  }
  for (const i of rng.permutation(edges.length).data) {
    const [a, b] = edges[i] ?? [0, 0]
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) {
      parent[ra] = rb
      m.carve(a, b)
    }
  }
}

/**
 * Remove dead ends with probability `p` each (Buck's braiding): a dead end, visited in random order, opens a wall to
 * a random closed neighbour (preferring neighbours that are dead ends too), which creates loops. A predicate can veto
 * openings (key-door tasks forbid loops around their doors).
 *
 * @param maze - The maze.
 * @param key - The key.
 * @param p - The probability of opening each dead end.
 * @param allowed - Whether the wall between two cells (row-major cell indices) may be opened (default always).
 * @returns The braided maze.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { braidMaze, deadEnds, generateMaze } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * deadEnds(braidMaze(generateMaze(key(0)), key(1), 1)).length // => 0
 * ```
 */
export function braidMaze(
  maze: Maze,
  key: Key,
  p: number,
  allowed: (a: number, b: number) => boolean = () => true,
): Maze {
  const m = new Carver(maze.rows, maze.cols, maze.grid)
  const rng = new Generator(key)
  const n = maze.rows * maze.cols
  for (const a of rng.permutation(n).data) {
    if (m.degree(a) !== 1 || (rng.uniform([]).data[0] ?? 1) >= p) continue
    const closed = cellNeighbours(m.rows, m.cols, a).filter((b) => !m.isOpen(a, b) && allowed(a, b))
    if (closed.length === 0) continue
    const ends = closed.filter((b) => m.degree(b) === 1)
    const pool = ends.length > 0 ? ends : closed
    m.carve(a, pool[Math.floor((rng.uniform([]).data[0] ?? 0) * pool.length)] ?? a)
  }
  return m.maze()
}

/**
 * The dead ends of a maze: cells with exactly one open side.
 *
 * @param maze - The maze.
 * @returns Their row-major cell indices.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { deadEnds } from 'whitebox-engine/methods/rl'
 * deadEnds({ rows: 1, cols: 2, grid: ['#####', '#...#', '#####'] }) // => [0, 1]
 * ```
 */
export function deadEnds(maze: Maze): number[] {
  const m = new Carver(maze.rows, maze.cols, maze.grid)
  return Array.from({ length: maze.rows * maze.cols }, (_, i) => i).filter((i) => m.degree(i) === 1)
}

/**
 * Generate a maze from a key: a perfect maze (a spanning tree of the cell grid, so exactly one path joins any two
 * cells) by the recursive backtracker (randomised depth-first search, long winding corridors), randomised Prim's
 * algorithm (many short dead ends) or randomised Kruskal's algorithm (uniformly scattered junctions), optionally braided
 * into a maze with loops.
 *
 * @param key - The key; the same key gives the same maze.
 * @param options - Size in cells, algorithm and braiding.
 * @returns The maze.
 * @throws WhiteboxError with code `value` for sizes below one cell.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { generateMaze } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * generateMaze(key(0), { rows: 3, cols: 4 }).grid.length // => 7
 * ```
 */
export function generateMaze(key: Key, options: MazeOptions = {}): Maze {
  const rows = options.rows ?? 5
  const cols = options.cols ?? 5
  if (!(Number.isInteger(rows) && Number.isInteger(cols) && rows >= 1 && cols >= 1)) {
    throw new WhiteboxError('value', 'generateMaze: rows and cols must be positive integers')
  }
  const m = new Carver(rows, cols)
  const rng = new Generator(foldIn(key, 0))
  const pick = (n: number): number => Math.floor((rng.uniform([]).data[0] ?? 0) * n)
  const algorithm = options.algorithm ?? 'backtracker'
  if (algorithm === 'prim') prim(m, pick)
  else if (algorithm === 'kruskal') kruskal(m, rng)
  else backtracker(m, pick)
  const maze = m.maze()
  return (options.braid ?? 0) > 0 ? braidMaze(maze, foldIn(key, 1), options.braid ?? 0) : maze
}

/**
 * Breadth-first distances on a character grid from one position, moving up, right, down and left through passable
 * characters (default everything but `#`).
 *
 * @param grid - The rows.
 * @param from - The start, as `[row, column]`.
 * @param passable - Whether a character can be entered.
 * @returns `distance[row * width + column]`, -1 where unreachable.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { gridDistances } from 'whitebox-engine/methods/rl'
 * Array.from(gridDistances(['...', '#.#'], [0, 0])) // => [0, 1, 2, -1, 2, -1]
 * ```
 */
export function gridDistances(
  grid: readonly string[],
  from: readonly [number, number],
  passable: (ch: string) => boolean = (ch) => ch !== '#',
): Int32Array {
  const h = grid.length
  const w = grid[0]?.length ?? 0
  const dist = new Int32Array(h * w).fill(-1)
  const start = from[0] * w + from[1]
  dist[start] = 0
  const queue = [start]
  for (let q = 0; q < queue.length; q++) {
    const cur = queue[q] ?? 0
    const r = Math.floor(cur / w)
    const c = cur % w
    for (const [dr, dc] of NEIGHBOURS) {
      const r2 = r + dr
      const c2 = c + dc
      if (r2 < 0 || r2 >= h || c2 < 0 || c2 >= w) continue
      const j = r2 * w + c2
      if (dist[j] !== -1 || !passable(grid[r2]?.[c2] ?? '#')) continue
      dist[j] = (dist[cur] ?? 0) + 1
      queue.push(j)
    }
  }
  return dist
}

// --------------------------------------------------------------------------------------------------------- Levels

/** Options of procedural levels and of the {@link proceduralMaze} and {@link keyDoor} environments. */
export interface ProceduralOptions extends MazeOptions {
  /** Key-door pairs (default 0; key `i` opens door `i`, and the doors lie in order on the path to the goal). */
  readonly keys?: number
  /** The seed every level is derived from (default 0). */
  readonly levelSeed?: number
  /** The first level of the allowed range (default 0). */
  readonly startLevel?: number
  /** How many levels to draw from at reset (default 0: any level, as Procgen's `num_levels = 0`). */
  readonly numLevels?: number
  /** Where the goal goes: the cell farthest from the start (default) or a random cell. */
  readonly goal?: 'farthest' | 'random'
  /** What the agent observes: the state index, the full grid, or a local view (default `'grid'`). */
  readonly observation?: 'index' | 'grid' | 'view'
  /** The radius of the local view (default 2, a 5 x 5 window). */
  readonly viewRadius?: number
  readonly goalReward?: number
  readonly stepReward?: number
  /** The time limit `make` applies (default `8 rows cols (keys + 1)`). */
  readonly maxEpisodeSteps?: number
}

/**
 * One procedural level: its maze and the positions (`[row, column]` on the character grid) of the start, the goal and
 * every key and door; `map` is the grid with `S`, `G`, keys `a`, `b`, ... and doors `A`, `B`, ...
 */
export interface ProceduralLevel {
  readonly level: number
  readonly maze: Maze
  readonly start: readonly [number, number]
  readonly goal: readonly [number, number]
  readonly keys: readonly (readonly [number, number])[]
  readonly doors: readonly (readonly [number, number])[]
  readonly map: readonly string[]
}

const KEY_COLOURS: readonly Paint[] = ['#cccc00', 'tab3', 'tab0', 'tab4']
const MAX_KEYS = 4

const cellPos = (maze: Maze, i: number): [number, number] => [
  2 * Math.floor(i / maze.cols) + 1,
  2 * (i % maze.cols) + 1,
]

/**
 * Build level `level` of a procedural family: the maze from `foldIn(key(levelSeed), level)`, a random start cell, the
 * goal at the farthest cell (or a random one), and for key-door tasks the doors on the path from start to goal, each
 * key placed where it can be reached through the earlier doors but not past its own (so every key is needed). Braiding
 * never opens a loop that would bypass a door.
 *
 * @param level - The level number.
 * @param options - Maze, keys, seed and goal placement.
 * @returns The level.
 * @throws WhiteboxError with code `value` when the maze is too small for its doors.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { proceduralLevel } from 'whitebox-engine/methods/rl'
 * proceduralLevel(3, { rows: 4, cols: 4, keys: 1 }).doors.length // => 1
 * ```
 */
export function proceduralLevel(level: number, options: ProceduralOptions = {}): ProceduralLevel {
  const nKeys = options.keys ?? 0
  if (!(Number.isInteger(nKeys) && nKeys >= 0 && nKeys <= MAX_KEYS)) {
    throw new WhiteboxError('value', `proceduralLevel: keys must be an integer from 0 to ${MAX_KEYS}`)
  }
  const lk = foldIn(makeKey(options.levelSeed ?? 0), level)
  const perfect = generateMaze(foldIn(lk, 0), { ...options, braid: 0 })
  const rng = new Generator(foldIn(lk, 1))
  const pick = (n: number): number => Math.floor((rng.uniform([]).data[0] ?? 0) * n)
  const nCells = perfect.rows * perfect.cols
  const startCell = pick(nCells)
  const start = cellPos(perfect, startCell)
  const fromStart = gridDistances(perfect.grid, start)
  const width = perfect.grid[0]?.length ?? 0
  let goalCell: number
  if (options.goal === 'random' && nCells > 1) {
    goalCell = pick(nCells - 1)
    if (goalCell >= startCell) goalCell += 1
  } else {
    goalCell = startCell
    for (let i = 0; i < nCells; i++) {
      const [r, c] = cellPos(perfect, i)
      const [gr, gc] = cellPos(perfect, goalCell)
      if ((fromStart[r * width + c] ?? 0) > (fromStart[gr * width + gc] ?? 0)) goalCell = i
    }
  }
  const goal = cellPos(perfect, goalCell)
  // The path from start to goal, by walking down the distances from the goal.
  const path: [number, number][] = [goal]
  for (let cur = goal; cur[0] !== start[0] || cur[1] !== start[1]; ) {
    const d = fromStart[cur[0] * width + cur[1]] ?? 0
    const prev = NEIGHBOURS.map(([dr, dc]) => [cur[0] + dr, cur[1] + dc] as [number, number]).find(
      ([r, c]) => fromStart[r * width + c] === d - 1,
    )
    if (prev === undefined) break
    path.push(prev)
    cur = prev
  }
  path.reverse()
  const interior = path.slice(1, -1)
  if (interior.length < 2 * nKeys) {
    throw new WhiteboxError('value', `proceduralLevel: the path to the goal is too short for ${nKeys} doors`)
  }
  // Doors at increasing positions along the path, leaving room before each for its key's region to grow.
  const doorIdx: number[] = []
  for (let i = 0; i < nKeys; i++) {
    const lo = i === 0 ? 1 : (doorIdx[i - 1] ?? 0) + 2
    const hi = interior.length - 2 * (nKeys - i - 1)
    doorIdx.push(lo + pick(Math.max(hi - lo, 1)))
  }
  const doors = doorIdx.map((i) => interior[Math.min(i, interior.length - 1)] ?? goal)
  const doorAt = new Map(doors.map(([r, c], i) => [r * width + c, i]))
  // side[j] is the first door that cannot be passed to reach j (nKeys beyond the last door). Braiding may not open a
  // wall between cells on different sides, so loops never bypass a door.
  const side = new Int32Array(width * perfect.grid.length).fill(nKeys)
  for (let i = nKeys - 1; i >= 0; i--) {
    const reach = reachable(perfect.grid, start, doorAt, i)
    for (const [j, d] of reach.entries()) if (d >= 0) side[j] = i
  }
  const maze =
    (options.braid ?? 0) > 0
      ? braidMaze(perfect, foldIn(lk, 2), options.braid ?? 0, (a, b) => {
          const [ra, ca] = cellPos(perfect, a)
          const [rb, cb] = cellPos(perfect, b)
          return side[ra * width + ca] === side[rb * width + cb]
        })
      : perfect
  const used = new Set([start[0] * width + start[1], goal[0] * width + goal[1], ...doorAt.keys()])
  const keys: [number, number][] = []
  for (let i = 0; i < nKeys; i++) {
    // Prefer the cells just unlocked by the previous door; fall back to anything on this side of door i.
    const candidates = (exact: boolean): number[] => {
      const out: number[] = []
      for (let j = 0; j < side.length; j++) {
        const ch = maze.grid[Math.floor(j / width)]?.[j % width] ?? '#'
        if (ch === '#' || used.has(j)) continue
        if (exact ? side[j] === i : (side[j] ?? nKeys) <= i) out.push(j)
      }
      return out
    }
    const pool = candidates(true).length > 0 ? candidates(true) : candidates(false)
    const j = pool[pick(pool.length)] ?? 0
    used.add(j)
    keys.push([Math.floor(j / width), j % width])
  }
  const map = maze.grid.map((row, r) =>
    row
      .split('')
      .map((ch, c) => {
        if (r === start[0] && c === start[1]) return 'S'
        if (r === goal[0] && c === goal[1]) return 'G'
        const k = keys.findIndex(([kr, kc]) => kr === r && kc === c)
        if (k >= 0) return String.fromCharCode(97 + k)
        const d = doors.findIndex(([dr, dc]) => dr === r && dc === c)
        return d >= 0 ? String.fromCharCode(65 + d) : ch
      })
      .join(''),
  )
  return { level, maze, start, goal, keys, doors, map }
}

/** Distances from the start with doors `>= closedFrom` closed and the earlier ones open. */
function reachable(
  grid: readonly string[],
  start: readonly [number, number],
  doorAt: ReadonlyMap<number, number>,
  closedFrom: number,
): Int32Array {
  const width = grid[0]?.length ?? 0
  const blocked = grid.map((row, r) =>
    row
      .split('')
      .map((ch, c) => ((doorAt.get(r * width + c) ?? -1) >= closedFrom ? '#' : ch))
      .join(''),
  )
  return gridDistances(blocked, start)
}

// ---------------------------------------------------------------------------------------------------- Environments

/** The state of a procedural environment: the level, the position (row-major on the character grid) and held keys. */
export interface ProceduralState {
  readonly level: number
  readonly pos: number
  /** Held keys as a bit mask (bit `i` for key `i`). */
  readonly keys: number
}

/** What a procedural environment observes: a state index, or a flat grid or local view with held-key flags. */
export type ProceduralObservation = number | readonly number[]

/** A procedural environment: an {@link Env} that also builds any of its levels and reports its level range. */
export interface ProceduralEnv extends Env<ProceduralState, ProceduralObservation, number> {
  /** Build (and cache) a level. */
  level(n: number): ProceduralLevel
  /** The levels reset draws from (`numLevels = 0` means any level). */
  readonly levels: { readonly startLevel: number; readonly numLevels: number }
  /** The channel names of the grid or view observations (empty for index observations). */
  readonly channelNames: readonly string[]
}

/** A procedural environment definition (what {@link proceduralMaze} and {@link keyDoor} are). */
export interface ProceduralDefinition extends EnvDefinition<ProceduralState, ProceduralObservation, number> {
  create(options?: Readonly<Record<string, unknown>>): ProceduralEnv
}

const MAX_MODEL_LEVELS = 256
const ALL_LEVELS = 2 ** 31

const isKeyChar = (ch: string): boolean => ch >= 'a' && ch <= 'd'
const isDoorChar = (ch: string): boolean => ch >= 'A' && ch <= 'D'

/**
 * Build a procedural maze or key-door environment (Procgen-style level control): each reset draws a level uniformly
 * from `startLevel, ..., startLevel + numLevels - 1` (any level when `numLevels = 0`), so a fixed set of training levels
 * and a disjoint set of test levels come from the same definition. Moves are up, right, down, left; walls and locked
 * doors block; entering a key picks it up; the goal pays `goalReward` and ends the episode, other steps pay
 * `stepReward`. With a finite level range (up to 256 levels) the environment has an explicit model, so value iteration
 * solves it level by level.
 *
 * @param options - Maze, keys, levels, observation and rewards.
 * @param id - The environment id.
 * @returns The environment.
 * @throws WhiteboxError with code `value` for index observations over unlimited levels.
 * @concept procedural-generation
 * @example
 * ```ts
 * import { makeProcedural } from 'whitebox-engine/methods/rl'
 * makeProcedural({ rows: 3, cols: 3, numLevels: 2, observation: 'index' }).observationSpace.kind // => 'discrete'
 * ```
 */
export function makeProcedural(options: ProceduralOptions = {}, id = 'ProceduralMaze-v0'): ProceduralEnv {
  const rows = options.rows ?? 5
  const cols = options.cols ?? 5
  const nKeys = options.keys ?? 0
  const h = 2 * rows + 1
  const w = 2 * cols + 1
  const masks = 2 ** nKeys
  const startLevel = options.startLevel ?? 0
  const numLevels = options.numLevels ?? 0
  const mode = options.observation ?? 'grid'
  const radius = options.viewRadius ?? 2
  const goalReward = options.goalReward ?? 1
  const stepReward = options.stepReward ?? 0
  if (mode === 'index' && numLevels === 0) {
    throw new WhiteboxError('value', 'procedural: index observations need a finite numLevels')
  }
  const cache = new Map<number, ProceduralLevel>()
  const level = (n: number): ProceduralLevel => {
    const hit = cache.get(n)
    if (hit !== undefined) return hit
    if (cache.size >= 4096) cache.clear()
    const made = proceduralLevel(n, options)
    cache.set(n, made)
    return made
  }
  const charAt = (s: ProceduralState, pos: number): string => level(s.level).map[Math.floor(pos / w)]?.[pos % w] ?? '#'
  const transition = (s: ProceduralState, a: number): Outcome<ProceduralState> => {
    if (charAt(s, s.pos) === 'G') return { probability: 1, state: s, reward: 0, terminated: true }
    const [dr, dc] = NEIGHBOURS[a] ?? [0, 0]
    const r = Math.floor(s.pos / w) + dr
    const c = (s.pos % w) + dc
    const next = r * w + c
    const ch = r < 0 || r >= h || c < 0 || c >= w ? '#' : charAt(s, next)
    const locked = isDoorChar(ch) && (s.keys & (1 << (ch.charCodeAt(0) - 65))) === 0
    if (ch === '#' || locked) return { probability: 1, state: s, reward: stepReward, terminated: false }
    const keys = isKeyChar(ch) ? s.keys | (1 << (ch.charCodeAt(0) - 97)) : s.keys
    const state = { level: s.level, pos: next, keys }
    return ch === 'G'
      ? { probability: 1, state, reward: goalReward, terminated: true }
      : { probability: 1, state, reward: stepReward, terminated: false }
  }
  const keyNames = Array.from({ length: nKeys }, (_, i) => `key_${String.fromCharCode(97 + i)}`)
  const doorNames = Array.from({ length: nKeys }, (_, i) => `door_${String.fromCharCode(65 + i)}`)
  const view = mode === 'view'
  const channelNames = view
    ? ['wall', 'goal', ...keyNames, ...doorNames]
    : ['wall', 'agent', 'goal', ...keyNames, ...doorNames]
  const nc = channelNames.length
  const stateIndex = (s: ProceduralState): number => ((s.level - startLevel) * h * w + s.pos) * masks + s.keys
  // The channels of one cell (the agent channel is filled in by the caller); held keys no longer show.
  const features = (ch: string, held: number): number[] => {
    const f = view ? [ch === '#' ? 1 : 0, ch === 'G' ? 1 : 0] : [ch === '#' ? 1 : 0, 0, ch === 'G' ? 1 : 0]
    for (let i = 0; i < nKeys; i++) f.push(ch === String.fromCharCode(97 + i) && (held & (1 << i)) === 0 ? 1 : 0)
    for (let i = 0; i < nKeys; i++) f.push(ch === String.fromCharCode(65 + i) ? 1 : 0)
    return f
  }
  const observe = (s: ProceduralState): ProceduralObservation => {
    if (mode === 'index') return stateIndex(s)
    const out: number[] = []
    if (!view) {
      for (let p = 0; p < h * w; p++) {
        const f = features(charAt(s, p), s.keys)
        if (p === s.pos) f[1] = 1
        out.push(...f)
      }
    } else {
      const r0 = Math.floor(s.pos / w)
      const c0 = s.pos % w
      for (let dr = -radius; dr <= radius; dr++) {
        for (let dc = -radius; dc <= radius; dc++) {
          const r = r0 + dr
          const c = c0 + dc
          out.push(...features(r < 0 || r >= h || c < 0 || c >= w ? '#' : charAt(s, r * w + c), s.keys))
        }
      }
    }
    for (let i = 0; i < nKeys; i++) out.push((s.keys >> i) & 1)
    return out
  }
  const side = 2 * radius + 1
  const observationSpace: Space =
    mode === 'index'
      ? discrete(numLevels * h * w * masks, { label: '((level - startLevel) * cells + pos) * 2^keys + held keys' })
      : box(0, 1, [(view ? side * side : h * w) * nc + nKeys], {
          label: `${channelNames.join(', ')} per cell, then held keys`,
        })
  const startOf = (n: number): ProceduralState => {
    const L = level(n)
    return { level: n, pos: L.start[0] * w + L.start[1], keys: 0 }
  }
  let states: ProceduralState[] | undefined
  const model: EnvModel<ProceduralState, number> | undefined =
    numLevels > 0 && numLevels <= MAX_MODEL_LEVELS
      ? {
          get states(): readonly ProceduralState[] {
            states ??= Array.from({ length: numLevels * h * w * masks }, (_, i) => ({
              level: startLevel + Math.floor(i / (h * w * masks)),
              pos: Math.floor(i / masks) % (h * w),
              keys: i % masks,
            }))
            return states
          },
          actions: [0, 1, 2, 3],
          stateIndex,
          actionIndex: (a) => a,
          initial: Array.from({ length: numLevels }, (_, i) => ({
            probability: 1 / numLevels,
            state: startOf(startLevel + i),
          })),
          transitions: (s, a) => [transition(s, a)],
        }
      : undefined
  const render = (s: ProceduralState): Scene => {
    const L = level(s.level)
    const cell = Math.max(16, Math.floor(480 / Math.max(h, w)))
    const drawn = L.map.map((row) =>
      row
        .split('')
        .map((ch) => (isKeyChar(ch) ? '.' : isDoorChar(ch) ? 'D' : ch))
        .join(''),
    )
    const overlay = [
      ...L.keys.flatMap(([r, c], i) =>
        (s.keys & (1 << i)) !== 0
          ? []
          : [
              circle(c * cell + cell / 2, r * cell + cell / 2, cell / 4, {
                fill: KEY_COLOURS[i] ?? 'fg',
                title: `key ${String.fromCharCode(97 + i)}`,
              }),
            ],
      ),
      ...L.doors.map(([r, c], i) =>
        rect(c * cell + 2, r * cell + 2, cell - 4, cell - 4, {
          fill: 'none',
          stroke: KEY_COLOURS[i] ?? 'fg',
          strokeWidth: 3,
          title: `door ${String.fromCharCode(65 + i)}${(s.keys & (1 << i)) !== 0 ? ' (unlocked)' : ''}`,
        }),
      ),
    ]
    return renderGrid(drawn, [Math.floor(s.pos / w), s.pos % w], { cell, title: `${id} level ${s.level}`, overlay })
  }
  return {
    spec: {
      id,
      description: nKeys > 0 ? 'Fetch the keys, open the doors and reach the goal.' : 'Find the goal of a maze.',
      maxEpisodeSteps: options.maxEpisodeSteps ?? 8 * rows * cols * (nKeys + 1),
      deterministic: true,
    },
    observationSpace,
    actionSpace: discrete(4, { label: 'up (0), right (1), down (2), left (3)' }),
    channelNames: mode === 'index' ? [] : channelNames,
    levels: { startLevel, numLevels },
    level,
    reset: (k) => {
      const u = new Generator(k).uniform([]).data[0] ?? 0
      const n = numLevels > 0 ? startLevel + Math.floor(u * numLevels) : Math.floor(u * ALL_LEVELS)
      const s = startOf(n)
      return { state: s, observation: observe(s), info: { level: n } }
    },
    step: (s, a) => {
      const o = transition(s, a)
      return {
        state: o.state,
        observation: observe(o.state),
        reward: o.reward,
        terminated: o.terminated,
        truncated: false,
        info: { level: s.level, keys: o.state.keys },
      }
    },
    render,
    observe,
    ...(model === undefined ? {} : { model }),
  }
}

/**
 * Disjoint training and test level ranges (Procgen's protocol): `train` levels from `startLevel`, then `test` levels
 * right after them.
 *
 * @param train - Number of training levels.
 * @param test - Number of test levels.
 * @param startLevel - The first training level (default 0).
 * @returns The two ranges, as `startLevel` and `numLevels` options.
 * @concept generalisation
 * @example
 * ```ts
 * import { levelSplit } from 'whitebox-engine/methods/rl'
 * levelSplit(200, 50).test // => { startLevel: 200, numLevels: 50 }
 * ```
 */
export function levelSplit(
  train: number,
  test: number,
  startLevel = 0,
): {
  readonly train: { readonly startLevel: number; readonly numLevels: number }
  readonly test: { readonly startLevel: number; readonly numLevels: number }
} {
  return { train: { startLevel, numLevels: train }, test: { startLevel: startLevel + train, numLevels: test } }
}

const levelOptions = {
  rows: int(2, 20, { default: 5, description: 'maze cells per column' }),
  cols: int(2, 20, { default: 5, description: 'maze cells per row' }),
  algorithm: choice(['backtracker', 'prim', 'kruskal'], { default: 'backtracker' }),
  braid: real(0, 1, { default: 0, description: 'probability of opening a loop at each dead end' }),
  levelSeed: int(0, 2 ** 31 - 1, { default: 0 }),
  startLevel: int(0, 2 ** 31 - 1, { default: 0 }),
  numLevels: int(0, 100_000, { default: 0, description: 'levels to draw from at reset (0: any level)' }),
  goal: choice(['farthest', 'random'], { default: 'farthest' }),
  observation: choice(['grid', 'view', 'index'], { default: 'grid' }),
  viewRadius: int(1, 5, { default: 2 }),
  goalReward: real(0, 10, { default: 1 }),
  stepReward: real(-1, 0, { default: 0 }),
}

/**
 * ProceduralMaze-v0: a maze generated per level (recursive backtracker, Prim's or Kruskal's algorithm, optionally
 * braided), started at a random cell with the goal at the farthest cell; levels are drawn at reset from a Procgen-style
 * range (`levelSeed`, `startLevel`, `numLevels`), for generalisation studies on held-out levels.
 *
 * @concept procedural-generation
 * @example
 * ```ts
 * import { proceduralMaze } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const level = proceduralMaze.create({ numLevels: 10, startLevel: 5 }).reset(key(0)).info['level'] as number
 * level >= 5 && level < 15 // => true
 * ```
 */
export const proceduralMaze: ProceduralDefinition = {
  id: 'ProceduralMaze-v0',
  description: 'Find the goal of a procedurally generated maze.',
  options: dict(levelOptions),
  create: (options) => makeProcedural({ ...((options ?? {}) as ProceduralOptions), keys: 0 }, 'ProceduralMaze-v0'),
}

/**
 * KeyDoor-v0: a procedural maze with one to four coloured key-door pairs on the way to the goal; each door opens only
 * with its key, which lies before it (through the earlier doors), so the agent must fetch every key in turn.
 *
 * @concept procedural-generation
 * @example
 * ```ts
 * import { keyDoor } from 'whitebox-engine/methods/rl'
 * keyDoor.create({ keys: 2, numLevels: 1 }).level(0).keys.length // => 2
 * ```
 */
export const keyDoor: ProceduralDefinition = {
  id: 'KeyDoor-v0',
  description: 'Fetch the keys, open the matching doors and reach the goal of a procedurally generated maze.',
  options: dict({ ...levelOptions, keys: int(1, MAX_KEYS, { default: 1, description: 'key-door pairs' }) }),
  create: (options) => {
    const o = (options ?? {}) as ProceduralOptions
    return makeProcedural({ ...o, keys: o.keys ?? 1 }, 'KeyDoor-v0')
  },
}
