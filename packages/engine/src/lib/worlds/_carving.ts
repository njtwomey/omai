// The step machines of the maze generators: each takes a maze state and the step's uniforms and returns the next
// state, copying what it changes. Private to worlds (generators.ts wraps them as traced algorithms).
import type { Key } from '../random/index.ts'
import { foldIn, permutation, uniform } from '../random/index.ts'
import { oppositeSide, sideBetween, sideCount, step } from './_topology.ts'
import type { MazeAlgorithmName, MazeOptions, MazeState } from './generators.ts'
import { type Topology, type World, world } from './world.ts'

export const SQUARE_ONLY: readonly MazeAlgorithmName[] = ['eller', 'binaryTree', 'sidewinder', 'division']

export function baseWorld(options: MazeOptions): World {
  if (options.base !== undefined) return { ...options.base, walls: new Uint32Array(options.base.n) }
  const rows = options.rows ?? 8
  const cols = options.cols ?? 8
  const t: Topology = { kind: options.topology === 'hex' ? 'hex' : 'square', rows, cols }
  const w = world({ topology: t })
  return { ...w, starts: [0], goals: [w.n - 1] }
}

/** Every side neighbour of a cell, as `[side, cell]`. */
export function sides(t: Topology, cell: number): [number, number][] {
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
export function toggled(t: Topology, walls: Uint32Array, a: number, b: number, open: boolean): Uint32Array {
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

export const draw = (k: Key, n: number): Float64Array => uniform(k, [n]).data as Float64Array
export const pick = (u: number, n: number): number => Math.min(n - 1, Math.floor(u * n))

export function initial(base: World, algorithm: MazeAlgorithmName, key: Key): MazeState {
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

export function stepBacktracker(s: MazeState, u: Float64Array): MazeState {
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

export function stepPrim(s: MazeState, u: Float64Array): MazeState {
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

export function stepKruskal(s: MazeState): MazeState {
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

export function stepWilson(s: MazeState, u: Float64Array): MazeState {
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

export function stepAldousBroder(s: MazeState, u: Float64Array): MazeState {
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

export function stepEller(s: MazeState, u: Float64Array, rows: number, cols: number): MazeState {
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

export function stepBinaryTree(s: MazeState, u: Float64Array, rows: number, cols: number, bias: string): MazeState {
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

export function stepSidewinder(s: MazeState, u: Float64Array, rows: number, cols: number): MazeState {
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

export function stepDivision(s: MazeState, u: Float64Array, cols: number): MazeState {
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
