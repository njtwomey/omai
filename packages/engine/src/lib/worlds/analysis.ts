import { heap, WhiteboxError } from '../base/index.ts'
import { axial } from './_topology.ts'
import {
  cellCentre,
  cellCoords,
  exits,
  type MoveOptions,
  neighbours,
  passable,
  terrainOf,
  type World,
} from './world.ts'

const sources = (w: World, from: number | readonly number[]): number[] => {
  const list = typeof from === 'number' ? [from] : [...from]
  for (const c of list) {
    if (!Number.isInteger(c) || c < 0 || c >= w.n) throw new WhiteboxError('value', `worlds: cell ${c} is off the map`)
  }
  return list
}

/**
 * The cells reachable from some cells by legal moves (walls, terrain, one-way cells and corners respected).
 *
 * @param w - The world.
 * @param from - A cell or cells.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns A mask, 1 for each reachable cell (the sources included).
 * @throws WhiteboxError with code `value` for a source off the map.
 * @example
 * ```ts
 * import { fromAscii, reachable } from 'whitebox-engine/worlds'
 * Array.from(reachable(fromAscii(['..#.']), 0)) // => [1, 1, 0, 0]
 * ```
 */
export function reachable(w: World, from: number | readonly number[], options: MoveOptions = {}): Uint8Array {
  const seen = new Uint8Array(w.n)
  const queue = sources(w, from)
  for (const c of queue) seen[c] = 1
  for (let q = 0; q < queue.length; q++) {
    for (const to of neighbours(w, queue[q] ?? 0, options)) {
      if (seen[to] === 1) continue
      seen[to] = 1
      queue.push(to)
    }
  }
  return seen
}

/** Undirected adjacency of the passable cells: `u` and `v` are joined when either can move to the other. */
function undirected(w: World, options: MoveOptions): number[][] {
  const adj: Set<number>[] = Array.from({ length: w.n }, () => new Set<number>())
  for (let c = 0; c < w.n; c++) {
    if (!passable(w, c) || options.blocked?.(c) === true) continue
    for (const to of neighbours(w, c, options)) {
      adj[c]?.add(to)
      adj[to]?.add(c)
    }
  }
  return adj.map((s) => [...s].sort((a, b) => a - b))
}

/** Connected regions of a world. */
export interface Regions {
  /** The region of each cell (-1 for impassable cells), numbered in order of each region's first cell. */
  readonly labels: Int32Array
  readonly count: number
  /** Cells per region. */
  readonly sizes: readonly number[]
}

/**
 * The connected regions of the passable cells (weakly connected: a one-way move joins its cells).
 *
 * @param w - The world.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns Labels, the count and the sizes.
 * @example
 * ```ts
 * import { fromAscii, regions } from 'whitebox-engine/worlds'
 * const r = regions(fromAscii(['..#.', '..#.']))
 * [r.count, r.sizes] // => [2, [4, 2]]
 * ```
 */
export function regions(w: World, options: MoveOptions = {}): Regions {
  const adj = undirected(w, options)
  const labels = new Int32Array(w.n).fill(-1)
  const sizes: number[] = []
  for (let c = 0; c < w.n; c++) {
    if (labels[c] !== -1 || !passable(w, c) || options.blocked?.(c) === true) continue
    const id = sizes.length
    labels[c] = id
    const queue = [c]
    for (let q = 0; q < queue.length; q++) {
      for (const to of adj[queue[q] ?? 0] ?? []) {
        if (labels[to] !== -1) continue
        labels[to] = id
        queue.push(to)
      }
    }
    sizes.push(queue.length)
  }
  return { labels, count: sizes.length, sizes }
}

/** Options of {@link distanceField} and {@link shortestPath}. */
export interface DistanceOptions extends MoveOptions {
  /**
   * Weigh moves by the terrain cost of the cell entered times the step length (`sqrt 2` for diagonals), by Dijkstra's
   * algorithm; otherwise count moves, by breadth-first search (default).
   */
  readonly weighted?: boolean
}

function edgeCost(w: World, direction: number, to: number): number {
  return terrainOf(w, to).cost * (w.topology.kind === 'square' && direction >= 4 ? Math.SQRT2 : 1)
}

function search(
  w: World,
  from: readonly number[],
  options: DistanceOptions,
): { readonly dist: Float64Array; readonly parent: Int32Array } {
  const dist = new Float64Array(w.n).fill(Number.POSITIVE_INFINITY)
  const parent = new Int32Array(w.n).fill(-1)
  for (const c of from) dist[c] = 0
  if (options.weighted !== true) {
    const queue = [...from]
    for (let q = 0; q < queue.length; q++) {
      const u = queue[q] ?? 0
      for (const v of neighbours(w, u, options)) {
        if (dist[v] !== Number.POSITIVE_INFINITY) continue
        dist[v] = (dist[u] ?? 0) + 1
        parent[v] = u
        queue.push(v)
      }
    }
    return { dist, parent }
  }
  const h = heap<readonly [number, number]>(
    (a, b) => a[0] - b[0] || a[1] - b[1],
    from.map((c) => [0, c] as const),
  )
  const done = new Uint8Array(w.n)
  for (let top = h.pop(); top !== undefined; top = h.pop()) {
    const [d, u] = top
    if (done[u] === 1) continue
    done[u] = 1
    for (const [dir, v] of exits(w, u, options)) {
      const nd = d + edgeCost(w, dir, v)
      if (nd < (dist[v] ?? Number.POSITIVE_INFINITY)) {
        dist[v] = nd
        parent[v] = u
        h.push([nd, v])
      }
    }
  }
  return { dist, parent }
}

/**
 * Shortest-path distances from some cells to every cell, as a field over cells: move counts by breadth-first search,
 * or path costs by Dijkstra's algorithm when `weighted`. A breadth-first field satisfies `d(v) = 1 + min d(u)` over
 * the cells `u` that move to `v`, the discrete eikonal equation.
 *
 * @param w - The world.
 * @param from - A cell or cells (distance 0).
 * @param options - Weighting, connectivity, corner rule and extra blocked cells.
 * @returns The distance of each cell (`Infinity` where unreachable).
 * @throws WhiteboxError with code `value` for a source off the map.
 * @example
 * ```ts
 * import { distanceField, fromAscii } from 'whitebox-engine/worlds'
 * Array.from(distanceField(fromAscii(['...', '#.#']), 0)) // => [0, 1, 2, Infinity, 2, Infinity]
 * Array.from(distanceField(fromAscii(['.~.']), 0, { weighted: true })) // => [0, 5, 6]
 * ```
 */
export function distanceField(w: World, from: number | readonly number[], options: DistanceOptions = {}): Float64Array {
  return search(w, sources(w, from), options).dist
}

/**
 * A shortest path between two cells (fewest moves, or least cost when `weighted`).
 *
 * @param w - The world.
 * @param from - The start cell.
 * @param to - The goal cell.
 * @param options - Weighting, connectivity, corner rule and extra blocked cells.
 * @returns The cells of the path, both ends included, or null when the goal is unreachable.
 * @throws WhiteboxError with code `value` for cells off the map.
 * @example
 * ```ts
 * import { fromAscii, shortestPath } from 'whitebox-engine/worlds'
 * shortestPath(fromAscii(['...', '#.#', '...']), 0, 6) // => [0, 1, 4, 7, 6]
 * ```
 */
export function shortestPath(w: World, from: number, to: number, options: DistanceOptions = {}): number[] | null {
  sources(w, to)
  const { dist, parent } = search(w, sources(w, from), options)
  if (dist[to] === Number.POSITIVE_INFINITY) return null
  const out = [to]
  for (let c = to; c !== from; ) {
    c = parent[c] ?? from
    out.push(c)
    if (out.length > w.n) throw new WhiteboxError('value', 'shortestPath: parent chain did not end')
  }
  return out.reverse()
}

/**
 * The dead ends of a world: passable cells with exactly one neighbour (in a perfect maze, the leaves of its tree).
 *
 * @param w - The world.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns The cells, in order.
 * @example
 * ```ts
 * import { deadEnds, fromAscii } from 'whitebox-engine/worlds'
 * deadEnds(fromAscii(['...', '#.#'])) // => [0, 2, 4]
 * ```
 */
export function deadEnds(w: World, options: MoveOptions = {}): number[] {
  const adj = undirected(w, options)
  const out: number[] = []
  for (let c = 0; c < w.n; c++) if (passable(w, c) && adj[c]?.length === 1) out.push(c)
  return out
}

/** Articulation points and bridges of the undirected cell graph, by Tarjan's low-link algorithm (iteratively). */
function lowLinks(w: World, options: MoveOptions): { readonly cuts: number[]; readonly bridges: [number, number][] } {
  const adj = undirected(w, options)
  const disc = new Int32Array(w.n).fill(-1)
  const low = new Int32Array(w.n)
  const cut = new Uint8Array(w.n)
  const bridges: [number, number][] = []
  let time = 0
  for (let root = 0; root < w.n; root++) {
    if (disc[root] !== -1 || !passable(w, root)) continue
    disc[root] = time
    low[root] = time
    time++
    let rootChildren = 0
    const stack: [number, number, number][] = [[root, -1, 0]]
    while (stack.length > 0) {
      const frame = stack[stack.length - 1] as [number, number, number]
      const [u, parent, i] = frame
      const list = adj[u] ?? []
      if (i < list.length) {
        frame[2] = i + 1
        const v = list[i] ?? 0
        if (v === parent) continue
        if (disc[v] === -1) {
          disc[v] = time
          low[v] = time
          time++
          if (u === root) rootChildren++
          stack.push([v, u, 0])
        } else low[u] = Math.min(low[u] ?? 0, disc[v] ?? 0)
        continue
      }
      stack.pop()
      if (parent >= 0) {
        low[parent] = Math.min(low[parent] ?? 0, low[u] ?? 0)
        if ((low[u] ?? 0) > (disc[parent] ?? 0)) bridges.push(parent < u ? [parent, u] : [u, parent])
        if (parent !== root && (low[u] ?? 0) >= (disc[parent] ?? 0)) cut[parent] = 1
      }
    }
    if (rootChildren > 1) cut[root] = 1
  }
  const cuts: number[] = []
  for (let c = 0; c < w.n; c++) if (cut[c] === 1) cuts.push(c)
  bridges.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return { cuts, bridges }
}

/**
 * The choke points of a world: passable cells whose removal disconnects their region (the articulation points of the
 * cell graph, as NetworkX's `articulation_points`): doorways, corridors and hallways.
 *
 * @param w - The world.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns The cells, in order.
 * @example
 * ```ts
 * import { chokePoints, fromAscii } from 'whitebox-engine/worlds'
 * chokePoints(fromAscii(['..#..', '.....', '..#..'])) // => [6, 7, 8]
 * ```
 */
export function chokePoints(w: World, options: MoveOptions = {}): number[] {
  return lowLinks(w, options).cuts
}

/**
 * The bridges of a world: adjacent pairs of cells whose connection is the only route between its two sides (as
 * NetworkX's `bridges`); in a perfect maze every open passage is a bridge.
 *
 * @param w - The world.
 * @param options - Connectivity, corner rule and extra blocked cells.
 * @returns Pairs `[a, b]` with `a < b`, sorted.
 * @example
 * ```ts
 * import { bridges, fromAscii } from 'whitebox-engine/worlds'
 * bridges(fromAscii(['..#..', '.....', '..#..'])) // => [[6, 7], [7, 8]]
 * bridges(fromAscii(['...'])) // => [[0, 1], [1, 2]]
 * ```
 */
export function bridges(w: World, options: MoveOptions = {}): [number, number][] {
  return lowLinks(w, options).bridges
}

/** A world's cell graph as edges `graph` (`whitebox-engine/graph`) can build: `graph(edges, { nodes: labels, directed })`. */
export interface Adjacency {
  /** The number of nodes. */
  readonly n: number
  /** Node labels (`r,c` on grids, the node index on graphs). */
  readonly labels: readonly string[]
  /** The cell of each node. */
  readonly cells: readonly number[]
  /** Edges between node indices, weighted by move cost (1 unless weighted). */
  readonly edges: readonly (readonly [number, number, number])[]
  /** Directed when some move cannot be reversed at the same cost (one-way cells, weighted terrain). */
  readonly directed: boolean
}

/**
 * The cell graph of a world in a form graph algorithms consume: nodes for the passable cells (or all cells), an edge
 * per legal move, weighted by terrain cost times step length when `weighted`. Undirected (one edge per pair) when
 * every move can be reversed at the same cost; otherwise directed with one edge per move. Build it with
 * `graph(adj.edges, { nodes: adj.labels, directed: adj.directed })` from `whitebox-engine/graph`.
 *
 * @param w - The world.
 * @param options - Which cells become nodes, weighting, connectivity, corner rule and extra blocked cells.
 * @returns The adjacency.
 * @example
 * ```ts
 * import { fromAscii, toAdjacency } from 'whitebox-engine/worlds'
 * import { graph } from 'whitebox-engine/graph'
 * const adj = toAdjacency(fromAscii(['..', '#.']))
 * [adj.labels, adj.edges, adj.directed] // => [['0,0', '0,1', '1,1'], [[0, 1, 1], [1, 2, 1]], false]
 * graph(adj.edges, { nodes: adj.labels }).n // => 3
 * ```
 */
export function toAdjacency(
  w: World,
  options: DistanceOptions & { readonly cells?: 'passable' | 'all' } = {},
): Adjacency {
  const cells: number[] = []
  for (let c = 0; c < w.n; c++) if (options.cells === 'all' || passable(w, c)) cells.push(c)
  const node = new Int32Array(w.n).fill(-1)
  for (const [i, c] of cells.entries()) node[c] = i
  const moves = new Map<string, number>()
  for (const c of cells) {
    if (!passable(w, c) || options.blocked?.(c) === true) continue
    for (const [dir, to] of exits(w, c, options)) {
      moves.set(`${c},${to}`, options.weighted === true ? edgeCost(w, dir, to) : 1)
    }
  }
  let directed = false
  for (const [k, cost] of moves) {
    const [a, b] = k.split(',')
    if (moves.get(`${b},${a}`) !== cost) directed = true
  }
  const edges: [number, number, number][] = []
  for (const [k, cost] of moves) {
    const [a = 0, b = 0] = k.split(',').map(Number)
    if (!directed && b < a) continue
    edges.push([node[a] ?? 0, node[b] ?? 0, cost])
  }
  const labels = cells.map((c) => (w.topology.kind === 'graph' ? String(c) : cellCoords(w, c).join(',')))
  return { n: cells.length, labels, cells, edges, directed }
}

/** A distance between cells for heuristics. */
export type CellMetric = 'manhattan' | 'chebyshev' | 'octile' | 'euclidean' | 'hex'

/**
 * A distance between two cells: Manhattan, Chebyshev or octile on square grids (rows and columns), the hexagonal
 * step distance on hexagonal grids, or the Euclidean distance between centres anywhere. Each is an admissible A*
 * heuristic for the matching moves and unit costs: Manhattan for 4-connected moves, octile for 8-connected moves with
 * `sqrt 2` diagonals, Chebyshev for 8-connected unit moves, hexagonal for hexagons.
 *
 * @param w - The world.
 * @param a - A cell.
 * @param b - Another cell.
 * @param metric - The metric (default Manhattan on squares, hexagonal on hexagons, Euclidean on graphs).
 * @returns The distance.
 * @throws WhiteboxError with code `value` for a grid metric on a graph world.
 * @example
 * ```ts
 * import { cellDistance, fromAscii } from 'whitebox-engine/worlds'
 * const w = fromAscii(['....', '....', '....'])
 * [cellDistance(w, 0, 11), cellDistance(w, 0, 11, 'chebyshev'), cellDistance(w, 0, 11, 'euclidean')] // => [5, 3, Math.hypot(2, 3)]
 * ```
 */
export function cellDistance(w: World, a: number, b: number, metric?: CellMetric): number {
  const t = w.topology
  const m = metric ?? (t.kind === 'square' ? 'manhattan' : t.kind === 'hex' ? 'hex' : 'euclidean')
  if (m === 'euclidean') {
    const [x1, y1] = cellCentre(w, a)
    const [x2, y2] = cellCentre(w, b)
    return Math.hypot(x2 - x1, y2 - y1)
  }
  if (t.kind === 'graph') throw new WhiteboxError('value', `cellDistance: '${m}' needs a grid`)
  if (m === 'hex') {
    const [q1, r1] = axial(t.cols, a)
    const [q2, r2] = axial(t.cols, b)
    return (Math.abs(q1 - q2) + Math.abs(r1 - r2) + Math.abs(q1 + r1 - q2 - r2)) / 2
  }
  const [r1, c1] = cellCoords(w, a)
  const [r2, c2] = cellCoords(w, b)
  const dr = Math.abs(r1 - r2)
  const dc = Math.abs(c1 - c2)
  if (m === 'manhattan') return dr + dc
  if (m === 'chebyshev') return Math.max(dr, dc)
  return Math.max(dr, dc) + (Math.SQRT2 - 1) * Math.min(dr, dc)
}

/**
 * A heuristic to the nearest of some goal cells (the world's goals by default), for A* and greedy search.
 *
 * @param w - The world.
 * @param options - The goals and the metric (as {@link cellDistance}).
 * @returns A function from a cell to its estimated distance.
 * @example
 * ```ts
 * import { fromAscii, heuristicTo } from 'whitebox-engine/worlds'
 * heuristicTo(fromAscii(['S..G', '...G']))(0) // => 3
 * ```
 */
export function heuristicTo(
  w: World,
  options: { readonly goals?: readonly number[]; readonly metric?: CellMetric } = {},
): (cell: number) => number {
  const goals = options.goals ?? w.goals
  return (cell) => {
    let best = Number.POSITIVE_INFINITY
    for (const g of goals) best = Math.min(best, cellDistance(w, cell, g, options.metric))
    return best
  }
}
