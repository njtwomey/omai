/**
 * The shortest-path algorithms as steppable algorithms: Dijkstra and A* (one node settled per step, with the frontier
 * and the distances so far), Bellman-Ford (one round of relaxations per step) and Floyd-Warshall (one intermediate
 * node per step). Each has a state that serialises and resumes and logs the paths of {@link shortestPathContract};
 * `dijkstra`, `bellmanFord`, `aStar` and `floydWarshall` in `paths.ts` are these run to the end.
 */

import { type Json, OmaiError } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  event,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { floatsFromJson, floatsToJson, weightFunction } from './_run.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { IndexedQueue } from './structures.ts'

/** The paths the shortest-path algorithms log, as a type. */
export type ShortestPathPath =
  | 'sp/settled'
  | 'sp/priority'
  | 'sp/relaxed'
  | 'sp/frontier'
  | 'sp/distance'
  | 'sp/predecessor'
  | 'sp/round'
  | 'sp/pivot'
  | 'sp/improved'
  | 'sp/negative-cycle'

/**
 * What the shortest-path algorithms log, by path. Dijkstra and A* log the node settled and its priority, how many
 * edges improved a distance, the frontier in the order it will be settled, and the distances and predecessors so far;
 * Bellman-Ford logs the round, the edges improved and the same arrays; Floyd-Warshall logs the pivot, the pairs
 * improved and the matrices. `sp/negative-cycle` fires when a negative cycle is found.
 */
export const shortestPathContract: Contract<unknown, ShortestPathPath> = defineContract({
  'sp/settled': scalar({ description: 'the node settled at this step (Dijkstra, A*)' }),
  'sp/priority': scalar({ description: 'the priority of the settled node: its distance plus the weighted heuristic' }),
  'sp/relaxed': scalar({ description: 'edges that improved a distance at this step' }),
  'sp/frontier': tensor({
    role: 'state',
    shape: ['frontier'],
    description: 'the nodes waiting to be settled, in the order they will be',
  }),
  'sp/distance': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the distance found so far (Infinity when unreached); for Floyd-Warshall an n by n matrix, flat',
  }),
  'sp/predecessor': tensor({
    role: 'state',
    shape: ['nodes'],
    description:
      'the node before each on its best path so far (-1 for none); for Floyd-Warshall the n by n matrix, flat',
  }),
  'sp/round': scalar({ description: 'the round of Bellman-Ford relaxations, counting from 1' }),
  'sp/pivot': scalar({ description: 'the intermediate node Floyd-Warshall allows at this step' }),
  'sp/improved': scalar({ description: 'pairs whose distance the pivot improved' }),
  'sp/negative-cycle': event({ context: { step: 'the step at which the cycle showed' } }),
})

/** What every single-source shortest-path state holds: the distances and predecessors found so far. */
export interface ShortestPathState {
  /** The best distance found from the source (Infinity when not yet reached). */
  readonly distance: Float64Array
  /** The node before each on its best path (-1 for the source and unreached nodes). */
  readonly predecessor: Int32Array
  /** Edges that improved a distance at the last step. */
  readonly relaxed: number
}

/**
 * The state of Dijkstra's algorithm and A*: the best costs and predecessors, which nodes are settled, the frontier as a
 * priority queue, and the settling order. Steps update the arrays in place (the search only ever improves them), so
 * keep a copy of a state you want to look at later.
 */
export interface PathSearchState extends ShortestPathState {
  /** 1 for a settled node. */
  readonly closed: Uint8Array
  /** The frontier, with the priority (cost plus weighted heuristic) of each queued node. */
  readonly queue: IndexedQueue
  /** Nodes in the order they were settled. */
  readonly expanded: number[]
  /** The node settled at the last step (-1 before the first). */
  readonly current: number
}

/** The state of {@link bellmanFordAlgorithm}. */
export interface BellmanFordState extends ShortestPathState {
  /** Rounds of relaxations done. */
  readonly rounds: number
  /** True once a round changed nothing: the distances are final. */
  readonly stable: boolean
  /** True when the n-th round still improved a distance: a negative cycle is reachable. */
  readonly negativeCycle: boolean
}

/** The state of {@link floydWarshallAlgorithm}: the n by n matrices, row-major. */
export interface FloydWarshallState {
  readonly n: number
  /** `distance[i * n + j]`: the shortest distance from i to j using the pivots so far as intermediates. */
  readonly distance: Float64Array
  /** `predecessor[i * n + j]`: the node before j on that path (-1 for none). */
  readonly predecessor: Int32Array
  /** The next intermediate node to allow. */
  readonly pivot: number
  /** Pairs improved by the last pivot. */
  readonly improved: number
  /** True when a node has a negative distance to itself. */
  readonly negativeCycle: boolean
}

/** Options shared by the shortest-path algorithms. */
export interface ShortestPathOptions {
  /** Use edge weights (default true); false counts hops. */
  readonly weighted?: boolean
}

/** Options of {@link dijkstraAlgorithm}. */
export interface DijkstraAlgorithmOptions extends ShortestPathOptions {
  /** Stop once this node is settled. */
  readonly target?: NodeId
  /** Edges (by position in `graph.edges`) the search may not use, as Yen's algorithm removes them. */
  readonly blockedEdges?: ReadonlySet<number>
  /** Nodes the search may not enter. */
  readonly blockedNodes?: ReadonlySet<number>
  /** The weight of each adjacency entry of `graph.out` in place of the edge weights (Johnson's reweighting). */
  readonly weights?: (entry: number) => number
}

/** Options of {@link aStarAlgorithm}. */
export interface AStarAlgorithmOptions extends ShortestPathOptions {
  /** A lower bound on the remaining distance to the target (default 0, which is Dijkstra). */
  readonly heuristic?: (v: number) => number
  /**
   * The weight on the heuristic (default 1: A*). Above 1 the search is greedier and faster, and the path it finds is
   * at most this factor longer than the best when the heuristic is admissible.
   */
  readonly weight?: number
}

/** What builds a path search: shared by Dijkstra and A*, with the blocked parts Yen's algorithm needs. */
interface PathSearchSpec {
  readonly name: string
  readonly g: Graph
  readonly source: number
  readonly target: number
  readonly weight: (k: number) => number
  readonly heuristic: (v: number) => number
  readonly heuristicWeight: number
  readonly negativeMessage: string
  readonly blockedEdges?: ReadonlySet<number> | undefined
  readonly blockedNodes?: ReadonlySet<number> | undefined
}

const INF = Number.POSITIVE_INFINITY

/** The queued nodes in the order they will come out: by priority, ties by node index. */
function frontierOf(queue: IndexedQueue, n: number): number[] {
  const nodes: number[] = []
  for (let v = 0; v < n; v++) if (queue.has(v)) nodes.push(v)
  return nodes.sort((a, b) => (queue.priority(a) ?? 0) - (queue.priority(b) ?? 0) || a - b)
}

/** Best-first search over a graph's out-edges: Dijkstra with a zero heuristic, A* with another. */
function pathSearch(spec: PathSearchSpec): Algorithm<PathSearchState, ShortestPathPath> {
  const { g, source, target, heuristic: h, heuristicWeight: hw } = spec
  const a = g.out
  const n = g.n
  return {
    name: spec.name,
    config: { nodes: n, source, target, heuristicWeight: hw },
    init: () => {
      const distance = new Float64Array(n).fill(INF)
      distance[source] = 0
      const queue = new IndexedQueue(n)
      queue.push(source, hw * h(source))
      return {
        distance,
        predecessor: new Int32Array(n).fill(-1),
        closed: new Uint8Array(n),
        queue,
        expanded: [],
        current: -1,
        relaxed: 0,
      }
    },
    step: (s, ctx) => {
      const u = s.queue.pop()
      if (u === undefined) return s
      s.closed[u] = 1
      s.expanded.push(u)
      let relaxed = 0
      if (u !== target) {
        for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
          const v = a.indices[k] ?? 0
          if (
            s.closed[v] === 1 ||
            spec.blockedEdges?.has(a.edges[k] ?? -1) === true ||
            spec.blockedNodes?.has(v) === true
          )
            continue
          const w = spec.weight(k)
          if (w < 0) throw new OmaiError('domain', spec.negativeMessage)
          const alt = (s.distance[u] ?? 0) + w
          if (alt < (s.distance[v] ?? INF)) {
            s.distance[v] = alt
            s.predecessor[v] = u
            s.queue.push(v, alt + hw * h(v))
            relaxed += 1
          }
        }
      }
      if (ctx.wants('sp/settled')) ctx.log('sp/settled', u)
      if (ctx.wants('sp/priority')) ctx.log('sp/priority', (s.distance[u] ?? 0) + hw * h(u))
      if (ctx.wants('sp/relaxed')) ctx.log('sp/relaxed', relaxed)
      if (ctx.wants('sp/frontier')) ctx.log('sp/frontier', frontierOf(s.queue, n))
      if (ctx.wants('sp/distance')) ctx.log('sp/distance', Float64Array.from(s.distance))
      if (ctx.wants('sp/predecessor')) ctx.log('sp/predecessor', Int32Array.from(s.predecessor))
      return { ...s, current: u, relaxed }
    },
    done: (s) => (s.queue.size === 0 || (target >= 0 && s.current === target) ? 'converged' : undefined),
    serialise: (s) => {
      const queued = frontierOf(s.queue, n)
      return {
        distance: floatsToJson(s.distance),
        predecessor: stateToJson(s.predecessor),
        closed: stateToJson(s.closed),
        queue: queued.map((v) => [v, s.queue.priority(v) ?? 0]),
        expanded: [...s.expanded],
        current: s.current,
        relaxed: s.relaxed,
      }
    },
    deserialise: (j) => {
      const o = j as { readonly [key: string]: Json }
      const queue = new IndexedQueue(n)
      for (const [v, p] of o['queue'] as unknown as readonly (readonly [number, number])[]) queue.push(v, p)
      return {
        distance: floatsFromJson(o['distance'] as Json),
        predecessor: stateFromJson(o['predecessor'] as Json) as Int32Array,
        closed: stateFromJson(o['closed'] as Json) as Uint8Array,
        queue,
        expanded: [...(o['expanded'] as unknown as readonly number[])],
        current: o['current'] as number,
        relaxed: o['relaxed'] as number,
      }
    },
  }
}

/**
 * Dijkstra's algorithm as a steppable algorithm: each step settles the queued node of least distance (ties in node
 * order) and relaxes its out-edges, so the frontier, the distances found and the predecessors can be watched growing.
 * The state serialises, and a run resumes from a checkpoint. A negative weight, when reached, throws.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param options - `weighted` (default true), and a `target` to stop at once it is settled.
 * @returns The algorithm; it is done when the frontier is empty or the target is settled.
 * @throws OmaiError with code `domain` when a negative weight is reached while stepping.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { dijkstraAlgorithm, graph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = graph([[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 5]])
 * const run = runAlgorithm(dijkstraAlgorithm(g, 0), { maxSteps: 10 })
 * [run.steps, run.state.expanded, Array.from(run.state.distance)] // => [4, [0, 2, 1, 3], [0, 3, 1, 8]]
 * ```
 */
export function dijkstraAlgorithm(
  g: Graph,
  source: NodeId,
  options: DijkstraAlgorithmOptions = {},
): Algorithm<PathSearchState, ShortestPathPath> {
  return pathSearch({
    name: 'graph/dijkstra',
    g,
    source: nodeIndex(g, source),
    target: options.target === undefined ? -1 : nodeIndex(g, options.target),
    weight: options.weights ?? weightFunction(g, options.weighted),
    heuristic: () => 0,
    heuristicWeight: 1,
    negativeMessage: 'dijkstra: negative edge weights need bellmanFord or johnson',
    blockedEdges: options.blockedEdges,
    blockedNodes: options.blockedNodes,
  })
}

/**
 * A* search as a steppable algorithm: Dijkstra ordered by cost so far plus a heuristic estimate of the rest, so a good
 * heuristic settles far fewer nodes. Each step settles one node; it is done when the target is settled. With the
 * default heuristic (zero) it is Dijkstra; a `weight` above one trades the optimality guarantee for speed.
 *
 * @param g - The graph (non-negative weights).
 * @param source - The source.
 * @param target - The target.
 * @param options - The heuristic (default 0), its weight (default 1) and `weighted` (default true).
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a weight below 1, and `domain` for a negative edge weight while stepping.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { aStarAlgorithm, gridGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = gridGraph(5, 5)
 * const manhattan = (v: number) => Math.abs(Math.floor(v / 5) - 1) + Math.abs((v % 5) - 1)
 * const informed = runAlgorithm(aStarAlgorithm(g, 0, 6, { heuristic: manhattan }), { maxSteps: 30 })
 * const blind = runAlgorithm(aStarAlgorithm(g, 0, 6), { maxSteps: 30 })
 * [informed.state.distance[6], informed.state.expanded.length < blind.state.expanded.length] // => [2, true]
 * ```
 */
export function aStarAlgorithm(
  g: Graph,
  source: NodeId,
  target: NodeId,
  options: ShortestPathOptions & AStarAlgorithmOptions = {},
): Algorithm<PathSearchState, ShortestPathPath> {
  const weight = options.weight ?? 1
  if (!(weight >= 1)) throw new OmaiError('value', `aStarAlgorithm: the heuristic weight ${weight} must be at least 1`)
  return pathSearch({
    name: 'graph/a-star',
    g,
    source: nodeIndex(g, source),
    target: nodeIndex(g, target),
    weight: weightFunction(g, options.weighted),
    heuristic: options.heuristic ?? (() => 0),
    heuristicWeight: weight,
    negativeMessage: 'aStar: negative edge weights are not allowed',
  })
}

/**
 * Bellman-Ford as a steppable algorithm: each step is one round that relaxes every edge in node order, so the
 * shortest paths of at most r edges are known after round r. It is done when a round changes nothing, or, when round
 * n still improves a distance, flags a negative cycle (status `diverged`; `bellmanFord` then throws).
 *
 * @param g - The graph.
 * @param source - The source.
 * @param options - `weighted` (default true).
 * @returns The algorithm.
 * @concept shortest-paths
 * @example
 * ```ts
 * import { bellmanFordAlgorithm, digraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(bellmanFordAlgorithm(digraph([[0, 1, 4], [0, 2, 5], [2, 1, -3]]), 0), { maxSteps: 10 })
 * [run.state.rounds, run.status, Array.from(run.state.distance)] // => [2, 'converged', [0, 2, 5]]
 * ```
 */
export function bellmanFordAlgorithm(
  g: Graph,
  source: NodeId,
  options: ShortestPathOptions = {},
): Algorithm<BellmanFordState, ShortestPathPath> {
  const s0 = nodeIndex(g, source)
  const weight = weightFunction(g, options.weighted)
  const a = g.out
  return {
    name: 'graph/bellman-ford',
    config: { nodes: g.n, source: s0 },
    init: () => {
      const distance = new Float64Array(g.n).fill(INF)
      distance[s0] = 0
      return {
        distance,
        predecessor: new Int32Array(g.n).fill(-1),
        relaxed: 0,
        rounds: 0,
        stable: false,
        negativeCycle: false,
      }
    },
    step: (s, ctx) => {
      const distance = Float64Array.from(s.distance)
      const predecessor = Int32Array.from(s.predecessor)
      let relaxed = 0
      for (let u = 0; u < g.n; u++) {
        const du = distance[u] ?? INF
        if (du === INF) continue
        for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
          const v = a.indices[k] ?? 0
          const alt = du + weight(k)
          if (alt < (distance[v] ?? INF)) {
            distance[v] = alt
            predecessor[v] = u
            relaxed += 1
          }
        }
      }
      const rounds = s.rounds + 1
      const negativeCycle = relaxed > 0 && rounds >= g.n
      if (ctx.wants('sp/round')) ctx.log('sp/round', rounds)
      if (ctx.wants('sp/relaxed')) ctx.log('sp/relaxed', relaxed)
      if (ctx.wants('sp/distance')) ctx.log('sp/distance', Float64Array.from(distance))
      if (ctx.wants('sp/predecessor')) ctx.log('sp/predecessor', Int32Array.from(predecessor))
      if (negativeCycle) ctx.emit('sp/negative-cycle', { step: ctx.t })
      return { distance, predecessor, relaxed, rounds, stable: relaxed === 0, negativeCycle }
    },
    done: (s) => (s.negativeCycle ? 'diverged' : s.stable ? 'converged' : undefined),
    serialise: (s) => ({
      distance: floatsToJson(s.distance),
      predecessor: stateToJson(s.predecessor),
      relaxed: s.relaxed,
      rounds: s.rounds,
      stable: s.stable,
      negativeCycle: s.negativeCycle,
    }),
    deserialise: (j) => {
      const o = j as { readonly [key: string]: Json }
      return {
        distance: floatsFromJson(o['distance'] as Json),
        predecessor: stateFromJson(o['predecessor'] as Json) as Int32Array,
        relaxed: o['relaxed'] as number,
        rounds: o['rounds'] as number,
        stable: o['stable'] as boolean,
        negativeCycle: o['negativeCycle'] as boolean,
      }
    },
  }
}

/**
 * Floyd-Warshall as a steppable algorithm: each step allows one more node as an intermediate and relaxes every pair
 * through it, so after step k the matrices are exact for paths whose intermediates are among the first k nodes. It
 * stops early, with status `diverged`, as soon as a node has a negative distance to itself.
 *
 * @param g - The graph.
 * @param options - `weighted` (default true).
 * @returns The algorithm; it is done after the last pivot.
 * @throws OmaiError with code `value` for a negative self-loop (a negative cycle of one edge).
 * @concept shortest-paths
 * @example
 * ```ts
 * import { floydWarshallAlgorithm, pathGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(floydWarshallAlgorithm(pathGraph(3)), { maxSteps: 3 })
 * [run.state.pivot, Array.from(run.state.distance)] // => [3, [0, 1, 2, 1, 0, 1, 2, 1, 0]]
 * ```
 */
export function floydWarshallAlgorithm(
  g: Graph,
  options: ShortestPathOptions = {},
): Algorithm<FloydWarshallState, ShortestPathPath> {
  const n = g.n
  const weight = weightFunction(g, options.weighted)
  const a = g.out
  return {
    name: 'graph/floyd-warshall',
    config: { nodes: n },
    init: () => {
      const d = new Float64Array(n * n).fill(INF)
      const p = new Int32Array(n * n).fill(-1)
      for (let i = 0; i < n; i++) d[i * n + i] = 0
      for (let u = 0; u < n; u++) {
        for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
          const v = a.indices[k] ?? 0
          const w = weight(k)
          if (u === v) {
            if (w < 0) throw new OmaiError('value', 'floydWarshall: the graph has a negative cycle')
            continue
          }
          if (w < (d[u * n + v] ?? INF)) {
            d[u * n + v] = w
            p[u * n + v] = u
          }
        }
      }
      return { n, distance: d, predecessor: p, pivot: 0, improved: 0, negativeCycle: false }
    },
    step: (s, ctx) => {
      const k = s.pivot
      const d = Float64Array.from(s.distance)
      const p = Int32Array.from(s.predecessor)
      let improved = 0
      for (let i = 0; i < n; i++) {
        const dik = d[i * n + k] ?? INF
        if (dik === INF) continue
        for (let j = 0; j < n; j++) {
          const alt = dik + (d[k * n + j] ?? INF)
          if (alt < (d[i * n + j] ?? INF)) {
            d[i * n + j] = alt
            p[i * n + j] = p[k * n + j] ?? -1
            improved += 1
          }
        }
      }
      let negativeCycle = false
      for (let i = 0; i < n; i++) if ((d[i * n + i] ?? 0) < 0) negativeCycle = true
      if (ctx.wants('sp/pivot')) ctx.log('sp/pivot', k)
      if (ctx.wants('sp/improved')) ctx.log('sp/improved', improved)
      if (ctx.wants('sp/distance')) ctx.log('sp/distance', Float64Array.from(d))
      if (ctx.wants('sp/predecessor')) ctx.log('sp/predecessor', Int32Array.from(p))
      if (negativeCycle) ctx.emit('sp/negative-cycle', { step: ctx.t })
      return { n, distance: d, predecessor: p, pivot: k + 1, improved, negativeCycle }
    },
    done: (s) => (s.negativeCycle ? 'diverged' : s.pivot >= n ? 'converged' : undefined),
    serialise: (s) => ({
      n: s.n,
      distance: floatsToJson(s.distance),
      predecessor: stateToJson(s.predecessor),
      pivot: s.pivot,
      improved: s.improved,
      negativeCycle: s.negativeCycle,
    }),
    deserialise: (j) => {
      const o = j as { readonly [key: string]: Json }
      return {
        n: o['n'] as number,
        distance: floatsFromJson(o['distance'] as Json),
        predecessor: stateFromJson(o['predecessor'] as Json) as Int32Array,
        pivot: o['pivot'] as number,
        improved: o['improved'] as number,
        negativeCycle: o['negativeCycle'] as boolean,
      }
    },
  }
}
