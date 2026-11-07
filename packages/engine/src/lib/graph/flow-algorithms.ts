/**
 * The flow algorithms as steppable algorithms: maximum flow by augmenting paths (Edmonds-Karp or Dinic, one augmentation
 * per step, with the residual capacities, the path and the source side of the cut) and minimum-cost flow by successive
 * shortest paths (one augmentation per step, with the node potentials). Each state serialises and resumes and logs the
 * paths of {@link flowContract}; the slot is the option table {@link maxFlowMethods}. `maxFlow`, `minCut`, `minCostFlow`
 * and `maxFlowMinCost` in `flows.ts` are these run to the end.
 */

import { type Json, kahanSum, MdmlError } from '../base/index.ts'
import { dict, knobOption, type OptionTable, optionTable } from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { augment, EPS, type ResidualNetwork, residualNetwork, sourceSide } from './_residual.ts'
import { floatsFromJson, floatsToJson } from './_run.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { IndexedQueue } from './structures.ts'

/** The paths the flow algorithms log, as a type. */
export type FlowPath =
  | 'flow/augment'
  | 'flow/value'
  | 'flow/cost'
  | 'flow/phase'
  | 'flow/residual'
  | 'flow/path'
  | 'flow/cut'
  | 'flow/potential'

/**
 * What the flow algorithms log, by path: the amount the augmentation pushed, the flow value (and, for minimum-cost
 * flow, the cost) so far, the Dinic phase, the residual capacities of every arc (arc 2e runs along edge e, arc 2e + 1
 * against it), the arcs of the augmenting path, the source side of the cut (the nodes still reachable from the source;
 * at the last step it is a minimum cut) and the node potentials of minimum-cost flow.
 */
export const flowContract: Contract<unknown, FlowPath> = defineContract({
  'flow/augment': scalar({ description: 'the amount pushed along the augmenting path at this step' }),
  'flow/value': scalar({ description: 'the flow value so far' }),
  'flow/cost': scalar({ description: 'the cost of the flow so far (minimum-cost flow)' }),
  'flow/phase': scalar({ description: 'the Dinic phase (blocking flow) the augmentation belongs to, from 1' }),
  'flow/residual': tensor({ role: 'state', shape: ['arcs'], description: 'the residual capacity of every arc' }),
  'flow/path': tensor({
    role: 'state',
    shape: ['path'],
    description: 'the arcs of the augmenting path, from the source',
  }),
  'flow/cut': tensor({
    role: 'state',
    shape: ['nodes'],
    description: '1 for nodes reachable from the source in the residual network: the source side of the cut',
  }),
  'flow/potential': tensor({ role: 'state', shape: ['nodes'], description: 'the node potentials (minimum-cost flow)' }),
})

/** Options of the flow functions. */
export interface FlowOptions {
  /** Capacity per edge, in edge order (default: the edge weights); may be Infinity. */
  readonly capacity?: ArrayLike<number>
}

/** The method of a maximum flow. */
export type MaxFlowMethod = 'edmonds-karp' | 'dinic'

/** Options of {@link maxFlowAlgorithm}. */
export interface MaxFlowAlgorithmOptions extends FlowOptions {
  /** How augmenting paths are found (default Dinic). */
  readonly method?: MaxFlowMethod
}

/** The state of {@link maxFlowAlgorithm}. */
export interface MaxFlowState {
  /** The residual capacity of every arc (arc 2e along edge e, arc 2e + 1 against it). */
  readonly residual: Float64Array
  /** The flow value so far. */
  readonly value: number
  /** Augmentations applied so far. */
  readonly augmentations: number
  /** The augmenting path found for the next step, as arcs from the source (empty when the flow is maximum). */
  readonly pending: readonly number[]
  /** The path applied at the last step. */
  readonly last: readonly number[]
  /** The amount pushed at the last step. */
  readonly pushed: number
  /** Dinic: the BFS level of each node in the current phase (-1 for none or a dead end). */
  readonly level: Int32Array
  /** Dinic: the current-arc pointer of each node. */
  readonly next: Int32Array
  /** Dinic: whether `level` is valid for the current phase. */
  readonly leveled: boolean
  /** Dinic: phases started. */
  readonly phases: number
}

type Mutable<T> = { -readonly [K in keyof T]: T[K] }

/** Edmonds-Karp: the shortest augmenting path by breadth-first search ([] when there is none). */
function shortestAugmenting(r: ResidualNetwork, s: number, t: number): number[] {
  const via = new Int32Array(r.n).fill(-1)
  const seen = new Uint8Array(r.n)
  seen[s] = 1
  const queue = [s]
  for (let head = 0; head < queue.length && seen[t] === 0; head++) {
    const u = queue[head] ?? 0
    for (const a of r.arcs[u] ?? []) {
      const v = r.to[a] ?? 0
      if (seen[v] === 1 || (r.residual[a] ?? 0) <= EPS) continue
      seen[v] = 1
      via[v] = a
      queue.push(v)
    }
  }
  if (seen[t] === 0) return []
  const path: number[] = []
  for (let v = t; v !== s; v = r.to[(via[v] ?? 0) ^ 1] ?? s) path.push(via[v] ?? 0)
  return path.reverse()
}

/** Dinic: the next path of the blocking flow, starting a new phase when the last one is exhausted ([] when done). */
function blockingPath(r: ResidualNetwork, s: number, t: number, st: Mutable<MaxFlowState>): number[] {
  for (;;) {
    if (!st.leveled) {
      st.level.fill(-1)
      st.next.fill(0)
      st.level[s] = 0
      const queue = [s]
      for (let head = 0; head < queue.length; head++) {
        const u = queue[head] ?? 0
        for (const a of r.arcs[u] ?? []) {
          const v = r.to[a] ?? 0
          if (st.level[v] === -1 && (r.residual[a] ?? 0) > EPS) {
            st.level[v] = (st.level[u] ?? 0) + 1
            queue.push(v)
          }
        }
      }
      st.leveled = true
      st.phases += 1
      if (st.level[t] === -1) return []
    }
    // Depth-first along level-increasing arcs, with a current-arc pointer per node.
    const path: number[] = []
    let u = s
    while (u !== t) {
      const list = r.arcs[u] ?? new Int32Array(0)
      let advanced = false
      while ((st.next[u] ?? 0) < list.length) {
        const a = list[st.next[u] ?? 0] ?? 0
        const v = r.to[a] ?? 0
        if ((r.residual[a] ?? 0) > EPS && st.level[v] === (st.level[u] ?? 0) + 1) {
          path.push(a)
          u = v
          advanced = true
          break
        }
        st.next[u] = (st.next[u] ?? 0) + 1
      }
      if (advanced) continue
      // Dead end: retreat and skip the arc that led here.
      st.level[u] = -1
      const back = path.pop()
      if (back === undefined) break
      u = r.to[back ^ 1] ?? s
      st.next[u] = (st.next[u] ?? 0) + 1
    }
    if (u === t) return path
    st.leveled = false
  }
}

/**
 * Maximum flow as a steppable algorithm: each step applies one augmenting path (found by Edmonds-Karp's breadth-first
 * search, or by Dinic's blocking flow on level graphs) and logs the amount pushed, the flow value, the residual
 * capacities, the path and the source side of the cut. It is done when no augmenting path is left; the source side of
 * the cut at that point is a minimum cut.
 *
 * @param g - The graph.
 * @param source - The source.
 * @param sink - The sink.
 * @param options - Capacities (default the weights) and `method` (default `'dinic'`).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a negative capacity or source equal to sink, and when stepping meets an
 *   unbounded flow.
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, maxFlowAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = digraph([[0, 1, 3], [0, 2, 2], [1, 2, 1], [1, 3, 2], [2, 3, 3]])
 * const run = runAlgorithm(maxFlowAlgorithm(g, 0, 3, { method: 'edmonds-karp' }), { maxSteps: 20 })
 * [run.state.value, run.state.augmentations] // => [5, 3]
 * ```
 */
export function maxFlowAlgorithm(
  g: Graph,
  source: NodeId,
  sink: NodeId,
  options: MaxFlowAlgorithmOptions = {},
): Algorithm<MaxFlowState, FlowPath> {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, sink)
  if (s === t) throw new MdmlError('value', 'maxFlow: source and sink are the same node')
  const method = options.method ?? 'dinic'
  const net = residualNetwork(g, options.capacity)
  const view = (residual: Float64Array): ResidualNetwork => ({ ...net, residual })
  const find = (st: Mutable<MaxFlowState>): number[] =>
    method === 'dinic' ? blockingPath(view(st.residual), s, t, st) : shortestAugmenting(view(st.residual), s, t)
  return {
    name: `graph/max-flow-${method}`,
    config: { nodes: g.n, edges: g.edges.length, source: s, sink: t, method },
    init: () => {
      const st: Mutable<MaxFlowState> = {
        residual: Float64Array.from(net.residual),
        value: 0,
        augmentations: 0,
        pending: [],
        last: [],
        pushed: 0,
        level: new Int32Array(g.n).fill(-1),
        next: new Int32Array(g.n),
        leveled: false,
        phases: 0,
      }
      st.pending = find(st)
      return st
    },
    step: (state, ctx) => {
      if (state.pending.length === 0) return state
      const st: Mutable<MaxFlowState> = { ...state }
      const path = state.pending
      const r = view(st.residual)
      const pushed = augment(r, path)
      st.value += pushed
      st.augmentations += 1
      st.last = path
      st.pushed = pushed
      st.pending = find(st)
      if (ctx.wants('flow/augment')) ctx.log('flow/augment', pushed)
      if (ctx.wants('flow/value')) ctx.log('flow/value', st.value)
      if (ctx.wants('flow/phase')) ctx.log('flow/phase', st.phases)
      if (ctx.wants('flow/residual')) ctx.log('flow/residual', Float64Array.from(st.residual))
      if (ctx.wants('flow/path')) ctx.log('flow/path', Int32Array.from(path))
      if (ctx.wants('flow/cut')) ctx.log('flow/cut', sourceSide(r, s))
      return st
    },
    done: (st) => (st.pending.length === 0 ? 'converged' : undefined),
    serialise: (st) => ({
      residual: floatsToJson(st.residual),
      value: st.value,
      augmentations: st.augmentations,
      pending: [...st.pending],
      last: [...st.last],
      pushed: st.pushed,
      level: stateToJson(st.level),
      next: stateToJson(st.next),
      leveled: st.leveled,
      phases: st.phases,
    }),
    deserialise: (j) => {
      const o = j as { readonly [key: string]: Json }
      return {
        residual: floatsFromJson(o['residual'] as Json),
        value: o['value'] as number,
        augmentations: o['augmentations'] as number,
        pending: [...(o['pending'] as unknown as readonly number[])],
        last: [...(o['last'] as unknown as readonly number[])],
        pushed: o['pushed'] as number,
        level: stateFromJson(o['level'] as Json) as Int32Array,
        next: stateFromJson(o['next'] as Json) as Int32Array,
        leveled: o['leveled'] as boolean,
        phases: o['phases'] as number,
      }
    },
  }
}

/** What builds a maximum-flow algorithm: a graph, a source, a sink and the capacities. */
export type MaxFlowBuilder = (
  g: Graph,
  source: NodeId,
  sink: NodeId,
  options?: FlowOptions,
) => Algorithm<MaxFlowState, FlowPath>

/** What a maximum-flow method can do, for validity rules and selectors. */
export interface MaxFlowProperties {
  /** Whether the number of augmentations is bounded by the graph alone, not by the capacities. */
  readonly strongly_polynomial: boolean
  /** Whether augmentations are grouped in phases of a level graph. */
  readonly phases: boolean
}

/**
 * The maximum-flow methods as one slot: `dinic` (blocking flows on level graphs) and `edmonds-karp` (the shortest
 * augmenting path by breadth-first search).
 *
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, maxFlowMethods } from 'mdml-engine/graph'
 * maxFlowMethods.make({ kind: 'edmonds-karp' })(digraph([[0, 1, 1]]), 0, 1).name // => 'graph/max-flow-edmonds-karp'
 * ```
 */
export const maxFlowMethods: OptionTable<MaxFlowBuilder, MaxFlowProperties> = optionTable(
  {
    dinic: knobOption<MaxFlowBuilder, MaxFlowProperties>(
      'blocking flows on breadth-first level graphs, with a current-arc pointer per node',
      { strongly_polynomial: true, phases: true },
      dict({}),
      () => (g, source, sink, options) => maxFlowAlgorithm(g, source, sink, { ...options, method: 'dinic' }),
    ),
    'edmonds-karp': knobOption<MaxFlowBuilder, MaxFlowProperties>(
      'augment along the fewest-edge path of the residual network each time',
      { strongly_polynomial: true, phases: false },
      dict({}),
      () => (g, source, sink, options) => maxFlowAlgorithm(g, source, sink, { ...options, method: 'edmonds-karp' }),
    ),
  },
  {
    default: 'dinic',
    label: 'max-flow method',
    description: 'How augmenting paths are found: blocking flows on level graphs, or the shortest path each time.',
  },
)

/** Options of {@link minCostFlowAlgorithm}. */
export interface MinCostFlowAlgorithmOptions extends FlowOptions {
  /** The cost of an edge, in edge order (default 1). */
  readonly cost?: ArrayLike<number>
}

/** The state of {@link minCostFlowAlgorithm}. */
export interface MinCostFlowState {
  /** The residual capacity of every arc of the network with the virtual source and sink. */
  readonly residual: Float64Array
  /** The node potentials (Johnson), for reduced costs. */
  readonly potential: Float64Array
  /** The supply routed so far. */
  readonly routed: number
  /** The cost of the paths pushed so far (the saturated negative edges come on top: see `cost`). */
  readonly pathCost: number
  /** The cost of the flow so far, saturated negative-cost edges included. */
  readonly cost: number
  /** The flow on each edge so far, in edge order. */
  readonly flow: Float64Array
  /** The arcs of the last augmenting path, from the virtual source. */
  readonly last: readonly number[]
  /** The amount pushed at the last step. */
  readonly pushed: number
}

/**
 * Minimum-cost flow as a steppable algorithm: negative-cost edges are saturated first (in `init`), then each step
 * routes supply from a virtual source to a virtual sink along one shortest path of reduced costs (successive shortest
 * paths with Johnson potentials). Done when all the supply is routed; an infeasible demand throws while stepping.
 *
 * @param g - A directed graph.
 * @param demand - The demand of each node (negative for supply); the demands sum to zero.
 * @param options - Capacities (default the weights) and the `cost` per edge (default 1).
 * @returns The algorithm.
 * @throws MdmlError with code `value` for an undirected graph, unbalanced demands, an unbounded negative-cost edge
 *   or, while stepping, infeasible demands.
 * @concept network-flows
 * @example
 * ```ts
 * import { digraph, minCostFlowAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = digraph([[0, 1], [0, 2], [1, 3], [2, 3]])
 * const alg = minCostFlowAlgorithm(g, [-4, 0, 0, 4], { capacity: [4, 2, 3, 5], cost: [2, 1, 1, 3] })
 * const run = runAlgorithm(alg, { maxSteps: 20 })
 * [run.state.cost, Array.from(run.state.flow)] // => [13, [3, 1, 3, 1]]
 * ```
 */
export function minCostFlowAlgorithm(
  g: Graph,
  demand: ArrayLike<number>,
  options: MinCostFlowAlgorithmOptions = {},
): Algorithm<MinCostFlowState, FlowPath> {
  if (!g.directed) throw new MdmlError('value', 'minCostFlow: the graph must be directed')
  if (demand.length !== g.n) throw new MdmlError('value', 'minCostFlow: give one demand per node')
  const total = kahanSum(demand)
  if (Math.abs(total) > 1e-9) throw new MdmlError('value', 'minCostFlow: the demands do not sum to zero')
  const m = g.edges.length
  const cost = Float64Array.from({ length: m }, (_, e) => options.cost?.[e] ?? 1)
  // Arcs 0 .. 2m - 1 as in the residual network, then 2m + 2v (virtual source to v) and 2m + 2v + 1 back, and
  // 2m + 2n + 2v (v to virtual sink) and its reverse.
  const S = g.n
  const T = g.n + 1
  const arcCount = 2 * m + 4 * g.n
  const to = new Int32Array(arcCount)
  const residual0 = new Float64Array(arcCount)
  const arcCost = new Float64Array(arcCount)
  const lists: number[][] = Array.from({ length: g.n + 2 }, () => [])
  const imbalance = Float64Array.from({ length: g.n }, (_, v) => -(demand[v] ?? 0))
  let base = 0
  const saturated = new Float64Array(m)
  const addArc = (a: number, from: number, head: number, cap: number, c: number): void => {
    to[a] = head
    to[a ^ 1] = from
    residual0[a] = cap
    residual0[a ^ 1] = 0
    arcCost[a] = c
    arcCost[a ^ 1] = -c
    lists[from]?.push(a)
    lists[head]?.push(a ^ 1)
  }
  for (const [e, edge] of g.edges.entries()) {
    const cap = options.capacity === undefined ? edge.weight : (options.capacity[e] ?? 0)
    if (!(cap >= 0)) throw new MdmlError('value', `minCostFlow: edge ${e} has a negative or missing capacity`)
    const c = cost[e] ?? 0
    if (c < 0) {
      // Saturate: the reverse arc then has positive cost, and the endpoints' balances change.
      if (cap === Number.POSITIVE_INFINITY) throw new MdmlError('value', 'minCostFlow: unbounded negative cost')
      addArc(2 * e + 1, edge.target, edge.source, cap, -c)
      saturated[e] = cap
      base += c * cap
      imbalance[edge.source] = (imbalance[edge.source] ?? 0) - cap
      imbalance[edge.target] = (imbalance[edge.target] ?? 0) + cap
    } else addArc(2 * e, edge.source, edge.target, cap, c)
  }
  let required = 0
  for (let v = 0; v < g.n; v++) {
    const b = imbalance[v] ?? 0
    if (b > 0) {
      addArc(2 * m + 2 * v, S, v, b, 0)
      required += b
    } else if (b < 0) addArc(2 * m + 2 * g.n + 2 * v, v, T, -b, 0)
  }
  const nodes = g.n + 2
  /** Net flow per edge: arcs along the edge carry it forwards; a saturated negative edge loses what its reversed arc carries. */
  const flowOf = (residual: Float64Array): Float64Array =>
    Float64Array.from({ length: m }, (_, e) =>
      (cost[e] ?? 0) < 0 ? (saturated[e] ?? 0) - (residual[2 * e] ?? 0) : (residual[2 * e + 1] ?? 0),
    )
  return {
    name: 'graph/min-cost-flow',
    config: { nodes: g.n, edges: m, required },
    init: () => ({
      residual: Float64Array.from(residual0),
      potential: new Float64Array(nodes),
      routed: 0,
      pathCost: 0,
      cost: base,
      flow: flowOf(residual0),
      last: [],
      pushed: 0,
    }),
    step: (s, ctx) => {
      if (s.routed >= required - 1e-12) return s
      const residual = Float64Array.from(s.residual)
      const potential = Float64Array.from(s.potential)
      const dist = new Float64Array(nodes).fill(Number.POSITIVE_INFINITY)
      const via = new Int32Array(nodes).fill(-1)
      const queue = new IndexedQueue(nodes)
      dist[S] = 0
      queue.push(S, 0)
      for (let u = queue.pop(); u !== undefined; u = queue.pop()) {
        for (const a of lists[u] ?? []) {
          if ((residual[a] ?? 0) <= EPS) continue
          const v = to[a] ?? 0
          const reduced = (arcCost[a] ?? 0) + (potential[u] ?? 0) - (potential[v] ?? 0)
          const alt = (dist[u] ?? 0) + Math.max(0, reduced)
          if (alt < (dist[v] ?? Number.POSITIVE_INFINITY) - 1e-15) {
            dist[v] = alt
            via[v] = a
            queue.push(v, alt)
          }
        }
      }
      if (dist[T] === Number.POSITIVE_INFINITY) throw new MdmlError('value', 'minCostFlow: the demands are infeasible')
      for (let v = 0; v < nodes; v++)
        if (dist[v] !== Number.POSITIVE_INFINITY) potential[v] = (potential[v] ?? 0) + (dist[v] ?? 0)
      let push = required - s.routed
      for (let v = T; v !== S; v = to[(via[v] ?? 0) ^ 1] ?? S) push = Math.min(push, residual[via[v] ?? 0] ?? 0)
      let pathCost = s.pathCost
      const path: number[] = []
      for (let v = T; v !== S; v = to[(via[v] ?? 0) ^ 1] ?? S) {
        const a = via[v] ?? 0
        residual[a] = (residual[a] ?? 0) - push
        residual[a ^ 1] = (residual[a ^ 1] ?? 0) + push
        pathCost += push * (arcCost[a] ?? 0)
        path.push(a)
      }
      path.reverse()
      const routed = s.routed + push
      const next: MinCostFlowState = {
        residual,
        potential,
        routed,
        pathCost,
        cost: base + pathCost,
        flow: flowOf(residual),
        last: path,
        pushed: push,
      }
      if (ctx.wants('flow/augment')) ctx.log('flow/augment', push)
      if (ctx.wants('flow/value')) ctx.log('flow/value', routed)
      if (ctx.wants('flow/cost')) ctx.log('flow/cost', next.cost)
      if (ctx.wants('flow/residual')) ctx.log('flow/residual', Float64Array.from(residual))
      if (ctx.wants('flow/path')) ctx.log('flow/path', Int32Array.from(path))
      if (ctx.wants('flow/potential')) ctx.log('flow/potential', Float64Array.from(potential))
      return next
    },
    done: (s) => (s.routed >= required - 1e-12 ? 'converged' : undefined),
    serialise: (s) => ({
      residual: floatsToJson(s.residual),
      potential: floatsToJson(s.potential),
      routed: s.routed,
      pathCost: s.pathCost,
      cost: s.cost,
      flow: floatsToJson(s.flow),
      last: [...s.last],
      pushed: s.pushed,
    }),
    deserialise: (j) => {
      const o = j as { readonly [key: string]: Json }
      return {
        residual: floatsFromJson(o['residual'] as Json),
        potential: floatsFromJson(o['potential'] as Json),
        routed: o['routed'] as number,
        pathCost: o['pathCost'] as number,
        cost: o['cost'] as number,
        flow: floatsFromJson(o['flow'] as Json),
        last: [...(o['last'] as unknown as readonly number[])],
        pushed: o['pushed'] as number,
      }
    },
  }
}
