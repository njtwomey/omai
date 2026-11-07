/**
 * The depth-first connectivity algorithms as steppable algorithms: strongly connected components by Tarjan's lowlink
 * search or Kosaraju's two passes, and the Hopcroft-Tarjan search for articulation points, bridges and biconnected
 * components. Each step is one action of the search (a node entered, an edge examined, a node finished), and each
 * state serialises and resumes and logs the paths of {@link dfsContract}; the slot for strong components is the option
 * table {@link componentMethods}. The functions of `components.ts` are these run to the end.
 */

import { MdmlError } from '../base/index.ts'
import { dict, knobOption, type OptionTable, optionTable } from '../space/index.ts'
import { type Algorithm, type Contract, defineContract, scalar, tensor } from '../trace/index.ts'
import { codec } from './_run.ts'
import type { Graph } from './graph.ts'

/** The paths the depth-first connectivity algorithms log, as a type. */
export type DfsPath =
  | 'dfs/node'
  | 'dfs/phase'
  | 'dfs/discover'
  | 'dfs/low'
  | 'dfs/component'
  | 'dfs/stack'
  | 'dfs/count'
  | 'dfs/cut'
  | 'dfs/bridges'

/**
 * What the depth-first connectivity algorithms log, by path: the node the step acted on, the pass (Kosaraju), the
 * discovery times and lowlinks so far (-1 for unseen), the component of every node (-1 while unassigned), the stack,
 * the components or blocks found so far, and for biconnectivity the articulation flags and the number of bridges.
 */
export const dfsContract: Contract<unknown, DfsPath> = defineContract({
  'dfs/node': scalar({ description: 'the node the step entered, finished or expanded' }),
  'dfs/phase': scalar({ description: 'the pass of Kosaraju: 1 finishing order, 2 components on the reversed graph' }),
  'dfs/discover': tensor({ role: 'state', shape: ['nodes'], description: 'discovery time of each node (-1: unseen)' }),
  'dfs/low': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the lowlink of each node: the earliest discovery it reaches',
  }),
  'dfs/component': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the strong component of each node (-1 while unassigned)',
  }),
  'dfs/stack': tensor({
    role: 'state',
    shape: ['stack'],
    description: 'the nodes on the algorithm stack, bottom first',
  }),
  'dfs/count': scalar({ description: 'strong components (or biconnected blocks) found so far' }),
  'dfs/cut': tensor({
    role: 'state',
    shape: ['nodes'],
    description: '1 for the nodes found to be articulation points',
  }),
  'dfs/bridges': scalar({ description: 'bridges found so far' }),
})

/** The state of the strong-component algorithms. Steps update the arrays in place; keep a copy to look at later. */
export interface ComponentSearchState {
  /** Discovery time of each node (-1 unseen); Kosaraju's first pass. */
  readonly discovery: Int32Array
  /** Tarjan: the lowlink of each node. */
  readonly low: Int32Array
  /** The strong component of each node (-1 while unassigned), numbered in the order they were completed. */
  readonly component: Int32Array
  /** Depth-first call frames: a node and the position of its next adjacency entry. */
  readonly frames: [number, number][]
  /** Tarjan: the component stack; Kosaraju second pass: nodes waiting to be expanded. */
  readonly stack: number[]
  /** Tarjan: 1 for nodes on the component stack. */
  readonly onStack: Uint8Array
  /** Kosaraju: nodes in the order they finished in the first pass. */
  readonly finished: number[]
  /** Kosaraju: the pass (1 or 2); Tarjan stays in 1. */
  readonly phase: 1 | 2
  /** The next discovery time. */
  readonly time: number
  /** Components completed so far. */
  readonly count: number
  /** The next candidate root (Kosaraju second pass: the position in `finished`, counting down). */
  readonly cursor: number
  /** The node the last step acted on (-1 before the first). */
  readonly node: number
}

function componentInit(g: Graph, kosaraju: boolean): ComponentSearchState {
  return {
    discovery: new Int32Array(g.n).fill(-1),
    low: new Int32Array(g.n),
    component: new Int32Array(g.n).fill(-1),
    frames: [],
    stack: [],
    onStack: new Uint8Array(g.n),
    finished: [],
    phase: kosaraju && g.n === 0 ? 2 : 1,
    time: 0,
    count: 0,
    cursor: kosaraju && g.n === 0 ? -1 : 0,
    node: -1,
  }
}

function logComponent(
  s: ComponentSearchState,
  ctx: Parameters<Algorithm<ComponentSearchState, DfsPath>['step']>[1],
): void {
  if (ctx.wants('dfs/node')) ctx.log('dfs/node', s.node)
  if (ctx.wants('dfs/phase')) ctx.log('dfs/phase', s.phase)
  if (ctx.wants('dfs/discover')) ctx.log('dfs/discover', Int32Array.from(s.discovery))
  if (ctx.wants('dfs/low')) ctx.log('dfs/low', Int32Array.from(s.low))
  if (ctx.wants('dfs/component')) ctx.log('dfs/component', Int32Array.from(s.component))
  if (ctx.wants('dfs/stack')) ctx.log('dfs/stack', Int32Array.from(s.stack))
  if (ctx.wants('dfs/count')) ctx.log('dfs/count', s.count)
}

/** The first node at or after `from` that is not discovered. */
function nextUnseen(s: ComponentSearchState, from: number, n: number): number {
  let r = from
  while (r < n && s.discovery[r] !== -1) r++
  return r
}

/**
 * Tarjan's strongly connected components as a steppable algorithm: each step enters a node, examines one edge (a tree
 * edge enters the child, an edge to a node on the stack lowers the lowlink) or finishes a node, popping its component
 * when the lowlink equals the discovery time. Done when every node has a component.
 *
 * @param g - A graph (an undirected graph's strong components are its connected components).
 * @returns The algorithm.
 * @concept connected-components
 * @reference R. Tarjan, Depth-first search and linear graph algorithms, SIAM J. Computing 1 (1972).
 * @example
 * ```ts
 * import { digraph, tarjanAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = digraph([[0, 1], [1, 0], [1, 2], [2, 3], [3, 2]])
 * const run = runAlgorithm(tarjanAlgorithm(g), { maxSteps: 40 })
 * [run.state.count, Array.from(run.state.component)] // => [2, [1, 1, 0, 0]]
 * ```
 */
export function tarjanAlgorithm(g: Graph): Algorithm<ComponentSearchState, DfsPath> {
  const a = g.out
  const enter = (s: ComponentSearchState, v: number): void => {
    s.discovery[v] = s.time
    s.low[v] = s.time
    s.stack.push(v)
    s.onStack[v] = 1
    s.frames.push([v, a.indptr[v] ?? 0])
  }
  return {
    name: 'graph/tarjan',
    config: { nodes: g.n },
    init: () => componentInit(g, false),
    step: (s, ctx) => {
      if (s.frames.length === 0) {
        const root = nextUnseen(s, s.cursor, g.n)
        if (root >= g.n) return s
        enter(s, root)
        const next = { ...s, time: s.time + 1, cursor: root + 1, node: root }
        logComponent(next, ctx)
        return next
      }
      const frame = s.frames[s.frames.length - 1] as [number, number]
      const [v, k] = frame
      let time = s.time
      let count = s.count
      let node = v
      if (k < (a.indptr[v + 1] ?? 0)) {
        frame[1] = k + 1
        const w = a.indices[k] ?? 0
        if (s.discovery[w] === -1) {
          enter(s, w)
          time += 1
          node = w
        } else if (s.onStack[w] === 1) s.low[v] = Math.min(s.low[v] ?? 0, s.discovery[w] ?? 0)
      } else {
        s.frames.pop()
        const parent = s.frames[s.frames.length - 1]
        if (parent !== undefined) s.low[parent[0]] = Math.min(s.low[parent[0]] ?? 0, s.low[v] ?? 0)
        if (s.low[v] === s.discovery[v]) {
          for (;;) {
            const w = s.stack.pop() ?? v
            s.onStack[w] = 0
            s.component[w] = count
            if (w === v) break
          }
          count += 1
        }
      }
      const next = { ...s, time, count, node }
      logComponent(next, ctx)
      return next
    },
    done: (s) => (s.frames.length === 0 && nextUnseen(s, s.cursor, g.n) >= g.n ? 'converged' : undefined),
    ...codec<ComponentSearchState>(),
  }
}

/**
 * Kosaraju's strongly connected components as a steppable algorithm. The first pass is a depth-first search of the
 * graph that records the order nodes finish in (one action per step); the second takes the nodes in reverse finishing
 * order and, from each unassigned one, collects what reaches it on the reversed graph (one node expanded per step).
 *
 * @param g - A graph.
 * @returns The algorithm.
 * @concept connected-components
 * @example
 * ```ts
 * import { digraph, kosarajuAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const g = digraph([[0, 1], [1, 0], [1, 2], [2, 3], [3, 2]])
 * const run = runAlgorithm(kosarajuAlgorithm(g), { maxSteps: 60 })
 * [run.state.count, Array.from(run.state.component)] // => [2, [0, 0, 1, 1]]
 * ```
 */
export function kosarajuAlgorithm(g: Graph): Algorithm<ComponentSearchState, DfsPath> {
  const out = g.out
  const into = g.in
  /** Move on to the second pass once the first has seen every node. */
  const advance = (s: ComponentSearchState): ComponentSearchState =>
    s.frames.length === 0 && nextUnseen(s, s.cursor, g.n) >= g.n ? { ...s, phase: 2, cursor: s.finished.length - 1 } : s
  return {
    name: 'graph/kosaraju',
    config: { nodes: g.n },
    init: () => componentInit(g, true),
    step: (s, ctx) => {
      if (s.phase === 1) {
        if (s.frames.length === 0) {
          const root = nextUnseen(s, s.cursor, g.n)
          s.discovery[root] = s.time
          s.frames.push([root, out.indptr[root] ?? 0])
          const next = advance({ ...s, time: s.time + 1, cursor: root + 1, node: root })
          logComponent(next, ctx)
          return next
        }
        const frame = s.frames[s.frames.length - 1] as [number, number]
        const [v, k] = frame
        let time = s.time
        let node = v
        if (k < (out.indptr[v + 1] ?? 0)) {
          frame[1] = k + 1
          const w = out.indices[k] ?? 0
          if (s.discovery[w] === -1) {
            s.discovery[w] = time
            time += 1
            s.frames.push([w, out.indptr[w] ?? 0])
            node = w
          }
        } else {
          s.frames.pop()
          s.finished.push(v)
        }
        const next = advance({ ...s, time, node })
        logComponent(next, ctx)
        return next
      }
      // Second pass: expand one node of the component being collected, starting a new one when none is.
      let { count, cursor } = s
      if (s.stack.length === 0) {
        while (cursor >= 0 && s.component[s.finished[cursor] ?? 0] !== -1) cursor--
        if (cursor < 0) return s
        const root = s.finished[cursor] ?? 0
        s.component[root] = count
        s.stack.push(root)
        count += 1
      }
      const u = s.stack.pop() ?? 0
      for (let k = into.indptr[u] ?? 0; k < (into.indptr[u + 1] ?? 0); k++) {
        const w = into.indices[k] ?? 0
        if (s.component[w] === -1) {
          s.component[w] = count - 1
          s.stack.push(w)
        }
      }
      const next = { ...s, count, cursor, node: u }
      logComponent(next, ctx)
      return next
    },
    done: (s) => {
      if (s.phase === 1 || s.stack.length > 0) return undefined
      let c = s.cursor
      while (c >= 0 && s.component[s.finished[c] ?? 0] !== -1) c--
      return c < 0 ? 'converged' : undefined
    },
    ...codec<ComponentSearchState>(),
  }
}

/** What builds a strong-component algorithm. */
export type ComponentBuilder = (g: Graph) => Algorithm<ComponentSearchState, DfsPath>

/** What a strong-component method can do, for validity rules and selectors. */
export interface ComponentProperties {
  /** Graph traversals it makes: one (Tarjan) or two (Kosaraju, which also needs the reversed graph). */
  readonly passes: 1 | 2
}

/**
 * The strong-component methods as one slot: `tarjan` (one pass with lowlinks and a component stack) and `kosaraju`
 * (a finishing-order pass, then the reversed graph).
 *
 * @concept connected-components
 * @example
 * ```ts
 * import { componentMethods, digraph } from 'mdml-engine/graph'
 * componentMethods.make({ kind: 'kosaraju' })(digraph([[0, 1]])).name // => 'graph/kosaraju'
 * ```
 */
export const componentMethods: OptionTable<ComponentBuilder, ComponentProperties> = optionTable(
  {
    tarjan: knobOption<ComponentBuilder, ComponentProperties>(
      'one depth-first pass with lowlinks and a component stack',
      { passes: 1 },
      dict({}),
      () => (g) => tarjanAlgorithm(g),
    ),
    kosaraju: knobOption<ComponentBuilder, ComponentProperties>(
      'depth-first finishing order, then components on the reversed graph',
      { passes: 2 },
      dict({}),
      () => (g) => kosarajuAlgorithm(g),
    ),
  },
  {
    default: 'tarjan',
    label: 'strong-component method',
    description: 'How strongly connected components are found: by lowlinks in one pass, or by two passes.',
  },
)

/** The state of {@link biconnectivityAlgorithm}. Steps update the arrays in place; keep a copy to look at later. */
export interface BiconnectivityState {
  /** Discovery time of each node (-1 unseen). */
  readonly discovery: Int32Array
  /** The lowpoint of each node: the earliest discovery reachable by its subtree and one back edge. */
  readonly low: Int32Array
  /** 1 for articulation points found. */
  readonly cut: Uint8Array
  /** Bridges found, as `[min, max]` pairs. */
  readonly bridges: [number, number][]
  /** Biconnected blocks found, each its edges as visited. */
  readonly blocks: [number, number][][]
  /** The edge stack of the search. */
  readonly edgeStack: [number, number][]
  /** Call frames: the node, the edge it was entered by and the position of its next adjacency entry. */
  readonly frames: [number, number, number][]
  /** The next discovery time. */
  readonly time: number
  /** The next candidate root. */
  readonly cursor: number
  /** Tree children of the current root (two or more make it an articulation point). */
  readonly rootChildren: number
  /** The root of the current search tree. */
  readonly root: number
  /** The node the last step acted on. */
  readonly node: number
}

/**
 * The Hopcroft-Tarjan biconnectivity search as a steppable algorithm: each step enters a node, examines an edge or
 * finishes a node, lowering lowpoints and emitting a block (and marking an articulation point or a bridge) whenever
 * a child's lowpoint reaches its parent. Done when every node was visited.
 *
 * @param g - An undirected graph.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for a directed graph.
 * @concept biconnectivity
 * @example
 * ```ts
 * import { barbellGraph, biconnectivityAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(biconnectivityAlgorithm(barbellGraph(3, 1)), { maxSteps: 100 })
 * [run.state.blocks.length, run.state.bridges.length] // => [4, 2]
 * ```
 */
export function biconnectivityAlgorithm(g: Graph): Algorithm<BiconnectivityState, DfsPath> {
  if (g.directed) throw new MdmlError('value', 'biconnectivityAlgorithm: the graph must be undirected')
  const a = g.out
  const firstUnseen = (s: BiconnectivityState): number => {
    let r = s.cursor
    while (r < g.n && s.discovery[r] !== -1) r++
    return r
  }
  return {
    name: 'graph/biconnectivity',
    config: { nodes: g.n },
    init: () => ({
      discovery: new Int32Array(g.n).fill(-1),
      low: new Int32Array(g.n),
      cut: new Uint8Array(g.n),
      bridges: [],
      blocks: [],
      edgeStack: [],
      frames: [],
      time: 0,
      cursor: 0,
      rootChildren: 0,
      root: -1,
      node: -1,
    }),
    step: (s, ctx) => {
      let next: BiconnectivityState
      if (s.frames.length === 0) {
        const root = firstUnseen(s)
        if (root >= g.n) return s
        s.discovery[root] = s.time
        s.low[root] = s.time
        s.frames.push([root, -1, a.indptr[root] ?? 0])
        next = { ...s, time: s.time + 1, cursor: root + 1, rootChildren: 0, root, node: root }
      } else {
        next = s
        for (;;) {
          const frame = s.frames[s.frames.length - 1] as [number, number, number]
          const [v, via, k] = frame
          if (k < (a.indptr[v + 1] ?? 0)) {
            frame[2] = k + 1
            const w = a.indices[k] ?? 0
            const e = a.edges[k] ?? -1
            if (w === v || e === via) continue
            if (s.discovery[w] === -1) {
              s.edgeStack.push([v, w])
              s.discovery[w] = s.time
              s.low[w] = s.time
              s.frames.push([w, e, a.indptr[w] ?? 0])
              next = { ...s, time: s.time + 1, node: w }
            } else if ((s.discovery[w] ?? 0) < (s.discovery[v] ?? 0)) {
              s.low[v] = Math.min(s.low[v] ?? 0, s.discovery[w] ?? 0)
              s.edgeStack.push([v, w])
              next = { ...s, node: v }
            } else continue
            break
          }
          s.frames.pop()
          const parent = s.frames[s.frames.length - 1]
          let rootChildren = s.rootChildren
          if (parent === undefined) {
            if (rootChildren >= 2) s.cut[s.root] = 1
          } else {
            const p = parent[0]
            s.low[p] = Math.min(s.low[p] ?? 0, s.low[v] ?? 0)
            if (p === s.root) rootChildren += 1
            if ((s.low[v] ?? 0) >= (s.discovery[p] ?? 0)) {
              if (p !== s.root) s.cut[p] = 1
              const block: [number, number][] = []
              for (;;) {
                const top = s.edgeStack.pop()
                if (top === undefined) break
                block.push(top)
                if (top[0] === p && top[1] === v) break
              }
              s.blocks.push(block)
            }
            if ((s.low[v] ?? 0) > (s.discovery[p] ?? 0)) s.bridges.push([Math.min(p, v), Math.max(p, v)])
          }
          next = { ...s, rootChildren, node: v }
          break
        }
      }
      if (ctx.wants('dfs/node')) ctx.log('dfs/node', next.node)
      if (ctx.wants('dfs/discover')) ctx.log('dfs/discover', Int32Array.from(next.discovery))
      if (ctx.wants('dfs/low')) ctx.log('dfs/low', Int32Array.from(next.low))
      if (ctx.wants('dfs/cut')) ctx.log('dfs/cut', Uint8Array.from(next.cut))
      if (ctx.wants('dfs/count')) ctx.log('dfs/count', next.blocks.length)
      if (ctx.wants('dfs/bridges')) ctx.log('dfs/bridges', next.bridges.length)
      if (ctx.wants('dfs/stack')) ctx.log('dfs/stack', Int32Array.from(next.frames.map((f) => f[0])))
      return next
    },
    done: (s) => (s.frames.length === 0 && firstUnseen(s) >= g.n ? 'converged' : undefined),
    ...codec<BiconnectivityState>(),
  }
}
