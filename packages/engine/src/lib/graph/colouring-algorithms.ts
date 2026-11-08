/**
 * The combinatorial set and colouring algorithms as steppable algorithms: greedy colouring (one node coloured per step,
 * under an ordering strategy of the table {@link colouringStrategies}), maximal independent sets (one node chosen per
 * step, by the rules of {@link independentSetRules}) and the Bron-Kerbosch search for maximal cliques (one recursive
 * call per step, with its R, P and X sets, under the pivot rules of {@link pivotRules}). Each state serialises and
 * resumes and logs the paths of {@link colouringContract}, {@link independentSetContract} or {@link cliqueContract};
 * `greedyColouring`, `maximalIndependentSet` and `maximalCliques` in `colouring.ts` are these run to the end.
 */

import { OmaiError } from '../base/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { dict, knobOption, type Option, type OptionTable, optionTable } from '../space/index.ts'
import { type Algorithm, type Contract, defineContract, scalar, tensor } from '../trace/index.ts'
import { neighbourSets } from './_neighbours.ts'
import { codec } from './_run.ts'
import { connectedComponents } from './components.ts'
import { degree, type Graph, type NodeId, nodeIndex } from './graph.ts'
import { bfs, dfs } from './traversal.ts'

function smallestFree(used: ReadonlySet<number>): number {
  let c = 0
  while (used.has(c)) c++
  return c
}

/** The remaining node of least degree among the remaining (ties to the smallest index). */
function minDegreeNode(nb: readonly Set<number>[], remaining: ReadonlySet<number>): number {
  let best = -1
  let bestDegree = Number.POSITIVE_INFINITY
  for (const v of [...remaining].sort((p, q) => p - q)) {
    let d = 0
    for (const u of nb[v] ?? []) if (remaining.has(u)) d++
    if (d < bestDegree) {
      best = v
      bestDegree = d
    }
  }
  return best
}

/** Repeatedly take the node of least degree in what remains (ties to the smallest index), dropping its neighbours. */
function minDegreeIndependent(nb: readonly Set<number>[], within: ReadonlySet<number>): number[] {
  const remaining = new Set(within)
  const out: number[] = []
  while (remaining.size > 0) {
    const best = minDegreeNode(nb, remaining)
    out.push(best)
    remaining.delete(best)
    for (const u of nb[best] ?? []) remaining.delete(u)
  }
  return out
}

function smallestLastOrder(nb: readonly Set<number>[]): number[] {
  const n = nb.length
  const deg = Int32Array.from(nb, (s) => s.size)
  const removed = new Uint8Array(n)
  const order: number[] = []
  for (let i = 0; i < n; i++) {
    let best = -1
    for (let v = 0; v < n; v++) if (removed[v] === 0 && (best < 0 || (deg[v] ?? 0) < (deg[best] ?? 0))) best = v
    removed[best] = 1
    order.push(best)
    for (const u of nb[best] ?? []) if (removed[u] === 0) deg[u] = (deg[u] ?? 0) - 1
  }
  return order.reverse()
}

/** The graph itself when undirected; otherwise the same nodes with undirected neighbour sets, for traversal. */
function undirectedView(g: Graph, nb: readonly Set<number>[]): Graph {
  if (!g.directed) return g
  const indptr = new Int32Array(g.n + 1)
  const indices: number[] = []
  for (let u = 0; u < g.n; u++) {
    indices.push(...[...(nb[u] ?? [])].sort((p, q) => p - q))
    indptr[u + 1] = indices.length
  }
  const adj = {
    indptr,
    indices: Int32Array.from(indices),
    weights: new Float64Array(indices.length).fill(1),
    edges: new Int32Array(indices.length),
  }
  return { ...g, directed: false, out: adj, in: adj }
}

// ---------------------------------------------------------------------------------------------------------------------
// Greedy colouring

/** The order in which {@link greedyColouring} colours the nodes (NetworkX greedy_color strategies). */
export type ColouringStrategy =
  | 'largest-first'
  | 'smallest-last'
  | 'random-sequential'
  | 'independent-set'
  | 'connected-sequential-bfs'
  | 'connected-sequential-dfs'
  | 'dsatur'

/** The paths greedy colouring logs, as a type. */
export type ColouringPath = 'colour/node' | 'colour/assign' | 'colour/count' | 'colour/colours' | 'colour/saturation'

/**
 * What greedy colouring logs, by path: the node coloured at this step and the colour it got, the number of colours
 * used so far, the colour of every node (-1 while uncoloured) and the saturation of every node (the number of distinct
 * colours among its neighbours).
 */
export const colouringContract: Contract<unknown, ColouringPath> = defineContract({
  'colour/node': scalar({ description: 'the node coloured at this step' }),
  'colour/assign': scalar({ description: 'the colour it was given (the smallest not used by a neighbour)' }),
  'colour/count': scalar({ description: 'the number of colours used so far' }),
  'colour/colours': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the colour of each node (-1: uncoloured)',
  }),
  'colour/saturation': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the number of distinct colours among the neighbours of each node',
  }),
})

/** What a colouring strategy is given. */
export interface OrderingInput {
  readonly g: Graph
  /** Neighbour sets ignoring direction and self-loops. */
  readonly nb: readonly Set<number>[]
  /** The key for the random strategies. */
  readonly key?: Key | undefined
}

/** Builds the order a strategy colours in; `null` for a strategy that chooses the next node as it goes (DSATUR). */
export type OrderBuilder = (input: OrderingInput) => number[] | null

/** What a colouring strategy can do, for validity rules and selectors. */
export interface StrategyProperties {
  /** Whether the next node depends on the colours so far (DSATUR) rather than on a fixed order. */
  readonly dynamic: boolean
  /** Whether it needs a key. */
  readonly needsKey: boolean
}

function strategy(
  description: string,
  properties: StrategyProperties,
  build: OrderBuilder,
): Option<OrderBuilder, StrategyProperties> {
  return knobOption<OrderBuilder, StrategyProperties>(description, properties, dict({}), () => build)
}

/**
 * The colouring strategies as one slot, in NetworkX's names: `largest-first` (by degree), `smallest-last` (repeatedly
 * remove a node of least degree, colour in reverse), `random-sequential` (shuffled, with a key), `independent-set`
 * (peel off maximal independent sets), `connected-sequential-bfs` and `-dfs` (a traversal of each component), and
 * `dsatur` (always the node with most distinctly coloured neighbours, Brelaz 1979). Ties go to the smaller index.
 *
 * @concept graph-colouring
 * @example
 * ```ts
 * import { colouringStrategies } from 'omai-engine/graph'
 * colouringStrategies.where((p) => p.dynamic) // => ['dsatur']
 * ```
 */
export const colouringStrategies: OptionTable<OrderBuilder, StrategyProperties> = optionTable(
  {
    'largest-first': strategy('by degree, largest first', { dynamic: false, needsKey: false }, ({ g }) => {
      const deg = degree(g)
      return Array.from({ length: g.n }, (_, v) => v).sort((p, q) => (deg[q] ?? 0) - (deg[p] ?? 0) || p - q)
    }),
    'smallest-last': strategy(
      'repeatedly remove a node of least degree, colour in reverse',
      { dynamic: false, needsKey: false },
      ({ nb }) => smallestLastOrder(nb),
    ),
    'random-sequential': strategy(
      'in a random order given by a key',
      { dynamic: false, needsKey: true },
      ({ g, key }) => {
        if (key === undefined) throw new OmaiError('value', "greedyColouring: 'random-sequential' needs a key")
        return Array.from(permutation(key, g.n).data)
      },
    ),
    'independent-set': strategy(
      'peel off maximal independent sets and colour them in turn',
      { dynamic: false, needsKey: false },
      ({ g, nb }) => {
        const order: number[] = []
        const remaining = new Set(Array.from({ length: g.n }, (_, v) => v))
        while (remaining.size > 0) {
          for (const v of minDegreeIndependent(nb, remaining)) {
            remaining.delete(v)
            order.push(v)
          }
        }
        return order
      },
    ),
    'connected-sequential-bfs': strategy(
      'a breadth-first order of each component',
      { dynamic: false, needsKey: false },
      ({ g }) => connectedComponents(g).components.flatMap((c) => bfs(g, c[0] ?? 0, { direction: 'both' }).order),
    ),
    'connected-sequential-dfs': strategy(
      'a depth-first order of each component',
      { dynamic: false, needsKey: false },
      ({ g, nb }) => connectedComponents(g).components.flatMap((c) => dfs(undirectedView(g, nb), c[0] ?? 0).preorder),
    ),
    dsatur: strategy(
      'always the node whose neighbours show the most distinct colours',
      { dynamic: true, needsKey: false },
      () => null,
    ),
  },
  {
    default: 'largest-first',
    label: 'colouring strategy',
    description: 'The order in which greedy colouring visits the nodes.',
  },
)

/** The state of {@link greedyColouringAlgorithm}. */
export interface ColouringState {
  /** The colour of each node (-1 uncoloured). */
  readonly colours: Int32Array
  /** The order the nodes are coloured in (empty for DSATUR, which chooses as it goes). */
  readonly order: readonly number[]
  /** Nodes coloured so far. */
  readonly position: number
  /** DSATUR: the distinct neighbour colours of each node, ascending. */
  readonly saturation: readonly (readonly number[])[]
  /** The node coloured at the last step (-1 before the first). */
  readonly node: number
  /** The colour it received. */
  readonly colour: number
}

/**
 * Greedy colouring as a steppable algorithm: each step colours one node with the smallest colour not used by a
 * neighbour, the node chosen by the strategy (a fixed order, or for DSATUR the most saturated node). Edge directions
 * are ignored.
 *
 * @param g - The graph.
 * @param options - The `strategy` (default `'largest-first'`, see {@link colouringStrategies}) and a `key` for
 *   `'random-sequential'`.
 * @returns The algorithm; it is done when every node has a colour.
 * @throws OmaiError with code `value` when `'random-sequential'` has no key.
 * @concept graph-colouring
 * @example
 * ```ts
 * import { cycleGraph, greedyColouringAlgorithm } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(greedyColouringAlgorithm(cycleGraph(5)), { maxSteps: 10 })
 * [run.steps, Array.from(run.state.colours)] // => [5, [0, 1, 0, 1, 2]]
 * ```
 */
export function greedyColouringAlgorithm(
  g: Graph,
  options: { readonly strategy?: ColouringStrategy; readonly key?: Key } = {},
): Algorithm<ColouringState, ColouringPath> {
  const name = options.strategy ?? 'largest-first'
  const nb = neighbourSets(g)
  const deg = degree(g)
  const builder = colouringStrategies.make({ kind: name })
  const order = builder({ g, nb, key: options.key })
  const used = (colours: Int32Array, u: number): Set<number> => {
    const s = new Set<number>()
    for (const v of nb[u] ?? []) if ((colours[v] ?? -1) >= 0) s.add(colours[v] ?? 0)
    return s
  }
  return {
    name: 'graph/greedy-colouring',
    config: { nodes: g.n, strategy: name },
    init: () => ({
      colours: new Int32Array(g.n).fill(-1),
      order: order ?? [],
      position: 0,
      saturation: order === null ? Array.from({ length: g.n }, () => []) : [],
      node: -1,
      colour: -1,
    }),
    step: (s, ctx) => {
      if (s.position >= g.n) return s
      let u = -1
      if (order === null) {
        for (let v = 0; v < g.n; v++) {
          if ((s.colours[v] ?? -1) >= 0) continue
          const sv = s.saturation[v]?.length ?? 0
          const sb = s.saturation[u]?.length ?? 0
          if (u < 0 || sv > sb || (sv === sb && (deg[v] ?? 0) > (deg[u] ?? 0))) u = v
        }
      } else u = order[s.position] ?? 0
      const colour = smallestFree(used(s.colours, u))
      s.colours[u] = colour
      const saturation =
        order === null
          ? s.saturation.map((list, v) =>
              nb[u]?.has(v) === true && !list.includes(colour) ? [...list, colour].sort((p, q) => p - q) : list,
            )
          : s.saturation
      const next: ColouringState = { ...s, position: s.position + 1, saturation, node: u, colour }
      if (ctx.wants('colour/node')) ctx.log('colour/node', u)
      if (ctx.wants('colour/assign')) ctx.log('colour/assign', colour)
      if (ctx.wants('colour/count'))
        ctx.log(
          'colour/count',
          s.colours.reduce((m, c) => Math.max(m, c + 1), 0),
        )
      if (ctx.wants('colour/colours')) ctx.log('colour/colours', Int32Array.from(s.colours))
      if (ctx.wants('colour/saturation'))
        ctx.log(
          'colour/saturation',
          Int32Array.from({ length: g.n }, (_, v) => used(s.colours, v).size),
        )
      return next
    },
    done: (s) => (s.position >= g.n ? 'converged' : undefined),
    ...codec<ColouringState>(),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Maximal independent sets

/** The paths maximal-independent-set search logs, as a type. */
export type IndependentSetPath = 'mis/node' | 'mis/size' | 'mis/free' | 'mis/chosen'

/**
 * What the independent-set search logs, by path: the node chosen at this step, the size of the set so far, and the
 * nodes still free to be chosen and those chosen.
 */
export const independentSetContract: Contract<unknown, IndependentSetPath> = defineContract({
  'mis/node': scalar({ description: 'the node added to the set at this step' }),
  'mis/size': scalar({ description: 'the size of the set so far' }),
  'mis/free': tensor({
    role: 'state',
    shape: ['free'],
    description: 'the nodes still free: not chosen and not next to one',
  }),
  'mis/chosen': tensor({ role: 'state', shape: ['chosen'], description: 'the nodes chosen so far, ascending' }),
})

/** The state of {@link maximalIndependentSetAlgorithm}. */
export interface IndependentSetState {
  /** The nodes chosen, ascending. */
  readonly chosen: readonly number[]
  /** The nodes still free, ascending. */
  readonly free: readonly number[]
  /** The random order to consider nodes in (empty for the minimum-degree rule). */
  readonly order: readonly number[]
  /** The position in `order`. */
  readonly cursor: number
  /** The node added at the last step (-1 before the first). */
  readonly node: number
}

/** What builds an independent-set search: the graph, the forced nodes and the key of the random rule. */
export type IndependentSetBuilder = (
  g: Graph,
  options?: { readonly nodes?: readonly NodeId[]; readonly key?: Key },
) => Algorithm<IndependentSetState, IndependentSetPath>

/** What an independent-set rule can do, for validity rules and selectors. */
export interface IndependentSetProperties {
  /** Whether it needs a key. */
  readonly needsKey: boolean
}

/**
 * Maximal independent sets as a steppable algorithm: each step adds one free node to the set and frees none of its
 * neighbours. With no key the node of least remaining degree goes first; with a key nodes are considered in a random
 * order. Nodes given in `nodes` are in the set from the start.
 *
 * @param g - The graph (directions ignored).
 * @param options - Nodes to include, and a key for a random set.
 * @returns The algorithm; it is done when no node is free.
 * @throws OmaiError with code `value` when the forced nodes are not independent.
 * @concept independent-sets
 * @example
 * ```ts
 * import { maximalIndependentSetAlgorithm, starGraph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(maximalIndependentSetAlgorithm(starGraph(3)), { maxSteps: 10 })
 * [run.steps, run.state.chosen] // => [3, [1, 2, 3]]
 * ```
 */
export function maximalIndependentSetAlgorithm(
  g: Graph,
  options: { readonly nodes?: readonly NodeId[]; readonly key?: Key } = {},
): Algorithm<IndependentSetState, IndependentSetPath> {
  const nb = neighbourSets(g)
  const forced = (options.nodes ?? []).map((v) => nodeIndex(g, v))
  const chosen = new Set<number>()
  const blocked = new Set<number>()
  for (const v of forced) {
    if (blocked.has(v)) throw new OmaiError('value', 'maximalIndependentSet: the given nodes are not independent')
    chosen.add(v)
    blocked.add(v)
    for (const u of nb[v] ?? []) blocked.add(u)
  }
  const free = Array.from({ length: g.n }, (_, v) => v).filter((v) => !blocked.has(v))
  const order = options.key === undefined ? [] : Array.from(permutation(options.key, g.n).data)
  return {
    name: 'graph/maximal-independent-set',
    config: { nodes: g.n, rule: options.key === undefined ? 'min-degree' : 'random', forced: forced.length },
    init: () => ({ chosen: [...chosen].sort((p, q) => p - q), free, order, cursor: 0, node: -1 }),
    step: (s, ctx) => {
      if (s.free.length === 0) return s
      const open = new Set(s.free)
      let v: number
      let cursor = s.cursor
      if (options.key === undefined) v = minDegreeNode(nb, open)
      else {
        while (!open.has(s.order[cursor] ?? -1)) cursor++
        v = s.order[cursor] ?? 0
        cursor++
      }
      open.delete(v)
      for (const u of nb[v] ?? []) open.delete(u)
      const next: IndependentSetState = {
        chosen: [...s.chosen, v].sort((p, q) => p - q),
        free: s.free.filter((u) => open.has(u)),
        order: s.order,
        cursor,
        node: v,
      }
      if (ctx.wants('mis/node')) ctx.log('mis/node', v)
      if (ctx.wants('mis/size')) ctx.log('mis/size', next.chosen.length)
      if (ctx.wants('mis/free')) ctx.log('mis/free', Int32Array.from(next.free))
      if (ctx.wants('mis/chosen')) ctx.log('mis/chosen', Int32Array.from(next.chosen))
      return next
    },
    done: (s) => (s.free.length === 0 ? 'converged' : undefined),
    ...codec<IndependentSetState>(),
  }
}

/**
 * The independent-set rules as one slot: `min-degree` (the free node of least remaining degree first) and `random`
 * (free nodes in the order of a random permutation; needs a key).
 *
 * @concept independent-sets
 * @example
 * ```ts
 * import { independentSetRules, starGraph } from 'omai-engine/graph'
 * independentSetRules.make({ kind: 'min-degree' })(starGraph(3)).name // => 'graph/maximal-independent-set'
 * ```
 */
export const independentSetRules: OptionTable<IndependentSetBuilder, IndependentSetProperties> = optionTable(
  {
    'min-degree': knobOption<IndependentSetBuilder, IndependentSetProperties>(
      'the free node of least remaining degree first',
      { needsKey: false },
      dict({}),
      () => (g, options) =>
        maximalIndependentSetAlgorithm(g, { ...(options?.nodes === undefined ? {} : { nodes: options.nodes }) }),
    ),
    random: knobOption<IndependentSetBuilder, IndependentSetProperties>(
      'free nodes in the order of a random permutation',
      { needsKey: true },
      dict({}),
      () => (g, options) => {
        if (options?.key === undefined) throw new OmaiError('value', "independentSetRules: 'random' needs a key")
        return maximalIndependentSetAlgorithm(g, options)
      },
    ),
  },
  {
    default: 'min-degree',
    label: 'independent-set rule',
    description: 'Which free node joins the independent set next.',
  },
)

// ---------------------------------------------------------------------------------------------------------------------
// Bron-Kerbosch

/** The paths the clique search logs, as a type. */
export type CliquePath = 'clique/R' | 'clique/P' | 'clique/X' | 'clique/pivot' | 'clique/found'

/**
 * What the Bron-Kerbosch search logs, by path: the sets R (the clique being grown), P (candidates) and X (excluded) of
 * the recursive call made at this step, its pivot (-1 when it ends at once) and the number of maximal cliques found.
 */
export const cliqueContract: Contract<unknown, CliquePath> = defineContract({
  'clique/R': tensor({ role: 'state', shape: ['r'], description: 'the clique being grown, in the order nodes joined' }),
  'clique/P': tensor({
    role: 'state',
    shape: ['p'],
    description: 'the nodes adjacent to all of R that can still extend it',
  }),
  'clique/X': tensor({
    role: 'state',
    shape: ['x'],
    description: 'the nodes adjacent to all of R already tried: they forbid R',
  }),
  'clique/pivot': scalar({
    description: 'the pivot whose neighbours need not be branched on (-1: no call was pushed)',
  }),
  'clique/found': scalar({ description: 'maximal cliques found so far' }),
})

/** One recursive call of Bron-Kerbosch waiting on the stack. */
export interface CliqueFrame {
  readonly r: readonly number[]
  readonly p: readonly number[]
  readonly x: readonly number[]
  /** The candidates `P \ N(pivot)` still to branch on, ascending. */
  readonly todo: readonly number[]
}

/** The state of {@link bronKerboschAlgorithm}. The list of cliques grows in place; copy a state to keep it. */
export interface CliqueState {
  /** The calls waiting, innermost last. */
  readonly stack: readonly CliqueFrame[]
  /** The maximal cliques found, each ascending, in the order found. */
  readonly cliques: readonly (readonly number[])[]
  /** The call made at the last step. */
  readonly last: {
    readonly r: readonly number[]
    readonly p: readonly number[]
    readonly x: readonly number[]
    readonly pivot: number
  }
}

/** Chooses the pivot of a call: from P and X, given the neighbour sets; -1 for none. */
export type PivotRule = (nb: readonly Set<number>[], p: readonly number[], x: readonly number[]) => number

/** What a pivot rule can do, for validity rules and selectors. */
export interface PivotProperties {
  /** Whether it keeps the worst-case O(3^(n/3)) bound. */
  readonly worstCaseOptimal: boolean
}

const mostNeighboursIn = (nb: readonly Set<number>[], candidates: readonly number[], p: readonly number[]): number => {
  const inP = new Set(p)
  let pivot = -1
  let most = -1
  for (const u of candidates) {
    let c = 0
    for (const v of nb[u] ?? []) if (inP.has(v)) c++
    if (c > most) {
      most = c
      pivot = u
    }
  }
  return pivot
}

/**
 * The pivot rules of Bron-Kerbosch as one slot: `tomita` (the node of P and X with most neighbours in P, so the fewest
 * branches remain), `p-only` (the same over P alone) and `none` (no pivot: branch on every candidate, the original
 * algorithm).
 *
 * @concept cliques
 * @example
 * ```ts
 * import { pivotRules } from 'omai-engine/graph'
 * pivotRules.where((p) => p.worstCaseOptimal) // => ['tomita']
 * ```
 */
export const pivotRules: OptionTable<PivotRule, PivotProperties> = optionTable(
  {
    tomita: knobOption<PivotRule, PivotProperties>(
      'the node of P and X with most neighbours in P (Tomita)',
      { worstCaseOptimal: true },
      dict({}),
      () => (nb, p, x) => mostNeighboursIn(nb, [...p, ...x], p),
    ),
    'p-only': knobOption<PivotRule, PivotProperties>(
      'the node of P with most neighbours in P',
      { worstCaseOptimal: false },
      dict({}),
      () => (nb, p) => mostNeighboursIn(nb, p, p),
    ),
    none: knobOption<PivotRule, PivotProperties>(
      'no pivot: branch on every candidate (Bron and Kerbosch 1973)',
      { worstCaseOptimal: false },
      dict({}),
      () => () => -1,
    ),
  },
  {
    default: 'tomita',
    label: 'pivot rule',
    description: 'Which node prunes the branching of Bron-Kerbosch.',
  },
)

/**
 * Bron-Kerbosch as a steppable algorithm: each step makes one recursive call, branching on the next candidate of the
 * innermost call (R grows by that node; P and X shrink to its neighbours). A call with P and X both empty reports a
 * maximal clique. Done when every call is exhausted.
 *
 * @param g - An undirected graph.
 * @param options - The pivot rule (default `'tomita'`).
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a directed graph.
 * @concept cliques
 * @example
 * ```ts
 * import { bronKerboschAlgorithm, graph } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(bronKerboschAlgorithm(graph([[0, 1], [1, 2], [2, 0], [2, 3]])), { maxSteps: 50 })
 * run.state.cliques.length // => 2
 * ```
 */
export function bronKerboschAlgorithm(
  g: Graph,
  options: { readonly pivot?: 'tomita' | 'p-only' | 'none' } = {},
): Algorithm<CliqueState, CliquePath> {
  if (g.directed) throw new OmaiError('value', 'maximalCliques: the graph must be undirected')
  const rule = pivotRules.make({ kind: options.pivot ?? 'tomita' })
  const nb = neighbourSets(g)
  /** The call (R, P, X): a clique when P and X are empty, else a frame (when P is not empty). */
  const call = (
    r: readonly number[],
    p: readonly number[],
    x: readonly number[],
    stack: CliqueFrame[],
    cliques: (readonly number[])[],
  ): number => {
    if (p.length === 0) {
      if (x.length === 0) cliques.push([...r].sort((a, b) => a - b))
      return -1
    }
    const pivot = rule(nb, p, x)
    const todo = p.filter((v) => !(nb[pivot]?.has(v) ?? false)).sort((a, b) => a - b)
    stack.push({ r, p, x, todo })
    return pivot
  }
  /** Drop the calls with nothing left to branch on. */
  const settle = (stack: CliqueFrame[]): void => {
    while (stack.length > 0 && (stack[stack.length - 1]?.todo.length ?? 0) === 0) stack.pop()
  }
  return {
    name: 'graph/bron-kerbosch',
    config: { nodes: g.n, pivot: options.pivot ?? 'tomita' },
    init: () => {
      const stack: CliqueFrame[] = []
      const cliques: (readonly number[])[] = []
      const p = Array.from({ length: g.n }, (_, v) => v)
      const pivot = call([], p, [], stack, cliques)
      settle(stack)
      return { stack, cliques, last: { r: [], p, x: [], pivot } }
    },
    step: (s, ctx) => {
      const stack = [...s.stack]
      const top = stack[stack.length - 1]
      if (top === undefined) return s
      const [v, ...rest] = top.todo
      const node = v ?? 0
      const nv = nb[node] ?? new Set<number>()
      stack[stack.length - 1] = { r: top.r, p: top.p.filter((u) => u !== node), x: [...top.x, node], todo: rest }
      const cliques = s.cliques as (readonly number[])[]
      const r = [...top.r, node]
      const p = top.p.filter((u) => nv.has(u))
      const x = top.x.filter((u) => nv.has(u))
      const pivot = call(r, p, x, stack, cliques)
      settle(stack)
      if (ctx.wants('clique/R')) ctx.log('clique/R', Int32Array.from(r))
      if (ctx.wants('clique/P')) ctx.log('clique/P', Int32Array.from(p))
      if (ctx.wants('clique/X')) ctx.log('clique/X', Int32Array.from(x))
      if (ctx.wants('clique/pivot')) ctx.log('clique/pivot', pivot)
      if (ctx.wants('clique/found')) ctx.log('clique/found', cliques.length)
      return { stack, cliques, last: { r, p, x, pivot } }
    },
    done: (s) => (s.stack.length === 0 ? 'converged' : undefined),
    ...codec<CliqueState>(),
  }
}
