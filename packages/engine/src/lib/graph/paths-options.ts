/**
 * The slots of the shortest-path algorithms as option tables: the heuristic that guides A* (none, the fewest hops to the
 * target, or distances to landmarks), and the single-source method itself (Dijkstra, Bellman-Ford, A*), each with the
 * knobs of its own. Each table is one `union` space for controls and grids, and builds the heuristic or the algorithm
 * from a configuration.
 */

import { MdmlError } from '../base/index.ts'
import {
  choice,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { type Graph, type NodeId, nodeIndex, reverse } from './graph.ts'
import { dijkstra } from './paths.ts'
import {
  aStarAlgorithm,
  bellmanFordAlgorithm,
  dijkstraAlgorithm,
  type ShortestPathOptions,
  type ShortestPathPath,
  type ShortestPathState,
} from './shortest-paths.ts'

/** Builds a heuristic for the target of a search: an estimate of the distance left from each node. */
export type HeuristicBuilder = (g: Graph, target: number, weighted: boolean) => (v: number) => number

/** What a heuristic guarantees, for validity rules and selectors. */
export interface HeuristicProperties {
  /** Whether it never overestimates the distance left (so A* finds a shortest path). */
  readonly admissible: boolean
  /** Whether it needs precomputed distances (landmarks), as opposed to being free. */
  readonly precomputed: boolean
}

/** The knobs of a configuration read with the option's defaults filling those not given. */
type Knobs = (name: string) => number | string

function knobsOf(space: ReturnType<typeof dict>, config: Readonly<Record<string, SpaceValue>>): Knobs {
  const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
  return (name) => full[name] as number | string
}

function heuristicOption(
  description: string,
  properties: HeuristicProperties,
  space: ReturnType<typeof dict>,
  build: (k: Knobs) => HeuristicBuilder,
): Option<HeuristicBuilder, HeuristicProperties> {
  return { description, properties, space, make: (config) => build(knobsOf(space, config)) }
}

/** Hops from every node to the target along edges (a breadth-first search on the reversed graph), -1 when none. */
function hopsToTarget(g: Graph, target: number): Int32Array {
  const hops = new Int32Array(g.n).fill(-1)
  const into = g.directed ? g.in : g.out
  hops[target] = 0
  const queue = [target]
  for (let i = 0; i < queue.length; i++) {
    const u = queue[i] ?? 0
    for (let k = into.indptr[u] ?? 0; k < (into.indptr[u + 1] ?? 0); k++) {
      const v = into.indices[k] ?? 0
      if (hops[v] === -1) {
        hops[v] = (hops[u] ?? 0) + 1
        queue.push(v)
      }
    }
  }
  return hops
}

/**
 * The heuristics of A* as one slot: `zero` (none: A* is Dijkstra), `hops` (the cheapest edge times the fewest hops to
 * the target, free and admissible) and `landmarks` (ALT: the triangle inequality against distances to and from a few
 * landmark nodes chosen far apart, admissible and consistent, paid for with two Dijkstra runs per landmark).
 *
 * @concept shortest-paths
 * @example
 * ```ts
 * import { gridGraph, pathHeuristics } from 'mdml-engine/graph'
 * const h = pathHeuristics.make({ kind: 'hops' })(gridGraph(3, 3), 8, true)
 * [h(0), h(8)] // => [4, 0]
 * ```
 */
export const pathHeuristics: OptionTable<HeuristicBuilder, HeuristicProperties> = optionTable(
  {
    zero: heuristicOption(
      'no estimate: A* settles nodes by distance alone, as Dijkstra does',
      { admissible: true, precomputed: false },
      dict({}),
      () => () => () => 0,
    ),
    hops: heuristicOption(
      'the cheapest edge weight times the fewest hops to the target',
      { admissible: true, precomputed: false },
      dict({}),
      () => (g, target, weighted) => {
        const hops = hopsToTarget(g, target)
        let cheapest = Number.POSITIVE_INFINITY
        for (const e of g.edges) cheapest = Math.min(cheapest, weighted ? e.weight : 1)
        const unit = Number.isFinite(cheapest) ? Math.max(0, cheapest) : 0
        return (v) => ((hops[v] ?? -1) < 0 ? 0 : unit * (hops[v] ?? 0))
      },
    ),
    landmarks: heuristicOption(
      'ALT: the largest gap the triangle inequality leaves against distances to and from landmark nodes',
      { admissible: true, precomputed: true },
      dict({ count: int(1, 16, { default: 4, label: 'landmarks' }) }),
      (k) => (g, target, weighted) => {
        const count = Math.max(1, Math.min(g.n, Math.round(k('count') as number)))
        const back = reverse(g)
        const options = { weighted }
        // Farthest-first: the first landmark is the node farthest from the target, each next the node farthest from all so far.
        const from: Float64Array[] = []
        const to: Float64Array[] = []
        const nearest = new Float64Array(g.n).fill(Number.POSITIVE_INFINITY)
        let pick = target
        const first = dijkstra(g, target, options).distance
        let best = -1
        for (let v = 0; v < g.n; v++) {
          if (Number.isFinite(first[v]) && (first[v] ?? 0) > best) {
            best = first[v] ?? 0
            pick = v
          }
        }
        for (let i = 0; i < count; i++) {
          const forward = dijkstra(g, pick, options).distance
          from.push(forward)
          to.push(g.directed ? dijkstra(back, pick, options).distance : forward)
          let far = -1
          let next = pick
          for (let v = 0; v < g.n; v++) {
            nearest[v] = Math.min(nearest[v] ?? 0, forward[v] ?? Number.POSITIVE_INFINITY)
            if (Number.isFinite(nearest[v]) && (nearest[v] ?? 0) > far) {
              far = nearest[v] ?? 0
              next = v
            }
          }
          pick = next
        }
        return (v) => {
          let h = 0
          for (let i = 0; i < from.length; i++) {
            const lt = from[i]?.[target] ?? Number.POSITIVE_INFINITY
            const lv = from[i]?.[v] ?? Number.POSITIVE_INFINITY
            const vl = to[i]?.[v] ?? Number.POSITIVE_INFINITY
            const tl = to[i]?.[target] ?? Number.POSITIVE_INFINITY
            if (Number.isFinite(lt) && Number.isFinite(lv)) h = Math.max(h, lt - lv)
            if (Number.isFinite(vl) && Number.isFinite(tl)) h = Math.max(h, vl - tl)
          }
          return h
        }
      },
    ),
  },
  {
    default: 'hops',
    label: 'heuristic',
    description: 'The estimate of the distance left that guides A*, from none to distances to landmarks.',
  },
)

/** What a single-source method is built from: a graph, a source, an optional target and the weights flag. */
export type SingleSourceBuilder = (
  g: Graph,
  source: NodeId,
  options?: ShortestPathOptions & { readonly target?: NodeId },
) => Algorithm<ShortestPathState, ShortestPathPath>

/** What a single-source method can do, for validity rules and selectors. */
export interface SingleSourceProperties {
  /** Whether negative edge weights are allowed. */
  readonly negativeWeights: boolean
  /** Whether it needs a target (the search is guided to it). */
  readonly needsTarget: boolean
  /** Whether a step settles a node (Dijkstra, A*) or relaxes every edge (Bellman-Ford). */
  readonly step: 'node' | 'round'
}

function method(
  description: string,
  properties: SingleSourceProperties,
  space: ReturnType<typeof dict>,
  build: (k: Knobs) => SingleSourceBuilder,
): Option<SingleSourceBuilder, SingleSourceProperties> {
  return { description, properties, space, make: (config) => build(knobsOf(space, config)) }
}

/**
 * The single-source shortest-path methods as one slot: `dijkstra` (non-negative weights, a node settled per step),
 * `bellman-ford` (negative weights allowed, a round of relaxations per step) and `a-star` (Dijkstra guided to a
 * target by a heuristic of {@link pathHeuristics}, with a weight above 1 for a greedier search).
 *
 * @concept shortest-paths
 * @example
 * ```ts
 * import { graph, shortestPathMethods } from 'mdml-engine/graph'
 * shortestPathMethods.make({ kind: 'dijkstra' })(graph([[0, 1]]), 0).name // => 'graph/dijkstra'
 * ```
 */
export const shortestPathMethods: OptionTable<SingleSourceBuilder, SingleSourceProperties> = optionTable(
  {
    dijkstra: method(
      'settle the nearest unsettled node each step; non-negative weights',
      { negativeWeights: false, needsTarget: false, step: 'node' },
      dict({}),
      () => (g, source, options) => dijkstraAlgorithm(g, source, options),
    ),
    'bellman-ford': method(
      'relax every edge each round; negative weights allowed, negative cycles detected',
      { negativeWeights: true, needsTarget: false, step: 'round' },
      dict({}),
      () => (g, source, options) => bellmanFordAlgorithm(g, source, options),
    ),
    'a-star': method(
      'Dijkstra guided to a target by a heuristic estimate of the distance left',
      { negativeWeights: false, needsTarget: true, step: 'node' },
      dict({
        heuristic: choice(pathHeuristics.names, { default: 'hops', label: 'heuristic' }),
        landmarks: int(1, 16, { default: 4, label: 'landmarks (for the landmark heuristic)' }),
        weight: real(1, 10, { default: 1, label: 'heuristic weight (1 is A*)' }),
      }),
      (k) => (g, source, options) => {
        if (options?.target === undefined) throw new MdmlError('value', 'shortestPathMethods: a-star needs a target')
        const target = nodeIndex(g, options.target)
        const weighted = options.weighted !== false
        const heuristic = pathHeuristics.make({ kind: k('heuristic') as string, count: k('landmarks') as number })(
          g,
          target,
          weighted,
        )
        return aStarAlgorithm(g, source, target, { weighted, heuristic, weight: k('weight') as number })
      },
    ),
  },
  {
    default: 'dijkstra',
    label: 'shortest-path method',
    description: 'How single-source shortest paths are found: by settling nodes, by rounds of relaxation, or guided.',
  },
)
