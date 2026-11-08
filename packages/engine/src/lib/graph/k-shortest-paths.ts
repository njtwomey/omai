/**
 * Yen's k shortest loopless paths as a steppable algorithm: each step computes the shortest deviation (spur path) from
 * one node of the latest accepted path, with the edges earlier paths take from the same root and the root's nodes
 * removed; once every node of that path has been tried, the cheapest candidate is accepted. The state serialises and
 * resumes and logs the paths of {@link yenContract}; `kShortestPaths` is this run to the end.
 */

import { OmaiError } from '../base/index.ts'
import { type Algorithm, type Contract, defineContract, runAlgorithm, scalar, tensor } from '../trace/index.ts'
import { codec, weightFunction } from './_run.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { type PathOptions, reconstructPath, type WeightedPath } from './paths.ts'
import { dijkstraAlgorithm } from './shortest-paths.ts'

/** The paths Yen's algorithm logs, as a type. */
export type YenPath = 'yen/spur' | 'yen/candidate' | 'yen/candidates' | 'yen/found' | 'yen/cost'

/**
 * What Yen's algorithm logs, by path: the spur node tried at this step, the candidate path it produced (empty when the
 * deviation does not exist or repeats a known path), the size of the candidate pool, the number of paths accepted and
 * the cost of the latest accepted path.
 */
export const yenContract: Contract<unknown, YenPath> = defineContract({
  'yen/spur': scalar({ description: 'the node of the latest accepted path from which a deviation was searched' }),
  'yen/candidate': tensor({
    role: 'state',
    shape: ['path'],
    description:
      'the candidate path the spur gave: the root followed by the shortest way on (empty when none or known)',
  }),
  'yen/candidates': scalar({ description: 'candidate paths in the pool' }),
  'yen/found': scalar({ description: 'paths accepted so far' }),
  'yen/cost': scalar({ description: 'the cost of the latest accepted path' }),
})

/** The state of {@link yenAlgorithm}. */
export interface YenState {
  /** The accepted paths, cheapest first. */
  readonly found: readonly WeightedPath[]
  /** The candidates not yet accepted. */
  readonly candidates: readonly WeightedPath[]
  /** The position in the latest accepted path of the next spur node. */
  readonly spur: number
  /** True once no candidate is left. */
  readonly exhausted: boolean
  /** The spur node of the last step (-1 before the first) and the candidate it made (empty for none). */
  readonly lastSpur: number
  readonly lastCandidate: readonly number[]
}

/** Order candidates by cost, then by node sequence. */
function byCost(x: WeightedPath, y: WeightedPath): number {
  if (x.cost !== y.cost) return x.cost - y.cost
  for (let i = 0; i < Math.min(x.path.length, y.path.length); i++) {
    if (x.path[i] !== y.path[i]) return (x.path[i] ?? 0) - (y.path[i] ?? 0)
  }
  return x.path.length - y.path.length
}

/**
 * Yen's algorithm as a steppable algorithm. Each step tries one spur node of the latest accepted path; the step that
 * tries its last node also accepts the cheapest candidate. Done when `k` paths are accepted or no candidate is left.
 *
 * @param g - The graph (non-negative weights).
 * @param source - The source.
 * @param target - The target.
 * @param k - How many paths.
 * @param options - `weighted` (default true).
 * @returns The algorithm.
 * @concept shortest-paths
 * @reference J. Y. Yen, Finding the k shortest loopless paths in a network, Management Science 17 (1971).
 * @example
 * ```ts
 * import { cycleGraph, yenAlgorithm } from 'omai-engine/graph'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const run = runAlgorithm(yenAlgorithm(cycleGraph(5), 0, 2, 2), { maxSteps: 20 })
 * run.state.found.map((p) => p.path) // => [[0, 1, 2], [0, 4, 3, 2]]
 * ```
 */
export function yenAlgorithm(
  g: Graph,
  source: NodeId,
  target: NodeId,
  k: number,
  options: PathOptions = {},
): Algorithm<YenState, YenPath> {
  const s = nodeIndex(g, source)
  const t = nodeIndex(g, target)
  const weight = weightFunction(g, options.weighted)
  const a = g.out
  const edgeBetween = (u: number, v: number): number => {
    for (let q = a.indptr[u] ?? 0; q < (a.indptr[u + 1] ?? 0); q++) if (a.indices[q] === v) return q
    return -1
  }
  const costOf = (path: readonly number[]): number => {
    let c = 0
    for (let i = 0; i + 1 < path.length; i++) c += weight(edgeBetween(path[i] ?? 0, path[i + 1] ?? 0))
    return c
  }
  const shortest = (
    from: number,
    blockedEdges?: ReadonlySet<number>,
    blockedNodes?: ReadonlySet<number>,
  ): { distance: Float64Array; predecessor: Int32Array } => {
    const alg = dijkstraAlgorithm(g, from, {
      target: t,
      weights: weight,
      ...(blockedEdges === undefined ? {} : { blockedEdges }),
      ...(blockedNodes === undefined ? {} : { blockedNodes }),
    })
    return runAlgorithm(alg, { maxSteps: g.n + 1 }).state
  }
  const key = (path: readonly number[]): string => path.join(',')
  return {
    name: 'graph/yen',
    config: { nodes: g.n, source: s, target: t, k },
    init: () => {
      const first = shortest(s)
      const p0 = reconstructPath(first.predecessor, s, t)
      if (p0 === null || k <= 0)
        return { found: [], candidates: [], spur: 0, exhausted: true, lastSpur: -1, lastCandidate: [] }
      return {
        found: [{ path: p0, cost: first.distance[t] ?? 0 }],
        candidates: [],
        spur: 0,
        exhausted: false,
        lastSpur: -1,
        lastCandidate: [],
      }
    },
    step: (st, ctx) => {
      if (st.exhausted || st.found.length >= k) return st
      const last = (st.found[st.found.length - 1] as WeightedPath).path
      const found = [...st.found]
      const candidates = [...st.candidates]
      const i = st.spur
      let lastSpur = -1
      let lastCandidate: number[] = []
      if (i + 1 < last.length) {
        const spur = last[i] ?? 0
        lastSpur = spur
        const root = last.slice(0, i + 1)
        const blockedEdges = new Set<number>()
        for (const { path } of found) {
          if (path.length > i + 1 && root.every((v, j) => path[j] === v)) {
            const q = edgeBetween(path[i] ?? 0, path[i + 1] ?? 0)
            if (q >= 0) blockedEdges.add(a.edges[q] ?? -1)
          }
        }
        const r = shortest(spur, blockedEdges, new Set(root.slice(0, -1)))
        const tail = reconstructPath(r.predecessor, spur, t)
        if (tail !== null) {
          const path = [...root.slice(0, -1), ...tail]
          const id = key(path)
          if (!candidates.some((c) => key(c.path) === id) && !found.some((f) => key(f.path) === id)) {
            candidates.push({ path, cost: costOf(path) })
            lastCandidate = path
          }
        }
      }
      let spurNext = i + 1
      let exhausted = false
      if (spurNext + 1 >= last.length) {
        // Every spur of the latest path has been tried: accept the cheapest candidate.
        if (candidates.length === 0) exhausted = true
        else {
          candidates.sort(byCost)
          found.push(candidates.shift() as WeightedPath)
          spurNext = 0
        }
      }
      if (ctx.wants('yen/spur')) ctx.log('yen/spur', lastSpur)
      if (ctx.wants('yen/candidate')) ctx.log('yen/candidate', Int32Array.from(lastCandidate))
      if (ctx.wants('yen/candidates')) ctx.log('yen/candidates', candidates.length)
      if (ctx.wants('yen/found')) ctx.log('yen/found', found.length)
      if (ctx.wants('yen/cost')) ctx.log('yen/cost', (found[found.length - 1] as WeightedPath).cost)
      return { found, candidates, spur: spurNext, exhausted, lastSpur, lastCandidate }
    },
    done: (st) => (st.exhausted || st.found.length >= k ? 'converged' : undefined),
    ...codec<YenState>(),
  }
}

/**
 * The k shortest loopless paths from a source to a target by Yen's algorithm (NetworkX shortest_simple_paths, which
 * lists them lazily in the same order): each new path deviates from a previous one at a spur node, with the edges
 * that previous paths take from the same root and the root's nodes removed. Ties are broken by the node sequence.
 *
 * @param g - The graph (non-negative weights).
 * @param source - The source.
 * @param target - The target.
 * @param k - How many paths.
 * @param options - `weighted` (default true).
 * @returns Up to k paths with their costs, in non-decreasing cost.
 * @throws OmaiError with code `value` for a non-integer k.
 * @concept shortest-paths
 * @reference J. Y. Yen, Finding the k shortest loopless paths in a network, Management Science 17 (1971).
 * @example
 * ```ts
 * import { cycleGraph, kShortestPaths } from 'omai-engine/graph'
 * kShortestPaths(cycleGraph(5), 0, 2, 2).map((p) => p.path) // => [[0, 1, 2], [0, 4, 3, 2]]
 * ```
 */
export function kShortestPaths(
  g: Graph,
  source: NodeId,
  target: NodeId,
  k: number,
  options: PathOptions = {},
): WeightedPath[] {
  if (!Number.isFinite(k)) throw new OmaiError('value', 'kShortestPaths: k must be finite')
  const { found } = runAlgorithm(yenAlgorithm(g, source, target, k, options), {
    maxSteps: Number.MAX_SAFE_INTEGER,
  }).state
  return [...found]
}
