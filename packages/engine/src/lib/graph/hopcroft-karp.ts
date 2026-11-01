/**
 * Hopcroft-Karp maximum bipartite matching as a steppable algorithm: each step is one phase, which layers the left nodes
 * by alternating-path distance from the free ones and then augments along a maximal set of vertex-disjoint shortest
 * augmenting paths. The state serialises and resumes and logs the paths of {@link matchingContract};
 * `bipartiteMatching` in `flows.ts` is this run to the end.
 */

import { type Json, MdmlError } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { floatsFromJson, floatsToJson } from './_run.ts'
import { type Graph, type NodeId, nodeIndex } from './graph.ts'
import { bipartiteColouring } from './traversal.ts'

/** The paths the matching algorithm logs, as a type. */
export type MatchingPath = 'match/phase' | 'match/size' | 'match/augmenting' | 'match/layers' | 'match/mate'

/**
 * What Hopcroft-Karp logs, by path: the phase, the matching size after it, the augmenting paths found in it, the layer
 * of each left node in it (Infinity for unreached) and every node's partner.
 */
export const matchingContract: Contract<unknown, MatchingPath> = defineContract({
  'match/phase': scalar({ description: 'the phase just run, counting from 1' }),
  'match/size': scalar({ description: 'the size of the matching after the phase' }),
  'match/augmenting': scalar({ description: 'vertex-disjoint augmenting paths applied in the phase' }),
  'match/layers': tensor({
    role: 'state',
    shape: ['nodes'],
    description: 'the alternating-path distance of each left node from a free one in this phase (Infinity: unreached)',
  }),
  'match/mate': tensor({ role: 'state', shape: ['nodes'], description: "each node's partner (-1 when unmatched)" }),
})

/** The state of {@link hopcroftKarpAlgorithm}. */
export interface MatchingState {
  /** The left nodes, in order. */
  readonly left: readonly number[]
  /** Each node's partner (-1 when unmatched). */
  readonly mate: Int32Array
  /** The matching size. */
  readonly size: number
  /** Phases run. */
  readonly phase: number
  /** The layer of each left node for the next phase (Infinity for unreached, and for right nodes). */
  readonly layer: Float64Array
  /** The layer at which the first free right node is reached (Infinity: no augmenting path is left). */
  readonly found: number
  /** Augmenting paths applied in the last phase. */
  readonly augmented: number
}

const INF = Number.POSITIVE_INFINITY

/**
 * Hopcroft-Karp as a steppable algorithm (one phase per step).
 *
 * @param g - A bipartite graph (edge directions are ignored).
 * @param options - The left nodes (default: colour 0 of {@link bipartiteColouring}).
 * @returns The algorithm; it is done when no augmenting path is left.
 * @throws MdmlError with code `value` when the graph is not bipartite or an edge joins two nodes of one side.
 * @concept matching
 * @example
 * ```ts
 * import { graph, hopcroftKarpAlgorithm } from 'mdml-engine/graph'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const run = runAlgorithm(hopcroftKarpAlgorithm(graph([[0, 3], [0, 4], [1, 3], [2, 5]])), { maxSteps: 10 })
 * [run.state.size, run.state.phase] // => [3, 2]
 * ```
 */
export function hopcroftKarpAlgorithm(
  g: Graph,
  options: { readonly left?: readonly NodeId[] } = {},
): Algorithm<MatchingState, MatchingPath> {
  const isLeft = new Uint8Array(g.n)
  if (options.left === undefined) {
    const colour = bipartiteColouring(g)
    if (colour === null) throw new MdmlError('value', 'bipartiteMatching: the graph is not bipartite')
    for (let v = 0; v < g.n; v++) isLeft[v] = colour[v] === 0 ? 1 : 0
  } else for (const v of options.left) isLeft[nodeIndex(g, v)] = 1
  const adj: number[][] = Array.from({ length: g.n }, () => [])
  for (const e of g.edges) {
    if (isLeft[e.source] === isLeft[e.target]) {
      throw new MdmlError('value', `bipartiteMatching: edge (${e.source}, ${e.target}) lies within one side`)
    }
    const [l, r] = isLeft[e.source] === 1 ? [e.source, e.target] : [e.target, e.source]
    adj[l]?.push(r)
  }
  for (const list of adj) list.sort((a, b) => a - b)
  const left = Array.from({ length: g.n }, (_, v) => v).filter((v) => isLeft[v] === 1)

  /** Layer the left nodes by alternating-path distance from the free ones; returns the layer of the first free right node. */
  const layers = (mate: Int32Array, dist: Float64Array): number => {
    const queue: number[] = []
    for (const u of left) {
      if (mate[u] === -1) {
        dist[u] = 0
        queue.push(u)
      } else dist[u] = INF
    }
    let found = INF
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head] ?? 0
      if ((dist[u] ?? INF) >= found) continue
      for (const v of adj[u] ?? []) {
        const w = mate[v] ?? -1
        if (w === -1) found = Math.min(found, (dist[u] ?? 0) + 1)
        else if (dist[w] === INF) {
          dist[w] = (dist[u] ?? 0) + 1
          queue.push(w)
        }
      }
    }
    return found
  }

  return {
    name: 'graph/hopcroft-karp',
    config: { nodes: g.n, left: left.length },
    init: () => {
      const mate = new Int32Array(g.n).fill(-1)
      const layer = new Float64Array(g.n).fill(INF)
      const found = layers(mate, layer)
      return { left, mate, size: 0, phase: 0, layer, found, augmented: 0 }
    },
    step: (s, ctx) => {
      if (s.found === INF) return s
      const mate = Int32Array.from(s.mate)
      const dist = Float64Array.from(s.layer)
      const shown = Float64Array.from(s.layer)
      const found = s.found
      let size = s.size
      let augmented = 0
      // Vertex-disjoint shortest augmenting paths by iterative depth-first search.
      const next = new Int32Array(g.n)
      for (const root of left) {
        if (mate[root] !== -1) continue
        const stack = [root]
        const taken: number[] = []
        while (stack.length > 0) {
          const u = stack[stack.length - 1] ?? 0
          const list = adj[u] ?? []
          let moved = false
          while ((next[u] ?? 0) < list.length) {
            const v = list[next[u] ?? 0] ?? 0
            next[u] = (next[u] ?? 0) + 1
            const w = mate[v] ?? -1
            if (w === -1 ? (dist[u] ?? 0) + 1 === found : dist[w] === (dist[u] ?? 0) + 1) {
              taken.push(v)
              if (w === -1) {
                // Augment along the stack.
                for (let i = 0; i < stack.length; i++) {
                  const a = stack[i] ?? 0
                  const b = taken[i] ?? 0
                  mate[a] = b
                  mate[b] = a
                }
                size++
                augmented++
                stack.length = 0
              } else stack.push(w)
              moved = true
              break
            }
          }
          if (moved) continue
          dist[u] = INF
          stack.pop()
          taken.pop()
        }
      }
      const layer = new Float64Array(g.n).fill(INF)
      const nextFound = layers(mate, layer)
      const phase = s.phase + 1
      if (ctx.wants('match/phase')) ctx.log('match/phase', phase)
      if (ctx.wants('match/size')) ctx.log('match/size', size)
      if (ctx.wants('match/augmenting')) ctx.log('match/augmenting', augmented)
      if (ctx.wants('match/layers')) ctx.log('match/layers', shown)
      if (ctx.wants('match/mate')) ctx.log('match/mate', Int32Array.from(mate))
      return { left, mate, size, phase, layer, found: nextFound, augmented }
    },
    done: (s) => (s.found === INF ? 'converged' : undefined),
    serialise: (s) => ({
      left: [...s.left],
      mate: stateToJson(s.mate),
      size: s.size,
      phase: s.phase,
      layer: floatsToJson(s.layer),
      found: Number.isFinite(s.found) ? s.found : 'Infinity',
      augmented: s.augmented,
    }),
    deserialise: (j) => {
      const o = j as { readonly [key: string]: Json }
      return {
        left: [...(o['left'] as unknown as readonly number[])],
        mate: stateFromJson(o['mate'] as Json) as Int32Array,
        size: o['size'] as number,
        phase: o['phase'] as number,
        layer: floatsFromJson(o['layer'] as Json),
        found: Number(o['found']),
        augmented: o['augmented'] as number,
      }
    },
  }
}
