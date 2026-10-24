import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { dijkstra, type Graph, reverse } from '../graph/index.ts'
import {
  aStarSearch,
  beamSearch,
  bidirectionalSearch,
  breadthFirstSearch,
  graphProblem,
  greedyBestFirstSearch,
  idaStarSearch,
  isAdmissible,
  isConsistent,
  iterativeDeepeningSearch,
  recursiveBestFirstSearch,
  replay,
  uniformCostSearch,
  weightedAStarSearch,
} from './index.ts'

const weighted = fc.oneof(
  arbitraryGraph({ minNodes: 3, maxNodes: 9, maxWeight: 9, density: 0.4 }),
  arbitraryGraph({ minNodes: 3, maxNodes: 9, maxWeight: 9, density: 0.4, directed: true }),
)

/** The true cost to the target from every node, by Dijkstra on the reversed graph. */
function costToGo(g: Graph, target: number): Float64Array {
  return dijkstra(g.directed ? reverse(g) : g, target).distance
}

const case_ = weighted.chain((g) =>
  fc.record({
    g: fc.constant(g),
    s: fc.nat(g.n - 1),
    t: fc.nat(g.n - 1),
    scale: fc.double({ min: 0, max: 1, noNaN: true }),
  }),
)

describe('path searches on random graphs', () => {
  it('A*, IDA*, RBFS and bidirectional search with admissible heuristics match uniform-cost search', () => {
    fc.assert(
      fc.property(case_, ({ g, s, t, scale }) => {
        const hStar = costToGo(g, t)
        const h = (v: number): number => {
          const d = hStar[v] ?? Number.POSITIVE_INFINITY
          return Number.isFinite(d) ? scale * d : 0
        }
        const problem = graphProblem(g, s, t, { heuristic: h })
        const ucs = uniformCostSearch(problem)
        const optimum = hStar[s] ?? Number.POSITIVE_INFINITY
        expect(ucs.cost).toBe(optimum)
        expect(ucs.status).toBe(Number.isFinite(optimum) ? 'solved' : 'failure')
        expect(aStarSearch(problem).cost).toBe(optimum)
        expect(idaStarSearch(problem).cost).toBe(optimum)
        // RBFS is a tree search: without a goal it cycles until its budget runs out.
        const rbfs = recursiveBestFirstSearch(problem, { maxExpansions: Number.isFinite(optimum) ? 100_000 : 300 })
        expect(rbfs.cost).toBe(optimum)
        if (!Number.isFinite(optimum)) expect(['budget', 'failure']).toContain(rbfs.status)
        const back = bidirectionalSearch({
          ...problem,
          goal: t,
          predecessors: (v) => {
            const out: { state: number; action: number; cost: number }[] = []
            for (let k = g.in.indptr[v] ?? 0; k < (g.in.indptr[v + 1] ?? 0); k++) {
              const u = g.in.indices[k] ?? 0
              out.push({ state: u, action: v, cost: g.in.weights[k] ?? 1 })
            }
            return out
          },
        })
        expect(back.cost).toBe(optimum)
        if (ucs.status === 'solved') {
          expect(replay(problem, ucs.actions)).toEqual({ state: t, cost: optimum, goal: true })
          expect(isAdmissible(problem).holds).toBe(true)
          expect(isConsistent(problem).holds).toBe(true)
        }
      }),
      { numRuns: 120 },
    )
  })

  it('weighted A* stays within its weight of the optimum, and greedy and beam searches return valid paths', () => {
    fc.assert(
      fc.property(case_, fc.double({ min: 1, max: 4, noNaN: true }), ({ g, s, t }, w) => {
        const hStar = costToGo(g, t)
        const h = (v: number): number => (Number.isFinite(hStar[v] ?? 0) ? (hStar[v] ?? 0) : 0)
        const problem = graphProblem(g, s, t, { heuristic: h })
        const optimum = hStar[s] ?? Number.POSITIVE_INFINITY
        const r = weightedAStarSearch(problem, w)
        if (!Number.isFinite(optimum)) {
          expect(r.status).toBe('failure')
          return
        }
        expect(r.cost).toBeLessThanOrEqual(w * optimum + 1e-9)
        for (const x of [r, greedyBestFirstSearch(problem), beamSearch(problem, { width: 2 })]) {
          if (x.status === 'solved') expect(replay(problem, x.actions).goal).toBe(true)
        }
      }),
      { numRuns: 120 },
    )
  })

  it('breadth-first and iterative deepening search find the fewest hops', () => {
    fc.assert(
      fc.property(case_, ({ g, s, t }) => {
        const hops = dijkstra(g, s, { weighted: false }).distance[t] ?? Number.POSITIVE_INFINITY
        const bfs = breadthFirstSearch(graphProblem(g, s, t))
        expect(bfs.status === 'solved' ? bfs.depth : Number.POSITIVE_INFINITY).toBe(hops)
        const ids = iterativeDeepeningSearch(graphProblem(g, s, t), { maxDepth: g.n, avoidCycles: true })
        expect(ids.status === 'solved' ? ids.depth : Number.POSITIVE_INFINITY).toBe(hops)
      }),
      { numRuns: 120 },
    )
  })

  it('inflated heuristics are caught as inadmissible', () => {
    fc.assert(
      fc.property(case_, ({ g, s, t }) => {
        const hStar = costToGo(g, t)
        if (!Number.isFinite(hStar[s] ?? Number.POSITIVE_INFINITY) || s === t) return
        const problem = graphProblem(g, s, t, { heuristic: (v) => 2 * (hStar[v] ?? 0) + 1 })
        expect(isAdmissible(problem).holds).toBe(false)
        expect(isConsistent(problem).holds).toBe(false)
      }),
      { numRuns: 60 },
    )
  })
})
