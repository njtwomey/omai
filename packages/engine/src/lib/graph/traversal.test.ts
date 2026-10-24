import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { arbitraryGraph } from '../../testing/graphs.ts'
import { iterate, policies, recorder } from '../trace/index.ts'
import {
  bfs,
  bipartiteColouring,
  cycleGraph,
  dfs,
  dfsAlgorithm,
  digraph,
  findCycle,
  graph,
  hasEdge,
  isBipartite,
  isDag,
  kHopNeighbourhood,
  pathGraph,
  topologicalSort,
} from './index.ts'

describe('traversal', () => {
  it('classifies undirected DFS edges as tree or back, each edge once', () => {
    const g = graph([
      [0, 1],
      [1, 2],
      [2, 0],
      [2, 3],
      [3, 3],
    ])
    const r = dfs(g)
    expect(r.edges.map((e) => [e.source, e.target, e.kind])).toEqual([
      [0, 1, 'tree'],
      [1, 2, 'tree'],
      [2, 0, 'back'],
      [2, 3, 'tree'],
      [3, 3, 'back'],
    ])
    expect(Array.from(r.discovery)).toEqual([0, 1, 2, 3])
    expect(Array.from(r.finish)).toEqual([7, 6, 5, 4])
  })

  it('finds cross edges in a directed forest and visits every root', () => {
    const g = digraph([
      [0, 1],
      [2, 1],
      [2, 3],
    ])
    const r = dfs(g)
    expect(r.preorder).toEqual([0, 1, 2, 3])
    expect(r.edges.find((e) => e.source === 2 && e.target === 1)?.kind).toBe('cross')
    expect(Array.from(r.parent)).toEqual([-1, 0, -1, 2])
  })

  it('runs BFS from several sources, against edges, and to a depth', () => {
    const g = digraph([
      [0, 1],
      [1, 2],
      [3, 2],
    ])
    expect(bfs(g, [0, 3]).layers).toEqual([
      [0, 3],
      [1, 2],
    ])
    expect(bfs(g, 2, { direction: 'in' }).order).toEqual([2, 1, 3, 0])
    expect(bfs(g, 2, { direction: 'both' }).order).toEqual([2, 1, 3, 0])
    expect(bfs(pathGraph(6), 0, { maxDepth: 2 }).order).toEqual([0, 1, 2])
    expect(kHopNeighbourhood(g, 1, 1, { direction: 'in' })).toEqual([0, 1])
  })

  it('traces BFS layers and DFS events', () => {
    const rec = recorder()
    bfs(cycleGraph(6), 0, { tracer: rec, policy: policies.teaching })
    const layers = rec.events.filter((e) => e.type === 'message')
    expect(layers.length).toBe(5)
    const states = Array.from(iterate(dfsAlgorithm(pathGraph(3), 0), { maxSteps: 100 }), (s) => s.state.event?.kind)
    expect(states.filter((k) => k !== undefined)).toEqual(['discover', 'tree', 'tree', 'finish', 'finish', 'finish'])
  })

  it('sorts topologically and detects cycles', () => {
    expect(() => topologicalSort(graph([[0, 1]]))).toThrow()
    expect(() =>
      topologicalSort(
        digraph([
          [0, 1],
          [1, 0],
        ]),
      ),
    ).toThrow()
    expect(isDag(graph([[0, 1]]))).toBe(false)
    expect(findCycle(graph([[0, 0]]))).toEqual([[0, 0]])
    expect(findCycle(cycleGraph(4))).toEqual([
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
    ])
    expect(
      findCycle(
        digraph([
          [0, 1],
          [1, 2],
        ]),
        0,
      ),
    ).toBeNull()
  })

  it('returns valid cycles and orders (property)', () => {
    fc.assert(
      fc.property(arbitraryGraph({ directed: true, maxNodes: 8, density: 0.25 }), (g) => {
        const cycle = findCycle(g)
        expect(cycle === null).toBe(isDag(g))
        if (cycle !== null) {
          for (const [u, v] of cycle) expect(hasEdge(g, u, v)).toBe(true)
          expect(cycle[0]?.[0]).toBe(cycle[cycle.length - 1]?.[1])
        } else {
          const order = topologicalSort(g)
          const position = new Map(order.map((v, i) => [v, i]))
          for (const e of g.edges) expect(position.get(e.source) ?? 0).toBeLessThan(position.get(e.target) ?? 0)
        }
      }),
    )
  })

  it('colours bipartite graphs properly (property)', () => {
    fc.assert(
      fc.property(arbitraryGraph({ maxNodes: 9 }), (g) => {
        const colour = bipartiteColouring(g)
        expect(colour !== null).toBe(isBipartite(g))
        if (colour !== null) for (const e of g.edges) expect(colour[e.source]).not.toBe(colour[e.target])
        // A BFS tree's depths differ by at most one along every edge.
        const d = bfs(g, 0).distance
        for (const e of g.edges) {
          if ((d[e.source] ?? -1) >= 0) expect(Math.abs((d[e.source] ?? 0) - (d[e.target] ?? 0))).toBeLessThanOrEqual(1)
        }
      }),
    )
  })
})
