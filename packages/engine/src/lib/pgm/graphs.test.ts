import { describe, expect, it } from 'vitest'
import * as P from './index.ts'

describe('directed and undirected graphs', () => {
  const g = P.dag([
    ['a', 'b'],
    ['b', 'c'],
    ['a', 'd'],
    ['d', 'c'],
    ['c', 'e'],
  ])

  it('orders nodes topologically and rejects cycles', () => {
    expect(g.nodes.map((n) => n.name)).toEqual(['a', 'b', 'd', 'c', 'e'])
    expect(() =>
      P.dag([
        ['a', 'b'],
        ['b', 'a'],
      ]),
    ).toThrow(/cycle/)
    expect(P.dagEdges(g)).toHaveLength(5)
  })

  it('finds ancestors, descendants, children and blankets', () => {
    expect(P.ancestors(g, ['c'])).toEqual(['a', 'b', 'c', 'd'])
    expect(P.descendants(g, 'b')).toEqual(['b', 'c', 'e'])
    expect(P.children(g, 'a')).toEqual(['b', 'd'])
    expect(P.markovBlanket(g, 'b')).toEqual(['a', 'c', 'd'])
  })

  it('d-separation agrees with Bayes-ball reachability and minimal separators separate', () => {
    for (const x of ['a', 'b', 'd']) {
      for (const observed of [[], ['c'], ['e'], ['b', 'd']]) {
        const reach = P.activeTrailNodes(g, x, observed)
        for (const y of ['a', 'b', 'c', 'd', 'e']) {
          if (y === x || observed.includes(y)) continue
          expect(P.dSeparated(g, [x], [y], observed)).toBe(!reach.includes(y))
        }
      }
    }
    const sep = P.minimalDSeparator(g, 'a', 'c')
    expect(sep).toEqual(['b', 'd'])
    expect(P.minimalDSeparator(g, 'a', 'b')).toBeNull()
    for (const s of P.independencies(g)) expect(P.dSeparated(g, s.x, s.y, s.given)).toBe(true)
  })

  it('moral graphs marry co-parents; interaction graphs join factor scopes', () => {
    expect(P.graphEdges(P.moralGraph(g))).toEqual([
      ['a', 'b'],
      ['a', 'd'],
      ['b', 'c'],
      ['b', 'd'],
      ['c', 'd'],
      ['c', 'e'],
    ])
    const f = P.factor(['x', 'y', 'z'], [2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1])
    expect(P.graphEdges(P.interactionGraph([f], ['w']))).toEqual([
      ['x', 'y'],
      ['x', 'z'],
      ['y', 'z'],
    ])
    expect(P.connectedComponents(P.interactionGraph([f], ['w']))).toEqual([['w'], ['x', 'y', 'z']])
  })
})

describe('graph algorithms from the graph domain keep pgm orders', () => {
  it('topologicalOrder keeps the given order among ready nodes and reports cycles and unknown parents', () => {
    const names = (ns: readonly P.DagNode[]): string[] => ns.map((n) => n.name)
    expect(
      names(
        P.topologicalOrder([
          { name: 'z', parents: [] },
          { name: 'b', parents: ['z', 'z'] },
          { name: 'a', parents: [] },
          { name: 'c', parents: ['a'] },
        ]),
      ),
    ).toEqual(['z', 'b', 'a', 'c'])
    expect(() =>
      P.topologicalOrder([
        { name: 'a', parents: ['b'] },
        { name: 'b', parents: ['a'] },
      ]),
    ).toThrow(/cycle/)
    expect(() => P.topologicalOrder([{ name: 'a', parents: ['q'] }])).toThrow(/unknown parent q/)
  })

  it('components and cliques are sorted by name, cliques in locale order', () => {
    const u = P.undirectedGraph(
      [
        ['b', 'B'],
        ['B', 'a'],
        ['a', 'b'],
        ['a', 'c'],
      ],
      ['Z', 'y'],
    )
    expect(P.connectedComponents(u)).toEqual([['B', 'a', 'b', 'c'], ['Z'], ['y']])
    expect(P.maximalCliques(u)).toEqual([['a', 'c'], ['B', 'a', 'b'], ['y'], ['Z']])
  })
})

describe('triangulation and treewidth', () => {
  const grid = (k: number): Map<string, Set<string>> => {
    const edges: [string, string][] = []
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        if (c + 1 < k) edges.push([`${r}${c}`, `${r}${c + 1}`])
        if (r + 1 < k) edges.push([`${r}${c}`, `${r + 1}${c}`])
      }
    }
    return P.undirectedGraph(edges)
  }

  it('has the known treewidths of grids, cycles, trees and cliques', () => {
    expect(P.treewidth(grid(2))).toBe(2)
    expect(P.treewidth(grid(3))).toBe(3)
    expect(P.treewidth(grid(4))).toBe(4)
    const cycle = P.undirectedGraph(Array.from({ length: 7 }, (_, i): [string, string] => [`v${i}`, `v${(i + 1) % 7}`]))
    expect(P.treewidth(cycle)).toBe(2)
    const star = P.undirectedGraph(['b', 'c', 'd', 'e'].map((v): [string, string] => ['a', v]))
    expect(P.treewidth(star)).toBe(1)
    const k5 = P.undirectedGraph(
      ['a', 'b', 'c', 'd', 'e'].flatMap((x, i, all) => all.slice(i + 1).map((y): [string, string] => [x, y])),
    )
    expect(P.treewidth(k5)).toBe(4)
    expect(P.treewidth(grid(3), 'heuristic')).toBeGreaterThanOrEqual(3)
  })

  it('computes elimination costs and orders by each heuristic', () => {
    const g = P.undirectedGraph([
      ['a', 'b'],
      ['a', 'c'],
      ['a', 'd'],
      ['b', 'c'],
    ])
    const card = { a: 2, b: 3, c: 4, d: 5 }
    expect(P.eliminationCost(g, 'a', 'minDegree')).toBe(3)
    expect(P.eliminationCost(g, 'a', 'minFill')).toBe(2)
    expect(P.eliminationCost(g, 'a', 'minWeight', card)).toBe(60)
    expect(P.eliminationCost(g, 'a', 'weightedMinFill', card)).toBe(3 * 5 + 4 * 5)
    for (const h of ['minDegree', 'minFill', 'minWeight', 'weightedMinFill'] as const) {
      const order = P.eliminationOrder(g, ['a', 'b', 'c', 'd'], h, card)
      expect([...order].sort()).toEqual(['a', 'b', 'c', 'd'])
      expect(P.inducedWidth(g, order)).toBe(2)
    }
    expect(P.maximumCardinalitySearch(g)[0]).toBe('a')
  })

  it('triangulates cycles with one chord per extra vertex and joins cliques into a junction tree', () => {
    const square = P.undirectedGraph([
      ['a', 'b'],
      ['b', 'c'],
      ['c', 'd'],
      ['d', 'a'],
    ])
    expect(P.isChordal(square)).toBe(false)
    const t = P.triangulate(square, ['a', 'b', 'c', 'd'])
    expect(t.fillIns).toEqual([['b', 'd']])
    expect(t.cliques).toEqual([
      ['a', 'b', 'd'],
      ['b', 'c', 'd'],
    ])
    const jt = P.junctionTree(square, ['a', 'b', 'c', 'd'])
    expect(jt.edges).toEqual([{ from: 0, to: 1, separator: ['b', 'd'] }])
    expect(() => P.triangulate(square, ['a', 'b'])).toThrow(/every vertex/)
  })
})
