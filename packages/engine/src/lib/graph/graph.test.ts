import { describe, expect, it } from 'vitest'
import { isStructure } from '../scene/index.ts'
import {
  degree,
  digraph,
  edgeList,
  edgeWeight,
  graph,
  hasEdge,
  inDegree,
  isGraph,
  neighbours,
  nodeIndex,
  outDegree,
  predecessors,
  reverse,
  subgraph,
  successors,
  toDirected,
  toStructure,
  toUndirected,
} from './index.ts'

describe('graph', () => {
  it('resolves labels and indices, growing the node set', () => {
    const g = graph(
      [
        ['a', 'b'],
        ['b', 'c'],
      ],
      { nodes: ['z'] },
    )
    expect(g.labels).toEqual(['z', 'a', 'b', 'c'])
    expect(nodeIndex(g, 'c')).toBe(3)
    expect(nodeIndex(g, 0)).toBe(0)
    expect(() => nodeIndex(g, 'q')).toThrow()
    expect(() => nodeIndex(g, 9)).toThrow()
    expect(graph([[3, 1]]).labels).toEqual(['0', '1', '2', '3'])
    expect(() => graph([[-1, 0]])).toThrow()
    expect(() => graph([[0, 1, Number.NaN]])).toThrow()
    expect(isGraph(g)).toBe(true)
    expect(isGraph({ kind: 'scene' })).toBe(false)
  })

  it('combines repeated edges as asked', () => {
    const edges: [number, number, number][] = [
      [0, 1, 2],
      [1, 0, 5],
    ]
    expect(graph(edges).edges[0]?.weight).toBe(5)
    expect(graph(edges, { duplicates: 'sum' }).edges[0]?.weight).toBe(7)
    expect(graph(edges, { duplicates: 'min' }).edges[0]?.weight).toBe(2)
    expect(graph(edges, { duplicates: 'max' }).edges[0]?.weight).toBe(5)
    expect(() => graph(edges, { duplicates: 'error' })).toThrow()
    // Directed graphs keep both orientations.
    expect(digraph(edges).edges.length).toBe(2)
    expect(graph([{ source: 'x', target: 'y', weight: 3, data: { kind: 'road' } }]).edgeData[0]).toEqual({
      kind: 'road',
    })
  })

  it('counts degrees with self-loops twice when undirected, as NetworkX', () => {
    const g = graph([
      [0, 0, 2],
      [0, 1],
    ])
    expect(degree(g)).toEqual([3, 1])
    expect(degree(g, { weighted: true })).toEqual([5, 1])
    expect(neighbours(g, 0)).toEqual([0, 1])
    const d = digraph([
      [0, 1],
      [2, 1],
      [1, 0],
      [1, 1],
    ])
    expect(inDegree(d)).toEqual([1, 3, 0])
    expect(outDegree(d)).toEqual([1, 2, 1])
    expect(degree(d)).toEqual([2, 5, 1])
    expect(successors(d, 1)).toEqual([0, 1])
    expect(predecessors(d, 1)).toEqual([0, 1, 2])
    expect(neighbours(d, 1)).toEqual([0, 1, 2])
  })

  it('looks up edges both ways when undirected and one way when directed', () => {
    const g = graph([[2, 0, 4]])
    expect(edgeWeight(g, 0, 2)).toBe(4)
    expect(hasEdge(g, 2, 0)).toBe(true)
    const d = digraph([[2, 0, 4]])
    expect(hasEdge(d, 0, 2)).toBe(false)
    expect(edgeList(d)).toEqual([[2, 0, 4]])
  })

  it('takes induced subgraphs, reverses and converts directedness', () => {
    const d = digraph(
      [
        ['a', 'b', 1],
        ['b', 'a', 4],
        ['b', 'c', 2],
      ],
      { nodeData: [{ x: 1 }, { x: 2 }, { x: 3 }] },
    )
    const s = subgraph(d, ['c', 'b'])
    expect(s.labels).toEqual(['c', 'b'])
    expect(s.edges).toEqual([{ source: 1, target: 0, weight: 2 }])
    expect(s.nodeData).toEqual([{ x: 3 }, { x: 2 }])
    expect(successors(reverse(d), 'c')).toEqual([1])
    expect(toUndirected(d).edges).toEqual([
      { source: 0, target: 1, weight: 4 },
      { source: 1, target: 2, weight: 2 },
    ])
    expect(toUndirected(d, { combine: 'mean' }).edges[0]?.weight).toBe(2.5)
    expect(toUndirected(d, { combine: 'min' }).edges[0]?.weight).toBe(1)
    const u = graph([
      [0, 1],
      [1, 1],
    ])
    expect(reverse(u)).toBe(u)
    expect(toUndirected(u)).toBe(u)
    expect(toDirected(u).edges.length).toBe(3)
    expect(toDirected(d)).toBe(d)
  })

  it('describes itself as a structure with groups and highlights', () => {
    const g = digraph([
      ['a', 'b'],
      ['b', 'c'],
    ])
    const s = toStructure(g, { groups: [0, 0, 1], highlight: ['b'], title: 'chain' })
    expect(isStructure(s)).toBe(true)
    expect(s.type).toBe('digraph')
    expect(s.groups.map((x) => x.id)).toEqual(['c0', 'c1'])
    expect(s.nodes[2]?.group).toBe('c1')
    expect(s.edges.every((e) => e.directed === true && e.label === undefined)).toBe(true)
    expect(s.annotations?.[0]?.target).toBe('n1')
    expect(s.title).toBe('chain')
  })
})
