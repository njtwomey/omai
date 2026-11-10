import { describe, expect, it } from 'vitest'
import { graphOracle } from './graph-oracle.ts'
import {
  fullyConnectedInput,
  gridCrf,
  gridInput,
  higherOrderChainCrf,
  higherOrderChainInput,
  skipChainInput,
  treeInput,
} from './graph-structures.ts'

const blank = (n: number): Record<string, number>[] => Array.from({ length: n }, () => ({}))
const pairs = (x: { edges: readonly { from: number; to: number }[] }): number[][] => x.edges.map((e) => [e.from, e.to])

describe('structure builders', () => {
  it('trees: parent arrays and edge lists make the same forest, and cycles, repeats and bad indices are refused', () => {
    expect(pairs(treeInput(blank(4), { parents: [-1, 0, 1, 1] }))).toEqual([
      [0, 1],
      [1, 2],
      [1, 3],
    ])
    expect(
      pairs(
        treeInput(blank(4), {
          edges: [
            [0, 1],
            [1, 2],
            [1, 3],
          ],
        }),
      ),
    ).toEqual([
      [0, 1],
      [1, 2],
      [1, 3],
    ])
    expect(() =>
      treeInput(blank(3), {
        edges: [
          [0, 1],
          [1, 2],
          [2, 0],
        ],
      }),
    ).toThrow(/cycle/)
    expect(() =>
      treeInput(blank(3), {
        edges: [
          [0, 1],
          [1, 0],
        ],
      }),
    ).toThrow(/repeated/)
    expect(() => treeInput(blank(3), { edges: [[0, 5]] })).toThrow(/range/)
    expect(() => treeInput(blank(3), { parents: [-1, 0] })).toThrow(/parent per node/)
    expect(treeInput([], { parents: [] }).edges).toEqual([])
  })

  it('grids: 4- and 8-neighbourhoods, undirected and directional types, one-column and one-row grids', () => {
    const g4 = gridInput(blank(6), 2, 3)
    const g8 = gridInput(blank(6), 2, 3, { neighbourhood: 8 })
    expect([g4.edges.length, g8.edges.length]).toEqual([7, 11])
    expect(new Set(g8.edges.map((e) => e.type))).toEqual(new Set(['grid', 'diagonal']))
    const directional = gridInput(blank(6), 2, 3, { neighbourhood: 8, directional: true })
    expect(directional.edges.filter((e) => e.type === 'right')).toHaveLength(4)
    expect(directional.edges.filter((e) => e.type === 'down')).toHaveLength(3)
    expect(directional.edges.filter((e) => e.type === 'down-right')).toHaveLength(2)
    expect(directional.edges.filter((e) => e.type === 'down-left')).toHaveLength(2)
    expect(gridInput(blank(3), 3, 1, { directional: true }).edges.map((e) => e.type)).toEqual(['down', 'down'])
    expect(gridInput(blank(3), 1, 3, { directional: true }).edges.map((e) => e.type)).toEqual(['right', 'right'])
    expect(() => gridInput(blank(5), 2, 3)).toThrow(/cells/)
    // the model made for the same options knows every type the input uses
    const model = gridCrf(['a', 'b'], [], { neighbourhood: 8, directional: true })
    expect(() => graphOracle(model).expectations(new Float64Array(model.layout.size), directional)).not.toThrow()
  })

  it('edge attributes are computed from the two ends', () => {
    const g = gridInput([{ f: 1 }, { f: 4 }], 1, 2, {
      edgeAttributes: (a, b) => ({ d: (b['f'] ?? 0) - (a['f'] ?? 0) }),
    })
    expect(g.edges[0]?.attributes).toEqual({ d: 3 })
  })

  it('skip chains: skip edges join repeated keys that are not adjacent, once, within the distance limit', () => {
    const x = skipChainInput(blank(6), {
      keys: ['x', 'x', undefined, 'x', 'y', 'y'],
      skips: [
        [0, 5],
        [5, 0],
        [0, 1],
      ],
    })
    expect(x.edges.filter((e) => e.type === 'chain')).toHaveLength(5)
    expect(pairs({ edges: x.edges.filter((e) => e.type === 'skip') })).toEqual([
      [0, 3],
      [1, 3],
      [0, 5],
    ])
    expect(
      skipChainInput(blank(6), { keys: ['x', 'x', undefined, 'x', 'y', 'y'], maxDistance: 2 }).edges.filter(
        (e) => e.type === 'skip',
      ),
    ).toHaveLength(1)
    expect(() => skipChainInput(blank(3), { skips: [[0, 9]] })).toThrow(/outside/)
    expect(() => skipChainInput(blank(3), { keys: ['a'] })).toThrow(/one key/)
  })

  it('higher-order chains: windows of every size up to order + 1, tied weights, order 1 has no cliques', () => {
    const x = higherOrderChainInput(blank(5), 3)
    expect(x.cliques?.map((c) => [c.type, c.nodes.join('')])).toEqual([
      ['ngram3', '012'],
      ['ngram3', '123'],
      ['ngram3', '234'],
      ['ngram4', '0123'],
      ['ngram4', '1234'],
    ])
    expect(higherOrderChainInput(blank(5), 1).cliques).toEqual([])
    expect(higherOrderChainCrf(['a', 'b'], [], 3, { bias: false }).layout.size).toBe(4 + 8 + 16)
    expect(() => higherOrderChainInput(blank(3), 0)).toThrow(/order/)
    expect(() => higherOrderChainCrf(['a'], [], 0)).toThrow(/order/)
  })

  it('fully connected graphs have an edge per pair', () => {
    expect(fullyConnectedInput(blank(5)).edges).toHaveLength(10)
    expect(fullyConnectedInput([]).edges).toEqual([])
  })
})
