import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { add, matmul, sum, swapaxes, tensor, toArray, type Value } from '../tensor/index.ts'
import {
  adjacencyMatrix,
  cycleGraph,
  degreeMatrix,
  fromAdjacency,
  graph,
  incidenceMatrix,
  laplacian,
  laplacianPositionalEncoding,
  modularityMatrix,
  normalisedLaplacian,
  pathGraph,
  randomWalkLaplacian,
  randomWalkPositionalEncoding,
  transitionMatrix,
} from './index.ts'

// A symmetric weighted adjacency with distinct Laplacian eigenvalues and one isolated-free pattern.
const W = tensor([
  [0, 1.2, 0.3, 0],
  [1.2, 0, 0.7, 0.4],
  [0.3, 0.7, 0, 1.9],
  [0, 0.4, 1.9, 0],
])
const symmetric = (f: (a: Value) => Value) => (a: Value) => f(add(a, swapaxes(a, -1, -2)))

describe('graph matrices', () => {
  it('builds Laplacians whose rows sum to zero and whose incidence factorisation holds', () => {
    const g = graph([
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [0, 2],
    ])
    expect(toArray(sum(laplacian(g), -1))).toEqual([0, 0, 0, 0])
    const b = incidenceMatrix(g, { oriented: true })
    expect(toArray(matmul(b, swapaxes(b, 0, 1)))).toEqual(toArray(laplacian(g)))
    expect(toArray(degreeMatrix(g))).toEqual([
      [3, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 2],
    ])
    expect(toArray(incidenceMatrix(pathGraph(3)))).toEqual([
      [1, 0],
      [1, 1],
      [0, 1],
    ])
  })

  it('round-trips through adjacency matrices, dense and directed', () => {
    const g = graph(
      [
        [0, 1, 2],
        [1, 2, 3],
      ],
      { nodes: 4 },
    )
    const back = fromAdjacency(adjacencyMatrix(g))
    expect(back.edges).toEqual(g.edges)
    expect(back.n).toBe(4)
    const d = fromAdjacency(
      [
        [0, 1, 0],
        [0, 0, 2],
        [5, 0, 0],
      ],
      { directed: true },
    )
    expect(d.edges.map((e) => [e.source, e.target, e.weight])).toEqual([
      [0, 1, 1],
      [1, 2, 2],
      [2, 0, 5],
    ])
    expect(
      fromAdjacency([
        [0, 0],
        [3, 0],
      ]).edges,
    ).toEqual([{ source: 0, target: 1, weight: 3 }])
    expect(() => fromAdjacency([[0, 1, 2]])).toThrow()
  })

  it('handles isolated and dangling nodes', () => {
    const g = graph([[0, 1]], { nodes: 3 })
    expect((toArray(normalisedLaplacian(g)) as number[][])[2]).toEqual([0, 0, 0])
    expect((toArray(randomWalkLaplacian(g)) as number[][])[2]).toEqual([0, 0, 0])
    expect((toArray(transitionMatrix(g)) as number[][])[2]).toEqual([1 / 3, 1 / 3, 1 / 3])
    expect((toArray(transitionMatrix(g, { dangling: 'self' })) as number[][])[2]).toEqual([0, 0, 1])
    expect((toArray(transitionMatrix(g, { dangling: 'zero' })) as number[][])[2]).toEqual([0, 0, 0])
    expect(() => transitionMatrix(g, { laziness: 2 })).toThrow()
  })

  it('makes modularity and transition matrices with the expected sums', () => {
    expect((toArray(sum(modularityMatrix(cycleGraph(5)), -1)) as number[]).map((v) => Math.abs(v) < 1e-12)).toEqual([
      true,
      true,
      true,
      true,
      true,
    ])
    expect((toArray(sum(transitionMatrix(W), -1)) as number[]).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([
      1, 1, 1, 1,
    ])
  })

  it('makes positional encodings of the right shape and range', () => {
    expect(laplacianPositionalEncoding(cycleGraph(5), 2).shape).toEqual([5, 2])
    expect(() => laplacianPositionalEncoding(cycleGraph(5), 5)).toThrow()
    expect(() => randomWalkPositionalEncoding(cycleGraph(5), 0)).toThrow()
    const rw = toArray(randomWalkPositionalEncoding(pathGraph(3), 3)) as number[][]
    expect(rw[1]).toEqual([0, 1, 0])
  })

  it('differentiates the Laplacians, transition matrices and encodings in the weights (checkPrimitive)', () => {
    for (const f of [laplacian, normalisedLaplacian, randomWalkLaplacian, modularityMatrix, degreeMatrix]) {
      const report = checkPrimitive(
        symmetric((a) => f(a)),
        [W],
      )
      expect(report.failures).toEqual([])
    }
    expect(
      checkPrimitive(
        symmetric((a) => transitionMatrix(a, { laziness: 0.3 })),
        [W],
      ).failures,
    ).toEqual([])
    expect(
      checkPrimitive(
        symmetric((a) => randomWalkPositionalEncoding(a, 3)),
        [W],
      ).failures,
    ).toEqual([])
    expect(
      checkPrimitive(
        symmetric((a) => laplacianPositionalEncoding(a, 2)),
        [W],
        { rtol: 1e-4, atol: 1e-6 },
      ).failures,
    ).toEqual([])
  })
})
