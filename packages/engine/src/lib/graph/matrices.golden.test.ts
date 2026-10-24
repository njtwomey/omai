import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { toDense } from '../numerics/sparse/index.ts'
import {
  adjacencyCsr,
  adjacencyMatrix,
  incidenceMatrix,
  laplacian,
  laplacianCsr,
  laplacianPositionalEncoding,
  modularityMatrix,
  normalisedAdjacency,
  normalisedLaplacian,
  randomWalkLaplacian,
  randomWalkPositionalEncoding,
  transitionMatrix,
} from './index.ts'

describe('graph matrices against networkx and scipy.sparse.csgraph', () => {
  for (const c of loadFixture('graph/matrices').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const match = (x: Parameters<typeof expectMatches>[0]): void => expectMatches(x, c.output, c.rtol, c.atol, c.name)
      switch (c.fn) {
        case 'adjacency':
          match(adjacencyMatrix(g))
          match(toDense(adjacencyCsr(g)))
          // From the dense matrix, the graph functions agree with the matrix functions.
          return
        case 'incidence':
          return match(incidenceMatrix(g, { oriented: true }))
        case 'transition':
          return match(transitionMatrix(g))
        case 'laplacian':
          match(laplacian(g))
          match(laplacian(adjacencyMatrix(g)))
          return match(toDense(laplacianCsr(g)))
        case 'normalised':
          match(normalisedLaplacian(g))
          return match(toDense(laplacianCsr(g, { normalised: true })))
        case 'randomWalk':
          return match(randomWalkLaplacian(g))
        case 'modularityMatrix':
          return match(modularityMatrix(g))
        case 'gcn':
          return match(normalisedAdjacency(g))
        case 'meanAggregation':
          return match(normalisedAdjacency(g, { kind: 'rw' }))
        case 'rwpe':
          return match(randomWalkPositionalEncoding(g, 4))
        case 'lappe':
          expect(c.params['minGap'] as number).toBeGreaterThan(1e-6)
          return match(laplacianPositionalEncoding(g, 3))
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
