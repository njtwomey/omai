import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { fromStorage, stack } from '../tensor/index.ts'
import {
  betweennessCentrality,
  closenessCentrality,
  degreeCentrality,
  edgeBetweennessCentrality,
  eigenvectorCentrality,
  hits,
  katzCentrality,
  pagerank,
} from './index.ts'

const vec = (x: Float64Array) => fromStorage(x, [x.length], 'float64')

describe('centrality against networkx', () => {
  for (const c of loadFixture('graph/centrality').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const p = c.params
      const check = (x: Float64Array): void => expectMatches(vec(x), c.output, c.rtol, c.atol, c.name)
      switch (c.fn) {
        case 'degree':
          return check(degreeCentrality(g))
        case 'closeness':
          return check(closenessCentrality(g))
        case 'closenessWeighted':
          return check(closenessCentrality(g, { weighted: true }))
        case 'betweenness':
          return check(
            betweennessCentrality(g, { normalized: p['normalized'] as boolean, endpoints: p['endpoints'] as boolean }),
          )
        case 'betweennessWeighted':
          return check(betweennessCentrality(g, { weighted: true }))
        case 'edgeBetweenness':
          return check(edgeBetweennessCentrality(g))
        case 'pagerank':
          return check(pagerank(g, { tol: 1e-13, maxIter: 1000 }))
        case 'pagerankPersonal':
          return check(
            pagerank(g, {
              alpha: 0.7,
              personalization: p['personalization'] as number[],
              tol: 1e-13,
              maxIter: 1000,
            }),
          )
        case 'hits': {
          const r = hits(g, { tol: 1e-14, maxIter: 10000 })
          return expectMatches(stack([vec(r.hubs), vec(r.authorities)]), c.output, c.rtol, c.atol, c.name)
        }
        case 'katz':
          return check(katzCentrality(g, { alpha: 0.05, tol: 1e-13, maxIter: 10000 }))
        case 'eigenvector':
          return check(eigenvectorCentrality(g, { tol: 1e-13, maxIter: 10000 }))
        default:
          throw new Error(`unknown case ${c.fn}`)
      }
    })
  }
})
