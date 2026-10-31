import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import { fromStorage } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  averageClustering,
  averageShortestPathLength,
  effectiveResistance,
  growthGraph,
  kirchhoffIndex,
  rewiringAlgorithm,
  ringLatticePathLength,
  spanningTreeCount,
  wattsStrogatzClustering,
} from './index.ts'

describe('effective resistance and path lengths against networkx', () => {
  for (const c of loadFixture('graph/resistance').cases) {
    it(c.name, () => {
      const p = c.params
      if (c.fn === 'ring') {
        const n = p['n'] as number
        const k = p['k'] as number
        const g = growthGraph(n, runAlgorithm(rewiringAlgorithm(n, k, 0), { seed: 0, maxSteps: n * k }).state.edges)
        const out = fromStorage(Float64Array.of(averageClustering(g), averageShortestPathLength(g)), [2], 'float64')
        expectMatches(out, c.output, c.rtol, c.atol, c.name)
        expect(ringLatticePathLength(n, k)).toBeCloseTo(averageShortestPathLength(g), 12)
        if (k > 2) expect(wattsStrogatzClustering(k, 0)).toBeCloseTo(averageClustering(g), 12)
        return
      }
      const g = graphFromCase(p)
      const match = (y: Parameters<typeof expectMatches>[0]): void => expectMatches(y, c.output, c.rtol, c.atol, c.name)
      switch (c.fn) {
        case 'resistance':
          return match(effectiveResistance(g))
        case 'kirchhoff':
          return match(fromStorage(Float64Array.of(kirchhoffIndex(g)), [], 'float64'))
        case 'trees':
          return match(
            fromStorage(
              Float64Array.of(spanningTreeCount(g), spanningTreeCount(g, { weighted: true })),
              [2],
              'float64',
            ),
          )
        case 'averagePathLength':
          return match(fromStorage(Float64Array.of(averageShortestPathLength(g)), [], 'float64'))
        default:
          throw new Error(`unknown case function ${c.fn}`)
      }
    })
  }
})
