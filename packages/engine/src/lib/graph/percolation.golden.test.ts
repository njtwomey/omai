import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { graphFromCase } from '../../testing/graphs.ts'
import type { Tensor } from '../tensor/index.ts'
import { iterate } from '../trace/index.ts'
import { percolationAlgorithm } from './index.ts'

describe('percolation bookkeeping against networkx components', () => {
  for (const c of loadFixture('graph/percolation').cases) {
    it(c.name, () => {
      const g = graphFromCase(c.params)
      const order = c.params['order'] as number[]
      const mode = c.fn === 'bond' ? 'bond' : 'site'
      const want = decode(c.output) as Tensor
      const cols = 2
      const alg = percolationAlgorithm(g, { mode, order })
      for (const { t, state } of iterate(alg, { seed: 0, maxSteps: order.length })) {
        // Site percolation counts only occupied sites; bond percolation starts from n singletons.
        expect(state.clusters).toBe(want.data[t * cols])
        expect(state.largest).toBe(want.data[t * cols + 1])
      }
    })
  }
})
