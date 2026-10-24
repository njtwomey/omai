import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { summarise } from './summaries.ts'

describe('tensor summaries against numpy', () => {
  for (const c of loadFixture('trace/summaries').cases) {
    it(c.name, () => {
      const x = (decode(c.inputs[0]) as Tensor).data
      const bins = typeof c.params['bins'] === 'number' ? c.params['bins'] : 10
      const s = summarise(x, bins)
      const got =
        c.fn === 'moments'
          ? [s.n, s.mean, s.std, s.min, s.max, s.norm, s.zeros]
          : c.fn === 'histogram'
            ? [...s.histogram.counts]
            : [...s.histogram.edges]
      expectMatches(tensor(got), c.output, c.rtol, c.atol, c.name)
    })
  }
})
