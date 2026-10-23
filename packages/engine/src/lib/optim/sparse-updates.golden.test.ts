import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { trajectory } from '../../testing/trajectories.ts'
import * as O from './index.ts'

type Options = Readonly<Record<string, unknown>>

const FACTORIES: Readonly<Record<string, (o: Options) => O.GradientTransformation>> = {
  sparseAdam: (o) => O.sparseAdam(o as unknown as O.SparseAdamOptions),
  lazyAdagrad: (o) => O.lazyAdagrad(o as unknown as O.LazyAdagradOptions),
}

describe('sparse-gradient optimisers against torch.optim on sparse gradients', () => {
  for (const c of loadFixture('optim/sparse_updates').cases) {
    it(c.name, () => {
      const make = FACTORIES[c.fn] as (o: Options) => O.GradientTransformation
      const mask = c.params['mask'] as 'element' | 'rows'
      expectMatches(trajectory(c, make(c.params), { mask }).path, c.output, c.rtol, c.atol, c.name)
    })
  }
})
