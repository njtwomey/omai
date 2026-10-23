import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { trajectory } from '../../testing/trajectories.ts'
import * as O from './index.ts'

type Options = Readonly<Record<string, unknown>>

const FACTORIES: Readonly<Record<string, (o: Options) => O.GradientTransformation>> = {
  adam: (o) => O.adam(o as unknown as O.AdamOptions),
  adamw: (o) => O.adamw(o as unknown as O.AdamOptions),
  amsgrad: (o) => O.amsgrad(o as unknown as O.AdamOptions),
  adamax: (o) => O.adamax(o as unknown as O.AdamOptions),
  nadam: (o) => O.nadam(o as unknown as O.NadamOptions),
  radam: (o) => O.radam(o as unknown as O.NadamOptions),
  adafactor: (o) => O.adafactor(o as O.AdafactorOptions),
  lion: (o) => O.lion(o as unknown as O.LionOptions),
  lamb: (o) => O.lamb(o as unknown as O.AdamOptions),
}

describe('adaptive optimisers against torch.optim and optax', () => {
  for (const c of loadFixture('optim/adaptive').cases) {
    it(c.name, () => {
      const make = FACTORIES[c.fn] as (o: Options) => O.GradientTransformation
      expectMatches(trajectory(c, make(c.params)).path, c.output, c.rtol, c.atol, c.name)
    })
  }
})
