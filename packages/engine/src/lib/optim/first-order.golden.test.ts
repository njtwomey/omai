import { describe, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { trajectory } from '../../testing/trajectories.ts'
import * as O from './index.ts'

type Options = Readonly<Record<string, unknown>>

const FACTORIES: Readonly<Record<string, (o: Options) => O.GradientTransformation>> = {
  sgd: (o) => O.sgd(o as unknown as O.SgdOptions),
  rmsprop: (o) => O.rmsprop(o as unknown as O.RmspropOptions),
  adagrad: (o) => O.adagrad(o as unknown as O.AdagradOptions),
  adadelta: (o) => O.adadelta(o as O.AdadeltaOptions),
  rprop: (o) => O.rprop(o as O.RpropOptions),
  asgd: (o) => O.asgd(o as O.AsgdOptions),
  signSgd: (o) => O.signSgd(o as unknown as O.SignSgdOptions),
}

describe('first-order optimisers against torch.optim and optax', () => {
  for (const c of loadFixture('optim/first_order').cases) {
    it(c.name, () => {
      const make = FACTORIES[c.fn] as (o: Options) => O.GradientTransformation
      const { path, state } = trajectory(c, make(c.params))
      if (c.fn === 'asgd') {
        const out = c.output as { path: unknown; ax: unknown }
        expectMatches(path, out.path, c.rtol, c.atol, c.name)
        expectMatches((state as O.AsgdState).ax as never, out.ax, c.rtol, c.atol, `${c.name} average`)
      } else expectMatches(path, c.output, c.rtol, c.atol, c.name)
    })
  }
})
