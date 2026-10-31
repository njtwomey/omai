import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor } from '../../tensor/index.ts'
import { CrossNetwork, dotInteraction, FactorisationMachine, FieldAwareFm, NeuralCF } from './recommender.ts'

type Params = Readonly<Record<string, unknown>>

const stateOf = (p: Params): Record<string, Tensor> =>
  Object.fromEntries(Object.entries(p['state'] as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))

describe('recommender blocks against torch', () => {
  for (const c of loadFixture('nn/blocks/recommender').cases) {
    it(c.name, () => {
      const cfg = (c.params['config'] ?? {}) as Params
      const xs = c.inputs.map((v) => decode(v) as Tensor)
      const check = (got: Parameters<typeof expectMatches>[0]): void =>
        expectMatches(got, c.output, 1e-11, 1e-12, c.name)
      switch (c.fn) {
        case 'fm': {
          const m = new FactorisationMachine(cfg['features'] as number, cfg['factors'] as number).loadStateDict(
            stateOf(c.params),
          )
          return check(m.forward({ ids: xs[0] as Tensor, ...(xs[1] === undefined ? {} : { values: xs[1] }) }))
        }
        case 'ffm': {
          const m = new FieldAwareFm(
            cfg['features'] as number,
            cfg['fields'] as number,
            cfg['factors'] as number,
          ).loadStateDict(stateOf(c.params))
          return check(m.forward({ ids: xs[0] as Tensor, values: xs[1] as Tensor }))
        }
        case 'ncf': {
          const m = new NeuralCF(cfg['users'] as number, cfg['items'] as number, {
            gmfDim: cfg['gmf'] as number,
            mlpDim: cfg['mlp'] as number,
            hidden: cfg['hidden'] as number[],
          }).loadStateDict(stateOf(c.params))
          return check(m.forward({ user: xs[0] as Tensor, item: xs[1] as Tensor }))
        }
        case 'dotInteraction':
          return check(dotInteraction(xs[0] as Tensor))
        default: {
          const m = new CrossNetwork(cfg['dim'] as number, cfg['layers'] as number).loadStateDict(stateOf(c.params))
          return check(m.forward(xs[0] as Tensor))
        }
      }
    })
  }
})
