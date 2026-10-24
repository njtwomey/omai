import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { MODELS } from '../../testing/probprog.ts'
import { key } from '../random/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { autoNormal, elboAt, type Model, renyiBound, substitute, traceMeanFieldElbo } from './index.ts'

function values(x: unknown): Record<string, Tensor> {
  return Object.fromEntries(Object.entries(x as Record<string, unknown>).map(([k, v]) => [k, decode(v) as Tensor]))
}

describe('probprog/elbo against NumPyro', () => {
  for (const c of loadFixture('probprog/elbo').cases) {
    it(c.name, () => {
      const model = MODELS[c.params['model'] as string] as Model
      const guide = autoNormal(model)
      const params = values(c.inputs[0])
      switch (c.fn) {
        case 'elboAt':
          expectMatches(
            elboAt(model, guide, undefined, params, values(c.inputs[1])).elbo,
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          return
        case 'meanField': {
          const fixed = substitute(guide, values(c.inputs[1]))
          const r = traceMeanFieldElbo().estimate(key(0), model, fixed, undefined, params, null)
          expectMatches(r.elbo, c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'renyi': {
          const w = c.inputs.slice(1).map((d) => elboAt(model, guide, undefined, params, values(d)).elbo)
          expectMatches(renyiBound(w, c.params['alpha'] as number), c.output, c.rtol, c.atol, c.name)
          return
        }
        default:
          throw new Error(`no function ${c.fn}`)
      }
    })
  }
})
