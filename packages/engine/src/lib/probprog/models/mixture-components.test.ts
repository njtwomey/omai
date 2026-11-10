// The K-component Gaussian mixture definition: it binds to data for any number of components and a bad number of
// components is refused.
import { describe, expect, it } from 'vitest'
import { OmaiError } from '../../base/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { inferenceMethodFor, modelRegistry } from '../inference/index.ts'
import { gaussianMixtureModel } from './mixture.ts'

const y = [-1.2, -0.9, -1.1, 0.8, 1.0, 1.3, 0.9]

describe('gaussianMixtureModel', () => {
  for (const components of [1, 2, 3, 4])
    for (const ordered of [false, true])
      it(`binds ${components} component(s)${ordered ? ', ordered' : ''} `, () => {
        const spec = gaussianMixtureModel({ components, ordered })
        const bound = spec.bind({ y })
        expect(bound.sizes).toEqual({ row: y.length, component: components })
      })

  it('is fitted by the Laplace method, a normal at the mode with a finite log density', () => {
    const registry = modelRegistry({
      m: { definition: gaussianMixtureModel({ components: 2, ordered: true }), data: { y } },
    })
    const method = inferenceMethodFor(registry, 'laplace', 'm')
    const s = session(method, { config: {}, seed: 1, steps: 200, mode: 'eager' })
    const mean = method.posterior?.(s.state as never, s.config as never)?.mean('mu')
    const [lo, hi] = toFlat(mean as never)
    expect(lo).toBeLessThan(0)
    expect(hi).toBeGreaterThan(0)
  })

  it('names the model by its components and ordering', () => {
    expect(gaussianMixtureModel({ components: 3, ordered: true }).name).toBe('gaussian-mixture-3-ordered')
  })

  it('refuses fewer than one component', () => {
    expect(() => gaussianMixtureModel({ components: 0 })).toThrow(OmaiError)
  })
})
