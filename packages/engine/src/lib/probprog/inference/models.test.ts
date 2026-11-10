// A model reaches a method run by name through a registry: the worker's module registers models, lists the methods it
// offers (`<kind>/<model>`), and a request carries only a name, a seed and JSON knobs. These tests are the engine side of a
// worker: methods are looked up by name from a table as the producer's `methodRegistry` does, knobs and data go through JSON
// as they do on the wire, and the run equals the one made directly on the main thread.
import { describe, expect, it } from 'vitest'
import { item, type Tensor } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { analyseSpec, defineModel } from '../model/index.ts'
import { inferenceMethodFor, inferenceMethods, methodKinds, modelRegistry } from './models.ts'
import { nutsMethod } from './nuts-method.ts'

const normalMean = defineModel('normal-mean', (b) => {
  b.observe('y', d.normal(b.latent('mu', d.normal(0, 2)), 1.5), b.data('y', { dims: ['data'] }))
})
const models = modelRegistry({ 'normal-mean': { definition: normalMean, data: { y: [0.3, -1.2, 2.1, 0.8] } } })
const exactMean = (y: number[]): number => y.reduce((a, v) => a + v, 0) / 2.25 / (1 / 4 + y.length / 2.25)

describe('the model registry', () => {
  it('holds named models bound to their data, and rebinds them to JSON data', () => {
    expect(models.names()).toEqual(['normal-mean'])
    expect(models.need('normal-mean').sizes).toEqual({ data: 4 })
    expect(models.bind('normal-mean', JSON.parse('{ "y": [1, 2, 3] }')).sizes).toEqual({ data: 3 })
    expect(() => models.need('radon')).toThrow(/no model called 'radon' \(known: normal-mean\)/)
    expect(() => models.bind('normal-mean', { z: [1] })).toThrow(/z/)
  })

  it('offers a method per kind and model, named kind/model, and refuses a kind it does not know', () => {
    const offered = inferenceMethods(models, ['nuts', 'svi', 'laplace'])
    expect(offered.map((m) => m.name)).toEqual(['nuts/normal-mean', 'svi/normal-mean', 'laplace/normal-mean'])
    expect(Object.keys(methodKinds)).toContain('abc-smc')
    expect(() => inferenceMethodFor(models, 'nonesuch', 'normal-mean')).toThrow(
      /no inference method of kind 'nonesuch'/,
    )
  })
})

describe('a method run by name', () => {
  // As the worker: a table of the methods on offer, a name, a seed and knobs that crossed as JSON.
  const table = new Map(inferenceMethods(models, ['nuts', 'svi', 'conjugate', 'gibbs']).map((m) => [m.name, m]))
  const wire = (name: string, config: unknown, seed: number, steps: number) => {
    const m = table.get(name)
    if (m === undefined) throw new Error(`no method called '${name}'`)
    return { m, s: session(m, { config: JSON.parse(JSON.stringify(config)), seed, steps, mode: 'eager' }) }
  }

  it('equals the run made directly on the main thread, for the same seed and knobs', () => {
    const knobs = { numChains: 2, numWarmup: 40, numDraws: 30 }
    const remote = wire('nuts/normal-mean', knobs, 7, 200)
    const direct = session(nutsMethod(models.need('normal-mean')), {
      config: knobs,
      seed: 7,
      steps: 200,
      mode: 'eager',
    })
    expect(remote.s.t).toBe(direct.t)
    const a = remote.m.posterior?.(remote.s.state, remote.s.config)
    const b = nutsMethod(models.need('normal-mean')).posterior?.(direct.state, direct.config)
    expect(a?.mean('mu').data).toEqual(b?.mean('mu').data)
  })

  it('takes the data of a run from JSON and builds the method from the registered spec', () => {
    const y = [3, 4, 5, 6, 7]
    const m = inferenceMethodFor(models, 'conjugate', 'normal-mean', JSON.parse(JSON.stringify({ y })))
    const s = session(m, { steps: 3, mode: 'eager' })
    expect(Math.abs(item(m.posterior?.(s.state, s.config).mean('mu') as Tensor) - exactMean(y))).toBeLessThan(1e-9)
    const svi = inferenceMethodFor(models, 'svi', 'normal-mean', { y })
    expect(svi.check?.(analyseSpec(normalMean)).applies).toBe(true)
  })

  it('every method on offer carries its own knobs and tape, only numbers, flags and choices cross', () => {
    for (const m of table.values()) {
      const cross = (v: unknown): boolean => {
        if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') return true
        if (Array.isArray(v)) return v.every(cross)
        return typeof v === 'object' && v !== null && Object.values(v).every(cross)
      }
      expect(cross(session(m, { steps: 1 }).config), m.name).toBe(true)
      expect(JSON.parse(JSON.stringify(session(m, { steps: 1 }).config))).toEqual(session(m, { steps: 1 }).config)
    }
  })
})
