// The GPLVM with random Fourier features: its pieces (features, data, alignment) and the route EP takes. The direct formulation, a
// Gaussian process prior whose covariance is a custom function of the random coordinates, is refused with the compiler's reason; the
// finite-feature one is accepted and recovers the coordinates of a small data set up to the alignment.
import { describe, expect, it } from 'vitest'
import { gram, rbfKernel } from '../../kernels/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, reshape, tensor, toArray, toFlat, type Value, zeros } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { epRoute } from '../inference/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, toFactorGraph } from '../model/index.ts'
import { alignLatents, fourierFeatures, gplvmFourierModel, simulateGplvm } from './gplvm.ts'

const SIZE = { features: 6, coords: 1, outputs: 5 } as const

describe('fourierFeatures and simulateGplvm', () => {
  it('draw the shapes that were asked for, the same ones for the same key', () => {
    const a = fourierFeatures(key(3), { features: 5, coords: 2, lengthScale: 2 })
    expect([a.frequencies.shape, a.phases.shape]).toEqual([[2, 5], [5]])
    expect(toArray(fourierFeatures(key(3), { features: 5, coords: 2, lengthScale: 2 }).frequencies)).toEqual(
      toArray(a.frequencies),
    )
    const data = simulateGplvm(key(3), { points: 9, coords: 2, outputs: 4 })
    expect([data.y.shape, data.latent.shape, data.clean.shape]).toEqual([
      [9, 4],
      [9, 2],
      [9, 4],
    ])
    expect(Math.max(...toFlat(data.latent).map(Math.abs))).toBeLessThanOrEqual(1.7)
  })

  it('scale the frequencies by the inverse of the length scale', () => {
    const wide = toFlat(fourierFeatures(key(0), { features: 4000, coords: 1, lengthScale: 1 }).frequencies)
    const narrow = toFlat(fourierFeatures(key(0), { features: 4000, coords: 1, lengthScale: 4 }).frequencies)
    const sd = (v: readonly number[]): number => Math.sqrt(v.reduce((a, b) => a + b * b, 0) / v.length)
    expect(sd(wide) / sd(narrow)).toBeCloseTo(4, 10)
    expect(sd(wide)).toBeCloseTo(1, 1)
  })

  it('refuse a length scale that is not positive and a count that is not a whole number', () => {
    expect(() => fourierFeatures(key(0), { features: 3, coords: 1, lengthScale: 0 })).toThrow(/length scale/)
    expect(() => gplvmFourierModel({ features: 0, coords: 1, outputs: 3 })).toThrow(/features is a whole number/)
  })
})

describe('alignLatents', () => {
  it('give a perfect fit to a rotated, reflected and rescaled copy of the truth', () => {
    const truth = tensor([
      [1, 0.2],
      [-0.5, 1.4],
      [0.3, -1.1],
      [-1.2, -0.4],
      [0.9, 0.7],
    ])
    // a reflection of the second coordinate, a rotation of 0.6 radians, a scale of 2.5 and a shift
    const c = Math.cos(0.6)
    const s = Math.sin(0.6)
    const moved = [
      [1, -0.2],
      [-0.5, -1.4],
      [0.3, 1.1],
      [-1.2, 0.4],
      [0.9, -0.7],
    ].map(([u, v]) => [
      2.5 * (c * (u as number) - s * (v as number)) + 3,
      2.5 * (s * (u as number) + c * (v as number)) - 1,
    ])
    const fit = alignLatents(moved, truth)
    for (const r of toFlat(fit.correlation)) expect(r).toBeCloseTo(1, 10)
    expect(item(fit.error)).toBeLessThan(1e-10)
  })

  it('score unrelated coordinates low', () => {
    const a = normal(key(1), [60, 1])
    const b = normal(key(2), [60, 1])
    expect(Math.abs(toFlat(alignLatents(a, b).correlation)[0] as number)).toBeLessThan(0.4)
  })
})

describe('the GPLVM as a model for expectation propagation', () => {
  it('is refused when the covariance is a custom function of the random coordinates', () => {
    const direct = defineModel('gplvm-direct', (m) => {
      const point = m.range('point')
      const output = m.range('output', { size: 3 })
      const y = m.data('y', { dims: [point, output], role: 'response' })
      const x = m.latent('x', d.normal(0, 1), { dims: [point] })
      const cov = m.custom(
        'rbfGram',
        ([xs]: readonly { readonly value: Value }[]) =>
          gram(rbfKernel({ lengthScale: 1 }), reshape((xs as { value: Value }).value, [5, 1]), { jitter: 1e-6 }),
        [x],
        { eventRank: 2 },
      )
      const f = m.latent('f', d.multivariateNormal(zeros([5]), { covariance: cov }), { dims: [point, output] })
      m.observe('y', d.normal(f, 0.1), y)
    })
    const route = epRoute(direct.bind({ y: normal(key(0), [5, 3]) }))
    expect(route.via).toBe('none')
    expect(route.verdict.reason).toMatch(/'f' is opaque to message passing/)
    expect(route.verdict.missing).toEqual(['exponential-family factors'])
  })

  it('is accepted with the features, as a graph of the registry operators, and is not a network layer', () => {
    const { frequencies, phases } = fourierFeatures(key(0), SIZE)
    const bound = gplvmFourierModel(SIZE).bind({
      omega: frequencies,
      phase: phases,
      y: simulateGplvm(key(1), { points: 6, coords: 1, outputs: SIZE.outputs }).y,
    })
    expect(toFactorGraph(bound).issues).toEqual([])
    const route = epRoute(bound)
    expect(route.via).toBe('ep')
    expect(route.verdict.applies).toBe(true)
  })

  it('recovers the coordinates of a small data set, up to the alignment, and reports spreads of the right order', () => {
    const k = key(4)
    const options = { ...SIZE, noise: 0.1 }
    const { frequencies, phases } = fourierFeatures(k, options)
    const data = simulateGplvm(k, { points: 20, coords: 1, outputs: SIZE.outputs })
    const bound = gplvmFourierModel(options).bind({ omega: frequencies, phase: phases, y: data.y })
    const route = epRoute(bound)
    if (route.via !== 'ep') throw new Error(route.verdict.reason)
    const run = session(route.method, { steps: 70, mode: 'eager', config: { tol: 1e-6, damping: 0.7 } })
    const post = route.method.posterior?.(run.state, run.config)
    if (post === undefined) throw new Error('no posterior')
    const fit = alignLatents(post.mean('x'), data.latent)
    expect(toFlat(fit.correlation)[0]).toBeGreaterThan(0.95)
    expect(item(fit.error)).toBeLessThan(0.25)
    // the spreads are small but of the order of the error that is left (EP's spreads in a loopy graph are the narrow ones)
    const sd = toFlat(post.stddev('x'))
    const meanSd = sd.reduce((a, b) => a + b, 0) / sd.length
    expect(meanSd).toBeGreaterThan(0.05)
    expect(meanSd).toBeLessThan(1)
  }, 60_000)
})
