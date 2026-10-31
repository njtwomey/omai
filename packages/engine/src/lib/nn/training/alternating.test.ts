import { describe, expect, it } from 'vitest'
import { gaussianMixtureDensity } from '../../datasets/index.ts'
import { adam, linearSumAssignment, sgd } from '../../optim/index.ts'
import { key, normal } from '../../random/index.ts'
import { add, square, sub, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../../trace/index.ts'
import { ReLU } from '../module/activations.ts'
import { mlp, Sequential } from '../module/containers.ts'
import { Linear } from '../module/layers.ts'
import type { Params } from '../module/module.ts'
import { SpectralNorm } from '../module/parametrisation.ts'
import { type AlternatingState, alternatingAlgorithm, ganAlgorithm, modeCoverage } from './alternating.ts'

const flat = (v: unknown): number[] => toFlat(v as Value)

/** The exact Wasserstein-1 distance between two equal-size samples: the mean cost of the best matching. */
function w1(a: Tensor, b: Tensor): number {
  const n = a.shape[0] ?? 0
  const cost = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      Math.hypot(
        (a.data[2 * i] as number) - (b.data[2 * j] as number),
        (a.data[2 * i + 1] as number) - (b.data[2 * j + 1] as number),
      ),
    ),
  )
  return linearSumAssignment(cost).cost / n
}

describe('alternatingAlgorithm', () => {
  // Two players with their own quadratic losses that each depend on the other's parameter: the equilibrium of
  // x -> (x - y)^2 + (x - 1)^2 and y -> (y - x)^2 + (y + 1)^2 is x = 1/3, y = -1/3.
  const make = (xSteps: number, ySteps: number, frozenY = false) => {
    const px = new Linear(1, 1, { bias: false, key: key(0) })
    const py = new Linear(1, 1, { bias: false, key: key(1) })
    if (frozenY) py.requiresGrad(false)
    const one = tensor([[1]])
    const w = (p: Params): Value => p['weight'] as Value
    const alg = alternatingAlgorithm({
      players: [
        {
          name: 'x',
          module: px,
          optimizer: sgd({ learningRate: 0.2 }),
          loss: (own, all) => add(square(sub(w(own), w(all['y'] as Params))), square(sub(w(own), one))),
        },
        {
          name: 'y',
          module: py,
          optimizer: sgd({ learningRate: 0.2 }),
          loss: (own, all) => add(square(sub(w(own), w(all['x'] as Params))), square(add(w(own), one))),
        },
      ],
      schedule: [
        { player: 'x', times: xSteps },
        { player: 'y', times: ySteps },
      ],
      batch: () => 0,
      rounds: 200,
    })
    return { alg, px, py }
  }
  it('reaches the equilibrium of two coupled players', () => {
    const { alg } = make(1, 1)
    const r = runAlgorithm(alg, { maxSteps: 200, seed: 0 })
    expect(flat((r.state.params['x'] as Params)['weight'])[0]).toBeCloseTo(1 / 3, 6)
    expect(flat((r.state.params['y'] as Params)['weight'])[0]).toBeCloseTo(-1 / 3, 6)
  })

  it('counts updates by the schedule and logs both players', () => {
    const { alg } = make(3, 1)
    const rec = recorder()
    const r = runAlgorithm(alg, { maxSteps: 4, seed: 0, tracer: rec })
    expect(r.state.updates).toEqual({ x: 12, y: 4 })
    expect(metricSeries(rec.run(), 'player/x/loss').values).toHaveLength(4)
    expect(metricSeries(rec.run(), 'schedule/round').values).toEqual([1, 2, 3, 4])
  })

  it('leaves a frozen player where it is', () => {
    const { alg, py } = make(1, 1, true)
    const r = runAlgorithm(alg, { maxSteps: 20, seed: 0 })
    expect(flat((r.state.params['y'] as Params)['weight'])).toEqual(flat(py.params['weight']))
    expect(flat((r.state.params['x'] as Params)['weight'])[0]).not.toBeCloseTo(1 / 3, 3)
  })

  it('is deterministic in its key and its state serialises', () => {
    const run = (): AlternatingState => runAlgorithm(make(1, 1).alg, { maxSteps: 5, seed: 4 }).state
    expect(flat((run().params['x'] as Params)['weight'])).toEqual(flat((run().params['x'] as Params)['weight']))
    const { alg } = make(1, 1)
    const s = run()
    const serialise = alg.serialise
    const deserialise = alg.deserialise
    if (serialise === undefined || deserialise === undefined) throw new Error('the algorithm should serialise')
    expect(deserialise(serialise(s)).round).toBe(5)
    expect(() =>
      alternatingAlgorithm({ players: [], schedule: [{ player: 'nobody' }], batch: () => 0, rounds: 1 }),
    ).toThrow(/no player/)
  })
})

describe('GAN on a two-dimensional mixture', () => {
  const density = gaussianMixtureDensity({ layout: 'circle', count: 4, radius: 2, sigma: 0.1 })
  const real = (n: number, k: ReturnType<typeof key>): Tensor => density.sample(k, n).x
  const probe = normal(key(77), [400, 2]) as Tensor

  it('a non-saturating GAN covers all four modes, the discriminator is fooled and the Wasserstein distance falls', {
    timeout: 60000,
  }, () => {
    const generator = mlp([2, 24, 24, 2], { key: key(0) })
    const alg = ganAlgorithm({
      generator,
      discriminator: mlp([2, 24, 24, 1], { key: key(1) }),
      generatorOptimizer: adam({ learningRate: 2e-3, b1: 0.5, b2: 0.9 }),
      discriminatorOptimizer: adam({ learningRate: 2e-3, b1: 0.5, b2: 0.9 }),
      noiseDim: 2,
      sampleReal: real,
      batchSize: 64,
      rounds: 1500,
      objective: 'non-saturating',
      trackEvery: 10,
    })
    const rec = recorder()
    const r = runAlgorithm(alg, { maxSteps: 1500, seed: 0, tracer: rec })
    const before = generator.apply(generator.params, probe) as Tensor
    const after = generator.apply(r.state.params['generator'] as Params, probe) as Tensor
    const coverage = modeCoverage(after, density.means, 0.4)
    expect(modeCoverage(before, density.means, 0.4).covered).toBe(0)
    expect(coverage.covered).toBe(4)
    expect(coverage.highQuality).toBeGreaterThan(0.8)
    const reference = real(400, key(5))
    expect(w1(after, reference)).toBeLessThan(0.5 * w1(before, reference))
    const accuracy = metricSeries(rec.run(), 'gan/discriminatorAccuracy').values
    expect(accuracy).toHaveLength(150)
    expect(accuracy.slice(-10).reduce((a, b) => a + b, 0) / 10).toBeLessThan(0.8)
    expect(r.state.updates).toEqual({ discriminator: 1500, generator: 1500 })
  })

  it('with the generator frozen, the gradient-penalty critic estimates the exact Wasserstein distance', {
    timeout: 60000,
  }, () => {
    const generator = mlp([2, 16, 2], { key: key(3) })
    generator.requiresGrad(false)
    const rounds = 250
    const alg = ganAlgorithm({
      generator,
      discriminator: mlp([2, 32, 32, 1], { key: key(4) }),
      generatorOptimizer: adam({ learningRate: 1e-3 }),
      discriminatorOptimizer: adam({ learningRate: 2e-3, b1: 0.5, b2: 0.9 }),
      noiseDim: 2,
      sampleReal: real,
      batchSize: 64,
      rounds,
      evalBatch: 400,
      nCritic: 3,
      trackEvery: 5,
    })
    const rec = recorder()
    const r = runAlgorithm(alg, { maxSteps: rounds, seed: 1, tracer: rec })
    const fake = generator.apply(generator.params, normal(key(9), [400, 2])) as Tensor
    const exact = w1(fake, real(400, key(10)))
    const estimate = metricSeries(rec.run(), 'gan/wasserstein').values
    const final = estimate.slice(-10).reduce((a, b) => a + b, 0) / 10
    // The penalty keeps the critic near 1-Lipschitz (softly, so it can overshoot a little): its estimate is the distance to within a quarter.
    expect(final).toBeLessThan(exact * 1.25)
    expect(final).toBeGreaterThan(exact * 0.75)
    expect(estimate[0] as number).toBeLessThan(final)
    expect(metricSeries(rec.run(), 'gan/penalty').values.every((v) => Number.isFinite(v) && v >= 0)).toBe(true)
    const trained = (r.state.params['generator'] as Params)['0'] as Params
    expect(flat(trained['weight'])).toEqual(flat((generator.params['0'] as Params)['weight']))
  })

  it('carries a spectrally normalised discriminator’s power-iteration buffers through the rounds', {
    timeout: 60000,
  }, () => {
    const d = new Sequential(
      new SpectralNorm(new Linear(2, 16, { key: key(1) }), { key: key(2) }),
      new ReLU(),
      new SpectralNorm(new Linear(16, 1, { key: key(3) }), { key: key(4) }),
    )
    const alg = ganAlgorithm({
      generator: mlp([2, 16, 2], { key: key(0) }),
      discriminator: d,
      generatorOptimizer: adam({ learningRate: 1e-3 }),
      discriminatorOptimizer: adam({ learningRate: 1e-3 }),
      noiseDim: 2,
      sampleReal: real,
      batchSize: 32,
      rounds: 20,
      objective: 'hinge',
      trackEvery: 20,
    })
    const r = runAlgorithm(alg, { maxSteps: 20, seed: 0 })
    const before = flat((d.buffers['0'] as Params)['weight_u'])
    const after = flat(((r.state.buffers['discriminator'] as Params)['0'] as Params)['weight_u'])
    expect(after).not.toEqual(before)
    expect(Math.hypot(...after)).toBeCloseTo(1, 10)
    // Both layers' weights, divided by their estimated spectral norms, are what the discriminator applies.
    expect(Number.isFinite(r.state.losses['discriminator'])).toBe(true)
  })

  it('modeCoverage counts samples near modes and modes with enough of them', () => {
    const c = modeCoverage(
      [
        [0, 0],
        [0.1, 0],
        [5, 5],
        [20, 20],
      ],
      [
        [0, 0],
        [5, 5],
        [-5, -5],
      ],
      0.5,
    )
    expect(c.counts).toEqual([2, 1, 0])
    expect(c.covered).toBe(2)
    expect(c.highQuality).toBe(0.75)
  })
})
