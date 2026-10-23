import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../../testing/golden.ts'
import { xy } from '../../data/index.ts'
import { mse } from '../../losses/index.ts'
import { sgd } from '../../optim/index.ts'
import { key, normal } from '../../random/index.ts'
import { item, square, sum, type Tensor, toFlat, type Value } from '../../tensor/index.ts'
import { metricSeries, recorder, tensorSeries, tracePolicy } from '../../trace/index.ts'
import * as NN from '../index.ts'

const loss = (p: Value, t: Value): Value => mse(p, t)
const xs = Array.from({ length: 8 }, (_, i) => [i / 4 - 1, (i % 3) / 2])
const ys = xs.map(([a = 0, b = 0]) => [2 * a - b + 0.5])
const data = xy(xs, ys)

describe('training state', () => {
  it('a functional train step matches one fit step', () => {
    const opt = sgd({ learningRate: 0.1 })
    const a = new NN.Linear(2, 1, { key: key(0) })
    const s = NN.trainStep(a, NN.trainState(a, opt), data.x, (y) => mse(y, data.y), opt)
    const b = new NN.Linear(2, 1, { key: key(0) })
    NN.fit(b, data, { loss, optimizer: opt, epochs: 1 })
    expect(toFlat(s.params['weight'] as Value)).toEqual(toFlat(b.params['weight'] as Value))
    expect(s.updates).toBe(1)
  })

  it('fit carries batch-norm statistics and loads them into the model', () => {
    const net = new NN.Sequential(
      new NN.Linear(2, 3, { key: key(1) }),
      new NN.BatchNorm1d(3),
      new NN.Linear(3, 1, { key: key(2) }),
    )
    const r = NN.fit(net, data, { loss, optimizer: sgd({ learningRate: 0.05 }), epochs: 3, batchSize: 4, key: key(3) })
    expect(item(net.stateDict()['1.num_batches_tracked'] as Tensor)).toBe(6)
    expect(toFlat(net.stateDict()['1.running_mean'] as Tensor)).not.toEqual([0, 0, 0])
    expect(Object.keys(r.buffers)).toEqual(['1'])
  })

  it('accumulating two half batches equals one full batch with plain SGD', () => {
    const a = new NN.Linear(2, 1, { key: key(4) })
    const b = new NN.Linear(2, 1, { key: key(4) })
    NN.fit(a, data, { loss, optimizer: sgd({ learningRate: 0.1 }), epochs: 2 })
    // Unshuffled halves: give batchSize without shuffling by using the same order (batches come from epochBatches).
    const halves = NN.fitAlgorithm(b, data, {
      loss,
      optimizer: sgd({ learningRate: 0.1 }),
      epochs: 2,
      batchSize: 4,
      accumulateGradients: 2,
    })
    expect(halves.name).toBe('fit')
    const r = NN.fit(b, data, {
      loss,
      optimizer: sgd({ learningRate: 0.1 }),
      epochs: 2,
      batchSize: 4,
      accumulateGradients: 2,
    })
    expect(r.steps).toBe(4)
    for (const [i, v] of toFlat(a.params['weight'] as Value).entries())
      expect(v).toBeCloseTo(toFlat(b.params['weight'] as Value)[i] ?? 0, 12)
  })

  it('frozen parameters do not move', () => {
    const net = new NN.Sequential(
      new NN.Linear(2, 3, { key: key(5) }),
      new NN.Tanh(),
      new NN.Linear(3, 1, { key: key(6) }),
    )
    ;(net.layers[0] as NN.Module).requiresGrad(false)
    const before = toFlat(net.stateDict()['0.weight'] as Tensor)
    const head = toFlat(net.stateDict()['2.weight'] as Tensor)
    NN.fit(net, data, { loss, optimizer: sgd({ learningRate: 0.1 }), epochs: 5 })
    expect(toFlat(net.stateDict()['0.weight'] as Tensor)).toEqual(before)
    expect(toFlat(net.stateDict()['2.weight'] as Tensor)).not.toEqual(head)
  })

  it('reduces the learning rate on a plateau and records it', () => {
    const plateau = NN.reduceLROnPlateau({ monitor: 'loss', patience: 0, factor: 0.5, threshold: 10 })
    const rec = recorder()
    NN.fit(new NN.Linear(2, 1, { key: key(7) }), data, {
      loss,
      optimizer: sgd({ learningRate: 0.1 }),
      learningRate: 0.1,
      epochs: 4,
      callbacks: [plateau],
      tracer: rec,
    })
    expect(plateau.scale()).toBeLessThan(1)
    const lr = metricSeries(rec.run(), 'lr').values
    expect(lr[0]).toBe(0.1)
    expect(lr.at(-1)).toBeLessThan(0.1)
    expect(metricSeries(rec.run(), 'lr_scale').values.length).toBeGreaterThan(0)
  })

  it('records per-layer norms, gradients and activations by policy', () => {
    const rec = recorder()
    const probe = new NN.Lambda((v, ctx) => NN.tap(ctx, 'hidden', v))
    const net = new NN.Sequential(new NN.Linear(2, 3, { key: key(8) }), probe, new NN.Linear(3, 1, { key: key(9) }))
    NN.fit(net, data, {
      loss,
      optimizer: sgd({ learningRate: 0.1 }),
      epochs: 2,
      tracer: rec,
      policy: tracePolicy({ tensors: true, roles: ['grad', 'param', 'activation'] }),
      callbacks: [NN.activationLogger({ norms: true })],
    })
    const run = rec.run()
    expect(metricSeries(run, 'gradNorm/0.weight').values.length).toBe(2)
    expect(metricSeries(run, 'paramNorm/2.bias').values.length).toBe(2)
    expect(metricSeries(run, 'activation_norm/1.hidden').values.length).toBe(2)
    expect(tensorSeries(run, 'activations.1.hidden').length).toBe(2)
  })
})

describe('serialisation', () => {
  it('round-trips parameters and buffers through JSON text', () => {
    const a = new NN.Sequential(new NN.Linear(2, 2, { key: key(0) }), new NN.BatchNorm1d(2))
    a.forward(normal(key(1), [4, 2]))
    const b = NN.stateFromText(
      new NN.Sequential(new NN.Linear(2, 2, { key: key(5) }), new NN.BatchNorm1d(2)),
      NN.stateToText(a),
    )
    expect(b.stateDict()).toEqual(a.stateDict())
    expect(() => NN.loadState(b, { format: 'other' })).toThrow()
  })

  it('loads weights exported by the oracle', () => {
    const c = loadFixture('nn/torch_reference').cases.find((k) => k.fn === 'mlpForward')
    const state = c?.params['state'] as Record<string, unknown>
    const net = new NN.Sequential(
      new NN.Linear(3, 5),
      new NN.Tanh(),
      new NN.Linear(5, 4),
      new NN.ReLU(),
      new NN.Linear(4, 2),
    )
    NN.loadTorchStateDict(net, state)
    const want = decode(c?.output) as Tensor
    const got = toFlat(net.forward(decode(c?.inputs[0]) as Tensor))
    for (const [i, v] of toFlat(want).entries()) expect(got[i]).toBeCloseTo(v, 12)
    expect(() => NN.loadTorchStateDict(net, { weight: 3 })).toThrow()
    expect(item(sum(square(net.forward([[0, 0, 0]])))) >= 0).toBe(true)
  })
})
