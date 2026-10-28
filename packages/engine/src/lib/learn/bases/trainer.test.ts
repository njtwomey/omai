import { describe, expect, it } from 'vitest'
import { generate } from '../../data/index.ts'
import { moons } from '../../datasets/index.ts'
import { crossEntropy, mse } from '../../losses/index.ts'
import { Linear, mlp } from '../../nn/index.ts'
import { adam, sgd } from '../../optim/index.ts'
import { key } from '../../random/index.ts'
import { dict, real } from '../../space/index.ts'
import { type Tensor, toFlat } from '../../tensor/index.ts'
import { Log, messages, metricSeries, tensorSeries } from '../../trace/index.ts'
import type { LearnData } from '../estimator.ts'
import { defineTrainer } from './trainer.ts'

const draw = (n: number, seed: number): LearnData => {
  const d = generate(moons, { knobs: { n, noise: 0.1 }, key: key(seed) }).data
  return { x: d.x, y: d.y }
}
const flat = (t: Tensor | undefined): number[] => (t === undefined ? [] : toFlat(t))
const train = draw(120, 1)
const valid = draw(60, 2)

interface Hyper {
  readonly hidden: number
  readonly lr: number
  readonly unit: 'step' | 'epoch'
}

/** A tiny MLP classifier on 2-D data, with a cosine schedule and a validation set. */
const net = defineTrainer<Hyper>({
  name: 'moons-mlp',
  kind: 'classifier',
  defaults: { hidden: 16, lr: 0.05, unit: 'step' },
  hyperSpace: (h) => dict({ lr: real(1e-3, 1, { default: h.lr, log: true }) }),
  model: (_data, h, k) => mlp([2, h.hidden, 2], { key: k }),
  objective: () => (p, t) => crossEntropy(p, t),
  optimiser: (lr) => adam({ learningRate: lr }),
  settings: (h) => ({
    learningRate: h.lr,
    schedule: { kind: 'cosine', steps: 160, endRatio: 0.1 },
    epochs: 40,
    batchSize: 30,
    unit: h.unit,
  }),
  validation: () => valid,
})

describe('the trainer base', () => {
  it('trains a tiny MLP on moons and logs the standard paths', () => {
    const fitted = net.fit(train, { key: key(3) })
    const accuracy = fitted.score?.(valid) ?? 0
    expect(accuracy).toBeGreaterThan(0.9)
    expect(fitted.classes).toEqual([0, 1])
    const loss = metricSeries(fitted.run, 'train/loss')
    expect(loss.values).toHaveLength(160)
    expect(loss.values.at(-1)).toBeLessThan(loss.values[0] ?? 0)
    // The cosine schedule decays from its peak to a tenth of it.
    const lr = metricSeries(fitted.run, 'train/lr').values
    expect(lr[0]).toBeCloseTo(0.05, 12)
    expect(lr.at(-1)).toBeLessThan(0.01)
    // Validation at each of the 40 epoch ends (4 minibatches an epoch), and no more.
    const acc = metricSeries(fitted.run, 'eval/accuracy')
    expect(acc.steps.slice(0, 3)).toEqual([3, 7, 11])
    expect(acc.values.at(-1)).toBeGreaterThan(0.9)
    expect(metricSeries(fitted.run, 'eval/loss').values).toHaveLength(40)
    // The training curves are the default record.
    expect(fitted.exposition.map((r) => r.kind)).toEqual(['curves'])
  })

  it('keeps layer summaries, activations and flattened parameters when asked, and nothing otherwise', () => {
    expect(tensorSeries(net.fit(train, { key: key(3) }).run, 'layer/0/weights')).toEqual([])
    const fitted = net.fit(train, {
      key: key(3),
      logs: {
        'layer/0/weights': Log({ every: 40 }),
        'layer/2/params': Log({ every: 40, detail: 'full' }),
        'layer/1/activations': Log({ every: 40 }),
      },
    })
    const weights = tensorSeries(fitted.run, 'layer/0/weights')
    expect(weights.map((e) => e.step)).toEqual([0, 40, 80, 120])
    expect(weights[0]?.summary).toBeDefined()
    expect(weights[0]?.values).toBeUndefined()
    const params = tensorSeries(fitted.run, 'layer/2/params')
    expect(params[0]?.values).toHaveLength(16 * 2 + 2)
    expect(tensorSeries(fitted.run, 'layer/1/activations')[0]?.summary).toBeDefined()
    // The ReLU between the linear layers has activations but no weights.
    expect(Object.keys(net.session(train).contract)).toContain('layer/1/activations')
    expect(Object.keys(net.session(train).contract)).not.toContain('layer/1/weights')
  })

  it('checkpoints the parameters and optimiser state so any step can be probed, seeked and forked', () => {
    const s = net.session(train, { key: key(3), mode: 'live' })
    s.advance({ steps: 50 })
    expect(s.t).toBe(50)
    const early = s.fittedAt(10)
    const late = s.fittedAt(50)
    expect(flat(early.predictProba?.(valid.x)).length).toBe(120)
    expect(flat(early.decisionFunction?.(valid.x))).not.toEqual(flat(late.decisionFunction?.(valid.x)))
    // A seek re-runs from the nearest checkpoint with the original keys: the same parameters as the live run had.
    const params = (t: number): number[] =>
      Object.values(s.stateAt(t).params).flatMap((l) => toFlat((l as { weight: never }).weight))
    expect(params(50)).toEqual(Object.values(s.state.params).flatMap((l) => toFlat((l as { weight: never }).weight)))
    // Continuing a fork from step 20 reproduces the straight run, optimiser moments included.
    const fork = s.fork(20)
    fork.advance({ steps: 30 })
    const weightsOf = (state: typeof s.state): number[] =>
      Object.values(state.params).flatMap((l) => toFlat((l as { weight: never }).weight))
    expect(weightsOf(fork.state)).toEqual(weightsOf(s.state))
    const surface = s.probe(30, 'model/decision-surface', { resolution: 5 }) as { readonly kind: string }
    expect(surface.kind).toBe('decisionSurface')
    expect(Object.keys(s.probe(30, 'model/params') as object)).toEqual(['params', 'buffers'])
    s.advance()
    expect(s.status).toBe('completed')
    expect(s.fittedAt(s.t).score?.(valid)).toBeGreaterThan(0.9)
    expect(messages(s.run(), 'train/epoch/end')).toEqual([])
    expect(s.stepsOf('train/epoch/end')).toHaveLength(40)
  })

  it('runs an epoch to a step, and revives a fitted network from its parameters', () => {
    const byEpoch = net.with({ unit: 'epoch' })
    const fitted = byEpoch.fit(train, { key: key(3) })
    expect(metricSeries(fitted.run, 'train/loss').values).toHaveLength(40)
    expect(fitted.score?.(valid)).toBeGreaterThan(0.85)
    const revived = byEpoch.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(flat(revived.predict?.(valid.x))).toEqual(flat(fitted.predict?.(valid.x)))
    expect(net.with({ lr: 0.1 }).hyper.lr).toBe(0.1)
  })

  it('draws a stream into the rows its budget needs, and fits a regressor without validation', () => {
    const line = defineTrainer({
      name: 'line',
      kind: 'regressor',
      defaults: {},
      model: (_d, _h, k) => new Linear(1, 1, { key: k }),
      objective: () => (p, t) => mse(p, t),
      optimiser: (lr) => sgd({ learningRate: lr }),
      settings: () => ({ learningRate: 0.2, steps: 200, batchSize: 16 }),
      source: () => ({
        draw: (n) => ({
          x: Array.from({ length: n }, (_, i) => [(i % 17) / 8 - 1]),
          y: Array.from({ length: n }, (_, i) => [3 * ((i % 17) / 8 - 1) + 1]),
        }),
      }),
    })
    const fitted = line.fit({ x: [[0]], y: [[0]] }, { key: key(1) })
    expect(flat(fitted.predict?.([[0.5]]))[0]).toBeCloseTo(2.5, 2)
    expect(metricSeries(fitted.run, 'eval/loss').values).toEqual([])
  })
})
