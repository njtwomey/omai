import { describe, expect, it } from 'vitest'
import { xy } from '../../data/index.ts'
import { mse } from '../../losses/index.ts'
import { sgd } from '../../optim/index.ts'
import { key } from '../../random/index.ts'
import { metricSeries, recorder, summary, tensorSeries, tracePolicy } from '../../trace/index.ts'
import * as NN from '../index.ts'

const data = xy([[0], [1], [2], [3]], [[1], [3], [5], [7]])
const loss = (p: Parameters<typeof mse>[0], t: Parameters<typeof mse>[1]) => mse(p, t)

describe('training hooks', () => {
  it('run in Lightning order', () => {
    const calls: string[] = []
    const spy: NN.Callback = {
      name: 'spy',
      onTrainStart: () => calls.push('trainStart'),
      onEpochStart: (c) => calls.push(`epochStart ${c.epoch}`),
      onBatchStart: (c) => calls.push(`batchStart ${c.batch}`),
      onAfterBackward: () => calls.push('afterBackward'),
      onBeforeOptimizerStep: () => calls.push('beforeOptimizerStep'),
      onBatchEnd: () => calls.push('batchEnd'),
      onEpochEnd: (c) => calls.push(`epochEnd ${c.epoch}`),
      onValidationEnd: () => calls.push('validationEnd'),
      onTrainEnd: (c) => calls.push(`trainEnd ${c.status}`),
    }
    NN.fit(new NN.Linear(1, 1, { key: key(0) }), data, {
      loss,
      optimizer: sgd({ learningRate: 0.01 }),
      epochs: 1,
      batchSize: 2,
      callbacks: [spy],
      validation: { data },
    })
    expect(calls).toEqual([
      'trainStart',
      'epochStart 0',
      'batchStart 0',
      'afterBackward',
      'beforeOptimizerStep',
      'batchEnd',
      'batchStart 1',
      'afterBackward',
      'beforeOptimizerStep',
      'batchEnd',
      'epochEnd 0',
      'validationEnd',
      'trainEnd completed',
    ])
  })

  it('log gradient and parameter norms, values and the learning rate into the trace', () => {
    const rec = recorder()
    NN.fit(NN.mlp([1, 3, 1], { key: key(1) }), data, {
      loss,
      optimizer: sgd({ learningRate: 0.01 }),
      epochs: 3,
      callbacks: [NN.gradNormLogger(), NN.paramNormLogger(), NN.valueLogger(), NN.learningRateMonitor(0.01)],
      tracer: rec,
      policy: tracePolicy({ tensors: true, roles: ['grad', 'param'], detail: 'full' }),
    })
    const run = rec.run()
    const total = metricSeries(run, 'grad_norm/total').values
    expect(total).toHaveLength(3)
    // The total is the norm of the per-parameter norms.
    const parts = ['0.bias', '0.weight', '2.bias', '2.weight'].map(
      (p) => metricSeries(run, `grad_norm/${p}`).values[0] ?? 0,
    )
    expect(total[0]).toBeCloseTo(Math.sqrt(parts.reduce((s, v) => s + v * v, 0)), 12)
    expect(metricSeries(run, 'param_norm/total').values).toHaveLength(3)
    expect(metricSeries(run, 'lr').values).toEqual([0.01, 0.01, 0.01])
    expect(tensorSeries(run, 'grad/0.weight')[0]?.values).toHaveLength(3)
  })

  it('stop early when validation stops improving, keeping the best checkpoint', () => {
    const best = NN.bestCheckpoint()
    const stopper = NN.earlyStopping({ patience: 2 })
    // A learning rate this large makes the loss grow, so validation never improves after the first epoch.
    const rec = recorder()
    const r = NN.fit(new NN.Linear(1, 1, { key: key(2) }), data, {
      loss,
      optimizer: sgd({ learningRate: 0.5 }),
      epochs: 50,
      callbacks: [best, stopper],
      validation: { data },
      tracer: rec,
    })
    expect(r.status).toBe('stopped')
    expect(r.steps).toBe(3)
    expect(best.best().epoch).toBe(0)
    expect(best.best().value).toBe(stopper.best())
    expect(metricSeries(rec.run(), 'valLoss').values).toHaveLength(3)
    expect(Object.values(summary(rec.run()))[0]?.status).toBe('stopped')
  })

  it('validate with extra metrics', () => {
    const rec = recorder()
    NN.fit(new NN.Linear(1, 1, { key: key(3) }), data, {
      loss,
      optimizer: sgd({ learningRate: 0.05 }),
      epochs: 4,
      validation: {
        data,
        every: 2,
        metrics: {
          mae: (p, t) =>
            Math.abs(
              Number((p as never as { data: Float64Array }).data[0]) -
                Number((t as never as { data: Float64Array }).data[0]),
            ),
        },
      },
      tracer: rec,
    })
    expect(metricSeries(rec.run(), 'valLoss').steps).toEqual([2, 4])
    expect(metricSeries(rec.run(), 'val_mae').values).toHaveLength(2)
  })
})
