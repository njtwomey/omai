// A torch-like method on the contract: an MLP classifier on a two-dimensional dataset, one minibatch per step, logging
// its loss, its parameters and activations under paths, firing an epoch event, and offering its decision surface and
// accuracy as derived paths. The data and the network are rebuilt from the same key wherever they are needed.
import { decisionSurface } from 'mdml-engine/exposition'
import { crossEntropy } from 'mdml-engine/losses'
import {
  activationModule,
  applyGradients,
  Linear,
  lossAndGrads,
  Sequential,
  type TrainState,
  trainState,
} from 'mdml-engine/nn'
import { adam } from 'mdml-engine/optim'
import { foldIn, type Key, permutation } from 'mdml-engine/random'
import { choice, dict, int, real } from 'mdml-engine/space'
import { argmax, asTensor, softmax, toFlat } from 'mdml-engine/tensor'
import {
  type Algorithm,
  defineMethod,
  derived,
  event,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from 'mdml-engine/trace'
import { DATASETS, type DatasetName, gridOver, leaves, rowsOf, twoClassData } from './data.ts'

export type MlpConfig = {
  readonly dataset: DatasetName
  readonly n: number
  readonly hidden: number
  readonly lr: number
  readonly batch: number
}

export interface MlpState {
  readonly train: TrainState
  readonly order: readonly number[]
  readonly batch: number
  readonly epoch: number
  readonly lossSum: number
}

function built(c: MlpConfig, key: Key): { data: ReturnType<typeof twoClassData>; net: Sequential } {
  const data = twoClassData(c.dataset, c.n, foldIn(key, 0))
  const k = foldIn(key, 1)
  const net = new Sequential(
    new Linear(2, c.hidden, { key: foldIn(k, 0) }),
    activationModule('relu'),
    new Linear(c.hidden, c.hidden, { key: foldIn(k, 1) }),
    activationModule('relu'),
    new Linear(c.hidden, data.k, { key: foldIn(k, 2) }),
  )
  return { data, net }
}

export const mlpMethod: Method<MlpState, MlpConfig> = defineMethod<MlpState, MlpConfig>({
  name: 'poc/mlp',
  config: dict({
    dataset: choice([...DATASETS], { default: 'moons' }),
    n: int(50, 1000, { default: 200 }),
    hidden: int(2, 64, { default: 16 }),
    lr: real(1e-4, 0.1, { log: true, default: 0.01 }),
    batch: int(4, 200, { default: 32 }),
  }),
  make: (c, key) => {
    const { data, net } = built(c, key)
    const optimizer = adam({ learningRate: c.lr })
    const perEpoch = Math.ceil(c.n / c.batch)
    const alg: Algorithm<MlpState> = {
      name: 'poc/mlp',
      config: { ...c },
      init: (k) => ({
        train: trainState(net, optimizer, foldIn(k, 0)),
        order: toFlat(permutation(foldIn(k, 1), c.n)),
        batch: 0,
        epoch: 0,
        lossSum: 0,
      }),
      step: (s, ctx) => {
        const rows = s.order.slice(s.batch * c.batch, (s.batch + 1) * c.batch)
        const xb = asTensor(rows.map((i) => data.x[i] ?? [0, 0]))
        const yb = asTensor(
          rows.map((i) => data.y[i] ?? 0),
          'int32',
        )
        const r = lossAndGrads(net, s.train.params, s.train.buffers, xb, (out) => crossEntropy(out, yb), {
          key: foldIn(ctx.key, 0),
          taps: ctx.taps('model/activations'),
        })
        const applied = applyGradients(s.train, r.grads, optimizer).state
        ctx.log('training/loss', r.loss)
        for (const [name, p] of leaves(applied.params)) ctx.log(`model/params/${name}`, p)
        const train: TrainState = { ...applied, buffers: r.buffers, step: s.train.step + 1, loss: r.loss }
        const batch = s.batch + 1
        const lossSum = s.lossSum + r.loss
        if (batch < perEpoch) return { ...s, train, batch, lossSum }
        const meanLoss = lossSum / perEpoch
        ctx.log('training/epoch/loss', meanLoss)
        ctx.emit('training/epoch/end', { epoch: s.epoch, meanLoss })
        return {
          train,
          order: toFlat(permutation(foldIn(ctx.key, 1), c.n)),
          batch: 0,
          epoch: s.epoch + 1,
          lossSum: 0,
        }
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as MlpState,
    }
    return alg
  },
  contract: (c, key) => {
    const { data, net } = built(c, key)
    const predict = (s: MlpState, points: readonly (readonly number[])[]): number[] =>
      toFlat(argmax(net.apply(s.train.params, asTensor(points)), 1))
    return {
      'training/loss': scalar({ description: 'minibatch cross-entropy', direction: 'lower' }),
      'training/epoch/loss': scalar({ description: 'mean loss over the epoch', direction: 'lower' }),
      'training/epoch/end': event({ context: { epoch: 'the epoch finished', meanLoss: 'its mean loss' } }),
      'model/params/*': tensor({ role: 'param', description: 'a layer’s weights or biases' }),
      'model/activations/*': tensor({ role: 'activation', description: 'a layer’s output on the batch' }),
      'model/accuracy': derived({
        description: 'training accuracy',
        compute: (s) => {
          const p = predict(s, data.x)
          return p.filter((v, i) => v === data.y[i]).length / p.length
        },
      }),
      'model/decision-surface': derived({
        description: 'the class probabilities over the plane (the view draws them smoothly, boundary at 0.5)',
        cost: 'dear',
        compute: (s, args) => {
          const res = ((args as { readonly resolution?: number } | undefined)?.resolution ?? 40) | 0
          const grid = gridOver(data.bounds, res)
          // probabilities[c][i][j]: class c at (x[j], y[i]); the flat softmax is [points, k], points row by row.
          const p = toFlat(softmax(net.apply(s.train.params, asTensor(grid.points)), -1))
          const k = data.k
          const probabilities = Array.from({ length: k }, (_, c) =>
            rowsOf(
              Array.from({ length: grid.points.length }, (_, i) => p[i * k + c] ?? 0),
              res,
            ),
          )
          return decisionSurface({
            features: ['x', 'y'],
            x: grid.xs,
            y: grid.ys,
            labels: rowsOf(predict(s, grid.points), res),
            probabilities,
            rows: { points: data.x, features: ['x', 'y'], labels: data.y },
          })
        },
      }),
    }
  },
  defaults: { 'training/loss': Log(), 'training/epoch/loss': Log() },
  hooks: {
    stopBelow: (o) => (h) => {
      const target = (o as { readonly loss?: number } | undefined)?.loss ?? 0.1
      if ((h.context as { readonly meanLoss: number }).meanLoss < target) h.stop(`loss below ${target}`)
    },
  },
})
