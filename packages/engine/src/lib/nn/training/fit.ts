import { valueAndGrad } from '../../autodiff/index.ts'
import type { Json } from '../../base/index.ts'
import { epochBatches, rowCount, type TabularData } from '../../data/index.ts'
import { applyUpdates, clipByGlobalNorm, type GradientTransformation, globalNorm } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { item, treePaths, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import type { Module, Params } from '../module/module.ts'

/** The state of a training run. */
export interface TrainState {
  readonly params: Params
  readonly optState: unknown
  /** Updates applied so far. */
  readonly step: number
  readonly epoch: number
  /** The loss and gradient norm of the last minibatch. */
  readonly loss: number
  readonly gradNorm: number
}

/** Options of {@link fitAlgorithm} and {@link fit}. */
export interface FitOptions {
  /** The training objective of predictions and targets (for example `mse` or `crossEntropy`). */
  readonly loss: (prediction: Value, target: Value) => Value
  readonly optimizer: GradientTransformation
  readonly epochs: number
  /** Minibatch size (default: the whole dataset). */
  readonly batchSize?: number
  /** Shuffling and dropout keys derive from this (default key(0)). */
  readonly key?: Key
  /** Clip the gradient's global norm before the optimiser. */
  readonly clipNorm?: number
}

/**
 * Minibatch training of a module as an {@link Algorithm}: each step evaluates the loss on one minibatch, differentiates
 * it with respect to the parameters and applies the optimiser; epochs reshuffle with their own keys. Records the loss
 * and gradient norm each step and, by policy, the parameters and gradients by layer.
 *
 * @param model - The module (its current parameters are the starting point; it is not modified).
 * @param data - The training data.
 * @param options - Loss, optimiser, epochs, batch size, key and gradient clipping.
 * @returns The algorithm.
 * @concept training
 * @example
 * ```ts
 * import { fitAlgorithm, mlp } from 'whitebox-engine/nn'
 * import { sgd } from 'whitebox-engine/optim'
 * import { mse } from 'whitebox-engine/losses'
 * import { xy } from 'whitebox-engine/data'
 * fitAlgorithm(mlp([1, 1]), xy([[1]], [2]), { loss: (p, t) => mse(p, t), optimizer: sgd({ learningRate: 0.1 }), epochs: 1 }).name // => 'fit'
 * ```
 */
export function fitAlgorithm(model: Module, data: TabularData, options: FitOptions): Algorithm<TrainState> {
  const n = rowCount(data)
  const batchSize = options.batchSize ?? n
  const perEpoch = Math.ceil(n / batchSize)
  const base = options.key ?? makeKey(0)
  const shuffleKey = foldIn(base, 0)
  const optimizer = options.clipNorm === undefined ? options.optimizer : chainClip(options.clipNorm, options.optimizer)
  let cached: { epoch: number; batches: TabularData[] } | undefined
  const batchAt = (epoch: number, index: number): TabularData => {
    if (cached?.epoch !== epoch) {
      cached = {
        epoch,
        batches: epochBatches(data, epoch, {
          batchSize,
          ...(options.batchSize === undefined ? {} : { key: shuffleKey }),
        }),
      }
    }
    return cached.batches[index] as TabularData
  }
  return {
    name: 'fit',
    config: { model: model.kind, optimizer: optimizer.name, epochs: options.epochs, batchSize } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'minibatch loss' },
        gradNorm: { description: 'global gradient norm' },
        epoch: { description: 'epoch' },
      },
      tensors: {
        params: { role: 'param', description: 'parameters by layer' },
        grad: { role: 'grad', description: 'gradients by layer' },
      },
    }),
    init: () => ({
      params: model.params,
      optState: optimizer.init(model.params),
      step: 0,
      epoch: 0,
      loss: Number.NaN,
      gradNorm: Number.NaN,
    }),
    step: (s, ctx) => {
      const epoch = Math.floor(s.step / perEpoch)
      const batch = batchAt(epoch, s.step % perEpoch)
      const dropoutKey = foldIn(foldIn(base, 1), s.step)
      const objective = (p: Params): Value =>
        options.loss(model.apply(p, batch.x, { train: true, key: dropoutKey }), batch.y)
      const { value, grad } = valueAndGrad(objective)(s.params)
      const { updates, state } = optimizer.update(grad, s.optState, s.params)
      if (ctx.run.wantsTensor('grad', 'grad')) {
        for (const [path, g] of Object.entries(treePaths(grad))) ctx.run.tensor(s.step, `grad.${path}`, 'grad', g)
      }
      return {
        params: applyUpdates(s.params, updates) as Params,
        optState: state,
        step: s.step + 1,
        epoch: Math.floor((s.step + 1) / perEpoch),
        loss: item(value),
        gradNorm: item(globalNorm(grad)),
      }
    },
    done: (s) =>
      s.step >= options.epochs * perEpoch
        ? 'completed'
        : !Number.isNaN(s.loss) && !Number.isFinite(s.loss)
          ? 'diverged'
          : undefined,
    observe: (s) => {
      const tensors: Record<string, { role: 'param'; value: Value }> = {}
      for (const [path, v] of Object.entries(treePaths(s.params)))
        tensors[`params.${path}`] = { role: 'param', value: v }
      return { metrics: Number.isNaN(s.loss) ? {} : { loss: s.loss, gradNorm: s.gradNorm, epoch: s.epoch }, tensors }
    },
  }
}

function chainClip(maxNorm: number, optimizer: GradientTransformation): GradientTransformation<unknown[]> {
  const clip = clipByGlobalNorm(maxNorm)
  return {
    name: `clip(${maxNorm}) + ${optimizer.name}`,
    init: (p) => [undefined, optimizer.init(p)],
    update: (g, state, p) => {
      const clipped = clip.update(g, undefined).updates
      const r = optimizer.update(clipped, state[1], p)
      return { updates: r.updates, state: [undefined, r.state] }
    },
  }
}

/** Result of {@link fit}. */
export interface FitResult {
  /** The trained parameters (also loaded into the returned model). */
  readonly params: Params
  /** The minibatch losses, one per step. */
  readonly losses: readonly number[]
  readonly steps: number
  readonly status: string
}

/**
 * Train a module on a dataset and load the trained parameters into it.
 *
 * @param model - The module (updated in place with the trained parameters).
 * @param data - The training data.
 * @param options - Loss, optimiser, epochs, batch size, key, clipping, and the run's tracer and policy.
 * @returns The trained parameters, per-step losses and the run status.
 * @concept training
 * @example
 * ```ts
 * import { fit, Linear } from 'whitebox-engine/nn'
 * import { sgd } from 'whitebox-engine/optim'
 * import { mse } from 'whitebox-engine/losses'
 * import { xy } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * const r = fit(new Linear(1, 1, { key: key(0) }), xy([[0], [1], [2]], [[1], [3], [5]]), { loss: (p, t) => mse(p, t), optimizer: sgd({ learningRate: 0.1 }), epochs: 500 })
 * (r.losses.at(-1) ?? 1) < 1e-8 // => true
 * ```
 */
export function fit(
  model: Module,
  data: TabularData,
  options: FitOptions & Partial<Omit<RunOptions, 'maxSteps'>>,
): FitResult {
  const { loss, optimizer, epochs, batchSize, key, clipNorm, ...run } = options
  const fitOptions: FitOptions = {
    loss,
    optimizer,
    epochs,
    ...(batchSize === undefined ? {} : { batchSize }),
    ...(key === undefined ? {} : { key }),
    ...(clipNorm === undefined ? {} : { clipNorm }),
  }
  const alg = fitAlgorithm(model, data, fitOptions)
  const losses: number[] = []
  const tracked: Algorithm<TrainState> = {
    ...alg,
    step: (s, ctx) => {
      const next = alg.step(s, ctx)
      losses.push(next.loss)
      return next
    },
  }
  const perEpoch = Math.ceil(rowCount(data) / (batchSize ?? rowCount(data)))
  const result = runAlgorithm(tracked, { ...run, maxSteps: epochs * perEpoch })
  model.setParams(result.state.params)
  return { params: result.state.params, losses, steps: result.steps, status: result.status }
}
