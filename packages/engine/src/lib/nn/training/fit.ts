import { valueAndGrad } from '../../autodiff/index.ts'
import type { Json } from '../../base/index.ts'
import { epochBatches, rowCount, type TabularData } from '../../data/index.ts'
import { applyUpdates, clipByGlobalNorm, type GradientTransformation, globalNorm } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { item, treePaths, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import type { Module, Params } from '../module/module.ts'
import type { Callback, HookContext } from './callbacks.ts'

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
  /** The sum of this epoch's minibatch losses so far (for the epoch mean). */
  readonly epochLossSum: number
  /** Set when a callback asked to stop. */
  readonly stopped?: string
}

/** Validation during training. */
export interface ValidationOptions {
  readonly data: TabularData
  /** Validate every this many epochs (default 1). */
  readonly every?: number
  /** Further metrics of predictions and targets, logged as `val_<name>`. */
  readonly metrics?: Readonly<Record<string, (prediction: Value, target: Value) => number>>
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
  /** Hooks called in Lightning's order (logging, early stopping, checkpoints). */
  readonly callbacks?: readonly Callback[]
  readonly validation?: ValidationOptions
}

/**
 * Minibatch training of a module as an {@link Algorithm}: each step evaluates the loss on one minibatch, differentiates
 * it with respect to the parameters and applies the optimiser; epochs reshuffle with their own keys. Records the loss
 * and gradient norm each step, the epoch's mean loss and validation metrics at the end of each epoch, and calls the
 * callbacks' hooks in Lightning's order: train start, epoch start, batch start, after backward, before optimiser step,
 * batch end, epoch end, validation end, train end.
 *
 * @param model - The module (its current parameters are the starting point; it is not modified).
 * @param data - The training data.
 * @param options - Loss, optimiser, epochs, batch size, key, clipping, callbacks and validation.
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
  const total = options.epochs * perEpoch
  const base = options.key ?? makeKey(0)
  const shuffleKey = foldIn(base, 0)
  const optimizer = options.clipNorm === undefined ? options.optimizer : chainClip(options.clipNorm, options.optimizer)
  const callbacks = options.callbacks ?? []
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
  const validate = (params: Params): Record<string, number> => {
    const v = options.validation
    if (v === undefined) return {}
    const prediction = model.apply(params, v.data.x, { train: false })
    const metrics: Record<string, number> = { valLoss: item(options.loss(prediction, v.data.y)) }
    for (const [name, f] of Object.entries(v.metrics ?? {})) metrics[`val_${name}`] = f(prediction, v.data.y)
    return metrics
  }
  return {
    name: 'fit',
    config: {
      model: model.kind,
      optimizer: optimizer.name,
      epochs: options.epochs,
      batchSize,
      callbacks: callbacks.map((c) => c.name),
    } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'minibatch loss' },
        gradNorm: { description: 'global gradient norm' },
        epochLoss: { description: 'mean minibatch loss of the epoch' },
        valLoss: { description: 'validation loss' },
      },
      tensors: { params: { role: 'param', description: 'parameters by layer' } },
    }),
    init: () => ({
      params: model.params,
      optState: optimizer.init(model.params),
      step: 0,
      epoch: 0,
      loss: Number.NaN,
      gradNorm: Number.NaN,
      epochLossSum: 0,
    }),
    step: (s, ctx) => {
      const epoch = Math.floor(s.step / perEpoch)
      const batchIndex = s.step % perEpoch
      let stopped: string | undefined
      const hook = (params: Params, step: number): HookContext => ({
        step,
        epoch,
        batch: batchIndex,
        params,
        run: ctx.run,
        log: (name, value) => ctx.run.metric(step, name, value),
        logTensor: (name, role, value) => ctx.run.tensor(step, name, role, value),
        stop: (reason) => {
          stopped ??= reason
        },
      })
      const call = <K extends keyof Callback>(name: K, extra: object, params: Params, step = s.step): void => {
        for (const c of callbacks)
          (c[name] as ((x: unknown) => void) | undefined)?.({ ...hook(params, step), ...extra })
      }
      if (s.step === 0) call('onTrainStart', {}, s.params)
      if (batchIndex === 0) call('onEpochStart', {}, s.params)
      call('onBatchStart', {}, s.params)
      const batch = batchAt(epoch, batchIndex)
      const dropoutKey = foldIn(foldIn(base, 1), s.step)
      const objective = (p: Params): Value =>
        options.loss(model.apply(p, batch.x, { train: true, key: dropoutKey }), batch.y)
      const { value, grad } = valueAndGrad(objective)(s.params)
      const loss = item(value)
      call('onAfterBackward', { loss, grads: grad }, s.params)
      const { updates, state } = optimizer.update(grad, s.optState, s.params)
      call('onBeforeOptimizerStep', { grads: grad, updates }, s.params)
      const params = applyUpdates(s.params, updates) as Params
      const step = s.step + 1
      call('onBatchEnd', { loss }, params, step)
      let epochLossSum = s.epochLossSum + loss
      if (step % perEpoch === 0) {
        const meanLoss = epochLossSum / perEpoch
        ctx.run.metric(step, 'epochLoss', meanLoss)
        call('onEpochEnd', { meanLoss }, params, step)
        if (options.validation !== undefined && (epoch + 1) % (options.validation.every ?? 1) === 0) {
          const metrics = validate(params)
          for (const [k, v] of Object.entries(metrics)) ctx.run.metric(step, k, v)
          call('onValidationEnd', { metrics }, params, step)
        }
        epochLossSum = 0
      }
      const status = stopped !== undefined ? 'stopped' : 'completed'
      if (stopped !== undefined || step >= total) call('onTrainEnd', { status }, params, step)
      if (stopped !== undefined) ctx.run.log(step, 'info', stopped)
      return {
        params,
        optState: state,
        step,
        epoch: Math.floor(step / perEpoch),
        loss,
        gradNorm: item(globalNorm(grad)),
        epochLossSum,
        ...(stopped === undefined ? {} : { stopped }),
      }
    },
    done: (s) =>
      s.stopped !== undefined
        ? 'stopped'
        : s.step >= total
          ? 'completed'
          : !Number.isNaN(s.loss) && !Number.isFinite(s.loss)
            ? 'diverged'
            : undefined,
    observe: (s) => {
      const tensors: Record<string, { role: 'param'; value: Value }> = {}
      for (const [path, v] of Object.entries(treePaths(s.params)))
        tensors[`params.${path}`] = { role: 'param', value: v }
      return { metrics: Number.isNaN(s.loss) ? {} : { loss: s.loss, gradNorm: s.gradNorm }, tensors }
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
  const { loss, optimizer, epochs, batchSize, key, clipNorm, callbacks, validation, ...run } = options
  const fitOptions: FitOptions = {
    loss,
    optimizer,
    epochs,
    ...(batchSize === undefined ? {} : { batchSize }),
    ...(key === undefined ? {} : { key }),
    ...(clipNorm === undefined ? {} : { clipNorm }),
    ...(callbacks === undefined ? {} : { callbacks }),
    ...(validation === undefined ? {} : { validation }),
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
