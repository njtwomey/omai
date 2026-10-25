/**
 * Minibatch training of a module: `fitAlgorithm` runs the loop as a traced `Algorithm` (loss on each minibatch in
 * training mode, gradients, optimiser update, optional validation and callbacks), and `fit` runs it on a dataset and
 * loads the trained parameters and buffers back into the module.
 */

import type { Json } from '../../base/index.ts'
import { epochBatches, rowCount, type TabularData } from '../../data/index.ts'
import { clipByGlobalNorm, type GradientTransformation, globalNorm, type LearningRate } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { div, item, norm, treeMap, treePaths, type Value } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type RunHandle,
  type RunOptions,
  runAlgorithm,
  tapContext,
} from '../../trace/index.ts'
import type { Module, Params } from '../module/module.ts'
import type { Callback, HookContext } from './callbacks.ts'
import { accumulate, applyGradients, hasFrozen, lossAndGrads, type TrainState, trainState } from './state.ts'

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
  /** Sum gradients over this many minibatches per update (default 1); the epoch's last update takes what is left. */
  readonly accumulateGradients?: number
  /** The optimiser's learning rate (or schedule), recorded as `lr` times the plateau scale. */
  readonly learningRate?: LearningRate
  /** Hooks called in Lightning's order (logging, early stopping, checkpoints, learning-rate reduction). */
  readonly callbacks?: readonly Callback[]
  readonly validation?: ValidationOptions
}

const rate = (lr: LearningRate, step: number): number => (typeof lr === 'number' ? lr : lr(step))

/** Per-layer norms, gradients, parameters and activations, as the run's policy asks (roles grad, param, activation). */
function recordLayers(run: RunHandle, step: number, params: Params, grads: Params | undefined): void {
  if (!run.due(step)) return
  const roles = run.policy.roles
  if (roles.includes('grad') && grads !== undefined) {
    for (const [path, g] of Object.entries(treePaths(grads))) {
      run.metric(step, `gradNorm/${path}`, item(norm(g)))
      run.tensor(step, `grads.${path}`, 'grad', g)
    }
  }
  if (roles.includes('param')) {
    for (const [path, p] of Object.entries(treePaths(params))) {
      run.metric(step, `paramNorm/${path}`, item(norm(p)))
      run.tensor(step, `params.${path}`, 'param', p)
    }
  }
}

/**
 * Minibatch training of a module as an {@link Algorithm}: each step evaluates the loss on one minibatch (in training
 * mode, with the step's dropout key and the current buffers), differentiates it with respect to the parameters, and
 * applies the optimiser (after `accumulateGradients` minibatches, with frozen parameters masked and the update scaled
 * by the plateau factor); epochs reshuffle with their own keys and buffers (batch statistics) are carried along.
 * Records loss, gradient norm, learning rate, epoch loss and validation metrics; with the policy's tensor roles,
 * per-layer gradient and parameter norms and values (`grad`, `param`) and tapped activations (`activation`). Calls the
 * callbacks' hooks in Lightning's order: train start, epoch start, batch start, after backward, before optimiser step,
 * batch end, epoch end, validation end, train end. A non-finite loss ends the run as `diverged`.
 *
 * @param model - The module (its current parameters and buffers are the starting point; it is not modified).
 * @param data - The training data.
 * @param options - Loss, optimiser, epochs, batch size, key, clipping, accumulation, learning rate, callbacks and
 *   validation.
 * @returns The algorithm.
 * @concept training
 * @example
 * ```ts
 * import { fitAlgorithm, mlp } from 'mdml-engine/nn'
 * import { sgd } from 'mdml-engine/optim'
 * import { mse } from 'mdml-engine/losses'
 * import { xy } from 'mdml-engine/data'
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
  const window = Math.max(1, options.accumulateGradients ?? 1)
  const mask = model.trainableMask()
  const frozen = hasFrozen(mask)
  const wantsActivations = callbacks.some((c) => c.tapsActivations === true)
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
  const validate = (params: Params, buffers: Params): Record<string, number> => {
    const v = options.validation
    if (v === undefined) return {}
    const prediction = model.apply(params, v.data.x, { train: false, buffers })
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
      accumulateGradients: window,
      callbacks: callbacks.map((c) => c.name),
    } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'minibatch loss' },
        gradNorm: { description: 'global gradient norm' },
        lr: { description: 'learning rate (times the plateau scale)' },
        epochLoss: { description: 'mean minibatch loss of the epoch' },
        valLoss: { description: 'validation loss' },
      },
      tensors: {
        params: { role: 'param', description: 'parameters by layer' },
        grads: { role: 'grad', description: 'gradients by layer' },
        activations: { role: 'activation', description: 'tapped activations' },
      },
    }),
    init: () => ({ ...trainState(model, optimizer, base) }),
    step: (s, ctx) => {
      const epoch = Math.floor(s.step / perEpoch)
      const batchIndex = s.step % perEpoch
      let stopped: string | undefined
      let lrScale = s.lrScale
      const hook = (params: Params, step: number): HookContext => ({
        step,
        epoch,
        batch: batchIndex,
        params,
        lrScale,
        run: ctx.run,
        log: (name, value) => ctx.run.metric(step, name, value),
        logTensor: (name, role, value) => ctx.run.tensor(step, name, role, value),
        stop: (reason) => {
          stopped ??= reason
        },
        setLearningRateScale: (scale) => {
          lrScale = scale
        },
      })
      const call = <K extends keyof Callback>(name: K, extra: object, params: Params, step = s.step): void => {
        for (const c of callbacks) {
          const f = c[name]
          if (typeof f === 'function') (f as (x: unknown) => void).call(c, { ...hook(params, step), ...extra })
        }
      }
      if (s.step === 0) call('onTrainStart', {}, s.params)
      if (batchIndex === 0) call('onEpochStart', {}, s.params)
      call('onBatchStart', {}, s.params)
      const batch = batchAt(epoch, batchIndex)
      const taps =
        wantsActivations || (ctx.run.policy.roles.includes('activation') && ctx.run.due(s.step))
          ? tapContext()
          : undefined
      const r = lossAndGrads(model, s.params, s.buffers, batch.x, (y) => options.loss(y, batch.y), {
        key: foldIn(foldIn(base, 1), s.step),
        ...(taps === undefined ? {} : { taps }),
      })
      call('onAfterBackward', { loss: r.loss, grads: r.grads }, s.params)
      const acc = accumulate(s.accumulated, r.grads)
      const lastOfEpoch = batchIndex === perEpoch - 1
      let state: TrainState = { ...s, buffers: r.buffers, lrScale }
      if (acc.count >= window || lastOfEpoch) {
        const grads = acc.count === 1 ? acc.grads : treeMap((g) => div(g, acc.count), acc.grads)
        const applied = applyGradients(state, grads, optimizer, { ...(frozen ? { mask } : {}), scale: lrScale })
        call('onBeforeOptimizerStep', { grads, updates: applied.updates }, s.params)
        const { accumulated: _, ...rest } = applied.state
        state = rest
      } else {
        state = { ...state, accumulated: acc }
      }
      const step = s.step + 1
      const activations =
        taps === undefined ? new Map<string, Value>() : new Map([...taps.entries].map(([k, v]) => [k, v.value]))
      for (const [name, value] of activations) ctx.run.tensor(s.step, `activations.${name}`, 'activation', value)
      recordLayers(ctx.run, s.step, state.params, r.grads)
      if (options.learningRate !== undefined)
        ctx.run.metric(s.step, 'lr', rate(options.learningRate, s.updates) * lrScale)
      call('onBatchEnd', { loss: r.loss, activations }, state.params, step)
      let epochLossSum = s.epochLossSum + r.loss
      if (step % perEpoch === 0) {
        const meanLoss = epochLossSum / perEpoch
        ctx.run.metric(step, 'epochLoss', meanLoss)
        call('onEpochEnd', { meanLoss }, state.params, step)
        if (options.validation !== undefined && (epoch + 1) % (options.validation.every ?? 1) === 0) {
          const metrics = validate(state.params, state.buffers)
          for (const [k, v] of Object.entries(metrics)) ctx.run.metric(step, k, v)
          call('onValidationEnd', { metrics }, state.params, step)
        }
        epochLossSum = 0
      }
      const status = stopped !== undefined ? 'stopped' : 'completed'
      if (stopped !== undefined || step >= total) call('onTrainEnd', { status }, state.params, step)
      if (stopped !== undefined) ctx.run.log(step, 'info', stopped)
      return {
        ...state,
        lrScale,
        step,
        epoch: Math.floor(step / perEpoch),
        loss: r.loss,
        gradNorm: item(globalNorm(r.grads)),
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
      return {
        metrics: Number.isNaN(s.loss) ? {} : { loss: s.loss, gradNorm: s.gradNorm, lrScale: s.lrScale },
        tensors,
      }
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
  /** The final buffers (also loaded into the model). */
  readonly buffers: Params
  /** The minibatch losses, one per step. */
  readonly losses: readonly number[]
  readonly steps: number
  readonly status: string
}

/**
 * Train a module on a dataset and load the trained parameters and buffers into it.
 *
 * @param model - The module (updated in place with the trained parameters and buffers).
 * @param data - The training data.
 * @param options - Loss, optimiser, epochs, batch size, key, clipping, accumulation, callbacks, validation, and the
 *   run's tracer and policy.
 * @returns The trained parameters and buffers, per-step losses and the run status.
 * @concept training
 * @example
 * ```ts
 * import { fit, Linear } from 'mdml-engine/nn'
 * import { sgd } from 'mdml-engine/optim'
 * import { mse } from 'mdml-engine/losses'
 * import { xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const r = fit(new Linear(1, 1, { key: key(0) }), xy([[0], [1], [2]], [[1], [3], [5]]), { loss: (p, t) => mse(p, t), optimizer: sgd({ learningRate: 0.1 }), epochs: 500 })
 * (r.losses.at(-1) ?? 1) < 1e-8 // => true
 * ```
 */
export function fit(
  model: Module,
  data: TabularData,
  options: FitOptions & Partial<Omit<RunOptions, 'maxSteps'>>,
): FitResult {
  const {
    loss,
    optimizer,
    epochs,
    batchSize,
    key,
    clipNorm,
    accumulateGradients,
    learningRate,
    callbacks,
    validation,
    ...run
  } = options
  const fitOptions: FitOptions = {
    loss,
    optimizer,
    epochs,
    ...(batchSize === undefined ? {} : { batchSize }),
    ...(key === undefined ? {} : { key }),
    ...(clipNorm === undefined ? {} : { clipNorm }),
    ...(accumulateGradients === undefined ? {} : { accumulateGradients }),
    ...(learningRate === undefined ? {} : { learningRate }),
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
  model.setBuffers(result.state.buffers)
  return {
    params: result.state.params,
    buffers: result.state.buffers,
    losses,
    steps: result.steps,
    status: result.status,
  }
}
