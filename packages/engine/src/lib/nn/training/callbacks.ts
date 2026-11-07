/**
 * Training callbacks in the spirit of PyTorch Lightning's hooks: loggers of gradient norms, parameter norms, full
 * tensors, learning rates and tapped activations into the run's trace, plus early stopping, best-checkpoint keeping and
 * reducing the learning rate on a plateau.
 */

import { item, norm, treePaths, type Value } from '../../tensor/index.ts'
import type { RunHandle, TensorRole } from '../../trace/index.ts'
import type { Params } from '../module/module.ts'

/**
 * The paths the logging callbacks write, which have a computed part (a parameter's path in the network, a tap's name),
 * hence template literals: the norms (`grad_norm/total`, `grad_norm/<parameter>`, `param_norm/...`), the full tensors
 * (`grad/<parameter>`, `param/<parameter>`), the learning rate (`lr`, `lr_scale`) and the tapped activations
 * (`activation/<name>`, `activation_norm/<name>`).
 *
 * @example
 * ```ts
 * import type { CallbackPath } from 'mdml-engine/nn'
 * const path: CallbackPath = 'grad_norm/layers/0/weight'
 * path // => 'grad_norm/layers/0/weight'
 * ```
 */
export type CallbackPath =
  | 'grad_norm/total'
  | `grad_norm/${string}`
  | 'param_norm/total'
  | `param_norm/${string}`
  | `grad/${string}`
  | `param/${string}`
  | 'lr'
  | 'lr_scale'
  | `activation/${string}`
  | `activation_norm/${string}`

/**
 * What a training hook sees (the ideas of PyTorch Lightning's hooks, without its machinery): where training is, the
 * current parameters, and ways to log into the run's trace and to stop. The parameter is the type of the paths a
 * callback logs (`string`: any; the library's callbacks declare theirs, see {@link CallbackPath}).
 */
export interface TrainingHookContext<K extends string = string> {
  /** Optimiser updates applied so far. */
  readonly step: number
  readonly epoch: number
  /** The index of the minibatch within the epoch. */
  readonly batch: number
  readonly params: Params
  /** Log a scalar into the run's trace at this step (Lightning's self.log); recorded when the policy selects it. */
  log(name: K, value: number): void
  /** Log a tensor (by role) into the trace, recorded when the policy selects the name and role. */
  logTensor(name: K, role: TensorRole, value: Value): void
  /** Stop training after this hook (the run ends with status `stopped`). */
  stop(reason: string): void
  /** The factor every update is multiplied by (1 unless lowered). */
  readonly lrScale: number
  /** Set the update factor from this step on (learning-rate reduction, warm restarts by hand). */
  setLearningRateScale(scale: number): void
  /** The underlying run, for anything else (messages, child runs). */
  readonly run: RunHandle
}

/**
 * A training callback: any subset of the hooks, called by `fit` in Lightning's order. Callbacks may keep their own state
 * (early stopping's patience counter, the best checkpoint).
 *
 * @concept training-callbacks
 */
export interface Callback<K extends string = string> {
  readonly name: string
  /** Ask the loop to tap activations in the forward pass (delivered to `onBatchEnd`). */
  readonly tapsActivations?: boolean
  onTrainStart?(ctx: TrainingHookContext<K>): void
  onEpochStart?(ctx: TrainingHookContext<K>): void
  onBatchStart?(ctx: TrainingHookContext<K>): void
  /** After the loss has been differentiated: the gradients by parameter path's tree. */
  onAfterBackward?(ctx: TrainingHookContext<K> & { readonly loss: number; readonly grads: Params }): void
  /** Just before the updates are applied (after any clipping by the optimiser chain). */
  onBeforeOptimizerStep?(ctx: TrainingHookContext<K> & { readonly grads: Params; readonly updates: Params }): void
  /** After the update, with the loss and the activations tapped in the forward pass (when a callback asked). */
  onBatchEnd?(
    ctx: TrainingHookContext<K> & { readonly loss: number; readonly activations: ReadonlyMap<string, Value> },
  ): void
  onEpochEnd?(ctx: TrainingHookContext<K> & { readonly meanLoss: number }): void
  /** After validation, with every validation metric (always `valLoss`). */
  onValidationEnd?(ctx: TrainingHookContext<K> & { readonly metrics: Readonly<Record<string, number>> }): void
  onTrainEnd?(ctx: TrainingHookContext<K> & { readonly status: 'completed' | 'stopped' }): void
}

/** The L2 norm of a value, as a number. */
const l2 = (v: Value): number => item(norm(v))

/** Options of the logging callbacks. */
export interface LoggerOptions {
  /** Log every this many steps (default 1). */
  readonly every?: number
  /** Also log one entry per parameter tensor (default true). */
  readonly perParameter?: boolean
}

/**
 * Log gradient norms after each backward pass: `grad_norm/total` (the global L2 norm) and, per parameter,
 * `grad_norm/<path>` (as Lightning's `grad_norm` utility).
 *
 * @param options - Frequency and whether to log per parameter.
 * @returns The callback.
 * @concept training-callbacks
 * @example
 * ```ts
 * import { gradNormLogger } from 'mdml-engine/nn'
 * gradNormLogger().name // => 'gradNormLogger'
 * ```
 */
export function gradNormLogger(options: LoggerOptions = {}): Callback<'grad_norm/total' | `grad_norm/${string}`> {
  const every = options.every ?? 1
  return {
    name: 'gradNormLogger',
    onAfterBackward: (ctx) => {
      if (ctx.step % every !== 0) return
      let total = 0
      for (const [path, g] of Object.entries(treePaths(ctx.grads))) {
        const n = l2(g)
        total += n * n
        if (options.perParameter ?? true) ctx.log(`grad_norm/${path}`, n)
      }
      ctx.log('grad_norm/total', Math.sqrt(total))
    },
  }
}

/**
 * Log parameter norms after each step: `param_norm/total` and `param_norm/<path>`.
 *
 * @param options - Frequency and whether to log per parameter.
 * @returns The callback.
 * @example
 * ```ts
 * import { paramNormLogger } from 'mdml-engine/nn'
 * paramNormLogger({ every: 10 }).name // => 'paramNormLogger'
 * ```
 */
export function paramNormLogger(options: LoggerOptions = {}): Callback<'param_norm/total' | `param_norm/${string}`> {
  const every = options.every ?? 1
  return {
    name: 'paramNormLogger',
    onBatchEnd: (ctx) => {
      if (ctx.step % every !== 0) return
      let total = 0
      for (const [path, p] of Object.entries(treePaths(ctx.params))) {
        const n = l2(p)
        total += n * n
        if (options.perParameter ?? true) ctx.log(`param_norm/${path}`, n)
      }
      ctx.log('param_norm/total', Math.sqrt(total))
    },
  }
}

/**
 * Log the gradient and parameter tensors themselves (roles `grad` and `param`, under `grad/<path>` and `param/<path>`),
 * recorded in full or as summaries according to the run's policy.
 *
 * @param options - Frequency.
 * @returns The callback.
 * @example
 * ```ts
 * import { valueLogger } from 'mdml-engine/nn'
 * valueLogger().name // => 'valueLogger'
 * ```
 */
export function valueLogger(options: { readonly every?: number } = {}): Callback<`grad/${string}` | `param/${string}`> {
  const every = options.every ?? 1
  return {
    name: 'valueLogger',
    onAfterBackward: (ctx) => {
      if (ctx.step % every !== 0) return
      for (const [path, g] of Object.entries(treePaths(ctx.grads))) ctx.logTensor(`grad/${path}`, 'grad', g)
      for (const [path, p] of Object.entries(treePaths(ctx.params))) ctx.logTensor(`param/${path}`, 'param', p)
    },
  }
}

/**
 * Log the learning rate each step (Lightning's LearningRateMonitor), from the same schedule the optimiser uses.
 *
 * @param learningRate - A constant or the schedule (a function of the step).
 * @returns The callback.
 * @example
 * ```ts
 * import { learningRateMonitor } from 'mdml-engine/nn'
 * learningRateMonitor((t) => 0.1 / (1 + t)).name // => 'learningRateMonitor'
 * ```
 */
export function learningRateMonitor(learningRate: number | ((step: number) => number)): Callback<'lr'> {
  return {
    name: 'learningRateMonitor',
    onBatchStart: (ctx) => ctx.log('lr', typeof learningRate === 'number' ? learningRate : learningRate(ctx.step)),
  }
}

/** Options of {@link earlyStopping}. */
export interface EarlyStoppingOptions {
  /** The quantity to watch: `valLoss` (needs validation data), `loss` (the epoch's mean training loss) or a metric. */
  readonly monitor?: string
  /** Epochs without improvement before stopping (default 3). */
  readonly patience?: number
  /** The least change that counts as an improvement (default 0). */
  readonly minDelta?: number
  /** Whether smaller (`min`, default) or larger values are better. */
  readonly mode?: 'min' | 'max'
}

/**
 * Stop when the monitored quantity has not improved for `patience` epochs (Lightning's EarlyStopping). Checked after
 * validation for validation metrics and at the end of each epoch for `loss`.
 *
 * @param options - Monitor, patience, minimum delta and mode.
 * @returns The callback, with its best value so far.
 * @concept training-callbacks
 * @example
 * ```ts
 * import { earlyStopping } from 'mdml-engine/nn'
 * earlyStopping({ patience: 5 }).name // => 'earlyStopping'
 * ```
 */
export function earlyStopping(options: EarlyStoppingOptions = {}): Callback<never> & { readonly best: () => number } {
  const monitor = options.monitor ?? 'valLoss'
  const patience = options.patience ?? 3
  const delta = options.minDelta ?? 0
  const sign = options.mode === 'max' ? -1 : 1
  let best = Number.POSITIVE_INFINITY
  let wait = 0
  const check = (ctx: TrainingHookContext<never>, value: number | undefined): void => {
    if (value === undefined) return
    if (sign * value < best - delta) {
      best = sign * value
      wait = 0
      return
    }
    wait += 1
    if (wait >= patience) ctx.stop(`earlyStopping: ${monitor} did not improve for ${patience} epochs`)
  }
  return {
    name: 'earlyStopping',
    best: () => sign * best,
    onValidationEnd: (ctx) => {
      if (monitor !== 'loss') check(ctx, ctx.metrics[monitor])
    },
    onEpochEnd: (ctx) => {
      if (monitor === 'loss') check(ctx, ctx.meanLoss)
    },
  }
}

/**
 * Keep the parameters with the best monitored value (Lightning's ModelCheckpoint with save_top_k = 1, in memory).
 *
 * @param options - The quantity to monitor (default `valLoss`; `loss` is the epoch's mean training loss) and mode.
 * @returns The callback, with `best()` returning the best value and its parameters.
 * @example
 * ```ts
 * import { bestCheckpoint } from 'mdml-engine/nn'
 * bestCheckpoint().best().params // => undefined
 * ```
 */
export function bestCheckpoint(
  options: { readonly monitor?: string; readonly mode?: 'min' | 'max' } = {},
): Callback<never> & {
  readonly best: () => { readonly value: number; readonly params: Params | undefined; readonly epoch: number }
} {
  const monitor = options.monitor ?? 'valLoss'
  const sign = options.mode === 'max' ? -1 : 1
  let value = Number.POSITIVE_INFINITY
  let params: Params | undefined
  let epoch = -1
  const offer = (ctx: TrainingHookContext<never>, v: number | undefined): void => {
    if (v === undefined || !(sign * v < value)) return
    value = sign * v
    params = ctx.params
    epoch = ctx.epoch
  }
  return {
    name: 'bestCheckpoint',
    best: () => ({ value: sign * value, params, epoch }),
    onValidationEnd: (ctx) => {
      if (monitor !== 'loss') offer(ctx, ctx.metrics[monitor])
    },
    onEpochEnd: (ctx) => {
      if (monitor === 'loss') offer(ctx, ctx.meanLoss)
    },
  }
}

/** Options of {@link reduceLROnPlateau}. */
export interface PlateauOptions {
  /** The quantity to watch (default `valLoss`; `loss` is the epoch's mean training loss). */
  readonly monitor?: string
  readonly mode?: 'min' | 'max'
  /** Multiply the learning rate by this on a plateau (default 0.1). */
  readonly factor?: number
  /** Epochs without improvement before reducing (default 10). */
  readonly patience?: number
  /** The least relative change that counts as an improvement (default 1e-4, PyTorch's 'rel' threshold). */
  readonly threshold?: number
  /** Epochs to wait after a reduction before counting again (default 0). */
  readonly cooldown?: number
  /** The smallest scale (default 0). */
  readonly minScale?: number
}

/**
 * Reduce the learning rate when a metric stops improving (torch.optim.lr_scheduler.ReduceLROnPlateau): after
 * `patience` epochs without a relative improvement of `threshold`, the update scale is multiplied by `factor` and
 * logged as `lr_scale`.
 *
 * @param options - Monitor, mode, factor, patience, threshold, cooldown and minimum scale.
 * @returns The callback, with the current scale.
 * @concept training-callbacks
 * @example
 * ```ts
 * import { reduceLROnPlateau } from 'mdml-engine/nn'
 * reduceLROnPlateau({ patience: 2 }).scale() // => 1
 * ```
 */
export function reduceLROnPlateau(
  options: PlateauOptions = {},
): Callback<'lr_scale'> & { readonly scale: () => number } {
  const monitor = options.monitor ?? 'valLoss'
  const sign = options.mode === 'max' ? -1 : 1
  const factor = options.factor ?? 0.1
  const patience = options.patience ?? 10
  const threshold = options.threshold ?? 1e-4
  const cooldown = options.cooldown ?? 0
  const minScale = options.minScale ?? 0
  let best = Number.POSITIVE_INFINITY
  let bad = 0
  let cooling = 0
  let scale = 1
  const check = (ctx: TrainingHookContext<'lr_scale'>, value: number | undefined): void => {
    if (value === undefined) return
    const v = sign * value
    // PyTorch's 'rel' mode: improvement means v < best (1 - threshold) for positive best.
    if (v < best - Math.abs(best) * threshold) {
      best = v
      bad = 0
    } else bad += 1
    if (cooling > 0) {
      cooling -= 1
      bad = 0
    }
    if (bad > patience) {
      const next = Math.max(scale * factor, minScale)
      if (next < scale) {
        scale = next
        ctx.setLearningRateScale(scale)
        ctx.log('lr_scale', scale)
      }
      cooling = cooldown
      bad = 0
    }
  }
  return {
    name: 'reduceLROnPlateau',
    scale: () => scale,
    onValidationEnd: (ctx) => {
      if (monitor !== 'loss') check(ctx, ctx.metrics[monitor])
    },
    onEpochEnd: (ctx) => {
      if (monitor === 'loss') check(ctx, ctx.meanLoss)
    },
  }
}

/**
 * Log activations tapped in the forward pass (`tap(ctx, name, value)` inside modules: attention weights, hidden
 * states) as tensors with role `activation` under `activation/<path>`, recorded in full or as summaries by the run's
 * policy, and optionally their norms as metrics.
 *
 * @param options - Frequency, a name filter, and whether to log norms.
 * @returns The callback.
 * @concept training-callbacks
 * @example
 * ```ts
 * import { activationLogger } from 'mdml-engine/nn'
 * activationLogger().tapsActivations // => true
 * ```
 */
export function activationLogger(
  options: { readonly every?: number; readonly filter?: (name: string) => boolean; readonly norms?: boolean } = {},
): Callback<`activation/${string}` | `activation_norm/${string}`> {
  const every = options.every ?? 1
  return {
    name: 'activationLogger',
    tapsActivations: true,
    onBatchEnd: (ctx) => {
      if ((ctx.step - 1) % every !== 0) return
      for (const [name, value] of ctx.activations) {
        if (options.filter !== undefined && !options.filter(name)) continue
        ctx.logTensor(`activation/${name}`, 'activation', value)
        if (options.norms === true) ctx.log(`activation_norm/${name}`, l2(value))
      }
    },
  }
}
