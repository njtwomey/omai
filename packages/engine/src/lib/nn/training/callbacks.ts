import { item, sqrt, square, sum, treePaths, type Value } from '../../tensor/index.ts'
import type { RunHandle, TensorRole } from '../../trace/index.ts'
import type { Params } from '../module/module.ts'

/**
 * What a training hook sees (the ideas of PyTorch Lightning's hooks, without its machinery): where training is, the
 * current parameters, and ways to log into the run's trace and to stop.
 */
export interface HookContext {
  /** Optimiser updates applied so far. */
  readonly step: number
  readonly epoch: number
  /** The index of the minibatch within the epoch. */
  readonly batch: number
  readonly params: Params
  /** Log a scalar into the run's trace at this step (Lightning's self.log); recorded when the policy selects it. */
  log(name: string, value: number): void
  /** Log a tensor (by role) into the trace, recorded when the policy selects the name and role. */
  logTensor(name: string, role: TensorRole, value: Value): void
  /** Stop training after this hook (the run ends with status `stopped`). */
  stop(reason: string): void
  /** The underlying run, for anything else (messages, child runs). */
  readonly run: RunHandle
}

/**
 * A training callback: any subset of the hooks, called by `fit` in Lightning's order. Callbacks may keep their own state
 * (early stopping's patience counter, the best checkpoint).
 *
 * @concept training-callbacks
 */
export interface Callback {
  readonly name: string
  onTrainStart?(ctx: HookContext): void
  onEpochStart?(ctx: HookContext): void
  onBatchStart?(ctx: HookContext): void
  /** After the loss has been differentiated: the gradients by parameter path's tree. */
  onAfterBackward?(ctx: HookContext & { readonly loss: number; readonly grads: Params }): void
  /** Just before the updates are applied (after any clipping by the optimiser chain). */
  onBeforeOptimizerStep?(ctx: HookContext & { readonly grads: Params; readonly updates: Params }): void
  onBatchEnd?(ctx: HookContext & { readonly loss: number }): void
  onEpochEnd?(ctx: HookContext & { readonly meanLoss: number }): void
  /** After validation, with every validation metric (always `valLoss`). */
  onValidationEnd?(ctx: HookContext & { readonly metrics: Readonly<Record<string, number>> }): void
  onTrainEnd?(ctx: HookContext & { readonly status: 'completed' | 'stopped' }): void
}

const norm = (v: Value): number => item(sqrt(sum(square(v))))

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
 * import { gradNormLogger } from 'whitebox-engine/nn'
 * gradNormLogger().name // => 'gradNormLogger'
 * ```
 */
export function gradNormLogger(options: LoggerOptions = {}): Callback {
  const every = options.every ?? 1
  return {
    name: 'gradNormLogger',
    onAfterBackward: (ctx) => {
      if (ctx.step % every !== 0) return
      let total = 0
      for (const [path, g] of Object.entries(treePaths(ctx.grads))) {
        const n = norm(g)
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
 * import { paramNormLogger } from 'whitebox-engine/nn'
 * paramNormLogger({ every: 10 }).name // => 'paramNormLogger'
 * ```
 */
export function paramNormLogger(options: LoggerOptions = {}): Callback {
  const every = options.every ?? 1
  return {
    name: 'paramNormLogger',
    onBatchEnd: (ctx) => {
      if (ctx.step % every !== 0) return
      let total = 0
      for (const [path, p] of Object.entries(treePaths(ctx.params))) {
        const n = norm(p)
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
 * import { valueLogger } from 'whitebox-engine/nn'
 * valueLogger().name // => 'valueLogger'
 * ```
 */
export function valueLogger(options: { readonly every?: number } = {}): Callback {
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
 * import { learningRateMonitor } from 'whitebox-engine/nn'
 * learningRateMonitor((t) => 0.1 / (1 + t)).name // => 'learningRateMonitor'
 * ```
 */
export function learningRateMonitor(learningRate: number | ((step: number) => number)): Callback {
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
 * import { earlyStopping } from 'whitebox-engine/nn'
 * earlyStopping({ patience: 5 }).name // => 'earlyStopping'
 * ```
 */
export function earlyStopping(options: EarlyStoppingOptions = {}): Callback & { readonly best: () => number } {
  const monitor = options.monitor ?? 'valLoss'
  const patience = options.patience ?? 3
  const delta = options.minDelta ?? 0
  const sign = options.mode === 'max' ? -1 : 1
  let best = Number.POSITIVE_INFINITY
  let wait = 0
  const check = (ctx: HookContext, value: number | undefined): void => {
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
 * import { bestCheckpoint } from 'whitebox-engine/nn'
 * bestCheckpoint().best().params // => undefined
 * ```
 */
export function bestCheckpoint(
  options: { readonly monitor?: string; readonly mode?: 'min' | 'max' } = {},
): Callback & {
  readonly best: () => { readonly value: number; readonly params: Params | undefined; readonly epoch: number }
} {
  const monitor = options.monitor ?? 'valLoss'
  const sign = options.mode === 'max' ? -1 : 1
  let value = Number.POSITIVE_INFINITY
  let params: Params | undefined
  let epoch = -1
  const offer = (ctx: HookContext, v: number | undefined): void => {
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
