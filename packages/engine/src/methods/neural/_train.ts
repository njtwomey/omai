// Private training plumbing of the neural methods (methods/neural): lib/nn's minibatch `fitAlgorithm`, run through
// `runAlgorithm`, with what an explanation of training needs recorded on the way: the loss and score of every epoch on
// the training and validation rows, the minibatch loss of every step, and the parameters at a few dozen epochs spaced
// geometrically (dense early, where training changes fastest), so a page can play the training back. Adds
// scikit-learn's stopping rules (no improvement by `tol` for `nIterNoChange` epochs, on the training loss or the
// validation score) and its 'adaptive' learning rate (divide by five instead of stopping).
import { rowCount, type TabularData, take, xy } from '../../lib/data/index.ts'
import { type FitOptions, fitRunOptions, geometricSteps } from '../../lib/learn/index.ts'
import { fitAlgorithm, type Module, type Params, type TrainState } from '../../lib/nn/index.ts'
import type { GradientTransformation, LearningRate } from '../../lib/optim/index.ts'
import { type Key, key as makeKey, permutation } from '../../lib/random/index.ts'
import { item, type Tensor, toFlat, type Value } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'

/** The rows a network trains or validates on: inputs `[n, ...]` and targets. */
export interface Rows {
  readonly x: Tensor
  readonly y: Tensor
}

/** The parameters at one recorded epoch (0 is the initial network). */
export interface Snapshot {
  readonly epoch: number
  readonly params: Params
  readonly buffers: Params
}

/** What one finished epoch reports (for progress and live drawing). */
export interface EpochReport {
  readonly epoch: number
  readonly epochs: number
  readonly loss: number
  readonly snapshot?: Snapshot
}

/** Options of {@link train}. */
export interface TrainOptions {
  /** The run's name (the key is derived under it). */
  readonly name: string
  readonly model: Module
  readonly train: Rows
  readonly validation?: Rows
  /** The objective of a prediction and its targets (data term only; penalties go in the optimiser). */
  readonly loss: (prediction: Value, target: Value) => Value
  /** A score of eval-mode predictions and targets, higher is better (accuracy, R²). */
  readonly score?: (prediction: Tensor, target: Tensor) => number
  readonly optimizer: GradientTransformation
  /** The optimiser's learning rate, for the record. */
  readonly learningRate?: LearningRate
  readonly epochs: number
  /** Minibatch size (default the whole training set). */
  readonly batchSize?: number
  /** Stop after `nIterNoChange` epochs without improving by `tol` (scikit-learn); undefined never stops early. */
  readonly stopping?: {
    readonly tol: number
    readonly nIterNoChange: number
    /** Monitor the validation score (scikit-learn's early_stopping) rather than the training loss. */
    readonly monitor: 'loss' | 'validationScore'
    /** Divide the learning rate by five instead of stopping (scikit-learn's 'adaptive'), down to 1e-6 of it. */
    readonly adaptive?: boolean
  }
  /** The most snapshots kept (default 40). */
  readonly snapshots?: number
  /** Extra epochs to snapshot besides the geometric schedule. */
  readonly snapshotAt?: readonly number[]
  readonly fit: FitOptions
  /** Called after every epoch (progress for live pages). */
  readonly onEpoch?: (report: EpochReport) => void
}

/** What {@link train} returns. */
export interface Training {
  readonly params: Params
  readonly buffers: Params
  /** Epoch numbers 1..E of the epochs run. */
  readonly epochs: readonly number[]
  /** Mean minibatch loss of each epoch (scikit-learn's loss_curve_). */
  readonly trainLoss: readonly number[]
  /** The loss on all the training rows (eval mode) at the end of each epoch. */
  readonly evalLoss: readonly number[]
  /** The loss and score of the initial network on the training rows. */
  readonly initial: { readonly loss: number; readonly score?: number }
  readonly trainScore?: readonly number[]
  readonly validationLoss?: readonly number[]
  readonly validationScore?: readonly number[]
  /** The minibatch loss of every step. */
  readonly stepLoss: readonly number[]
  readonly snapshots: readonly Snapshot[]
  readonly status: string
  /** The epoch whose parameters were kept (the best validation score with validation stopping, else the last). */
  readonly bestEpoch: number
  readonly steps: number
}

/**
 * Hold out a stratified (classification) or random fraction of the rows for validation, as scikit-learn's
 * early_stopping does.
 */
export function holdOut(rows: Rows, fraction: number, key: Key, stratify: boolean): { train: Rows; validation: Rows } {
  const n = rows.x.shape[0] ?? 0
  const order = toFlat(permutation(key, n))
  const pick: number[] = []
  if (stratify) {
    const y = toFlat(rows.y)
    const byClass = new Map<number, number[]>()
    for (const i of order) byClass.set(y[i] ?? 0, [...(byClass.get(y[i] ?? 0) ?? []), i])
    for (const members of byClass.values())
      pick.push(...members.slice(0, Math.max(1, Math.round(fraction * members.length))))
  } else pick.push(...order.slice(0, Math.max(1, Math.round(fraction * n))))
  const held = new Set(pick)
  const rest = order.filter((i) => !held.has(i))
  const data = xy(rows.x, rows.y)
  const part = (idx: readonly number[]): Rows => {
    const d = take(data, idx)
    return { x: d.x, y: d.y }
  }
  return { train: part(rest), validation: part([...pick].sort((a, b) => a - b)) }
}

/** The model's eval-mode output on some rows. */
export function forward(model: Module, params: Params, buffers: Params, x: Tensor): Tensor {
  return model.apply(params, x, { train: false, buffers }) as Tensor
}

/**
 * Train a module with minibatches, recording per-epoch curves, per-step losses and geometric snapshots.
 *
 * @param options - The model, rows, loss, score, optimiser, epochs, batching, stopping, snapshots and run options.
 * @returns The trained parameters and the record of training.
 */
export function train(options: TrainOptions): Training {
  const { model } = options
  const data = xy(options.train.x, options.train.y)
  const n = rowCount(data)
  const batchSize = Math.min(n, options.batchSize ?? n)
  const perEpoch = Math.ceil(n / batchSize)
  const run = fitRunOptions(options.fit, options.name)
  const base = run.key ?? makeKey(0)
  const inner = fitAlgorithm(model, data as TabularData, {
    loss: options.loss,
    optimizer: options.optimizer,
    epochs: options.epochs,
    ...(batchSize < n ? { batchSize } : {}),
    key: base,
    ...(options.learningRate === undefined ? {} : { learningRate: options.learningRate }),
  })
  const schedule = new Set([...geometricSteps(options.epochs, options.snapshots ?? 40), ...(options.snapshotAt ?? [])])
  const snapshots: Snapshot[] = []
  const epochs: number[] = []
  const trainLoss: number[] = []
  const trainScore: number[] = []
  const evalLoss: number[] = []
  let initial: { loss: number; score?: number } = { loss: Number.NaN }
  const validationLoss: number[] = []
  const validationScore: number[] = []
  const stepLoss: number[] = []
  const stopping = options.stopping
  let best = Number.POSITIVE_INFINITY
  let noChange = 0
  let stopReason: string | undefined
  let kept: { epoch: number; params: Params; buffers: Params } | undefined
  let epochSum = 0
  const evaluate = (rows: Rows, params: Params, buffers: Params): { loss: number; score?: number } => {
    const p = forward(model, params, buffers, rows.x)
    const loss = item(options.loss(p, rows.y))
    return options.score === undefined ? { loss } : { loss, score: options.score(p, rows.y) }
  }
  const algorithm: Algorithm<TrainState> = {
    ...inner,
    name: options.name,
    init: (k) => {
      const s = inner.init(k)
      initial = evaluate(options.train, s.params, s.buffers)
      if (schedule.has(0)) snapshots.push({ epoch: 0, params: s.params, buffers: s.buffers })
      return s
    },
    step: (s, ctx) => {
      const next = inner.step(s, ctx)
      stepLoss.push(next.loss)
      epochSum += next.loss
      if (next.step % perEpoch !== 0) return next
      const epoch = next.step / perEpoch
      const meanLoss = epochSum / perEpoch
      epochSum = 0
      epochs.push(epoch)
      trainLoss.push(meanLoss)
      const t = evaluate(options.train, next.params, next.buffers)
      evalLoss.push(t.loss)
      if (t.score !== undefined) trainScore.push(t.score)
      let vScore: number | undefined
      if (options.validation !== undefined) {
        const v = evaluate(options.validation, next.params, next.buffers)
        validationLoss.push(v.loss)
        if (v.score !== undefined) validationScore.push(v.score)
        vScore = v.score
        ctx.run.metric(next.step, 'valLoss', v.loss)
      }
      let state = next
      if (stopping !== undefined) {
        const monitored = stopping.monitor === 'validationScore' && vScore !== undefined ? -vScore : meanLoss
        if (monitored > best - stopping.tol) noChange++
        else noChange = 0
        if (monitored < best) {
          best = monitored
          if (stopping.monitor === 'validationScore') kept = { epoch, params: next.params, buffers: next.buffers }
        }
        if (noChange >= stopping.nIterNoChange) {
          if (stopping.adaptive === true && next.lrScale > 1e-6) {
            state = { ...next, lrScale: next.lrScale / 5 }
            noChange = 0
          } else stopReason = `no improvement by ${stopping.tol} for ${stopping.nIterNoChange} epochs (epoch ${epoch})`
        }
      }
      const last = stopReason !== undefined || epoch >= options.epochs || !Number.isFinite(meanLoss)
      let snapshot: Snapshot | undefined
      if (schedule.has(epoch) || last) {
        snapshot = { epoch, params: state.params, buffers: state.buffers }
        if (snapshots.at(-1)?.epoch !== epoch) snapshots.push(snapshot)
      }
      options.onEpoch?.({
        epoch,
        epochs: options.epochs,
        loss: meanLoss,
        ...(snapshot === undefined ? {} : { snapshot }),
      })
      return state
    },
    done: (s, t) => (stopReason !== undefined ? 'converged' : inner.done?.(s, t)),
  }
  const result = runAlgorithm(algorithm, { ...run, maxSteps: options.epochs * perEpoch })
  const final = kept ?? { epoch: epochs.at(-1) ?? 0, params: result.state.params, buffers: result.state.buffers }
  return {
    params: final.params,
    buffers: final.buffers,
    epochs,
    trainLoss,
    evalLoss,
    initial,
    ...(options.score === undefined ? {} : { trainScore }),
    ...(options.validation === undefined ? {} : { validationLoss }),
    ...(options.validation !== undefined && options.score !== undefined ? { validationScore } : {}),
    stepLoss,
    snapshots,
    status: result.status,
    bestEpoch: final.epoch,
    steps: result.steps,
  }
}
