/**
 * Recurrent networks on toy sequence tasks: `recurrentTagger` trains an Elman RNN, GRU or LSTM on tasks from
 * `lib/datasets` such as running parity, copying or Dyck languages, with a prediction at every position or from the
 * last one. Records the hidden states and predictions of a sequence as training goes on, and accuracy by position.
 */

import { MdmlError } from '../../../lib/base/index.ts'
import type { SequenceData } from '../../../lib/data/index.ts'
import {
  diagramRecord,
  type Explained,
  type Exposition,
  learningCurve,
  valueGrids,
} from '../../../lib/exposition/index.ts'
import type { FitOptions } from '../../../lib/learn/index.ts'
import { crossEntropy } from '../../../lib/losses/index.ts'
import {
  type ApplyContext,
  Composite,
  describe,
  Embedding,
  GRU,
  Linear,
  LSTM,
  type Params,
  type RecurrentOutput,
  RNN,
} from '../../../lib/nn/index.ts'
import { foldIn, type Key, key as makeKey } from '../../../lib/random/index.ts'
import type { Structure } from '../../../lib/scene/index.ts'
import { argmax, fromStorage, reshape, slice, type Tensor, toFlat, type Value } from '../../../lib/tensor/index.ts'
import { optimiserOf } from '../_nets.ts'
import { lossAtSnapshots, lossCurve, progressReporter, scoreCurve, type TrainingProgress } from '../_records.ts'
import { holdOut, type Rows, train } from '../_train.ts'

/** The recurrent cells a tagger can use. */
export type RecurrentCell = 'rnn' | 'gru' | 'lstm'

/** Options of {@link recurrentTagger}. */
export interface RecurrentTaggerOptions {
  /** Elman RNN (tanh, default), GRU or LSTM. */
  readonly cell?: RecurrentCell
  /** Hidden units (default 16). */
  readonly hidden?: number
  /** Embedding size of the input tokens (default 4). */
  readonly embedding?: number
  /** Epochs (default 80). */
  readonly epochs?: number
  /** Minibatch size (default all the training sequences). */
  readonly batchSize?: number
  /** Adam's learning rate (default 0.02). */
  readonly learningRate?: number
  /** Sequences held out for validation (default 0.25). */
  readonly validationFraction?: number
  /** The most snapshots of the hidden states (geometric over steps, default 30). */
  readonly snapshots?: number
  /** Which validation sequence is tapped (default 0). */
  readonly tapRow?: number
  readonly onProgress?: (progress: TrainingProgress) => void
}

/**
 * Embedding, a recurrent layer and a linear read-out: one prediction per position (tagging), or one from the last
 * position (`final`), over the vocabulary.
 */
export class RecurrentTagger extends Composite {
  readonly kind = 'RecurrentTagger'
  readonly vocabulary: number
  readonly final: boolean

  constructor(
    vocabulary: number,
    options: { cell: RecurrentCell; hidden: number; embedding: number; final: boolean; key: Key },
  ) {
    const Cell = options.cell === 'gru' ? GRU : options.cell === 'lstm' ? LSTM : RNN
    super({
      embed: new Embedding(vocabulary, options.embedding, { key: foldIn(options.key, 0) }),
      recurrent: new Cell(options.embedding, options.hidden, { batchFirst: true, key: foldIn(options.key, 1) }),
      head: new Linear(options.hidden, vocabulary, { key: foldIn(options.key, 2) }),
    })
    this.vocabulary = vocabulary
    this.final = options.final
  }

  /** The hidden states `[B, T, H]` of token ids `[B, T]`. */
  hiddenStates(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const e = p.run<Value, Value>('embed', ids(x))
    const h = p.run<Value, RecurrentOutput>('recurrent', e).output as Value
    p.done()
    return h
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const p = this.parts(params, ctx)
    const e = p.run<Value, Value>('embed', ids(x))
    const h = p.run<Value, RecurrentOutput>('recurrent', e).output as Value
    const [b = 0, t = 0, units = 0] = h.shape
    const read = this.final ? reshape(slice(h, [0, t - 1, 0], [b, t, units]), [b, units]) : h
    const y = p.run<Value, Value>('head', read)
    p.done()
    return y
  }

  outputShape(s: readonly number[]): number[] {
    return this.final ? [s[0] ?? 1, this.vocabulary] : [s[0] ?? 1, s[1] ?? 1, this.vocabulary]
  }
}

// Token ids as an int32 tensor (the training rows arrive as floats).
function ids(x: Value): Tensor {
  return fromStorage(Int32Array.from(toFlat(x)), [...x.shape], 'int32') as Tensor
}

/** What {@link recurrentTagger} returns. */
export interface RecurrentTaggerResult {
  readonly cell: RecurrentCell
  /** Whether one prediction is made per position (tagging) or one per sequence (`final`). */
  readonly final: boolean
  /** Token accuracy on the training and validation sequences (padding ignored). */
  readonly trainAccuracy: number
  readonly validationAccuracy: number
  /** The fraction of validation sequences predicted entirely right. */
  readonly sequenceAccuracy: number
  /** Predicted target ids for some input sequences. */
  readonly predict: (inputs: readonly (readonly number[])[]) => number[][]
  readonly describe: () => Structure
}

/**
 * A recurrent network (Elman RNN, GRU or LSTM) on a toy sequence task of `lib/datasets` (running parity, copying,
 * Dyck languages...): token embeddings, the recurrence and a linear read-out at every position when the targets are as
 * long as the inputs (sequence labelling) or from the last position when they are one token, trained by Adam on the
 * cross-entropy with padding ignored. Exposition, in order: the loss and token accuracy per epoch (training against
 * held-out sequences), the accuracy per position (where recurrences fail first), the hidden states of one held-out
 * sequence (units against positions) at about 30 geometrically spaced steps, and its input, target and predicted
 * tokens at the same steps.
 *
 * @param data - A sequence dataset (`inputs`, `targets`, `padId`, `vocabulary`).
 * @param options - Cell, sizes, epochs, batching, rate, hold-out, snapshots and progress.
 * @param fit - The key, tracer and policy.
 * @returns The accuracies, a predictor and the exposition records.
 */
export function recurrentTagger(
  data: SequenceData,
  options: RecurrentTaggerOptions = {},
  fit: FitOptions = {},
): Explained<RecurrentTaggerResult> {
  const key = fit.key ?? makeKey(0)
  const [n = 0, tIn = 0] = data.inputs.shape
  const tOut = data.targets.shape[1] ?? 0
  const final = tOut === 1 && tIn > 1
  if (!final && tOut !== tIn)
    throw new MdmlError(
      'shape',
      `recurrentTagger: targets of length ${tOut} for inputs of length ${tIn} (needs equal lengths, or one target)`,
    )
  const vocabulary = data.vocabulary.length
  const cell = options.cell ?? 'rnn'
  const all: Rows = {
    x: fromStorage(Float64Array.from(data.inputs.data), [n, tIn], 'float64') as Tensor,
    y: fromStorage(Float64Array.from(data.targets.data), [n, tOut], 'float64') as Tensor,
  }
  const parts = holdOut(all, options.validationFraction ?? 0.25, foldIn(key, 7), false)
  const net = new RecurrentTagger(vocabulary, {
    cell,
    hidden: options.hidden ?? 16,
    embedding: options.embedding ?? 4,
    final,
    key: foldIn(key, 1),
  })
  const pad = data.padId
  const logitsOf = (p: Value): Value => (final ? p : reshape(p, [(p.shape[0] ?? 0) * tIn, vocabulary]))
  const targetsOf = (t: Value): Value => reshape(t, [(t.shape[0] ?? 0) * tOut])
  const loss = (p: Value, t: Value): Value => crossEntropy(logitsOf(p), targetsOf(t), { ignoreIndex: pad })
  const tokenAccuracy = (p: Tensor, t: Tensor): number => {
    const predicted = toFlat(argmax(p, -1))
    const truth = toFlat(t)
    let right = 0
    let total = 0
    for (const [i, y] of truth.entries()) {
      if (y === pad) continue
      total++
      if (predicted[i] === y) right++
    }
    return total === 0 ? Number.NaN : right / total
  }
  const tapRows = parts.validation
  const tap = Math.min(Math.max(0, options.tapRow ?? 0), (tapRows.x.shape[0] ?? 1) - 1)
  const tapX = fromStorage(
    Float64Array.from(toFlat(tapRows.x).slice(tap * tIn, (tap + 1) * tIn)),
    [1, tIn],
    'float64',
  ) as Tensor
  const tapY = toFlat(tapRows.y).slice(tap * tOut, (tap + 1) * tOut)
  const hiddenMap = (params: Params): number[][] => {
    const h = toFlat(net.hiddenStates(params, tapX, { train: false }))
    const units = h.length / tIn
    // Units down, positions across.
    return Array.from({ length: units }, (_, u) => Array.from({ length: tIn }, (_, t) => h[t * units + u] ?? 0))
  }
  const tokens = (params: Params): number[][] => {
    const predicted = toFlat(argmax(net.apply(params, tapX, { train: false }), -1))
    const target = final ? [...new Array<number>(tIn - 1).fill(pad), tapY[0] ?? pad] : tapY
    const guess = final ? [...new Array<number>(tIn - 1).fill(pad), predicted[0] ?? pad] : predicted
    return [toFlat(tapX), target, guess]
  }
  const epochs = options.epochs ?? 80
  const training = train({
    name: `recurrentTagger/${cell}`,
    model: net,
    train: parts.train,
    validation: parts.validation,
    loss,
    score: tokenAccuracy,
    optimizer: optimiserOf('adam', options.learningRate ?? 0.02),
    learningRate: options.learningRate ?? 0.02,
    epochs,
    ...(options.batchSize === undefined ? {} : { batchSize: options.batchSize }),
    snapshots: options.snapshots ?? 30,
    snapshotUnit: 'step',
    fit: { ...fit, key: foldIn(key, 2) },
    ...(options.onProgress === undefined
      ? {}
      : {
          onEpoch: progressReporter(options.onProgress, (s) => [
            valueGrids({
              grids: [{ name: 'hidden states', values: hiddenMap(s.params) }],
              diverging: true,
              axes: ['hidden unit', 'position'],
              valueName: 'state',
            }),
          ]),
        }),
  })
  net.setParams(training.params)
  const predictIds = (rows: Tensor): number[] => toFlat(argmax(net.apply(training.params, rows, { train: false }), -1))
  const perPosition = (rows: Rows): number[] => {
    const predicted = predictIds(rows.x)
    const truth = toFlat(rows.y)
    return Array.from({ length: tOut }, (_, t) => {
      let right = 0
      let total = 0
      for (let i = 0; i < truth.length / tOut; i++) {
        const y = truth[i * tOut + t] ?? pad
        if (y === pad) continue
        total++
        if (predicted[i * tOut + t] === y) right++
      }
      return total === 0 ? Number.NaN : right / total
    })
  }
  const vPredicted = predictIds(parts.validation.x)
  const vTruth = toFlat(parts.validation.y)
  const nv = vTruth.length / tOut
  let wholeRight = 0
  for (let i = 0; i < nv; i++) {
    let ok = true
    for (let t = 0; t < tOut; t++) {
      const y = vTruth[i * tOut + t] ?? pad
      if (y !== pad && vPredicted[i * tOut + t] !== y) ok = false
    }
    if (ok) wholeRight++
  }
  const iterations = training.snapshots.map((s) => s.step)
  const objective = lossAtSnapshots(training)
  const stepsOf = (map: (p: Params) => number[][]) =>
    training.snapshots.map((s, t) => ({
      iteration: iterations[t] ?? t,
      grids: [map(s.params)],
      ...(Number.isFinite(objective[t]) ? { objective: objective[t] as number } : {}),
    }))
  const hiddenRecord = valueGrids({
    grids: [{ name: 'hidden states', values: hiddenMap(training.params) }],
    diverging: true,
    axes: ['hidden unit', 'position'],
    valueName: 'state',
    steps: stepsOf(hiddenMap),
    stepName: 'step',
  })
  const tokenRecord = valueGrids({
    grids: [{ name: 'input, target, predicted', values: tokens(training.params) }],
    axes: ['input · target · predicted', 'position'],
    valueName: 'token',
    steps: stepsOf(tokens),
    stepName: 'step',
  })
  const exposition: Exposition[] = [
    lossCurve(training),
    ...(scoreCurve(training, 'token accuracy') === undefined
      ? []
      : [scoreCurve(training, 'token accuracy') as Exposition]),
    learningCurve({
      sizes: Array.from({ length: tOut }, (_, t) => t + 1),
      train: perPosition(parts.train),
      validation: perPosition(parts.validation),
      resource: 'position',
      labels: { train: 'training accuracy', validation: 'validation accuracy' },
    }),
    hiddenRecord,
    tokenRecord,
    diagramRecord({ structure: describe(net, [1, tIn]) }),
  ]
  return {
    cell,
    final,
    trainAccuracy: training.trainScore?.at(-1) ?? Number.NaN,
    validationAccuracy: training.validationScore?.at(-1) ?? Number.NaN,
    sequenceAccuracy: nv === 0 ? Number.NaN : wholeRight / nv,
    predict: (inputs) => {
      const rows = fromStorage(Float64Array.from(inputs.flat()), [inputs.length, tIn], 'float64') as Tensor
      const p = predictIds(rows)
      return Array.from({ length: inputs.length }, (_, i) => p.slice(i * tOut, (i + 1) * tOut))
    },
    describe: () => describe(net, [1, tIn]),
    exposition,
  }
}
