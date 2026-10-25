/**
 * A study of regularisation against overfitting: the same over-sized network trained with each dropout rate or
 * weight-decay strength, scored on training and held-out rows, so the gap between them can be seen to close.
 */

import { WhiteboxError } from '../../../lib/base/index.ts'
import { type Explained, type Exposition, learningCurve } from '../../../lib/exposition/index.ts'
import type { FitOptions, LearnData } from '../../../lib/learn/index.ts'
import type { ActivationName } from '../../../lib/nn/index.ts'
import { foldIn, key as makeKey } from '../../../lib/random/index.ts'
import {
  buildMlp,
  DEMO_LEARNING_RATES,
  isRegression,
  lossOf,
  type OptimiserName,
  optimiserOf,
  scoreOf,
  surfaceOf,
  taskOf,
  weightMask,
} from '../_nets.ts'
import { lossCurve, studyCurves, type TrainingProgress } from '../_records.ts'
import { type Training, train } from '../_train.ts'

/** Options of {@link regularisationStudy}. */
export interface RegularisationStudyOptions {
  /** Dropout after every hidden layer, or an L2 penalty on the weights (weight decay; default 'dropout'). */
  readonly kind?: 'dropout' | 'weightDecay'
  /** The strengths compared (default [0, 0.2, 0.5] for dropout, [0, 1e-3, 1e-2, 1e-1] for weight decay). */
  readonly values?: readonly number[]
  /** The fraction of rows held out for validation (default 0.3). */
  readonly validationFraction?: number
  /** Hidden widths (default [32, 32]: large enough to overfit small data). */
  readonly hiddenLayerSizes?: readonly number[]
  readonly activation?: ActivationName
  readonly optimiser?: OptimiserName
  readonly learningRate?: number
  /** Epochs (default 150). */
  readonly epochs?: number
  /** Minibatch size (default all the rows). */
  readonly batchSize?: number
  readonly resolution?: number
  readonly onProgress?: (progress: TrainingProgress) => void
}

/** One strength's outcome. */
export interface RegularisationResult {
  readonly value: number
  readonly trainScore: number
  readonly validationScore: number
  readonly trainLoss: number
  readonly validationLoss: number
}

/**
 * Regularisation against overfitting: the same over-sized network trained with each dropout rate (or weight decay),
 * scored on the training rows and on held-out rows. Exposition, in order: the validation curve (training and
 * validation score against the strength), one decision surface per strength (wiggly without, smoother with), one
 * loss curve per strength (training against validation loss per epoch: the gap is the overfitting), and every
 * strength's training and validation loss as one `curves` record.
 *
 * @param data - Classification (default) or regression data.
 * @param options - Kind, strengths, hold-out fraction, network, optimiser, epochs, batch size and progress.
 * @param fit - The key (one split, one initialisation and one shuffle for every strength), tracer and policy.
 * @returns The results per strength and the exposition records.
 */
export function regularisationStudy(
  data: LearnData,
  options: RegularisationStudyOptions = {},
  fit: FitOptions = {},
): Explained<{ readonly results: readonly RegularisationResult[] }> {
  const kind = options.kind ?? 'dropout'
  const values = options.values ?? (kind === 'dropout' ? [0, 0.2, 0.5] : [0, 1e-3, 1e-2, 1e-1])
  if (values.length === 0) throw new WhiteboxError('value', 'regularisationStudy: needs at least one strength')
  if (kind === 'dropout' && values.some((v) => !(v >= 0 && v < 1)))
    throw new WhiteboxError('value', 'regularisationStudy: dropout rates must be in [0, 1)')
  const key = fit.key ?? makeKey(0)
  const task = taskOf(data, !isRegression(data), {
    who: 'regularisationStudy',
    key,
    holdOut: options.validationFraction ?? 0.3,
  })
  const d = task.x.shape[1] ?? 1
  const sizes = [d, ...(options.hiddenLayerSizes ?? [32, 32]), task.nOut]
  const optimiser = options.optimiser ?? 'adam'
  const learningRate = options.learningRate ?? DEMO_LEARNING_RATES[optimiser]
  const epochs = options.epochs ?? 150
  const surfaces: Exposition[] = []
  const curves: Exposition[] = []
  const results: RegularisationResult[] = []
  const runs: Training[] = []
  for (const [k, value] of values.entries()) {
    const net = buildMlp({
      sizes,
      activation: options.activation ?? 'relu',
      ...(kind === 'dropout' ? { dropout: value } : {}),
      key: foldIn(key, 1),
    })
    const training = train({
      name: `regularisationStudy/${value}`,
      model: net,
      train: task.rows,
      ...(task.validation === undefined ? {} : { validation: task.validation }),
      loss: lossOf(task.output),
      score: scoreOf(task.output),
      optimizer: optimiserOf(
        optimiser,
        learningRate,
        kind === 'weightDecay' ? { weightDecay: value, mask: weightMask(net) } : {},
      ),
      learningRate,
      epochs,
      ...(options.batchSize === undefined ? {} : { batchSize: options.batchSize }),
      snapshots: 20,
      fit: { ...fit, key: foldIn(key, 2) },
      ...(options.onProgress === undefined
        ? {}
        : {
            onEpoch: (r) => {
              if (r.epoch % Math.max(1, Math.floor(epochs / 10)) !== 0 && r.epoch !== epochs) return
              options.onProgress?.({
                fraction: (k + r.epoch / epochs) / values.length,
                label: `${kind} ${value}: epoch ${r.epoch} of ${epochs}`,
                exposition: [...surfaces, ...curves],
              })
            },
          }),
    })
    surfaces.push(surfaceOf(net, training.params, training.buffers, task, options.resolution ?? 30, training))
    curves.push(lossCurve(training))
    runs.push(training)
    results.push({
      value,
      trainScore: training.trainScore?.at(-1) ?? Number.NaN,
      validationScore: training.validationScore?.at(-1) ?? Number.NaN,
      trainLoss: training.evalLoss.at(-1) ?? Number.NaN,
      validationLoss: training.validationLoss?.at(-1) ?? Number.NaN,
    })
  }
  const scoreName = task.classifier ? 'accuracy' : 'R²'
  const validation = learningCurve({
    sizes: [...values],
    train: results.map((r) => r.trainScore),
    validation: results.map((r) => r.validationScore),
    resource: kind === 'dropout' ? 'dropout rate' : 'weight decay',
    labels: { train: `training ${scoreName}`, validation: `validation ${scoreName}` },
  })
  const all = studyCurves(
    values.map((v) => `${kind === 'dropout' ? 'dropout' : 'decay'} ${v}`),
    runs,
  )
  return { results, exposition: [validation, ...surfaces, ...curves, all] }
}
