import { WhiteboxError } from '../../../lib/base/index.ts'
import {
  type Explained,
  type Exposition,
  learningCurve,
  type SearchTrials,
  searchTrials,
} from '../../../lib/exposition/index.ts'
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
} from '../_nets.ts'
import { lossCurve, studyCurves, type TrainingProgress } from '../_records.ts'
import { type Training, train } from '../_train.ts'

/** Options of {@link optimiserRace}. */
export interface OptimiserRaceOptions {
  /** The optimisers raced (default SGD, momentum, RMSprop and Adam). */
  readonly optimisers?: readonly OptimiserName[]
  /** A learning rate per optimiser (defaults that suit each on small networks). */
  readonly learningRates?: Partial<Readonly<Record<OptimiserName, number>>>
  /** Hidden widths (default [16, 16]). */
  readonly hiddenLayerSizes?: readonly number[]
  /** Hidden activation (default 'tanh'). */
  readonly activation?: ActivationName
  /** Epochs (default 60). */
  readonly epochs?: number
  /** Minibatch size (default 32). */
  readonly batchSize?: number
  /** Decision-surface grid points per axis (default 30). */
  readonly resolution?: number
  /** Called as each optimiser trains, with the records so far. */
  readonly onProgress?: (progress: TrainingProgress) => void
}

/** One optimiser's outcome in a race. */
export interface RaceResult {
  readonly optimiser: OptimiserName
  readonly learningRate: number
  /** The training loss at the end (eval mode, all rows). */
  readonly finalLoss: number
  /** The training accuracy (or R²) at the end. */
  readonly score: number
  /** The first epoch whose loss fell below twice the best final loss of the race (how fast it got close). */
  readonly epochsToNear: number
  readonly losses: readonly number[]
}

/**
 * The same network from the same initial weights, trained on the same minibatches by several optimisers: SGD, SGD
 * with momentum or Nesterov momentum, RMSprop, Adagrad, Adam, AdamW. Exposition, in order: one decision surface per
 * optimiser (the fitted networks side by side), one loss curve per optimiser, the race as trials (final loss per
 * optimiser, its loss per epoch as the intermediate values), and every optimiser's loss per epoch as one `curves`
 * record.
 *
 * @param data - Classification (default) or regression data (a dataset of kind 'regression').
 * @param options - Optimisers, learning rates, network, epochs, batch size and progress.
 * @param fit - The key (one initialisation and one shuffle for every optimiser), tracer and policy.
 * @returns The results per optimiser and the exposition records.
 */
export function optimiserRace(
  data: LearnData,
  options: OptimiserRaceOptions = {},
  fit: FitOptions = {},
): Explained<{ readonly results: readonly RaceResult[] }> {
  const optimisers = options.optimisers ?? ['sgd', 'momentum', 'rmsprop', 'adam']
  if (optimisers.length === 0) throw new WhiteboxError('value', 'optimiserRace: needs at least one optimiser')
  const key = fit.key ?? makeKey(0)
  const task = taskOf(data, !isRegression(data), { who: 'optimiserRace', key })
  const d = task.x.shape[1] ?? 1
  const sizes = [d, ...(options.hiddenLayerSizes ?? [16, 16]), task.nOut]
  const epochs = options.epochs ?? 60
  const surfaces: Exposition[] = []
  const curves: Exposition[] = []
  const runs: { optimiser: OptimiserName; learningRate: number; losses: number[]; finalLoss: number; score: number }[] =
    []
  const trainings: Training[] = []
  const label = (o: OptimiserName, lr: number): string => `${o} (lr ${lr})`
  for (const [k, optimiser] of optimisers.entries()) {
    const learningRate = options.learningRates?.[optimiser] ?? DEMO_LEARNING_RATES[optimiser]
    // Every optimiser starts from the same weights and sees the same minibatches.
    const net = buildMlp({ sizes, activation: options.activation ?? 'tanh', key: foldIn(key, 1) })
    const training = train({
      name: `optimiserRace/${optimiser}`,
      model: net,
      train: task.rows,
      loss: lossOf(task.output),
      score: scoreOf(task.output),
      optimizer: optimiserOf(optimiser, learningRate),
      learningRate,
      epochs,
      batchSize: options.batchSize ?? 32,
      snapshots: 20,
      fit: { ...fit, key: foldIn(key, 2) },
      ...(options.onProgress === undefined
        ? {}
        : {
            onEpoch: (r) => {
              if (r.epoch % Math.max(1, Math.floor(epochs / 10)) !== 0 && r.epoch !== epochs) return
              options.onProgress?.({
                fraction: (k + r.epoch / epochs) / optimisers.length,
                label: `${optimiser}: epoch ${r.epoch} of ${epochs}`,
                exposition: [
                  ...surfaces,
                  ...curves,
                  ...(trainings.length === 0
                    ? []
                    : [
                        studyCurves(
                          runs.map((q) => label(q.optimiser, q.learningRate)),
                          trainings,
                        ),
                      ]),
                ],
              })
            },
          }),
    })
    surfaces.push(surfaceOf(net, training.params, training.buffers, task, options.resolution ?? 30, training))
    curves.push(lossCurve(training))
    trainings.push(training)
    runs.push({
      optimiser,
      learningRate,
      losses: [...training.evalLoss],
      finalLoss: training.evalLoss.at(-1) ?? Number.NaN,
      score: training.trainScore?.at(-1) ?? Number.NaN,
    })
  }
  const best = Math.min(...runs.map((r) => r.finalLoss).filter(Number.isFinite))
  const results: RaceResult[] = runs.map((r) => {
    const near = r.losses.findIndex((l) => l <= 2 * best)
    return { ...r, epochsToNear: near < 0 ? Number.POSITIVE_INFINITY : near + 1 }
  })
  const trials: SearchTrials = searchTrials({
    method: 'optimiser race',
    parameters: ['optimizer', 'learningRate'],
    trials: results.map((r) => ({
      params: { optimizer: r.optimiser, learningRate: r.learningRate },
      value: r.finalLoss,
      intermediate: r.losses,
      state: Number.isFinite(r.finalLoss) ? ('complete' as const) : ('failed' as const),
    })),
    higherIsBetter: false,
    best: results.findIndex((r) => r.finalLoss === best),
  })
  const race = studyCurves(
    runs.map((r) => label(r.optimiser, r.learningRate)),
    trainings,
  )
  return { results, exposition: [...surfaces, ...curves, trials, race] }
}

/** Options of {@link learningRateFinder}. */
export interface LearningRateFinderOptions {
  readonly minLearningRate?: number
  readonly maxLearningRate?: number
  /** Minibatch steps, the rate growing geometrically from the least to the most (default 100). */
  readonly steps?: number
  /** The optimiser whose rate is swept (default 'sgd'). */
  readonly optimiser?: OptimiserName
  readonly hiddenLayerSizes?: readonly number[]
  readonly activation?: ActivationName
  readonly batchSize?: number
  /** Stop once the smoothed loss exceeds this multiple of its least value (default 4). */
  readonly divergence?: number
}

/** What {@link learningRateFinder} returns. */
export interface LearningRateFinderResult {
  readonly learningRates: readonly number[]
  readonly losses: readonly number[]
  /** The losses smoothed by a debiased exponential moving average (β = 0.98, fastai's). */
  readonly smoothed: readonly number[]
  /** The rate where the smoothed loss falls fastest (steepest descent in log rate). */
  readonly steepest: number
  /** A tenth of the rate at the least smoothed loss (the other common rule). */
  readonly minimumOverTen: number
}

/**
 * Leslie Smith's learning-rate range test (fastai's lr_find): train a fresh network for one sweep of minibatch steps
 * whose learning rate grows geometrically, recording the loss at each, until the loss blows up; suggest the rate where
 * the smoothed loss falls fastest, and a tenth of the rate at its least. Exposition: the loss and the smoothed loss
 * against the learning rate.
 *
 * @param data - Classification (default) or regression data.
 * @param options - The rate range, steps, optimiser, network, batch size and divergence threshold.
 * @param fit - The key, tracer and policy.
 * @returns The rates, losses, suggestions and the exposition record.
 */
export function learningRateFinder(
  data: LearnData,
  options: LearningRateFinderOptions = {},
  fit: FitOptions = {},
): Explained<LearningRateFinderResult> {
  const lo = options.minLearningRate ?? 1e-5
  const hi = options.maxLearningRate ?? 10
  const steps = options.steps ?? 100
  if (!(lo > 0 && hi > lo))
    throw new WhiteboxError('value', 'learningRateFinder: needs 0 < minLearningRate < maxLearningRate')
  const key = fit.key ?? makeKey(0)
  const task = taskOf(data, !isRegression(data), { who: 'learningRateFinder', key })
  const d = task.x.shape[1] ?? 1
  const net = buildMlp({
    sizes: [d, ...(options.hiddenLayerSizes ?? [16, 16]), task.nOut],
    activation: options.activation ?? 'tanh',
    key: foldIn(key, 1),
  })
  const rate = (u: number): number => lo * (hi / lo) ** (Math.min(u, steps - 1) / Math.max(1, steps - 1))
  const n = task.rows.x.shape[0] ?? 0
  const batchSize = Math.min(n, options.batchSize ?? 32)
  const perEpoch = Math.ceil(n / batchSize)
  const training = train({
    name: 'learningRateFinder',
    model: net,
    train: task.rows,
    loss: lossOf(task.output),
    optimizer: optimiserOf(options.optimiser ?? 'sgd', rate),
    learningRate: rate,
    epochs: Math.ceil(steps / perEpoch),
    batchSize,
    snapshots: 2,
    fit: { ...fit, key: foldIn(key, 2) },
  })
  const raw = training.stepLoss.slice(0, steps)
  const learningRates: number[] = []
  const losses: number[] = []
  const smoothed: number[] = []
  let avg = 0
  let least = Number.POSITIVE_INFINITY
  for (const [t, loss] of raw.entries()) {
    if (!Number.isFinite(loss)) break
    avg = 0.98 * avg + 0.02 * loss
    const s = avg / (1 - 0.98 ** (t + 1))
    learningRates.push(rate(t))
    losses.push(loss)
    smoothed.push(s)
    least = Math.min(least, s)
    if (t > 5 && s > (options.divergence ?? 4) * least) break
  }
  let steepest = learningRates[0] ?? lo
  let slope = Number.POSITIVE_INFINITY
  // fastai skips the first ten and the last five points, where the average is still settling or blowing up.
  const first = smoothed.length > 20 ? 10 : 1
  const last = smoothed.length > 20 ? smoothed.length - 5 : smoothed.length
  for (let t = first; t < last; t++) {
    const g =
      ((smoothed[t] ?? 0) - (smoothed[t - 1] ?? 0)) / Math.log((learningRates[t] ?? 1) / (learningRates[t - 1] ?? 1))
    if (g < slope) {
      slope = g
      steepest = learningRates[t] ?? steepest
    }
  }
  const atLeast = learningRates[smoothed.indexOf(least)] ?? lo
  return {
    learningRates,
    losses,
    smoothed,
    steepest,
    minimumOverTen: atLeast / 10,
    exposition: [
      learningCurve({
        sizes: learningRates,
        train: losses,
        validation: smoothed,
        resource: 'learning rate',
        labels: { train: 'minibatch loss', validation: 'smoothed loss' },
      }),
    ],
  }
}
