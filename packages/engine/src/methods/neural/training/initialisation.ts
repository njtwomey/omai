import { valueAndGrad } from '../../../lib/autodiff/index.ts'
import { WhiteboxError } from '../../../lib/base/index.ts'
import {
  curveSet,
  type Explained,
  type Exposition,
  learningCurve,
  type SearchTrials,
  searchTrials,
} from '../../../lib/exposition/index.ts'
import type { FitOptions, LearnData } from '../../../lib/learn/index.ts'
import type { ActivationName, Params } from '../../../lib/nn/index.ts'
import { foldIn, type Key, key as makeKey } from '../../../lib/random/index.ts'
import { item, norm, type Tensor, toFlat } from '../../../lib/tensor/index.ts'
import {
  buildMlp,
  DEMO_LEARNING_RATES,
  hiddenActivations,
  type InitScheme,
  isRegression,
  linearNames,
  lossOf,
  type OptimiserName,
  optimiserOf,
  scoreOf,
  surfaceOf,
  type Task,
  taskOf,
} from '../_nets.ts'
import { lossCurve, spansDecades, studyCurves } from '../_records.ts'
import { type Training, train } from '../_train.ts'

/** Options shared by the initialisation and normalisation studies. */
export interface DeepNetStudyOptions {
  /** Hidden layers (default 6 for initialisation, 4 for normalisation). */
  readonly depth?: number
  /** Units per hidden layer (default 32). */
  readonly width?: number
  readonly activation?: ActivationName
  readonly optimiser?: OptimiserName
  readonly learningRate?: number
  /** Epochs (default 30). */
  readonly epochs?: number
  /** Minibatch size (default 50). */
  readonly batchSize?: number
  /** Decision-surface grid points per axis (default 30). */
  readonly resolution?: number
}

/** Options of {@link initialisationStudy}. */
export interface InitialisationStudyOptions extends DeepNetStudyOptions {
  /** The schemes compared (default N(0, 0.01²), Xavier, Kaiming and N(0, 1)). */
  readonly schemes?: readonly InitScheme[]
}

/** The forward and backward statistics of one network at initialisation. */
export interface LayerStatistics {
  readonly scheme: string
  /** The standard deviation of each hidden layer's activations over the rows, then the output layer's. */
  readonly activationStd: readonly number[]
  /** The norm of the loss gradient for each linear layer's weight. */
  readonly gradientNorm: readonly number[]
  readonly finalLoss: number
  readonly score: number
}

function std(t: Tensor): number {
  const v = toFlat(t)
  const m = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, v.length))
}

function trainDeep(
  name: string,
  task: Task,
  net: ReturnType<typeof buildMlp>,
  options: DeepNetStudyOptions,
  key: Key,
  fit: FitOptions,
  fallbackOptimiser: OptimiserName,
): Training {
  const optimiser = options.optimiser ?? fallbackOptimiser
  const learningRate = options.learningRate ?? DEMO_LEARNING_RATES[optimiser]
  return train({
    name,
    model: net,
    train: task.rows,
    loss: lossOf(task.output),
    score: scoreOf(task.output),
    optimizer: optimiserOf(optimiser, learningRate),
    learningRate,
    epochs: options.epochs ?? 30,
    batchSize: options.batchSize ?? 50,
    snapshots: 20,
    fit: { ...fit, key: foldIn(key, 2) },
  })
}

function summary(method: string, parameter: string, names: readonly string[], runs: readonly Training[]): SearchTrials {
  const finals = runs.map((r) => r.evalLoss.at(-1) ?? Number.NaN)
  const finite = finals.filter(Number.isFinite)
  const best = finite.length === 0 ? undefined : finals.indexOf(Math.min(...finite))
  return searchTrials({
    method,
    parameters: [parameter],
    trials: runs.map((r, i) => ({
      params: { [parameter]: names[i] ?? '' },
      value: finals[i] ?? Number.NaN,
      intermediate: [...r.evalLoss],
      state: Number.isFinite(finals[i] ?? Number.NaN) ? ('complete' as const) : ('failed' as const),
    })),
    higherIsBetter: false,
    ...(best === undefined ? {} : { best }),
  })
}

/**
 * How initialisation shapes a deep network: for each scheme (zeros, N(0, 0.01²), Xavier/Glorot, Kaiming/He, N(0, 1)),
 * the spread of every hidden layer's activations and the size of every layer's weight gradient at initialisation
 * (vanishing or exploding with depth), then how the network trains from there. Exposition, in order: one per-layer
 * statistics curve per scheme (activation standard deviation and gradient norm against depth), one loss curve per
 * scheme, the schemes as trials (final loss, with the loss per epoch), then as `curves` records: every scheme's
 * activation spread by layer, its gradient norm by layer, and its loss per epoch.
 *
 * @param data - Classification (default) or regression data.
 * @param options - Schemes, depth, width, activation, optimiser, learning rate, epochs and batch size.
 * @param fit - The key, tracer and policy.
 * @returns The statistics per scheme and the exposition records.
 */
export function initialisationStudy(
  data: LearnData,
  options: InitialisationStudyOptions = {},
  fit: FitOptions = {},
): Explained<{ readonly schemes: readonly LayerStatistics[] }> {
  const schemes = options.schemes ?? ['small', 'xavier', 'kaiming', 'large']
  if (schemes.length === 0) throw new WhiteboxError('value', 'initialisationStudy: needs at least one scheme')
  const key = fit.key ?? makeKey(0)
  const task = taskOf(data, !isRegression(data), { who: 'initialisationStudy', key })
  const d = task.x.shape[1] ?? 1
  const depth = options.depth ?? 6
  const width = options.width ?? 32
  const sizes = [d, ...Array.from({ length: depth }, () => width), task.nOut]
  const loss = lossOf(task.output)
  const stats: LayerStatistics[] = []
  const layerCurves: Exposition[] = []
  const runs: Training[] = []
  for (const scheme of schemes) {
    const net = buildMlp({
      sizes,
      activation: options.activation ?? 'tanh',
      init: scheme,
      key: foldIn(key, 1),
    })
    // Each hidden layer's activations, then the output layer's (the logits).
    const activationStd = [
      ...hiddenActivations(net, net.params, task.rows.x),
      net.apply(net.params, task.rows.x, { train: false }) as Tensor,
    ].map(std)
    const names = linearNames(net)
    const g = valueAndGrad((p: Params) => loss(net.apply(p, task.rows.x, { train: false }), task.rows.y))(net.params)
      .grad as Params
    const gradientNorm = names.map((n) => item(norm((g[n] as { readonly weight: Tensor }).weight)))
    layerCurves.push(
      learningCurve({
        sizes: names.map((_, l) => l + 1),
        train: activationStd,
        validation: gradientNorm,
        resource: 'layer',
        labels: { train: `activation std (${scheme})`, validation: `weight-gradient norm (${scheme})` },
      }),
    )
    const training = trainDeep(`initialisationStudy/${scheme}`, task, net, options, key, fit, 'sgd')
    runs.push(training)
    stats.push({
      scheme,
      activationStd,
      gradientNorm,
      finalLoss: training.evalLoss.at(-1) ?? Number.NaN,
      score: training.trainScore?.at(-1) ?? Number.NaN,
    })
  }
  // Every scheme on one chart per statistic: how the spread and the gradients change with depth.
  const byLayer = (field: 'activationStd' | 'gradientNorm', yLabel: string) => {
    const series = stats.map((s) => ({ name: s.scheme, x: s[field].map((_, l) => l + 1), y: [...s[field]] }))
    return curveSet({ series, xLabel: 'layer', yLabel, yLog: spansDecades(series.flatMap((q) => q.y)) })
  }
  return {
    schemes: stats,
    exposition: [
      ...layerCurves,
      ...runs.map(lossCurve),
      summary('initialisation', 'init', schemes, runs),
      byLayer('activationStd', 'activation std'),
      byLayer('gradientNorm', 'weight-gradient norm'),
      studyCurves(schemes, runs),
    ],
  }
}

/** Options of {@link normalisationStudy}. */
export interface NormalisationStudyOptions extends DeepNetStudyOptions {
  /** The normalisations compared (default none, batch and layer). */
  readonly norms?: readonly ('none' | 'batch' | 'layer')[]
}

/**
 * Batch and layer normalisation against none on the same deep network from the same initial weights: how each trains
 * and what it fits. Exposition, in order: one decision surface per normalisation, one loss curve per normalisation,
 * the normalisations as trials (final loss, with the loss per epoch), and every normalisation's loss per epoch as one
 * `curves` record.
 *
 * @param data - Classification (default) or regression data.
 * @param options - Normalisations, depth, width, activation, optimiser, learning rate, epochs and batch size.
 * @param fit - The key, tracer and policy.
 * @returns The final loss and score per normalisation and the exposition records.
 */
export function normalisationStudy(
  data: LearnData,
  options: NormalisationStudyOptions = {},
  fit: FitOptions = {},
): Explained<{
  readonly results: readonly { readonly norm: string; readonly finalLoss: number; readonly score: number }[]
}> {
  const norms = options.norms ?? ['none', 'batch', 'layer']
  const key = fit.key ?? makeKey(0)
  const task = taskOf(data, !isRegression(data), { who: 'normalisationStudy', key })
  const d = task.x.shape[1] ?? 1
  const depth = options.depth ?? 4
  const sizes = [d, ...Array.from({ length: depth }, () => options.width ?? 32), task.nOut]
  const surfaces: Exposition[] = []
  const runs: Training[] = []
  for (const n of norms) {
    const net = buildMlp({
      sizes,
      activation: options.activation ?? 'relu',
      init: 'kaiming',
      norm: n,
      key: foldIn(key, 1),
    })
    const training = trainDeep(`normalisationStudy/${n}`, task, net, options, key, fit, 'sgd')
    runs.push(training)
    surfaces.push(surfaceOf(net, training.params, training.buffers, task, options.resolution ?? 30, training))
  }
  return {
    results: norms.map((n, i) => ({
      norm: n,
      finalLoss: runs[i]?.evalLoss.at(-1) ?? Number.NaN,
      score: runs[i]?.trainScore?.at(-1) ?? Number.NaN,
    })),
    exposition: [
      ...surfaces,
      ...runs.map(lossCurve),
      summary('normalisation', 'norm', norms, runs),
      studyCurves(norms, runs),
    ],
  }
}
