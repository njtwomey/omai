/**
 * Universal approximation in one dimension (Cybenko, Hornik): one hidden layer of increasing width fitted to the same
 * curve, showing the fit improve with the number of units and how the widest network builds the curve from its
 * individual units.
 */

import { MdmlError } from '../../../lib/base/index.ts'
import { type Explained, type Exposition, learningCurve } from '../../../lib/exposition/index.ts'
import type { FitOptions, LearnData } from '../../../lib/learn/index.ts'
import type { ActivationName, Params } from '../../../lib/nn/index.ts'
import { foldIn, key as makeKey } from '../../../lib/random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../../lib/tensor/index.ts'
import { buildMlp, hiddenActivations, linearNames, lossOf, optimiserOf, scoreOf, surfaceOf, taskOf } from '../_nets.ts'
import { mapSteps } from '../_records.ts'
import { train } from '../_train.ts'

/** Options of {@link universalApproximation}. */
export interface UniversalApproximationOptions {
  /** Hidden widths compared (default 1, 2, 4, 8, 16, 32). */
  readonly widths?: readonly number[]
  /** Hidden activation (default 'tanh'; any non-polynomial one approximates). */
  readonly activation?: ActivationName
  /** Epochs (default 300). */
  readonly epochs?: number
  /** Adam's learning rate (default 0.1). */
  readonly learningRate?: number
  /** Minibatch size (default all the rows). */
  readonly batchSize?: number
  /** Rows held out to score each fit (default 0.25). */
  readonly validationFraction?: number
}

/** One width's fit. */
export interface ApproximationResult {
  readonly width: number
  readonly trainR2: number
  readonly validationR2: number
}

/**
 * Universal approximation in one dimension: one hidden layer of increasing width fitted to the same curve, showing the
 * fit improve with the number of units (Cybenko, Hornik), and how the widest network builds its curve as a sum of
 * scaled, shifted activations. Exposition, in order: the training and validation R² against the width, one fitted
 * curve per width (each with its training steps), and the widest network's units' contributions (output weight times
 * activation) across the input, unit by unit.
 *
 * @param data - One-dimensional regression data.
 * @param options - Widths, activation, epochs, learning rate, batch size and hold-out.
 * @param fit - The key, tracer and policy.
 * @returns The scores per width and the exposition records.
 */
export function universalApproximation(
  data: LearnData,
  options: UniversalApproximationOptions = {},
  fit: FitOptions = {},
): Explained<{ readonly results: readonly ApproximationResult[] }> {
  const widths = options.widths ?? [1, 2, 4, 8, 16, 32]
  if (widths.length === 0) throw new MdmlError('value', 'universalApproximation: needs at least one width')
  const key = fit.key ?? makeKey(0)
  const task = taskOf(data, false, {
    who: 'universalApproximation',
    key,
    holdOut: options.validationFraction ?? 0.25,
  })
  const d = task.x.shape[1] ?? 0
  if (d !== 1) throw new MdmlError('shape', `universalApproximation: needs one feature, got ${d}`)
  const surfaces: Exposition[] = []
  const results: ApproximationResult[] = []
  let widest: { net: ReturnType<typeof buildMlp>; params: Params; width: number } | undefined
  for (const width of widths) {
    const net = buildMlp({
      sizes: [1, width, task.nOut],
      activation: options.activation ?? 'tanh',
      key: foldIn(key, 1),
    })
    const training = train({
      name: `universalApproximation/${width}`,
      model: net,
      train: task.rows,
      ...(task.validation === undefined ? {} : { validation: task.validation }),
      loss: lossOf('identity'),
      score: scoreOf('identity'),
      optimizer: optimiserOf('adam', options.learningRate ?? 0.1),
      epochs: options.epochs ?? 300,
      ...(options.batchSize === undefined ? {} : { batchSize: options.batchSize }),
      snapshots: 15,
      fit: { ...fit, key: foldIn(key, 2) },
    })
    surfaces.push(surfaceOf(net, training.params, training.buffers, task, 60, training))
    results.push({
      width,
      trainR2: training.trainScore?.at(-1) ?? Number.NaN,
      validationR2: training.validationScore?.at(-1) ?? Number.NaN,
    })
    if (widest === undefined || width >= widest.width) widest = { net, params: training.params, width }
  }
  const curve = learningCurve({
    sizes: [...widths],
    train: results.map((r) => r.trainR2),
    validation: results.map((r) => r.validationR2),
    resource: 'hidden units',
    labels: { train: 'training R²', validation: 'validation R²' },
  })
  return { results, exposition: [curve, ...surfaces, ...(widest === undefined ? [] : [contributions(widest, task.x)])] }
}

// Each hidden unit's contribution to the output, w_k a_k(x), over a grid spanning the inputs: units down, x across.
function contributions(widest: { net: ReturnType<typeof buildMlp>; params: Params }, x: Tensor): Exposition {
  const v = toFlat(x)
  const lo = Math.min(...v)
  const hi = Math.max(...v)
  const res = 60
  const grid = fromStorage(
    Float64Array.from({ length: res }, (_, i) => lo + ((hi - lo) * i) / (res - 1)),
    [res, 1],
    'float64',
  )
  const a = hiddenActivations(widest.net, widest.params, grid)[0] as Tensor
  const units = a.shape[1] ?? 0
  const out = widest.params[linearNames(widest.net).at(-1) ?? ''] as { readonly weight: Tensor }
  const w = toFlat(out.weight)
  const av = toFlat(a)
  const map = Array.from({ length: units }, (_, k) =>
    Array.from({ length: res }, (_, i) => (w[k] ?? 0) * (av[i * units + k] ?? 0)),
  )
  return {
    ...mapSteps([[map]], [0], 1, undefined, false),
    features: ['input', 'hidden unit: output weight × activation'],
  }
}
