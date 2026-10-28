/**
 * The parts of the trainer base that do not depend on a particular: the plan arithmetic, the data to tabular form, the
 * default metrics of a kind, the layers of a network and what is logged about them, evaluation on held-out rows, the
 * training curves of a tape, and the verbs of a classifier or regressor over a network's parameters.
 */

import { type TabularData, xy, xyClasses } from '../../data/index.ts'
import { type CurveSeries, curveSet, type Exposition } from '../../exposition/index.ts'
import { accuracyScore, r2Score } from '../../metrics/index.ts'
import { type Module, type Params, Sequential, type TrainState } from '../../nn/index.ts'
import {
  argmax,
  asTensor,
  fromStorage,
  item,
  reshape,
  softmax,
  type Tensor,
  type TensorLike,
  toFlat,
  treePaths,
  type Value,
} from '../../tensor/index.ts'
import { metricSeries, type Run, type StepContext } from '../../trace/index.ts'
import { type EstimatorKind, type FittedFacts, type LearnData, requireTargets, type Verbs } from '../estimator.ts'

/** The rows held out of `n` by a fraction (at least one when a fraction is asked for). */
export const heldOut = (n: number, fraction: number | undefined): number =>
  fraction === undefined || fraction <= 0 ? 0 : Math.max(1, Math.round(n * fraction))

/** Rows for `fitAlgorithm`: class labels for a classifier, else targets (the features themselves without any). */
export const tabular = (kind: EstimatorKind, d: LearnData): TabularData =>
  kind === 'classifier' && d.y !== undefined ? xyClasses(d.x, d.y) : xy(d.x, d.y ?? d.x)

export type Metrics = Readonly<Record<string, (prediction: Value, target: Value) => number>>

export const defaultMetrics = (kind: EstimatorKind): Metrics =>
  kind === 'classifier'
    ? {
        accuracy: (p, t) => {
          const a = toFlat(argmax(p, 1))
          const y = toFlat(t)
          return a.reduce((n, v, i) => n + (v === y[i] ? 1 : 0), 0) / Math.max(1, a.length)
        },
      }
    : {
        rmse: (p, t) => {
          const a = toFlat(p)
          const y = toFlat(t)
          return Math.sqrt(a.reduce((n, v, i) => n + (v - (y[i] ?? 0)) ** 2, 0) / Math.max(1, a.length))
        },
      }

/** The layers of a module: the children of a `Sequential`, else the module as one layer. */
export interface Layer {
  readonly index: number
  readonly name: string | undefined
  readonly module: Module
}

export const layersOf = (model: Module): readonly Layer[] =>
  model instanceof Sequential
    ? model.layers.map((module, index) => ({ index, name: String(index), module }))
    : [{ index: 0, name: undefined, module: model }]

/** The part of a parameter or buffer tree that belongs to a layer. */
export const under = (tree: Params, l: Layer): Params =>
  l.name === undefined ? tree : ((tree[l.name] as Params) ?? {})

/** The layers' parameters, flattened parameters and outputs on the probe rows, for the paths that are due. */
export function recordLayers(layers: readonly Layer[], probe: Tensor, state: TrainState, ctx: StepContext): void {
  for (const l of layers) {
    const own = Object.entries(treePaths(under(state.params, l)))
    if (own.length === 0) continue
    if (ctx.wants(`layer/${l.index}/weights`))
      ctx.log(`layer/${l.index}/weights`, (own.find(([n]) => n === 'weight') ?? own[0])?.[1] as Tensor)
    if (ctx.wants(`layer/${l.index}/params`)) {
      const flat = own.flatMap(([, v]) => Array.from(toFlat(v)))
      ctx.log(`layer/${l.index}/params`, fromStorage(Float64Array.from(flat), [flat.length], 'float64'))
    }
  }
  if (!layers.some((l) => ctx.wants(`layer/${l.index}/activations`))) return
  let h: Value = probe
  for (const l of layers) {
    h = l.module.apply(under(state.params, l), h, { train: false, buffers: under(state.buffers, l) })
    ctx.log(`layer/${l.index}/activations`, h as Tensor)
  }
}

/** The objective and the metrics on held-out rows, with the network in evaluation mode. */
export function evaluate(
  model: Module,
  rows: TabularData,
  state: TrainState,
  loss: (prediction: Value, target: Value) => Value,
  metrics: Metrics,
): Record<string, number> {
  const prediction = model.apply(state.params, rows.x, { train: false, buffers: state.buffers })
  const out: Record<string, number> = { loss: item(loss(prediction, rows.y)) }
  for (const [name, f] of Object.entries(metrics)) out[name] = f(prediction, rows.y)
  return out
}

/** The training curves of a tape: the losses on the left axis, the validation metrics on the right. */
export function curvesOf(run: Run, unit: string): Exposition | undefined {
  const series: CurveSeries[] = []
  const add = (path: string, name: string, extra: Partial<CurveSeries>): void => {
    const s = metricSeries(run, path)
    if (s.steps.length > 0) series.push({ name, x: s.steps, y: s.values, ...extra })
  }
  add('train/loss', 'train loss', { group: 'loss' })
  add('eval/loss', 'validation loss', { group: 'loss' })
  const metrics = new Set(
    run.events.flatMap((e) => (e.type === 'metric' && e.name.startsWith('eval/') ? [e.name] : [])),
  )
  for (const path of metrics) if (path !== 'eval/loss') add(path, path.slice(5), { axis: 'right' })
  return series.length === 0 ? undefined : curveSet({ series, xLabel: unit, yLabel: 'loss', y2Label: 'metric' })
}

/** The verbs of a network's kind over parameters and buffers: probabilities, labels and scores, or predictions. */
export function networkVerbs(
  kind: EstimatorKind,
  model: Module,
  state: { readonly params: Params; readonly buffers: Params },
  classes: readonly number[] | undefined,
): Verbs & FittedFacts {
  const forward = (x: TensorLike): Tensor =>
    model.apply(state.params, asTensor(x), { train: false, buffers: state.buffers }) as Tensor
  const scored = (v: Verbs & FittedFacts): Verbs & FittedFacts => {
    const predict = v.predict
    if (predict === undefined) return v
    const metric = kind === 'classifier' ? accuracyScore : r2Score
    return { ...v, score: (d) => metric(requireTargets(d, 'score'), predict(d.x)).estimate.value }
  }
  if (kind !== 'classifier') {
    return scored({
      predict: (x) => {
        const z = forward(x)
        return z.shape.length === 2 && z.shape[1] === 1 ? (reshape(z, [z.shape[0] ?? 0]) as Tensor) : z
      },
    })
  }
  const labels = classes ?? []
  return scored({
    classes: labels,
    predictProba: (x) => softmax(forward(x), 1) as Tensor,
    predict: (x) => {
      const idx = toFlat(argmax(forward(x), 1))
      return fromStorage(
        Float64Array.from(idx, (i) => labels[i] ?? i),
        [idx.length],
        'float64',
      )
    },
    decisionFunction: (x) => {
      const z = forward(x)
      if (labels.length !== 2) return z
      const f = toFlat(z)
      const n = f.length / 2
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => (f[2 * i + 1] ?? 0) - (f[2 * i] ?? 0)),
        [n],
        'float64',
      )
    },
  })
}
