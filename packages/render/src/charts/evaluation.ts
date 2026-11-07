/**
 * Model evaluation figures from the engine's `metrics`: ROC, precision-recall and precision-recall-gain curves, reliability (calibration)
 * diagrams, learning and validation curves with spread bands, residual plots, and forest plots and bars of metric
 * results with their intervals.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { Estimate, MetricResult } from 'mdml-engine/base'
import type { CalibrationCurve, PrecisionRecallCurve, PrGainCurve, RocCurve } from 'mdml-engine/metrics'
import type { Tensor, TensorLike } from 'mdml-engine/tensor'
import { extentOf } from '../figure/limits.ts'
import { formatNumber } from '../theme/format.ts'
import { colourFor, modeOf, numbers } from './_data.ts'
import { bars } from './bars.ts'
import { band, lines } from './lines.ts'
import { cartesian, type Layer, layer, type PlotOptions } from './plot.ts'
import { points } from './points.ts'

/** A curve of the engine with its tensor vectors given as tensor-likes, so plain arrays draw as well. */
type Loose<T> = { readonly [K in keyof T]: T[K] extends Tensor<'float64'> ? TensorLike : T[K] }

function named<T>(items: T | readonly { readonly name?: string; readonly curve: T }[]): { name?: string; curve: T }[] {
  return Array.isArray(items)
    ? (items as readonly { readonly name?: string; readonly curve: T }[]).map((i) => ({ ...i }))
    : [{ curve: items as T }]
}

function diagonal(mode: PlotOptions['mode'], from: [number, number], to: [number, number], name: string): Layer {
  return lines([{ x: [from[0], to[0]], y: [from[1], to[1]], colour: 'muted', dash: 'dashed', width: 1, name }], {
    mode: mode ?? 'light',
  })
}

/**
 * ROC curves (from mdml-engine's `metrics.rocCurve`) with the chance diagonal, on the unit square at equal aspect;
 * names may carry the AUC.
 *
 * @param curves - One curve, or named curves.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @index function A ROC curve chart: `rocChart(curve)`.
 * @example
 * ```ts
 * import { rocChart } from 'mdml-render/charts'
 * const o = rocChart([{ name: 'model (AUC 0.75)', curve: { fpr: [0, 0, 0.5, 1], tpr: [0, 0.5, 1, 1], thresholds: [Infinity, 0.8, 0.4, 0.1] } }])
 * o.xAxis[0].name // => 'false positive rate'
 * ```
 */
export function rocChart(
  curves: Loose<RocCurve> | readonly { readonly name?: string; readonly curve: Loose<RocCurve> }[],
  options: PlotOptions = {},
): EChartsOption {
  const mode = modeOf(options)
  const list = named(curves)
  const l = lines(
    list.map((c) => ({ x: c.curve.fpr, y: c.curve.tpr, ...(c.name === undefined ? {} : { name: c.name }) })),
    { mode },
  )
  return cartesian([diagonal(mode, [0, 0], [1, 1], '__chance'), l], {
    x: { label: 'false positive rate', limits: [0, 1] },
    y: { label: 'true positive rate', limits: [0, 1] },
    tooltip: 'item',
    ...options,
  })
}

/**
 * Precision-recall curves (from mdml-engine's `metrics.precisionRecallCurve`), recall across and precision up,
 * with the no-skill line at the prevalence when given.
 *
 * @param curves - One curve, or named curves.
 * @param options - Plot options and the positive-class prevalence.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { precisionRecallChart } from 'mdml-render/charts'
 * precisionRecallChart({ precision: [0.5, 1, 1], recall: [1, 0.5, 0], thresholds: [0.1, 0.8] }, { prevalence: 0.5 }).series.length // => 2
 * ```
 */
export function precisionRecallChart(
  curves: Loose<PrecisionRecallCurve> | readonly { readonly name?: string; readonly curve: Loose<PrecisionRecallCurve> }[],
  options: PlotOptions & { readonly prevalence?: number } = {},
): EChartsOption {
  const mode = modeOf(options)
  const { prevalence, ...plot } = options
  const l = lines(
    named(curves).map((c) => ({
      x: c.curve.recall,
      y: c.curve.precision,
      step: 'start' as const,
      ...(c.name === undefined ? {} : { name: c.name }),
    })),
    { mode },
  )
  const layers = prevalence === undefined ? [l] : [diagonal(mode, [0, prevalence], [1, prevalence], '__no-skill'), l]
  return cartesian(layers, {
    x: { label: 'recall', limits: [0, 1] },
    y: { label: 'precision', limits: [0, 1.02] },
    ...plot,
  })
}

/**
 * Precision-recall-gain curves (Flach and Kull, 2015; from mdml-engine's `metrics.prGainCurve`): recall gain across
 * and precision gain up, on the unit square where the always-positive classifier sits at the origin, so the curve's
 * area is meaningful (unlike the precision-recall curve's) and the F-gain isometrics are straight lines; the dashed
 * minor diagonal is F-gain 1/2.
 *
 * @param curves - One curve, or named curves.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { prGainChart } from 'mdml-render/charts'
 * prGainChart({ precisionGain: [1, 0.5, 0], recallGain: [0, 0.5, 1], thresholds: [0.9, 0.5, 0.1], isCrossing: [true, false, true] }).xAxis[0].name // => 'recall gain'
 * ```
 */
export function prGainChart(
  curves: Loose<PrGainCurve> | readonly { readonly name?: string; readonly curve: Loose<PrGainCurve> }[],
  options: PlotOptions = {},
): EChartsOption {
  const mode = modeOf(options)
  // The curve on the unit square: points with negative recall gain lie off to the left and are left out.
  const l = lines(
    named(curves).map((c) => {
      const recallGain = numbers(c.curve.recallGain)
      const keep = recallGain.map((r) => r >= 0)
      return {
        x: recallGain.filter((_, i) => keep[i]),
        y: numbers(c.curve.precisionGain).filter((_, i) => keep[i]),
        ...(c.name === undefined ? {} : { name: c.name }),
      }
    }),
    { mode },
  )
  return cartesian([diagonal(mode, [0, 1], [1, 0], '__F-gain 1/2'), l], {
    x: { label: 'recall gain', limits: [0, 1] },
    y: { label: 'precision gain', limits: [0, 1] },
    ...options,
  })
}

/**
 * A reliability diagram (from mdml-engine's `metrics.calibrationCurve`): observed frequency against mean forecast
 * per bin, sized by count, with the perfect-calibration diagonal.
 *
 * @param curves - One curve, or named curves.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { calibrationChart } from 'mdml-render/charts'
 * calibrationChart({ probTrue: [0.1, 0.8], probPred: [0.2, 0.7], counts: [10, 30] }).series.length // => 3
 * ```
 */
export function calibrationChart(
  curves: Loose<CalibrationCurve> | readonly { readonly name?: string; readonly curve: Loose<CalibrationCurve> }[],
  options: PlotOptions = {},
): EChartsOption {
  const mode = modeOf(options)
  const list = named(curves)
  const l = lines(
    list.map((c) => ({ x: c.curve.probPred, y: c.curve.probTrue, ...(c.name === undefined ? {} : { name: c.name }) })),
    { mode },
  )
  const p = points(
    list.map((c) => ({ x: c.curve.probPred, y: c.curve.probTrue, sizes: numbers(c.curve.counts), sizeRange: [4, 16] as const })),
    { mode },
  )
  return cartesian([diagonal(mode, [0, 0], [1, 1], '__perfect'), l, p], {
    x: { label: 'mean predicted probability', limits: [0, 1] },
    y: { label: 'fraction of positives', limits: [0, 1] },
    aspect: 'equal',
    ...options,
  })
}

/** Scores at each training size: the mean over folds and optionally its spread. */
export interface ScoreSeries {
  readonly mean: readonly number[]
  readonly std?: readonly number[]
}

/**
 * A learning (or validation) curve: train and test scores against training size (or a parameter), each with a band of
 * one standard deviation.
 *
 * @param sizes - The training sizes or parameter values.
 * @param train - Training scores.
 * @param test - Validation scores.
 * @param options - Plot options, and the x and y labels.
 * @returns The ECharts option.
 * @index function Train and validation scores against training size: `learningCurveChart(sizes, train, test)`.
 * @example
 * ```ts
 * import { learningCurveChart } from 'mdml-render/charts'
 * learningCurveChart([10, 20], { mean: [0.9, 0.85], std: [0.05, 0.03] }, { mean: [0.6, 0.7] }).legend.data // => ['train', 'validation']
 * ```
 */
export function learningCurveChart(
  sizes: readonly number[],
  train: ScoreSeries,
  test: ScoreSeries,
  options: PlotOptions = {},
): EChartsOption {
  const mode = modeOf(options)
  const layers: Layer[] = []
  ;[train, test].forEach((s, i) => {
    if (s.std !== undefined) {
      const sd = s.std
      layers.push(
        band(
          [
            {
              x: sizes,
              lower: s.mean.map((m, k) => m - (sd[k] ?? 0)),
              upper: s.mean.map((m, k) => m + (sd[k] ?? 0)),
              colour: `tab${i}`,
            },
          ],
          { mode },
        ),
      )
    }
  })
  layers.push(
    lines(
      [
        { x: sizes, y: train.mean, name: 'train', markers: true, colour: 'tab0' },
        { x: sizes, y: test.mean, name: 'validation', markers: true, colour: 'tab1' },
      ],
      { mode },
    ),
  )
  return cartesian(layers, { x: { label: 'training size' }, y: { label: 'score' }, tooltip: 'axis', ...options })
}

/**
 * A residual plot: residuals against fitted values with the zero line.
 *
 * @param fitted - Fitted values.
 * @param residuals - Residuals (observed minus fitted).
 * @param options - Plot options.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { residualChart } from 'mdml-render/charts'
 * residualChart([1, 2, 3], [0.1, -0.2, 0.05]).yAxis[0].name // => 'residual'
 * ```
 */
export function residualChart(
  fitted: readonly number[],
  residuals: readonly number[],
  options: PlotOptions = {},
): EChartsOption {
  const mode = modeOf(options)
  const p = points([{ x: fitted, y: residuals, name: '__residuals', opacity: 0.7 }], { mode })
  const e = extentOf(fitted) ?? [0, 1]
  const zero = lines([{ x: [e[0], e[1]], y: [0, 0], colour: 'muted', dash: 'dashed', width: 1, name: '__zero' }], {
    mode,
  })
  return cartesian([zero, p], { x: { label: 'fitted' }, y: { label: 'residual' }, ...options })
}

interface IntervalApi {
  coord(p: readonly (number | string)[]): number[]
  value(d: number): number
}

/**
 * Estimates with intervals (a forest plot): each metric result's or estimate's value as a point and its interval as a
 * horizontal bar, one row per item, top to bottom.
 *
 * @param items - Named estimates or metric results (`estimate.interval` draws the bar when present).
 * @param options - Plot options and a reference value (a vertical line, e.g. zero effect).
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { intervalChart } from 'mdml-render/charts'
 * const o = intervalChart([{ name: 'a', value: { value: 0.8, interval: [0.7, 0.9] } }])
 * o.yAxis[0].data // => ['a']
 * ```
 */
export function intervalChart(
  items: readonly { readonly name: string; readonly value: Estimate | MetricResult }[],
  options: PlotOptions & { readonly reference?: number } = {},
): EChartsOption {
  const mode = modeOf(options)
  const { reference, ...plot } = options
  const est = items.map((i) => ('estimate' in i.value ? i.value.estimate : i.value))
  const names = items.map((i) => i.name).reverse()
  const rows = est.map((e, k) => [
    items.length - 1 - k,
    e.value,
    e.interval?.[0] ?? e.value,
    e.interval?.[1] ?? e.value,
  ])
  const colour = colourFor('tab0', mode)
  const fg = colourFor('fg', mode)
  const series: SeriesOption[] = [
    {
      type: 'custom',
      name: '__intervals',
      data: rows,
      renderItem: (_p: unknown, api: IntervalApi): Record<string, unknown> => {
        const row = names[api.value(0)] ?? ''
        const [x0 = 0, y = 0] = api.coord([api.value(2), row])
        const [x1 = 0] = api.coord([api.value(3), row])
        const [xm = 0] = api.coord([api.value(1), row])
        return {
          type: 'group',
          children: [
            { type: 'line', shape: { x1: x0, y1: y, x2: x1, y2: y }, style: { stroke: fg, lineWidth: 1.5 } },
            { type: 'circle', shape: { cx: xm, cy: y, r: 4.5 }, style: { fill: colour } },
          ],
        }
      },
      encode: { x: [1, 2, 3], y: 0 },
      tooltip: {
        formatter: (p: { value: number[] }) =>
          `${names[p.value[0] ?? 0] ?? ''}: ${formatNumber(p.value[1] ?? 0)} [${formatNumber(p.value[2] ?? 0)}, ${formatNumber(p.value[3] ?? 0)}]`,
      },
    } as unknown as SeriesOption,
  ]
  const xs = rows.flatMap((r) => r.slice(1))
  const layers: Layer[] = [
    layer(
      series,
      { x: extentOf([...xs, ...(reference === undefined ? [] : [reference])]) ?? [0, 1] },
      { yCategories: names },
    ),
  ]
  if (reference !== undefined)
    layers.unshift(
      layer(
        [
          {
            type: 'line',
            name: '__reference',
            data: [],
            markLine: {
              silent: true,
              symbol: 'none',
              lineStyle: { color: colourFor('muted', mode), type: 'dashed' },
              data: [{ xAxis: reference }],
              label: { show: false },
            },
          } as SeriesOption,
        ],
        {},
        { yCategories: names },
      ),
    )
  return cartesian(layers, plot)
}

/**
 * Metric results as bars with their interval as error bars, one bar per metric (or per model).
 *
 * @param results - Named metric results.
 * @param options - Plot options.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { metricBarChart } from 'mdml-render/charts'
 * metricBarChart([{ name: 'accuracy', value: { metric: 'accuracy', estimate: { value: 0.9, interval: [0.85, 0.95] }, n: 100 } }]).xAxis[0].data // => ['accuracy']
 * ```
 */
export function metricBarChart(
  results: readonly { readonly name: string; readonly value: MetricResult }[],
  options: PlotOptions = {},
): EChartsOption {
  const values = results.map((r) => r.value.estimate.value)
  const errors = results.map((r) => {
    const i = r.value.estimate.interval
    return i === undefined ? 0 : (i[1] - i[0]) / 2
  })
  return cartesian(
    [
      bars(
        results.map((r) => r.name),
        [{ values, errors }],
        { mode: modeOf(options) },
      ),
    ],
    { tooltip: 'axis', ...options },
  )
}
