/**
 * Views of feature attributions (SHAP values and other local explanations) as shap draws them: a waterfall from the
 * expected output to one prediction (played as a sampled estimate converges), a beeswarm summary coloured by feature
 * value, mean |attribution| bars and dependence scatters coloured by the interacting feature, and permutation
 * importances as bars with their spread.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import type { FeatureAttributions, Importances } from 'whitebox-engine/exposition'
import { bars } from '../charts/bars.ts'
import { referenceLines } from '../charts/functions.ts'
import { cartesian, type Layer, layer } from '../charts/plot.ts'
import type { Figure } from '../figure/figure.ts'
import { formatNumber } from '../theme/format.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { short, spanOf, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** The colour of a positive attribution (pushing the output up), as shap draws it. */
export const ATTRIBUTION_UP = '#ff0d57'
/** The colour of a negative attribution (pushing the output down). */
export const ATTRIBUTION_DOWN = '#1e88e5'
/** Low to high feature values, blue through purple to red (shap's red-blue). */
export const FEATURE_VALUE_RAMP: readonly string[] = ['#1e88e5', '#7b4fd8', '#ff0d57']

/** Options of the attribution views: how many features to show (the rest summed into one row). */
export interface AttributionViewOptions extends ViewOptions {
  /** Features shown (default 10); the rest are summed into one row (the waterfall) or left out. */
  readonly maxFeatures?: number
}

/** Options of {@link waterfallView} and {@link attributionFrames}. */
export interface WaterfallViewOptions extends AttributionViewOptions {
  /** Which explained row (default 0). */
  readonly row?: number
}

/**
 * The features of an attribution record in order of their mean |attribution| over the rows, largest first.
 *
 * @param record - The record.
 * @returns Feature indices.
 * @example
 * ```ts
 * import { featureAttributions } from 'whitebox-engine/exposition'
 * import { attributionOrder } from 'whitebox-render/views'
 * const r = featureAttributions({ method: 'exact', features: ['a', 'b'], output: 'y', baseValue: 0, values: [[0.1, -2]], data: [[0, 0]], predictions: [-1.9] })
 * attributionOrder(r) // => [1, 0]
 * ```
 */
export function attributionOrder(record: FeatureAttributions): number[] {
  const n = Math.max(1, record.values.length)
  const mean = record.features.map((_, j) => record.values.reduce((s, row) => s + Math.abs(row[j] ?? 0), 0) / n)
  return mean
    .map((v, j) => ({ v, j }))
    .sort((a, b) => b.v - a.v || a.j - b.j)
    .map((e) => e.j)
}

function signed(v: number): string {
  return `${v >= 0 ? '+' : '−'}${short(Math.abs(v))}`
}

// Room on the left for category labels of this many characters.
function labelPadding(fig: Figure, labels: readonly string[]): Figure {
  const longest = Math.max(4, ...labels.map((l) => l.length))
  const left = Math.min(220, Math.round(16 + longest * fig.fontSize * 0.6))
  return { ...fig, padding: { ...fig.padding, left: Math.max(fig.padding.left, left) } }
}

interface WaterfallBar {
  readonly label: string
  readonly start: number
  readonly end: number
}

// The waterfall's rows from the bottom up: the summed rest (if any) first, then the largest at the top, each starting
// where the one beneath it ended, from the base value.
function waterfallBars(
  record: FeatureAttributions,
  values: readonly number[],
  row: number,
  order: readonly number[],
  maxFeatures: number,
): { readonly bars: WaterfallBar[]; readonly base: number; readonly end: number } {
  const d = record.features.length
  const k = d <= maxFeatures ? d : Math.max(1, maxFeatures - 1)
  const shown = order.slice(0, k)
  const rest = order.slice(k)
  const base = record.baseValues?.[row] ?? record.baseValue
  const data = record.data[row] ?? []
  const items: { label: string; value: number }[] = []
  if (rest.length > 0)
    items.push({
      label: `${rest.length} other features`,
      value: rest.reduce((s, j) => s + (values[j] ?? 0), 0),
    })
  for (const j of [...shown].reverse())
    items.push({ label: `${record.features[j] ?? `x${j}`} = ${short(data[j] ?? 0)}`, value: values[j] ?? 0 })
  let at = base
  const out = items.map((it) => {
    const b = { label: it.label, start: at, end: at + it.value }
    at += it.value
    return b
  })
  return { bars: out, base, end: at }
}

interface SwarmApi {
  value(dim: number): number
  coord(p: readonly (number | string)[]): number[]
  size(p: readonly number[]): number[]
  visual(name: string): string
}

interface ArrowApi {
  value(dim: number): number
  coord(p: readonly (number | string)[]): number[]
  size(p: readonly number[]): number[]
}

function waterfallLayer(bars: readonly WaterfallBar[], mode: ColourMode, fontSize: number): Layer {
  const t = tokens(mode)
  const series = {
    type: 'custom',
    name: '__waterfall',
    data: bars.map((b, i) => [b.start, b.end, i]),
    renderItem: (_p: unknown, api: ArrowApi): Record<string, unknown> => {
      const s = api.value(0)
      const e = api.value(1)
      const i = api.value(2)
      const cat = bars[i]?.label ?? ''
      const a = api.coord([s, cat])
      const b = api.coord([e, cat])
      const band = api.size([0, 1])[1] ?? 20
      const h = Math.max(6, Math.min(22, band * 0.62))
      const x0 = a[0] ?? 0
      const x1 = b[0] ?? 0
      const y = a[1] ?? 0
      const up = e >= s
      const dir = up ? 1 : -1
      const tip = Math.min(7, Math.abs(x1 - x0))
      const colour = up ? ATTRIBUTION_UP : ATTRIBUTION_DOWN
      const v = e - s
      return {
        type: 'group',
        children: [
          {
            type: 'polygon',
            shape: {
              points: [
                [x0, y - h / 2],
                [x1 - dir * tip, y - h / 2],
                [x1, y],
                [x1 - dir * tip, y + h / 2],
                [x0, y + h / 2],
              ],
            },
            style: { fill: colour, stroke: t.bg, lineWidth: 0.5 },
          },
          {
            type: 'text',
            style: {
              text: signed(v),
              x: x1 + dir * 5,
              y,
              fill: colour,
              align: up ? 'left' : 'right',
              verticalAlign: 'middle',
              font: `600 ${fontSize}px ${t.font}`,
            },
          },
        ],
      }
    },
    tooltip: {
      formatter: (p: { dataIndex: number }) => {
        const b = bars[p.dataIndex]
        return b === undefined ? '' : `${b.label}<br/>${signed(b.end - b.start)}`
      },
    },
    z: 3,
  } as unknown as SeriesOption
  return layer([series], {}, { yCategories: bars.map((b) => b.label) })
}

function waterfallOption(
  record: FeatureAttributions,
  values: readonly number[],
  row: number,
  fig: Figure,
  span: readonly [number, number],
  maxFeatures: number,
): EChartsOption {
  const order = attributionOrder(record)
  const w = waterfallBars(record, values, row, order, maxFeatures)
  const prediction = record.predictions[row] ?? w.end
  const f = labelPadding(
    fig,
    w.bars.map((b) => b.label),
  )
  // Leave room either side for the value labels beyond the arrows' tips.
  const pad = (span[1] - span[0] || 1) * 0.16
  const local = Math.abs(prediction - w.end) > 1e-9 * (1 + Math.abs(prediction))
  const t = tokens(f.mode)
  const rises = prediction >= w.base
  const farApart = Math.abs(prediction - w.end) > 0.12 * (span[1] - span[0] || 1)
  const marks = {
    type: 'line',
    name: '__marks',
    data: [],
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: { color: t.muted, type: 'dashed', width: 1 },
      label: { color: t.fgSecondary, fontSize: f.fontSize - 1 },
      data: [
        // Both above the plot, each leaning away from the other.
        {
          xAxis: w.base,
          label: { formatter: `E[f(x)] = ${short(w.base)}`, position: 'end', align: rises ? 'right' : 'left' },
        },
        {
          xAxis: prediction,
          label: { formatter: `f(x) = ${short(prediction)}`, position: 'end', align: rises ? 'left' : 'right' },
        },
        // A local surrogate's own prediction (or a sampled estimate's sum), labelled when clear of f(x).
        ...(local
          ? [
              {
                xAxis: w.end,
                label: farApart
                  ? {
                      formatter: `${record.method === 'lime' ? 'local model' : 'sum'} = ${short(w.end)}`,
                      position: 'end',
                      align: w.end >= prediction ? 'left' : 'right',
                    }
                  : { show: false },
              },
            ]
          : []),
      ],
    },
  } as SeriesOption
  return cartesian([waterfallLayer(w.bars, f.mode, f.fontSize - 1), layer([marks], {})], {
    figure: f,
    mode: f.mode,
    x: { label: record.output, limits: [span[0] - pad, span[1] + pad] },
    tooltip: 'item',
    legend: false,
  })
}

function waterfallSpan(
  record: FeatureAttributions,
  steps: readonly (readonly number[])[],
  row: number,
  maxFeatures: number,
): [number, number] {
  const order = attributionOrder(record)
  const ends: number[] = [record.predictions[row] ?? 0]
  for (const v of steps) {
    const w = waterfallBars(record, v, row, order, maxFeatures)
    ends.push(w.base, w.end, ...w.bars.flatMap((b) => [b.start, b.end]))
  }
  return spanOf(ends)
}

/**
 * One prediction's attributions as a waterfall (shap.plots.waterfall): from the expected output E[f(x)] at the bottom,
 * each feature's push up (red) or down (blue) in turn, the largest at the top, ending at the prediction f(x); the
 * features beyond `maxFeatures` summed into one row. A local surrogate's waterfall ends at its own prediction, marked
 * beside the model's.
 *
 * @param record - The record.
 * @param options - row, maxFeatures, mode and figure.
 * @returns The ECharts option.
 */
export function waterfallView(record: FeatureAttributions, options: WaterfallViewOptions = {}): EChartsOption {
  const row = options.row ?? 0
  const max = options.maxFeatures ?? 10
  const values = record.values[row] ?? []
  const fig = viewFigure(options, { aspect: 1.5 })
  return waterfallOption(record, values, row, fig, waterfallSpan(record, [values], row, max), max)
}

/**
 * A waterfall that plays: with a sampled estimate's convergence, the waterfall of the running estimate after each
 * batch of samples, on fixed axes, so the bars settle; otherwise the one waterfall.
 *
 * @param record - The record.
 * @param options - row, maxFeatures, mode and figure.
 * @returns The frames.
 * @example
 * ```ts
 * import { featureAttributions } from 'whitebox-engine/exposition'
 * import { attributionFrames } from 'whitebox-render/views'
 * const r = featureAttributions({
 *   method: 'sampling', features: ['a', 'b'], output: 'y', baseValue: 0, values: [[1, -1]], data: [[0, 0]], predictions: [0],
 *   convergence: { row: 0, samples: [10, 20], estimates: [[0.8, -0.9], [1, -1]] },
 * })
 * attributionFrames(r).count // => 2
 * ```
 */
export function attributionFrames(record: FeatureAttributions, options: WaterfallViewOptions = {}): ViewFrames {
  const c = record.convergence
  const row = c?.row ?? options.row ?? 0
  const max = options.maxFeatures ?? 10
  const steps = c === undefined ? [record.values[row] ?? []] : c.estimates
  const fig = viewFigure(options, { aspect: 1.5 })
  const span = waterfallSpan(record, steps, row, max)
  return {
    count: steps.length,
    label: (i) => (c === undefined ? '' : `${c.samples[i] ?? 0} samples`),
    option: (i) => waterfallOption(record, steps[i] ?? [], row, fig, span, max),
  }
}

// A value's place between a feature's 5th and 95th percentiles, clipped to [0, 1] (shap's colouring).
function scaledColumn(values: readonly number[]): number[] {
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b)
  const q = (p: number): number =>
    sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))] ?? 0
  let lo = q(0.05)
  let hi = q(0.95)
  if (hi <= lo) {
    lo = sorted[0] ?? 0
    hi = sorted[sorted.length - 1] ?? 0
  }
  return values.map((v) => (hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5))
}

// shap's beeswarm offsets: bin the values, and stack each bin's points alternately above and below the row.
function swarm(values: readonly number[], bins = 100, height = 0.4): number[] {
  const [lo, hi] = spanOf(values)
  const quant = values.map((v) => Math.round((bins * (v - lo)) / (hi - lo + 1e-8)))
  const order = values.map((_, i) => i).sort((a, b) => (quant[a] ?? 0) - (quant[b] ?? 0) || a - b)
  const ys = new Array<number>(values.length).fill(0)
  let layerIndex = 0
  let last = -1
  for (const i of order) {
    if (quant[i] !== last) layerIndex = 0
    ys[i] = Math.ceil(layerIndex / 2) * ((layerIndex % 2) * 2 - 1)
    layerIndex += 1
    last = quant[i] ?? 0
  }
  const top = Math.max(1, ...ys.map((y) => Math.abs(y) + 1))
  return ys.map((y) => (0.9 * height * y) / top)
}

function relabelBar(option: EChartsOption, text: readonly [string, string]): EChartsOption {
  const maps = (option.visualMap ?? []) as Record<string, unknown>[]
  const visualMap = maps.map((m) => {
    const { formatter: _, ...rest } = m
    return { ...rest, text: [...text], calculable: false }
  }) as EChartsOption['visualMap']
  return { ...option, ...(visualMap === undefined ? {} : { visualMap }) }
}

/**
 * Many rows' attributions as a beeswarm summary (shap.plots.beeswarm): one row per feature in order of mean
 * |attribution|, a point per explained row at its attribution, stacked where they crowd, coloured by the feature's
 * value from low (blue) to high (red), so the direction of each feature's effect reads at a glance.
 *
 * @param record - The record.
 * @param options - maxFeatures (default 12), mode and figure.
 * @returns The ECharts option.
 */
export function beeswarmView(record: FeatureAttributions, options: AttributionViewOptions = {}): EChartsOption {
  const shown = attributionOrder(record).slice(0, options.maxFeatures ?? 12)
  const cats = [...shown].reverse()
  const labels = cats.map((j) => record.features[j] ?? `x${j}`)
  const fig = labelPadding(
    viewFigure(options, { height: Math.max(220, 70 + 30 * cats.length), padding: { right: 72 } }),
    labels,
  )
  // [attribution, feature row, offset within the row (in rows), colour value].
  const data: [number, number, number, number][] = []
  cats.forEach((j, r) => {
    const v = record.values.map((row) => row[j] ?? 0)
    const c = scaledColumn(record.data.map((row) => row[j] ?? 0))
    const dy = swarm(v)
    v.forEach((x, i) => {
      data.push([x, r, dy[i] ?? 0, c[i] ?? 0.5])
    })
  })
  const t = tokens(fig.mode)
  const radius = record.values.length > 400 ? 2 : 2.75
  const dots = layer(
    [
      {
        type: 'custom',
        name: '__swarm',
        data,
        // A category axis places only whole rows: each point is offset within its row's band in pixels.
        renderItem: (_p: unknown, api: SwarmApi): Record<string, unknown> => {
          const r = api.value(1)
          const at = api.coord([api.value(0), labels[r] ?? ''])
          const band = api.size([0, 1])[1] ?? 20
          return {
            type: 'circle',
            shape: { cx: at[0] ?? 0, cy: (at[1] ?? 0) - api.value(2) * band, r: radius },
            style: { fill: api.visual('color'), opacity: 0.9 },
          }
        },
        encode: { x: 0, y: 1, tooltip: [0] },
        tooltip: {
          formatter: (p: { value: number[] }) => `${labels[p.value[1] ?? 0] ?? ''}: ${signed(p.value[0] ?? 0)}`,
        },
        z: 3,
      } as unknown as SeriesOption,
    ],
    { x: spanOf(data.map((d) => d[0])) },
    {
      yCategories: labels,
      colourBar: { min: 0, max: 1, colours: [...FEATURE_VALUE_RAMP], label: 'feature value' },
    },
  )
  return relabelBar(
    cartesian([referenceLines({ x: [0] }, { mode: fig.mode, colour: t.muted, dash: 'solid' }), dots], {
      figure: fig,
      mode: fig.mode,
      x: { label: `attribution to ${record.output}` },
      tooltip: 'item',
      legend: false,
    }),
    ['high', 'low'],
  )
}

/**
 * The mean |attribution| of each feature over the rows as bars, largest at the top (shap.plots.bar): a global
 * importance built from local explanations.
 *
 * @param record - The record.
 * @param options - maxFeatures (default 12), mode and figure.
 * @returns The ECharts option.
 */
export function attributionBarView(record: FeatureAttributions, options: AttributionViewOptions = {}): EChartsOption {
  const shown = attributionOrder(record).slice(0, options.maxFeatures ?? 12)
  const cats = [...shown].reverse()
  const labels = cats.map((j) => record.features[j] ?? `x${j}`)
  const n = Math.max(1, record.values.length)
  const means = cats.map((j) => record.values.reduce((s, row) => s + Math.abs(row[j] ?? 0), 0) / n)
  const fig = labelPadding(viewFigure(options, { height: Math.max(200, 70 + 26 * cats.length) }), labels)
  return cartesian(
    [bars(labels, [{ name: '__mean', values: means, colour: ATTRIBUTION_UP }], { mode: fig.mode, horizontal: true })],
    {
      figure: { ...fig, padding: { ...fig.padding, right: Math.max(fig.padding.right, 48) } },
      mode: fig.mode,
      x: { label: `mean |attribution| to ${record.output}` },
      legend: false,
    },
  )
}

/** Options of {@link dependenceScatterView}. */
export interface DependenceScatterOptions extends ViewOptions {
  /** The feature on the x axis (default the most important). */
  readonly feature?: number
  /** The feature colouring the points (default the record's partner of `feature`; -1 for none). */
  readonly colourBy?: number
}

/**
 * A feature's dependence scatter (shap.plots.scatter): each explained row at its value of the feature and its
 * attribution, coloured by the feature it most interacts with, so a spread in the vertical shows the interaction.
 *
 * @param record - The record.
 * @param options - feature, colourBy, mode and figure.
 * @returns The ECharts option.
 */
export function dependenceScatterView(
  record: FeatureAttributions,
  options: DependenceScatterOptions = {},
): EChartsOption {
  const f = options.feature ?? attributionOrder(record)[0] ?? 0
  const g = options.colourBy ?? record.partners?.[f] ?? -1
  const fig = viewFigure(options, { aspect: 1.5, padding: g >= 0 ? { right: 80 } : {} })
  const t = tokens(fig.mode)
  const xs = record.data.map((row) => row[f] ?? 0)
  const ys = record.values.map((row) => row[f] ?? 0)
  const cs = g >= 0 ? record.data.map((row) => row[g] ?? 0) : undefined
  const sorted = cs === undefined ? [] : [...cs].sort((a, b) => a - b)
  const q = (p: number): number => sorted[Math.round(p * (sorted.length - 1))] ?? 0
  const [lo, hi] = cs === undefined ? [0, 1] : q(0.95) > q(0.05) ? [q(0.05), q(0.95)] : spanOf(sorted)
  const name = record.features[f] ?? `x${f}`
  const dots = layer(
    [
      {
        type: 'scatter',
        name: '__dependence',
        data: xs.map((x, i) => (cs === undefined ? [x, ys[i] ?? 0] : [x, ys[i] ?? 0, cs[i] ?? 0])),
        symbolSize: 6,
        itemStyle: { opacity: 0.85, ...(cs === undefined ? { color: ATTRIBUTION_DOWN } : {}) },
        z: 3,
      } as SeriesOption,
    ],
    { x: spanOf(xs), y: spanOf(ys) },
    cs === undefined
      ? {}
      : {
          colourBar: {
            min: lo,
            max: hi > lo ? hi : lo + 1,
            colours: [...FEATURE_VALUE_RAMP],
            label: record.features[g] ?? `x${g}`,
          },
        },
  )
  return cartesian([referenceLines({ y: [0] }, { mode: fig.mode, colour: t.muted, dash: 'solid' }), dots], {
    figure: fig,
    mode: fig.mode,
    x: { label: name },
    y: { label: `attribution of ${name}` },
    tooltip: 'item',
    legend: false,
  })
}

/**
 * Feature importances as bars with their spread (sklearn's permutation importance box plot, as bars): the mean
 * decrease in the metric when each feature is shuffled, largest at the top, a whisker of one spread either side and
 * every repeat as a dot.
 *
 * @param record - The record.
 * @param options - maxFeatures (default 15), mode and figure.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { importances } from 'whitebox-engine/exposition'
 * import { importanceView } from 'whitebox-render/views'
 * const r = importances({ method: 'permutation', features: ['a', 'b'], mean: [0.2, 0.01], spread: [0.02, 0.01], metric: 'accuracy', direction: 'higher' })
 * Array.isArray(importanceView(r).series) // => true
 * ```
 */
export function importanceView(record: Importances, options: AttributionViewOptions = {}): EChartsOption {
  const order = record.mean
    .map((v, j) => ({ v, j }))
    .sort((a, b) => b.v - a.v || a.j - b.j)
    .slice(0, options.maxFeatures ?? 15)
    .map((e) => e.j)
  const cats = [...order].reverse()
  const labels = cats.map((j) => record.features[j] ?? `x${j}`)
  const fig = labelPadding(viewFigure(options, { height: Math.max(200, 70 + 26 * cats.length) }), labels)
  const t = tokens(fig.mode)
  const mean = cats.map((j) => record.mean[j] ?? 0)
  const spread = record.spread === undefined ? undefined : cats.map((j) => record.spread?.[j] ?? 0)
  const layers: Layer[] = [
    bars(
      labels,
      [{ name: '__mean', values: mean, colour: 'accent', ...(spread === undefined ? {} : { errors: spread }) }],
      {
        mode: fig.mode,
        horizontal: true,
      },
    ),
  ]
  if (record.repeats !== undefined) {
    const dots: [number, string][] = []
    cats.forEach((j, r) => {
      for (const v of record.repeats?.[j] ?? []) dots.push([v, labels[r] ?? ''])
    })
    layers.push(
      layer(
        [
          {
            type: 'scatter',
            name: '__repeats',
            data: dots,
            symbolSize: 4,
            itemStyle: { color: t.fg, opacity: 0.55 },
            z: 5,
          } as SeriesOption,
        ],
        { x: spanOf(dots.map((d) => d[0])) },
      ),
    )
  }
  layers.push(referenceLines({ x: [0] }, { mode: fig.mode, colour: t.muted, dash: 'solid' }))
  const what =
    record.method === 'permutation'
      ? `decrease in ${record.metric}${record.baseline === undefined ? '' : ` (from ${formatNumber(record.baseline, { digits: 3 })})`}`
      : record.metric
  return cartesian(layers, {
    figure: fig,
    mode: fig.mode,
    x: { label: what },
    tooltip: 'item',
    legend: false,
  })
}

interface ForceSegment {
  readonly label: string
  readonly start: number
  readonly end: number
  readonly up: boolean
}

/**
 * The segments of a force plot for one row: features that raise the output stacked up to the prediction from the
 * left (largest nearest it), those that lower it stacked down from the right.
 *
 * @param record - The record.
 * @param row - The explained row.
 * @returns The segments, the base value and the end the attributions reach.
 * @example
 * ```ts
 * import { featureAttributions } from 'whitebox-engine/exposition'
 * import { forceSegments } from 'whitebox-render/views'
 * const r = featureAttributions({ method: 'exact', features: ['a', 'b'], output: 'y', baseValue: 1, values: [[2, -0.5]], data: [[0, 0]], predictions: [2.5] })
 * forceSegments(r, 0).segments.map((s) => [s.start, s.end]) // => [[0.5, 2.5], [2.5, 3]]
 * ```
 */
export function forceSegments(
  record: FeatureAttributions,
  row: number,
): { readonly segments: ForceSegment[]; readonly base: number; readonly end: number } {
  const values = record.values[row] ?? []
  const data = record.data[row] ?? []
  const base = record.baseValues?.[row] ?? record.baseValue
  const end = base + values.reduce((a, b) => a + b, 0)
  const label = (j: number): string => `${record.features[j] ?? `x${j}`} = ${short(data[j] ?? 0)}`
  const order = (sign: 1 | -1): number[] =>
    values
      .map((v, j) => ({ v, j }))
      .filter((e) => (sign > 0 ? e.v > 0 : e.v < 0))
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v) || a.j - b.j)
      .map((e) => e.j)
  const segments: ForceSegment[] = []
  let at = end
  for (const j of order(1)) {
    const v = values[j] ?? 0
    segments.push({ label: label(j), start: at - v, end: at, up: true })
    at -= v
  }
  at = end
  for (const j of order(-1)) {
    const v = -(values[j] ?? 0)
    segments.push({ label: label(j), start: at, end: at + v, up: false })
    at += v
  }
  return { segments, base, end }
}

/**
 * One prediction's attributions as a force plot (shap.plots.force): along the output axis, the features that push
 * the prediction up as red arrows ending at f(x) from the left and those that push it down as blue arrows from the
 * right, each as long as its attribution, the larger features labelled beneath, with the base value marked.
 *
 * @param record - The record.
 * @param options - row, mode and figure.
 * @returns The ECharts option.
 */
export function forceView(record: FeatureAttributions, options: WaterfallViewOptions = {}): EChartsOption {
  const row = options.row ?? 0
  const fig = viewFigure(options, { aspect: 3.2, padding: { left: 24, right: 24 } })
  const t = tokens(fig.mode)
  const { segments, base, end } = forceSegments(record, row)
  const prediction = record.predictions[row] ?? end
  const span = spanOf([base, prediction, end, ...segments.flatMap((s) => [s.start, s.end])])
  const width = span[1] - span[0] || 1
  const fs = fig.fontSize - 1
  const series = {
    type: 'custom',
    name: '__force',
    data: segments.map((s, i) => [s.start, s.end, i]),
    renderItem: (_p: unknown, api: ArrowApi): Record<string, unknown> => {
      const i = api.value(2)
      const seg = segments[i]
      if (seg === undefined) return { type: 'group', children: [] }
      const a = api.coord([seg.start, 0])
      const b = api.coord([seg.end, 0])
      const x0 = a[0] ?? 0
      const x1 = b[0] ?? 0
      const y = a[1] ?? 0
      const h = 18
      const tip = Math.min(6, Math.abs(x1 - x0) / 2)
      // Red arrows point right (towards higher output), blue ones left.
      const points = seg.up
        ? [
            [x0, y - h / 2],
            [x1 - tip, y - h / 2],
            [x1, y],
            [x1 - tip, y + h / 2],
            [x0, y + h / 2],
            [x0 + tip, y],
          ]
        : [
            [x0 + tip, y - h / 2],
            [x1, y - h / 2],
            [x1 - tip, y],
            [x1, y + h / 2],
            [x0 + tip, y + h / 2],
            [x0, y],
          ]
      const wide = Math.abs(seg.end - seg.start) > 0.07 * width
      return {
        type: 'group',
        children: [
          {
            type: 'polygon',
            shape: { points },
            style: { fill: seg.up ? ATTRIBUTION_UP : ATTRIBUTION_DOWN, stroke: t.bg, lineWidth: 1 },
          },
          ...(wide
            ? [
                {
                  type: 'text',
                  style: {
                    text: seg.label,
                    x: (x0 + x1) / 2,
                    y: y + h / 2 + 6 + (i % 2) * (fs + 3),
                    fill: seg.up ? ATTRIBUTION_UP : ATTRIBUTION_DOWN,
                    align: 'center',
                    verticalAlign: 'top',
                    font: `${fs}px ${t.font}`,
                  },
                },
              ]
            : []),
        ],
      }
    },
    tooltip: {
      formatter: (p: { dataIndex: number }) => {
        const s = segments[p.dataIndex]
        return s === undefined ? '' : `${s.label}<br/>${signed(s.up ? s.end - s.start : s.start - s.end)}`
      },
    },
    z: 3,
  } as unknown as SeriesOption
  const marks = {
    type: 'line',
    name: '__marks',
    data: [],
    markLine: {
      silent: true,
      symbol: 'none',
      lineStyle: { color: t.muted, type: 'dashed', width: 1 },
      label: { color: t.fgSecondary, fontSize: fs, position: 'end' },
      data: [
        { xAxis: base, label: { formatter: `base value ${short(base)}` } },
        { xAxis: prediction, label: { formatter: `f(x) = ${short(prediction)}`, fontWeight: 600 } },
      ],
    },
  } as SeriesOption
  return cartesian([layer([series], { x: span, y: [-1, 1] }), layer([marks], {})], {
    figure: fig,
    mode: fig.mode,
    x: { label: record.output, limits: [span[0] - 0.04 * width, span[1] + 0.04 * width] },
    y: { show: false, splitLine: false, limits: [-1.6, 1] },
    tooltip: 'item',
    legend: false,
  })
}
