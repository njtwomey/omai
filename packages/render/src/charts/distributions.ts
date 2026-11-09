/**
 * Figures of probability distributions and samples: densities from a `Distribution` on its plausible range, Gaussian
 * kernel density estimates, empirical CDFs, probability mass functions as stems, Tukey box plots and violin plots, QQ
 * plots against a reference distribution, and prior and posterior overlays, each as a layer or a ready chart.
 */

import type { EChartsOption, SeriesOption } from 'echarts'
import { maxOf, minOf, OmaiError } from 'omai-engine/base'
import { type Distribution, normal } from 'omai-engine/dists'
import { ecdf, gaussianKde, quantile } from 'omai-engine/stats'
import { type TensorLike, toFlat } from 'omai-engine/tensor'
import type { Range } from '../figure/aspect.ts'
import { extentOf, padRange } from '../figure/limits.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { colourFor, modeOf, numbers } from './_data.ts'
import { deferMode, isModeless } from './_mode.ts'
import { band, lines } from './lines.ts'
import { cartesian, type Layer, layer, type PlotOptions } from './plot.ts'
import { points } from './points.ts'

/** Options shared by the distribution layers. */
export interface DistributionLayerOptions {
  readonly mode?: ColourMode
  readonly name?: string
  readonly colour?: string
  readonly colourOffset?: number
  readonly dash?: 'solid' | 'dashed' | 'dotted'
  /** Fill under the curve with this opacity (default none). */
  readonly fill?: number
}

function scalar(v: unknown): number {
  const flat = toFlat(v as Parameters<typeof toFlat>[0])
  return flat[0] ?? Number.NaN
}

/**
 * The range a distribution's plot should cover: its 0.1% to 99.9% quantiles where finite, otherwise the mean plus
 * and minus four standard deviations.
 *
 * @param dist - A univariate distribution.
 * @param mass - The central probability to cover (default 0.998).
 * @returns The range.
 * @example
 * ```ts
 * import { distributionRange } from 'omai-render/charts'
 * import { normal } from 'omai-engine/dists'
 * distributionRange(normal(0, 1)).map((v) => Math.round(v * 100) / 100) // => [-3.09, 3.09]
 * ```
 */
export function distributionRange(dist: Distribution, mass = 0.998): Range {
  const tail = (1 - mass) / 2
  let lo = Number.NaN
  let hi = Number.NaN
  try {
    lo = scalar(dist.ppf(tail))
    hi = scalar(dist.ppf(1 - tail))
  } catch {
    // Some distributions have no quantile function; fall back to moments.
  }
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo, hi]
  const m = scalar(dist.mean())
  const s = scalar(dist.stddev())
  if (Number.isFinite(m) && Number.isFinite(s) && s > 0) return [m - 4 * s, m + 4 * s]
  return [0, 1]
}

/**
 * The density of a univariate distribution as a curve, from `dist.prob` on an even grid.
 *
 * @param dist - The distribution.
 * @param options - Range (default {@link distributionRange}), number of points (default 200), name, colour and fill.
 * @returns The layer.
 * @example
 * ```ts
 * import { density } from 'omai-render/charts'
 * import { normal } from 'omai-engine/dists'
 * const l = density(normal(0, 1), { range: [-1, 1], n: 3 })
 * l.series[0].data // => [[-1, 0.24197072451914337], [0, 0.3989422804014327], [1, 0.24197072451914337]]
 * ```
 */
export function density(
  dist: Distribution,
  options: DistributionLayerOptions & { readonly range?: Range; readonly n?: number } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => density(dist, { ...options, mode }))
  const [a, b] = options.range ?? distributionRange(dist)
  const n = Math.max(2, options.n ?? 200)
  const x = Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
  const y = toFlat(dist.prob(x))
  return curveLayer(x, y, options)
}

function curveLayer(x: readonly number[], y: readonly number[], options: DistributionLayerOptions): Layer {
  const l = lines(
    [
      {
        x,
        y,
        ...(options.name === undefined ? {} : { name: options.name }),
        ...(options.colour === undefined ? {} : { colour: options.colour }),
        ...(options.dash === undefined ? {} : { dash: options.dash }),
        ...(options.fill === undefined ? {} : { area: options.fill }),
      },
    ],
    { mode: options.mode ?? 'light', colourOffset: options.colourOffset ?? 0 },
  )
  return { ...l, extent: { ...l.extent, baseline: 'y' } }
}

/**
 * A Gaussian kernel density estimate of samples as a curve (omai-engine's `stats.gaussianKde`, Scott's rule by
 * default).
 *
 * @param samples - The sample.
 * @param options - Bandwidth (a rule or factor), range (default the data padded by three bandwidths), points, name and
 *   colour.
 * @returns The layer.
 * @index function A kernel density estimate layer of samples: `kde(samples)`.
 * @example
 * ```ts
 * import { kde } from 'omai-render/charts'
 * kde([0, 1, 2], { n: 50 }).series[0].data.length // => 50
 * ```
 */
export function kde(
  samples: TensorLike | readonly number[],
  options: DistributionLayerOptions & {
    readonly bandwidth?: 'scott' | 'silverman' | number
    readonly range?: Range
    readonly n?: number
  } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => kde(samples, { ...options, mode }))
  const data = numbers(samples).filter(Number.isFinite)
  const est = gaussianKde(data, options.bandwidth === undefined ? {} : { bandwidth: options.bandwidth })
  const sd = Math.sqrt(toFlat(est.covariance)[0] ?? 1)
  const e = extentOf(data) ?? [0, 1]
  const [a, b] = options.range ?? [e[0] - 3 * sd, e[1] + 3 * sd]
  const n = Math.max(2, options.n ?? 200)
  const x = Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
  return curveLayer(x, toFlat(est.evaluate(x)), options)
}

/**
 * The empirical CDF of samples as a step curve (omai-engine's `stats.ecdf`), from 0 before the least value to 1
 * after the greatest.
 *
 * @param samples - The sample.
 * @param options - Name, colour and colour mode.
 * @returns The layer.
 * @example
 * ```ts
 * import { ecdfCurve } from 'omai-render/charts'
 * ecdfCurve([1, 3, 3, 4]).series[0].data // => [[1, 0], [1, 0.25], [3, 0.75], [4, 1]]
 * ```
 */
export function ecdfCurve(samples: TensorLike | readonly number[], options: DistributionLayerOptions = {}): Layer {
  if (isModeless(options)) return deferMode((mode) => ecdfCurve(samples, { ...options, mode }))
  const F = ecdf(numbers(samples).filter(Number.isFinite))
  const q = toFlat(F.quantiles)
  const p = toFlat(F.probabilities)
  const x = [q[0] ?? 0, ...q]
  const y = [0, ...p]
  const l = lines(
    [
      {
        x,
        y,
        step: 'end',
        ...(options.name === undefined ? {} : { name: options.name }),
        ...(options.colour === undefined ? {} : { colour: options.colour }),
      },
    ],
    { mode: options.mode ?? 'light', colourOffset: options.colourOffset ?? 0 },
  )
  return { ...l, extent: { ...l.extent, y: [0, 1] } }
}

/**
 * The probability mass function of a discrete distribution as stems with markers.
 *
 * @param dist - The distribution.
 * @param options - The integer range (default from the support or the quantiles), name and colour.
 * @returns The layer.
 * @example
 * ```ts
 * import { pmf } from 'omai-render/charts'
 * import { bernoulli } from 'omai-engine/dists'
 * pmf(bernoulli({ probs: 0.25 }), { range: [0, 1] }).series[1].data // => [[0, 0.75], [1, 0.25]]
 * ```
 */
export function pmf(dist: Distribution, options: DistributionLayerOptions & { readonly range?: Range } = {}): Layer {
  if (isModeless(options)) return deferMode((mode) => pmf(dist, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const s = dist.support
  const [lo, hi] =
    options.range ??
    (s.kind === 'integerInterval' && Number.isFinite(s.high) && s.high - s.low <= 200
      ? [s.low, s.high]
      : s.kind === 'boolean'
        ? [0, 1]
        : (() => {
            const r = distributionRange(dist)
            return [
              Math.max(s.kind === 'integerInterval' ? s.low : Number.NEGATIVE_INFINITY, Math.floor(r[0])),
              Math.ceil(r[1]),
            ] as const
          })())
  const ks: number[] = []
  for (let k = Math.ceil(lo); k <= Math.floor(hi) && ks.length < 1000; k++) ks.push(k)
  const pk = toFlat(dist.prob(ks))
  const colour = colourFor(options.colour, mode, options.colourOffset ?? 0)
  const name = options.name ?? '__pmf'
  const stems: SeriesOption = {
    type: 'line',
    name: `__stems`,
    data: ks.flatMap((k, i) => [
      [k, 0],
      [k, pk[i] ?? 0],
      [k, null],
    ]),
    showSymbol: false,
    connectNulls: false,
    color: colour,
    lineStyle: { color: colour, width: 1.5 },
    silent: true,
    tooltip: { show: false },
  } as SeriesOption
  const heads: SeriesOption = {
    type: 'scatter',
    name,
    data: ks.map((k, i) => [k, pk[i] ?? 0]),
    symbolSize: 7,
    color: colour,
    itemStyle: { color: colour },
  } as SeriesOption
  return layer([stems, heads], {
    x: [ks[0] ?? 0, ks[ks.length - 1] ?? 1],
    y: [0, Math.max(0, maxOf(pk))],
    baseline: 'y',
  })
}

/** A named sample for box and violin plots. */
export interface SampleGroup {
  readonly name: string
  readonly values: TensorLike | readonly number[]
  readonly colour?: string
}

/** Tukey's box statistics of a sample: quartiles, whiskers at 1.5 IQR and the outliers beyond. */
export interface BoxStats {
  readonly low: number
  readonly q1: number
  readonly median: number
  readonly q3: number
  readonly high: number
  readonly outliers: readonly number[]
}

/**
 * Tukey's box statistics: quartiles (linear interpolation, numpy's default), whiskers at the most extreme data within
 * 1.5 IQR of the box, and outliers beyond.
 *
 * @param values - The sample.
 * @param whisker - The whisker reach in IQRs (default 1.5).
 * @returns The statistics.
 * @throws OmaiError with code `value` for an empty sample.
 * @example
 * ```ts
 * import { boxStats } from 'omai-render/charts'
 * boxStats([1, 2, 3, 4, 100]) // => { low: 1, q1: 2, median: 3, q3: 4, high: 4, outliers: [100] }
 * ```
 */
export function boxStats(values: TensorLike | readonly number[], whisker = 1.5): BoxStats {
  const v = numbers(values)
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  if (v.length === 0) throw new OmaiError('value', 'boxStats: the sample is empty')
  const [q1 = 0, median = 0, q3 = 0] = toFlat(quantile(v, [0.25, 0.5, 0.75]))
  const reach = whisker * (q3 - q1)
  const inside = v.filter((x) => x >= q1 - reach && x <= q3 + reach)
  return {
    low: inside[0] ?? q1,
    q1,
    median,
    q3,
    high: inside[inside.length - 1] ?? q3,
    outliers: v.filter((x) => x < q1 - reach || x > q3 + reach),
  }
}

/**
 * Box plots of samples, one per category, with outliers as points.
 *
 * @param groups - The samples.
 * @param options - Colour mode and colour.
 * @returns The layer, categories on the x axis.
 * @index function Box plots of groups: `boxPlot([{ name: 'a', values }])`.
 * @example
 * ```ts
 * import { boxPlot } from 'omai-render/charts'
 * boxPlot([{ name: 'a', values: [1, 2, 3, 4, 100] }]).series[1].data // => [[0, 100]]
 * ```
 */
export function boxPlot(
  groups: readonly SampleGroup[],
  options: { readonly mode?: ColourMode; readonly colour?: string } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => boxPlot(groups, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const stats = groups.map((g) => boxStats(g.values))
  const colour = colourFor(options.colour, mode, 0)
  const series: SeriesOption[] = [
    {
      type: 'boxplot',
      name: '__box',
      data: stats.map((s, i) => ({
        value: [s.low, s.q1, s.median, s.q3, s.high],
        itemStyle: { color: 'transparent', borderColor: colourFor(groups[i]?.colour, mode, i) },
      })),
      itemStyle: { color: 'transparent', borderColor: colour, borderWidth: 1.5 },
      boxWidth: [8, 40],
    } as SeriesOption,
    {
      type: 'scatter',
      name: '__outliers',
      data: stats.flatMap((s, i) => s.outliers.map((o) => [i, o])),
      symbolSize: 5,
      itemStyle: { color: 'transparent', borderColor: colourFor('muted', mode), borderWidth: 1 },
    } as SeriesOption,
  ]
  const all = stats.flatMap((s) => [s.low, s.high, ...s.outliers])
  const e = extentOf(all)
  return layer(series, e === undefined ? {} : { y: e }, { xCategories: groups.map((g) => g.name) })
}

interface ViolinApi {
  coord(p: readonly (number | string)[]): number[]
  size(p: readonly number[]): number[]
}

/**
 * Violin plots: a mirrored kernel density estimate of each sample, with its median and quartiles marked.
 *
 * @param groups - The samples.
 * @param options - Bandwidth, points per density (default 80), width as a fraction of a category (default 0.8), colour
 *   mode.
 * @returns The layer, categories on the x axis.
 * @example
 * ```ts
 * import { violinPlot } from 'omai-render/charts'
 * violinPlot([{ name: 'a', values: [0, 1, 1, 2] }]).xCategories // => ['a']
 * ```
 */
export function violinPlot(
  groups: readonly SampleGroup[],
  options: {
    readonly mode?: ColourMode
    readonly bandwidth?: 'scott' | 'silverman' | number
    readonly n?: number
    readonly width?: number
  } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => violinPlot(groups, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const width = options.width ?? 0.8
  const shapes = groups.map((g) => {
    const v = numbers(g.values).filter(Number.isFinite)
    const est = gaussianKde(v, options.bandwidth === undefined ? {} : { bandwidth: options.bandwidth })
    const sd = Math.sqrt(toFlat(est.covariance)[0] ?? 1)
    const e = extentOf(v) ?? [0, 1]
    const [a, b] = [e[0] - 2 * sd, e[1] + 2 * sd]
    const n = Math.max(8, options.n ?? 80)
    const y = Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
    const d = toFlat(est.evaluate(y))
    const peak = maxOf(d)
    return { y, d: d.map((x) => (peak > 0 ? x / peak : 0)), stats: boxStats(v), range: [a, b] as Range }
  })
  const fg = colourFor('fg', mode)
  const series: SeriesOption = {
    type: 'custom',
    name: '__violin',
    data: groups.map((_, i) => [i]),
    renderItem: (params: { dataIndex: number }, api: ViolinApi): Record<string, unknown> => {
      const i = params.dataIndex
      const s = shapes[i]
      if (s === undefined) return { type: 'group', children: [] }
      const [cx = 0] = api.coord([i, 0])
      const half = ((api.size([1, 0])[0] ?? 0) * width) / 2
      const right = s.y.map((y, k) => [cx + (s.d[k] ?? 0) * half, api.coord([i, y])[1] ?? 0])
      const left = s.y.map((y, k) => [cx - (s.d[k] ?? 0) * half, api.coord([i, y])[1] ?? 0]).reverse()
      const colour = colourFor(groups[i]?.colour, mode, i)
      const yq = (v: number): number => api.coord([i, v])[1] ?? 0
      return {
        type: 'group',
        children: [
          {
            type: 'polygon',
            shape: { points: [...right, ...left] },
            style: { fill: colour, opacity: 0.35, stroke: colour, lineWidth: 1 },
          },
          {
            type: 'line',
            shape: { x1: cx, y1: yq(s.stats.q1), x2: cx, y2: yq(s.stats.q3) },
            style: { stroke: fg, lineWidth: 3 },
          },
          {
            type: 'circle',
            shape: { cx, cy: yq(s.stats.median), r: 3 },
            style: { fill: colourFor('bg', mode), stroke: fg },
          },
        ],
      }
    },
    tooltip: { show: false },
  } as unknown as SeriesOption
  const e = extentOf(shapes.flatMap((s) => s.range))
  return layer([series], e === undefined ? {} : { y: e }, { xCategories: groups.map((g) => g.name) })
}

/**
 * A QQ plot: sorted samples against the quantiles of a reference distribution (standard normal by default) at the
 * plotting positions (i - 0.5) / n, with the line through the quartiles.
 *
 * @param samples - The sample.
 * @param options - Reference distribution, colour mode and colour.
 * @returns The layer.
 * @example
 * ```ts
 * import { qqPlot } from 'omai-render/charts'
 * qqPlot([3, 1, 2]).series[0].data.map((p) => p.value[1]) // => [1, 2, 3]
 * ```
 */
export function qqPlot(
  samples: TensorLike | readonly number[],
  options: { readonly reference?: Distribution; readonly mode?: ColourMode; readonly colour?: string } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => qqPlot(samples, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const ref = options.reference ?? normal(0, 1)
  const v = numbers(samples)
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  const n = v.length
  const theory = toFlat(ref.ppf(v.map((_, i) => (i + 0.5) / n)))
  const pts = points(
    [{ x: theory, y: v, name: '__qq', ...(options.colour === undefined ? {} : { colour: options.colour }) }],
    { mode },
  )
  const [t1 = 0, t3 = 1] = toFlat(ref.ppf([0.25, 0.75]))
  const [s1 = 0, s3 = 1] = toFlat(quantile(v, [0.25, 0.75]))
  const slope = t3 > t1 ? (s3 - s1) / (t3 - t1) : 1
  const xr = padRange(extentOf(theory) ?? [0, 1], 0.05)
  const ref2 = lines(
    [
      {
        x: [xr[0], xr[1]],
        y: [s1 + slope * (xr[0] - t1), s1 + slope * (xr[1] - t1)],
        name: '__reference',
        colour: 'muted',
        dash: 'dashed',
        width: 1.5,
      },
    ],
    { mode },
  )
  return { ...pts, series: [...pts.series, ...ref2.series] }
}

/**
 * Prior and posterior densities overlaid: the prior dashed and muted, the posterior filled, on a range covering both.
 *
 * @param prior - The prior.
 * @param posterior - The posterior.
 * @param options - Range, points, colour mode and the names (default `prior` and `posterior`).
 * @returns The layer.
 * @example
 * ```ts
 * import { priorPosterior } from 'omai-render/charts'
 * import { normal } from 'omai-engine/dists'
 * priorPosterior(normal(0, 2), normal(1, 0.5)).series.map((s) => s.name) // => ['prior', 'posterior']
 * ```
 */
export function priorPosterior(
  prior: Distribution,
  posterior: Distribution,
  options: {
    readonly range?: Range
    readonly n?: number
    readonly mode?: ColourMode
    readonly names?: readonly [string, string]
  } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => priorPosterior(prior, posterior, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const a = distributionRange(prior)
  const b = distributionRange(posterior)
  const range: Range = options.range ?? [Math.min(a[0], b[0]), Math.max(a[1], b[1])]
  const [pn = 'prior', qn = 'posterior'] = options.names ?? []
  const p = density(prior, {
    range,
    mode,
    name: pn,
    colour: 'muted',
    dash: 'dashed',
    ...(options.n === undefined ? {} : { n: options.n }),
  })
  const q = density(posterior, {
    range,
    mode,
    name: qn,
    colour: 'accent',
    fill: 0.2,
    ...(options.n === undefined ? {} : { n: options.n }),
  })
  return layer([...p.series, ...q.series], {
    x: range,
    y: [0, Math.max(p.extent.y?.[1] ?? 0, q.extent.y?.[1] ?? 0)],
    baseline: 'y',
  })
}

/**
 * Densities of several distributions on one chart.
 *
 * @param dists - Distributions with names.
 * @param options - Plot options, the range and points.
 * @returns The ECharts option.
 * @index function Densities of distributions on one chart: `densityChart([{ name, dist }])`.
 * @example
 * ```ts
 * import { densityChart } from 'omai-render/charts'
 * import { normal } from 'omai-engine/dists'
 * densityChart([{ name: 'N(0,1)', dist: normal(0, 1) }]).series.length // => 1
 * ```
 */
export function densityChart(
  dists: readonly { readonly name: string; readonly dist: Distribution; readonly colour?: string }[],
  options: PlotOptions & { readonly range?: Range; readonly n?: number } = {},
): EChartsOption {
  const mode = modeOf(options)
  const ranges = dists.map((d) => distributionRange(d.dist))
  const range: Range = options.range ?? [minOf(ranges.map((r) => r[0])), maxOf(ranges.map((r) => r[1]))]
  const layers = dists.map((d, i) =>
    density(d.dist, {
      range,
      mode,
      name: d.name,
      colourOffset: i,
      ...(d.colour === undefined ? {} : { colour: d.colour }),
      ...(options.n === undefined ? {} : { n: options.n }),
    }),
  )
  return cartesian(layers, { tooltip: 'axis', ...options })
}

/**
 * Mean curves with uncertainty bands, a common pairing (posterior predictive mean and interval, say).
 *
 * @param x - The x values.
 * @param mean - The mean at each x.
 * @param lower - The lower edge.
 * @param upper - The upper edge.
 * @param options - Name, colour and colour mode.
 * @returns The layer: the band below the curve.
 * @example
 * ```ts
 * import { meanBand } from 'omai-render/charts'
 * meanBand([0, 1], [0, 1], [-1, 0], [1, 2]).series.length // => 2
 * ```
 */
export function meanBand(
  x: TensorLike | readonly number[],
  mean: TensorLike | readonly number[],
  lower: TensorLike | readonly number[],
  upper: TensorLike | readonly number[],
  options: {
    readonly name?: string
    readonly colour?: string
    readonly mode?: ColourMode
    readonly opacity?: number
  } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => meanBand(x, mean, lower, upper, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const colour = options.colour ?? 'tab0'
  const b = band(
    [{ x, lower, upper, colour, ...(options.opacity === undefined ? {} : { opacity: options.opacity }) }],
    { mode },
  )
  const l = lines([{ x, y: mean, colour, ...(options.name === undefined ? {} : { name: options.name }) }], { mode })
  const e = extentOf([...(b.extent.y ?? []), ...(l.extent.y ?? [])])
  return layer([...b.series, ...l.series], {
    ...(l.extent.x === undefined ? {} : { x: l.extent.x }),
    ...(e === undefined ? {} : { y: e }),
  })
}

/**
 * A ridgeline plot (a joyplot): one kernel density per group, stacked one above another with a little overlap, so
 * many distributions compare at a glance. Group k's baseline sits at y = k (label the y axis with the groups' names).
 *
 * @param groups - The groups, bottom first: names and samples.
 * @param options - The shared range (default the pooled data's, padded), grid points (default 200), the overlap (the
 *   tallest density's height in rows, default 1.6) and colour mode.
 * @returns The layer: a filled band and an outline per group.
 * @example
 * ```ts
 * import { ridgeline } from 'omai-render/charts'
 * ridgeline([{ name: 'a', values: [0, 1, 2] }, { name: 'b', values: [1, 2, 3] }], { n: 5 }).series.length // => 4
 * ```
 */
export function ridgeline(
  groups: readonly { readonly name: string; readonly values: readonly number[] }[],
  options: { readonly range?: Range; readonly n?: number; readonly overlap?: number; readonly mode?: ColourMode } = {},
): Layer {
  if (isModeless(options)) return deferMode((mode) => ridgeline(groups, { ...options, mode }))
  const mode = options.mode ?? 'light'
  const all = groups.flatMap((g) => [...g.values])
  const range = options.range ?? padRange([minOf(all), maxOf(all)], 0.08)
  const n = Math.max(2, options.n ?? 200)
  const x = Array.from({ length: n }, (_, i) => range[0] + ((range[1] - range[0]) * i) / (n - 1))
  const dens = groups.map((g) =>
    g.values.length > 1 ? toFlat(gaussianKde([...g.values]).evaluate(x)) : x.map(() => 0),
  )
  const top = Math.max(1e-12, maxOf(dens.flat()))
  const scale = (options.overlap ?? 1.6) / top
  const bands = band(
    groups.map((g, k) => ({
      x,
      lower: x.map(() => k),
      upper: (dens[k] ?? []).map((d) => k + d * scale),
      name: g.name,
      opacity: 0.55,
      edges: false,
    })),
    { mode },
  )
  const outlines = lines(
    groups.map((g, k) => ({ x, y: (dens[k] ?? []).map((d) => k + d * scale), name: `__${g.name}`, width: 1.25 })),
    { mode },
  )
  return layer([...bands.series, ...outlines.series], {
    x: range,
    y: [0, groups.length - 1 + (options.overlap ?? 1.6)],
  })
}
