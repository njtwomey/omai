/**
 * The view of a `CurveSet`: several curves on shared axes (an optimiser race, a comparison of schemes), coloured by
 * group and dashed within one, with bands and a right-hand axis where asked, and frames that reveal the curves along x
 * for a player.
 */

import type { EChartsOption } from 'echarts'
import type { CurveSeries, CurveSet } from 'mdml-engine/exposition'
import { referenceLines } from '../charts/functions.ts'
import { band, lines } from '../charts/lines.ts'
import { type AxisSpec, cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { limitsOf, short, spanOf, type ViewFrames, type ViewOptions, viewFigure } from './_view.ts'

/** Options of {@link curveSetView} and {@link curveSetFrames}. */
export interface CurvesViewOptions extends ViewOptions {
  /** Most frames of a reveal (default 60). */
  readonly maxFrames?: number
}

const DASHES = ['solid', 'dashed', 'dotted'] as const

/**
 * The colour and dash of every series: one colour per group (or per series without groups), dashes within a group.
 *
 * @param series - The series, with their groups.
 * @returns One colour and dash per series, in order.
 */
export function curveStyles(series: readonly CurveSeries[]): { colour: string; dash: 'solid' | 'dashed' | 'dotted' }[] {
  const groups: string[] = []
  const seen = new Map<string, number>()
  return series.map((s, i) => {
    if (s.group === undefined) return { colour: categoricalColour(groups.length + i), dash: 'solid' }
    let g = groups.indexOf(s.group)
    if (g < 0) {
      groups.push(s.group)
      g = groups.length - 1
    }
    const k = seen.get(s.group) ?? 0
    seen.set(s.group, k + 1)
    return { colour: categoricalColour(g), dash: DASHES[k % DASHES.length] ?? 'solid' }
  })
}

/** The band of a series, if it has one: y ± spread or its explicit edges. */
function bandOf(s: CurveSeries): { lower: number[]; upper: number[] } | undefined {
  if (s.lower !== undefined && s.upper !== undefined) return { lower: [...s.lower], upper: [...s.upper] }
  const spread = s.spread
  if (spread === undefined) return undefined
  return { lower: s.y.map((v, i) => v - (spread[i] ?? 0)), upper: s.y.map((v, i) => v + (spread[i] ?? 0)) }
}

// Fixed limits over every value an axis reads (bands included), padded a little, so a reveal never rescales.
function axisLimits(values: readonly number[], log: boolean): [number, number] {
  if (log) return limitsOf(values, true)
  const [lo, hi] = spanOf(values)
  const pad = hi > lo ? 0.04 * (hi - lo) : Math.max(1e-9, Math.abs(hi) * 0.1) || 1
  return [lo - pad, hi + pad]
}

/**
 * Several curves on shared axes as one chart, drawn up to a position along x: each series with its band, coloured by
 * group (dashed within one), the right-hand axis for series that ask for it, and the record's marks. Up to `upTo`, the
 * curves stop there with a dot on each series' latest value; the axes always span the whole record.
 *
 * @param record - The record.
 * @param options - Mode and figure, and how far along x to draw (default all of it).
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { curveSet } from 'mdml-engine/exposition'
 * import { curveSetView } from 'mdml-render/views'
 * const r = curveSet({ x: [1, 2, 3], xLabel: 'epoch', yLabel: 'loss', series: [{ name: 'sgd', y: [1, 0.6, 0.4] }, { name: 'adam', y: [0.8, 0.3, 0.2] }] })
 * curveSetView(r).xAxis[0].name // => 'epoch'
 * ```
 */
export function curveSetView(
  record: CurveSet,
  options: CurvesViewOptions & { readonly upTo?: number } = {},
): EChartsOption {
  const fig = viewFigure(options, { aspect: 1.6 })
  const mode = fig.mode
  const styles = curveStyles(record.series)
  const upTo = options.upTo ?? Number.POSITIVE_INFINITY
  const left: number[] = []
  const right: number[] = []
  const xs: number[] = []
  for (const s of record.series) {
    const x = s.x ?? record.x ?? []
    xs.push(...x)
    const target = s.axis === 'right' ? right : left
    target.push(...s.y)
    const b = bandOf(s)
    if (b !== undefined) target.push(...b.lower, ...b.upper)
  }
  const layers: Layer[] = []
  const dots: Layer[] = []
  const partial = Number.isFinite(upTo)
  for (const [i, s] of record.series.entries()) {
    const style = styles[i] ?? { colour: 'tab0', dash: 'solid' as const }
    const x = s.x ?? record.x ?? []
    const keep = x.flatMap((v, j) => (v <= upTo ? [j] : []))
    const px = keep.map((j) => x[j] ?? 0)
    const py = keep.map((j) => s.y[j] ?? Number.NaN)
    const secondary = s.axis === 'right'
    const b = bandOf(s)
    if (b !== undefined && keep.length > 0) {
      const l = band(
        [
          {
            x: px,
            lower: keep.map((j) => b.lower[j] ?? Number.NaN),
            upper: keep.map((j) => b.upper[j] ?? Number.NaN),
            colour: style.colour,
            opacity: 0.16,
            name: `__band${i}`,
          },
        ],
        { mode },
      )
      layers.push(secondary ? { ...l, secondary: true } : l)
    }
    const l = lines([{ x: px, y: py, name: s.name, colour: style.colour, dash: style.dash, width: 1.6 }], { mode })
    layers.push(secondary ? { ...l, secondary: true } : l)
    if (partial && keep.length > 0) {
      const last = keep.length - 1
      const d = points([{ x: [px[last] ?? 0], y: [py[last] ?? 0], name: `__now${i}`, colour: style.colour, size: 7 }], {
        mode,
      })
      dots.push(secondary ? { ...d, secondary: true } : d)
    }
  }
  const marks = (record.marks ?? []).map((m) => (m.label === undefined ? m.x : { value: m.x, label: m.label }))
  if (marks.length > 0) layers.unshift(referenceLines({ x: marks }, { mode, colour: 'fgSecondary' }))
  const xLog = record.xLog === true
  const yLog = record.yLog === true
  const x: AxisSpec = { label: record.xLabel, scale: xLog ? 'log' : 'linear', limits: limitsOf(xs, xLog) }
  const y: AxisSpec = {
    ...(record.yLabel === undefined ? {} : { label: record.yLabel }),
    scale: yLog ? 'log' : 'linear',
    limits: axisLimits(left, yLog),
  }
  const shared = record.x !== undefined && record.series.every((s) => s.x === undefined)
  return cartesian([...layers, ...dots], {
    figure: fig,
    mode,
    x,
    y,
    ...(right.length > 0
      ? { y2: { ...(record.y2Label === undefined ? {} : { label: record.y2Label }), limits: axisLimits(right, false) } }
      : {}),
    tooltip: shared ? 'axis' : 'item',
    legend: record.series.length > 1,
  })
}

/**
 * A curve set as frames for a player, revealing the curves along x (a race run again): frame f draws every series up
 * to the f-th of (at most `maxFrames`) evenly spread positions, the last frame the whole record.
 *
 * @param record - The record.
 * @param options - Mode, figure and the most frames.
 * @returns The frames.
 */
export function curveSetFrames(record: CurveSet, options: CurvesViewOptions = {}): ViewFrames {
  const all = [...new Set(record.series.flatMap((s) => [...(s.x ?? record.x ?? [])]))]
    .filter(Number.isFinite)
    .sort((a, b) => a - b)
  const max = Math.max(2, options.maxFrames ?? 60)
  const stops =
    all.length <= max
      ? all
      : [...new Set(Array.from({ length: max }, (_, j) => all[Math.round((j * (all.length - 1)) / (max - 1))] ?? 0))]
  const count = Math.max(1, stops.length)
  const at = (frame: number): number => stops[Math.min(Math.max(frame, 0), count - 1)] ?? Number.POSITIVE_INFINITY
  return {
    count,
    label: (f) => `${record.xLabel} ${short(at(f))}`,
    option: (f) => curveSetView(record, { ...options, ...(f >= count - 1 ? {} : { upTo: at(f) }) }),
  }
}
