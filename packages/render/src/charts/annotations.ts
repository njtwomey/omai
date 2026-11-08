/**
 * Annotations over a chart: arrows from one point to another with a label at the tail, text labels at points,
 * highlighted spans of x (a period) and shaded regions (a rectangle, or a band across one axis), as one layer that
 * combines with any other.
 */

import type { SeriesOption } from 'echarts'
import { maxOf, minOf } from 'omai-engine/base'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { colourFor } from './_data.ts'
import { type Layer, layer } from './plot.ts'

/** A shaded region of the plot: a rectangle (`x` and `y`), or a band across the whole plot (only one of them). */
export interface Region {
  /** The x range of the region (omit for a horizontal band across the plot). */
  readonly x?: readonly [number, number]
  /** The y range of the region (omit for a vertical band). */
  readonly y?: readonly [number, number]
  /** Text at the region's top. */
  readonly label?: string
  /** The fill colour (default the accent); a token name or any CSS colour. */
  readonly colour?: string
}

/** What to annotate: arrows, labels, spans and regions, in data coordinates. */
export interface Annotations {
  readonly arrows?: readonly {
    readonly from: readonly [number, number]
    readonly to: readonly [number, number]
    readonly label?: string
  }[]
  readonly labels?: readonly { readonly at: readonly [number, number]; readonly text: string }[]
  readonly spans?: readonly { readonly x: readonly [number, number]; readonly label?: string }[]
  /** Shaded rectangles and bands, each with an optional label and colour. */
  readonly regions?: readonly Region[]
}

/**
 * Annotations as a layer: arrows (with their labels at the tail), labels, shaded spans of x and shaded regions (a
 * rectangle given by `x` and `y`, or a band across the plot given by one of them), with their labels at the top.
 *
 * @param notes - The arrows, labels, spans and regions.
 * @param options - Colour mode, colour (default the foreground) and label size (default the figure's type size).
 * @returns The layer.
 * @index function Arrows, labels, spans and shaded regions: `annotations({ regions })`.
 * @example
 * ```ts
 * import { annotations } from 'omai-render/charts'
 * annotations({ arrows: [{ from: [0, 0], to: [1, 1], label: 'here' }], spans: [{ x: [2, 3] }] }).series.length // => 3
 * annotations({ regions: [{ x: [0, 1], y: [2, 3], label: 'zone' }] }).series[2].markArea.data.length // => 1
 * ```
 */
export function annotations(
  notes: Annotations,
  options: { readonly mode?: ColourMode; readonly colour?: string; readonly fontSize?: number } = {},
): Layer {
  const mode = options.mode ?? 'light'
  const t = tokens(mode)
  const colour = options.colour === undefined ? t.fg : colourFor(options.colour, mode, 0)
  const arrows = notes.arrows ?? []
  const labels = notes.labels ?? []
  const spans = notes.spans ?? []
  const regions = (notes.regions ?? []).filter((r) => r.x !== undefined || r.y !== undefined)
  const size = options.fontSize === undefined ? {} : { fontSize: options.fontSize }
  const series: SeriesOption[] = [
    {
      type: 'lines',
      coordinateSystem: 'cartesian2d',
      name: '__arrows',
      data: arrows.map((a) => ({ coords: [[...a.from], [...a.to]] })),
      symbol: ['none', 'arrow'],
      symbolSize: 9,
      lineStyle: { color: colour, width: 1.5, curveness: 0.15 },
      silent: true,
      z: 10,
    } as unknown as SeriesOption,
    {
      type: 'scatter',
      name: '__labels',
      data: [
        ...arrows.filter((a) => a.label !== undefined).map((a) => ({ value: [...a.from], name: a.label })),
        ...labels.map((l) => ({ value: [...l.at], name: l.text })),
      ],
      symbolSize: 0,
      silent: true,
      label: { show: true, formatter: '{b}', color: colour, fontFamily: t.font, position: 'top', ...size },
      z: 10,
    } as SeriesOption,
    {
      type: 'line',
      name: '__spans',
      data: [],
      silent: true,
      markArea: {
        silent: true,
        itemStyle: { color: t.accent, opacity: 0.12 },
        label: { color: t.fgSecondary, fontFamily: t.font, position: 'insideTop', ...size },
        data: [
          ...spans.map((s) => [{ xAxis: s.x[0], name: s.label ?? '' }, { xAxis: s.x[1] }]),
          // A rectangle names both corners; a band names only the axis it spans (ECharts stretches it across the plot).
          ...regions.map((r) => [
            {
              name: r.label ?? '',
              ...(r.x === undefined ? {} : { xAxis: r.x[0] }),
              ...(r.y === undefined ? {} : { yAxis: r.y[0] }),
              ...(r.colour === undefined ? {} : { itemStyle: { color: colourFor(r.colour, mode, 0), opacity: 0.18 } }),
            },
            { ...(r.x === undefined ? {} : { xAxis: r.x[1] }), ...(r.y === undefined ? {} : { yAxis: r.y[1] }) },
          ]),
        ],
      },
    } as SeriesOption,
  ]
  const xs = [
    ...arrows.flatMap((a) => [a.from[0], a.to[0]]),
    ...labels.map((l) => l.at[0]),
    ...spans.flatMap((s) => [...s.x]),
    ...regions.flatMap((r) => [...(r.x ?? [])]),
  ]
  const ys = [
    ...arrows.flatMap((a) => [a.from[1], a.to[1]]),
    ...labels.map((l) => l.at[1]),
    ...regions.flatMap((r) => [...(r.y ?? [])]),
  ]
  return layer(series, {
    ...(xs.length > 0 ? { x: [minOf(xs), maxOf(xs)] as [number, number] } : {}),
    ...(ys.length > 0 ? { y: [minOf(ys), maxOf(ys)] as [number, number] } : {}),
  })
}
