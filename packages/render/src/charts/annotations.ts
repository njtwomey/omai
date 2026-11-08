/**
 * Annotations over a chart: arrows from one point to another with a label at the tail, text labels at points, and
 * highlighted spans of x (a period, a region of interest), as one layer that combines with any other.
 */

import type { SeriesOption } from 'echarts'
import { maxOf, minOf } from 'omai-engine/base'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { colourFor } from './_data.ts'
import { type Layer, layer } from './plot.ts'

/** What to annotate: arrows, labels and spans, in data coordinates. */
export interface Annotations {
  readonly arrows?: readonly {
    readonly from: readonly [number, number]
    readonly to: readonly [number, number]
    readonly label?: string
  }[]
  readonly labels?: readonly { readonly at: readonly [number, number]; readonly text: string }[]
  readonly spans?: readonly { readonly x: readonly [number, number]; readonly label?: string }[]
}

/**
 * Annotations as a layer: arrows (with their labels at the tail), labels and shaded spans of x with their labels at
 * the top.
 *
 * @param notes - The arrows, labels and spans.
 * @param options - Colour mode and colour (default the foreground).
 * @returns The layer.
 * @example
 * ```ts
 * import { annotations } from 'omai-render/charts'
 * annotations({ arrows: [{ from: [0, 0], to: [1, 1], label: 'here' }], spans: [{ x: [2, 3] }] }).series.length // => 3
 * ```
 */
export function annotations(
  notes: Annotations,
  options: { readonly mode?: ColourMode; readonly colour?: string } = {},
): Layer {
  const mode = options.mode ?? 'light'
  const t = tokens(mode)
  const colour = options.colour === undefined ? t.fg : colourFor(options.colour, mode, 0)
  const arrows = notes.arrows ?? []
  const labels = notes.labels ?? []
  const spans = notes.spans ?? []
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
      label: { show: true, formatter: '{b}', color: colour, fontFamily: t.font, position: 'top', fontSize: 12 },
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
        label: { color: t.fgSecondary, fontFamily: t.font, position: 'insideTop' },
        data: spans.map((s) => [{ xAxis: s.x[0], name: s.label ?? '' }, { xAxis: s.x[1] }]),
      },
    } as SeriesOption,
  ]
  const xs = [
    ...arrows.flatMap((a) => [a.from[0], a.to[0]]),
    ...labels.map((l) => l.at[0]),
    ...spans.flatMap((s) => [...s.x]),
  ]
  const ys = [...arrows.flatMap((a) => [a.from[1], a.to[1]]), ...labels.map((l) => l.at[1])]
  return layer(series, {
    ...(xs.length > 0 ? { x: [minOf(xs), maxOf(xs)] as [number, number] } : {}),
    ...(ys.length > 0 ? { y: [minOf(ys), maxOf(ys)] as [number, number] } : {}),
  })
}
