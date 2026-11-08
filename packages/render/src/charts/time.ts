/**
 * Charts of time: a calendar heatmap (one cell per day, laid out by week and weekday, a year at a glance) and a
 * candlestick chart of open, high, low and close per period with the volume beneath.
 */

import type { EChartsOption } from 'echarts'
import { maxOf, minOf } from 'omai-engine/base'
import { type Figure, figure as makeFigure } from '../figure/figure.ts'
import { sequentialRamp } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { chartFrame } from './plot.ts'

/**
 * A calendar heatmap: one cell per day of a year, weeks across and weekdays down, coloured by the day's value.
 *
 * @param days - Each day as `YYYY-MM-DD` and its value.
 * @param options - The year (default the first day's), mode, figure, title and the value's name.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { calendarHeatmap } from 'omai-render/charts'
 * calendarHeatmap([{ date: '2026-01-01', value: 3 }]).calendar.range // => '2026'
 * ```
 */
export function calendarHeatmap(
  days: readonly { readonly date: string; readonly value: number }[],
  options: {
    readonly year?: number
    readonly mode?: ColourMode
    readonly figure?: Figure
    readonly title?: string
    readonly label?: string
  } = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const year = options.year ?? Number((days[0]?.date ?? '2026').slice(0, 4))
  const values = days.map((d) => d.value)
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    visualMap: {
      min: Math.min(0, minOf(values)),
      max: Math.max(1, maxOf(values)),
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 4,
      text: [options.label ?? '', ''],
      inRange: { color: [...sequentialRamp('viridis', mode)] },
      textStyle: { color: t.fgSecondary },
    },
    calendar: {
      range: String(year),
      top: options.title === undefined ? 30 : 56,
      left: 40,
      right: 16,
      cellSize: ['auto', 14],
      itemStyle: { borderColor: t.bg, borderWidth: 2, color: t.surface },
      splitLine: { lineStyle: { color: t.grid } },
      dayLabel: { color: t.fgSecondary, firstDay: 1 },
      monthLabel: { color: t.fgSecondary },
      yearLabel: { show: false },
    },
    series: [{ type: 'heatmap', coordinateSystem: 'calendar', data: days.map((d) => [d.date, d.value]) }],
  } as EChartsOption
}

/** One period of a price series. */
export interface Candle {
  readonly label: string
  readonly open: number
  readonly high: number
  readonly low: number
  readonly close: number
  readonly volume?: number
}

/**
 * A candlestick chart: per period a box from open to close (green when it rose, red when it fell) with wicks to the
 * high and low, an optional moving average, and the volume beneath when given; drag the slider to zoom.
 *
 * @param candles - The periods in order.
 * @param options - Mode, figure, title and a moving-average window.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { candlestickChart } from 'omai-render/charts'
 * candlestickChart([{ label: 'd1', open: 1, high: 2, low: 0.5, close: 1.5 }]).series[0].type // => 'candlestick'
 * ```
 */
export function candlestickChart(
  candles: readonly Candle[],
  options: {
    readonly mode?: ColourMode
    readonly figure?: Figure
    readonly title?: string
    readonly average?: number
  } = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const up = '#3fa66b'
  const down = '#e5484d'
  const labels = candles.map((c) => c.label)
  const withVolume = candles.some((c) => c.volume !== undefined)
  const w = options.average
  const ma =
    w === undefined
      ? undefined
      : candles.map((_, i) => (i + 1 < w ? null : candles.slice(i + 1 - w, i + 1).reduce((a, c) => a + c.close, 0) / w))
  const axis = {
    axisLine: { lineStyle: { color: t.grid } },
    axisLabel: { color: t.fgSecondary },
    splitLine: { lineStyle: { color: t.grid } },
  }
  const top = options.title === undefined ? 24 : 48
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false, tooltip: 'axis' },
    ),
    grid: withVolume
      ? [
          { left: 56, right: 16, top, height: '58%' },
          { left: 56, right: 16, top: '76%', height: '12%' },
        ]
      : [{ left: 56, right: 16, top, bottom: 56 }],
    xAxis: withVolume
      ? [
          { type: 'category', data: labels, gridIndex: 0, ...axis, axisLabel: { show: false } },
          { type: 'category', data: labels, gridIndex: 1, ...axis },
        ]
      : [{ type: 'category', data: labels, ...axis }],
    yAxis: withVolume
      ? [
          { scale: true, gridIndex: 0, ...axis },
          { scale: true, gridIndex: 1, ...axis, splitNumber: 2 },
        ]
      : [{ scale: true, ...axis }],
    dataZoom: [
      { type: 'slider', xAxisIndex: withVolume ? [0, 1] : [0], bottom: 4, height: 18 },
      { type: 'inside', xAxisIndex: withVolume ? [0, 1] : [0] },
    ],
    series: [
      {
        type: 'candlestick',
        name: 'price',
        data: candles.map((c) => [c.open, c.close, c.low, c.high]),
        itemStyle: { color: up, color0: down, borderColor: up, borderColor0: down },
      },
      ...(ma === undefined
        ? []
        : [
            {
              type: 'line',
              name: `${w}-period average`,
              data: ma,
              showSymbol: false,
              smooth: true,
              lineStyle: { width: 1.5, color: t.accent },
            },
          ]),
      ...(withVolume
        ? [
            {
              type: 'bar',
              name: 'volume',
              xAxisIndex: 1,
              yAxisIndex: 1,
              data: candles.map((c) => ({
                value: c.volume ?? 0,
                itemStyle: { color: c.close >= c.open ? up : down, opacity: 0.6 },
              })),
            },
          ]
        : []),
    ],
  } as EChartsOption
}
