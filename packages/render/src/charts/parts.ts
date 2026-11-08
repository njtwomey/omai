/**
 * Parts of a whole and single readings: pie and donut charts, funnels (shares lost from stage to stage) and gauges
 * (one value against its range). Use them sparingly: bars compare parts more accurately than angles or areas.
 */

import type { EChartsOption } from 'echarts'
import { type Figure, figure as makeFigure } from '../figure/figure.ts'
import { formatNumber } from '../theme/format.ts'
import { categoricalColour } from '../theme/palettes.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { chartFrame } from './plot.ts'

/** Options of the parts charts. */
export interface PartsOptions {
  readonly mode?: ColourMode
  readonly figure?: Figure
  readonly title?: string
}

/**
 * A pie chart, or a donut with an inner radius, each slice labelled with its share.
 *
 * @param parts - The parts: names and values.
 * @param options - Mode, figure, title and the inner radius as a share of the outer (default 0: a pie).
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { pieChart } from 'omai-render/charts'
 * pieChart([{ name: 'a', value: 1 }, { name: 'b', value: 3 }], { inner: 0.5 }).series[0].radius // => ['40%', '80%']
 * ```
 */
export function pieChart(
  parts: readonly { readonly name: string; readonly value: number }[],
  options: PartsOptions & { readonly inner?: number } = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const inner = Math.max(0, Math.min(0.9, options.inner ?? 0))
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    series: [
      {
        type: 'pie',
        radius: [`${Math.round(80 * inner)}%`, '80%'],
        center: ['50%', '54%'],
        data: parts.map((p, i) => ({
          ...p,
          itemStyle: { color: categoricalColour(i), borderColor: t.bg, borderWidth: 2 },
        })),
        label: { color: t.fg, fontFamily: t.font, formatter: '{b}\n{d}%' },
        labelLine: { lineStyle: { color: t.grid } },
      },
    ],
  } as EChartsOption
}

/**
 * A funnel: stages from widest to narrowest, each labelled with its value and its share of the first.
 *
 * @param stages - The stages in order: names and values.
 * @param options - Mode, figure and title.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { funnelChart } from 'omai-render/charts'
 * funnelChart([{ name: 'visit', value: 100 }, { name: 'buy', value: 5 }]).series[0].type // => 'funnel'
 * ```
 */
export function funnelChart(
  stages: readonly { readonly name: string; readonly value: number }[],
  options: PartsOptions = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const first = stages[0]?.value ?? 1
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    series: [
      {
        type: 'funnel',
        top: options.title === undefined ? 16 : 44,
        bottom: 16,
        left: '12%',
        width: '76%',
        sort: 'none',
        gap: 2,
        data: stages.map((s, i) => ({ ...s, itemStyle: { color: categoricalColour(i), borderColor: t.bg } })),
        label: {
          color: '#fff',
          fontFamily: t.font,
          formatter: (p: { name: string; value: number }) =>
            `${p.name}: ${formatNumber(p.value)} (${formatNumber((100 * p.value) / first, { digits: 3 })}%)`,
        },
      },
    ],
  } as EChartsOption
}

/**
 * A gauge: one value on a dial over its range, with coloured bands (low, middle, high).
 *
 * @param value - The reading.
 * @param options - Mode, figure, title, the range (default [0, 100]), the reading's name and unit.
 * @returns The ECharts option.
 * @example
 * ```ts
 * import { gaugeChart } from 'omai-render/charts'
 * gaugeChart(72, { range: [0, 100] }).series[0].max // => 100
 * ```
 */
export function gaugeChart(
  value: number,
  options: PartsOptions & {
    readonly range?: readonly [number, number]
    readonly name?: string
    readonly unit?: string
  } = {},
): EChartsOption {
  const fig = options.figure ?? makeFigure({ mode: options.mode ?? 'light' })
  const mode = options.mode ?? fig.mode
  const t = tokens(mode)
  const [min, max] = options.range ?? [0, 100]
  return {
    ...chartFrame(
      { ...fig, mode },
      { ...(options.title === undefined ? {} : { title: options.title }), legend: false },
    ),
    series: [
      {
        type: 'gauge',
        min,
        max,
        center: ['50%', '60%'],
        radius: '85%',
        axisLine: {
          lineStyle: {
            width: 14,
            color: [
              [0.6, categoricalColour(2)],
              [0.85, categoricalColour(1)],
              [1, categoricalColour(3)],
            ],
          },
        },
        axisTick: { lineStyle: { color: t.grid } },
        splitLine: { lineStyle: { color: t.grid } },
        axisLabel: { color: t.fgSecondary, fontFamily: t.font },
        pointer: { itemStyle: { color: t.fg } },
        title: { color: t.fgSecondary, offsetCenter: [0, '30%'] },
        detail: {
          color: t.fg,
          fontFamily: t.font,
          offsetCenter: [0, '55%'],
          formatter: (v: number) => `${formatNumber(v, { digits: 3 })}${options.unit ?? ''}`,
        },
        data: [{ value, name: options.name ?? '' }],
      },
    ],
  } as EChartsOption
}
