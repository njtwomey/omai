/**
 * The ECharts themes of the light and dark colour modes: tab10 series colours, token-coloured text, axes, legends,
 * tooltips, visual maps and data zoom, with animation off so figures that update every frame do not tween, plus
 * `baseOption`, the base of every chart option.
 */

import { tab10 } from './palettes.ts'
import { type ColourMode, tokens } from './tokens.ts'

/**
 * The name under which a mode's theme is registered with ECharts (`echarts.registerTheme(themeName(mode),
 * echartsTheme(mode))`, done once by `mdml-render/react`'s `Chart`).
 *
 * @param mode - The colour mode.
 * @returns `mdml-light` or `mdml-dark`.
 * @example
 * ```ts
 * import { themeName } from 'mdml-render/theme'
 * themeName('dark') // => 'mdml-dark'
 * ```
 */
export function themeName(mode: ColourMode): string {
  return `mdml-${mode}`
}

/**
 * The style of a cartesian axis in a theme: lines, labels, split lines and the axis pointer, without ticks.
 *
 * @param mode - The colour mode.
 * @param fontSize - The base font size in pixels (default 11).
 * @returns An ECharts axis style fragment.
 * @example
 * ```ts
 * import { axisStyle } from 'mdml-render/theme'
 * axisStyle('light').splitLine.lineStyle.color // => '#d0d7de'
 * ```
 */
export function axisStyle(
  mode: ColourMode,
  fontSize = 11,
): {
  readonly axisLine: { readonly show: boolean; readonly lineStyle: { readonly color: string } }
  readonly axisTick: { readonly show: boolean }
  readonly axisLabel: {
    readonly color: string
    readonly fontSize: number
    readonly fontFamily: string
    readonly hideOverlap: boolean
  }
  readonly splitLine: { readonly show: boolean; readonly lineStyle: { readonly color: string; readonly width: number } }
  readonly nameTextStyle: { readonly color: string; readonly fontSize: number; readonly fontFamily: string }
  readonly nameLocation: 'middle'
  readonly nameGap: number
  readonly axisPointer: {
    readonly lineStyle: { readonly color: string; readonly width: number; readonly type: 'dashed' }
    readonly label: {
      readonly backgroundColor: string
      readonly color: string
      readonly fontSize: number
      readonly fontFamily: string
    }
  }
} {
  const t = tokens(mode)
  return {
    axisLine: { show: true, lineStyle: { color: t.axis } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: fontSize - 1, fontFamily: t.font, hideOverlap: true },
    splitLine: { show: true, lineStyle: { color: t.grid, width: 1 } },
    nameTextStyle: { color: t.fgSecondary, fontSize, fontFamily: t.font },
    nameLocation: 'middle',
    nameGap: Math.round(fontSize * 2.5),
    axisPointer: {
      lineStyle: { color: t.muted, width: 1, type: 'dashed' },
      label: { backgroundColor: t.fgSecondary, color: t.bg, fontSize: fontSize - 1, fontFamily: t.font },
    },
  }
}

/**
 * The ECharts theme of a colour mode: tab10 series colours, token-coloured text, axes, legends, tooltips, visual maps
 * and data zoom, and no animation (figures that update every frame must not tween).
 *
 * @param mode - The colour mode.
 * @returns A theme object for `echarts.registerTheme`.
 * @example
 * ```ts
 * import { echartsTheme } from 'mdml-render/theme'
 * echartsTheme('light').color[0] // => '#1f77b4'
 * ```
 */
export function echartsTheme(mode: ColourMode): {
  readonly color: readonly string[]
  readonly backgroundColor: string
  readonly [key: string]: unknown
} {
  const t = tokens(mode)
  const axis = axisStyle(mode)
  return {
    color: tab10,
    backgroundColor: 'transparent',
    textStyle: { fontFamily: t.font, color: t.fg, fontSize: 11 },
    title: { textStyle: { color: t.fg, fontWeight: 600, fontSize: 13 }, subtextStyle: { color: t.muted } },
    legend: { textStyle: { color: t.fgSecondary }, inactiveColor: t.grid, pageTextStyle: { color: t.muted } },
    tooltip: tooltipStyle(mode),
    categoryAxis: { ...axis, splitLine: { show: false, lineStyle: axis.splitLine.lineStyle } },
    valueAxis: axis,
    logAxis: axis,
    timeAxis: axis,
    line: { symbol: 'none', lineStyle: { width: 2 } },
    scatter: { symbolSize: 7 },
    visualMap: { textStyle: { color: t.muted } },
    dataZoom: {
      textStyle: { color: t.muted },
      borderColor: t.grid,
      fillerColor: mode === 'dark' ? 'rgba(47,129,247,0.15)' : 'rgba(9,105,218,0.12)',
      handleStyle: { color: t.surface, borderColor: t.muted },
    },
    markLine: { lineStyle: { color: t.muted }, label: { color: t.fgSecondary } },
    // ECharts describes a chart to a screen reader from its title, series and axes (a view's `ariaOf` text wins).
    aria: { enabled: true },
    animation: false,
  }
}

function tooltipStyle(mode: ColourMode): Record<string, unknown> {
  const t = tokens(mode)
  return {
    confine: true,
    backgroundColor: t.tooltip,
    borderColor: mode === 'dark' ? 'rgba(240,246,252,0.12)' : 'rgba(31,35,40,0.12)',
    textStyle: { color: t.fg, fontSize: 11, fontFamily: t.font },
    extraCssText: 'border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.10);',
  }
}

/**
 * The base of every chart option: transparent background, tab10, token-coloured text, tooltip and legend styling, and
 * no animation. Builders in `mdml-render/charts` start from it, so options look right even without the registered
 * theme (in Node, slides and SVG export).
 *
 * @param mode - The colour mode.
 * @param fontSize - The base font size in pixels (default 11).
 * @returns An ECharts option fragment.
 * @example
 * ```ts
 * import { baseOption } from 'mdml-render/theme'
 * baseOption('dark').animation // => false
 * ```
 */
export function baseOption(
  mode: ColourMode,
  fontSize = 11,
): {
  readonly backgroundColor: string
  readonly color: readonly string[]
  readonly animation: false
  readonly textStyle: Readonly<Record<string, unknown>>
  readonly tooltip: Readonly<Record<string, unknown>>
  readonly legend: Readonly<Record<string, unknown>>
} {
  const t = tokens(mode)
  return {
    backgroundColor: 'transparent',
    color: tab10,
    animation: false,
    textStyle: { fontFamily: t.font, color: t.fg, fontSize },
    tooltip: tooltipStyle(mode),
    legend: {
      show: false,
      top: 0,
      right: 0,
      type: 'plain',
      icon: 'roundRect',
      itemWidth: 12,
      itemHeight: 3,
      itemGap: 10,
      textStyle: { color: t.fgSecondary, fontSize, fontFamily: t.font },
    },
  }
}
