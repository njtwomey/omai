/**
 * Colour-bar range selection for layers ECharts cannot recolour itself (a heatmap drawn as one image, matrix cells with
 * baked colours): the selection read from a visual map as soft value weights, colours outside it dimmed towards the
 * background, and the option update that redraws the selected layer.
 */

import type { Pixels } from 'whitebox-engine/scene'
import { pngDataUrl } from '../svg/png.ts'

/**
 * Turns an image's pixels into what an ECharts image element draws: a PNG data URL by default, or, in a browser, a
 * canvas (drawn without decoding, so recolouring while a handle is dragged does not flicker).
 */
export type ImageEncoder = (pixels: Pixels) => unknown

/**
 * How far a value is inside a colour bar's selection: 1 inside, 0 outside. Given a `spread` (how much the value changes
 * across a pixel of an image) the weight ramps linearly over it at the selection's edges, so a dimmed region's boundary
 * is anti-aliased rather than stepped.
 */
export type ValueFilter = (value: number, spread?: number) => number

/**
 * A layer's answer to a colour bar's range selection, for series ECharts cannot recolour itself (an image drawn once,
 * cells with their colours baked in): the fields to merge into one of its series so the values outside the selection
 * are dimmed and those inside keep their colours.
 */
export interface ColourSelect {
  /** The index, within the layer's series, of the series it recolours. */
  readonly series: number
  /**
   * The series fields to merge for a selection.
   *
   * @param keep - How far a value is selected (undefined: every value is).
   * @param encode - How to encode a repainted image.
   * @returns The fields (a new `renderItem`, say).
   */
  readonly recolour: (keep: ValueFilter | undefined, encode: ImageEncoder) => Record<string, unknown>
}

/** How much of its colour a value outside a colour bar's selection keeps (the rest is the background's). */
export const DIMMED_OPACITY = 0.15

/**
 * A colour dimmed by how far its value is outside a selection: blended towards the background, all the way to
 * {@link DIMMED_OPACITY} of its own colour when the weight is 0.
 *
 * @param rgb - The colour's red, green and blue (0 to 255).
 * @param background - The background's red, green and blue.
 * @param weight - How far the value is selected (0 to 1).
 * @returns The dimmed red, green and blue.
 * @example
 * ```ts
 * import { dimColour } from 'whitebox-render/charts'
 * dimColour([200, 100, 0], [0, 0, 0], 1) // => [200, 100, 0]
 * dimColour([200, 100, 0], [0, 0, 0], 0) // => [30, 15, 0]
 * ```
 */
export function dimColour(rgb: readonly number[], background: readonly number[], weight: number): number[] {
  const f = DIMMED_OPACITY + (1 - DIMMED_OPACITY) * Math.min(1, Math.max(0, weight))
  return [0, 1, 2].map((i) => {
    const bg = background[i] ?? 0
    return Math.round(bg + ((rgb[i] ?? 0) - bg) * f)
  })
}

/** The key under which a visual map in a chart option carries its layer's {@link ColourSelect} and series id. */
export const SELECT_KEY = '__whiteboxSelect'

interface Selectable {
  readonly series: string
  readonly recolour: ColourSelect['recolour']
}

interface Piece {
  readonly min?: number
  readonly max?: number
}

// The weight of a value in [lo, hi] (either end may be infinite), ramping over `spread` at the ends.
function inside(v: number, lo: number, hi: number, spread: number): number {
  if (!(spread > 0)) return v >= lo && v <= hi ? 1 : 0
  const ramp = (d: number): number => Math.min(1, Math.max(0, 0.5 + d / spread))
  return Math.min(ramp(v - lo), ramp(hi - v))
}

/**
 * The weights of a visual map's selection: between a continuous map's two handles, or in a piecewise map's selected
 * pieces (adjacent ones merged). An end of the selection at an end of the bar is open, so the values there (a field
 * clipped to the bar's range, say) stay selected. Undefined when every value is selected.
 *
 * @param visualMap - The visual map's option (`min` and `max`, or `pieces`).
 * @param selected - The selection from a `datarangeselected` event: `[low, high]`, or selection flags by piece index.
 * @returns The weights, or undefined for the whole range.
 * @example
 * ```ts
 * import { selectionFilter } from 'whitebox-render/charts'
 * const keep = selectionFilter({ min: 0, max: 10 }, [2, 5])
 * keep?.(3) // => 1
 * keep?.(7) // => 0
 * keep?.(5, 2) // => 0.5
 * selectionFilter({ min: 0, max: 10 }, [0, 10]) // => undefined
 * selectionFilter({ pieces: [{ min: 0, max: 1 }, { min: 1, max: 2 }] }, { 0: false, 1: true })?.(0.5) // => 0
 * ```
 */
export function selectionFilter(
  visualMap: Readonly<Record<string, unknown>>,
  selected: unknown,
): ValueFilter | undefined {
  let intervals: [number, number][]
  if (Array.isArray(selected)) {
    const [a, b] = selected as unknown[]
    if (typeof a !== 'number' || typeof b !== 'number') return undefined
    const min = typeof visualMap['min'] === 'number' ? visualMap['min'] : Number.NEGATIVE_INFINITY
    const max = typeof visualMap['max'] === 'number' ? visualMap['max'] : Number.POSITIVE_INFINITY
    // A millionth of the bar's length: a handle left at an end leaves that end open.
    const tol = Number.isFinite(max - min) ? 1e-6 * (max - min) : 0
    const lo = Math.min(a, b) <= min + tol ? Number.NEGATIVE_INFINITY : Math.min(a, b)
    const hi = Math.max(a, b) >= max - tol ? Number.POSITIVE_INFINITY : Math.max(a, b)
    intervals = [[lo, hi]]
  } else if (selected !== null && typeof selected === 'object') {
    const pieces = (visualMap['pieces'] ?? []) as readonly Piece[]
    const flags = selected as Readonly<Record<string, unknown>>
    const chosen = pieces.filter((_, i) => flags[String(i)] !== false)
    if (chosen.length === pieces.length) return undefined
    const bottom = Math.min(...pieces.map((p) => p.min ?? Number.NEGATIVE_INFINITY))
    const top = Math.max(...pieces.map((p) => p.max ?? Number.POSITIVE_INFINITY))
    intervals = []
    for (const p of [...chosen].sort((x, y) => (x.min ?? 0) - (y.min ?? 0))) {
      const lo = p.min === undefined || p.min <= bottom ? Number.NEGATIVE_INFINITY : p.min
      const hi = p.max === undefined || p.max >= top ? Number.POSITIVE_INFINITY : p.max
      const last = intervals[intervals.length - 1]
      if (last !== undefined && lo <= last[1]) last[1] = Math.max(last[1], hi)
      else intervals.push([lo, hi])
    }
  } else return undefined
  if (
    intervals.length === 1 &&
    intervals[0]?.[0] === Number.NEGATIVE_INFINITY &&
    intervals[0][1] === Number.POSITIVE_INFINITY
  )
    return undefined
  return (v, spread = 0) => intervals.reduce((w, [lo, hi]) => Math.max(w, inside(v, lo, hi, spread)), 0)
}

/**
 * The option update that answers a colour bar's range selection: the selected layer's series recoloured (values
 * outside the selection dimmed), to merge into the chart (`chart.setOption(update)`). The React `Chart` applies it on
 * every `datarangeselected` event, once per animation frame; undefined when the event's colour bar has no selectable
 * layer.
 *
 * @param option - The chart's option, as built by `cartesian` or `subplots`.
 * @param event - The `datarangeselected` event's parameters (`visualMapId` and `selected`).
 * @param encode - How to encode repainted images (default a PNG data URL).
 * @returns The update, or undefined.
 * @example
 * ```ts
 * import { colourSelection, heatmapChart } from 'whitebox-render/charts'
 * const option = heatmapChart([[0, 1], [2, 3]])
 * const id = option.visualMap[0].id
 * colourSelection(option, { visualMapId: id, selected: [1, 2] })?.series[0].id // => '__selectable-0'
 * colourSelection(option, { visualMapId: 'elsewhere', selected: [1, 2] }) // => undefined
 * ```
 */
export function colourSelection(
  option: unknown,
  event: unknown,
  encode: ImageEncoder = pngDataUrl,
):
  | { readonly series: readonly Record<string, unknown>[]; readonly visualMap?: readonly Record<string, unknown>[] }
  | undefined {
  if (option === null || typeof option !== 'object' || event === null || typeof event !== 'object') return undefined
  const { visualMapId, selected } = event as { readonly visualMapId?: unknown; readonly selected?: unknown }
  const maps = (option as { readonly visualMap?: unknown }).visualMap
  const list = (Array.isArray(maps) ? maps : maps === undefined ? [] : [maps]) as readonly Record<string, unknown>[]
  const map = list.find((m) => m['id'] !== undefined && m['id'] === visualMapId)
  const select = map?.[SELECT_KEY] as Selectable | undefined
  if (map === undefined || select === undefined) return undefined
  const series = [{ id: select.series, ...select.recolour(selectionFilter(map, selected), encode) }]
  // Merging a series makes ECharts re-read its visual maps, and a piecewise map's selection would reset: restate it.
  if (Array.isArray(selected) || selected === null || typeof selected !== 'object') return { series }
  return { series, visualMap: [{ id: map['id'], selected: { ...selected } }] }
}
