/**
 * A hook giving the projection of a live chart's axes for an interaction overlay laid over a `Chart`, set once the
 * chart exists and again whenever it is recreated (as on a change of colour mode).
 */

import { useCallback, useReducer, useState } from 'react'
import { chartProjection, type PixelConverter, type Projection } from './projection.ts'

/**
 * The projection of a live chart's axes, for an {@link InteractionOverlay} laid over a `Chart`: pass `onInit` to the
 * chart, and the projection is set once the chart exists (and again if it is recreated, as on a change of colour mode).
 * Give the chart fixed axis limits so the projection does not move under the reader's hand; with numeric `min` and
 * `max` on its value axes and `outerBoundsMode: 'none'` on the grid (as omai's cartesian charts write them) the
 * projection is computed from the chart's option and ECharts is not asked at all; a hand-written grid (ECharts 6 merges
 * `outerBoundsMode: 'auto'` into it, which moves the plotting box) goes through ECharts' converters (see {@link chartProjection}). The host renders again when a redraw moves the plotting box or the
 * ranges, so handles follow a chart that is resized.
 *
 * @param gridIndex - Which grid (set of axes) of the chart (default 0).
 * @returns The projection (`null` until the chart exists; its `rect` is the plotting box in pixels, so a caller can
 * map data to pixels with the chart's own geometry) and the chart's `onInit` handler.
 * @index function React hook: projection of a live Chart for overlays: `useChartProjection()`
 */
export function useChartProjection(gridIndex = 0): {
  readonly projection: Projection | null
  readonly onInit: (chart: PixelConverter) => void
} {
  const [projection, setProjection] = useState<Projection | null>(null)
  // A redraw that moves the box or the ranges (a fluid chart resized) renders the host again, so handles follow it.
  const [, redraw] = useReducer((n: number) => n + 1, 0)
  const onInit = useCallback(
    (chart: PixelConverter) => setProjection(chartProjection(chart, gridIndex, redraw)),
    [gridIndex],
  )
  return { projection, onInit }
}
