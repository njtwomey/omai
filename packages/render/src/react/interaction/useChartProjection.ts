/**
 * A hook giving the projection of a live chart's axes for an interaction overlay laid over a `Chart`, set once the
 * chart exists and again whenever it is recreated (as on a change of colour mode).
 */

import { useCallback, useState } from 'react'
import { chartProjection, type PixelConverter, type Projection } from './projection.ts'

/**
 * The projection of a live chart's axes, for an {@link InteractionOverlay} laid over a `Chart`: pass `onInit` to the
 * chart, and the projection is set once the chart exists (and again if it is recreated, as on a change of colour mode).
 * Give the chart fixed axis limits so the projection does not move under the reader's hand.
 *
 * @param gridIndex - Which grid (set of axes) of the chart (default 0).
 * @returns The projection (`null` until the chart exists) and the chart's `onInit` handler.
 * @index function React hook: projection of a live Chart for overlays: `useChartProjection()`
 */
export function useChartProjection(gridIndex = 0): {
  readonly projection: Projection | null
  readonly onInit: (chart: PixelConverter) => void
} {
  const [projection, setProjection] = useState<Projection | null>(null)
  const onInit = useCallback((chart: PixelConverter) => setProjection(chartProjection(chart, gridIndex)), [gridIndex])
  return { projection, onInit }
}
