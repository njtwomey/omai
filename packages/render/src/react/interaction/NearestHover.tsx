/**
 * Hovering over data points: as the pointer moves, reports the index of the point nearest to it on screen within a
 * radius, or -1 when none is near.
 */

import type { Point2 } from 'omai-engine/numerics/geometry'
import type { ReactNode } from 'react'
import { Backdrop, useOverlay } from './InteractionOverlay.tsx'
import { nearestPoint } from './selection.ts'

/**
 * Hover over data points: as the pointer moves, `onHover` hears the index of the point nearest to it on screen (within
 * `radius` pixels), or -1 when none is near or the pointer leaves. Draws nothing itself, so the figures that share the
 * hovered index (through a `LinkStore`, say) mark it however they like.
 *
 * @param props - The points in data coordinates, the handler and the radius (default 12 pixels).
 * @returns The hover tool, inside an {@link InteractionOverlay}.
 */
export function NearestHover(props: {
  readonly points: readonly Point2[]
  readonly onHover: (index: number) => void
  readonly radius?: number
}): ReactNode {
  const { points, onHover, radius = 12 } = props
  const { projection, pixelOf } = useOverlay()
  return (
    <Backdrop
      cursor="default"
      onPointerMove={(e) => onHover(nearestPoint(points, pixelOf(e), projection, radius))}
      onPointerLeave={() => onHover(-1)}
    />
  )
}
