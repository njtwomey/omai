/**
 * A probe tool: a crosshair that follows the pointer, drawing whatever its children make of the data point under it (a
 * readout, a marker on a curve, a slice through a field).
 */

import type { Point2 } from 'mdml-engine/numerics/geometry'
import { type ReactNode, useState } from 'react'
import { Backdrop, useOverlay } from './InteractionOverlay.tsx'

/**
 * Probe the figure: follow the pointer with a crosshair, and draw whatever `children` makes of the data point under it
 * (a readout, a marker on a curve, a slice through a field); `onProbe` hears the point too, or `null` as the pointer
 * leaves.
 *
 * @param props - What to draw at the probed point, and whether to draw the crosshair (default true).
 * @returns The probe, inside an {@link InteractionOverlay}.
 * @index function React component: a crosshair reporting the data point under the pointer: `<Probe onProbe={setPoint} />`
 */
export function Probe(props: {
  readonly children?: (point: Point2, pixel: Point2) => ReactNode
  readonly onProbe?: (point: Point2 | null) => void
  readonly crosshair?: boolean
}): ReactNode {
  const { children, onProbe, crosshair = true } = props
  const { projection, width, height, dataOf } = useOverlay()
  const [at, setAt] = useState<Point2 | null>(null)
  const pixel = at === null ? null : projection.toPixel(at)
  return (
    <g>
      <Backdrop
        onPointerMove={(e) => {
          const p = dataOf(e)
          setAt(p)
          onProbe?.(p)
        }}
        onPointerLeave={() => {
          setAt(null)
          onProbe?.(null)
        }}
      />
      {at === null || pixel === null ? null : (
        <g pointerEvents="none">
          {crosshair ? (
            <g stroke="var(--wb-muted, #888)" strokeWidth={1} strokeDasharray="3 3">
              <line x1={pixel[0]} x2={pixel[0]} y1={0} y2={height} />
              <line x1={0} x2={width} y1={pixel[1]} y2={pixel[1]} />
            </g>
          ) : null}
          {children?.(at, pixel)}
        </g>
      )}
    </g>
  )
}
