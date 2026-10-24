import { type ReactNode, useState } from 'react'
import { Backdrop, useOverlay } from './InteractionOverlay.tsx'
import type { Point2 } from './projection.ts'
import { type DataRect, rectBetween } from './selection.ts'

/**
 * Brush a rectangle: press and drag across the figure to select a region (`onChange` follows the drag), press without
 * dragging to clear it. With `axis`, the brush spans the whole height (`'x'`) or width (`'y'`). Pair with
 * `pointsInRect` to select data, or with linked figures to filter them.
 *
 * @param props - The selection, what to call with a new one, and the brushed axes.
 * @returns The brush, inside an {@link InteractionOverlay}.
 */
export function Brush(props: {
  readonly value: DataRect | null
  readonly onChange: (rect: DataRect | null) => void
  readonly axis?: 'x' | 'y' | 'xy'
}): ReactNode {
  const { value, onChange, axis = 'xy' } = props
  const { projection, width, height, dataOf } = useOverlay()
  const [start, setStart] = useState<Point2 | null>(null)
  const full = (rect: DataRect): DataRect => {
    if (axis === 'xy') return rect
    const [a, b] = [projection.toData([0, height]), projection.toData([width, 0])]
    return axis === 'x' ? { ...rect, y0: a[1], y1: b[1] } : { ...rect, x0: a[0], x1: b[0] }
  }
  let shape: ReactNode = null
  if (value !== null) {
    const [x0, y0] = projection.toPixel([value.x0, value.y1])
    const [x1, y1] = projection.toPixel([value.x1, value.y0])
    shape = (
      <rect
        x={Math.min(x0, x1)}
        y={Math.min(y0, y1)}
        width={Math.abs(x1 - x0)}
        height={Math.abs(y1 - y0)}
        fill="var(--wb-accent, #1f77b4)"
        fillOpacity={0.12}
        stroke="var(--wb-accent, #1f77b4)"
        strokeDasharray="4 3"
        pointerEvents="none"
      />
    )
  }
  return (
    <g>
      <Backdrop
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          setStart(dataOf(e))
          onChange(null)
        }}
        onPointerMove={(e) => {
          if (start !== null && e.currentTarget.hasPointerCapture(e.pointerId))
            onChange(full(rectBetween(start, dataOf(e))))
        }}
        onPointerUp={() => setStart(null)}
      />
      {shape}
    </g>
  )
}
