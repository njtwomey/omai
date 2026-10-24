import { type KeyboardEvent, type ReactNode, useState } from 'react'
import type { Constraint } from './constraints.ts'
import { Backdrop, useOverlay } from './InteractionOverlay.tsx'
import type { Point2 } from './projection.ts'

/** Props of {@link DraggablePoints}. */
export interface DraggablePointsProps {
  readonly points: readonly Point2[]
  readonly onChange: (points: Point2[]) => void
  /** Where points may go (in a box, on a curve, on a grid). */
  readonly constrain?: Constraint
  /** A rule per point, by index, overriding `constrain` where it gives one. */
  readonly constrainEach?: (index: number) => Constraint | undefined
  /** A press on empty space adds a point there (default false). */
  readonly add?: boolean
  /** The most points `add` may make (default unlimited). */
  readonly max?: number
  /** A double click (or Delete key) on a point removes it (default false). */
  readonly remove?: boolean
  /** Handle radius in pixels (default 6). */
  readonly radius?: number
  /** Handle colour (default the accent token). */
  readonly colour?: string | ((index: number) => string)
  /** Accessible names of the handles (default "point 1", ...). */
  readonly labels?: readonly string[]
  /** Called as a drag starts and ends, with the point's index (and -1 at the end). */
  readonly onActive?: (index: number) => void
}

/**
 * Points the reader can drag (and, with `add`, place by pressing empty space; with `remove`, delete by double click),
 * each kept where its constraint allows: in a box, on a curve, on a grid. The points are controlled: the overlay
 * calls `onChange` with the new list and draws what it is given, so every figure reading the same state follows. Each
 * handle is focusable; arrow keys nudge it by a few pixels (more with Shift).
 *
 * @param props - The points, what to call with new ones, and the rules.
 * @returns The handles, inside an {@link InteractionOverlay}.
 */
export function DraggablePoints(props: DraggablePointsProps): ReactNode {
  const { points, onChange, add = false, remove = false, radius = 6, labels, onActive } = props
  const { projection, dataOf } = useOverlay()
  const [active, setActive] = useState(-1)
  const ruleFor = (i: number): Constraint | undefined => props.constrainEach?.(i) ?? props.constrain
  const place = (i: number, p: Point2): Point2 => {
    const rule = ruleFor(i)
    return rule === undefined ? p : rule(p)
  }
  const move = (i: number, p: Point2): void => {
    if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) return
    const next = [...points]
    next[i] = place(i, p)
    onChange(next)
  }
  const colourOf = (i: number): string =>
    typeof props.colour === 'function' ? props.colour(i) : (props.colour ?? 'var(--wb-accent, #1f77b4)')
  const canAdd = add && (props.max === undefined || points.length < props.max)
  const nudge = (i: number, e: KeyboardEvent): void => {
    const step = e.shiftKey ? 20 : 4
    const d: Record<string, Point2> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    }
    const delta = d[e.key]
    if (delta !== undefined) {
      e.preventDefault()
      const [px, py] = projection.toPixel(points[i] as Point2)
      move(i, projection.toData([px + delta[0], py + delta[1]]))
    } else if (remove && (e.key === 'Delete' || e.key === 'Backspace')) {
      e.preventDefault()
      onChange(points.filter((_, j) => j !== i))
    }
  }
  return (
    <g>
      {add ? (
        <Backdrop
          cursor={canAdd ? 'copy' : 'default'}
          onPointerDown={(e) => {
            if (!canAdd) return
            const i = points.length
            const next = [...points, place(i, dataOf(e))]
            onChange(next)
            // The new point follows the pointer until release, as if it had been grabbed.
            const target = e.currentTarget
            target.setPointerCapture(e.pointerId)
            setActive(i)
            onActive?.(i)
            const follow = (ev: PointerEvent): void => {
              const moved = [...next]
              moved[i] = place(i, dataOf(ev))
              onChange(moved)
              next[i] = moved[i] as Point2
            }
            const end = (): void => {
              target.removeEventListener('pointermove', follow)
              target.removeEventListener('pointerup', end)
              target.removeEventListener('pointercancel', end)
              setActive(-1)
              onActive?.(-1)
            }
            target.addEventListener('pointermove', follow)
            target.addEventListener('pointerup', end)
            target.addEventListener('pointercancel', end)
          }}
        />
      ) : null}
      {points.map((p, i) => {
        const [cx, cy] = projection.toPixel(p)
        if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null
        return (
          // biome-ignore lint/a11y/useSemanticElements: an SVG handle; there is no native draggable point
          <circle
            // biome-ignore lint/suspicious/noArrayIndexKey: points are identified by position in the list
            key={i}
            cx={cx}
            cy={cy}
            r={i === active ? radius + 2 : radius}
            role="button"
            tabIndex={0}
            aria-label={labels?.[i] ?? `point ${i + 1}`}
            fill={colourOf(i)}
            stroke="var(--wb-bg, #fff)"
            strokeWidth={2}
            style={{ cursor: i === active ? 'grabbing' : 'grab', outline: 'none' }}
            onPointerDown={(e) => {
              e.stopPropagation()
              e.currentTarget.setPointerCapture(e.pointerId)
              setActive(i)
              onActive?.(i)
            }}
            onPointerMove={(e) => {
              if (e.currentTarget.hasPointerCapture(e.pointerId)) move(i, dataOf(e))
            }}
            onPointerUp={() => {
              setActive(-1)
              onActive?.(-1)
            }}
            onDoubleClick={() => {
              if (remove) onChange(points.filter((_, j) => j !== i))
            }}
            onKeyDown={(e) => nudge(i, e)}
          />
        )
      })}
    </g>
  )
}
