/**
 * Points the reader can drag on a figure, add by pressing empty space and remove by double click, each kept where its
 * constraint allows (in a box, on a curve, on a grid), with keyboard nudges.
 */

import type { Point2 } from 'omai-engine/numerics/geometry'
import { type KeyboardEvent, type ReactNode, type PointerEvent as ReactPointerEvent, useState } from 'react'
import type { Constraint } from './constraints.ts'
import { Backdrop, useOverlay } from './InteractionOverlay.tsx'

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
  /**
   * A press on empty space puts the point there (default false): it makes the first point when there is none and
   * otherwise moves it (the single-point tool: click a map to place the point, click again to move it). The point then
   * follows the pointer until release. `add` takes precedence while it may still make points. Handles and the other
   * tools of the overlay ({@link Probe}, {@link Brush}) are unaffected: the press goes to every tool.
   */
  readonly place?: boolean
  /** The most points `add` may make (default unlimited). */
  readonly max?: number
  /** A double click (or Delete key) on a point removes it (default false). */
  readonly remove?: boolean
  /** Handle radius in pixels (default 6); a square handle has this as its half side. */
  readonly radius?: number
  /** The handle's shape (default `circle`). */
  readonly shape?: 'circle' | 'square'
  /** Handle colour (default the accent token). */
  readonly colour?: string | ((index: number) => string)
  /** Accessible names of the handles (default "point 1", ...). */
  readonly labels?: readonly string[]
  /** Called as a drag starts and ends, with the point's index (and -1 at the end). */
  readonly onActive?: (index: number) => void
}

/**
 * Points the reader can drag (and, with `add`, place by pressing empty space, or with `place` put the one point there; with `remove`, delete by double click),
 * each kept where its constraint allows: in a box, on a curve, on a grid. The points are controlled: the overlay
 * calls `onChange` with the new list and draws what it is given, so every figure reading the same state follows. Each
 * handle is focusable; arrow keys nudge it by a few pixels (more with Shift).
 *
 * @param props - The points, what to call with new ones, and the rules.
 * @returns The handles, inside an {@link InteractionOverlay}.
 * @index function React component: points the reader drags, adds or removes: `<DraggablePoints points={pts} onChange={setPts} add />`
 */
export function DraggablePoints(props: DraggablePointsProps): ReactNode {
  const {
    points,
    onChange,
    add = false,
    place: putOnPress = false,
    remove = false,
    radius = 6,
    shape = 'circle',
    labels,
    onActive,
  } = props
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
  const catches = add || putOnPress
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
      {catches ? (
        <Backdrop
          cursor={canAdd || putOnPress ? 'copy' : 'default'}
          onPointerDown={(e) => {
            const at = dataOf(e)
            // Adding makes a point; placing moves the first (or makes it when there is none).
            if (!(canAdd || putOnPress) || !Number.isFinite(at[0]) || !Number.isFinite(at[1])) return
            const i = canAdd || points.length === 0 ? points.length : 0
            const next = [...points]
            next[i] = place(i, at)
            onChange(next)
            // The new point follows the pointer until release, as if it had been grabbed.
            const target = e.currentTarget
            target.setPointerCapture(e.pointerId)
            setActive(i)
            onActive?.(i)
            const follow = (ev: PointerEvent): void => {
              const to = dataOf(ev)
              if (!Number.isFinite(to[0]) || !Number.isFinite(to[1])) return
              const moved = [...next]
              moved[i] = place(i, to)
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
        const r = i === active ? radius + 2 : radius
        const handle = {
          role: 'button',
          tabIndex: 0,
          'aria-label': labels?.[i] ?? `point ${i + 1}`,
          fill: colourOf(i),
          stroke: 'var(--wb-bg, #fff)',
          strokeWidth: 2,
          style: { cursor: i === active ? 'grabbing' : 'grab', outline: 'none' },
          onPointerDown: (e: ReactPointerEvent<SVGElement>): void => {
            e.stopPropagation()
            e.currentTarget.setPointerCapture(e.pointerId)
            setActive(i)
            onActive?.(i)
          },
          onPointerMove: (e: ReactPointerEvent<SVGElement>): void => {
            if (e.currentTarget.hasPointerCapture(e.pointerId)) move(i, dataOf(e))
          },
          onPointerUp: (): void => {
            setActive(-1)
            onActive?.(-1)
          },
          onDoubleClick: (): void => {
            if (remove) onChange(points.filter((_, j) => j !== i))
          },
          onKeyDown: (e: KeyboardEvent<SVGElement>): void => nudge(i, e),
        } as const
        // biome-ignore lint/suspicious/noArrayIndexKey: points are identified by position in the list
        if (shape === 'square') return <rect key={i} x={cx - r} y={cy - r} width={2 * r} height={2 * r} {...handle} />
        // biome-ignore lint/suspicious/noArrayIndexKey: points are identified by position in the list
        return <circle key={i} cx={cx} cy={cy} r={r} {...handle} />
      })}
    </g>
  )
}
