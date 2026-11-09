/**
 * A rectangle the reader drags and resizes on a figure (a zoom window, a region of interest, a crop), bound two-way to
 * its x and y ranges in data coordinates: eight grips (four corners, four edges), the body to move it whole, snapping to
 * a step, bounds, a minimum size, arrow keys, and Escape to cancel a drag.
 */

import { type KeyboardEvent, type PointerEvent, type ReactNode, useEffect, useRef, useState } from 'react'
import { Handle, type HandleShape } from './Handle.tsx'
import { useOverlay } from './InteractionOverlay.tsx'
import type { DataRect } from './selection.ts'

/** Which grips a {@link DraggableRect} shows: the four `corners`, the four `edges`, or `all` eight (the default). */
export type RectHandles = 'corners' | 'edges' | 'all'

/** A number for both axes, or one per axis (an axis left out is not limited, snapped or kept apart). */
export type PerAxis = number | { readonly x?: number; readonly y?: number }

/** Props of {@link DraggableRect}. */
export interface DraggableRectProps {
  /** The x range `[x0, x1]` in data coordinates (given in either order: the rectangle is normalised, low end first). */
  readonly x: readonly [number, number]
  /** The y range `[y0, y1]` in data coordinates (either order). */
  readonly y: readonly [number, number]
  /**
   * Called with the new rectangle, low end first on both axes, while dragging, nudging or cancelling (bind it to the
   * same state as `x` and `y`).
   */
  readonly onChange: (rect: DataRect) => void
  /** The lowest the rectangle may reach, per axis: `{ x: 0, y: -1 }` (default unbounded). */
  readonly min?: { readonly x?: number; readonly y?: number }
  /** The highest the rectangle may reach, per axis (default unbounded). */
  readonly max?: { readonly x?: number; readonly y?: number }
  /** The smallest width and height in data units, a number for both or one per axis (default 0). An edge stops there. */
  readonly minSize?: PerAxis
  /**
   * Round the moved edges to this step in data units, a number for both axes or one per axis. It is also the arrow
   * keys' step (Shift moves ten steps); without it an arrow moves one pixel's worth of data.
   */
  readonly step?: PerAxis
  /** Which grips to show (default `all`: eight). */
  readonly handles?: RectHandles
  /** The grips' shape (default `square`; a `bar` lies along its edge on an edge grip and is a square on a corner). */
  readonly handleShape?: HandleShape
  /** The grips' size in pixels (default 8). */
  readonly handleSize?: number
  /** Whether dragging the body moves the whole rectangle (default true). */
  readonly moveable?: boolean
  /** The outline and grip colour: a CSS colour or `var(--wb-…)` token (default the foreground). */
  readonly colour?: string
  /** The fill of the body (default none: only the outline). */
  readonly fill?: string
  /** The opacity of the fill (default 0.12). */
  readonly fillOpacity?: number
  /** Names the rectangle for assistive technology (default "rectangle"). */
  readonly label?: string
}

type Range = readonly [lo: number, hi: number]

/** Where a grip sits on the rectangle: -1 the low end of an axis, 1 the high end, 0 the middle (the whole axis moves). */
interface Grip {
  readonly sx: -1 | 0 | 1
  readonly sy: -1 | 0 | 1
}

const CORNERS: readonly Grip[] = [
  { sx: -1, sy: -1 },
  { sx: 1, sy: -1 },
  { sx: -1, sy: 1 },
  { sx: 1, sy: 1 },
]
const EDGES: readonly Grip[] = [
  { sx: -1, sy: 0 },
  { sx: 1, sy: 0 },
  { sx: 0, sy: -1 },
  { sx: 0, sy: 1 },
]
const BODY: Grip = { sx: 0, sy: 0 }

interface AxisRules {
  readonly min: number
  readonly max: number
  readonly size: number
  readonly step: number | undefined
}

const tidy = (v: number): number => Number(v.toPrecision(12))

const snapped = (v: number, step: number | undefined): number =>
  step === undefined || !(step > 0) ? v : Math.round(v / step) * step

const perAxis = (v: PerAxis | undefined, axis: 'x' | 'y'): number | undefined => (typeof v === 'number' ? v : v?.[axis])

// One edge of a range moved by `delta`, snapped, kept inside the bounds and at least `size` from the other edge.
function moveEdge(range: Range, side: -1 | 1, delta: number, r: AxisRules): Range {
  if (side < 0) {
    const lo = Math.max(r.min, Math.min(snapped(range[0] + delta, r.step), range[1] - r.size))
    return [tidy(lo), range[1]]
  }
  const hi = Math.min(r.max, Math.max(snapped(range[1] + delta, r.step), range[0] + r.size))
  return [range[0], tidy(hi)]
}

// A whole range moved by `delta`, its size kept, snapped by its low end, and held inside the bounds.
function moveRange(range: Range, delta: number, r: AxisRules): Range {
  const size = range[1] - range[0]
  const lo = Math.max(r.min, Math.min(snapped(range[0] + delta, r.step), r.max - size))
  return [tidy(lo), tidy(lo + size)]
}

/** The rectangle with a grip dragged by a data-space `delta` from where it stood (a grip with both sides 0 moves the whole). */
function moved(rect: DataRect, grip: Grip, delta: readonly [number, number], rx: AxisRules, ry: AxisRules): DataRect {
  if (grip.sx === 0 && grip.sy === 0) return { x: moveRange(rect.x, delta[0], rx), y: moveRange(rect.y, delta[1], ry) }
  return {
    x: grip.sx === 0 ? rect.x : moveEdge(rect.x, grip.sx, delta[0], rx),
    y: grip.sy === 0 ? rect.y : moveEdge(rect.y, grip.sy, delta[1], ry),
  }
}

const same = (a: DataRect, b: DataRect): boolean =>
  a.x[0] === b.x[0] && a.x[1] === b.x[1] && a.y[0] === b.y[0] && a.y[1] === b.y[1]

/**
 * A rectangle on a figure that the reader drags and resizes, inside an `InteractionOverlay`: a zoom window, a region of
 * interest, a crop. Bind `x`, `y` and `onChange` to the same state as the controls (or the inset it frames) and the
 * rectangle and the controls move together. It shows a grip on each corner and edge (`handles`), and dragging the body
 * moves the whole rectangle (`moveable`). Edges snap to `step`, stay inside `min` and `max` and at least `minSize` apart,
 * and the rectangle stays normalised (low end first) whichever way the axes run on screen, so an inverted y axis or a
 * flipped x axis need nothing from the caller.
 *
 * Every grip is focusable and the arrow keys nudge it by `step` (Shift: ten steps), the way they look on screen (the
 * right arrow moves right even on a flipped axis). On a corner the arrows move both edges, on an edge the one it sits
 * on, on the body the whole rectangle. Escape during a drag puts the rectangle back as it was when the drag began.
 *
 * @param props - The ranges, the change handler, the bounds, snapping and the look of the grips.
 * @returns The rectangle, its grips and its hit area.
 * @index function React component: a rectangle dragged and resized across a figure: `<DraggableRect x={x} y={y} onChange={set} />`
 * @example
 * ```tsx
 * const [zoom, setZoom] = useState<DataRect>({ x: [2, 6], y: [0.2, 0.8] })
 * <InteractionOverlay projection={projection} width={w} height={h}>
 *   <DraggableRect
 *     x={zoom.x}
 *     y={zoom.y}
 *     onChange={setZoom}
 *     min={{ x: 0, y: 0 }}
 *     max={{ x: 10, y: 1 }}
 *     minSize={{ x: 0.5, y: 0.05 }}
 *     step={0.1}
 *     fill="var(--wb-accent)"
 *   />
 * </InteractionOverlay>
 * ```
 */
export function DraggableRect(props: DraggableRectProps): ReactNode {
  const { onChange, handles = 'all', moveable = true } = props
  const { projection, dataOf } = useOverlay()
  const colour = props.colour ?? 'var(--wb-fg)'
  const rect: DataRect = {
    x: [Math.min(props.x[0], props.x[1]), Math.max(props.x[0], props.x[1])],
    y: [Math.min(props.y[0], props.y[1]), Math.max(props.y[0], props.y[1])],
  }
  const rules = (axis: 'x' | 'y'): AxisRules => ({
    min: props.min?.[axis] ?? Number.NEGATIVE_INFINITY,
    max: props.max?.[axis] ?? Number.POSITIVE_INFINITY,
    size: perAxis(props.minSize, axis) ?? 0,
    step: perAxis(props.step, axis),
  })
  // What a drag started from: the grip, where the pointer was in data, and the rectangle, so Escape can put it back.
  const drag = useRef<{
    readonly grip: Grip
    readonly from: readonly [number, number]
    readonly start: DataRect
  } | null>(null)
  const [dragging, setDragging] = useState(false)
  const latest = useRef({ onChange, rect })
  latest.current = { onChange, rect }
  const stop = (): void => {
    drag.current = null
    setDragging(false)
  }
  // Escape cancels the drag from wherever the focus is.
  useEffect(() => {
    if (!dragging) return
    const onKey = (e: { readonly key: string; preventDefault?: () => void }): void => {
      if (e.key !== 'Escape' || drag.current === null) return
      e.preventDefault?.()
      latest.current.onChange(drag.current.start)
      drag.current = null
      setDragging(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [dragging])

  const [p0, p1] = [projection.toPixel([rect.x[0], rect.y[1]]), projection.toPixel([rect.x[1], rect.y[0]])]
  if (![...p0, ...p1].every(Number.isFinite)) return null
  const [left, top] = [Math.min(p0[0], p1[0]), Math.min(p0[1], p1[1])]
  const [pw, ph] = [Math.abs(p1[0] - p0[0]), Math.abs(p1[1] - p0[1])]
  const centre: readonly [number, number] = [left + pw / 2, top + ph / 2]

  const emit = (next: DataRect): void => {
    if (!same(next, rect)) onChange(next)
  }
  const grab = (grip: Grip, e: PointerEvent<SVGElement>): void => {
    e.currentTarget.setPointerCapture(e.pointerId)
    const d = dataOf(e)
    if (!d.every(Number.isFinite)) return
    drag.current = { grip, from: [d[0], d[1]], start: rect }
    setDragging(true)
  }
  const follow = (e: PointerEvent<SVGElement>): void => {
    const active = drag.current
    if (active === null) return
    const d = dataOf(e)
    if (!d.every(Number.isFinite)) return
    emit(moved(active.start, active.grip, [d[0] - active.from[0], d[1] - active.from[1]], rules('x'), rules('y')))
  }
  const pointer = (grip: Grip): object => ({
    onPointerDown: (e: PointerEvent<SVGElement>) => grab(grip, e),
    onPointerMove: follow,
    onPointerUp: stop,
    onPointerCancel: stop,
  })

  // The arrow keys move what is on screen the way they point: the direction of the data along each axis is read from
  // the projection, so a flipped x axis or an inverted y axis needs no care from the caller.
  const direction = (axis: 'x' | 'y'): number => {
    const a = projection.toPixel([rect.x[0], rect.y[0]])
    const b = projection.toPixel(axis === 'x' ? [rect.x[1], rect.y[0]] : [rect.x[0], rect.y[1]])
    return Math.sign(axis === 'x' ? b[0] - a[0] : -(b[1] - a[1])) || 1
  }
  // One pixel's worth of data along an axis, the arrow keys' step when no `step` is given.
  const perPixel = (axis: 'x' | 'y'): number => {
    const [ux, uy] = projection.unproject(centre[0], centre[1])
    const [vx, vy] = projection.unproject(centre[0] + (axis === 'x' ? 1 : 0), centre[1] + (axis === 'y' ? 1 : 0))
    return Math.abs(axis === 'x' ? vx - ux : vy - uy) || 1
  }
  const nudge = (grip: Grip, e: KeyboardEvent<SVGElement>): void => {
    const arrows: Record<string, readonly [number, number]> = {
      ArrowRight: [1, 0],
      ArrowLeft: [-1, 0],
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
    }
    const a = arrows[e.key]
    if (a === undefined) return
    e.preventDefault()
    const times = e.shiftKey ? 10 : 1
    const size = (axis: 'x' | 'y'): number => (perAxis(props.step, axis) ?? perPixel(axis)) * times
    emit(
      moved(rect, grip, [a[0] * direction('x') * size('x'), a[1] * direction('y') * size('y')], rules('x'), rules('y')),
    )
  }

  const grips = handles === 'corners' ? CORNERS : handles === 'edges' ? EDGES : [...CORNERS, ...EDGES]
  const at = (g: Grip): readonly [number, number] => {
    const gx = g.sx < 0 ? rect.x[0] : g.sx > 0 ? rect.x[1] : (rect.x[0] + rect.x[1]) / 2
    const gy = g.sy < 0 ? rect.y[0] : g.sy > 0 ? rect.y[1] : (rect.y[0] + rect.y[1]) / 2
    return projection.toPixel([gx, gy]) as readonly [number, number]
  }
  // The resize cursor for the direction of the grip on screen.
  const cursorOf = (p: readonly [number, number], g: Grip): string => {
    const [ox, oy] = [p[0] - centre[0], p[1] - centre[1]]
    if (g.sx !== 0 && g.sy !== 0) return ox * oy > 0 ? 'nwse-resize' : 'nesw-resize'
    return Math.abs(ox) > Math.abs(oy) ? 'ew-resize' : 'ns-resize'
  }
  // The grip's name as the reader sees it, from where it is on screen (a flipped axis swaps left and right).
  const nameOf = (g: Grip, p: readonly [number, number]): string => {
    const horizontal = g.sx === 0 ? '' : p[0] < centre[0] ? 'left' : 'right'
    const vertical = g.sy === 0 ? '' : p[1] < centre[1] ? 'top' : 'bottom'
    return [vertical, horizontal].filter((w) => w !== '').join(' ') + (g.sx !== 0 && g.sy !== 0 ? ' corner' : ' edge')
  }
  const sizeOf = props.handleSize ?? 8
  const valueText = `x ${rect.x[0]} to ${rect.x[1]}, y ${rect.y[0]} to ${rect.y[1]}`
  const label = props.label ?? 'rectangle'
  const lowEdge = (g: Grip): number => (g.sx !== 0 ? rect.x[g.sx < 0 ? 0 : 1] : rect.y[g.sy < 0 ? 0 : 1])
  const body = (
    <rect
      x={left}
      y={top}
      width={pw}
      height={ph}
      fill={props.fill ?? 'transparent'}
      fillOpacity={props.fill === undefined ? undefined : (props.fillOpacity ?? 0.12)}
      pointerEvents={moveable ? 'all' : 'none'}
    />
  )
  return (
    <g>
      {/* The body: a transparent or filled area that moves the rectangle whole, focusable when it moves. */}
      {moveable ? (
        <g
          role="slider"
          tabIndex={0}
          aria-label={`${label}, move`}
          aria-valuenow={rect.x[0]}
          aria-valuetext={valueText}
          data-rect-grip="body"
          style={{ cursor: dragging ? 'grabbing' : 'move', outline: 'none' }}
          onKeyDown={(e) => nudge(BODY, e)}
          {...pointer(BODY)}
        >
          {body}
        </g>
      ) : (
        body
      )}
      <rect
        x={left}
        y={top}
        width={pw}
        height={ph}
        fill="none"
        stroke={colour}
        strokeWidth={dragging ? 2.5 : 1.5}
        strokeDasharray={dragging ? undefined : '5 4'}
        pointerEvents="none"
      />
      {grips.map((g) => {
        const p = at(g)
        const edge = g.sx === 0 || g.sy === 0
        const name = nameOf(g, p)
        return (
          <g
            key={`${g.sx},${g.sy}`}
            transform={`translate(${p[0]}, ${p[1]})`}
            role="slider"
            tabIndex={0}
            aria-label={`${label}, ${name}`}
            aria-valuenow={lowEdge(g)}
            aria-valuetext={valueText}
            data-rect-grip={name}
            style={{ cursor: cursorOf(p, g), outline: 'none' }}
            onKeyDown={(e) => nudge(g, e)}
            {...pointer(g)}
          >
            {/* A larger transparent hit area, so a small grip is easy to take. */}
            <rect
              x={-Math.max(sizeOf, 14) / 2}
              y={-Math.max(sizeOf, 14) / 2}
              width={Math.max(sizeOf, 14)}
              height={Math.max(sizeOf, 14)}
              fill="transparent"
            />
            <Handle
              shape={props.handleShape === 'bar' && !edge ? 'square' : (props.handleShape ?? 'square')}
              size={sizeOf}
              vertical={g.sx !== 0}
              fill={dragging ? colour : 'var(--wb-bg)'}
              stroke={colour}
            />
          </g>
        )
      })}
    </g>
  )
}
