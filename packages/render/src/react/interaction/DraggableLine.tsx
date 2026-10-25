import { type KeyboardEvent, type ReactNode, useState } from 'react'
import { useOverlay } from './InteractionOverlay.tsx'

/** Props of {@link DraggableLine}. */
export interface DraggableLineProps {
  /** `x`: a vertical line at x = value, dragged left and right; `y`: a horizontal line at y = value, dragged up and down. */
  readonly axis: 'x' | 'y'
  readonly value: number
  /** Called with the new value while dragging (bind it to the same state as the line's control). */
  readonly onChange: (value: number) => void
  /** The range of the other axis the line spans, in data units (default the whole overlay). */
  readonly span?: readonly [number, number]
  readonly min?: number
  readonly max?: number
  /** Round values to this step (also the arrow keys' step; Shift moves ten steps). */
  readonly step?: number
  /** Shown by the handle. */
  readonly label?: string
  /** A CSS colour or `var(--wb-…)` token (default the foreground). */
  readonly colour?: string
}

/**
 * A reference line on a figure that the reader drags, inside an `InteractionOverlay`: a threshold, a cut-off, a level.
 * Bind `value` and `onChange` to the same state as a control (a `NumberField`), and the line and the control move
 * together whichever one is used. It is focusable, and the arrow keys nudge it by `step`.
 *
 * @param props - The axis, value, change handler and bounds.
 * @returns The line, its handle and its hit area.
 * @example
 * ```tsx
 * <InteractionOverlay projection={projection} width={w} height={h}>
 *   <DraggableLine axis="x" value={threshold} onChange={setThreshold} span={[0, 40]} min={-3} max={5} step={0.05} />
 * </InteractionOverlay>
 * ```
 */
export function DraggableLine(props: DraggableLineProps): ReactNode {
  const { axis, value, onChange, span, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY } = props
  const { projection, width, height, dataOf } = useOverlay()
  const [dragging, setDragging] = useState(false)
  const colour = props.colour ?? 'var(--wb-fg)'
  const vertical = axis === 'x'
  const snap = (v: number): number => {
    const s = props.step
    const stepped = s === undefined || s <= 0 ? v : Math.round(v / s) * s
    return Math.min(max, Math.max(min, Number(stepped.toPrecision(12))))
  }
  // The line's ends in pixels: across the spanned range of the other axis, or the whole overlay.
  const at = (other: number): readonly [number, number] =>
    projection.toPixel(vertical ? [value, other] : [other, value]) as readonly [number, number]
  let a: readonly [number, number]
  let b: readonly [number, number]
  if (span === undefined) {
    const p = at(0)
    a = vertical ? [p[0], 0] : [0, p[1]]
    b = vertical ? [p[0], height] : [width, p[1]]
  } else {
    a = at(span[0])
    b = at(span[1])
  }
  if (![...a, ...b].every(Number.isFinite)) return null
  const top: readonly [number, number] = vertical ? (a[1] < b[1] ? a : b) : a[0] > b[0] ? a : b
  const move = (e: { readonly clientX: number; readonly clientY: number }): void => {
    const d = dataOf(e)
    const v = vertical ? d[0] : d[1]
    if (Number.isFinite(v)) onChange(snap(v))
  }
  const nudge = (e: KeyboardEvent<SVGGElement>): void => {
    const s = (props.step ?? (Number.isFinite(max - min) ? (max - min) / 100 : 0.1)) * (e.shiftKey ? 10 : 1)
    const keys = vertical ? { ArrowLeft: -1, ArrowRight: 1 } : { ArrowDown: -1, ArrowUp: 1 }
    const dir = (keys as Record<string, number>)[e.key]
    if (dir === undefined) return
    e.preventDefault()
    onChange(snap(value + dir * s))
  }
  return (
    // biome-ignore lint/a11y/useSemanticElements: an SVG group cannot be an <input type="range">; it carries the slider role.
    <g
      role="slider"
      tabIndex={0}
      aria-label={props.label ?? (vertical ? 'vertical line' : 'horizontal line')}
      aria-valuenow={value}
      aria-valuemin={Number.isFinite(min) ? min : undefined}
      aria-valuemax={Number.isFinite(max) ? max : undefined}
      aria-orientation={vertical ? 'horizontal' : 'vertical'}
      onKeyDown={nudge}
      style={{ cursor: vertical ? 'ew-resize' : 'ns-resize', outline: 'none' }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        setDragging(true)
        move(e)
      }}
      onPointerMove={(e) => {
        if (dragging) move(e)
      }}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
    >
      {/* A wide transparent hit area, so the line is easy to grab. */}
      <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="transparent" strokeWidth={14} />
      <line
        x1={a[0]}
        y1={a[1]}
        x2={b[0]}
        y2={b[1]}
        stroke={colour}
        strokeWidth={dragging ? 2.5 : 1.5}
        strokeDasharray={dragging ? undefined : '5 4'}
      />
      <g transform={`translate(${top[0]}, ${top[1]})`}>
        <rect
          x={vertical ? -5 : -10}
          y={vertical ? -10 : -5}
          width={vertical ? 10 : 20}
          height={vertical ? 20 : 10}
          rx={3}
          fill="var(--wb-bg)"
          stroke={colour}
          strokeWidth={1.5}
        />
        {props.label !== undefined && (
          <text
            x={vertical ? 8 : -14}
            y={vertical ? 24 : -8}
            fontSize={11}
            fill={colour}
            textAnchor={vertical ? 'start' : 'end'}
            style={{ pointerEvents: 'none', userSelect: 'none' }}
          >
            {props.label}
          </text>
        )}
      </g>
    </g>
  )
}
