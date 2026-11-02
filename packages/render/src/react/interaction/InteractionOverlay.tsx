/**
 * The SVG layer laid exactly over a figure, in which interaction tools draw and read the pointer through a projection
 * between data and pixels; `Overlaid` stacks a figure and its overlay, and `Backdrop` catches presses that miss every
 * handle.
 */

import { createContext, type PointerEvent, type ReactNode, useContext, useMemo, useRef } from 'react'
import type { Point2, Projection } from './projection.ts'

/** What overlay tools read from their {@link InteractionOverlay}. */
export interface OverlayContext {
  readonly projection: Projection
  readonly width: number
  readonly height: number
  /** The pointer's position in pixels of the overlay (corrected for any CSS scaling of the figure). */
  readonly pixelOf: (e: { readonly clientX: number; readonly clientY: number }) => Point2
  /** The pointer's position in data coordinates. */
  readonly dataOf: (e: { readonly clientX: number; readonly clientY: number }) => Point2
}

const Context = createContext<OverlayContext | null>(null)

/**
 * The overlay a tool is inside.
 *
 * @returns The overlay's projection, size and pointer readers.
 * @throws Error outside an {@link InteractionOverlay}.
 */
export function useOverlay(): OverlayContext {
  const ctx = useContext(Context)
  if (ctx === null) throw new Error('interaction tools must sit inside an <InteractionOverlay>')
  return ctx
}

/**
 * An SVG layer laid exactly over a figure (it fills its positioned parent, as {@link Overlaid} provides), in which
 * interaction tools draw and read the pointer: {@link DraggablePoints}, {@link Brush}, {@link Probe}, or anything
 * using {@link useOverlay}. The projection maps the figure's data to the overlay's pixels; a figure with a fixed size
 * and limits gives a projection that never moves under the reader's hand.
 *
 * @param props - The projection, the figure's size in pixels and the tools.
 * @returns The overlay.
 * @index function React component: SVG layer over a figure for tools: `<InteractionOverlay projection={p} width={w} height={h}>...`
 */
export function InteractionOverlay(props: {
  readonly projection: Projection
  readonly width: number
  readonly height: number
  readonly children: ReactNode
  readonly className?: string
}): ReactNode {
  const { projection, width, height, children, className } = props
  const svg = useRef<SVGSVGElement>(null)
  const ctx = useMemo((): OverlayContext => {
    const pixelOf = (e: { readonly clientX: number; readonly clientY: number }): Point2 => {
      const r = svg.current?.getBoundingClientRect()
      if (r === undefined || r.width === 0 || r.height === 0) return [Number.NaN, Number.NaN]
      return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height]
    }
    return { projection, width, height, pixelOf, dataOf: (e) => projection.toData(pixelOf(e)) }
  }, [projection, width, height])
  return (
    <svg
      ref={svg}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        overflow: 'visible',
        touchAction: 'none',
      }}
      role="presentation"
    >
      <Context.Provider value={ctx}>{children}</Context.Provider>
    </svg>
  )
}

/**
 * A figure with an overlay on top: a box of the figure's size holding the figure and, over it, the overlay.
 *
 * @param props - The figure's size, the figure and the overlay.
 * @returns The stack.
 * @index function React component: stack a figure and its overlay: `<Overlaid width={w} height={h} overlay={tools}>{chart}</Overlaid>`
 */
export function Overlaid(props: {
  readonly width: number
  readonly height: number
  readonly children: ReactNode
  readonly overlay: ReactNode
  readonly className?: string
}): ReactNode {
  const { width, height, children, overlay, className } = props
  return (
    <div className={className} style={{ position: 'relative', width, height, maxWidth: '100%' }}>
      {children}
      {overlay}
    </div>
  )
}

/** The transparent rectangle a tool listens on for presses that miss every handle. */
export function Backdrop(props: {
  readonly onPointerDown?: (e: PointerEvent<SVGRectElement>) => void
  readonly onPointerMove?: (e: PointerEvent<SVGRectElement>) => void
  readonly onPointerUp?: (e: PointerEvent<SVGRectElement>) => void
  readonly onPointerLeave?: (e: PointerEvent<SVGRectElement>) => void
  readonly cursor?: string
}): ReactNode {
  const { width, height } = useOverlay()
  return (
    <rect
      x={0}
      y={0}
      width={width}
      height={height}
      fill="transparent"
      style={{ cursor: props.cursor ?? 'crosshair' }}
      onPointerDown={props.onPointerDown}
      onPointerMove={props.onPointerMove}
      onPointerUp={props.onPointerUp}
      onPointerLeave={props.onPointerLeave}
    />
  )
}
