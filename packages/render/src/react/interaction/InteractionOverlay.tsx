/**
 * The SVG layer laid exactly over a figure, in which interaction tools draw and read the pointer through a projection
 * between data and pixels; `Overlaid` stacks a figure and its overlay, and `Backdrop` is how a tool hears presses and
 * moves that miss every handle (the overlay keeps one catching rectangle under all the tools and passes each event to
 * every backdrop, so tools stack without blocking one another).
 */

import type { Point2 } from 'omai-engine/numerics/geometry'
import {
  createContext,
  type PointerEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import type { Projection } from './projection.ts'

/** What overlay tools read from their {@link InteractionOverlay}. */
export interface OverlayContext {
  readonly projection: Projection
  readonly width: number
  readonly height: number
  /** The pointer's position in pixels of the overlay (corrected for any CSS scaling of the figure). */
  readonly pixelOf: (e: { readonly clientX: number; readonly clientY: number }) => Point2
  /** The pointer's position in data coordinates. */
  readonly dataOf: (e: { readonly clientX: number; readonly clientY: number }) => Point2
  /** Register a tool's backdrop handlers with the overlay's catching rectangle; returns the way to remove them. */
  readonly register: (handlers: RefObject<BackdropHandlers>) => () => void
}

/** What a tool hears through the overlay's catching rectangle (see {@link Backdrop}). */
export interface BackdropHandlers {
  readonly onPointerDown?: (e: PointerEvent<SVGRectElement>) => void
  readonly onPointerMove?: (e: PointerEvent<SVGRectElement>) => void
  readonly onPointerUp?: (e: PointerEvent<SVGRectElement>) => void
  readonly onPointerLeave?: (e: PointerEvent<SVGRectElement>) => void
  /** The cursor over empty space (default crosshair); the last tool to give one wins. */
  readonly cursor?: string
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
  // The backdrops of the tools inside, in the order they registered (the order of the tools in the tree).
  const backdrops = useRef<RefObject<BackdropHandlers>[]>([])
  const [catching, setCatching] = useState(0)
  const register = useCallback((handlers: RefObject<BackdropHandlers>): (() => void) => {
    backdrops.current.push(handlers)
    setCatching(backdrops.current.length)
    return () => {
      backdrops.current = backdrops.current.filter((h) => h !== handlers)
      setCatching(backdrops.current.length)
    }
  }, [])
  const ctx = useMemo((): OverlayContext => {
    const pixelOf = (e: { readonly clientX: number; readonly clientY: number }): Point2 => {
      const r = svg.current?.getBoundingClientRect()
      if (r === undefined || r.width === 0 || r.height === 0) return [Number.NaN, Number.NaN]
      return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height]
    }
    return { projection, width, height, pixelOf, dataOf: (e) => projection.toData(pixelOf(e)), register }
  }, [projection, width, height, register])
  const pass =
    (name: Exclude<keyof BackdropHandlers, 'cursor'>) =>
    (e: PointerEvent<SVGRectElement>): void => {
      for (const h of [...backdrops.current]) h.current?.[name]?.(e)
    }
  const cursor = (): string =>
    [...backdrops.current].reverse().find((h) => h.current?.cursor !== undefined)?.current?.cursor ?? 'crosshair'
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
      {catching === 0 ? null : (
        <rect
          x={0}
          y={0}
          width={width}
          height={height}
          fill="transparent"
          style={{ cursor: cursor() }}
          onPointerDown={pass('onPointerDown')}
          onPointerMove={(e) => {
            e.currentTarget.style.cursor = cursor()
            pass('onPointerMove')(e)
          }}
          onPointerUp={pass('onPointerUp')}
          onPointerLeave={pass('onPointerLeave')}
        />
      )}
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

/**
 * How a tool hears the presses and moves that miss every handle. It draws nothing itself: the overlay keeps one
 * transparent rectangle under all its tools and passes each pointer event to the backdrops of every tool inside, so a
 * {@link Probe} and a {@link DraggablePoints} that places a point work together, and handles (drawn above it) always
 * win a press that lands on them. Without a backdrop the overlay catches nothing, so the chart below still gets hover.
 *
 * @param props - Handlers for pointer down, move, up and leave, and the cursor (default crosshair; the last tool to give one wins).
 * @returns Nothing drawn.
 */
export function Backdrop(props: BackdropHandlers): ReactNode {
  const { register } = useOverlay()
  const latest = useRef(props)
  latest.current = props
  useEffect(() => register(latest), [register])
  return null
}
