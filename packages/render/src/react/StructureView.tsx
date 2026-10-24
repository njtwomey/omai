import { type ReactElement, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import type { Structure, StructureNode, StructureOverlay } from 'whitebox-engine/scene'
import { type DiagramOptions, diagram } from '../structures/diagram.ts'
import { cx, iconButtonClass } from './controls/classes.ts'
import { SceneView } from './SceneView.tsx'

/** Props of {@link StructureView}. */
export interface StructureViewProps extends Omit<DiagramOptions, 'overlay' | 'frames'> {
  readonly structure: Structure
  /** State laid over the diagram: active nodes and edges, badges (a frame of a message-passing animation, say). */
  readonly overlay?: StructureOverlay | undefined
  /** Every frame the overlay comes from, so nodes are sized once for all of them and the layout never jumps. */
  readonly frames?: readonly StructureOverlay[] | undefined
  /** Drawn width in pixels (default the diagram's own width, at most the container's). */
  readonly width?: number
  /** Called with a clicked node. */
  readonly onNodeClick?: (node: StructureNode) => void
  /** Show the hovered node's details under the diagram (default true). */
  readonly details?: boolean
  readonly className?: string
}

type View = readonly [number, number, number, number]

/**
 * A `Structure` drawn as a diagram (`diagram` in `whitebox-render/structures`), with pan (drag) and zoom (the buttons
 * under it, or the wheel with Ctrl or Cmd held), the hovered node's role, detail and badges shown beneath, and an
 * optional overlay of active nodes, edges and badges.
 *
 * @param props - The structure, overlay, layout options, width and handlers.
 * @returns The view.
 */
export function StructureView(props: StructureViewProps): ReactElement {
  const { structure, overlay, width, onNodeClick, details = true, className, frames, ...layout } = props
  const options: DiagramOptions = { ...layout, ...(overlay === undefined ? {} : { overlay }) }
  // biome-ignore lint/correctness/useExhaustiveDependencies: options are compared by content, frames by reference
  const drawing = useMemo(
    () => diagram(structure, frames === undefined ? options : { ...options, frames }),
    [structure, frames, JSON.stringify(options)],
  )
  const full: View = [0, 0, drawing.width, drawing.height]
  const [view, setView] = useState<View | null>(null)
  const shown = view ?? full
  const [hover, setHover] = useState<StructureNode | null>(null)
  const drag = useRef<{ x: number; y: number; view: View; moved: boolean } | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const byId = useMemo(() => new Map(structure.nodes.map((n) => [n.id, n])), [structure])
  const zoom = (factor: number, cxFrac = 0.5, cyFrac = 0.5): void => {
    const [x, y, w, h] = shown
    const nw = Math.min(Math.max(w * factor, drawing.width / 20), drawing.width * 4)
    const nh = (nw * h) / w
    setView([x + (w - nw) * cxFrac, y + (h - nh) * cyFrac, nw, nh])
  }
  const scale = (): number => {
    const r = box.current?.getBoundingClientRect()
    return r === undefined || r.width === 0 ? 1 : shown[2] / r.width
  }
  // Ctrl or Cmd with the wheel zooms about the pointer; a plain wheel scrolls the page. React's wheel handlers are
  // passive, so the listener is attached here to be allowed to prevent the page's zoom.
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom
  useEffect(() => {
    const el = box.current
    if (el === null) return
    const onWheel = (e: WheelEvent): void => {
      if (!(e.ctrlKey || e.metaKey)) return
      e.preventDefault()
      const r = el.getBoundingClientRect()
      zoomRef.current(e.deltaY > 0 ? 1.15 : 1 / 1.15, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])
  return (
    <div className={cx('flex w-fit max-w-full flex-col gap-2', className)}>
      <div
        ref={box}
        className="cursor-grab touch-none overflow-hidden rounded-lg border border-(--wb-grid) active:cursor-grabbing"
        style={{ width: width ?? drawing.width, maxWidth: '100%' }}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, view: shown, moved: false }
        }}
        onPointerMove={(e) => {
          const d = drag.current
          if (d === null || e.buttons === 0) return
          const dx = e.clientX - d.x
          const dy = e.clientY - d.y
          if (!d.moved && Math.hypot(dx, dy) < 4) return
          if (!d.moved) e.currentTarget.setPointerCapture(e.pointerId)
          d.moved = true
          const s = scale()
          setView([d.view[0] - dx * s, d.view[1] - dy * s, d.view[2], d.view[3]])
        }}
        onPointerUp={() => {
          // Leave the record until the click that follows, so a drag does not count as a click on a node.
          setTimeout(() => {
            drag.current = null
          }, 0)
        }}
      >
        <SceneView
          scene={drawing}
          width={width ?? drawing.width}
          viewBox={shown}
          style={{ width: '100%' }}
          onNodeHover={(id) => setHover(id === null ? null : (byId.get(id) ?? null))}
          {...(onNodeClick === undefined
            ? {}
            : {
                onNodeClick: (id: string) => {
                  const node = byId.get(id)
                  if (node !== undefined && drag.current?.moved !== true) onNodeClick(node)
                },
              })}
        />
      </div>
      {/* The hovered node's details and the zoom buttons share a row under the diagram, so neither covers it. */}
      <div className="flex min-w-0 items-center justify-between gap-3">
        {details ? <Details node={hover} overlay={overlay} /> : <span />}
        <div className="flex shrink-0">
          <button
            type="button"
            className={cx(iconButtonClass, 'size-7 rounded-l-md')}
            aria-label="Zoom in"
            onClick={() => zoom(1 / 1.3)}
          >
            +
          </button>
          <button
            type="button"
            className={cx(iconButtonClass, '-ml-px size-7')}
            aria-label="Zoom out"
            onClick={() => zoom(1.3)}
          >
            −
          </button>
          <button
            type="button"
            className={cx(iconButtonClass, '-ml-px h-7 w-auto rounded-r-md px-2 text-xs')}
            aria-label="Fit"
            onClick={() => setView(null)}
          >
            fit
          </button>
        </div>
      </div>
    </div>
  )
}

function Details(props: {
  readonly node: StructureNode | null
  readonly overlay: StructureOverlay | undefined
}): ReactNode {
  const { node, overlay } = props
  if (node === null)
    return <p className="m-0 h-5 min-w-0 flex-1 truncate text-xs text-(--wb-muted)">Hover a node for its details.</p>
  const badges = { ...node.badges, ...overlay?.badges?.[node.id] }
  return (
    <p className="m-0 h-5 min-w-0 flex-1 truncate text-xs text-(--wb-muted)">
      <span className="font-medium text-(--wb-fg)">{node.label}</span> · {node.role}
      {node.detail === undefined ? null : <> · {node.detail}</>}
      {Object.entries(badges).map(([k, v]) => (
        <span key={k}>
          {' '}
          · {k} <span className="font-mono text-(--wb-fg)">{String(v)}</span>
        </span>
      ))}
    </p>
  )
}
