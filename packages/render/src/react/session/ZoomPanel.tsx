/**
 * The zoom panel of a session: a picture of a grid (a fractal, a field, a density) that is panned and zoomed by the
 * wheel, by dragging and by pinching, and re-rendered for the new viewport by the method itself. The panel probes a
 * derived path with the viewport and a raster size as arguments and expects a `decisionSurface` with `values`; while
 * the gesture runs, the picture on screen is the last one drawn, moved and scaled to the new viewport, and once it has
 * stood still for a moment the probe is asked again. Beside the picture are a minimap of the whole domain with the
 * viewport marked, buttons (and keys) for zoom and reset, and a pixel inspector reading the value under the cursor.
 */

import type { DecisionSurface } from 'omai-engine/exposition'
import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import {
  cellAt,
  clampViewport,
  dataToPixel,
  gridRange,
  minimapRect,
  panViewport,
  pixelToData,
  sameViewport,
  surfaceRaster,
  type Viewport,
  zoomViewport,
} from '../../views/zoom.ts'
import { saveBlob } from '../_download.ts'
import { FigureChromeProvider, FigureFrame, type FrameSize } from '../FigureFrame.tsx'
import { useColourMode, useTokens } from '../theme.tsx'
import type { Panel } from './layout.ts'
import { ToolButton, useSettled } from './parts.tsx'
import { Note, usePanelFigure, useProbe } from './RowsPanel.tsx'
import type { LiveSession } from './useSession.ts'

type ZoomSpec = Extract<Panel, { kind: 'zoom' }>

// A decision surface with a grid of values, or nothing.
const surfaceOf = (value: unknown): DecisionSurface | undefined => {
  const s = value as DecisionSurface | undefined
  return s?.kind === 'decisionSurface' && s.values !== undefined ? s : undefined
}

// The raster of a surface on an offscreen canvas.
function rasterCanvas(
  surface: DecisionSurface,
  mode: Figure['mode'],
  range: ZoomSpec['range'],
): HTMLCanvasElement | undefined {
  if (typeof document === 'undefined') return undefined
  const pixels = surfaceRaster(surface, { mode, ...(range === undefined ? {} : { range }) })
  if (pixels.width === 0 || pixels.height === 0) return undefined
  const c = document.createElement('canvas')
  c.width = pixels.width
  c.height = pixels.height
  c.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(pixels.rgba), pixels.width, pixels.height), 0, 0)
  return c
}

// What the readout under the picture says.
function readout(surface: DecisionSurface | undefined, at: readonly [number, number] | undefined): string {
  if (at === undefined) return 'Move the pointer over the picture to read the value under it.'
  const cell = surface === undefined ? undefined : cellAt(surface, at[0], at[1])
  const names = surface?.features ?? ['x', 'y']
  const g = (v: number): string => v.toPrecision(6)
  return `${names[0]} = ${g(at[0])}, ${names[1]} = ${g(at[1])}, ${surface?.field?.name ?? 'value'} = ${cell === undefined ? 'outside' : g(cell.value)}`
}

/**
 * A zoomable picture of a probed grid with a minimap, zoom buttons and a pixel inspector. Pan by dragging, zoom by the
 * wheel or by pinching, reset by double-click or the button; the arrow keys pan and `+` and `-` zoom while the picture
 * has focus.
 *
 * @param props - The session, the viewed step, the panel and the frame's title and file name.
 * @returns The figure.
 */
export function ZoomPanel(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly live: LiveSession<any, any>
  readonly panel: ZoomSpec
  readonly t: number
  readonly title: string
  readonly filename: string
  /** Draw the figure's frame without its buttons (default true; the panel's provider or `PanelFigureProvider` may say). */
  readonly chrome?: boolean
}): ReactNode {
  const base = usePanelFigure()
  const { live, panel, t } = props
  const tk = useTokens()
  const mode = useColourMode()
  const domain = useMemo<Viewport>(() => panel.domain ?? panel.view, [panel.domain, panel.view])
  const [view, setView] = useState<Viewport>(panel.view)
  const settled = useSettled(view, 140)
  const [cols, rows] = panel.resolution ?? [220, 160]
  const args = {
    ...(panel.args as object | undefined),
    view: { x: [...settled.x], y: [...settled.y] },
    size: [cols, rows],
  }
  const probed = useProbe(live, t, panel.path, args as never)
  const overview = useProbe(live, t, panel.path, {
    ...(panel.args as object | undefined),
    view: { x: [...domain.x], y: [...domain.y] },
    size: [88, Math.max(8, Math.round((88 * (domain.y[1] - domain.y[0])) / (domain.x[1] - domain.x[0])))],
  } as never)
  // The picture on screen: the latest surface that arrived, with the viewport it was computed for.
  const last = useRef<{ surface: DecisionSurface; view: Viewport } | undefined>(undefined)
  const surface = surfaceOf(probed.value)
  if (surface !== undefined) last.current = { surface, view: settled }
  const shown = last.current
  const shownSurface = shown?.surface
  const overSurface = surfaceOf(overview.value)
  // One colour range for the picture and its minimap, from the overview when there is one.
  const range = useMemo(
    () => panel.range ?? (overSurface?.values === undefined ? undefined : gridRange(overSurface.values)),
    [panel.range, overSurface],
  )
  const picture = useMemo(
    () => (shownSurface === undefined ? undefined : rasterCanvas(shownSurface, mode, range)),
    [shownSurface, range, mode],
  )
  const mini = useMemo(
    () => (overSurface === undefined ? undefined : rasterCanvas(overSurface, mode, range)),
    [overSurface, range, mode],
  )
  const [pointer, setPointer] = useState<readonly [number, number] | undefined>(undefined)
  const problem =
    probed.error ??
    (surface === undefined && probed.value !== undefined
      ? 'the probe is not a decision surface with values'
      : undefined)
  return (
    <FigureChromeProvider chrome={props.chrome}>
      <FigureFrame
        figure={base}
        title={props.title}
        filename={props.filename}
        exportFormats={['png']}
        onExport={(_, frame) => {
          frame.querySelector('canvas')?.toBlob((b) => {
            if (b !== null) saveBlob(b, 'image/png', `${props.filename}.png`)
          })
        }}
        caption={
          <div className="flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-1">
              <ToolButton
                label="+"
                title="Zoom in"
                onClick={() => setView(zoomViewport(view, centre(view), 2, domain))}
              />
              <ToolButton
                label="−"
                title="Zoom out"
                onClick={() => setView(zoomViewport(view, centre(view), 0.5, domain))}
              />
              <ToolButton
                label="reset"
                title="Reset the viewport"
                disabled={sameViewport(view, panel.view)}
                onClick={() => setView(panel.view)}
              />
              <span className="ml-1 text-xs text-muted tabular-nums">
                {`zoom ${((domain.x[1] - domain.x[0]) / (view.x[1] - view.x[0])).toPrecision(3)}×`}
              </span>
            </div>
            <span className="text-xs text-(--wb-fg) tabular-nums" aria-live="polite" data-testid="inspector">
              {readout(shown?.surface, pointer)}
            </span>
            {problem === undefined ? null : <Note>unavailable: {problem}</Note>}
          </div>
        }
      >
        {(size) => (
          <Picture
            size={size}
            view={view}
            setView={setView}
            domain={domain}
            home={panel.view}
            shown={shown}
            picture={picture}
            mini={mini}
            border={tk.fg}
            bg={tk.bg}
            label={props.title}
            onPointer={setPointer}
          />
        )}
      </FigureFrame>
    </FigureChromeProvider>
  )
}

const centre = (v: Viewport): [number, number] => [(v.x[0] + v.x[1]) / 2, (v.y[0] + v.y[1]) / 2]

// The canvas, its gestures, the minimap and the cursor's crosshair.
function Picture(props: {
  readonly size: FrameSize
  readonly view: Viewport
  readonly setView: (updater: (v: Viewport) => Viewport) => void
  readonly domain: Viewport
  readonly home: Viewport
  readonly shown: { surface: DecisionSurface; view: Viewport } | undefined
  readonly picture: HTMLCanvasElement | undefined
  readonly mini: HTMLCanvasElement | undefined
  readonly border: string
  readonly bg: string
  readonly label: string
  readonly onPointer: (at: readonly [number, number] | undefined) => void
}): ReactNode {
  const { size, view, domain, shown, picture, mini } = props
  const W = size.width
  const H = size.height
  const canvas = useRef<HTMLCanvasElement>(null)
  const miniCanvas = useRef<HTMLCanvasElement>(null)
  const latest = useRef(props)
  latest.current = props
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const pinch = useRef<number | undefined>(undefined)
  const [crosshair, setCrosshair] = useState<readonly [number, number] | undefined>(undefined)

  // The last picture, moved and scaled from the viewport it was computed for to the one on screen.
  useEffect(() => {
    const ctx = canvas.current?.getContext('2d')
    if (ctx === null || ctx === undefined) return
    ctx.fillStyle = props.bg
    ctx.fillRect(0, 0, W, H)
    if (picture === undefined || shown === undefined) return
    const a = dataToPixel(view, [W, H], shown.view.x[0], shown.view.y[1])
    const b = dataToPixel(view, [W, H], shown.view.x[1], shown.view.y[0])
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(picture, a[0], a[1], b[0] - a[0], b[1] - a[1])
  }, [view, picture, shown, W, H, props.bg])
  useEffect(() => {
    const ctx = miniCanvas.current?.getContext('2d')
    if (ctx === null || ctx === undefined || mini === undefined) return
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(mini, 0, 0, mini.width, mini.height)
  }, [mini])

  const local = (e: { clientX: number; clientY: number }): [number, number] => {
    const r = canvas.current?.getBoundingClientRect()
    return r === undefined ? [0, 0] : [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H]
  }
  // The wheel must be a non-passive listener to stop the page from scrolling while the picture zooms.
  useEffect(() => {
    const el = canvas.current
    if (el === null) return
    const wheel = (e: WheelEvent): void => {
      e.preventDefault()
      const l = latest.current
      const r = el.getBoundingClientRect()
      const px = ((e.clientX - r.left) / r.width) * l.size.width
      const py = ((e.clientY - r.top) / r.height) * l.size.height
      const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015))
      l.setView((v) => zoomViewport(v, pixelToData(v, [l.size.width, l.size.height], px, py), factor, l.domain))
    }
    el.addEventListener('wheel', wheel, { passive: false })
    return () => el.removeEventListener('wheel', wheel)
  }, [])

  const move = (e: React.PointerEvent): void => {
    const [px, py] = local(e)
    const prev = pointers.current.get(e.pointerId)
    if (prev === undefined) {
      // Hovering: the inspector reads the value under the cursor.
      setCrosshair([px, py])
      props.onPointer(pixelToData(view, [W, H], px, py))
      return
    }
    const next = { x: px, y: py }
    pointers.current.set(e.pointerId, next)
    const all = [...pointers.current.values()]
    if (all.length === 1) {
      props.setView((v) => panViewport(v, -(px - prev.x) / W, (py - prev.y) / H, props.domain))
    } else if (all.length === 2) {
      const [p, q] = all as [{ x: number; y: number }, { x: number; y: number }]
      const d = Math.hypot(p.x - q.x, p.y - q.y)
      const before = pinch.current
      pinch.current = d
      if (before !== undefined && before > 0 && d > 0) {
        const mx = (p.x + q.x) / 2
        const my = (p.y + q.y) / 2
        props.setView((v) => zoomViewport(v, pixelToData(v, [W, H], mx, my), d / before, props.domain))
      }
    }
  }
  const key = (e: React.KeyboardEvent): void => {
    const pan: Record<string, [number, number]> = {
      ArrowLeft: [-0.1, 0],
      ArrowRight: [0.1, 0],
      ArrowUp: [0, 0.1],
      ArrowDown: [0, -0.1],
    }
    const p = pan[e.key]
    if (p !== undefined) props.setView((v) => panViewport(v, p[0], p[1], props.domain))
    else if (e.key === '+' || e.key === '=') props.setView((v) => zoomViewport(v, centre(v), 1.5, props.domain))
    else if (e.key === '-') props.setView((v) => zoomViewport(v, centre(v), 1 / 1.5, props.domain))
    else if (e.key === '0') props.setView(() => props.home)
    else return
    e.preventDefault()
  }
  const rect = minimapRect(clampViewport(view, domain), domain)
  const miniW = mini?.width ?? 88
  const miniH = mini?.height ?? 66
  return (
    <div
      role="application"
      // biome-ignore lint/a11y/noNoninteractiveTabindex: an application region takes focus so the keys pan and zoom
      tabIndex={0}
      aria-label={`${props.label}: drag to pan, scroll or pinch to zoom, arrow keys pan, plus and minus zoom, zero resets`}
      className="relative outline-none select-none focus-visible:ring-2 focus-visible:ring-(--wb-accent)/60"
      style={{ width: W, height: H }}
      onKeyDown={key}
    >
      <canvas
        ref={canvas}
        width={W}
        height={H}
        className="block cursor-grab touch-none active:cursor-grabbing"
        style={{ width: W, height: H }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          const [x, y] = local(e)
          pointers.current.set(e.pointerId, { x, y })
          pinch.current = undefined
        }}
        onPointerMove={move}
        onPointerUp={(e) => {
          pointers.current.delete(e.pointerId)
          pinch.current = undefined
        }}
        onPointerCancel={(e) => pointers.current.delete(e.pointerId)}
        onPointerLeave={() => {
          setCrosshair(undefined)
          props.onPointer(undefined)
        }}
        onDoubleClick={() => props.setView(() => props.home)}
      />
      {crosshair === undefined ? null : (
        <svg className="pointer-events-none absolute inset-0" width={W} height={H} aria-hidden="true">
          <line
            x1={crosshair[0]}
            x2={crosshair[0]}
            y1={0}
            y2={H}
            stroke={props.border}
            strokeOpacity={0.5}
            strokeDasharray="3 3"
          />
          <line
            y1={crosshair[1]}
            y2={crosshair[1]}
            x1={0}
            x2={W}
            stroke={props.border}
            strokeOpacity={0.5}
            strokeDasharray="3 3"
          />
        </svg>
      )}
      <div
        className="pointer-events-none absolute top-2 right-2"
        style={{ width: miniW, height: miniH, border: `1px solid ${props.border}`, background: props.bg }}
        role="img"
        aria-label={`Minimap: the viewport covers ${(rect.width * 100).toFixed(1)}% of the width of the domain`}
      >
        <canvas ref={miniCanvas} width={miniW} height={miniH} className="block" />
        <div
          className="absolute"
          style={{
            left: `${rect.left * 100}%`,
            top: `${rect.top * 100}%`,
            width: `${Math.max(rect.width * 100, 2)}%`,
            height: `${Math.max(rect.height * 100, 2)}%`,
            border: `2px solid ${props.border}`,
            boxShadow: `0 0 0 1px ${props.bg}`,
          }}
        />
      </div>
    </div>
  )
}
