/**
 * The fixed-size figure area every figure sits in: content never resizes it, an expand button opens it full screen in a
 * modal, and its menu exports the whole figure (charts, scenes, typeset overlays) as SVG or PNG, or records a playback
 * as GIF or WebM; a settings button (a modal applied only on confirm) can sit beside them. `useFrameSize` tells content the size it is drawn at.
 */

import type { Pixels } from 'omai-engine/scene'
import {
  type CSSProperties,
  createContext,
  type ReactElement,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react'
import { encodeGif } from '../animation/gif.ts'
import { canEncodeVideo, encodeWebm } from '../animation/video.ts'
import type { Figure } from '../figure/figure.ts'
import { texToHtml } from '../maths/tex.ts'
import { chartToSvg, composeSvg, type FigurePart, svgDataUrl, svgToPixels, svgToPng } from '../svg/export.ts'
import { escapeXml } from '../svg/svg.ts'
import { type ColourMode, tokens } from '../theme/tokens.ts'
import { nextPaint, saveBlob, saveUrl } from './_download.ts'
import { useFocusTrap } from './controls/_focus.ts'
import { menuItemClass } from './controls/classes.ts'
import { Popover } from './controls/Popover.tsx'
import { loadEcharts } from './echarts.ts'
import { useColourMode } from './theme.tsx'

/** The size a frame gives its content, and whether it is expanded in the modal. */
export interface FrameSize {
  readonly width: number
  readonly height: number
  readonly expanded: boolean
  /**
   * In the modal, content that is not a function of the size is drawn at its page size and magnified by this factor
   * (a chart renders at a higher pixel ratio to stay sharp).
   */
  readonly scale?: number
}

const FrameContext = createContext<FrameSize | undefined>(undefined)

/**
 * The size of the enclosing {@link FigureFrame} (its fixed in-page size, or the modal's size while expanded), or
 * `undefined` outside a frame.
 *
 * @returns The frame size.
 */
export function useFrameSize(): FrameSize | undefined {
  return useContext(FrameContext)
}

/** What a frame needs of a playback to record it: the frame count and rate, the current frame, seek and pause. */
export interface FramePlayback {
  readonly count: number
  readonly fps: number
  readonly frame: number
  readonly seek: (frame: number) => void
  readonly pause: () => void
}

/** Export formats offered by a frame's menu. */
export type ExportFormat = 'png' | 'svg' | 'gif' | 'mp4' | 'webm' | 'json'

/** Props of {@link FigureFrame}. */
export interface FigureFrameProps {
  /** The figure giving the in-page size (default 640 by 400). */
  readonly figure?: Figure
  readonly width?: number
  readonly height?: number
  readonly title?: ReactNode
  readonly caption?: ReactNode
  /** The content: nodes, or a function of the frame size (to draw larger in the modal). */
  readonly children: ReactNode | ((size: FrameSize) => ReactNode)
  /** Offer the expand-to-modal button (default true). */
  readonly expandable?: boolean
  /**
   * A settings control for the figure, drawn in the toolbar before the export and expand buttons: a `SettingsButton`
   * or `SpaceSettings` with `appearance="toolbar"`, which opens a modal of settings applied only when it is confirmed.
   */
  readonly settings?: ReactNode
  /** Formats in the export menu (default png and svg; empty hides the menu). */
  readonly exportFormats?: readonly ExportFormat[]
  /**
   * Export handler. Without one, png and svg export the first chart in the frame (`Chart` from this package) and
   * download it as `filename`.
   */
  readonly onExport?: (format: ExportFormat, frame: HTMLElement) => void
  readonly filename?: string
  /**
   * The playback driving an animated figure (`usePlayback`'s controller): with it, the export menu also records the
   * figure frame by frame as a GIF and, where the browser can encode video, a WebM.
   */
  readonly playback?: FramePlayback
  readonly className?: string
  readonly style?: CSSProperties
}

const ICON_EXPAND = 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5'
const ICON_CLOSE = 'M6 6l12 12M18 6L6 18'
const ICON_DOWNLOAD = 'M12 4v11m0 0l-4-4m4 4l4-4M5 20h14'

function Icon(props: { readonly path: string }): ReactElement {
  return (
    <svg
      width={16}
      height={16}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path d={props.path} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

/** Every frame of a playing figure as pixels: seek, wait for the figure to draw, rasterise its SVG; then seek back. */
async function recordFrames(
  root: HTMLElement,
  playback: FramePlayback,
  options: { readonly background: string; readonly mode: ColourMode },
  progress: (done: number) => void,
): Promise<Pixels[]> {
  const start = playback.frame
  playback.pause()
  // The figure's own size (an expanded frame's fixed content is magnified; its frames are not).
  const width = root.offsetWidth
  const height = root.offsetHeight
  const frames: Pixels[] = []
  try {
    for (let i = 0; i < Math.min(playback.count, 600); i++) {
      playback.seek(i)
      await nextPaint(20)
      const svg = figureSvg(root, options)
      if (svg === undefined) break
      // Some browsers refuse to read back a canvas drawn from HTML in an SVG; then the frame goes without overlays.
      const pixels = await svgToPixels(svg, width, height, { background: options.background }).catch(() => {
        const plain = figureSvg(root, { ...options, overlays: false }) ?? svg
        return svgToPixels(plain, width, height, { background: options.background })
      })
      frames.push(pixels)
      progress(i + 1)
    }
  } finally {
    playback.seek(start)
  }
  return frames
}

/**
 * The whole figure inside an element as one SVG document: every chart in it (re-drawn by ECharts' SVG renderer at its
 * place), every scene (`SceneView`, or an inline SVG marked `data-omai-svg`) as drawn, and every overlay cell
 * (`data-omai-overlay`: TeX from `data-tex` as MathML, or its text), on the background colour. Panels are parts of
 * the figure, so they export together. A magnified element (an expanded frame's fixed content) exports at its own,
 * unmagnified size.
 *
 * @param root - The element holding the figure.
 * @param options - Background colour, colour mode, and whether to include HTML overlays (default true).
 * @returns The SVG text, or `undefined` when the element holds nothing to export.
 */
export function figureSvg(
  root: HTMLElement,
  options: { readonly background: string; readonly mode: ColourMode; readonly overlays?: boolean },
): string | undefined {
  const box = root.getBoundingClientRect()
  // A CSS-magnified element (an expanded frame's fixed content) is measured in its own, unmagnified pixels.
  const k = root.offsetWidth > 0 ? box.width / root.offsetWidth : 1
  const place = (r: DOMRect): { x: number; y: number; width: number; height: number } => ({
    x: (r.left - box.left) / k,
    y: (r.top - box.top) / k,
    width: r.width / k,
    height: r.height / k,
  })
  const parts: FigurePart[] = []
  const echarts = loadEcharts()
  for (const el of root.querySelectorAll<HTMLElement>('[data-omai-chart]')) {
    const chart = echarts.getInstanceByDom(el)
    if (chart === undefined) continue
    const at = place(el.getBoundingClientRect())
    const width = Math.round(at.width)
    const height = Math.round(at.height)
    parts.push({ ...at, width, height, svg: chartToSvg(chart.getOption(), { width, height, mode: options.mode }) })
  }
  // Inline SVG drawings (a SceneView in a `data-omai-svg` element) go in as they are drawn.
  for (const el of root.querySelectorAll<HTMLElement>('[data-omai-svg]')) {
    const svg = el.querySelector('svg')
    if (svg === null) continue
    parts.push({ ...place(svg.getBoundingClientRect()), svg: svg.outerHTML })
  }
  // Scenes (diagrams, worlds) are SVG already: each goes in as drawn, at its place.
  for (const el of root.querySelectorAll<SVGSVGElement>('svg[data-omai-scene]')) {
    if (el.closest('[data-omai-svg]') !== null) continue
    const at = place(el.getBoundingClientRect())
    if (at.width === 0 || at.height === 0) continue
    const copy = el.cloneNode(true) as SVGSVGElement
    copy.setAttribute('width', String(at.width))
    copy.setAttribute('height', String(at.height))
    copy.removeAttribute('style')
    parts.push({ ...at, svg: new XMLSerializer().serializeToString(copy) })
  }
  if (options.overlays ?? true)
    for (const el of root.querySelectorAll<HTMLElement>('[data-omai-overlay]')) {
      const at = place(el.getBoundingClientRect())
      const tex = el.dataset['tex']
      const colour = getComputedStyle(el).color
      const html =
        tex === undefined
          ? `<div style="color:${colour};font:13px system-ui,sans-serif">${escapeXml(el.textContent ?? '')}</div>`
          : `<div style="color:${colour};font-size:18px">${texToHtml(tex, { display: true, output: 'mathml' })}</div>`
      parts.push({ ...at, html })
    }
  if (parts.length === 0) return undefined
  return composeSvg(Math.round(box.width / k), Math.round(box.height / k), parts, { background: options.background })
}
/**
 * A fixed-size figure area: content never resizes it. The figure is the unit: a grid of panels (`subplots`,
 * `FigureGrid`) goes in one frame, with one expand button and one export for the whole figure, never one per panel.
 * The expand button opens the same content full screen in a modal (Escape or a click outside closes it), and the
 * export menu saves the whole figure (every chart, scene and overlay cell) as SVG or PNG, and, given the `playback`
 * that animates it, records it frame by frame as a GIF or (where the browser can encode video) a WebM. Content that is
 * a function receives the current size, so a chart can draw at the modal's size; other content is drawn at its page
 * size and magnified in the modal (charts at a matching pixel ratio, so they stay sharp). `useFrameSize` gives the size
 * inside the frame.
 *
 * @param props - Size, title, caption, content, expansion and export settings.
 * @returns The frame.
 * @index function React component: wrap a chart with expand and export: `<FigureFrame title="Loss"><Chart option={o} /></FigureFrame>`
 */
export function FigureFrame(props: FigureFrameProps): ReactElement {
  const mode = useColourMode()
  const t = tokens(mode)
  const width = props.width ?? props.figure?.width ?? 640
  const height = props.height ?? props.figure?.height ?? 400
  const aspect = width / height
  const titleId = useId()
  const [expanded, setExpanded] = useState(false)
  // The export menu is open from the page's toolbar or from the modal's (never both).
  const [menu, setMenu] = useState<'page' | 'modal' | null>(null)
  const [viewport, setViewport] = useState({ width: 1200, height: 800 })
  const frameRef = useRef<HTMLDivElement | null>(null)
  const modalRef = useRef<HTMLDivElement | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const expandRef = useRef<HTMLButtonElement | null>(null)
  const pageExportRef = useRef<HTMLButtonElement | null>(null)
  const modalExportRef = useRef<HTMLButtonElement | null>(null)
  const scaledRef = useRef<HTMLDivElement | null>(null)
  const playback = props.playback
  const formats =
    props.exportFormats ??
    ([
      'png',
      'svg',
      ...(playback === undefined ? [] : ['gif', ...(canEncodeVideo() ? ['webm'] : [])]),
    ] as ExportFormat[])
  const [busy, setBusy] = useState<string | null>(null)
  // Why the last export gave nothing, shown beside the busy readout until the next export starts.
  const [note, setNote] = useState<string | null>(null)
  const hasTools = formats.length > 0 || (props.expandable ?? true) || props.settings !== undefined

  useEffect(() => {
    if (!expanded) return
    const measure = (): void => setViewport({ width: window.innerWidth, height: window.innerHeight })
    measure()
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') setExpanded(false)
    }
    window.addEventListener('resize', measure)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('resize', measure)
      window.removeEventListener('keydown', onKey)
    }
  }, [expanded])

  // Focus stays in the expanded figure and goes back to its Expand button when it closes.
  useFocusTrap(expanded, dialogRef)
  const wasExpanded = useRef(false)
  useEffect(() => {
    if (wasExpanded.current && !expanded) expandRef.current?.focus()
    wasExpanded.current = expanded
  }, [expanded])

  const onExport = useCallback(
    (format: ExportFormat) => {
      setMenu(null)
      setNote(null)
      const failed = (e: unknown): void => setNote(`export failed: ${e instanceof Error ? e.message : String(e)}`)
      const root = (expanded ? (scaledRef.current ?? modalRef.current) : frameRef.current) ?? frameRef.current
      if (root === null) return
      if (props.onExport !== undefined) {
        props.onExport(format, root)
        return
      }
      const name = `${props.filename ?? 'figure'}.${format}`
      if ((format === 'gif' || format === 'webm') && playback !== undefined && busy === null) {
        const count = Math.min(playback.count, 600)
        setBusy(`recording 0 / ${count}`)
        recordFrames(root, playback, { background: t.bg, mode }, (done) => setBusy(`recording ${done} / ${count}`))
          .then(async (frames) => {
            setBusy('encoding')
            const bytes =
              format === 'gif'
                ? encodeGif(frames, { fps: playback.fps })
                : await encodeWebm(frames, { fps: playback.fps })
            saveBlob(bytes as Uint8Array<ArrayBuffer>, format === 'gif' ? 'image/gif' : 'video/webm', name)
          })
          .catch(failed)
          .finally(() => setBusy(null))
        return
      }
      if (format !== 'png' && format !== 'svg') {
        setNote(`export failed: ${format} needs an export handler or a playback`)
        return
      }
      const svg = figureSvg(root, { background: t.bg, mode })
      if (svg === undefined) {
        setNote('export failed: the figure holds nothing to export')
        return
      }
      if (format === 'svg') {
        saveUrl(svgDataUrl(svg), name)
        return
      }
      const [width, height] = [root.offsetWidth, root.offsetHeight]
      // Some browsers refuse to rasterise HTML in an SVG; then the PNG goes without the overlay cells.
      svgToPng(svg, width, height, { background: t.bg })
        .catch(() => {
          const plain = figureSvg(root, { background: t.bg, mode, overlays: false }) ?? svg
          return svgToPng(plain, width, height, { background: t.bg })
        })
        .then((url) => saveUrl(url, name))
        .catch(failed)
    },
    [expanded, props.onExport, props.filename, t.bg, mode, playback, busy],
  )

  // The modal keeps the figure's aspect, as large as fits in 92% of the viewport.
  const modalWidth = Math.min(viewport.width * 0.92, (viewport.height * 0.86 - 36) * aspect)
  const modalSize: FrameSize = {
    width: Math.round(modalWidth),
    height: Math.round(modalWidth / aspect),
    expanded: true,
  }
  const pageSize: FrameSize = { width, height, expanded: false }
  const content = (size: FrameSize): ReactNode => (
    <FrameContext.Provider value={size}>
      {typeof props.children === 'function' ? props.children(size) : props.children}
    </FrameContext.Provider>
  )
  const button: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 28,
    height: 28,
    border: `1px solid ${t.grid}`,
    borderRadius: 6,
    background: t.surface,
    color: t.fgSecondary,
    cursor: 'pointer',
    padding: 0,
  }
  const toolbar = (inModal: boolean): ReactElement => (
    <div style={{ display: 'flex', gap: 4, marginLeft: 'auto', zIndex: 2, alignItems: 'center' }}>
      {busy !== null && <span style={{ font: `11px ${t.monoFont}`, color: t.muted, marginRight: 4 }}>{busy}</span>}
      {note !== null && (
        <span role="alert" style={{ font: `11px ${t.monoFont}`, color: t.danger, marginRight: 4 }}>
          {note}
        </span>
      )}
      {props.settings}
      {formats.length > 0 && (
        <>
          <button
            ref={inModal ? modalExportRef : pageExportRef}
            type="button"
            style={{ ...button, opacity: busy === null ? 1 : 0.5 }}
            aria-label="Export"
            aria-haspopup="menu"
            aria-expanded={menu === (inModal ? 'modal' : 'page')}
            title="Export"
            disabled={busy !== null}
            onClick={() => setMenu((m) => (m === null ? (inModal ? 'modal' : 'page') : null))}
          >
            <Icon path={ICON_DOWNLOAD} />
          </button>
          <Popover
            anchor={inModal ? modalExportRef : pageExportRef}
            open={menu === (inModal ? 'modal' : 'page')}
            onClose={() => setMenu(null)}
            align="end"
            role="menu"
            label="Export"
            className="min-w-24 p-1"
          >
            {formats.map((f) => (
              <button key={f} type="button" role="menuitem" className={menuItemClass} onClick={() => onExport(f)}>
                {f.toUpperCase()}
              </button>
            ))}
          </Popover>
        </>
      )}
      {(props.expandable ?? true) && (
        <button
          ref={inModal ? undefined : expandRef}
          type="button"
          style={button}
          aria-label={inModal ? 'Close' : 'Expand'}
          title={inModal ? 'Close' : 'Expand'}
          onClick={() => {
            setMenu(null)
            setExpanded(!inModal)
          }}
        >
          <Icon path={inModal ? ICON_CLOSE : ICON_EXPAND} />
        </button>
      )}
    </div>
  )
  return (
    <figure
      className={props.className}
      style={{ margin: 0, width, maxWidth: '100%', ...props.style }}
      {...(props.title === undefined ? {} : { 'aria-labelledby': titleId })}
    >
      {(props.title !== undefined || hasTools) && (
        // The title and the buttons sit above the figure area, so they never cover a legend or data.
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, minHeight: 28 }}>
          {props.title !== undefined && (
            <div id={titleId} style={{ font: `600 13px ${t.font}`, color: t.fg }}>
              {props.title}
            </div>
          )}
          {hasTools && toolbar(false)}
        </div>
      )}
      <div
        ref={frameRef}
        style={{ position: 'relative', width, height, maxWidth: '100%', overflow: 'hidden' }}
        data-omai-frame=""
      >
        {!expanded && content(pageSize)}
      </div>
      {props.caption !== undefined && (
        <figcaption style={{ font: `12px ${t.font}`, color: t.muted, marginTop: 6 }}>{props.caption}</figcaption>
      )}
      {expanded && (
        // biome-ignore lint/a11y/useKeyWithClickEvents: Escape closes it (a window listener); the backdrop click is an extra
        <div
          ref={dialogRef}
          role="dialog"
          aria-modal="true"
          {...(props.title === undefined ? { 'aria-label': 'Expanded figure' } : { 'aria-labelledby': titleId })}
          tabIndex={-1}
          onClick={(e) => {
            if (e.target === e.currentTarget) setExpanded(false)
          }}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            outline: 'none',
            background: mode === 'dark' ? 'rgba(1,4,9,0.8)' : 'rgba(31,35,40,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex' }}>{toolbar(true)}</div>
            <div
              ref={modalRef}
              style={{
                position: 'relative',
                width: modalSize.width,
                height: modalSize.height,
                background: t.bg,
                borderRadius: 8,
                boxShadow: '0 12px 40px rgba(0,0,0,0.35)',
                overflow: 'hidden',
              }}
            >
              {typeof props.children === 'function' ? (
                content(modalSize)
              ) : (
                // Fixed content is drawn at its page size and magnified to fill the modal, overlays and all.
                <div
                  ref={scaledRef}
                  style={{
                    width,
                    height,
                    transform: `scale(${modalSize.width / width})`,
                    transformOrigin: 'top left',
                  }}
                >
                  {content({ width, height, expanded: true, scale: modalSize.width / width })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </figure>
  )
}
