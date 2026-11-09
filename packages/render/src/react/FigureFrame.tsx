/**
 * The fixed-size figure area every figure sits in: content never resizes it, an expand button opens it full screen in a
 * modal, and its menu exports the whole figure (charts, scenes, typeset overlays) as SVG or PNG, or records a playback
 * as GIF or WebM; a settings button (a modal applied only on confirm) can sit beside them. `useFrameSize` tells content the size it is drawn at.
 * The frame also holds the figure's text: a purpose line under the title, a row of controls, readouts and a caption,
 * and its title is a permalink (a link to `#id`) so a reader can copy the address of this exact figure.
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
import type { ColourMode } from '../theme/tokens.ts'
import { nextPaint, saveBlob, saveUrl } from './_download.ts'
import { prefersReducedMotion } from './_motion.ts'
import { useFocusTrap } from './controls/_focus.ts'
import { menuItemClass } from './controls/classes.ts'
import { Popover } from './controls/Popover.tsx'
import { loadEcharts } from './echarts.ts'
import { Controls, type ReadoutItem, Readouts } from './FigureParts.tsx'
import { useColourMode, useTokens } from './theme.tsx'

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

const ChromeContext = createContext<boolean | undefined>(undefined)

/**
 * Set whether the frames below it draw their toolbar (the expand, export and settings buttons), for every
 * {@link FigureFrame} in the subtree whose own `chrome` prop is not given: the default for a player's panels, a page's
 * figures. A frame's own `chrome` always wins. With `chrome` undefined it passes the setting of an enclosing provider
 * through (and the frames' own default, true, without one). Nested, the innermost given setting wins.
 *
 * @param props - The setting (`undefined` leaves the enclosing one) and the frames.
 * @returns The frames, with the setting in context.
 * @example
 * ```tsx
 * import { FigureChromeProvider, SessionPanels } from 'omai-render/react'
 * const bare = (
 *   <FigureChromeProvider chrome={false}>
 *     <SessionPanels live={live} at={at} layout={layout} />
 *   </FigureChromeProvider>
 * )
 * ```
 */
export function FigureChromeProvider(props: {
  readonly chrome: boolean | undefined
  readonly children?: ReactNode
}): ReactNode {
  return props.chrome === undefined ? (
    props.children
  ) : (
    <ChromeContext.Provider value={props.chrome}>{props.children}</ChromeContext.Provider>
  )
}

/**
 * Whether a figure draws its toolbar: its own setting when given, else the one a {@link FigureChromeProvider} above it
 * gives, else true. For a component that decides about its own buttons (a zoom control) the way {@link FigureFrame}
 * does about its toolbar.
 *
 * @param own - The component's own `chrome` prop, if any.
 * @returns Whether to draw the buttons.
 */
export function useFigureChrome(own?: boolean): boolean {
  const inherited = useContext(ChromeContext)
  return own ?? inherited ?? true
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
  /** A short line under the title saying what the figure is for. */
  readonly purpose?: ReactNode
  /**
   * The controls that drive the figure (`NumberField`, `Toggle`, `Select`, `Choice`), drawn in a wrapping {@link Controls}
   * row between the title and the figure.
   */
  readonly controls?: ReactNode
  /** Label and value pairs the figure reports, drawn as {@link Readouts} under the figure. */
  readonly readouts?: readonly ReadoutItem[]
  /** Text under the figure (and its readouts). */
  readonly caption?: ReactNode
  /**
   * Make the title a permalink (default true when there is a title): the title links to `#id`, a click sets the page's
   * hash, scrolls the frame into view and copies the full address to the clipboard (announced politely as "Copied"),
   * and a page opened at that hash scrolls to the frame and outlines it briefly (not animated for a reader who asked
   * for less motion). False turns all of it off, and the frame gets an `id` only if one is given.
   */
  readonly permalink?: boolean
  /**
   * The frame's `id`, the fragment of its permalink. Default: a slug of a string `title` (`"Loss over time"` becomes
   * `loss-over-time`), else one made by React that is stable within a page render but not between builds. Two frames
   * on a page need different ids; give an explicit one wherever a link has to last.
   */
  readonly id?: string
  /** The content: nodes, or a function of the frame size (to draw larger in the modal). */
  readonly children: ReactNode | ((size: FrameSize) => ReactNode)
  /**
   * Draw the figure bare (default true draws the toolbar): false leaves out the expand, export and settings buttons whatever
   * the other props say, and keeps the title and caption, for a figure placed in a slide or a page of its own. Not given,
   * it is the setting of an enclosing {@link FigureChromeProvider} (or `PanelFigureProvider`), else true.
   */
  readonly chrome?: boolean
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
const ICON_LINK = 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1'

/** A fragment-safe slug of a title: lower case words joined by hyphens (empty when nothing is left). */
function slugOf(title: string): string {
  return title
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/** The hash of the page without its `#`, decoded; empty when there is none or no page. */
function pageHash(): string {
  const raw = globalThis.location?.hash ?? ''
  const hash = raw.startsWith('#') ? raw.slice(1) : raw
  try {
    return decodeURIComponent(hash)
  } catch {
    return hash
  }
}

/** The address of the page with its hash set to `id`. */
function permalinkUrl(id: string): string {
  const loc = globalThis.location
  if (loc === undefined) return `#${id}`
  return `${loc.origin ?? ''}${loc.pathname ?? ''}${loc.search ?? ''}#${id}`
}

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
  const t = useTokens()
  const width = props.width ?? props.figure?.width ?? 640
  const height = props.height ?? props.figure?.height ?? 400
  const aspect = width / height
  const titleId = useId()
  const purposeId = useId()
  const fallbackId = `figure-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const permalink = (props.permalink ?? true) && props.title !== undefined
  const slug = typeof props.title === 'string' ? slugOf(props.title) : ''
  const frameId = props.id ?? (slug === '' ? fallbackId : slug)
  const figureRef = useRef<HTMLElement | null>(null)
  // Permalink state: the pointer or focus is on the title (its link icon shows), "Copied" is announced, the frame is outlined.
  const [linkHot, setLinkHot] = useState(false)
  const [copied, setCopied] = useState(false)
  const [flash, setFlash] = useState(false)
  // Read after mount, so a server-rendered frame and its hydration agree.
  const [reduced, setReduced] = useState(false)
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const flashTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
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
  const chrome = useFigureChrome(props.chrome)
  const formats = !chrome
    ? []
    : (props.exportFormats ??
      ([
        'png',
        'svg',
        ...(playback === undefined ? [] : ['gif', ...(canEncodeVideo() ? ['webm'] : [])]),
      ] as ExportFormat[]))
  const expandable = chrome && (props.expandable ?? true)
  const [busy, setBusy] = useState<string | null>(null)
  // Why the last export gave nothing, shown beside the busy readout until the next export starts.
  const [note, setNote] = useState<string | null>(null)
  const hasTools = formats.length > 0 || expandable || (chrome && props.settings !== undefined)

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

  // A page opened at this frame's hash scrolls to it and outlines it; so does a later change of the hash to it.
  useEffect(() => {
    if (!permalink) return
    setReduced(prefersReducedMotion())
    const mark = (): void => {
      setFlash(true)
      clearTimeout(flashTimer.current)
      flashTimer.current = setTimeout(() => setFlash(false), 1800)
    }
    if (pageHash() === frameId) {
      figureRef.current?.scrollIntoView?.({ block: 'start' })
      mark()
    }
    const onHash = (): void => {
      if (pageHash() === frameId) mark()
    }
    window.addEventListener('hashchange', onHash)
    return () => {
      window.removeEventListener('hashchange', onHash)
      clearTimeout(flashTimer.current)
    }
  }, [permalink, frameId])
  useEffect(() => () => clearTimeout(copiedTimer.current), [])

  const followLink = useCallback(
    (e: { preventDefault: () => void }) => {
      e.preventDefault()
      const loc = globalThis.location
      if (loc !== undefined) loc.hash = frameId
      figureRef.current?.scrollIntoView?.({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' })
      const clipboard = globalThis.navigator?.clipboard
      if (clipboard === undefined) return
      clipboard.writeText(permalinkUrl(frameId)).then(
        () => {
          setCopied(true)
          clearTimeout(copiedTimer.current)
          copiedTimer.current = setTimeout(() => setCopied(false), 1800)
        },
        () => undefined,
      )
    },
    [frameId],
  )

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
      {expandable && (
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
      ref={figureRef}
      className={props.className}
      style={{
        margin: 0,
        width,
        maxWidth: '100%',
        ...(permalink
          ? {
              scrollMarginTop: 16,
              borderRadius: 4,
              outline: `2px solid ${flash ? `var(--wb-accent, ${t.accent})` : 'transparent'}`,
              outlineOffset: 6,
              transition: reduced ? 'none' : 'outline-color 0.6s ease',
            }
          : {}),
        ...props.style,
      }}
      {...(permalink || props.id !== undefined ? { id: frameId } : {})}
      {...(props.title === undefined ? {} : { 'aria-labelledby': titleId })}
      {...(props.purpose === undefined ? {} : { 'aria-describedby': purposeId })}
    >
      {(props.title !== undefined || hasTools) && (
        // The title and the buttons sit above the figure area, so they never cover a legend or data.
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, minHeight: 28 }}>
          {props.title !== undefined && (
            <div id={titleId} style={{ font: `600 13px ${t.font}`, color: `var(--wb-fg, ${t.fg})` }}>
              {permalink ? (
                <a
                  href={`#${frameId}`}
                  title="Copy link to this figure"
                  onClick={followLink}
                  onPointerEnter={() => setLinkHot(true)}
                  onPointerLeave={() => setLinkHot(false)}
                  onFocus={() => setLinkHot(true)}
                  onBlur={() => setLinkHot(false)}
                  style={{
                    color: 'inherit',
                    textDecoration: 'none',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  {props.title}
                  <span
                    aria-hidden="true"
                    style={{
                      display: 'inline-flex',
                      opacity: linkHot || copied ? 0.7 : 0,
                      color: `var(--wb-muted, ${t.muted})`,
                    }}
                  >
                    <svg
                      width={14}
                      height={14}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                      aria-hidden="true"
                    >
                      <path d={ICON_LINK} strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </span>
                </a>
              ) : (
                props.title
              )}
            </div>
          )}
          {permalink && (
            // Mounted always, so a screen reader announces the text when it appears.
            <span
              role="status"
              aria-live="polite"
              style={{ font: `11px ${t.monoFont}`, color: `var(--wb-muted, ${t.muted})` }}
            >
              {copied ? 'Copied' : ''}
            </span>
          )}
          {hasTools && toolbar(false)}
        </div>
      )}
      {props.purpose !== undefined && (
        <p
          id={purposeId}
          style={{ margin: '0 0 8px', font: `12px/1.45 ${t.font}`, color: `var(--wb-muted, ${t.muted})` }}
        >
          {props.purpose}
        </p>
      )}
      {props.controls !== undefined && <Controls style={{ marginBottom: 8 }}>{props.controls}</Controls>}
      <div
        ref={frameRef}
        style={{ position: 'relative', width, height, maxWidth: '100%', overflow: 'hidden' }}
        data-omai-frame=""
      >
        {!expanded && content(pageSize)}
      </div>
      {props.readouts !== undefined && props.readouts.length > 0 && (
        <Readouts items={props.readouts} style={{ marginTop: 8 }} />
      )}
      {props.caption !== undefined && (
        <figcaption style={{ font: `12px ${t.font}`, color: `var(--wb-muted, ${t.muted})`, marginTop: 6 }}>
          {props.caption}
        </figcaption>
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
