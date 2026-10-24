import {
  type CSSProperties,
  createContext,
  type ReactElement,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'
import type { Figure } from '../figure/figure.ts'
import { tokens } from '../theme/tokens.ts'
import { loadEcharts } from './echarts.ts'
import { useColourMode } from './theme.tsx'

/** The size a frame gives its content, and whether it is expanded in the modal. */
export interface FrameSize {
  readonly width: number
  readonly height: number
  readonly expanded: boolean
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
  /** Formats in the export menu (default png and svg; empty hides the menu). */
  readonly exportFormats?: readonly ExportFormat[]
  /**
   * Export handler. Without one, png and svg export the first chart in the frame (`Chart` from this package) and
   * download it as `filename`.
   */
  readonly onExport?: (format: ExportFormat, frame: HTMLElement) => void
  readonly filename?: string
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

function download(url: string, name: string): void {
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
}

/**
 * Export the first chart inside an element as PNG (twice the resolution) or SVG (re-rendered with ECharts' SVG
 * renderer when the chart draws on a canvas), as a data URL.
 *
 * @param root - The element containing the chart.
 * @param format - `png` or `svg`.
 * @param background - Background colour of the export.
 * @returns The data URL, or `undefined` when no chart is found.
 */
export function exportChartUrl(root: HTMLElement, format: 'png' | 'svg', background: string): string | undefined {
  const el = root.querySelector<HTMLElement>('[data-whitebox-chart]')
  if (el === null) return undefined
  const echarts = loadEcharts()
  const chart = echarts.getInstanceByDom(el)
  if (chart === undefined) return undefined
  if (format === 'png') return chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: background })
  const host = document.createElement('div')
  host.style.cssText = `position:fixed;left:-10000px;width:${el.clientWidth}px;height:${el.clientHeight}px`
  document.body.appendChild(host)
  try {
    const svg = echarts.init(host, undefined, { renderer: 'svg' })
    svg.setOption({ ...chart.getOption(), backgroundColor: background })
    const url = svg.getDataURL({ type: 'svg' })
    svg.dispose()
    return url
  } finally {
    host.remove()
  }
}

/**
 * A fixed-size figure area: content never resizes it. An expand button opens the same content full screen in a modal
 * (Escape or a click outside closes it), and an export menu saves the figure. Content that is a function receives the
 * current size, so a chart can draw at the modal's size; `useFrameSize` gives the same inside the frame.
 *
 * @param props - Size, title, caption, content, expansion and export settings.
 * @returns The frame.
 */
export function FigureFrame(props: FigureFrameProps): ReactElement {
  const mode = useColourMode()
  const t = tokens(mode)
  const width = props.width ?? props.figure?.width ?? 640
  const height = props.height ?? props.figure?.height ?? 400
  const aspect = width / height
  const [expanded, setExpanded] = useState(false)
  const [menu, setMenu] = useState(false)
  const [viewport, setViewport] = useState({ width: 1200, height: 800 })
  const frameRef = useRef<HTMLDivElement | null>(null)
  const modalRef = useRef<HTMLDivElement | null>(null)
  const formats = props.exportFormats ?? ['png', 'svg']

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

  const onExport = useCallback(
    (format: ExportFormat) => {
      setMenu(false)
      const root = (expanded ? modalRef.current : frameRef.current) ?? frameRef.current
      if (root === null) return
      if (props.onExport !== undefined) {
        props.onExport(format, root)
        return
      }
      if (format !== 'png' && format !== 'svg') return
      const url = exportChartUrl(root, format, t.bg)
      if (url !== undefined) download(url, `${props.filename ?? 'figure'}.${format}`)
    },
    [expanded, props.onExport, props.filename, t.bg],
  )

  // The modal keeps the figure's aspect, as large as fits in 92% of the viewport.
  const modalWidth = Math.min(viewport.width * 0.92, viewport.height * 0.86 * aspect)
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
    <div style={{ position: 'absolute', top: 6, right: 6, display: 'flex', gap: 4, zIndex: 2 }}>
      {formats.length > 0 && (
        <div style={{ position: 'relative' }}>
          <button type="button" style={button} aria-label="Export" title="Export" onClick={() => setMenu((m) => !m)}>
            <Icon path={ICON_DOWNLOAD} />
          </button>
          {menu && (
            <div
              role="menu"
              style={{
                position: 'absolute',
                right: 0,
                top: 32,
                minWidth: 96,
                background: t.tooltip,
                border: `1px solid ${t.grid}`,
                borderRadius: 6,
                padding: 4,
                boxShadow: '0 4px 14px rgba(0,0,0,0.15)',
              }}
            >
              {formats.map((f) => (
                <button
                  key={f}
                  type="button"
                  role="menuitem"
                  onClick={() => onExport(f)}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '4px 8px',
                    border: 'none',
                    background: 'transparent',
                    color: t.fg,
                    font: `12px ${t.font}`,
                    cursor: 'pointer',
                  }}
                >
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {(props.expandable ?? true) && (
        <button
          type="button"
          style={button}
          aria-label={inModal ? 'Close' : 'Expand'}
          title={inModal ? 'Close' : 'Expand'}
          onClick={() => {
            setMenu(false)
            setExpanded(!inModal)
          }}
        >
          <Icon path={inModal ? ICON_CLOSE : ICON_EXPAND} />
        </button>
      )}
    </div>
  )
  return (
    <figure className={props.className} style={{ margin: 0, width, maxWidth: '100%', ...props.style }}>
      {props.title !== undefined && (
        <div style={{ font: `600 13px ${t.font}`, color: t.fg, marginBottom: 6 }}>{props.title}</div>
      )}
      <div
        ref={frameRef}
        style={{ position: 'relative', width, height, maxWidth: '100%', overflow: 'hidden' }}
        data-whitebox-frame=""
      >
        {!expanded && content(pageSize)}
        {toolbar(false)}
      </div>
      {props.caption !== undefined && (
        <figcaption style={{ font: `12px ${t.font}`, color: t.muted, marginTop: 6 }}>{props.caption}</figcaption>
      )}
      {expanded && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setExpanded(false)
          }}
          onKeyDown={() => {}}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            background: mode === 'dark' ? 'rgba(1,4,9,0.8)' : 'rgba(31,35,40,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
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
            {content(modalSize)}
            {toolbar(true)}
          </div>
        </div>
      )}
    </figure>
  )
}
