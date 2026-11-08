/**
 * A figure laid out on a grid specification: chart panels drawn as one subplot chart (shared and linked axes, zero
 * gaps, marginals) and text, TeX or any React content placed in the other cells, scaled to fit.
 */

import { type ReactElement, type ReactNode, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { type Panel, type SubplotOptions, subplotRects, subplots } from '../charts/compose.ts'
import { type Figure, figure as makeFigure, scaleFigure } from '../figure/figure.ts'
import type { GridSpec } from '../figure/grid.ts'
import { Chart } from './Chart.tsx'
import { useFrameSize } from './FigureFrame.tsx'
import { Tex } from './MathView.tsx'
import { useColourMode } from './theme.tsx'

// Scale content down (never up) so it fits its cell: typeset equations grow with their values and must not be clipped.
function FitToCell(props: { readonly children: ReactNode; readonly watch: unknown }): ReactElement {
  const outer = useRef<HTMLDivElement | null>(null)
  const inner = useRef<HTMLDivElement | null>(null)
  const [scale, setScale] = useState(1)
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure when the content (`watch`) changes
  useLayoutEffect(() => {
    const o = outer.current
    const i = inner.current
    if (o === null || i === null) return
    const measure = (): void => {
      const w = i.scrollWidth
      const h = i.scrollHeight
      const next = w > 0 && h > 0 ? Math.min(1, (o.clientWidth - 4) / w, (o.clientHeight - 4) / h) : 1
      setScale((s) => (Math.abs(s - next) < 1e-3 ? s : next))
    }
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(o)
    observer.observe(i)
    return () => observer.disconnect()
  }, [props.watch])
  return (
    <div
      ref={outer}
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      <div ref={inner} style={{ width: 'max-content', transform: `scale(${scale})`, transformOrigin: 'center' }}>
        {props.children}
      </div>
    </div>
  )
}

/** Props of {@link FigureGrid}. */
export interface FigureGridProps extends Omit<SubplotOptions, 'figure' | 'mode' | 'textCells'> {
  readonly grid: GridSpec
  /** Chart panels by cell id. */
  readonly panels: Readonly<Record<string, Panel>>
  /** React content for other cells by id (text, controls, readouts); `text` and `tex` cells show their content. */
  readonly cells?: Readonly<Record<string, ReactNode>>
  readonly figure?: Figure
}

/**
 * A figure laid out on a grid spec: chart panels drawn as one chart (`subplots`: shared and linked axes, zero gaps,
 * marginals), and text, TeX or any React content placed in the other cells. Inside an expanded `FigureFrame` it
 * scales to the modal.
 *
 * @param props - The grid, the panels, other cell content, the figure and subplot options.
 * @returns The figure.
 * @index function React component: charts and text on a grid: `<FigureGrid grid={g} panels={p} />`
 */
export function FigureGrid(props: FigureGridProps): ReactElement {
  const mode = useColourMode()
  const frame = useFrameSize()
  const { grid, panels, cells, title, legend, tooltip, link, zoom } = props
  const fw = frame?.width
  const fh = frame?.height
  const expanded = frame?.expanded === true
  const fig = useMemo(() => {
    const size = fw === undefined || fh === undefined ? {} : { width: fw, height: fh }
    const base = props.figure ?? makeFigure({ mode, ...size })
    return expanded && fw !== undefined ? scaleFigure(base, fw) : base
  }, [props.figure, mode, fw, fh, expanded])
  const option = useMemo(
    () =>
      subplots(grid, panels, {
        figure: { ...fig, mode },
        mode,
        textCells: false,
        ...(title === undefined ? {} : { title }),
        ...(legend === undefined ? {} : { legend }),
        ...(tooltip === undefined ? {} : { tooltip }),
        ...(link === undefined ? {} : { link }),
        ...(zoom === undefined ? {} : { zoom }),
      }),
    [grid, panels, fig, mode, title, legend, tooltip, link, zoom],
  )
  const rects = subplotRects(grid, panels, fig)
  const overlays = grid.cells.flatMap((c) => {
    const r = rects[c.id]
    const node: ReactNode =
      cells?.[c.id] ??
      (c.kind === 'tex' ? (
        <Tex display style={{ fontSize: '1.35em' }}>
          {c.content ?? ''}
        </Tex>
      ) : c.kind === 'text' ? (
        c.content
      ) : undefined)
    if (r === undefined || node === undefined || node === null || panels[c.id] !== undefined) return []
    return [
      <div
        key={c.id}
        data-omai-overlay=""
        {...(c.kind === 'tex' && cells?.[c.id] === undefined ? { 'data-tex': c.content ?? '' } : {})}
        style={{
          position: 'absolute',
          left: r.x,
          top: r.y,
          width: r.width,
          height: r.height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          overflow: 'hidden',
          fontSize: fig.fontSize + 2,
        }}
      >
        <FitToCell watch={c.kind === 'tex' || c.kind === 'text' ? c.content : node}>{node}</FitToCell>
      </div>,
    ]
  })
  return (
    <div style={{ position: 'relative', width: fig.width, height: fig.height, maxWidth: '100%' }}>
      <Chart option={option} width={fig.width} height={fig.height} mode={mode} />
      {overlays}
    </div>
  )
}
