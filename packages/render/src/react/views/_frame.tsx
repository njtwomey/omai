import type { EChartsOption } from 'echarts'
import { type ReactElement, type ReactNode, useCallback, useEffect, useMemo, useRef } from 'react'
import type { Scene } from 'whitebox-engine/scene'
import { type Figure, figure as makeFigure, scaleFigure } from '../../figure/figure.ts'
import { tokens } from '../../theme/tokens.ts'
import type { TableData } from '../../trace/tables.ts'
import type { ViewFrames } from '../../views/_view.ts'
import { Chart, type ChartEventHandler } from '../Chart.tsx'
import { FigureFrame, useFrameSize } from '../FigureFrame.tsx'
import { Player } from '../Player.tsx'
import { SceneView } from '../SceneView.tsx'
import { useColourMode } from '../theme.tsx'
import { usePlayback } from '../usePlayback.ts'

/** Props every record view shares: the record, its figure and the frame's title, caption and file name. */
export interface RecordViewProps<R> {
  readonly record: R
  /** The figure (default the view's own size). */
  readonly figure?: Figure
  readonly title?: ReactNode
  readonly caption?: ReactNode
  /** The export file name (default the record's kind). */
  readonly filename?: string
}

/** The figure of a view: the one given, else the view's default, in the page's colour mode. */
export function useViewFigure(given: Figure | undefined, defaults: Parameters<typeof makeFigure>[0]): Figure {
  const mode = useColourMode()
  return useMemo(() => ({ ...(given ?? makeFigure(defaults)), mode }), [given, defaults, mode])
}

// The chart of a view, drawn at the frame's size (the modal's when expanded, scaled from the figure).
function ViewChart(props: {
  readonly figure: Figure
  readonly build: (fig: Figure) => EChartsOption
  readonly onEvents?: Readonly<Record<string, ChartEventHandler>>
}): ReactElement {
  const frame = useFrameSize()
  const { figure, build } = props
  const fig = frame?.expanded === true ? { ...scaleFigure(figure, frame.width), mode: figure.mode } : figure
  const option = useMemo(() => build(fig), [build, fig])
  return (
    <Chart
      option={option}
      width={fig.width}
      height={fig.height}
      mode={fig.mode}
      {...(props.onEvents === undefined ? {} : { onEvents: props.onEvents })}
    />
  )
}

/**
 * One record view as one figure: a fixed-size `FigureFrame` (one expand, one export) around a chart built for the
 * frame's size, and an optional footer beneath it (a player, a readout).
 *
 * @param props - The figure, the option builder (memoised by the caller), frame text, events and footer.
 * @returns The figure.
 */
export function RecordFigure(props: {
  readonly figure: Figure
  readonly build: (fig: Figure) => EChartsOption
  readonly filename: string
  readonly title?: ReactNode
  readonly caption?: ReactNode
  readonly onEvents?: Readonly<Record<string, ChartEventHandler>>
  readonly footer?: ReactNode
}): ReactElement {
  const { figure } = props
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: figure.width, maxWidth: '100%' }}>
      <FigureFrame
        figure={figure}
        filename={props.filename}
        {...(props.title === undefined ? {} : { title: props.title })}
        {...(props.caption === undefined ? {} : { caption: props.caption })}
      >
        <ViewChart
          figure={figure}
          build={props.build}
          {...(props.onEvents === undefined ? {} : { onEvents: props.onEvents })}
        />
      </FigureFrame>
      {props.footer}
    </div>
  )
}

/** Props of the players: the frames' figure and frame settings, the playback rate, and the frame to start on. */
export interface PlayerViewProps {
  /** Frames per second (default 10). */
  readonly fps?: number
  /** Start on the first frame rather than the last (default false: show the result, play from the start). */
  readonly startAtFirst?: boolean
}

// Frames from a builder of the figure, with a player beneath the figure. The view opens on its last frame (the
// result); playing from there starts again from the first.
export function FramesPlayer(props: {
  readonly figure: Figure
  readonly frames: (fig: Figure) => ViewFrames
  readonly filename: string
  readonly fps?: number
  readonly startAtFirst?: boolean
}): ReactElement {
  const { figure, frames } = props
  const count = frames(figure).count
  // By default a whole run plays in about ten seconds, at 2 to 30 frames per second.
  const playback = usePlayback(count, { fps: props.fps ?? 10 })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // A new record (new frames) opens on its result, or on its first frame when asked.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `frames` changes with the record, which resets the view
  useEffect(() => {
    seek.current(props.startAtFirst === true ? 0 : count - 1)
  }, [frames, count, props.startAtFirst])
  const frame = playback.frame
  const build = useCallback((fig: Figure) => frames(fig).option(frame), [frames, frame])
  const label = frames(figure).label
  return (
    <RecordFigure
      figure={figure}
      build={build}
      filename={props.filename}
      footer={count > 1 ? <Player controller={playback} stamp={label} /> : undefined}
    />
  )
}

/**
 * A scene as one figure: a fixed-size frame around the scene (scaled to fit, the modal's size when expanded), exported
 * with the rest of the frame.
 *
 * @param props - The figure, the scene, the export name, frame text and a node click handler.
 * @returns The figure.
 */
export function SceneFigure(props: {
  readonly figure: Figure
  readonly scene: Scene
  readonly filename: string
  readonly title?: ReactNode
  readonly caption?: ReactNode
  readonly onNodeClick?: (id: string) => void
  readonly footer?: ReactNode
}): ReactElement {
  const { figure } = props
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: figure.width, maxWidth: '100%' }}>
      <FigureFrame
        figure={figure}
        filename={props.filename}
        {...(props.title === undefined ? {} : { title: props.title })}
        {...(props.caption === undefined ? {} : { caption: props.caption })}
      >
        {(size) => (
          <div
            data-whitebox-svg=""
            style={{
              width: size.width,
              height: size.height,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <SceneView
              scene={props.scene}
              width={size.width}
              height={size.height}
              mode={figure.mode}
              {...(props.onNodeClick === undefined ? {} : { onNodeClick: (id: string) => props.onNodeClick?.(id) })}
            />
          </div>
        )}
      </FigureFrame>
      {props.footer}
    </div>
  )
}

/**
 * Table data as a compact, themed HTML table (beside a figure, not part of it).
 *
 * @param props - The table and an optional highlighted row.
 * @returns The table.
 */
export function DataTable(props: { readonly table: TableData; readonly highlight?: number }): ReactElement {
  const t = tokens(useColourMode())
  const cell = { padding: '3px 10px', textAlign: 'right' as const, borderBottom: `1px solid ${t.grid}` }
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <table style={{ borderCollapse: 'collapse', font: `12px ${t.monoFont}`, color: t.fg }}>
        <thead>
          <tr>
            {props.table.columns.map((c, j) => (
              <th
                key={c}
                style={{
                  ...cell,
                  color: t.fgSecondary,
                  fontWeight: 600,
                  textAlign: j === 0 ? 'left' : 'right',
                  fontFamily: t.font,
                }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.table.rows.map((r, i) => (
            <tr key={String(r[0] ?? i)} style={i === props.highlight ? { background: t.surface } : undefined}>
              {r.map((v, j) => (
                <td key={props.table.columns[j] ?? j} style={{ ...cell, textAlign: j === 0 ? 'left' : 'right' }}>
                  {v ?? ''}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
