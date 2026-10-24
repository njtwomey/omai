import type { EChartsOption } from 'echarts'
import { type ReactElement, type ReactNode, useMemo } from 'react'
import { type Figure, figure as makeFigure, scaleFigure } from '../../figure/figure.ts'
import { Chart, type ChartEventHandler } from '../Chart.tsx'
import { FigureFrame, useFrameSize } from '../FigureFrame.tsx'
import { useColourMode } from '../theme.tsx'

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
