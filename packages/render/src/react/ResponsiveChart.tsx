/**
 * A chart that fills its container: `ResponsiveChart` measures the box it is in, builds the option for that width with
 * a function of a `Figure`, rebuilds it when the width changes, switches to a taller shape on a narrow screen, can keep
 * the plotting area square, and carries an interaction overlay over the fluid chart.
 */

import type { EChartsCoreOption, ECharts as EChartsInstance } from 'echarts/core'
import {
  type CSSProperties,
  type ReactElement,
  type ReactNode,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { type Figure, figure, squarePlot } from '../figure/figure.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { Chart, type ChartProps } from './Chart.tsx'
import { useFrameSize } from './FigureFrame.tsx'
import { InteractionOverlay, Overlaid } from './interaction/InteractionOverlay.tsx'
import type { PixelConverter, Projection } from './interaction/projection.ts'
import { useChartProjection } from './interaction/useChartProjection.ts'
import { useColourMode } from './theme.tsx'

/** What a {@link ResponsiveChart} overlay receives: the chart's projection and the size it is drawn at. */
export interface ResponsiveOverlayContext {
  /** The map between data and the chart's pixels (see `chartProjection`). */
  readonly projection: Projection
  /** The figure's size in pixels, which is the size of the overlay too. */
  readonly size: { readonly width: number; readonly height: number }
}

/** Props of {@link ResponsiveChart}: those of {@link Chart} except the option and the size, which it makes. */
export interface ResponsiveChartProps
  extends Omit<ChartProps, 'option' | 'figure' | 'width' | 'height' | 'mode' | 'className' | 'style'> {
  /**
   * The option for a figure of the measured width. It is called again when the width changes by a pixel or more, or
   * when its own identity changes (wrap it in `useCallback` over what it reads, or it is rebuilt on every render).
   */
  readonly build: (fig: Figure) => EChartsCoreOption
  /** Width over height (default the page figure's 1.6), or a ratio such as `'16:9'`. */
  readonly aspect?: number | `${number}:${number}`
  /** The aspect used below `narrowBelow` pixels of width (default: the same as `aspect`, so the shape does not change). */
  readonly narrowAspect?: number | `${number}:${number}`
  /** The width in pixels under which `narrowAspect` applies (default 520). */
  readonly narrowBelow?: number
  /** Base font size of the figure in pixels (default the page figure's). */
  readonly fontSize?: number
  /**
   * Make the plotting area square (the figure is as tall as it is wide, less the padding) for surfaces whose units must
   * be equal, with `aspect: 'equal'` in the chart. It overrides `aspect` and `narrowAspect`.
   */
  readonly square?: boolean
  /** Colour mode (default the {@link useColourMode} mode). */
  readonly mode?: ColourMode
  /**
   * Tools drawn over the chart in an `InteractionOverlay`: nodes that read the overlay (`DraggableLine`,
   * `DraggablePoints`, `Brush`...), or a function of the projection and the size. Drawn once the chart exists.
   */
  readonly overlay?: ReactNode | ((context: ResponsiveOverlayContext) => ReactNode)
  /** Class of the box the chart fills. */
  readonly className?: string
  /** Style of the box the chart fills (it is `width: 100%` and takes its height from the chart). */
  readonly style?: CSSProperties
}

/** The width of a box that a chart can be built for: positive and finite, else undefined (hidden, not laid out yet). */
function usable(width: number | undefined): number | undefined {
  return width !== undefined && Number.isFinite(width) && width > 0 ? Math.round(width) : undefined
}

/**
 * A chart that fills the box it is in and is built for that box's width. A `Chart` has a fixed size from a `Figure`;
 * here the figure is made from the width the container has, as `build(figure({ width, aspect, fontSize, mode }))`, so
 * padding, ticks and the aspect are laid out for the size that is drawn.
 *
 * - The container is measured with a `ResizeObserver` (and once before the first paint). An empty size, a first
 *   measurement of 0 by 0 or a box hidden later (`display: none`), is ignored: the chart keeps the last size it had.
 * - The option is rebuilt only when the measured width changes by a pixel or more, not on every observer tick; the
 *   height follows from the aspect, which is `narrowAspect` below `narrowBelow` pixels and `aspect` above.
 * - `square` makes the plotting area square, for equal-unit surfaces.
 * - `overlay` draws interaction tools over the chart; the projection is read from the chart's option, so a drag
 *   reads it without asking ECharts, and the tools follow the chart when it is resized.
 * - Inside a {@link FigureFrame} the aspect is the frame's own unless `aspect` is given, and the container fills the
 *   frame, so the chart follows a frame that shrinks on a narrow page. Expanded, it takes the size the frame gives its
 *   content: the modal's width and height when the frame's children are a function of the size (`{(size) => <ResponsiveChart
 *   ... />}`, drawn afresh at the larger size), else the page size, which the frame magnifies. `square` and the
 *   narrow switch do not apply to a frame's fixed size.
 *
 * @param props - The builder, the shape (`aspect`, `narrowAspect`, `narrowBelow`, `square`), the type size, the
 *   overlay, and what a `Chart` takes (renderer, events, label, group).
 * @returns The box holding the chart.
 * @index function React component: a chart that fills its container, built for its width: `<ResponsiveChart build={(fig) => option} />`
 * @example
 * ```tsx
 * import { lineChart } from 'omai-render/charts'
 * import { ResponsiveChart } from 'omai-render/react'
 * const chart = (
 *   <ResponsiveChart aspect={2} narrowAspect={1.2} build={(fig) => lineChart([{ y: [0, 1, 4] }], { figure: fig })} />
 * )
 * ```
 */
export function ResponsiveChart(props: ResponsiveChartProps): ReactElement {
  const {
    build,
    aspect,
    narrowAspect,
    narrowBelow = 520,
    fontSize,
    square = false,
    overlay,
    onInit,
    ...chartProps
  } = props
  const contextMode = useColourMode()
  const mode = props.mode ?? contextMode
  const frame = useFrameSize()
  const modal = frame?.expanded === true ? frame : undefined
  const modalWidth = modal?.width
  const modalHeight = modal?.height
  const box = useRef<HTMLDivElement | null>(null)
  // The last usable width of the container; a hidden or not yet laid out box keeps the one before.
  const [measured, setMeasured] = useState<number | undefined>(undefined)
  const seen = useRef<number | undefined>(undefined)
  const take = useCallback((width: number | undefined): void => {
    const w = usable(width)
    if (w === undefined || (seen.current !== undefined && Math.abs(w - seen.current) < 1)) return
    seen.current = w
    setMeasured(w)
  }, [])
  useLayoutEffect(() => {
    const el = box.current
    if (el === null || modalWidth !== undefined) return
    take(el.clientWidth)
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver((entries) => take(entries[entries.length - 1]?.contentRect.width))
    observer.observe(el)
    return () => observer.disconnect()
  }, [take, modalWidth])

  const width = modalWidth ?? measured
  const frameAspect = frame === undefined ? undefined : frame.width / frame.height
  const shape =
    width !== undefined && width < narrowBelow && narrowAspect !== undefined ? narrowAspect : (aspect ?? frameAspect)
  const fig = useMemo((): Figure | undefined => {
    if (width === undefined) return undefined
    const made = figure({
      width,
      mode,
      ...(modalHeight === undefined ? (shape === undefined ? {} : { aspect: shape }) : { height: modalHeight }),
      ...(fontSize === undefined ? {} : { fontSize }),
    })
    return square && modalHeight === undefined ? squarePlot(made) : made
  }, [width, shape, fontSize, mode, square, modalHeight])
  const option = useMemo(() => (fig === undefined ? undefined : build(fig)), [build, fig])

  const { projection, onInit: onProjection } = useChartProjection()
  const init = useCallback(
    (chart: EChartsInstance): void => {
      onProjection(chart as unknown as PixelConverter)
      onInit?.(chart)
    },
    [onProjection, onInit],
  )

  return (
    <div ref={box} className={props.className} style={{ width: '100%', ...props.style }} data-omai-responsive="">
      {fig !== undefined && option !== undefined && (
        <Overlaid
          width={fig.width}
          height={fig.height}
          overlay={
            overlay === undefined || projection === null ? null : (
              <InteractionOverlay projection={projection} width={fig.width} height={fig.height}>
                {typeof overlay === 'function'
                  ? overlay({ projection, size: { width: fig.width, height: fig.height } })
                  : overlay}
              </InteractionOverlay>
            )
          }
        >
          <Chart {...chartProps} option={option} figure={fig} mode={mode} onInit={init} />
        </Overlaid>
      )}
    </div>
  )
}
