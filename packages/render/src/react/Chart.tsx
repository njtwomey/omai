/**
 * The React component for an ECharts option: `Chart` draws it themed for the colour mode, resizes with its container,
 * disposes on unmount, forwards events, connects chart groups and answers colour-bar selections.
 */

import type { SeriesOption } from 'echarts'
import type { EChartsCoreOption, ECharts as EChartsInstance } from 'echarts/core'
import type { Pixels } from 'mdml-engine/scene'
import { type CSSProperties, type ReactElement, useEffect, useRef, useState } from 'react'
import { reencodeImages } from '../charts/matrices.ts'
import { colourSelection, type ImageEncoder } from '../charts/selection.ts'
import type { Figure } from '../figure/figure.ts'
import { pngDataUrl } from '../svg/png.ts'
import { themeName } from '../theme/echarts.ts'
import type { ColourMode } from '../theme/tokens.ts'
import { loadEcharts } from './echarts.ts'
import { useFrameSize } from './FigureFrame.tsx'
import { useColourMode } from './theme.tsx'

/** An ECharts event handler: the event's parameters and the chart instance. */
export type ChartEventHandler = (params: unknown, chart: EChartsInstance) => void

// An image as a canvas (drawn at once, where a data URL would be decoded first and flicker: while dragging a colour bar
// handle, and at every step of a chart that updates live). One canvas per pixels object, so a replayed option reuses it.
const canvases = new WeakMap<Pixels, unknown>()
function canvasImage(pixels: Pixels): unknown {
  if (typeof document === 'undefined') return pngDataUrl(pixels)
  const known = canvases.get(pixels)
  if (known !== undefined) return known
  const canvas = document.createElement('canvas')
  canvas.width = pixels.width
  canvas.height = pixels.height
  const ctx = canvas.getContext('2d')
  if (ctx === null) return pngDataUrl(pixels)
  ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels.rgba), pixels.width, pixels.height), 0, 0)
  canvases.set(pixels, canvas)
  return canvas
}

// The option as the canvas renderer draws it: image series painted from canvases (see reencodeImages).
function forRenderer(option: EChartsCoreOption, renderer: 'canvas' | 'svg'): EChartsCoreOption {
  const series = (option as { series?: unknown }).series
  if (renderer !== 'canvas' || !Array.isArray(series)) return option
  return { ...option, series: reencodeImages(series as SeriesOption[], canvasImage) } as EChartsCoreOption
}

/** Props of {@link Chart}. */
export interface ChartProps {
  /** The option, from a `mdml-render/charts` builder or written by hand. */
  readonly option: EChartsCoreOption
  /** A figure giving the chart's fixed width and height (the builders' default figure is 640 by 400). */
  readonly figure?: Figure
  /** Width in pixels or a CSS length (default the figure's width, else 100%). */
  readonly width?: number | string
  /** Height in pixels (default the figure's height, else 400). */
  readonly height?: number
  /** Colour mode (default the {@link useColourMode} mode). */
  readonly mode?: ColourMode
  /** Canvas (default) or SVG rendering. */
  readonly renderer?: 'canvas' | 'svg'
  /** Event handlers by ECharts event name (`click`, `mouseover`, `datazoom`, ...). */
  readonly onEvents?: Readonly<Record<string, ChartEventHandler>>
  /** Called with the instance after it is created (for `dispatchAction`, export and the like). */
  readonly onInit?: (chart: EChartsInstance) => void
  /** Charts with the same group are connected: hover, tooltips and zoom are shared (`echarts.connect`). */
  readonly group?: string
  /** Merge new options into the old one instead of replacing it (default false: replace). */
  readonly merge?: boolean
  readonly className?: string
  readonly style?: CSSProperties
}

/**
 * An ECharts option as a live chart: themed for the colour mode, resized with its container, disposed on unmount, with
 * event handlers and connected groups. The area has a fixed size (from `figure`, or `width` and `height`), so content
 * never resizes it.
 *
 * Inside a {@link FigureFrame} it takes the frame's size by default, and the modal's size when the frame is expanded.
 *
 * @param props - The option, size, mode, renderer, events and group.
 * @returns The chart's container.
 */
export function Chart(props: ChartProps): ReactElement {
  const { option, figure, renderer = 'canvas', onEvents, onInit, group, merge = false } = props
  const contextMode = useColourMode()
  const frame = useFrameSize()
  const mode = props.mode ?? contextMode
  const ratio = frame?.scale
  const ref = useRef<HTMLDivElement | null>(null)
  const [chart, setChart] = useState<EChartsInstance | null>(null)
  const optionRef = useRef(option)
  optionRef.current = option
  const onInitRef = useRef(onInit)
  onInitRef.current = onInit

  // Create the instance for a renderer and theme; recreate when either changes.
  useEffect(() => {
    const el = ref.current
    if (el === null) return
    const echarts = loadEcharts()
    // Magnified in an expanded frame: render at a pixel ratio to match, so the canvas stays sharp.
    const instance = echarts.init(el, themeName(mode), {
      renderer,
      ...(ratio === undefined ? {} : { devicePixelRatio: (globalThis.devicePixelRatio || 1) * ratio }),
    })
    instance.setOption(forRenderer(optionRef.current, renderer), { notMerge: true })
    onInitRef.current?.(instance)
    setChart(instance)
    const observer = typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(() => instance.resize())
    observer?.observe(el)
    return () => {
      observer?.disconnect()
      setChart(null)
      instance.dispose()
    }
  }, [mode, renderer, ratio])

  useEffect(() => {
    chart?.setOption(forRenderer(option, renderer), { notMerge: !merge, lazyUpdate: true })
  }, [chart, option, merge, renderer])

  // Colour bar selections: layers ECharts cannot recolour (images, baked cells) are repainted with the values outside
  // the selection dimmed, at most once per animation frame while a handle is dragged.
  useEffect(() => {
    if (chart === null) return
    const encode: ImageEncoder = renderer === 'svg' ? pngDataUrl : canvasImage
    let pending: unknown
    let frame: number | undefined
    const flush = (): void => {
      frame = undefined
      const update = colourSelection(optionRef.current, pending, encode)
      if (update !== undefined && !chart.isDisposed()) chart.setOption(update)
    }
    const onSelect = (params: unknown): void => {
      pending = params
      if (frame === undefined) frame = requestAnimationFrame(flush)
    }
    chart.on('datarangeselected', onSelect)
    return () => {
      chart.off('datarangeselected', onSelect)
      if (frame !== undefined) cancelAnimationFrame(frame)
    }
  }, [chart, renderer])

  useEffect(() => {
    if (chart === null || onEvents === undefined) return
    const bound = Object.entries(onEvents).map(([name, handler]) => {
      const fn = (params: unknown): void => handler(params, chart)
      chart.on(name, fn)
      return [name, fn] as const
    })
    return () => {
      for (const [name, fn] of bound) chart.off(name, fn)
    }
  }, [chart, onEvents])

  useEffect(() => {
    if (chart === null || group === undefined) return
    chart.group = group
    loadEcharts().connect(group)
  }, [chart, group])

  // Inside an expanded FigureFrame the modal's size wins; otherwise explicit sizes, then the figure, then the frame.
  const width = frame?.expanded === true ? frame.width : (props.width ?? figure?.width ?? frame?.width ?? '100%')
  const height = frame?.expanded === true ? frame.height : (props.height ?? figure?.height ?? frame?.height ?? 400)
  return (
    <div
      ref={ref}
      className={props.className}
      style={{ width, height, maxWidth: '100%', ...props.style }}
      data-mdml-chart=""
    />
  )
}
