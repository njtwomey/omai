import { type ReactNode, useId } from 'react'
import { type Figure, figure, plotArea } from 'whitebox-render/figure'
import { linearProjection, type Point2, type Projection } from 'whitebox-render/react'
import { formatNumber, niceTicks } from 'whitebox-render/theme'

/**
 * A plain SVG plot for the gallery's recipes about controls, interaction and playback, until whitebox-render's
 * `Chart` lands: a figure of the default size with fixed limits, axes, ticks and grid lines, and marks drawn through
 * the same projection the interaction overlays use. (Replace with `Chart` and `chartProjection` when available.)
 */
export interface PlotFrame {
  readonly fig: Figure
  readonly projection: Projection
  readonly x: Point2
  readonly y: Point2
}

/** The frame of a plot: a figure (default page size) and the projection of fixed x and y limits onto its area. */
export function plotFrame(options: {
  readonly x: Point2
  readonly y: Point2
  readonly width?: number
  readonly aspect?: number
}): PlotFrame {
  const fig = figure({
    ...(options.width === undefined ? {} : { width: options.width }),
    ...(options.aspect === undefined ? {} : { aspect: options.aspect }),
  })
  const area = plotArea(fig)
  const projection = linearProjection({
    x: options.x,
    y: options.y,
    width: fig.width,
    height: fig.height,
    margins: {
      left: area.x,
      top: area.y,
      right: fig.width - area.x - area.width,
      bottom: fig.height - area.y - area.height,
    },
  })
  return { fig, projection, x: options.x, y: options.y }
}

/** The axes of a frame with its marks inside. */
export function Plot(props: {
  readonly frame: PlotFrame
  readonly children?: ReactNode
  readonly xLabel?: string
  readonly yLabel?: string
}): ReactNode {
  const { frame, children, xLabel, yLabel } = props
  const { fig, projection: p, x, y } = frame
  const area = plotArea(fig)
  const clip = `plot-clip-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`
  const xt = niceTicks(x[0], x[1], 6).ticks.filter((t) => t >= x[0] - 1e-9 && t <= x[1] + 1e-9)
  const yt = niceTicks(y[0], y[1], 5).ticks.filter((t) => t >= y[0] - 1e-9 && t <= y[1] + 1e-9)
  return (
    <svg
      width={fig.width}
      height={fig.height}
      viewBox={`0 0 ${fig.width} ${fig.height}`}
      className="block max-w-full"
      style={{ fontSize: fig.fontSize, fontFamily: 'var(--wb-font)' }}
      role="img"
    >
      <title>{[yLabel, xLabel].filter(Boolean).join(' against ') || 'plot'}</title>
      <g stroke="var(--wb-grid)" strokeWidth={1}>
        {xt.map((t) => (
          <line
            key={`x${t}`}
            x1={p.toPixel([t, 0])[0]}
            x2={p.toPixel([t, 0])[0]}
            y1={area.y}
            y2={area.y + area.height}
          />
        ))}
        {yt.map((t) => (
          <line
            key={`y${t}`}
            y1={p.toPixel([0, t])[1]}
            y2={p.toPixel([0, t])[1]}
            x1={area.x}
            x2={area.x + area.width}
          />
        ))}
      </g>
      <rect x={area.x} y={area.y} width={area.width} height={area.height} fill="none" stroke="var(--wb-axis)" />
      <g fill="var(--wb-muted)">
        {xt.map((t) => (
          <text key={`xl${t}`} x={p.toPixel([t, 0])[0]} y={area.y + area.height + 16} textAnchor="middle">
            {formatNumber(t)}
          </text>
        ))}
        {yt.map((t) => (
          <text key={`yl${t}`} x={area.x - 6} y={p.toPixel([0, t])[1] + 4} textAnchor="end">
            {formatNumber(t)}
          </text>
        ))}
        {xLabel === undefined ? null : (
          <text x={area.x + area.width / 2} y={fig.height - 8} textAnchor="middle" fill="var(--wb-fg-secondary)">
            {xLabel}
          </text>
        )}
        {yLabel === undefined ? null : (
          <text
            transform={`translate(14 ${area.y + area.height / 2}) rotate(-90)`}
            textAnchor="middle"
            fill="var(--wb-fg-secondary)"
          >
            {yLabel}
          </text>
        )}
      </g>
      <clipPath id={clip}>
        <rect x={area.x} y={area.y} width={area.width} height={area.height} />
      </clipPath>
      <g clipPath={`url(#${clip})`}>{children}</g>
    </svg>
  )
}

/** A polyline through points, in a palette slot (`tab0` .. `tab9`) or any colour. */
export function Line(props: {
  readonly frame: PlotFrame
  readonly points: readonly Point2[]
  readonly colour?: string
  readonly width?: number
  readonly dashed?: boolean
}): ReactNode {
  const { frame, points, colour = 'var(--wb-tab0)', width = 2, dashed = false } = props
  const d = points
    .map((q, i) => {
      const [px, py] = frame.projection.toPixel(q)
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(2)},${py.toFixed(2)}`
    })
    .join('')
  return (
    <path
      d={d}
      fill="none"
      stroke={colour}
      strokeWidth={width}
      strokeLinejoin="round"
      strokeDasharray={dashed ? '5 4' : undefined}
    />
  )
}

/** Dots at points. */
export function Dots(props: {
  readonly frame: PlotFrame
  readonly points: readonly Point2[]
  readonly colour?: string | ((index: number) => string)
  readonly radius?: number
  readonly opacity?: number
}): ReactNode {
  const { frame, points, colour = 'var(--wb-tab0)', radius = 3, opacity = 0.85 } = props
  return (
    <g fillOpacity={opacity}>
      {points.map((q, i) => {
        const [px, py] = frame.projection.toPixel(q)
        return (
          // biome-ignore lint/suspicious/noArrayIndexKey: points are drawn in order
          <circle key={i} cx={px} cy={py} r={radius} fill={typeof colour === 'function' ? colour(i) : colour} />
        )
      })}
    </g>
  )
}

/** Sample a function on a grid of `n` points across `[a, b]`. */
export function sampleCurve(f: (x: number) => number, a: number, b: number, n = 200): Point2[] {
  return Array.from({ length: n }, (_, i): Point2 => {
    const x = a + ((b - a) * i) / (n - 1)
    return [x, f(x)]
  })
}
