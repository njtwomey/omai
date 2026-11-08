/**
 * The page a figure is drawn on: an SVG document with a white background, and panels that hold a chart (an ECharts
 * option from `mdml-render`) or a scene under a heading. The text uses a generic font stack, so a figure reads the same
 * on GitHub as in a browser; everything is plain SVG (no foreign objects).
 */

import { isScene } from 'mdml-engine/scene'
import { chartToSvg, sceneToSvg } from 'mdml-render/svg'
import type { ReactElement, ReactNode } from 'react'
import { INK, PAPER } from './colours.ts'

const FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

/** A chart option (what `stillOf` and the chart builders return) or a scene. */
export type Still = unknown

/** A chart option or scene as SVG text, on white. */
export function svgOf(still: Still, width: number, height: number): string {
  if (isScene(still)) return sceneToSvg(still, { mode: 'light' })
  const option = { ...(still as Parameters<typeof chartToSvg>[0]), backgroundColor: PAPER }
  // ECharts draws an image element with the default aspect ratio rule (the picture kept square and centred), which leaves
  // blank bands beside a heat map; stretching it to the plotting area is what makes the shading fill the axes.
  return chartToSvg(option, { width, height, mode: 'light' }).replace(/<image /g, '<image preserveAspectRatio="none" ')
}

/** The page: `width` by `height` pixels, white. */
export function Figure(props: { width: number; height: number; title: string; children: ReactNode }): ReactElement {
  const { width, height } = props
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <title>{props.title}</title>
      <rect width="100%" height="100%" fill={PAPER} />
      {props.children}
    </svg>
  )
}

/** A chart or scene at a position, under an optional heading. */
export function Panel(props: {
  still: Still
  x?: number
  width: number
  height: number
  title?: string
}): ReactElement {
  const { x = 0, width, height, title } = props
  const top = title === undefined ? 0 : 26
  return (
    <>
      {title === undefined ? null : (
        <text x={x + width / 2} y={19} textAnchor="middle" fontFamily={FONT} fontSize={13} fontWeight={600} fill={INK}>
          {title}
        </text>
      )}
      <svg
        x={x}
        y={top}
        width={width}
        height={height}
        overflow="visible"
        // biome-ignore lint/security/noDangerouslySetInnerHtml: the chart is SVG text from our own renderer
        dangerouslySetInnerHTML={{ __html: svgOf(props.still, width, height) }}
      />
    </>
  )
}

/** Panels side by side, `width` each, with headings. */
export function Row(props: {
  panels: readonly { title?: string; still: Still }[]
  width: number
  height: number
  alt: string
}): ReactElement {
  const { panels, width, height } = props
  const heading = panels.some((p) => p.title !== undefined) ? 26 : 0
  return (
    <Figure width={width * panels.length} height={height + heading} title={props.alt}>
      {panels.map((p, i) => (
        <Panel
          key={p.title ?? i}
          x={i * width}
          width={width}
          height={height}
          {...(p.title === undefined ? {} : { title: p.title })}
          still={p.still}
        />
      ))}
    </Figure>
  )
}
