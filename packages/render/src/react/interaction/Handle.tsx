/**
 * The grip drawn on an interaction tool: a bar, a square, a circle or a diamond, centred on the point it is drawn at.
 * `DraggableLine` puts one on the end of its line and `DraggableRect` one on each corner and edge of its rectangle.
 */

import type { ReactNode } from 'react'
import { FOCUS_COLOUR } from './useFocusRing.ts'

/** The shapes of a handle: a `bar` (the default of a line), a `square`, a `circle` or a `diamond`. */
export type HandleShape = 'bar' | 'square' | 'circle' | 'diamond'

/** Props of {@link Handle}. */
export interface HandleProps {
  /** The shape. */
  readonly shape: HandleShape
  /**
   * The side of a square, the diameter of a circle, the width and height of a diamond; a bar is `size` wide and
   * twice that long (`size` by `2 * size`), the long side lying along the line or edge it sits on.
   */
  readonly size: number
  /** Whether the line or edge the handle sits on is vertical (it sets which way a bar lies). */
  readonly vertical: boolean
  /** The fill: a CSS colour or `var(--wb-…)` token. */
  readonly fill: string
  /** The outline colour. */
  readonly stroke: string
  /** Draw the keyboard focus ring around the handle (a ring in the accent colour, 4 pixels outside the outline; default off). */
  readonly ring?: boolean
}

/** What a handle is filled with: `solid` (the tool's colour), `bg` (the page colour: a hollow grip) or any CSS colour. */
export type HandleFill = 'solid' | 'bg' | (string & {})

/**
 * The CSS fill of a handle from the `handleFill` a tool was given: `solid` is the tool's `colour`, `bg` or nothing the
 * page colour (`var(--wb-bg)`), or the colour of `active` when `activeSolid` says a hollow grip fills while it is being
 * dragged; any other string is a CSS colour and is used as it is.
 *
 * @param fill - `solid`, `bg` or a CSS colour; undefined is `bg`.
 * @param colour - The tool's outline colour.
 * @param activeSolid - Fill a `bg` (or unset) handle with `colour` instead (while it is dragged; default false).
 * @returns A CSS colour.
 * @example
 * ```ts
 * handleFillOf('solid', 'red') // 'red'
 * handleFillOf(undefined, 'red') // 'var(--wb-bg)'
 * handleFillOf('bg', 'red', true) // 'red'
 * handleFillOf('#fc0', 'red', true) // '#fc0'
 * ```
 */
export function handleFillOf(fill: HandleFill | undefined, colour: string, activeSolid = false): string {
  if (fill === 'solid') return colour
  if (fill === undefined || fill === 'bg') return activeSolid ? colour : 'var(--wb-bg)'
  return fill
}

/**
 * The handle drawn at a point of an interaction tool, centred on the origin of the element it is placed in (put it
 * inside a `<g transform="translate(x, y)">`): a bar, a square, a circle or a diamond (a square turned on its corner,
 * of the same width and height), outlined 1.5 pixels. With `ring` it has the keyboard focus ring around it.
 *
 * @param props - The shape, size, direction of the line it sits on, fill, stroke and focus ring.
 * @returns The handle's SVG element.
 * @index function React component: the grip of an overlay tool: `<Handle shape="square" size={8} />`
 * @example
 * ```tsx
 * <g transform="translate(120, 40)">
 *   <Handle shape="circle" size={10} vertical={false} fill="var(--wb-bg)" stroke="var(--wb-accent)" />
 * </g>
 * ```
 */
export function Handle(props: HandleProps): ReactNode {
  const { shape, size, vertical, fill, stroke, ring = false } = props
  const mark = {
    fill: 'none',
    stroke: FOCUS_COLOUR,
    strokeWidth: 2,
    pointerEvents: 'none',
    'data-focus-ring': 'true',
  } as const
  if (shape === 'circle') {
    return (
      <>
        {ring && <circle r={size / 2 + 4} {...mark} />}
        <circle r={size / 2} fill={fill} stroke={stroke} strokeWidth={1.5} />
      </>
    )
  }
  if (shape === 'diamond') {
    const r = size / 2
    const out = r + 5
    return (
      <>
        {ring && <polygon points={`0,${-out} ${out},0 0,${out} ${-out},0`} strokeLinejoin="round" {...mark} />}
        <polygon
          points={`0,${-r} ${r},0 0,${r} ${-r},0`}
          fill={fill}
          stroke={stroke}
          strokeWidth={1.5}
          strokeLinejoin="round"
        />
      </>
    )
  }
  const [w, h] = shape === 'square' ? [size, size] : vertical ? [size, 2 * size] : [2 * size, size]
  return (
    <>
      {ring && <rect x={-w / 2 - 4} y={-h / 2 - 4} width={w + 8} height={h + 8} rx={6} {...mark} />}
      <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={3} fill={fill} stroke={stroke} strokeWidth={1.5} />
    </>
  )
}
