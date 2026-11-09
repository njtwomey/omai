/**
 * The grip drawn on an interaction tool: a bar, a square, a circle or a diamond, centred on the point it is drawn at.
 * `DraggableLine` puts one on the end of its line and `DraggableRect` one on each corner and edge of its rectangle.
 */

import type { ReactNode } from 'react'

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
}

/**
 * The handle drawn at a point of an interaction tool, centred on the origin of the element it is placed in (put it
 * inside a `<g transform="translate(x, y)">`): a bar, a square, a circle or a diamond (a square turned on its corner,
 * of the same width and height), outlined 1.5 pixels.
 *
 * @param props - The shape, size, direction of the line it sits on, fill and stroke.
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
  const { shape, size, vertical, fill, stroke } = props
  if (shape === 'circle') return <circle r={size / 2} fill={fill} stroke={stroke} strokeWidth={1.5} />
  if (shape === 'diamond') {
    const r = size / 2
    return (
      <polygon
        points={`0,${-r} ${r},0 0,${r} ${-r},0`}
        fill={fill}
        stroke={stroke}
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
    )
  }
  const [w, h] = shape === 'square' ? [size, size] : vertical ? [size, 2 * size] : [2 * size, size]
  return <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={3} fill={fill} stroke={stroke} strokeWidth={1.5} />
}
