/**
 * A colour in a scene: a theme token resolved by the renderer (so scenes follow light and dark themes), or a literal
 * CSS colour. `tab0` to `tab9` are the tab10 palette in order.
 */
export type Paint =
  | 'fg'
  | 'bg'
  | 'surface'
  | 'muted'
  | 'grid'
  | 'accent'
  | 'none'
  | `tab${0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9}`
  | `#${string}`
  | `rgba(${string})`

/** Stroke and fill shared by every drawable node. */
export interface Style {
  readonly fill?: Paint
  readonly stroke?: Paint
  readonly strokeWidth?: number
  readonly opacity?: number
  readonly dash?: readonly number[]
}

/** Common fields of every node. */
export interface NodeBase {
  readonly id?: string
  readonly title?: string
}

/** An axis-aligned rectangle with optional rounded corners. */
export interface RectNode extends Style, NodeBase {
  readonly kind: 'rect'
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly radius?: number
}

/** A circle. */
export interface CircleNode extends Style, NodeBase {
  readonly kind: 'circle'
  readonly cx: number
  readonly cy: number
  readonly r: number
}

/** An axis-aligned ellipse. */
export interface EllipseNode extends Style, NodeBase {
  readonly kind: 'ellipse'
  readonly cx: number
  readonly cy: number
  readonly rx: number
  readonly ry: number
}

/** A straight segment, optionally with an arrow head at the end. */
export interface LineNode extends Style, NodeBase {
  readonly kind: 'line'
  readonly x1: number
  readonly y1: number
  readonly x2: number
  readonly y2: number
  readonly arrow?: boolean
}

/** A polyline or polygon through points. */
export interface PathNode extends Style, NodeBase {
  readonly kind: 'path'
  readonly points: readonly (readonly [number, number])[]
  readonly closed?: boolean
  readonly arrow?: boolean
  readonly smooth?: boolean
}

/** A text label anchored at a point. */
export interface TextNode extends NodeBase {
  readonly kind: 'text'
  readonly x: number
  readonly y: number
  readonly text: string
  readonly size?: number
  readonly fill?: Paint
  readonly anchor?: 'start' | 'middle' | 'end'
  readonly baseline?: 'top' | 'middle' | 'bottom' | 'alphabetic'
  readonly weight?: 'normal' | 'bold'
}

/** A maths label in TeX, typeset by the renderer. */
export interface TexNode extends NodeBase {
  readonly kind: 'tex'
  readonly x: number
  readonly y: number
  readonly tex: string
  readonly size?: number
  readonly fill?: Paint
  readonly anchor?: 'start' | 'middle' | 'end'
}

/** A raster image: RGBA bytes, row-major, `width * height * 4` long. */
export interface ImageNode extends NodeBase {
  readonly kind: 'image'
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly pixels: { readonly width: number; readonly height: number; readonly rgba: Uint8ClampedArray }
  readonly smoothing?: boolean
}

/** A group of nodes under a translation, rotation (degrees, about the group origin) and uniform scale. */
export interface GroupNode extends NodeBase {
  readonly kind: 'group'
  readonly children: readonly SceneNode[]
  readonly translate?: readonly [number, number]
  readonly rotate?: number
  readonly scale?: number
  readonly opacity?: number
}

/** Any node of a scene graph. */
export type SceneNode =
  | RectNode
  | CircleNode
  | EllipseNode
  | LineNode
  | PathNode
  | TextNode
  | TexNode
  | ImageNode
  | GroupNode

/**
 * A complete drawing in its own coordinate system (y grows downwards, as in SVG). Scenes are plain data: environments,
 * diagrams and models produce them, and whitebox-render draws them as SVG, React elements, images or animation frames.
 *
 * @concept scene-graph
 */
export interface Scene {
  readonly kind: 'scene'
  readonly width: number
  readonly height: number
  readonly children: readonly SceneNode[]
  readonly title?: string
  readonly background?: Paint
}
