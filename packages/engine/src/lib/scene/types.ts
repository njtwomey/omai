/**
 * The scene graph types: theme tokens and paints that follow light and dark themes, styles, the node kinds (rectangles,
 * circles, ellipses, lines, paths, text, TeX, images, groups and 3-D layers), transforms, and `Scene`, a complete
 * drawing in SVG-like coordinates with y growing downwards.
 *
 * @index structure Scene graph node types: shapes, text, paths, 3-D layers.
 */

import type { Camera } from './projection.ts'
import type { Scene3D } from './three-d.ts'

/**
 * A colour named by role rather than value, resolved by the renderer for the current light or dark theme: `fg` (text
 * and default strokes), `bg` (the page), `surface` (panels and cards), `muted` (secondary text and guides), `grid`
 * (grid lines), `accent` (highlights), and `tab0` to `tab9` (the tab10 categorical palette, in order).
 */
export type ThemeToken =
  | 'fg'
  | 'bg'
  | 'surface'
  | 'muted'
  | 'grid'
  | 'accent'
  | `tab${0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9}`

/**
 * A colour in a scene: a {@link ThemeToken} (so scenes follow light and dark themes), `none`, or a literal CSS colour
 * (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, or `rgba(r, g, b, a)` with channels 0 to 255 and alpha 0 to 1).
 */
export type Paint = ThemeToken | 'none' | `#${string}` | `rgba(${string})`

/**
 * Stroke and fill shared by every drawable node. Every renderer applies the same defaults: a closed shape (rect,
 * circle, ellipse, closed path) with neither `fill` nor `stroke` set is filled with `fg`; a line or open path with no
 * `stroke` is stroked with `fg`; otherwise an unset paint draws nothing. `strokeWidth` defaults to 1, `opacity` to 1,
 * and `dash` lists alternating dash and gap lengths.
 */
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
  /**
   * Round the corners of the polyline with arcs of this radius, in scene units (default 0: sharp corners). Each corner's
   * radius is limited to half the shorter of its two segments, so the line still passes through the middle of every
   * segment; the ends and the arrow head are unchanged. Ignored when `smooth` is set (that splines the whole path).
   */
  readonly cornerRadius?: number
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

/**
 * Pixels of a raster image: straight (not premultiplied) RGBA bytes, row-major, `width * height * 4` long. The bytes are a
 * typed array, so a scene with images survives the trace codec (`$typed`) but not a plain `JSON.stringify`, which
 * writes the array as an object of indices; `validateScene` reports such a scene.
 */
export interface Pixels {
  readonly width: number
  readonly height: number
  readonly rgba: Uint8ClampedArray
}

/** A raster image drawn into a box. */
export interface ImageNode extends NodeBase {
  readonly kind: 'image'
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly pixels: Pixels
  readonly smoothing?: boolean
}

/**
 * A similarity transform of the plane: a point `p` maps to `translate + scale * R(rotate) * p`, with `rotate` in
 * degrees (clockwise on screen, since y grows downwards). Missing fields are the identity.
 */
export interface Transform {
  readonly translate?: readonly [number, number]
  readonly rotate?: number
  readonly scale?: number
}

/** A group of nodes under a {@link Transform} and an opacity. */
export interface GroupNode extends NodeBase, Transform {
  readonly kind: 'group'
  readonly children: readonly SceneNode[]
  readonly opacity?: number
}

/**
 * A 3-D layer: a {@link Scene3D} seen through a {@link Camera}, drawn into a box. Renderers project it (`projectLayers`
 * in `projection.ts`); interactive views re-project it when the camera orbits.
 */
export interface Layer3DNode extends NodeBase {
  readonly kind: 'layer3d'
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly scene: Scene3D
  readonly camera: Camera
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
  | Layer3DNode

/**
 * A complete drawing in its own coordinate system (y grows downwards, as in SVG). Scenes are plain data: environments,
 * diagrams and models produce them, and omai-render draws them as SVG, React elements, images or animation frames.
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
