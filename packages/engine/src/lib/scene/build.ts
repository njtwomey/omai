import type {
  CircleNode,
  EllipseNode,
  GroupNode,
  ImageNode,
  LineNode,
  NodeBase,
  Paint,
  PathNode,
  RectNode,
  Scene,
  SceneNode,
  Style,
  TexNode,
  TextNode,
} from './types.ts'

/**
 * Build a scene from its size and children.
 *
 * @param width - Width in scene units.
 * @param height - Height in scene units.
 * @param children - The nodes, back to front.
 * @param options - An accessible title and a background paint.
 * @returns The scene.
 * @concept scene-graph
 * @example
 * ```ts
 * import { scene, circle } from 'whitebox-engine/scene'
 * scene(100, 50, [circle(50, 25, 10, { fill: 'tab0' })]).children.length // => 1
 * ```
 */
export function scene(
  width: number,
  height: number,
  children: readonly SceneNode[],
  options: { readonly title?: string; readonly background?: Paint } = {},
): Scene {
  return { kind: 'scene', width, height, children, ...options }
}

/**
 * A rectangle.
 *
 * @param x - Left edge.
 * @param y - Top edge.
 * @param width - Width.
 * @param height - Height.
 * @param style - Fill, stroke, corner radius, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { rect } from 'whitebox-engine/scene'
 * rect(0, 0, 10, 5, { fill: 'tab1' }).width // => 10
 * ```
 */
export function rect(
  x: number,
  y: number,
  width: number,
  height: number,
  style: Style & NodeBase & { readonly radius?: number } = {},
): RectNode {
  return { kind: 'rect', x, y, width, height, ...style }
}

/**
 * A circle.
 *
 * @param cx - Centre x.
 * @param cy - Centre y.
 * @param r - Radius.
 * @param style - Fill, stroke, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { circle } from 'whitebox-engine/scene'
 * circle(5, 5, 2).r // => 2
 * ```
 */
export function circle(cx: number, cy: number, r: number, style: Style & NodeBase = {}): CircleNode {
  return { kind: 'circle', cx, cy, r, ...style }
}

/**
 * An axis-aligned ellipse.
 *
 * @param cx - Centre x.
 * @param cy - Centre y.
 * @param rx - Horizontal radius.
 * @param ry - Vertical radius.
 * @param style - Fill, stroke, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { ellipse } from 'whitebox-engine/scene'
 * ellipse(0, 0, 3, 1).rx // => 3
 * ```
 */
export function ellipse(cx: number, cy: number, rx: number, ry: number, style: Style & NodeBase = {}): EllipseNode {
  return { kind: 'ellipse', cx, cy, rx, ry, ...style }
}

/**
 * A line segment.
 *
 * @param x1 - Start x.
 * @param y1 - Start y.
 * @param x2 - End x.
 * @param y2 - End y.
 * @param style - Stroke, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { line } from 'whitebox-engine/scene'
 * line(0, 0, 3, 4).x2 // => 3
 * ```
 */
export function line(x1: number, y1: number, x2: number, y2: number, style: Style & NodeBase = {}): LineNode {
  return { kind: 'line', x1, y1, x2, y2, ...style }
}

/**
 * A line segment with an arrow head at its end.
 *
 * @param x1 - Start x.
 * @param y1 - Start y.
 * @param x2 - End x.
 * @param y2 - End y.
 * @param style - Stroke, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { arrow } from 'whitebox-engine/scene'
 * arrow(0, 0, 3, 4).arrow // => true
 * ```
 */
export function arrow(x1: number, y1: number, x2: number, y2: number, style: Style & NodeBase = {}): LineNode {
  return { kind: 'line', x1, y1, x2, y2, arrow: true, ...style }
}

/**
 * A polyline (open by default) or polygon.
 *
 * @param points - The vertices in order.
 * @param style - Stroke, fill, closure, smoothing, arrow head, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { path } from 'whitebox-engine/scene'
 * path([[0, 0], [1, 1], [2, 0]]).points.length // => 3
 * ```
 */
export function path(
  points: readonly (readonly [number, number])[],
  style: Style & NodeBase & { readonly closed?: boolean; readonly arrow?: boolean; readonly smooth?: boolean } = {},
): PathNode {
  return { kind: 'path', points, ...style }
}

/**
 * A closed polygon.
 *
 * @param points - The vertices in order.
 * @param style - Fill, stroke, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { polygon } from 'whitebox-engine/scene'
 * polygon([[0, 0], [1, 0], [0, 1]]).closed // => true
 * ```
 */
export function polygon(points: readonly (readonly [number, number])[], style: Style & NodeBase = {}): PathNode {
  return { kind: 'path', points, closed: true, ...style }
}

/**
 * A text label.
 *
 * @param x - Anchor x.
 * @param y - Anchor y.
 * @param content - The text.
 * @param style - Size, fill, anchor, baseline, weight, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { text } from 'whitebox-engine/scene'
 * text(0, 10, 'hello', { anchor: 'middle' }).text // => 'hello'
 * ```
 */
export function text(
  x: number,
  y: number,
  content: string,
  style: Omit<TextNode, 'kind' | 'x' | 'y' | 'text'> = {},
): TextNode {
  return { kind: 'text', x, y, text: content, ...style }
}

/**
 * A maths label in TeX.
 *
 * @param x - Anchor x.
 * @param y - Anchor y.
 * @param source - The TeX source.
 * @param style - Size, fill, anchor, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { tex } from 'whitebox-engine/scene'
 * tex(0, 0, '\\alpha').tex // => '\\alpha'
 * ```
 */
export function tex(
  x: number,
  y: number,
  source: string,
  style: Omit<TexNode, 'kind' | 'x' | 'y' | 'tex'> = {},
): TexNode {
  return { kind: 'tex', x, y, tex: source, ...style }
}

/**
 * A raster image from RGBA bytes.
 *
 * @param x - Left edge.
 * @param y - Top edge.
 * @param width - Drawn width.
 * @param height - Drawn height.
 * @param pixels - The image: its pixel size and RGBA bytes.
 * @param options - Smoothing when scaled (off by default, so pixels stay crisp), id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { image } from 'whitebox-engine/scene'
 * image(0, 0, 10, 10, { width: 1, height: 1, rgba: new Uint8ClampedArray(4) }).width // => 10
 * ```
 */
export function image(
  x: number,
  y: number,
  width: number,
  height: number,
  pixels: ImageNode['pixels'],
  options: NodeBase & { readonly smoothing?: boolean } = {},
): ImageNode {
  return { kind: 'image', x, y, width, height, pixels, ...options }
}

/**
 * A group that translates, rotates and scales its children.
 *
 * @param children - The grouped nodes.
 * @param transform - Translation, rotation in degrees, scale, opacity, id and title.
 * @returns The node.
 * @example
 * ```ts
 * import { group, circle } from 'whitebox-engine/scene'
 * group([circle(0, 0, 1)], { translate: [5, 5] }).translate // => [5, 5]
 * ```
 */
export function group(
  children: readonly SceneNode[],
  transform: NodeBase & {
    readonly translate?: readonly [number, number]
    readonly rotate?: number
    readonly scale?: number
    readonly opacity?: number
  } = {},
): GroupNode {
  return { kind: 'group', children, ...transform }
}

/**
 * Whether a value is a {@link Scene}; the example runner uses this for `// => figure`.
 *
 * @param value - Any value.
 * @returns `true` for scenes.
 * @example
 * ```ts
 * import { isScene, scene } from 'whitebox-engine/scene'
 * isScene(scene(1, 1, [])) // => true
 * isScene({}) // => false
 * ```
 */
export function isScene(value: unknown): value is Scene {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'scene'
}
