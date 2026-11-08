/**
 * Constructors for scenes, the engine's drawings as data: the scene itself, shapes (rectangles, circles, ellipses,
 * lines, arrows, paths, polygons), text and TeX labels, raster images from tensors or nested arrays (with `toPixels`
 * converting them as `imshow` would), and transforming groups.
 */

import { clamp, maxOf, minOf, OmaiError } from '../base/index.ts'
import { transform as checkTransform, finite, size, vertices } from './_check.ts'
import type {
  CircleNode,
  EllipseNode,
  GroupNode,
  ImageNode,
  LineNode,
  NodeBase,
  Paint,
  PathNode,
  Pixels,
  RectNode,
  Scene,
  SceneNode,
  Style,
  TexNode,
  TextNode,
  Transform,
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
 * @index function Build a Scene from nodes; rect, circle and path sit beside it.
 * @throws OmaiError `value` when the width or height is not a finite number >= 0.
 * @example
 * ```ts
 * import { scene, circle } from 'omai-engine/scene'
 * scene(100, 50, [circle(50, 25, 10, { fill: 'tab0' })]).children.length // => 1
 * ```
 */
export function scene(
  width: number,
  height: number,
  children: readonly SceneNode[],
  options: { readonly title?: string; readonly background?: Paint } = {},
): Scene {
  size('scene', { width, height })
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
 * @throws OmaiError `value` when a coordinate is not finite or the width or height is not a finite number >= 0.
 * @example
 * ```ts
 * import { rect } from 'omai-engine/scene'
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
  finite('rect', { x, y })
  size('rect', { width, height })
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
 * @throws OmaiError `value` when the centre is not finite or the radius is not a finite number >= 0.
 * @example
 * ```ts
 * import { circle } from 'omai-engine/scene'
 * circle(5, 5, 2).r // => 2
 * ```
 */
export function circle(cx: number, cy: number, r: number, style: Style & NodeBase = {}): CircleNode {
  finite('circle', { cx, cy })
  size('circle', { r })
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
 * @throws OmaiError `value` when the centre is not finite or a radius is not a finite number >= 0.
 * @example
 * ```ts
 * import { ellipse } from 'omai-engine/scene'
 * ellipse(0, 0, 3, 1).rx // => 3
 * ```
 */
export function ellipse(cx: number, cy: number, rx: number, ry: number, style: Style & NodeBase = {}): EllipseNode {
  finite('ellipse', { cx, cy })
  size('ellipse', { rx, ry })
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
 * @throws OmaiError `value` when a coordinate is not finite.
 * @example
 * ```ts
 * import { line } from 'omai-engine/scene'
 * line(0, 0, 3, 4).x2 // => 3
 * ```
 */
export function line(x1: number, y1: number, x2: number, y2: number, style: Style & NodeBase = {}): LineNode {
  finite('line', { x1, y1, x2, y2 })
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
 * @throws OmaiError `value` when a coordinate is not finite.
 * @example
 * ```ts
 * import { arrow } from 'omai-engine/scene'
 * arrow(0, 0, 3, 4).arrow // => true
 * ```
 */
export function arrow(x1: number, y1: number, x2: number, y2: number, style: Style & NodeBase = {}): LineNode {
  finite('arrow', { x1, y1, x2, y2 })
  return { kind: 'line', x1, y1, x2, y2, arrow: true, ...style }
}

/**
 * A polyline (open by default) or polygon.
 *
 * @param points - The vertices in order.
 * @param style - Stroke, fill, closure, smoothing, arrow head, id and title.
 * @returns The node.
 * @throws OmaiError `value` when a vertex is not a pair of finite numbers.
 * @example
 * ```ts
 * import { path } from 'omai-engine/scene'
 * path([[0, 0], [1, 1], [2, 0]]).points.length // => 3
 * ```
 */
export function path(
  points: readonly (readonly [number, number])[],
  style: Style & NodeBase & { readonly closed?: boolean; readonly arrow?: boolean; readonly smooth?: boolean } = {},
): PathNode {
  vertices('path', points, 2)
  return { kind: 'path', points, ...style }
}

/**
 * A closed polygon.
 *
 * @param points - The vertices in order.
 * @param style - Fill, stroke, id and title.
 * @returns The node.
 * @throws OmaiError `value` when a vertex is not a pair of finite numbers.
 * @example
 * ```ts
 * import { polygon } from 'omai-engine/scene'
 * polygon([[0, 0], [1, 0], [0, 1]]).closed // => true
 * ```
 */
export function polygon(points: readonly (readonly [number, number])[], style: Style & NodeBase = {}): PathNode {
  vertices('polygon', points, 2)
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
 * @throws OmaiError `value` when the anchor is not finite or the size is not a finite number >= 0.
 * @example
 * ```ts
 * import { text } from 'omai-engine/scene'
 * text(0, 10, 'hello', { anchor: 'middle' }).text // => 'hello'
 * ```
 */
export function text(
  x: number,
  y: number,
  content: string,
  style: Omit<TextNode, 'kind' | 'x' | 'y' | 'text'> = {},
): TextNode {
  finite('text', { x, y })
  if (style.size !== undefined) size('text', { size: style.size })
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
 * @throws OmaiError `value` when the anchor is not finite or the size is not a finite number >= 0.
 * @example
 * ```ts
 * import { tex } from 'omai-engine/scene'
 * tex(0, 0, '\\alpha').tex // => '\\alpha'
 * ```
 */
export function tex(
  x: number,
  y: number,
  source: string,
  style: Omit<TexNode, 'kind' | 'x' | 'y' | 'tex'> = {},
): TexNode {
  finite('tex', { x, y })
  if (style.size !== undefined) size('tex', { size: style.size })
  return { kind: 'tex', x, y, tex: source, ...style }
}

/**
 * Image data to turn into pixels: a tensor (anything with a `shape` and flat row-major `data`) of shape `[h, w]`
 * (grey), `[h, w, 1]`, `[h, w, 3]` (RGB) or `[h, w, 4]` (RGBA), or the same as nested arrays.
 */
export type ImageSource =
  | { readonly shape: readonly number[]; readonly data: ArrayLike<number> }
  | readonly (readonly number[])[]
  | readonly (readonly (readonly number[])[])[]

function flattenSource(source: ImageSource): { shape: number[]; data: ArrayLike<number> } {
  if (!Array.isArray(source)) {
    const t = source as { readonly shape: readonly number[]; readonly data: ArrayLike<number> }
    return { shape: [...t.shape], data: t.data }
  }
  const rows = source as readonly (readonly (number | readonly number[])[])[]
  const h = rows.length
  const w = rows[0]?.length ?? 0
  const first = rows[0]?.[0]
  const c = Array.isArray(first) ? first.length : 0
  const data: number[] = []
  for (const row of rows) {
    if (row.length !== w) throw new OmaiError('shape', 'toPixels: rows of different lengths')
    for (const cell of row) {
      if (c === 0) data.push(cell as number)
      else {
        const channels = cell as readonly number[]
        if (channels.length !== c) throw new OmaiError('shape', 'toPixels: pixels with different channel counts')
        data.push(...channels)
      }
    }
  }
  return { shape: c === 0 ? [h, w] : [h, w, c], data }
}

/**
 * Pixels from a 2-D (grey) or 3-D (RGB or RGBA) array, as `imshow` would show them. Grey values are scaled linearly
 * from `range` (default: their finite minimum and maximum) to black and white, and NaN is transparent. Colour values are
 * scaled from `range` (default `[0, 1]` when every value is at most 1, else `[0, 255]`).
 *
 * @param source - A tensor or nested arrays of shape `[h, w]`, `[h, w, 1]`, `[h, w, 3]` or `[h, w, 4]`.
 * @param options - The value range mapped to 0 to 255.
 * @returns The pixels.
 * @throws OmaiError with code `shape` for any other shape or ragged arrays.
 * @example
 * ```ts
 * import { toPixels } from 'omai-engine/scene'
 * Array.from(toPixels([[0, 1]]).rgba) // => [0, 0, 0, 255, 255, 255, 255, 255]
 * Array.from(toPixels([[[1, 0, 0]]]).rgba) // => [255, 0, 0, 255]
 * ```
 */
export function toPixels(source: ImageSource, options: { readonly range?: readonly [number, number] } = {}): Pixels {
  const { shape, data } = flattenSource(source)
  const [h = 0, w = 0, c = 1] = shape
  if (shape.length < 2 || shape.length > 3 || ![1, 3, 4].includes(c)) {
    throw new OmaiError('shape', `toPixels: need shape [h, w], [h, w, 1], [h, w, 3] or [h, w, 4], got [${shape}]`)
  }
  if (data.length !== h * w * c) throw new OmaiError('shape', 'toPixels: data does not match the shape')
  const finite: number[] = []
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number
    if (Number.isFinite(v)) finite.push(v)
  }
  const [lo, hi] = options.range ?? (c === 1 ? [minOf(finite), maxOf(finite)] : [0, maxOf(finite) <= 1 ? 1 : 255])
  const span = hi > lo ? hi - lo : 1
  const byte = (v: number): number => Math.round((255 * clamp(v - lo, 0, span)) / span)
  const rgba = new Uint8ClampedArray(h * w * 4)
  for (let p = 0; p < h * w; p++) {
    if (c === 1) {
      const v = data[p] as number
      const g = Number.isNaN(v) ? 0 : byte(v)
      rgba.set([g, g, g, Number.isNaN(v) ? 0 : 255], 4 * p)
    } else {
      for (let k = 0; k < 3; k++) rgba[4 * p + k] = byte(data[c * p + k] as number)
      rgba[4 * p + 3] = c === 4 ? byte(data[c * p + 3] as number) : 255
    }
  }
  return { width: w, height: h, rgba }
}

/**
 * A raster image, from pixels or from a 2-D (grey) or RGB(A) tensor or nested arrays (converted by {@link toPixels}).
 *
 * @param x - Left edge.
 * @param y - Top edge.
 * @param width - Drawn width.
 * @param height - Drawn height.
 * @param source - The pixels, or image data to convert.
 * @param options - Smoothing when scaled (off by default, so pixels stay crisp), id and title.
 * @returns The node.
 * @throws OmaiError `value` when the corner is not finite or the drawn width or height is not a finite number >= 0.
 * @example
 * ```ts
 * import { image } from 'omai-engine/scene'
 * image(0, 0, 10, 10, { width: 1, height: 1, rgba: new Uint8ClampedArray(4) }).width // => 10
 * image(0, 0, 20, 10, [[0, 0.5, 1]]).pixels.width // => 3
 * ```
 */
export function image(
  x: number,
  y: number,
  width: number,
  height: number,
  source: Pixels | ImageSource,
  options: NodeBase & { readonly smoothing?: boolean } = {},
): ImageNode {
  finite('image', { x, y })
  size('image', { width, height })
  const pixels = !Array.isArray(source) && 'rgba' in source ? source : toPixels(source as ImageSource)
  return { kind: 'image', x, y, width, height, pixels, ...options }
}

/**
 * A group that translates, rotates and scales its children.
 *
 * @param children - The grouped nodes.
 * @param transform - Translation, rotation in degrees, scale, opacity, id and title.
 * @returns The node.
 * @throws OmaiError `value` when the translation or rotation is not finite or the scale is not a finite number other than
 *   0.
 * @example
 * ```ts
 * import { group, circle } from 'omai-engine/scene'
 * group([circle(0, 0, 1)], { translate: [5, 5] }).translate // => [5, 5]
 * ```
 */
export function group(
  children: readonly SceneNode[],
  transform: NodeBase & Transform & { readonly opacity?: number } = {},
): GroupNode {
  checkTransform('group', transform)
  return { kind: 'group', children, ...transform }
}

/**
 * Whether a value is a {@link Scene}; the example runner uses this for `// => figure`.
 *
 * @param value - Any value.
 * @returns `true` for scenes.
 * @example
 * ```ts
 * import { isScene, scene } from 'omai-engine/scene'
 * isScene(scene(1, 1, [])) // => true
 * isScene({}) // => false
 * ```
 */
export function isScene(value: unknown): value is Scene {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'scene'
}

const recoloured = (
  node: SceneNode,
  tones: Readonly<Record<string, Paint>>,
  inherited: Paint | undefined,
): SceneNode => {
  const paint = (node.id === undefined ? undefined : tones[node.id]) ?? inherited
  if (node.kind === 'group') {
    const children = node.children.map((c) => recoloured(c, tones, paint))
    return children.some((c, i) => c !== node.children[i]) ? { ...node, children } : node
  }
  if (paint === undefined) return node
  if (node.kind === 'text' || node.kind === 'tex') return { ...node, fill: paint }
  if (node.kind === 'image' || node.kind === 'layer3d') return node
  // A shape with a fill takes the paint as its fill; a line, an open path or an outline takes it as its stroke.
  const filled = node.fill !== undefined && node.fill !== 'none'
  return filled || (node.stroke === undefined && node.kind !== 'line')
    ? { ...node, fill: paint }
    : { ...node, stroke: paint }
}

/**
 * A scene with some nodes drawn in another paint: a failed state in red, a chosen cell in the accent. A node named in
 * `tones` by its `id` takes the paint as its fill (its stroke if it has no fill, as a line has none); a group named
 * there recolours everything in it; text takes it as its text colour. The scene given is not changed.
 *
 * @param s - The scene.
 * @param tones - The paint to give each node, by node id; ids that are not in the scene are ignored.
 * @returns The recoloured scene (the same scene when nothing is named).
 * @concept scene-graph
 * @index function Redraw named nodes of a Scene in another paint.
 * @example
 * ```ts
 * import { circle, recolour, scene } from 'omai-engine/scene'
 * const s = scene(10, 10, [circle(5, 5, 2, { fill: 'tab0', id: 'dot' })])
 * const red = recolour(s, { dot: 'tab3' })
 * red.children[0]?.kind === 'circle' && red.children[0].fill // => 'tab3'
 * ```
 */
export function recolour(s: Scene, tones: Readonly<Record<string, Paint>>): Scene {
  if (Object.keys(tones).length === 0) return s
  const children = s.children.map((c) => recoloured(c, tones, undefined))
  return children.some((c, i) => c !== s.children[i]) ? { ...s, children } : s
}
