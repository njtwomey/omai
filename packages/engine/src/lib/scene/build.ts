/**
 * Constructors for scenes, the engine's drawings as data: the scene itself, shapes (rectangles, circles, ellipses,
 * lines, arrows, paths, polygons), text and TeX labels, raster images from tensors or nested arrays (with `toPixels`
 * converting them as `imshow` would), and transforming groups.
 */

import { clamp, MdmlError, maxOf, minOf } from '../base/index.ts'
import { finite, size } from './_check.ts'
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
 * @throws MdmlError `value` when the width or height is not a finite number >= 0.
 * @example
 * ```ts
 * import { scene, circle } from 'mdml-engine/scene'
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
 * @throws MdmlError `value` when a coordinate is not finite or the width or height is not a finite number >= 0.
 * @example
 * ```ts
 * import { rect } from 'mdml-engine/scene'
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
 * @throws MdmlError `value` when the centre is not finite or the radius is not a finite number >= 0.
 * @example
 * ```ts
 * import { circle } from 'mdml-engine/scene'
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
 * @throws MdmlError `value` when the centre is not finite or a radius is not a finite number >= 0.
 * @example
 * ```ts
 * import { ellipse } from 'mdml-engine/scene'
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
 * @throws MdmlError `value` when a coordinate is not finite.
 * @example
 * ```ts
 * import { line } from 'mdml-engine/scene'
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
 * @throws MdmlError `value` when a coordinate is not finite.
 * @example
 * ```ts
 * import { arrow } from 'mdml-engine/scene'
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
 * @example
 * ```ts
 * import { path } from 'mdml-engine/scene'
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
 * import { polygon } from 'mdml-engine/scene'
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
 * import { text } from 'mdml-engine/scene'
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
 * import { tex } from 'mdml-engine/scene'
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
    if (row.length !== w) throw new MdmlError('shape', 'toPixels: rows of different lengths')
    for (const cell of row) {
      if (c === 0) data.push(cell as number)
      else {
        const channels = cell as readonly number[]
        if (channels.length !== c) throw new MdmlError('shape', 'toPixels: pixels with different channel counts')
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
 * @throws MdmlError with code `shape` for any other shape or ragged arrays.
 * @example
 * ```ts
 * import { toPixels } from 'mdml-engine/scene'
 * Array.from(toPixels([[0, 1]]).rgba) // => [0, 0, 0, 255, 255, 255, 255, 255]
 * Array.from(toPixels([[[1, 0, 0]]]).rgba) // => [255, 0, 0, 255]
 * ```
 */
export function toPixels(source: ImageSource, options: { readonly range?: readonly [number, number] } = {}): Pixels {
  const { shape, data } = flattenSource(source)
  const [h = 0, w = 0, c = 1] = shape
  if (shape.length < 2 || shape.length > 3 || ![1, 3, 4].includes(c)) {
    throw new MdmlError('shape', `toPixels: need shape [h, w], [h, w, 1], [h, w, 3] or [h, w, 4], got [${shape}]`)
  }
  if (data.length !== h * w * c) throw new MdmlError('shape', 'toPixels: data does not match the shape')
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
 * @example
 * ```ts
 * import { image } from 'mdml-engine/scene'
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
  const pixels = !Array.isArray(source) && 'rgba' in source ? source : toPixels(source as ImageSource)
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
 * import { group, circle } from 'mdml-engine/scene'
 * group([circle(0, 0, 1)], { translate: [5, 5] }).translate // => [5, 5]
 * ```
 */
export function group(
  children: readonly SceneNode[],
  transform: NodeBase & Transform & { readonly opacity?: number } = {},
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
 * import { isScene, scene } from 'mdml-engine/scene'
 * isScene(scene(1, 1, [])) // => true
 * isScene({}) // => false
 * ```
 */
export function isScene(value: unknown): value is Scene {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'scene'
}
