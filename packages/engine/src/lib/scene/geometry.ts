/**
 * Geometry of scene nodes: bounding boxes, composing, applying and inverting the similarity transforms of groups, and
 * layout helpers that translate, rotate, scale and fit nodes into a box or tile several scenes into a grid.
 */

import { MdmlError } from '../base/index.ts'
import { group } from './build.ts'
import type { GroupNode, Scene, SceneNode, Transform } from './types.ts'

/** An axis-aligned bounding box. */
export interface Bounds {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

function corners(x0: number, y0: number, x1: number, y1: number): (readonly [number, number])[] {
  return [
    [x0, y0],
    [x1, y0],
    [x0, y1],
    [x1, y1],
  ]
}

// Points whose bounding box encloses the node: exact for unrotated nodes and polylines, a safe enclosure for rotated
// circles and ellipses (their box corners are rotated).
function nodePoints(node: SceneNode): (readonly [number, number])[] {
  switch (node.kind) {
    case 'rect':
    case 'image':
    case 'layer3d':
      return corners(node.x, node.y, node.x + node.width, node.y + node.height)
    case 'circle':
      return corners(node.cx - node.r, node.cy - node.r, node.cx + node.r, node.cy + node.r)
    case 'ellipse':
      return corners(node.cx - node.rx, node.cy - node.ry, node.cx + node.rx, node.cy + node.ry)
    case 'line':
      return [
        [node.x1, node.y1],
        [node.x2, node.y2],
      ]
    case 'path':
      return [...node.points]
    case 'text':
    case 'tex':
      return [[node.x, node.y]]
    case 'group':
      return node.children.flatMap(nodePoints).map((p) => applyTransform(node, p))
  }
}

/**
 * The bounding box of nodes (text is counted by its anchor only; rotated circles and ellipses by their rotated box).
 *
 * @param nodes - The nodes.
 * @returns The box, or a zero box for no nodes.
 * @example
 * ```ts
 * import { bounds, rect, circle } from 'mdml-engine/scene'
 * bounds([rect(0, 0, 2, 1), circle(5, 5, 1)]) // => { x: 0, y: 0, width: 6, height: 6 }
 * ```
 */
export function bounds(nodes: readonly SceneNode[]): Bounds {
  const points = nodes.flatMap(nodePoints)
  if (points.length === 0) return { x: 0, y: 0, width: 0, height: 0 }
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [x, y] of points) {
    if (x < x0) x0 = x
    if (y < y0) y0 = y
    if (x > x1) x1 = x
    if (y > y1) y1 = y
  }
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }
}

/**
 * Compose two transforms: the result applies `inner` first, then `outer` (as a group inside a group does).
 *
 * @param outer - The transform applied second.
 * @param inner - The transform applied first.
 * @returns The combined transform, with all three fields set.
 * @example
 * ```ts
 * import { composeTransforms } from 'mdml-engine/scene'
 * composeTransforms({ translate: [1, 0] }, { scale: 2 }) // => { translate: [1, 0], rotate: 0, scale: 2 }
 * composeTransforms({ scale: 2 }, { translate: [1, 0] }).translate // => [2, 0]
 * ```
 */
export function composeTransforms(outer: Transform, inner: Transform): Required<Transform> {
  const [tx, ty] = applyTransform(outer, inner.translate ?? [0, 0])
  return {
    translate: [tx, ty],
    rotate: (outer.rotate ?? 0) + (inner.rotate ?? 0),
    scale: (outer.scale ?? 1) * (inner.scale ?? 1),
  }
}

/**
 * Apply a transform to a point.
 *
 * @param t - The transform.
 * @param point - The point `[x, y]`.
 * @returns The transformed point.
 * @example
 * ```ts
 * import { applyTransform } from 'mdml-engine/scene'
 * applyTransform({ translate: [10, 0], scale: 2 }, [1, 1]) // => [12, 2]
 * ```
 */
export function applyTransform(t: Transform, point: readonly [number, number]): [number, number] {
  const s = t.scale ?? 1
  const [tx, ty] = t.translate ?? [0, 0]
  const [x, y] = point
  const degrees = t.rotate ?? 0
  if (degrees === 0) return [tx + s * x, ty + s * y]
  const theta = (degrees * Math.PI) / 180
  const c = Math.cos(theta)
  const sn = Math.sin(theta)
  return [tx + s * (c * x - sn * y), ty + s * (sn * x + c * y)]
}

/**
 * The inverse of a transform: `applyTransform(invertTransform(t), applyTransform(t, p))` is `p`.
 *
 * @param t - The transform.
 * @returns The inverse transform, with all three fields set.
 * @throws MdmlError with code `singular` when the scale is 0.
 * @example
 * ```ts
 * import { invertTransform } from 'mdml-engine/scene'
 * invertTransform({ translate: [4, 2], scale: 2 }) // => { translate: [-2, -1], rotate: 0, scale: 0.5 }
 * ```
 */
export function invertTransform(t: Transform): Required<Transform> {
  const s = t.scale ?? 1
  if (s === 0) throw new MdmlError('singular', 'invertTransform: a zero scale has no inverse')
  const inverse = { rotate: 0 - (t.rotate ?? 0), scale: 1 / s }
  const [tx, ty] = applyTransform(inverse, t.translate ?? [0, 0])
  return { translate: [0 - tx, 0 - ty], ...inverse }
}

function asNodes(target: SceneNode | readonly SceneNode[]): readonly SceneNode[] {
  return Array.isArray(target) ? (target as readonly SceneNode[]) : [target as SceneNode]
}

function transformed(target: SceneNode | readonly SceneNode[], t: Transform): GroupNode {
  const nodes = asNodes(target)
  const only = nodes.length === 1 && !Array.isArray(target) ? nodes[0] : undefined
  if (only?.kind === 'group') return { ...only, ...composeTransforms(t, only) }
  return group(nodes, t)
}

/**
 * Move nodes. A single group absorbs the translation into its own transform; anything else is wrapped in a new group.
 *
 * @param target - A node or nodes.
 * @param dx - Horizontal offset.
 * @param dy - Vertical offset.
 * @returns The moved group.
 * @example
 * ```ts
 * import { translate, bounds, rect } from 'mdml-engine/scene'
 * bounds([translate(rect(0, 0, 2, 1), 5, 5)]) // => { x: 5, y: 5, width: 2, height: 1 }
 * ```
 */
export function translate(target: SceneNode | readonly SceneNode[], dx: number, dy: number): GroupNode {
  return transformed(target, { translate: [dx, dy] })
}

/**
 * Rotate nodes about a point (clockwise on screen for positive angles, since y grows downwards).
 *
 * @param target - A node or nodes.
 * @param degrees - The angle in degrees.
 * @param about - The centre of rotation (default the origin).
 * @returns The rotated group.
 * @example
 * ```ts
 * import { rotate, bounds, line } from 'mdml-engine/scene'
 * const b = bounds([rotate(line(0, 0, 2, 0), 90)])
 * [Math.round(b.width), Math.round(b.height)] // => [0, 2]
 * ```
 */
export function rotate(
  target: SceneNode | readonly SceneNode[],
  degrees: number,
  about: readonly [number, number] = [0, 0],
): GroupNode {
  const [cx, cy] = about
  const [rx, ry] = applyTransform({ rotate: degrees }, about)
  return transformed(target, { translate: [cx - rx, cy - ry], rotate: degrees })
}

/**
 * Scale nodes uniformly about the top-left corner of their bounds so that they are at most `width` wide and `height`
 * high (either may be omitted).
 *
 * @param target - A node or nodes.
 * @param size - The target width, height or both.
 * @returns The scaled group.
 * @throws MdmlError with code `value` when neither size is given.
 * @example
 * ```ts
 * import { scaleTo, bounds, rect } from 'mdml-engine/scene'
 * bounds([scaleTo(rect(1, 1, 4, 2), { width: 2 })]) // => { x: 1, y: 1, width: 2, height: 1 }
 * ```
 */
export function scaleTo(
  target: SceneNode | readonly SceneNode[],
  size: { readonly width?: number; readonly height?: number },
): GroupNode {
  if (size.width === undefined && size.height === undefined) {
    throw new MdmlError('value', 'scaleTo: give a width, a height or both')
  }
  const b = bounds(asNodes(target))
  const ratio = (want: number | undefined, have: number): number =>
    want === undefined || have === 0 ? Infinity : want / have
  const r = Math.min(ratio(size.width, b.width), ratio(size.height, b.height))
  const s = Number.isFinite(r) ? r : 1
  return transformed(target, { translate: [b.x * (1 - s), b.y * (1 - s)], scale: s })
}

/**
 * Scale and translate nodes to fit a box, keeping their aspect ratio and centring them.
 *
 * @param nodes - The nodes.
 * @param box - The target box.
 * @param padding - Margin inside the box.
 * @returns A group with the fitted nodes.
 * @example
 * ```ts
 * import { fit, rect } from 'mdml-engine/scene'
 * fit([rect(0, 0, 1, 1)], { x: 0, y: 0, width: 10, height: 10 }).scale // => 10
 * ```
 */
export function fit(nodes: readonly SceneNode[], box: Bounds, padding = 0): GroupNode {
  const b = bounds(nodes)
  const w = box.width - 2 * padding
  const h = box.height - 2 * padding
  const s =
    b.width === 0 && b.height === 0
      ? 1
      : Math.min(b.width === 0 ? Infinity : w / b.width, b.height === 0 ? Infinity : h / b.height)
  const tx = box.x + padding + (w - s * b.width) / 2 - s * b.x
  const ty = box.y + padding + (h - s * b.height) / 2 - s * b.y
  return group(nodes, { translate: [tx, ty], scale: s })
}

/**
 * Tile scenes into a grid (a filmstrip), each scaled to fit its cell.
 *
 * @param scenes - The scenes.
 * @param columns - Cells per row.
 * @param cell - Cell width and height.
 * @returns One scene containing the grid.
 * @throws MdmlError `value` when `columns` is not an integer >= 1 or the cell is not finite and positive.
 * @example
 * ```ts
 * import { grid, scene } from 'mdml-engine/scene'
 * grid([scene(10, 10, []), scene(10, 10, [])], 2, [20, 20]).width // => 40
 * ```
 */
export function grid(scenes: readonly Scene[], columns: number, cell: readonly [number, number]): Scene {
  const [cw, ch] = cell
  if (!(Number.isInteger(columns) && columns >= 1))
    throw new MdmlError('value', `grid: columns is ${columns}, expected an integer >= 1`)
  if (!(cw > 0 && ch > 0 && Number.isFinite(cw) && Number.isFinite(ch)))
    throw new MdmlError('value', `grid: the cell is ${cw} by ${ch}, expected finite positive sizes`)
  const rows = Math.ceil(scenes.length / columns)
  const children = scenes.map((s, i) => {
    const scale = Math.min(cw / s.width, ch / s.height)
    return group(s.children, { translate: [(i % columns) * cw, Math.floor(i / columns) * ch], scale })
  })
  return { kind: 'scene', width: columns * cw, height: rows * ch, children }
}
