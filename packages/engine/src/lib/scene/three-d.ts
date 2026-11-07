/**
 * Small 3-D scenes as data, in world coordinates with z up: points, segments, arrows, polylines, faces and labels, with
 * constructors for each, surfaces `z = f(x, y)` as grids of faces, and labelled coordinate axes.
 */

import { MdmlError, type Vec3 } from '../base/index.ts'
import type { NodeBase, Paint, Style } from './types.ts'

/** A dot drawn at a point, `r` scene units across the projected image (default 3). */
export interface Point3D extends Style, NodeBase {
  readonly kind: 'point'
  readonly at: Vec3
  readonly r?: number
}

/** A straight segment, optionally with an arrow head at `to`. */
export interface Segment3D extends Style, NodeBase {
  readonly kind: 'segment'
  readonly from: Vec3
  readonly to: Vec3
  readonly arrow?: boolean
}

/** A polyline through points (closed into a loop when `closed`). Projected segment by segment, so it sorts in depth. */
export interface Polyline3D extends Style, NodeBase {
  readonly kind: 'polyline'
  readonly points: readonly Vec3[]
  readonly closed?: boolean
}

/** A flat face: a triangle or any planar polygon, filled and outlined like a 2-D polygon. */
export interface Face3D extends Style, NodeBase {
  readonly kind: 'face'
  readonly points: readonly Vec3[]
}

/** A text (or TeX) label anchored at a point. */
export interface Label3D extends NodeBase {
  readonly kind: 'label'
  readonly at: Vec3
  readonly text: string
  readonly tex?: boolean
  readonly size?: number
  readonly fill?: Paint
  readonly anchor?: 'start' | 'middle' | 'end'
}

/** Any item of a {@link Scene3D}. */
export type Item3D = Point3D | Segment3D | Polyline3D | Face3D | Label3D

/**
 * A small 3-D scene as data: points, segments, arrows, polylines, faces (triangles and surface patches) and labels in
 * world coordinates with z up. A camera and `project` (in `projection.ts`) turn it into an ordinary 2-D scene, so
 * vectors, rotations, surfaces and loss landscapes draw everywhere a scene does, without WebGL.
 *
 * @concept scene-graph
 */
export interface Scene3D {
  readonly kind: 'scene3d'
  readonly items: readonly Item3D[]
  readonly title?: string
}

/**
 * Build a 3-D scene.
 *
 * @param items - The items (order breaks ties in depth).
 * @param options - A title.
 * @returns The scene.
 * @concept scene-graph
 * @example
 * ```ts
 * import { scene3d, point3d } from 'mdml-engine/scene'
 * scene3d([point3d([0, 0, 0])]).items.length // => 1
 * ```
 */
export function scene3d(items: readonly Item3D[], options: { readonly title?: string } = {}): Scene3D {
  return { kind: 'scene3d', items, ...options }
}

/**
 * A point.
 *
 * @param at - The position.
 * @param style - Radius, fill, stroke, id and title.
 * @returns The item.
 * @example
 * ```ts
 * import { point3d } from 'mdml-engine/scene'
 * point3d([1, 2, 3], { fill: 'tab1' }).at // => [1, 2, 3]
 * ```
 */
export function point3d(at: Vec3, style: Style & NodeBase & { readonly r?: number } = {}): Point3D {
  return { kind: 'point', at, ...style }
}

/**
 * A segment.
 *
 * @param from - Start.
 * @param to - End.
 * @param style - Stroke, id and title.
 * @returns The item.
 * @example
 * ```ts
 * import { segment3d } from 'mdml-engine/scene'
 * segment3d([0, 0, 0], [1, 0, 0]).to // => [1, 0, 0]
 * ```
 */
export function segment3d(from: Vec3, to: Vec3, style: Style & NodeBase = {}): Segment3D {
  return { kind: 'segment', from, to, ...style }
}

/**
 * An arrow from one point to another (a vector drawn at a base point).
 *
 * @param from - The tail.
 * @param to - The head.
 * @param style - Stroke, id and title.
 * @returns The item.
 * @example
 * ```ts
 * import { arrow3d } from 'mdml-engine/scene'
 * arrow3d([0, 0, 0], [0, 0, 1]).arrow // => true
 * ```
 */
export function arrow3d(from: Vec3, to: Vec3, style: Style & NodeBase = {}): Segment3D {
  return { kind: 'segment', from, to, arrow: true, ...style }
}

/**
 * A polyline (a curve sampled at points).
 *
 * @param points - The vertices in order.
 * @param style - Stroke, closure, id and title.
 * @returns The item.
 * @example
 * ```ts
 * import { polyline3d } from 'mdml-engine/scene'
 * polyline3d([[0, 0, 0], [1, 0, 0], [1, 1, 0]], { closed: true }).closed // => true
 * ```
 */
export function polyline3d(
  points: readonly Vec3[],
  style: Style & NodeBase & { readonly closed?: boolean } = {},
): Polyline3D {
  return { kind: 'polyline', points, ...style }
}

/**
 * A triangle.
 *
 * @param a - First vertex.
 * @param b - Second vertex.
 * @param c - Third vertex.
 * @param style - Fill, stroke, id and title.
 * @returns The face.
 * @example
 * ```ts
 * import { triangle3d } from 'mdml-engine/scene'
 * triangle3d([0, 0, 0], [1, 0, 0], [0, 1, 0]).points.length // => 3
 * ```
 */
export function triangle3d(a: Vec3, b: Vec3, c: Vec3, style: Style & NodeBase = {}): Face3D {
  return { kind: 'face', points: [a, b, c], ...style }
}

/**
 * A planar polygon face (a square, a plane patch, a cube side).
 *
 * @param points - Three or more coplanar vertices in order.
 * @param style - Fill, stroke, id and title.
 * @returns The face.
 * @throws MdmlError with code `value` for fewer than three vertices.
 * @example
 * ```ts
 * import { face3d } from 'mdml-engine/scene'
 * face3d([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).kind // => 'face'
 * ```
 */
export function face3d(points: readonly Vec3[], style: Style & NodeBase = {}): Face3D {
  if (points.length < 3) throw new MdmlError('value', 'face3d: a face needs at least three vertices')
  return { kind: 'face', points, ...style }
}

/**
 * A label at a point.
 *
 * @param at - The anchor.
 * @param content - The text, or TeX source when `tex` is set.
 * @param style - TeX flag, size, fill, anchor, id and title.
 * @returns The item.
 * @example
 * ```ts
 * import { label3d } from 'mdml-engine/scene'
 * label3d([1, 0, 0], 'x').text // => 'x'
 * ```
 */
export function label3d(at: Vec3, content: string, style: Omit<Label3D, 'kind' | 'at' | 'text'> = {}): Label3D {
  return { kind: 'label', at, text: content, ...style }
}

/**
 * A surface `z = f(x, y)` over a grid, as one quadrilateral face per cell. Heights are `z[i][j] = f(x[j], y[i])`
 * (numpy `meshgrid` indexing), given as a matrix or a function.
 *
 * @param x - Grid coordinates along x (at least two).
 * @param y - Grid coordinates along y (at least two).
 * @param z - Heights as rows over `y` and columns over `x`, or the function.
 * @param style - Stroke and fill; `fill` may be a function of a cell's mean height and its row and column.
 * @returns The faces, `(y.length - 1) * (x.length - 1)` of them.
 * @throws MdmlError with code `shape` when the grid has fewer than two points per axis or `z` has the wrong shape.
 * @example
 * ```ts
 * import { surface3d } from 'mdml-engine/scene'
 * surface3d([0, 1, 2], [0, 1], (x, y) => x * y).length // => 2
 * ```
 */
export function surface3d(
  x: readonly number[],
  y: readonly number[],
  z: readonly (readonly number[])[] | ((x: number, y: number) => number),
  style: Omit<Style, 'fill'> & { readonly fill?: Paint | ((z: number, i: number, j: number) => Paint) } = {},
): Face3D[] {
  if (x.length < 2 || y.length < 2) throw new MdmlError('shape', 'surface3d: need at least two points per axis')
  const height = (i: number, j: number): number => {
    const xj = x[j] ?? 0
    const yi = y[i] ?? 0
    if (typeof z === 'function') return z(xj, yi)
    const v = z[i]?.[j]
    if (v === undefined) throw new MdmlError('shape', `surface3d: z needs ${y.length} rows of ${x.length} values`)
    return v
  }
  const { fill = 'tab0', ...rest } = style
  const faces: Face3D[] = []
  for (let i = 0; i + 1 < y.length; i++) {
    for (let j = 0; j + 1 < x.length; j++) {
      const z00 = height(i, j)
      const z01 = height(i, j + 1)
      const z11 = height(i + 1, j + 1)
      const z10 = height(i + 1, j)
      const corners: Vec3[] = [
        [x[j] ?? 0, y[i] ?? 0, z00],
        [x[j + 1] ?? 0, y[i] ?? 0, z01],
        [x[j + 1] ?? 0, y[i + 1] ?? 0, z11],
        [x[j] ?? 0, y[i + 1] ?? 0, z10],
      ]
      const mean = (z00 + z01 + z11 + z10) / 4
      faces.push({ kind: 'face', points: corners, ...rest, fill: typeof fill === 'function' ? fill(mean, i, j) : fill })
    }
  }
  return faces
}

/**
 * Coordinate axes: arrows along x, y and z from the origin, labelled.
 *
 * @param length - Arrow length (default 1).
 * @param style - Stroke for the arrows and the label size.
 * @returns Three arrows and three labels.
 * @example
 * ```ts
 * import { axes3d } from 'mdml-engine/scene'
 * axes3d(2).filter((item) => item.kind === 'label').map((item) => (item as { text: string }).text) // => ['x', 'y', 'z']
 * ```
 */
export function axes3d(length = 1, style: Style & { readonly size?: number } = {}): Item3D[] {
  const { size, ...stroke } = style
  const tips: readonly (readonly [string, Vec3])[] = [
    ['x', [length, 0, 0]],
    ['y', [0, length, 0]],
    ['z', [0, 0, length]],
  ]
  const items: Item3D[] = tips.map(([name, tip]) => arrow3d([0, 0, 0], tip, { stroke: 'muted', ...stroke, id: name }))
  for (const [name, [a, b, c]] of tips) {
    items.push(
      label3d([a * 1.1, b * 1.1, c * 1.1], name, { anchor: 'middle', ...(size === undefined ? {} : { size }) }),
    )
  }
  return items
}
