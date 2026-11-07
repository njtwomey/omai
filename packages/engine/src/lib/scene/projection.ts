/**
 * Viewing 3-D scenes: an orbit camera with matplotlib's azimuth and elevation conventions, orbiting and fitting the
 * camera to a scene, projection of points, and projection of a whole 3-D scene to a 2-D scene by the painter's
 * algorithm (farthest first), including the 3-D layers embedded in 2-D scenes.
 */

import { clamp, MdmlError, type Vec3 } from '../base/index.ts'
import { finite, size } from './_check.ts'
import { arrow, group, line, polygon, scene, tex, text } from './build.ts'
import type { Item3D, Scene3D } from './three-d.ts'
import type { Layer3DNode, NodeBase, Paint, Scene, SceneNode, Style } from './types.ts'

/**
 * How a 3-D scene is viewed: the eye orbits `target` at `distance`, at an `azimuth` (degrees about the z axis, from
 * the x axis) and an `elevation` (degrees above the xy plane), the conventions of matplotlib's `view_init`.
 * Perspective projection uses the field of view `fov` across the shorter side of the viewport; orthographic
 * projection shows `extent` world units from the centre to the nearer edge of the viewport.
 */
export interface Camera {
  readonly projection: 'perspective' | 'orthographic'
  readonly azimuth: number
  readonly elevation: number
  readonly distance: number
  readonly target: Vec3
  readonly fov: number
  readonly extent: number
}

/**
 * A camera. Defaults: perspective, azimuth -60, elevation 30 (matplotlib's default view), distance 5, target the
 * origin, a 35 degree field of view, and an orthographic extent equal to the half-height the perspective view shows
 * at the target (so switching projection keeps the framing).
 *
 * @param options - Any camera fields.
 * @returns The camera.
 * @throws MdmlError with code `value` for a distance or extent that is not finite and positive, a field of view outside
 * (0, 180), an elevation outside [-90, 90], or an azimuth, elevation or target coordinate that is not finite.
 * @example
 * ```ts
 * import { camera } from 'mdml-engine/scene'
 * camera({ projection: 'orthographic', azimuth: 0 }).elevation // => 30
 * ```
 */
export function camera(options: Partial<Camera> = {}): Camera {
  const distance = options.distance ?? 5
  const fov = options.fov ?? 35
  if (!(distance > 0 && Number.isFinite(distance))) throw new MdmlError('value', 'camera: distance must be positive')
  if (!(fov > 0 && fov < 180)) throw new MdmlError('value', 'camera: fov must be in (0, 180) degrees')
  const extent = options.extent ?? distance * Math.tan((fov * Math.PI) / 360)
  if (!(extent > 0 && Number.isFinite(extent))) throw new MdmlError('value', 'camera: extent must be positive')
  const azimuth = options.azimuth ?? -60
  const elevation = options.elevation ?? 30
  const target = options.target ?? ([0, 0, 0] as const)
  if (!Number.isFinite(azimuth)) throw new MdmlError('value', `camera: azimuth must be finite, got ${azimuth}`)
  if (!(elevation >= -90 && elevation <= 90))
    throw new MdmlError('value', `camera: elevation must be in [-90, 90] degrees, got ${elevation}`)
  if (!target.every(Number.isFinite)) throw new MdmlError('value', `camera: target must be finite, got [${target}]`)
  return {
    projection: options.projection ?? 'perspective',
    azimuth,
    elevation,
    distance,
    target,
    fov,
    extent,
  }
}

/**
 * Turn a camera about its target (dragging in a 3-D view). Elevation is clamped to [-90, 90].
 *
 * @param cam - The camera.
 * @param dAzimuth - Change of azimuth in degrees.
 * @param dElevation - Change of elevation in degrees.
 * @returns The moved camera.
 * @example
 * ```ts
 * import { camera, orbit } from 'mdml-engine/scene'
 * orbit(camera({ elevation: 80 }), 15, 30).elevation // => 90
 * ```
 */
export function orbit(cam: Camera, dAzimuth: number, dElevation: number): Camera {
  return {
    ...cam,
    azimuth: cam.azimuth + dAzimuth,
    elevation: clamp(cam.elevation + dElevation, -90, 90),
  }
}

function itemPoints(item: Item3D): readonly Vec3[] {
  switch (item.kind) {
    case 'point':
    case 'label':
      return [item.at]
    case 'segment':
      return [item.from, item.to]
    case 'polyline':
    case 'face':
      return item.points
  }
}

/**
 * Aim a camera at a 3-D scene: target the centre of its bounding box, and set the distance and extent so that the
 * bounding sphere fills the view with a margin.
 *
 * @param cam - The camera (its projection, angles and field of view are kept).
 * @param s - The scene.
 * @param margin - Extra room as a factor (default 1.1).
 * @returns The fitted camera (unchanged for an empty scene or one without a finite point; for a scene of one point only
 *   the target moves, the camera keeps its distance and extent).
 * @example
 * ```ts
 * import { camera, fitCamera, scene3d, point3d } from 'mdml-engine/scene'
 * fitCamera(camera(), scene3d([point3d([0, 0, 0]), point3d([2, 2, 2])])).target // => [1, 1, 1]
 * ```
 */
export function fitCamera(cam: Camera, s: Scene3D, margin = 1.1): Camera {
  const lo = [Infinity, Infinity, Infinity]
  const hi = [-Infinity, -Infinity, -Infinity]
  for (const item of s.items) {
    for (const p of itemPoints(item)) {
      for (let k = 0; k < 3; k++) {
        const v = p[k] as number
        if (!Number.isFinite(v)) continue
        if (v < (lo[k] as number)) lo[k] = v
        if (v > (hi[k] as number)) hi[k] = v
      }
    }
  }
  if (!Number.isFinite(lo[0] as number)) return cam
  const mid = (k: number): number => ((lo[k] as number) + (hi[k] as number)) / 2
  const half = (k: number): number => ((hi[k] as number) - (lo[k] as number)) / 2
  const radius = Math.hypot(half(0), half(1), half(2))
  // One point (or coincident ones) has no size to frame: aim at it and keep the camera's own distance and extent.
  if (!(radius > 0)) return { ...cam, target: [mid(0), mid(1), mid(2)] }
  const extent = margin * radius
  // The sphere fits the perspective view when its angular radius is at most half the field of view.
  const distance = extent / Math.sin((cam.fov * Math.PI) / 360)
  return { ...cam, target: [mid(0), mid(1), mid(2)], extent, distance }
}

/** A point on the image plane: scene coordinates and the depth along the view direction (larger is farther). */
export interface Projected {
  readonly x: number
  readonly y: number
  readonly depth: number
}

/**
 * Project one world point into a viewport of the given size, centred on the camera target. Returns `undefined` for a
 * point at or behind the eye under perspective projection.
 *
 * @param cam - The camera.
 * @param p - The world point.
 * @param viewport - The width and height of the image.
 * @returns Image coordinates (y down) and depth, or `undefined`.
 * @example
 * ```ts
 * import { camera, projectPoint } from 'mdml-engine/scene'
 * projectPoint(camera(), [0, 0, 0], { width: 100, height: 100 }) // => { x: 50, y: 50, depth: 5 }
 * ```
 */
export function projectPoint(
  cam: Camera,
  p: Vec3,
  viewport: { readonly width: number; readonly height: number },
): Projected | undefined {
  const q = viewSpace(cam, p)
  const depth = q.depth
  const half = Math.min(viewport.width, viewport.height) / 2
  let k: number
  if (cam.projection === 'perspective') {
    if (depth <= 1e-9 * cam.distance) return undefined
    k = half / (depth * Math.tan((cam.fov * Math.PI) / 360))
  } else {
    k = half / cam.extent
  }
  const x = viewport.width / 2 + k * q.xc
  const y = viewport.height / 2 - k * q.yc
  // A non-finite coordinate (NaN, Infinity) has no image: it is dropped like a point behind the eye.
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(depth) ? { x, y, depth } : undefined
}

/** A world point in the camera's frame: right, up and the depth along the view direction (larger is farther). */
function viewSpace(cam: Camera, p: Vec3): { readonly xc: number; readonly yc: number; readonly depth: number } {
  const az = (cam.azimuth * Math.PI) / 180
  const el = (cam.elevation * Math.PI) / 180
  const ca = Math.cos(az)
  const sa = Math.sin(az)
  const ce = Math.cos(el)
  const se = Math.sin(el)
  // Camera axes: eye direction e = (ce ca, ce sa, se) from the target, right r = (-sa, ca, 0), up u = r x (-e).
  const dx = p[0] - cam.target[0]
  const dy = p[1] - cam.target[1]
  const dz = p[2] - cam.target[2]
  const xc = -sa * dx + ca * dy
  const yc = -se * ca * dx - se * sa * dy + ce * dz
  const zc = ce * ca * dx + ce * sa * dy + se * dz
  return { xc, yc, depth: cam.distance - zc }
}

/** The view a projection reads: the camera's image of a point, its depth, and the near plane that clips (perspective). */
interface View {
  readonly at: (p: Vec3) => Projected | undefined
  readonly depth: (p: Vec3) => number
  readonly perspective: boolean
  readonly near: number
}

/**
 * A segment that starts or ends behind the eye, cut at the near plane (1% of the camera distance), with whether its end
 * `b` is still the original one (an arrow head belongs there); undefined if none of it is in front.
 */
function clipSegment(view: View, a: Vec3, b: Vec3): readonly [Vec3, Vec3, boolean] | undefined {
  const da = view.depth(a)
  const db = view.depth(b)
  if (!(Number.isFinite(da) && Number.isFinite(db)) || (da < view.near && db < view.near)) return undefined
  const cut = (from: Vec3, to: Vec3, df: number, dt: number): Vec3 => {
    const t = (df - view.near) / (df - dt)
    return [from[0] + t * (to[0] - from[0]), from[1] + t * (to[1] - from[1]), from[2] + t * (to[2] - from[2])]
  }
  return da >= view.near ? [a, cut(a, b, da, db), false] : [cut(b, a, db, da), b, true]
}

/** A polygon cut at the near plane (Sutherland-Hodgman); fewer than three vertices left is nothing to draw. */
function clipPolygon(view: View, points: readonly Vec3[]): readonly Vec3[] {
  const depths = points.map((p) => view.depth(p))
  if (!depths.every(Number.isFinite)) return []
  const out: Vec3[] = []
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length
    const [p, q] = [points[i] as Vec3, points[j] as Vec3]
    const [dp, dq] = [depths[i] as number, depths[j] as number]
    if (dp >= view.near) out.push(p)
    if (dp >= view.near !== dq >= view.near) {
      const t = (dp - view.near) / (dp - dq)
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])])
    }
  }
  return out.length < 3 ? [] : out
}

interface Drawn {
  readonly depth: number
  readonly order: number
  readonly node: SceneNode
  readonly label: boolean
}

function projectItem(item: Item3D, order: number, view: View, cull: boolean, out: Drawn[]): void {
  const at = view.at
  const segment = (a: Vec3, b: Vec3, style: Style & NodeBase, head: boolean): void => {
    let [pa, pb, tip] = [at(a), at(b), head]
    if (pa === undefined || pb === undefined) {
      // Under perspective the part of the segment in front of the eye is still drawn, cut at the near plane.
      const cut = view.perspective ? clipSegment(view, a, b) : undefined
      if (cut === undefined) return
      ;[pa, pb, tip] = [at(cut[0]), at(cut[1]), head && cut[2]]
      if (pa === undefined || pb === undefined) return
    }
    const node = (tip ? arrow : line)(pa.x, pa.y, pb.x, pb.y, style)
    out.push({ depth: (pa.depth + pb.depth) / 2, order, node, label: false })
  }
  switch (item.kind) {
    case 'point': {
      const p = at(item.at)
      if (p === undefined) return
      const { kind: _, at: __, r = 3, ...style } = item
      out.push({ depth: p.depth, order, node: { kind: 'circle', cx: p.x, cy: p.y, r, ...style }, label: false })
      return
    }
    case 'segment': {
      const { kind: _, from, to, arrow: head = false, ...style } = item
      segment(from, to, style, head)
      return
    }
    case 'polyline': {
      const { kind: _, points, closed = false, ...style } = item
      const n = points.length
      const last = closed ? n : n - 1
      for (let i = 0; i < last; i++) segment(points[i] as Vec3, points[(i + 1) % n] as Vec3, style, false)
      return
    }
    case 'face': {
      const { kind: _, points, ...style } = item
      const pts: Projected[] = []
      let vertices = points
      if (view.perspective && points.some((q) => at(q) === undefined)) {
        vertices = clipPolygon(view, points)
        if (vertices.length === 0) return
      }
      for (const q of vertices) {
        const p = at(q)
        if (p === undefined) return
        pts.push(p)
      }
      if (cull) {
        let area = 0
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i] as Projected
          const q = pts[(i + 1) % pts.length] as Projected
          area += p.x * q.y - q.x * p.y
        }
        // y grows downwards, so a face wound counter-clockwise as seen from the eye has negative area here.
        if (area > 0) return
      }
      let depth = 0
      for (const p of pts) depth += p.depth
      const node = polygon(
        pts.map((p) => [p.x, p.y] as const),
        style,
      )
      out.push({ depth: depth / pts.length, order, node, label: false })
      return
    }
    case 'label': {
      const p = at(item.at)
      if (p === undefined) return
      const { kind: _, at: __, text: content, tex: isTex = false, ...style } = item
      const node = isTex ? tex(p.x, p.y, content, style) : text(p.x, p.y, content, { baseline: 'middle', ...style })
      out.push({ depth: p.depth, order, node, label: true })
      return
    }
  }
}

/**
 * Project a 3-D scene to a 2-D {@link Scene} with the painter's algorithm: every point, segment, polyline segment and
 * face becomes a 2-D node, and nodes are drawn from the farthest (by mean vertex depth) to the nearest, ties in item
 * order. Labels are drawn last unless `labelsOnTop` is false. Under perspective, a segment or face reaching behind the
 * eye is cut at a near plane (1% of the camera distance) and its visible part drawn, and a point or label behind the
 * eye, or with a non-finite coordinate, is dropped. With `cull`, faces wound clockwise as seen from the eye (their backs) are dropped.
 *
 * @param s - The 3-D scene.
 * @param cam - The camera.
 * @param options - Image size (default 400 by 400), background, title, culling and label placement.
 * @returns The 2-D scene.
 * @concept scene-graph
 * @example
 * ```ts
 * import { camera, project, scene3d, segment3d } from 'mdml-engine/scene'
 * const view = camera({ azimuth: -90, elevation: 0 })
 * project(scene3d([segment3d([0, 0, 0], [1, 0, 0])]), view, { width: 100, height: 100 }).children[0].kind // => 'line'
 * ```
 */
export function project(
  s: Scene3D,
  cam: Camera,
  options: {
    readonly width?: number
    readonly height?: number
    readonly background?: Paint
    readonly title?: string
    readonly cull?: boolean
    readonly labelsOnTop?: boolean
  } = {},
): Scene {
  const { width = 400, height = 400, cull = false, labelsOnTop = true } = options
  const viewport = { width, height }
  const view: View = {
    at: (p) => projectPoint(cam, p, viewport),
    depth: (p) => viewSpace(cam, p).depth,
    perspective: cam.projection === 'perspective',
    near: 0.01 * cam.distance,
  }
  const drawn: Drawn[] = []
  for (let order = 0; order < s.items.length; order++) projectItem(s.items[order] as Item3D, order, view, cull, drawn)
  const rank = (d: Drawn): number => (labelsOnTop && d.label ? 1 : 0)
  drawn.sort((a, b) => rank(a) - rank(b) || b.depth - a.depth || a.order - b.order)
  const title = options.title ?? s.title
  return scene(
    width,
    height,
    drawn.map((d) => d.node),
    {
      ...(options.background === undefined ? {} : { background: options.background }),
      ...(title === undefined ? {} : { title }),
    },
  )
}

/**
 * A 3-D layer node: a 3-D scene seen through a camera, drawn into a box of a 2-D scene.
 *
 * @param s - The 3-D scene.
 * @param cam - The camera.
 * @param box - Where the layer sits in the enclosing scene.
 * @param options - Id and title.
 * @returns The node.
 * @throws MdmlError `value` when the box corner is not finite or its width or height is not a finite number >= 0.
 * @example
 * ```ts
 * import { camera, layer3d, scene3d } from 'mdml-engine/scene'
 * layer3d(scene3d([]), camera(), { x: 0, y: 0, width: 200, height: 200 }).kind // => 'layer3d'
 * ```
 */
export function layer3d(
  s: Scene3D,
  cam: Camera,
  box: { readonly x: number; readonly y: number; readonly width: number; readonly height: number },
  options: NodeBase = {},
): Layer3DNode {
  finite('layer3d', { x: box.x, y: box.y })
  size('layer3d', { width: box.width, height: box.height })
  return { kind: 'layer3d', ...box, scene: s, camera: cam, ...options }
}

function flattenLayers(nodes: readonly SceneNode[]): SceneNode[] {
  return nodes.map((node) => {
    if (node.kind === 'group') return { ...node, children: flattenLayers(node.children) }
    if (node.kind !== 'layer3d') return node
    const flat = project(node.scene, node.camera, { width: node.width, height: node.height })
    return group(flat.children, {
      translate: [node.x, node.y],
      ...(node.id === undefined ? {} : { id: node.id }),
      ...(node.title === undefined ? {} : { title: node.title }),
    })
  })
}

/**
 * Replace every 3-D layer of a scene (inside groups too) by its projection, leaving a purely 2-D scene. Renderers call
 * this before drawing.
 *
 * @param s - The scene.
 * @returns The scene without 3-D layers.
 * @example
 * ```ts
 * import { camera, layer3d, point3d, projectLayers, scene, scene3d } from 'mdml-engine/scene'
 * const layer = layer3d(scene3d([point3d([0, 0, 0])]), camera(), { x: 10, y: 10, width: 80, height: 80 })
 * projectLayers(scene(100, 100, [layer])).children[0].kind // => 'group'
 * ```
 */
export function projectLayers(s: Scene): Scene {
  return { ...s, children: flattenLayers(s.children) }
}
