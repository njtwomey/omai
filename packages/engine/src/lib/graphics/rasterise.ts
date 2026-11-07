/**
 * Triangle rasterisation, the pipeline GPUs use: transform vertices to clip space, cull back faces, clip against the
 * view volume, cover pixels with edge functions under the top-left rule, interpolate attributes and keep the nearest
 * fragment in a depth buffer. The stages are exported separately, and `rasteriseMesh` runs them all on a mesh with a
 * record of what happened to each triangle.
 */

import { MdmlError } from '../base/index.ts'
import type { Point2 as PlanePoint2 } from '../numerics/geometry/index.ts'
import type { Vec3 } from '../scene/index.ts'
import { check3, cross3, dot3, normalise3, sub3 } from './_vec3.ts'
import { cameraBasis, type RenderCamera, transformPoint, viewProjection } from './camera.ts'
import { type Light, shadePoints } from './lights.ts'
import type { Material } from './shading.ts'
import { type FloatImage, floatImage, sampleBilinear } from './textures.ts'

// Triangle rasterisation, the counterpart of ray casting: instead of a ray per pixel finding its nearest surface, each
// triangle finds the pixels it covers, and a depth buffer keeps the nearest. Screen coordinates have y down and pixel
// (x, y) has its centre at (x + 0.5, y + 0.5), as `camera.ts` projects.

/** A point on the screen: the plane point of `numerics/geometry`. */
export type Point2 = PlanePoint2

/**
 * The edge function `E(a, b, p) = (b_x - a_x)(p_y - a_y) - (b_y - a_y)(p_x - a_x)`: twice the signed area of the
 * triangle `a, b, p`, positive when `p` lies to the right of the edge `a -> b` on a y-down screen.
 *
 * @param a - Edge start.
 * @param b - Edge end.
 * @param p - The point.
 * @returns The edge function.
 * @concept rasterisation
 * @example
 * ```ts
 * import { edgeFunction } from 'mdml-engine/graphics'
 * [edgeFunction([0, 0], [1, 0], [0, 1]), edgeFunction([0, 0], [1, 0], [0, -1])] // => [1, -1]
 * ```
 */
export function edgeFunction(a: Point2, b: Point2, p: Point2): number {
  return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/** A pixel a triangle covers, with the screen-space barycentric weights of its centre. */
export interface Fragment {
  readonly x: number
  readonly y: number
  readonly weights: readonly [number, number, number]
}

// The top-left rule for a triangle wound with positive area (clockwise on a y-down screen): an edge running right
// along the top, or upwards on the left, owns the pixel centres that lie exactly on it.
const topLeft = (a: Point2, b: Point2): boolean => (b[1] === a[1] && b[0] > a[0]) || b[1] < a[1]

/** Subpixel precision of the rasteriser: vertices snap to 1/256 of a pixel, as GPUs snap them to 8 fractional bits. */
export const SUBPIXEL = 256

/**
 * The pixels a triangle covers: those whose centres lie inside it (by the three edge functions, within the bounding
 * box clipped to the screen), with centres exactly on an edge assigned by the top-left rule, so triangles sharing an
 * edge cover each pixel once. Vertices are first snapped to {@link SUBPIXEL} steps, which makes every edge function
 * an exact integer computation, so the rule holds without rounding error. Either winding is accepted; degenerate
 * triangles cover nothing. Weights are the barycentric coordinates of the pixel centre in the given (unsnapped)
 * triangle (`w0` of vertex `a`), so interpolation is exact; within a 512th of a pixel of an edge one may be slightly
 * negative.
 *
 * @param a - First vertex in screen coordinates.
 * @param b - Second vertex.
 * @param c - Third vertex.
 * @param width - Screen width in pixels.
 * @param height - Screen height in pixels.
 * @returns The fragments, row by row.
 * @concept rasterisation
 * @example
 * ```ts
 * import { rasteriseTriangle } from 'mdml-engine/graphics'
 * rasteriseTriangle([0, 0], [4, 0], [0, 4], 4, 4).map((f) => [f.x, f.y]) // => [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [0, 2]]
 * ```
 */
export function rasteriseTriangle(a: Point2, b: Point2, c: Point2, width: number, height: number): Fragment[] {
  if (![a, b, c].every((v) => Number.isFinite(v[0]) && Number.isFinite(v[1]))) return []
  const snap = (v: Point2): Point2 => [Math.round(v[0] * SUBPIXEL), Math.round(v[1] * SUBPIXEL)]
  let [p, q, r] = [snap(a), snap(b), snap(c)]
  const area = edgeFunction(p, q, r)
  if (area < 0) [q, r] = [r, q]
  if (area === 0) return []
  const exact = edgeFunction(a, b, c)
  const x0 = Math.max(0, Math.floor(Math.min(p[0], q[0], r[0]) / SUBPIXEL))
  const x1 = Math.min(width - 1, Math.ceil(Math.max(p[0], q[0], r[0]) / SUBPIXEL))
  const y0 = Math.max(0, Math.floor(Math.min(p[1], q[1], r[1]) / SUBPIXEL))
  const y1 = Math.min(height - 1, Math.ceil(Math.max(p[1], q[1], r[1]) / SUBPIXEL))
  const tlQR = topLeft(q, r)
  const tlRP = topLeft(r, p)
  const tlPQ = topLeft(p, q)
  const inside = (e: number, tl: boolean): boolean => e > 0 || (e === 0 && tl)
  const out: Fragment[] = []
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const centre: Point2 = [(x + 0.5) * SUBPIXEL, (y + 0.5) * SUBPIXEL]
      const e0 = edgeFunction(q, r, centre)
      const e1 = edgeFunction(r, p, centre)
      const e2 = edgeFunction(p, q, centre)
      if (!(inside(e0, tlQR) && inside(e1, tlRP) && inside(e2, tlPQ))) continue
      const at: Point2 = [x + 0.5, y + 0.5]
      out.push({
        x,
        y,
        weights: [edgeFunction(b, c, at) / exact, edgeFunction(c, a, at) / exact, edgeFunction(a, b, at) / exact],
      })
    }
  }
  return out
}

/** A vertex in clip space with attributes to interpolate. */
export interface ClipVertex {
  readonly clip: readonly [number, number, number, number]
  readonly attributes: readonly number[]
}

/**
 * Clip a convex polygon in homogeneous clip space against the view volume `-w <= x, y, z <= w` (Sutherland and
 * Hodgman, one plane at a time), interpolating attributes linearly, which is exact in clip space. Clipping before the
 * division by `w` is what keeps triangles that cross the eye plane from wrapping around the screen.
 *
 * @param polygon - The vertices, in order.
 * @returns The clipped polygon (empty when wholly outside).
 * @concept clipping
 * @example
 * ```ts
 * import { clipPolygon } from 'mdml-engine/graphics'
 * const tri = [{ clip: [0, 0, 0, 1], attributes: [0] }, { clip: [2, 0, 0, 1], attributes: [1] }, { clip: [0, 0.5, 0, 1], attributes: [0] }]
 * clipPolygon(tri).map((v) => v.clip[0]) // => [0, 1, 1, 0]
 * ```
 */
export function clipPolygon(polygon: readonly ClipVertex[]): ClipVertex[] {
  // Each plane as a signed distance (inside when >= 0): w + x, w - x, w + y, w - y, w + z, w - z.
  const planes: ((c: readonly number[]) => number)[] = [
    (c) => (c[3] ?? 0) + (c[0] ?? 0),
    (c) => (c[3] ?? 0) - (c[0] ?? 0),
    (c) => (c[3] ?? 0) + (c[1] ?? 0),
    (c) => (c[3] ?? 0) - (c[1] ?? 0),
    (c) => (c[3] ?? 0) + (c[2] ?? 0),
    (c) => (c[3] ?? 0) - (c[2] ?? 0),
  ]
  let poly = [...polygon]
  for (const dist of planes) {
    if (poly.length === 0) break
    const next: ClipVertex[] = []
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i] as ClipVertex
      const b = poly[(i + 1) % poly.length] as ClipVertex
      const da = dist(a.clip)
      const db = dist(b.clip)
      if (da >= 0) next.push(a)
      if (da >= 0 !== db >= 0) {
        const t = da / (da - db)
        const lerp = (x: readonly number[], y: readonly number[]): number[] =>
          x.map((v, k) => v + t * ((y[k] ?? 0) - v))
        const clip = lerp(a.clip, b.clip)
        next.push({
          clip: [clip[0] ?? 0, clip[1] ?? 0, clip[2] ?? 0, clip[3] ?? 0],
          attributes: lerp(a.attributes, b.attributes),
        })
      }
    }
    poly = next
  }
  return poly
}

/**
 * Whether a world-space triangle faces away from the camera (its normal `(b - a) x (c - a)`, counter-clockwise
 * winding seen from the front, points away from the eye), the test back-face culling applies before rasterising.
 *
 * @param cam - The camera.
 * @param a - First vertex.
 * @param b - Second vertex.
 * @param c - Third vertex.
 * @returns Whether the triangle is a back face (degenerate triangles count as back faces).
 * @concept back-face-culling
 * @example
 * ```ts
 * import { isBackFacing, pinholeCamera } from 'mdml-engine/graphics'
 * const cam = pinholeCamera()
 * [isBackFacing(cam, [0, 0, 0], [1, 0, 0], [0, 1, 0]), isBackFacing(cam, [0, 0, 0], [0, 1, 0], [1, 0, 0])] // => [false, true]
 * ```
 */
export function isBackFacing(cam: RenderCamera, a: Vec3, b: Vec3, c: Vec3): boolean {
  const n = cross3(sub3(b, a), sub3(c, a))
  const toEye = cam.kind === 'pinhole' ? sub3(cam.eye, a) : sub3([0, 0, 0], cameraBasis(cam).forward)
  return dot3(n, toEye) <= 0
}

/** A depth buffer: the nearest depth written at each pixel (Infinity where nothing is). */
export interface DepthBuffer {
  readonly width: number
  readonly height: number
  readonly depth: Float64Array
}

/**
 * An empty depth buffer.
 *
 * @param width - Width in pixels.
 * @param height - Height in pixels.
 * @returns The buffer, every pixel at Infinity.
 * @concept depth-buffer
 * @example
 * ```ts
 * import { depthBuffer } from 'mdml-engine/graphics'
 * depthBuffer(2, 1).depth // => Float64Array.from([Infinity, Infinity])
 * ```
 */
export function depthBuffer(width: number, height: number): DepthBuffer {
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new MdmlError('value', 'depthBuffer: the size must be positive integers')
  }
  return { width, height, depth: new Float64Array(width * height).fill(Number.POSITIVE_INFINITY) }
}

/**
 * The depth test: keep a fragment when it is nearer than what the buffer holds at its pixel, and record its depth.
 *
 * @param buffer - The depth buffer (updated in place: it is the renderer's working memory).
 * @param x - Pixel column.
 * @param y - Pixel row.
 * @param z - The fragment's depth (smaller is nearer).
 * @returns Whether the fragment passed (and was written).
 * @concept depth-buffer
 * @example
 * ```ts
 * import { depthBuffer, depthTest } from 'mdml-engine/graphics'
 * const b = depthBuffer(1, 1)
 * [depthTest(b, 0, 0, 0.5), depthTest(b, 0, 0, 0.7), depthTest(b, 0, 0, 0.2)] // => [true, false, true]
 * ```
 */
export function depthTest(buffer: DepthBuffer, x: number, y: number, z: number): boolean {
  const i = y * buffer.width + x
  if (!(z < (buffer.depth[i] ?? Number.NEGATIVE_INFINITY))) return false
  buffer.depth[i] = z
  return true
}

// -------------------------------------------------------------------------------------------------- the pipeline

/**
 * A triangle mesh: positions, triangles as vertex index triples (counter-clockwise seen from the front), and optional
 * per-vertex colours, normals and texture coordinates.
 *
 * @concept rasterisation
 */
export interface Mesh {
  readonly positions: readonly Vec3[]
  readonly triangles: readonly (readonly [number, number, number])[]
  readonly colours?: readonly Vec3[]
  readonly normals?: readonly Vec3[]
  readonly uvs?: readonly (readonly [number, number])[]
}

/**
 * A mesh, checked: every index in range, and every attribute with one entry per vertex.
 *
 * @param positions - Vertex positions.
 * @param triangles - Index triples.
 * @param attributes - Optional colours, normals and texture coordinates per vertex.
 * @returns The mesh.
 * @throws MdmlError with code `value` for bad indices or attribute counts.
 * @concept rasterisation
 * @example
 * ```ts
 * import { mesh } from 'mdml-engine/graphics'
 * mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]).triangles.length // => 1
 * ```
 */
export function mesh(
  positions: readonly Vec3[],
  triangles: readonly (readonly [number, number, number])[],
  attributes: Pick<Mesh, 'colours' | 'normals' | 'uvs'> = {},
): Mesh {
  const n = positions.length
  for (const [i, p] of positions.entries()) check3(p, `position ${i}`)
  for (const t of triangles) {
    if (!t.every((i) => Number.isInteger(i) && i >= 0 && i < n))
      throw new MdmlError('value', 'mesh: index out of range')
  }
  for (const [name, list] of Object.entries(attributes)) {
    if (list !== undefined && (list as readonly unknown[]).length !== n) {
      throw new MdmlError('value', `mesh: ${name} needs one entry per vertex`)
    }
  }
  return { positions, triangles, ...attributes }
}

/** Options of {@link rasteriseMesh}. */
export interface RasteriseOptions {
  readonly width: number
  readonly height: number
  /** Which faces to drop (default `'back'`). */
  readonly cull?: 'back' | 'front' | 'none'
  /** Background colour (default black). */
  readonly background?: Vec3
  /** Multiplies the interpolated colour, sampled at the interpolated texture coordinates (bilinear, tiled). */
  readonly texture?: FloatImage
  /** Shade the visible surface (deferred: once per covered pixel, in one batch), with the colour as albedo. */
  readonly shading?: { readonly material: Material; readonly lights: readonly Light[]; readonly ambient?: Vec3 }
  /** Interpolate attributes perspective-correctly (default true); off, they are affine in screen space. */
  readonly perspectiveCorrect?: boolean
}

/** What happened to one triangle of a mesh in the pipeline. */
export interface TriangleRecord {
  readonly index: number
  readonly culled: boolean
  /** Vertices of the polygon after clipping (3 when untouched, 0 when wholly outside). */
  readonly clippedVertices: number
  /** Pixels it covered, and pixels where it passed the depth test. */
  readonly fragments: number
  readonly written: number
}

/** The output of {@link rasteriseMesh}: the image, the depth buffer (NDC depth), which triangle won each pixel, and the per-triangle record. */
export interface RasterResult {
  readonly image: FloatImage
  readonly depth: Float64Array
  readonly triangleIds: Int32Array
  readonly triangles: readonly TriangleRecord[]
}

/**
 * Rasterise a mesh through a camera: transform vertices to clip space, cull back (or front) faces, clip against the
 * view volume, divide by `w`, map to the screen, cover pixels by edge functions with the top-left rule, interpolate
 * depth linearly in screen space and attributes perspective-correctly (by `1 / w`), keep the nearest fragment by a
 * depth buffer, and finally shade the surviving pixels (deferred, in one batched call). The per-triangle record and
 * the buffers make every stage visible to a figure.
 *
 * @param m - The mesh.
 * @param cam - The camera.
 * @param options - Size, culling, background, texture, shading and interpolation.
 * @returns The image, depth buffer, triangle ids and records.
 * @concept rasterisation
 * @index function Triangle rasterisation with depth buffer.
 * @example
 * ```ts
 * import { mesh, pinholeCamera, rasteriseMesh } from 'mdml-engine/graphics'
 * const quad = mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]], { colours: [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]] })
 * const r = rasteriseMesh(quad, pinholeCamera({ eye: [0, 0, 3] }), { width: 8, height: 8 })
 * [Array.from(r.image.rgb.slice(3 * 36, 3 * 36 + 3)), r.triangles.map((t) => t.culled)] // => [[1, 0, 0], [false, false]]
 * ```
 */
export function rasteriseMesh(m: Mesh, cam: RenderCamera, options: RasteriseOptions): RasterResult {
  const { width, height } = options
  const buffer = depthBuffer(width, height)
  const ids = new Int32Array(width * height).fill(-1)
  const vp = viewProjection(cam, width / height)
  const cull = options.cull ?? 'back'
  const perspective = options.perspectiveCorrect !== false
  // Attributes per vertex: world position (3), colour (3), normal (3), uv (2).
  const stride = 11
  const pixelAttr = new Float64Array(stride * width * height)
  const records: TriangleRecord[] = []
  for (const [index, tri] of m.triangles.entries()) {
    const [ia, ib, ic] = tri
    const pa = m.positions[ia] as Vec3
    const pb = m.positions[ib] as Vec3
    const pc = m.positions[ic] as Vec3
    const back = isBackFacing(cam, pa, pb, pc)
    if ((cull === 'back' && back) || (cull === 'front' && !back)) {
      records.push({ index, culled: true, clippedVertices: 0, fragments: 0, written: 0 })
      continue
    }
    const faceNormal = (() => {
      const n = cross3(sub3(pb, pa), sub3(pc, pa))
      return dot3(n, n) > 0 ? normalise3(n) : ([0, 0, 1] as Vec3)
    })()
    const vertex = (i: number, p: Vec3): ClipVertex => {
      const colour = m.colours?.[i] ?? [1, 1, 1]
      const normal = m.normals?.[i] ?? faceNormal
      const uv = m.uvs?.[i] ?? [0, 0]
      return { clip: transformPoint(vp, p), attributes: [...p, ...colour, ...normal, uv[0], uv[1]] }
    }
    const poly = clipPolygon([vertex(ia, pa), vertex(ib, pb), vertex(ic, pc)])
    let fragments = 0
    let written = 0
    for (let k = 1; k + 1 < poly.length; k++) {
      const verts = [poly[0], poly[k], poly[k + 1]] as ClipVertex[]
      const screen = verts.map((v): [number, number, number, number] => {
        const w = v.clip[3]
        return [((v.clip[0] / w + 1) * width) / 2, ((1 - v.clip[1] / w) * height) / 2, v.clip[2] / w, 1 / w]
      })
      const [sa, sb, sc] = screen as [
        [number, number, number, number],
        [number, number, number, number],
        [number, number, number, number],
      ]
      for (const f of rasteriseTriangle([sa[0], sa[1]], [sb[0], sb[1]], [sc[0], sc[1]], width, height)) {
        fragments++
        const [w0, w1, w2] = f.weights
        const z = w0 * sa[2] + w1 * sb[2] + w2 * sc[2]
        if (!depthTest(buffer, f.x, f.y, z)) continue
        written++
        const pix = f.y * width + f.x
        ids[pix] = index
        let q0 = w0
        let q1 = w1
        let q2 = w2
        if (perspective) {
          const s = w0 * sa[3] + w1 * sb[3] + w2 * sc[3]
          q0 = (w0 * sa[3]) / s
          q1 = (w1 * sb[3]) / s
          q2 = (w2 * sc[3]) / s
        }
        const [A, B, C] = verts as [ClipVertex, ClipVertex, ClipVertex]
        for (let k2 = 0; k2 < stride; k2++) {
          pixelAttr[stride * pix + k2] =
            q0 * (A.attributes[k2] ?? 0) + q1 * (B.attributes[k2] ?? 0) + q2 * (C.attributes[k2] ?? 0)
        }
      }
    }
    records.push({ index, culled: false, clippedVertices: poly.length, fragments, written })
  }
  // Resolve: background, interpolated colour times texture, then deferred shading of the covered pixels.
  const bg = options.background ?? [0, 0, 0]
  const image = floatImage(width, height, () => bg)
  const covered: number[] = []
  const albedo: number[] = []
  for (let pix = 0; pix < width * height; pix++) {
    if ((ids[pix] ?? -1) < 0) continue
    const a = pixelAttr.subarray(stride * pix, stride * pix + stride)
    let colour: Vec3 = [a[3] ?? 0, a[4] ?? 0, a[5] ?? 0]
    if (options.texture !== undefined) {
      const t = sampleBilinear(options.texture, a[9] ?? 0, a[10] ?? 0)
      colour = [colour[0] * t[0], colour[1] * t[1], colour[2] * t[2]]
    }
    covered.push(pix)
    albedo.push(...colour)
    if (options.shading === undefined) image.rgb.set(colour, 3 * pix)
  }
  if (options.shading !== undefined && covered.length > 0) {
    const points = new Float64Array(3 * covered.length)
    const normals = new Float64Array(3 * covered.length)
    const views = new Float64Array(3 * covered.length)
    const { forward } = cameraBasis(cam)
    for (const [r, pix] of covered.entries()) {
      const a = pixelAttr.subarray(stride * pix, stride * pix + stride)
      const p: Vec3 = [a[0] ?? 0, a[1] ?? 0, a[2] ?? 0]
      const view = cam.kind === 'pinhole' ? normalise3(sub3(cam.eye, p)) : (sub3([0, 0, 0], forward) as Vec3)
      let n: Vec3 = [a[6] ?? 0, a[7] ?? 0, a[8] ?? 1]
      n = dot3(n, n) > 0 ? normalise3(n) : [0, 0, 1]
      // Shade the side facing the viewer.
      if (dot3(n, view) < 0) n = [-n[0], -n[1], -n[2]]
      points.set(p, 3 * r)
      normals.set(n, 3 * r)
      views.set(view, 3 * r)
    }
    const { material, lights, ambient } = options.shading
    const lit = shadePoints(material, points, normals, views, lights, {
      albedo: Float64Array.from(albedo),
      ...(ambient === undefined ? {} : { ambient }),
    })
    for (const [r, pix] of covered.entries()) image.rgb.set(lit.subarray(3 * r, 3 * r + 3), 3 * pix)
  }
  return { image, depth: buffer.depth, triangleIds: ids, triangles: records }
}
