/**
 * Cameras and projection for 3-D rendering: pinhole and orthographic cameras, the OpenGL-style view and projection
 * matrices (`lookAt`, `perspective`, `orthographic`), projection of points to pixels (also differentiably in the
 * points), per-pixel ray generation with optional jitter for antialiasing, and view-frustum planes with point, sphere
 * and box culling tests.
 */

import { WhiteboxError } from '../base/index.ts'
import { applyHomogeneous } from '../foundations/index.ts'
import { ddot } from '../numerics/linalg/blas.ts'
import { type Key, uniform } from '../random/index.ts'
import type { Camera as SceneCamera, Vec3 } from '../scene/index.ts'
import { add, mul, promoteFloat, sliceAxis, squeeze, stack, sub, type TensorLike, type Value } from '../tensor/index.ts'
import { check3, cross3, dot3, madd3, normalise3, sub3 } from './_vec3.ts'

/** A 4 x 4 matrix, row-major, acting on column vectors (`p' = M p`), as numpy and OpenGL's mathematics write it. */
export type Mat4 = readonly (readonly number[])[]

/**
 * A pinhole (perspective) camera at `eye` looking at `target`, with `up` fixing the roll, a vertical field of view
 * `fov` in degrees, and near and far clipping distances.
 *
 * @concept pinhole-camera
 */
export interface PinholeCamera {
  readonly kind: 'pinhole'
  readonly eye: Vec3
  readonly target: Vec3
  readonly up: Vec3
  readonly fov: number
  readonly near: number
  readonly far: number
}

/**
 * An orthographic camera: parallel rays along the view direction; `height` is the half-height of the view in world
 * units (the half-width is `height * aspect`).
 *
 * @concept orthographic-projection
 */
export interface OrthographicCamera {
  readonly kind: 'orthographic'
  readonly eye: Vec3
  readonly target: Vec3
  readonly up: Vec3
  readonly height: number
  readonly near: number
  readonly far: number
}

/** A camera for rendering: pinhole or orthographic. */
export type RenderCamera = PinholeCamera | OrthographicCamera

function checkPose(eye: Vec3, target: Vec3, up: Vec3): void {
  const f = normalise3(sub3(target, eye), 'target - eye')
  normalise3(cross3(f, normalise3(up, 'up')), 'the up vector (it must not be parallel to the view direction)')
}

function checkClip(near: number, far: number): void {
  if (!(near > 0 && far > near)) throw new WhiteboxError('value', 'camera: need 0 < near < far')
}

/**
 * A pinhole camera. Defaults: eye `[0, 0, 5]`, target the origin, up `+y`, a 45 degree vertical field of view, near
 * 0.1 and far 100.
 *
 * @param options - Any camera fields.
 * @returns The camera.
 * @throws WhiteboxError with code `value` for an eye at the target, an up vector along the view, a field of view
 *   outside (0, 180) or bad clipping distances.
 * @concept pinhole-camera
 * @example
 * ```ts
 * import { pinholeCamera } from 'whitebox-engine/graphics'
 * pinholeCamera({ fov: 60 }).eye // => [0, 0, 5]
 * ```
 */
export function pinholeCamera(options: Partial<Omit<PinholeCamera, 'kind'>> = {}): PinholeCamera {
  const cam: PinholeCamera = {
    kind: 'pinhole',
    eye: check3(options.eye ?? [0, 0, 5], 'eye'),
    target: check3(options.target ?? [0, 0, 0], 'target'),
    up: check3(options.up ?? [0, 1, 0], 'up'),
    fov: options.fov ?? 45,
    near: options.near ?? 0.1,
    far: options.far ?? 100,
  }
  checkPose(cam.eye, cam.target, cam.up)
  if (!(cam.fov > 0 && cam.fov < 180)) throw new WhiteboxError('value', 'pinholeCamera: fov must be in (0, 180)')
  checkClip(cam.near, cam.far)
  return cam
}

/**
 * An orthographic camera. Defaults as {@link pinholeCamera}, with a half-height of 1.
 *
 * @param options - Any camera fields.
 * @returns The camera.
 * @throws WhiteboxError with code `value` for a bad pose, a non-positive height or bad clipping distances.
 * @concept orthographic-projection
 * @example
 * ```ts
 * import { orthographicCamera } from 'whitebox-engine/graphics'
 * orthographicCamera({ height: 2 }).kind // => 'orthographic'
 * ```
 */
export function orthographicCamera(options: Partial<Omit<OrthographicCamera, 'kind'>> = {}): OrthographicCamera {
  const cam: OrthographicCamera = {
    kind: 'orthographic',
    eye: check3(options.eye ?? [0, 0, 5], 'eye'),
    target: check3(options.target ?? [0, 0, 0], 'target'),
    up: check3(options.up ?? [0, 1, 0], 'up'),
    height: options.height ?? 1,
    near: options.near ?? 0.1,
    far: options.far ?? 100,
  }
  checkPose(cam.eye, cam.target, cam.up)
  if (!(cam.height > 0)) throw new WhiteboxError('value', 'orthographicCamera: height must be positive')
  checkClip(cam.near, cam.far)
  return cam
}

/**
 * The render camera that sees what a `scene` orbit camera sees in a viewport: the same eye, target and roll, and the
 * vertical field of view (or half-height) that gives the scene camera's view across the shorter side. Projecting with
 * either gives the same pixels (see `projectPoint` in `scene/projection.ts`).
 *
 * @param cam - The orbit camera of 3-D scene layers.
 * @param viewport - The image size.
 * @returns The equivalent render camera.
 * @concept pinhole-camera
 * @example
 * ```ts
 * import { fromSceneCamera } from 'whitebox-engine/graphics'
 * import { camera } from 'whitebox-engine/scene'
 * const c = fromSceneCamera(camera({ azimuth: 0, elevation: 0, distance: 4 }), { width: 100, height: 100 })
 * [c.eye, c.up, c.kind] // => [[4, 0, 0], [0, 0, 1], 'pinhole']
 * ```
 */
export function fromSceneCamera(
  cam: SceneCamera,
  viewport: { readonly width: number; readonly height: number },
): RenderCamera {
  const az = (cam.azimuth * Math.PI) / 180
  const el = (cam.elevation * Math.PI) / 180
  const [ca, sa, ce, se] = [Math.cos(az), Math.sin(az), Math.cos(el), Math.sin(el)]
  const eye = madd3(cam.target, [ce * ca, ce * sa, se], cam.distance)
  // Adding 0 turns -0 into 0.
  const up: Vec3 = [-se * ca + 0, -se * sa + 0, ce + 0]
  // The scene camera frames the shorter side; a render camera frames the height.
  const ratio = viewport.width >= viewport.height ? 1 : viewport.height / viewport.width
  const near = 1e-6 * cam.distance
  const far = 1e6 * cam.distance
  if (cam.projection === 'orthographic') {
    return orthographicCamera({ eye, target: cam.target, up, height: cam.extent * ratio, near, far })
  }
  const fov = (360 / Math.PI) * Math.atan(Math.tan((cam.fov * Math.PI) / 360) * ratio)
  return pinholeCamera({ eye, target: cam.target, up, fov, near, far })
}

/** The camera's orthonormal frame: `right`, `up` and `forward` (towards the target). */
export interface CameraBasis {
  readonly right: Vec3
  readonly up: Vec3
  readonly forward: Vec3
}

/**
 * The camera's orthonormal frame (forward towards the target, right = forward x up, up = right x forward).
 *
 * @param cam - The camera.
 * @returns The frame.
 * @concept pinhole-camera
 * @example
 * ```ts
 * import { cameraBasis, pinholeCamera } from 'whitebox-engine/graphics'
 * cameraBasis(pinholeCamera()) // => { right: [1, 0, 0], up: [0, 1, 0], forward: [0, 0, -1] }
 * ```
 */
export function cameraBasis(cam: RenderCamera): CameraBasis {
  const forward = normalise3(sub3(cam.target, cam.eye))
  const right = normalise3(cross3(forward, cam.up))
  return { right, up: cross3(right, forward), forward }
}

/**
 * The view matrix of `gluLookAt`: a rigid motion taking the eye to the origin and the view direction to `-z`, with
 * `up` towards `+y`.
 *
 * @param eye - The eye.
 * @param target - The point looked at.
 * @param up - The up direction.
 * @returns The matrix.
 * @throws WhiteboxError with code `value` for an eye at the target or an up vector along the view.
 * @concept view-matrix
 * @example
 * ```ts
 * import { lookAt } from 'whitebox-engine/graphics'
 * lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0])[2] // => [0, 0, 1, -5]
 * ```
 */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  checkPose(eye, target, up)
  const { right: s, up: u, forward: f } = cameraBasis({ kind: 'pinhole', eye, target, up, fov: 45, near: 1, far: 2 })
  return [
    [s[0], s[1], s[2], -dot3(s, eye) + 0],
    [u[0], u[1], u[2], -dot3(u, eye) + 0],
    [-f[0] + 0, -f[1] + 0, -f[2] + 0, dot3(f, eye) + 0],
    [0, 0, 0, 1],
  ]
}

/**
 * The perspective projection matrix of `gluPerspective`: eye space to clip space, mapping the view frustum to the
 * cube `[-1, 1]^3` after division by `w = -z_eye` (near to -1, far to +1).
 *
 * @param fovY - Vertical field of view in degrees.
 * @param aspect - Width over height.
 * @param near - Near clipping distance.
 * @param far - Far clipping distance.
 * @returns The matrix.
 * @throws WhiteboxError with code `value` for bad arguments.
 * @concept perspective-projection
 * @example
 * ```ts
 * import { perspective } from 'whitebox-engine/graphics'
 * perspective(90, 1, 1, 3).map((row) => row.map((x) => Math.round(x * 1e12) / 1e12)) // => [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -2, -3], [0, 0, -1, 0]]
 * ```
 */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  if (!(fovY > 0 && fovY < 180 && aspect > 0)) throw new WhiteboxError('value', 'perspective: bad fov or aspect')
  checkClip(near, far)
  const f = 1 / Math.tan((fovY * Math.PI) / 360)
  return [
    [f / aspect, 0, 0, 0],
    [0, f, 0, 0],
    [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
    [0, 0, -1, 0],
  ]
}

/**
 * The orthographic projection matrix of `glOrtho`: the box `[left, right] x [bottom, top] x [-near, -far]` in eye
 * space to the cube `[-1, 1]^3`.
 *
 * @param left - Left plane.
 * @param right - Right plane.
 * @param bottom - Bottom plane.
 * @param top - Top plane.
 * @param near - Near distance.
 * @param far - Far distance.
 * @returns The matrix.
 * @throws WhiteboxError with code `value` for an empty box.
 * @concept orthographic-projection
 * @example
 * ```ts
 * import { orthographic } from 'whitebox-engine/graphics'
 * orthographic(-1, 1, -1, 1, 1, 3)[2] // => [0, 0, -1, -2]
 * ```
 */
export function orthographic(
  left: number,
  right: number,
  bottom: number,
  top: number,
  near: number,
  far: number,
): Mat4 {
  if (!(right > left && top > bottom && far > near)) throw new WhiteboxError('value', 'orthographic: empty view box')
  return [
    [2 / (right - left), 0, 0, -(right + left) / (right - left) + 0],
    [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom) + 0],
    [0, 0, -2 / (far - near), -(far + near) / (far - near) + 0],
    [0, 0, 0, 1],
  ]
}

/**
 * The camera's view matrix ({@link lookAt} of its pose).
 *
 * @param cam - The camera.
 * @returns The matrix.
 * @concept view-matrix
 * @example
 * ```ts
 * import { pinholeCamera, viewMatrix } from 'whitebox-engine/graphics'
 * viewMatrix(pinholeCamera({ eye: [0, 0, 2] }))[2][3] // => -2
 * ```
 */
export function viewMatrix(cam: RenderCamera): Mat4 {
  return lookAt(cam.eye, cam.target, cam.up)
}

/**
 * The camera's projection matrix for an image of the given aspect ratio.
 *
 * @param cam - The camera.
 * @param aspect - Width over height.
 * @returns The matrix.
 * @concept perspective-projection
 * @example
 * ```ts
 * import { orthographicCamera, projectionMatrix } from 'whitebox-engine/graphics'
 * projectionMatrix(orthographicCamera({ height: 2 }), 2)[0][0] // => 0.25
 * ```
 */
export function projectionMatrix(cam: RenderCamera, aspect: number): Mat4 {
  if (cam.kind === 'pinhole') return perspective(cam.fov, aspect, cam.near, cam.far)
  return orthographic(-cam.height * aspect, cam.height * aspect, -cam.height, cam.height, cam.near, cam.far)
}

function multiply4(a: Mat4, b: Mat4): Mat4 {
  const out: number[][] = []
  for (let i = 0; i < 4; i++) {
    const row: number[] = []
    for (let j = 0; j < 4; j++)
      row.push(ddot(a[i] ?? [], [b[0]?.[j] ?? 0, b[1]?.[j] ?? 0, b[2]?.[j] ?? 0, b[3]?.[j] ?? 0]))
    out.push(row)
  }
  return out
}

/**
 * The combined view-projection matrix `P V`: world space to clip space.
 *
 * @param cam - The camera.
 * @param aspect - Width over height.
 * @returns The matrix.
 * @concept perspective-projection
 * @example
 * ```ts
 * import { pinholeCamera, viewProjection } from 'whitebox-engine/graphics'
 * viewProjection(pinholeCamera({ fov: 90 }), 1)[3] // => [0, 0, -1, 5]
 * ```
 */
export function viewProjection(cam: RenderCamera, aspect: number): Mat4 {
  return multiply4(projectionMatrix(cam, aspect), viewMatrix(cam))
}

/**
 * Apply a 4 x 4 matrix to a point in homogeneous coordinates `[x, y, z, 1]`.
 *
 * @param m - The matrix.
 * @param p - The point.
 * @returns `[x, y, z, w]`.
 * @concept homogeneous-coordinates
 * @example
 * ```ts
 * import { lookAt, transformPoint } from 'whitebox-engine/graphics'
 * transformPoint(lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]), [1, 2, 0]) // => [1, 2, -5, 1]
 * ```
 */
export function transformPoint(m: Mat4, p: Vec3): [number, number, number, number] {
  const h = [p[0], p[1], p[2], 1]
  return [ddot(m[0] ?? [], h), ddot(m[1] ?? [], h), ddot(m[2] ?? [], h), ddot(m[3] ?? [], h)]
}

/** A point projected into an image. */
export interface PixelPoint {
  /** Image coordinates (y down, pixel centres at half-integers). */
  readonly x: number
  readonly y: number
  /** Distance in front of the eye along the view direction. */
  readonly depth: number
  /** Normalised device depth in [-1, 1] between the near and far planes. */
  readonly ndcZ: number
}

/**
 * Project a world point into an image: clip space by {@link viewProjection}, division by `w`, then the viewport map
 * `x = (ndc_x + 1) w / 2`, `y = (1 - ndc_y) h / 2`.
 *
 * @param cam - The camera.
 * @param p - The point.
 * @param width - Image width.
 * @param height - Image height.
 * @returns The projected point, or `undefined` at or behind the eye of a pinhole camera.
 * @concept perspective-projection
 * @example
 * ```ts
 * import { pinholeCamera, projectToPixel } from 'whitebox-engine/graphics'
 * const p = projectToPixel(pinholeCamera({ fov: 90 }), [0, 0, 0], 100, 50)
 * [p?.x, p?.y, p?.depth] // => [50, 25, 5]
 * ```
 */
export function projectToPixel(cam: RenderCamera, p: Vec3, width: number, height: number): PixelPoint | undefined {
  const clip = transformPoint(viewProjection(cam, width / height), p)
  const w = clip[3]
  if (!(w > 0)) return undefined
  const { forward } = cameraBasis(cam)
  return {
    x: ((clip[0] / w + 1) * width) / 2,
    y: ((1 - clip[1] / w) * height) / 2,
    depth: dot3(sub3(p, cam.eye), forward),
    ndcZ: clip[2] / w,
  }
}

/**
 * Project many points at once, differentiably in the points (for fitting, or gradients of a figure's layout): image
 * coordinates and normalised device depth, `[n, 3]`, with no clipping.
 *
 * @param cam - The camera.
 * @param points - World points `[n, 3]`.
 * @param width - Image width.
 * @param height - Image height.
 * @returns `[x, y, ndcZ]` per point `[n, 3]`.
 * @concept perspective-projection
 * @example
 * ```ts
 * import { pinholeCamera, projectPoints } from 'whitebox-engine/graphics'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(projectPoints(pinholeCamera({ fov: 90 }), [[0, 0, 0]], 100, 50))[0].slice(0, 2) // => [50, 25]
 * ```
 */
export function projectPoints(cam: RenderCamera, points: TensorLike, width: number, height: number): Value {
  const [pts] = promoteFloat(points) as [Value]
  const ndc = applyHomogeneous(viewProjection(cam, width / height) as number[][], pts)
  const col = (k: number): Value => squeeze(sliceAxis(ndc, -1, k, k + 1), -1)
  return stack([mul(add(col(0), 1), width / 2), mul(sub(1, col(1)), height / 2), col(2)], -1)
}

/** A ray: an origin and a unit direction. */
export interface Ray {
  readonly origin: Vec3
  readonly direction: Vec3
}

/**
 * The ray through a point of the image (pixel `(i, j)` has its centre at `(i + 0.5, j + 0.5)`): from the eye through
 * the image plane for a pinhole camera, parallel to the view from the image plane for an orthographic one.
 *
 * @param cam - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param x - Image x (to the right).
 * @param y - Image y (down).
 * @returns The ray.
 * @concept ray-generation
 * @example
 * ```ts
 * import { pinholeCamera, pixelRay } from 'whitebox-engine/graphics'
 * pixelRay(pinholeCamera(), 64, 48, 32, 24) // => { origin: [0, 0, 5], direction: [0, 0, -1] }
 * ```
 */
export function pixelRay(cam: RenderCamera, width: number, height: number, x: number, y: number): Ray {
  const { right, up, forward } = cameraBasis(cam)
  const aspect = width / height
  const sx = (2 * x) / width - 1
  const sy = 1 - (2 * y) / height
  if (cam.kind === 'orthographic') {
    const origin = madd3(madd3(cam.eye, right, sx * cam.height * aspect), up, sy * cam.height)
    return { origin, direction: forward }
  }
  const t = Math.tan((cam.fov * Math.PI) / 360)
  const d = madd3(madd3(forward, right, sx * t * aspect), up, sy * t)
  return { origin: cam.eye, direction: normalise3(d) }
}

/** Rays for every pixel, row-major from the top-left: flat origins and unit directions, three numbers per pixel. */
export interface RayBundle {
  readonly width: number
  readonly height: number
  readonly origins: Float64Array
  readonly directions: Float64Array
}

/**
 * Generate a ray per pixel through the pixel centres, or through uniformly jittered points inside each pixel when a
 * key is given (for antialiasing by averaging samples).
 *
 * @param cam - The camera.
 * @param width - Image width.
 * @param height - Image height.
 * @param options - A key for jittered samples.
 * @returns The rays.
 * @concept ray-generation
 * @example
 * ```ts
 * import { cameraRays, pinholeCamera } from 'whitebox-engine/graphics'
 * const rays = cameraRays(pinholeCamera(), 4, 2)
 * [rays.origins.length, rays.directions[2] < 0] // => [24, true]
 * ```
 */
export function cameraRays(
  cam: RenderCamera,
  width: number,
  height: number,
  options: { readonly key?: Key } = {},
): RayBundle {
  if (!(Number.isInteger(width) && Number.isInteger(height) && width > 0 && height > 0)) {
    throw new WhiteboxError('value', 'cameraRays: the image size must be positive integers')
  }
  const jitter = options.key === undefined ? undefined : uniform(options.key, [2 * width * height]).data
  const origins = new Float64Array(3 * width * height)
  const directions = new Float64Array(3 * width * height)
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const k = j * width + i
      const r = pixelRay(cam, width, height, i + (jitter?.[2 * k] ?? 0.5), j + (jitter?.[2 * k + 1] ?? 0.5))
      origins.set(r.origin, 3 * k)
      directions.set(r.direction, 3 * k)
    }
  }
  return { width, height, origins, directions }
}

// ------------------------------------------------------------------------------------------------- frustum tests

/** A plane `a x + b y + c z + d = 0` with unit normal `(a, b, c)` pointing into the frustum. */
export type Plane4 = readonly [number, number, number, number]

/**
 * The six planes of the view frustum of a view-projection matrix (Gribb and Hartmann's extraction: rows of the
 * matrix added and subtracted), in the order left, right, bottom, top, near, far, normalised, pointing inwards.
 *
 * @param m - A view-projection matrix (or a projection matrix for eye-space planes).
 * @returns The planes.
 * @concept view-frustum
 * @example
 * ```ts
 * import { frustumPlanes, perspective } from 'whitebox-engine/graphics'
 * frustumPlanes(perspective(90, 1, 1, 10))[4] // => [0, 0, -1, -1]
 * ```
 */
export function frustumPlanes(m: Mat4): Plane4[] {
  const row = (i: number): readonly number[] => m[i] ?? [0, 0, 0, 0]
  const combine = (i: number, s: number): Plane4 => {
    const a = row(3)
    const b = row(i)
    const p = [0, 1, 2, 3].map((k) => (a[k] ?? 0) + s * (b[k] ?? 0))
    const n = Math.hypot(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0)
    return [(p[0] ?? 0) / n, (p[1] ?? 0) / n, (p[2] ?? 0) / n, (p[3] ?? 0) / n]
  }
  return [combine(0, 1), combine(0, -1), combine(1, 1), combine(1, -1), combine(2, 1), combine(2, -1)]
}

/**
 * The frustum planes of a camera.
 *
 * @param cam - The camera.
 * @param aspect - Width over height.
 * @returns The planes (left, right, bottom, top, near, far).
 * @concept view-frustum
 * @example
 * ```ts
 * import { cameraFrustum, pinholeCamera } from 'whitebox-engine/graphics'
 * cameraFrustum(pinholeCamera(), 1).length // => 6
 * ```
 */
export function cameraFrustum(cam: RenderCamera, aspect: number): Plane4[] {
  return frustumPlanes(viewProjection(cam, aspect))
}

const distance = (pl: Plane4, p: Vec3): number => pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3]

/**
 * Whether a point lies inside the frustum (on or inside every plane).
 *
 * @param planes - Frustum planes.
 * @param p - The point.
 * @returns Whether it is inside.
 * @concept view-frustum
 * @example
 * ```ts
 * import { cameraFrustum, pinholeCamera, pointInFrustum } from 'whitebox-engine/graphics'
 * const f = cameraFrustum(pinholeCamera(), 1)
 * [pointInFrustum(f, [0, 0, 0]), pointInFrustum(f, [0, 0, 6])] // => [true, false]
 * ```
 */
export function pointInFrustum(planes: readonly Plane4[], p: Vec3): boolean {
  return planes.every((pl) => distance(pl, p) >= 0)
}

/** Where a volume lies relative to a frustum. */
export type Containment = 'inside' | 'intersect' | 'outside'

/**
 * Classify a sphere against a frustum: outside when it lies wholly behind a plane, inside when wholly in front of all,
 * otherwise intersecting (conservative near the frustum's corners, as view culling wants).
 *
 * @param planes - Frustum planes.
 * @param centre - The sphere's centre.
 * @param radius - Its radius.
 * @returns The classification.
 * @concept view-frustum
 * @example
 * ```ts
 * import { cameraFrustum, pinholeCamera, sphereInFrustum } from 'whitebox-engine/graphics'
 * const f = cameraFrustum(pinholeCamera(), 1)
 * [sphereInFrustum(f, [0, 0, 0], 0.5), sphereInFrustum(f, [0, 0, 5], 0.5), sphereInFrustum(f, [0, 0, 8], 1)] // => ['inside', 'intersect', 'outside']
 * ```
 */
export function sphereInFrustum(planes: readonly Plane4[], centre: Vec3, radius: number): Containment {
  let inside = true
  for (const pl of planes) {
    const d = distance(pl, centre)
    if (d < -radius) return 'outside'
    if (d < radius) inside = false
  }
  return inside ? 'inside' : 'intersect'
}

/**
 * Classify an axis-aligned box against a frustum by its positive and negative vertices per plane (outside when the
 * vertex farthest along a plane's normal is behind it; conservative, like {@link sphereInFrustum}).
 *
 * @param planes - Frustum planes.
 * @param min - The box's minimum corner.
 * @param max - The box's maximum corner.
 * @returns The classification.
 * @concept view-frustum
 * @example
 * ```ts
 * import { boxInFrustum, cameraFrustum, pinholeCamera } from 'whitebox-engine/graphics'
 * const f = cameraFrustum(pinholeCamera(), 1)
 * [boxInFrustum(f, [-1, -1, -1], [1, 1, 1]), boxInFrustum(f, [10, 10, 10], [11, 11, 11])] // => ['inside', 'outside']
 * ```
 */
export function boxInFrustum(planes: readonly Plane4[], min: Vec3, max: Vec3): Containment {
  let inside = true
  for (const pl of planes) {
    const pos: Vec3 = [pl[0] >= 0 ? max[0] : min[0], pl[1] >= 0 ? max[1] : min[1], pl[2] >= 0 ? max[2] : min[2]]
    const neg: Vec3 = [pl[0] >= 0 ? min[0] : max[0], pl[1] >= 0 ? min[1] : max[1], pl[2] >= 0 ? min[2] : max[2]]
    if (distance(pl, pos) < 0) return 'outside'
    if (distance(pl, neg) < 0) inside = false
  }
  return inside ? 'inside' : 'intersect'
}
