import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concreteOrThrow,
  div,
  fromStorage,
  matmul,
  mean,
  mul,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { det, inv, svd } from '../linalg/index.ts'

type Quat = readonly [number, number, number, number]
type Mat3 = readonly number[]

const qnorm = (q: Quat): Quat => {
  const n = Math.hypot(q[0], q[1], q[2], q[3])
  if (n === 0) throw new WhiteboxError('value', 'rotation: zero quaternion')
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/** Hamilton product p q (scalar last). */
const qmul = (p: Quat, q: Quat): Quat => [
  p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1],
  p[3] * q[1] - p[0] * q[2] + p[1] * q[3] + p[2] * q[0],
  p[3] * q[2] + p[0] * q[1] - p[1] * q[0] + p[2] * q[3],
  p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
]

function quatToMatrix(q: Quat): number[] {
  const [x, y, z, w] = q
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    1 - 2 * (x * x + y * y),
  ]
}

function matrixToQuat(m: Mat3): Quat {
  // scipy's choice of the largest of the diagonal and the trace, for stability.
  const at = (i: number, j: number): number => m[3 * i + j] ?? 0
  const trace = at(0, 0) + at(1, 1) + at(2, 2)
  const decision = [at(0, 0), at(1, 1), at(2, 2), trace]
  let choice = 0
  decision.forEach((v, i) => {
    if (v > (decision[choice] ?? 0)) choice = i
  })
  const q = [0, 0, 0, 0]
  if (choice !== 3) {
    const i = choice
    const j = (i + 1) % 3
    const k = (j + 1) % 3
    q[i] = 1 - trace + 2 * at(i, i)
    q[j] = at(j, i) + at(i, j)
    q[k] = at(k, i) + at(i, k)
    q[3] = at(k, j) - at(j, k)
  } else {
    q[0] = at(2, 1) - at(1, 2)
    q[1] = at(0, 2) - at(2, 0)
    q[2] = at(1, 0) - at(0, 1)
    q[3] = 1 + trace
  }
  return qnorm(q as unknown as Quat)
}

const AXES: Readonly<Record<string, number>> = { x: 0, y: 1, z: 2 }

function elementary(axis: number, angle: number): Quat {
  const q: [number, number, number, number] = [0, 0, 0, Math.cos(angle / 2)]
  q[axis] = Math.sin(angle / 2)
  return q
}

/** A rotation in three dimensions (scipy.spatial.transform.Rotation), stored as a unit quaternion [x, y, z, w]. */
export interface Rotation {
  readonly quaternion: Quat
  /** Rotate vectors `[3]` or `[n, 3]` (differentiable in the vectors). */
  readonly apply: (vectors: TensorLike, inverse?: boolean) => Value
  /** The quaternion [x, y, z, w]; canonical puts w >= 0. */
  readonly asQuat: (canonical?: boolean) => [number, number, number, number]
  readonly asMatrix: () => number[][]
  /** The rotation vector: axis times angle (radians, or degrees). */
  readonly asRotvec: (degrees?: boolean) => [number, number, number]
  /** Euler angles for a sequence of three axes: lowercase extrinsic ('xyz'), uppercase intrinsic ('XYZ'). */
  readonly asEuler: (seq: string, degrees?: boolean) => [number, number, number]
  readonly inv: () => Rotation
  /** this * other: apply other first, then this (as scipy's r1 * r2). */
  readonly compose: (other: Rotation) => Rotation
  /** The rotation angle in [0, pi]. */
  readonly magnitude: () => number
}

/**
 * A rotation from a quaternion [x, y, z, w] (scalar last, as scipy; normalised).
 *
 * @param q - The quaternion.
 * @returns The rotation.
 * @concept rotations
 * @example
 * ```ts
 * import { rotationFromQuat } from 'whitebox-engine/numerics/geometry'
 * rotationFromQuat([0, 0, Math.SQRT1_2, Math.SQRT1_2]).asEuler('xyz', true).map((v) => Math.round(v * 1e10) / 1e10) // => [0, 0, 90]
 * ```
 */
export function rotationFromQuat(q: readonly number[]): Rotation {
  if (q.length !== 4) throw new WhiteboxError('shape', 'rotationFromQuat: a quaternion has four entries')
  const quat = qnorm([q[0] ?? 0, q[1] ?? 0, q[2] ?? 0, q[3] ?? 0])
  const matrix = quatToMatrix(quat)
  const self: Rotation = {
    quaternion: quat,
    apply: (vectors, inverse = false) => {
      const [v] = promoteFloat(vectors) as [Value]
      const m = fromStorage(Float64Array.from(matrix), [3, 3], 'float64')
      // Row vectors: v R^T (or v R for the inverse).
      const r = inverse ? m : swapaxes(m, 0, 1)
      return v.shape.length === 1 ? reshape(matmul(reshape(v, [1, 3]), r), [3]) : matmul(v, r)
    },
    asQuat: (canonical = false) => {
      const flip =
        canonical &&
        (quat[3] < 0 ||
          (quat[3] === 0 && (quat[0] < 0 || (quat[0] === 0 && (quat[1] < 0 || (quat[1] === 0 && quat[2] < 0))))))
      return flip ? [-quat[0], -quat[1], -quat[2], -quat[3]] : [quat[0], quat[1], quat[2], quat[3]]
    },
    asMatrix: () => [matrix.slice(0, 3), matrix.slice(3, 6), matrix.slice(6, 9)],
    asRotvec: (degrees = false) => {
      const q2 = quat[3] < 0 ? quat.map((v) => -v) : [...quat]
      const s = Math.hypot(q2[0] ?? 0, q2[1] ?? 0, q2[2] ?? 0)
      const angle = 2 * Math.atan2(s, q2[3] ?? 1)
      // angle / sin(angle / 2), with its series near zero.
      const scale = angle <= 1e-3 ? 2 + angle ** 2 / 12 + (7 * angle ** 4) / 2880 : angle / Math.sin(angle / 2)
      const k = degrees ? 180 / Math.PI : 1
      return [(q2[0] ?? 0) * scale * k, (q2[1] ?? 0) * scale * k, (q2[2] ?? 0) * scale * k]
    },
    asEuler: (seq, degrees = false) => eulerFromQuat(quat, seq, degrees),
    inv: () => rotationFromQuat([-quat[0], -quat[1], -quat[2], quat[3]]),
    compose: (other) => rotationFromQuat([...qmul(quat, other.quaternion)]),
    magnitude: () => 2 * Math.atan2(Math.hypot(quat[0], quat[1], quat[2]), Math.abs(quat[3])),
  }
  return self
}

function parseSeq(seq: string): { axes: number[]; extrinsic: boolean } {
  if (!/^([xyz]{3}|[XYZ]{3})$/.test(seq)) throw new WhiteboxError('value', `rotation: bad Euler sequence ${seq}`)
  const lower = seq.toLowerCase()
  const axes = [...lower].map((c) => AXES[c] ?? 0)
  if (axes[0] === axes[1] || axes[1] === axes[2])
    throw new WhiteboxError('value', 'rotation: consecutive Euler axes must differ')
  return { axes, extrinsic: seq === lower }
}

/** Euler angles from a quaternion (Bernardes and Bezerra 2022, as scipy). */
function eulerFromQuat(quat: Quat, seq: string, degrees: boolean): [number, number, number] {
  const parsed = parseSeq(seq)
  const extrinsic = parsed.extrinsic
  const axes = extrinsic ? parsed.axes : [...parsed.axes].reverse()
  const i = axes[0] as number
  const j = axes[1] as number
  let k = axes[2] as number
  const symmetric = i === k
  if (symmetric) k = 3 - i - j
  const sign = ((i - j) * (j - k) * (k - i)) / 2
  const q = quat
  let a: number
  let b: number
  let c: number
  let d: number
  if (symmetric) {
    a = q[3]
    b = q[i] ?? 0
    c = q[j] ?? 0
    d = (q[k] ?? 0) * sign
  } else {
    a = q[3] - (q[j] ?? 0)
    b = (q[i] ?? 0) + (q[k] ?? 0) * sign
    c = (q[j] ?? 0) + q[3]
    d = (q[k] ?? 0) * sign - (q[i] ?? 0)
  }
  const angles = [0, 2 * Math.atan2(Math.hypot(c, d), Math.hypot(a, b)), 0]
  const eps = 1e-7
  const first = extrinsic ? 0 : 2
  const third = extrinsic ? 2 : 0
  const halfSum = Math.atan2(b, a)
  const halfDiff = Math.atan2(d, c)
  const case1 = Math.abs(angles[1] ?? 0) <= eps
  const case2 = Math.abs((angles[1] ?? 0) - Math.PI) <= eps
  if (!case1 && !case2) {
    angles[first] = halfSum - halfDiff
    angles[third] = halfSum + halfDiff
  } else {
    // Gimbal lock: the third angle is set to zero.
    angles[2] = 0
    angles[0] = case1 ? 2 * halfSum : 2 * halfDiff * (extrinsic ? -1 : 1)
  }
  if (!symmetric) {
    angles[third] = (angles[third] ?? 0) * sign
    angles[1] = (angles[1] ?? 0) - Math.PI / 2
  }
  const wrapped = angles.map((v) => (v < -Math.PI ? v + 2 * Math.PI : v > Math.PI ? v - 2 * Math.PI : v))
  const k2 = degrees ? 180 / Math.PI : 1
  return [(wrapped[0] ?? 0) * k2, (wrapped[1] ?? 0) * k2, (wrapped[2] ?? 0) * k2]
}

/**
 * A rotation from a 3 x 3 rotation matrix (scipy's Rotation.from_matrix, by the stable largest-component method).
 *
 * @param m - The matrix.
 * @returns The rotation.
 * @example
 * ```ts
 * import { rotationFromMatrix } from 'whitebox-engine/numerics/geometry'
 * rotationFromMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).magnitude() // => 1.5707963267948966
 * ```
 */
export function rotationFromMatrix(m: TensorLike): Rotation {
  const t = concreteOrThrow(promoteFloat(m)[0] as Tensor, 'rotationFromMatrix')
  if (t.shape.length !== 2 || t.shape[0] !== 3 || t.shape[1] !== 3)
    throw new WhiteboxError('shape', 'rotationFromMatrix: expected a 3 x 3 matrix')
  return rotationFromQuat([...matrixToQuat(toFlat(t))])
}

/**
 * A rotation from a rotation vector (axis times angle).
 *
 * @param v - The vector.
 * @param degrees - Whether the angle is in degrees.
 * @returns The rotation.
 * @example
 * ```ts
 * import { rotationFromRotvec } from 'whitebox-engine/numerics/geometry'
 * rotationFromRotvec([0, 0, Math.PI]).asQuat().map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0, 1, 0]
 * ```
 */
export function rotationFromRotvec(v: readonly number[], degrees = false): Rotation {
  const k = degrees ? Math.PI / 180 : 1
  const r = v.map((x) => x * k)
  const angle = Math.hypot(r[0] ?? 0, r[1] ?? 0, r[2] ?? 0)
  // sin(angle / 2) / angle, with its series near zero.
  const scale = angle <= 1e-3 ? 0.5 - angle ** 2 / 48 + angle ** 4 / 3840 : Math.sin(angle / 2) / angle
  return rotationFromQuat([(r[0] ?? 0) * scale, (r[1] ?? 0) * scale, (r[2] ?? 0) * scale, Math.cos(angle / 2)])
}

/**
 * A rotation from Euler angles: lowercase sequences are extrinsic (about the fixed axes, in order), uppercase intrinsic
 * (about the rotating axes).
 *
 * @param seq - One to three axes, such as 'z', 'xyz', 'ZYX' or 'zxz'.
 * @param angles - One angle per axis.
 * @param degrees - Whether the angles are in degrees.
 * @returns The rotation.
 * @example
 * ```ts
 * import { rotationFromEuler } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(rotationFromEuler('z', [90], true).apply([1, 0, 0])).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1, 0]
 * ```
 */
export function rotationFromEuler(seq: string, angles: readonly number[], degrees = false): Rotation {
  if (!/^([xyz]{1,3}|[XYZ]{1,3})$/.test(seq) || angles.length !== seq.length) {
    throw new WhiteboxError('value', `rotationFromEuler: ${seq} needs one angle per axis`)
  }
  const k = degrees ? Math.PI / 180 : 1
  const extrinsic = seq === seq.toLowerCase()
  let q: Quat = [0, 0, 0, 1]
  ;[...seq.toLowerCase()].forEach((c, i) => {
    const e = elementary(AXES[c] ?? 0, (angles[i] ?? 0) * k)
    // Extrinsic: later rotations act on the left; intrinsic: on the right.
    q = extrinsic ? qmul(e, q) : qmul(q, e)
  })
  return rotationFromQuat([...q])
}

/**
 * Spherical linear interpolation between two rotations (scipy's Slerp for two key rotations), at t in [0, 1].
 *
 * @param a - Start rotation.
 * @param b - End rotation.
 * @param t - Fraction.
 * @returns The interpolated rotation.
 * @example
 * ```ts
 * import { rotationFromEuler, slerp } from 'whitebox-engine/numerics/geometry'
 * Math.round(slerp(rotationFromEuler('z', [0]), rotationFromEuler('z', [90], true), 0.5).asEuler('xyz', true)[2]) // => 45
 * ```
 */
export function slerp(a: Rotation, b: Rotation, t: number): Rotation {
  const delta = a.inv().compose(b)
  const rv = delta.asRotvec()
  return a.compose(rotationFromRotvec(rv.map((v) => v * t)))
}

/** An affine map x -> A x + b in d dimensions. */
export interface AffineMap {
  readonly matrix: Value
  readonly translation: Value
  /** Map points `[n, d]` or `[d]` (differentiable in the points and the map). */
  readonly apply: (points: TensorLike) => Value
  /** this after other. */
  readonly compose: (other: AffineMap) => AffineMap
  readonly inverse: () => AffineMap
  /** The (d + 1) x (d + 1) homogeneous matrix. */
  readonly homogeneous: () => number[][]
}

/**
 * An affine map from its linear part and translation.
 *
 * @param matrix - A `[d, d]`.
 * @param translation - b `[d]` (default zero).
 * @returns The map.
 * @example
 * ```ts
 * import { affine } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(affine([[2, 0], [0, 1]], [1, 1]).apply([[1, 1]])) // => [[3, 2]]
 * ```
 */
export function affine(matrix: TensorLike, translation?: TensorLike): AffineMap {
  const [m] = promoteFloat(matrix) as [Value]
  const d = m.shape[0] ?? 0
  if (m.shape.length !== 2 || m.shape[1] !== d)
    throw new WhiteboxError('shape', 'affine: the linear part must be square')
  const [t] = promoteFloat(translation ?? new Array<number>(d).fill(0)) as [Value]
  const self: AffineMap = {
    matrix: m,
    translation: t,
    apply: (points) => {
      const [p] = promoteFloat(points) as [Value]
      const flat = p.shape.length === 1 ? reshape(p, [1, d]) : p
      const out = add(matmul(flat, swapaxes(m, 0, 1)), reshape(t, [1, d]))
      return p.shape.length === 1 ? reshape(out, [d]) : out
    },
    compose: (other) =>
      affine(matmul(m, other.matrix), add(reshape(matmul(m, reshape(other.translation, [d, 1])), [d]), t)),
    inverse: () => {
      const mi = inv(m)
      return affine(mi, mul(reshape(matmul(mi, reshape(t, [d, 1])), [d]), -1))
    },
    homogeneous: () => {
      const mv = toFlat(m as Tensor)
      const tv = toFlat(t as Tensor)
      return [
        ...Array.from({ length: d }, (_, i) => [...mv.slice(i * d, i * d + d), tv[i] ?? 0]),
        [...new Array<number>(d).fill(0), 1],
      ]
    },
  }
  return self
}

/** Points with their Hartley normalisation. */
export interface NormalisedPoints {
  readonly points: Tensor
  /** The 3 x 3 similarity T with normalised = T [x, y, 1]. */
  readonly transform: number[][]
}

/**
 * Hartley's normalisation of planar points: translate the centroid to the origin and scale so the root-mean-square
 * distance from it is sqrt(2) (as skimage; Hartley's paper uses the mean distance). Conditioning for the DLT and
 * eight-point algorithms.
 *
 * @param points - Points `[n, 2]`.
 * @returns The normalised points and the transform.
 * @example
 * ```ts
 * import { normalisePoints } from 'whitebox-engine/numerics/geometry'
 * normalisePoints([[0, 0], [2, 0], [0, 2], [2, 2]]).transform[0][0] // => 1
 * ```
 */
export function normalisePoints(points: TensorLike): NormalisedPoints {
  const t = concreteOrThrow(promoteFloat(points)[0] as Tensor, 'normalisePoints')
  const f = toFlat(t)
  const n = t.shape[0] ?? 0
  let cx = 0
  let cy = 0
  for (let i = 0; i < n; i++) {
    cx += (f[2 * i] ?? 0) / n
    cy += (f[2 * i + 1] ?? 0) / n
  }
  let meanSquare = 0
  for (let i = 0; i < n; i++) meanSquare += (((f[2 * i] ?? 0) - cx) ** 2 + ((f[2 * i + 1] ?? 0) - cy) ** 2) / n
  if (meanSquare === 0) throw new WhiteboxError('value', 'normalisePoints: all points coincide')
  const s = Math.SQRT2 / Math.sqrt(meanSquare)
  const out = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    out[2 * i] = s * ((f[2 * i] ?? 0) - cx)
    out[2 * i + 1] = s * ((f[2 * i + 1] ?? 0) - cy)
  }
  return {
    points: fromStorage(out, [n, 2], 'float64'),
    transform: [
      [s, 0, -s * cx],
      [0, s, -s * cy],
      [0, 0, 1],
    ],
  }
}

const mat3mul = (a: readonly number[][], b: readonly number[][]): number[][] =>
  a.map((row) => [0, 1, 2].map((j) => row.reduce((s, v, k) => s + v * (b[k]?.[j] ?? 0), 0)))

function invert3(m: readonly number[][]): number[][] {
  return toNested(toFlat(inv(m.map((r) => [...r])) as Tensor), 3)
}

function toNested(f: readonly number[], n: number): number[][] {
  return Array.from({ length: n }, (_, i) => f.slice(i * n, i * n + n))
}

/**
 * The right singular vector of the smallest singular value of a row-major m x 9 matrix. The SVD is the reduced one,
 * whose V has only min(m, 9) columns, so fewer than nine rows (four correspondences give eight) are padded with zero
 * rows: the singular vector that spans the null space is then the ninth column.
 */
function nullVector(rows: readonly number[][]): number[] {
  const padded = [
    ...rows.map((r) => [...r]),
    ...Array.from({ length: Math.max(0, 9 - rows.length) }, () => Array(9).fill(0)),
  ]
  const { vh } = svd(padded)
  return toFlat(vh as Tensor).slice(8 * 9, 9 * 9)
}

/**
 * The homography H with dst ~ H src from four or more point correspondences, by the normalised direct linear transform
 * (Hartley and Zisserman, algorithm 4.2), scaled so H[2][2] = 1 (as skimage's ProjectiveTransform.estimate).
 *
 * @param src - Source points `[n, 2]`, n >= 4.
 * @param dst - Destination points `[n, 2]`.
 * @returns The 3 x 3 homography.
 * @concept homography
 * @example
 * ```ts
 * import { homography } from 'whitebox-engine/numerics/geometry'
 * homography([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [2, 0], [2, 2], [0, 2]]).map((r) => r.map((v) => Math.round(v * 1e10) / 1e10)) // => [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
 * ```
 */
export function homography(src: TensorLike, dst: TensorLike): number[][] {
  const a = normalisePoints(src)
  const b = normalisePoints(dst)
  const n = a.points.shape[0] ?? 0
  if (n < 4 || b.points.shape[0] !== n)
    throw new WhiteboxError('value', 'homography: at least four correspondences are needed')
  const p = toFlat(a.points)
  const q = toFlat(b.points)
  const rows: number[][] = []
  for (let i = 0; i < n; i++) {
    const x = p[2 * i] ?? 0
    const y = p[2 * i + 1] ?? 0
    const u = q[2 * i] ?? 0
    const v = q[2 * i + 1] ?? 0
    rows.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
    rows.push([0, 0, 0, -x, -y, -1, v * x, v * y, v])
  }
  const h = toNested(nullVector(rows), 3)
  const full = mat3mul(mat3mul(invert3(b.transform), h), a.transform)
  const s = full[2]?.[2] ?? 1
  return full.map((r) => r.map((v) => v / s))
}

/**
 * Apply a homography to planar points (with the perspective division).
 *
 * @param h - The 3 x 3 matrix.
 * @param points - Points `[n, 2]`.
 * @returns Mapped points `[n, 2]` (differentiable in the points).
 * @example
 * ```ts
 * import { applyHomography } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(applyHomography([[1, 0, 0], [0, 1, 0], [0, 1, 1]], [[2, 1]])) // => [[1, 0.5]]
 * ```
 */
export function applyHomography(h: readonly (readonly number[])[], points: TensorLike): Value {
  const [p] = promoteFloat(points) as [Value]
  // [x, y] -> H [x, y, 1]: the first two columns of H act on the points, the third is added.
  const hom = add(
    matmul(p, swapaxes(sliceAxis(promoteFloat(h.map((r) => [...r]))[0] as Value, 1, 0, 2), 0, 1)),
    reshape(promoteFloat(h.map((r) => r[2] ?? 0))[0] as Value, [1, 3]),
  )
  return div(sliceAxis(hom, 1, 0, 2), sliceAxis(hom, 1, 2, 3))
}

/**
 * The fundamental matrix F with x2^T F x1 = 0 from eight or more correspondences, by the normalised eight-point
 * algorithm with the rank-2 constraint enforced by zeroing the smallest singular value; scaled to unit Frobenius norm
 * with F[2][2] >= 0.
 *
 * @param x1 - Points in the first image `[n, 2]`.
 * @param x2 - Matching points in the second `[n, 2]`.
 * @returns The 3 x 3 matrix.
 * @concept epipolar-geometry
 * @example
 * ```ts
 * import { fundamentalMatrix } from 'whitebox-engine/numerics/geometry'
 * const x1 = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [3, 1], [2, 3]]
 * fundamentalMatrix(x1, x1.map(([x, y]) => [x + 1, y])).length // => 3
 * ```
 */
export function fundamentalMatrix(x1: TensorLike, x2: TensorLike): number[][] {
  const a = normalisePoints(x1)
  const b = normalisePoints(x2)
  const n = a.points.shape[0] ?? 0
  if (n < 8 || b.points.shape[0] !== n)
    throw new WhiteboxError('value', 'fundamentalMatrix: at least eight correspondences are needed')
  const p = toFlat(a.points)
  const q = toFlat(b.points)
  const rows: number[][] = []
  for (let i = 0; i < n; i++) {
    const x = p[2 * i] ?? 0
    const y = p[2 * i + 1] ?? 0
    const u = q[2 * i] ?? 0
    const v = q[2 * i + 1] ?? 0
    rows.push([u * x, u * y, u, v * x, v * y, v, x, y, 1])
  }
  const f0 = toNested(nullVector(rows), 3)
  // Enforce rank 2.
  const { u, s, vh } = svd(f0)
  const sv = toFlat(s as Tensor)
  const uv = toNested(toFlat(u as Tensor), 3)
  const vv = toNested(toFlat(vh as Tensor), 3)
  const sdiag = [
    [sv[0] ?? 0, 0, 0],
    [0, sv[1] ?? 0, 0],
    [0, 0, 0],
  ]
  const f1 = mat3mul(mat3mul(uv, sdiag), vv)
  const bt = (b.transform as number[][]).map((_, i) => [0, 1, 2].map((j) => b.transform[j]?.[i] ?? 0))
  const full = mat3mul(mat3mul(bt, f1), a.transform)
  const norm = Math.hypot(...full.flat())
  const sign = (full[2]?.[2] ?? 0) < 0 ? -1 : 1
  return full.map((r) => r.map((v) => (sign * v) / norm))
}

/**
 * The orthogonal matrix R minimising |A R - B| (scipy.linalg.orthogonal_procrustes): R = U V^T from the SVD of A^T B,
 * with the scale sum(s). Differentiable through the SVD.
 *
 * @param a - Matrix `[n, d]`.
 * @param b - Matrix `[n, d]`.
 * @returns R and the scale.
 * @example
 * ```ts
 * import { orthogonalProcrustes } from 'whitebox-engine/numerics/geometry'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(orthogonalProcrustes([[1, 0], [0, 1]], [[0, 1], [-1, 0]]).rotation).map((r) => r.map((v) => Math.round(v * 1e12) / 1e12)) // => [[0, 1], [-1, 0]]
 * ```
 */
export function orthogonalProcrustes(a: TensorLike, b: TensorLike): { rotation: Value; scale: Value } {
  const [av, bv] = promoteFloat(a, b) as [Value, Value]
  const { u, s, vh } = svd(matmul(swapaxes(av, 0, 1), bv))
  return { rotation: matmul(u, vh), scale: sum(s) }
}

/**
 * Procrustes analysis (scipy.spatial.procrustes): centre both configurations, scale each to unit Frobenius norm, then
 * rotate and scale the second onto the first; the disparity is the remaining sum of squared differences.
 *
 * @param data1 - Reference points `[n, d]`.
 * @param data2 - Points to fit `[n, d]`.
 * @returns The standardised first configuration, the fitted second, and the disparity.
 * @example
 * ```ts
 * import { procrustes } from 'whitebox-engine/numerics/geometry'
 * import { item } from 'whitebox-engine/tensor'
 * // A rotated, scaled and shifted copy fits perfectly.
 * item(procrustes([[0, 0], [1, 0], [0, 2]], [[5, 5], [5, 7], [1, 5]]).disparity) < 1e-24 // => true
 * ```
 */
export function procrustes(data1: TensorLike, data2: TensorLike): { mtx1: Value; mtx2: Value; disparity: Value } {
  const [a0, b0] = promoteFloat(data1, data2) as [Value, Value]
  if (a0.shape.length !== 2 || a0.shape[0] !== b0.shape[0] || a0.shape[1] !== b0.shape[1])
    throw new WhiteboxError('shape', 'procrustes: inputs must have the same shape')
  const centre = (x: Value): Value => sub(x, mean(x, 0, true))
  const unit = (x: Value): Value => div(x, sqrt(sum(mul(x, x))))
  const mtx1 = unit(centre(a0))
  const b = unit(centre(b0))
  const { rotation, scale } = orthogonalProcrustes(mtx1, b)
  const mtx2 = mul(matmul(b, swapaxes(rotation, 0, 1)), scale)
  const d = sub(mtx1, mtx2)
  return { mtx1, mtx2, disparity: sum(mul(d, d)) }
}

/**
 * The Kabsch algorithm: the proper rotation R (det +1) and translation t minimising sum |R p_i + t - q_i|^2, from the SVD
 * of the cross-covariance with the reflection corrected, and the root-mean-square deviation after alignment.
 * Differentiable through the SVD.
 *
 * @param p - Points to move `[n, d]`.
 * @param q - Target points `[n, d]`.
 * @returns Rotation `[d, d]`, translation `[d]` and RMSD.
 * @concept rigid-alignment
 * @example
 * ```ts
 * import { kabsch } from 'whitebox-engine/numerics/geometry'
 * import { item } from 'whitebox-engine/tensor'
 * item(kabsch([[0, 0], [1, 0], [0, 1]], [[1, 1], [1, 2], [0, 1]]).rmsd) < 1e-12 // => true
 * ```
 */
export function kabsch(p: TensorLike, q: TensorLike): { rotation: Value; translation: Value; rmsd: Value } {
  const [pv, qv] = promoteFloat(p, q) as [Value, Value]
  const d = pv.shape[1] ?? 0
  const n = pv.shape[0] ?? 0
  const pc = mean(pv, 0, true)
  const qc = mean(qv, 0, true)
  const h = matmul(swapaxes(sub(pv, pc), 0, 1), sub(qv, qc))
  const { u, vh } = svd(h)
  // Correct a reflection: flip the last singular direction when det(V U^T) < 0.
  const sign = toFlat(det(matmul(swapaxes(vh, 0, 1), swapaxes(u, 0, 1))) as Tensor)[0] ?? 1
  const fix = fromStorage(
    Float64Array.from({ length: d * d }, (_, k) =>
      Math.floor(k / d) === k % d ? (k % d === d - 1 && sign < 0 ? -1 : 1) : 0,
    ),
    [d, d],
    'float64',
  )
  const rotation = matmul(matmul(swapaxes(vh, 0, 1), fix), swapaxes(u, 0, 1))
  const translation = reshape(sub(qc, matmul(pc, swapaxes(rotation, 0, 1))), [d])
  const moved = add(matmul(pv, swapaxes(rotation, 0, 1)), reshape(translation, [1, d]))
  const diff = sub(moved, qv)
  return { rotation, translation, rmsd: sqrt(div(sum(mul(diff, diff)), n)) }
}
