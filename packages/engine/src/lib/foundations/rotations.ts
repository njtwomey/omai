/**
 * Rotations in two and three dimensions: rotation matrices, Rodrigues' formula and axis-angle form, Euler angles in all
 * 24 conventions with gimbal-lock detection, quaternions (products, conversions, rotating vectors, slerp), the angle
 * between orientations, chordal and Karcher means of rotations, and a check that a matrix is a rotation, following
 * scipy's Rotation where they overlap.
 */

import { OmaiError } from '../base/index.ts'
import { type Rotation, rotationFromMatrix, rotationFromQuat, rotationFromRotvec } from '../numerics/geometry/index.ts'
import { det, eigh } from '../numerics/linalg/index.ts'
import {
  add,
  arctan2,
  concreteOrThrow,
  cos,
  div,
  item,
  less,
  matmul,
  mul,
  neg,
  onesLike,
  promoteFloat,
  sin,
  sliceAxis,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  toArray,
  toFlat,
  type Value,
  where,
  zerosLike,
} from '../tensor/index.ts'
import { crossProduct } from './vectors.ts'

const part = (v: Value, i: number): Value => squeeze(sliceAxis(v, -1, i, i + 1), -1)
const keep = (x: Value): Value => stack([x], -1)
const rows3 = (r: readonly (readonly Value[])[]): Value =>
  stack(
    r.map((row) => stack([...row], -1)),
    -2,
  )

/**
 * The 2-D rotation matrix by angle `t` counter-clockwise: `[[cos t, -sin t], [sin t, cos t]]`. Differentiable.
 *
 * @param theta - Angles `[...]` in radians.
 * @returns Matrices `[..., 2, 2]`.
 * @concept rotation-matrix
 * @example
 * ```ts
 * import { rotation2d } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rotation2d(0)) // => [[1, 0], [0, 1]]
 * ```
 */
export function rotation2d(theta: TensorLike): Value {
  const [t] = promoteFloat(theta) as [Value]
  const c = cos(t)
  const s = sin(t)
  return rows3([
    [c, neg(s)],
    [s, c],
  ])
}

/**
 * Rotate 2-D points about a centre by an angle (counter-clockwise). Differentiable.
 *
 * @param points - Points `[..., 2]`.
 * @param theta - The angle (radians), a scalar.
 * @param centre - The fixed point (default the origin).
 * @returns The rotated points `[..., 2]`.
 * @concept rotation-matrix
 * @example
 * ```ts
 * import { rotate2d } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rotate2d([2, 1], Math.PI, [1, 1])).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 1]
 * ```
 */
export function rotate2d(points: TensorLike, theta: TensorLike, centre: TensorLike = [0, 0]): Value {
  const [p, t, c] = promoteFloat(points, theta, centre) as [Value, Value, Value]
  return add(matmul(sub(p, c), swapaxes(rotation2d(t), -1, -2)), c)
}

/** A coordinate axis of 3-D space. */
export type Axis = 'x' | 'y' | 'z'

/**
 * The elementary rotation about a coordinate axis, `R_x`, `R_y` or `R_z` (active, right-handed). Differentiable.
 *
 * @param axis - 'x', 'y' or 'z'.
 * @param angle - Angles `[...]` in radians.
 * @returns Matrices `[..., 3, 3]`.
 * @concept rotation-matrix
 * @example
 * ```ts
 * import { elementaryRotation } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(elementaryRotation('z', Math.PI / 2)).map((r) => r.map(Math.round)) // => [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
 * ```
 */
export function elementaryRotation(axis: Axis, angle: TensorLike): Value {
  const [t] = promoteFloat(angle) as [Value]
  const c = cos(t)
  const s = sin(t)
  const o = zerosLike(t)
  const l = onesLike(t)
  if (axis === 'x') {
    return rows3([
      [l, o, o],
      [o, c, neg(s)],
      [o, s, c],
    ])
  }
  if (axis === 'y') {
    return rows3([
      [c, o, s],
      [o, l, o],
      [neg(s), o, c],
    ])
  }
  return rows3([
    [c, neg(s), o],
    [s, c, o],
    [o, o, l],
  ])
}

/**
 * The rotation matrix about a unit axis by an angle, by Rodrigues' formula `R = I + sin t K + (1 - cos t) K^2` with K
 * the cross-product matrix of the axis (normalised first). Differentiable.
 *
 * @param axis - Axes `[..., 3]` (nonzero).
 * @param angle - Angles `[...]` in radians.
 * @returns Matrices `[..., 3, 3]`.
 * @concept axis-angle
 * @example
 * ```ts
 * import { axisAngleToMatrix } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(axisAngleToMatrix([0, 0, 2], Math.PI)).map((r) => r.map(Math.round)) // => [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
 * ```
 */
export function axisAngleToMatrix(axis: TensorLike, angle: TensorLike): Value {
  const [a, t] = promoteFloat(axis, angle) as [Value, Value]
  const u = div(a, keep(sqrt(sum(mul(a, a), -1))))
  const [x, y, z] = [part(u, 0), part(u, 1), part(u, 2)]
  const s = sin(t)
  const v = sub(1, cos(t))
  const c = cos(t)
  return rows3([
    [add(c, mul(v, mul(x, x))), sub(mul(v, mul(x, y)), mul(s, z)), add(mul(v, mul(x, z)), mul(s, y))],
    [add(mul(v, mul(x, y)), mul(s, z)), add(c, mul(v, mul(y, y))), sub(mul(v, mul(y, z)), mul(s, x))],
    [sub(mul(v, mul(x, z)), mul(s, y)), add(mul(v, mul(y, z)), mul(s, x)), add(c, mul(v, mul(z, z)))],
  ])
}

/**
 * Rotate vectors about an axis by an angle with Rodrigues' rotation formula
 * `v cos t + (k x v) sin t + k (k . v)(1 - cos t)`. Differentiable.
 *
 * @param v - Vectors `[..., 3]`.
 * @param axis - Axes `[..., 3]` (nonzero; normalised).
 * @param angle - Angles `[...]` in radians.
 * @returns The rotated vectors `[..., 3]`.
 * @concept axis-angle
 * @example
 * ```ts
 * import { rotateAboutAxis } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rotateAboutAxis([1, 0, 0], [0, 0, 1], Math.PI / 2)).map(Math.round) // => [0, 1, 0]
 * ```
 */
export function rotateAboutAxis(v: TensorLike, axis: TensorLike, angle: TensorLike): Value {
  const [x, a, t] = promoteFloat(v, axis, angle) as [Value, Value, Value]
  const k = div(a, keep(sqrt(sum(mul(a, a), -1))))
  const c = keep(cos(t))
  return add(add(mul(x, c), mul(crossProduct(k, x), keep(sin(t)))), mul(k, mul(keep(sum(mul(k, x), -1)), sub(1, c))))
}

/**
 * The axis and angle of a rotation matrix (the angle in `[0, pi]`, the axis a unit vector; the x-axis for the
 * identity).
 *
 * @param matrix - A rotation matrix `[3, 3]`.
 * @returns `{ axis, angle }`.
 * @concept axis-angle
 * @example
 * ```ts
 * import { elementaryRotation, matrixToAxisAngle } from 'omai-engine/foundations'
 * matrixToAxisAngle(elementaryRotation('y', 0.5)).angle // => 0.5
 * ```
 */
export function matrixToAxisAngle(matrix: TensorLike): { readonly axis: number[]; readonly angle: number } {
  const rv = rotationFromMatrix(matrix).asRotvec()
  const angle = Math.hypot(...rv)
  return { axis: angle === 0 ? [1, 0, 0] : rv.map((v) => v / angle), angle }
}

/** An Euler-angle sequence: lowercase for extrinsic (fixed axes), uppercase for intrinsic (moving axes), as scipy. */
export interface EulerSequence {
  readonly seq: string
  readonly extrinsic: boolean
  /** Proper Euler angles repeat the first axis (zxz); Tait-Bryan angles use all three (xyz). */
  readonly kind: 'proper' | 'tait-bryan'
}

/**
 * All 24 Euler-angle conventions: the 6 Tait-Bryan and 6 proper axis sequences, each extrinsic and intrinsic.
 *
 * @concept euler-angles
 * @example
 * ```ts
 * import { eulerSequences } from 'omai-engine/foundations'
 * eulerSequences.length // => 24
 * eulerSequences.filter((s) => s.kind === 'proper' && s.extrinsic).map((s) => s.seq) // => ['xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz']
 * ```
 */
export const eulerSequences: readonly EulerSequence[] = (() => {
  const out: EulerSequence[] = []
  for (const extrinsic of [true, false]) {
    for (const kind of ['tait-bryan', 'proper'] as const) {
      for (const a of 'xyz') {
        for (const b of 'xyz') {
          if (b === a) continue
          const c = kind === 'proper' ? a : 'xyz'.replace(a, '').replace(b, '')
          const seq = `${a}${b}${c}`
          out.push({ seq: extrinsic ? seq : seq.toUpperCase(), extrinsic, kind })
        }
      }
    }
  }
  return out
})()

function parseSequence(seq: string): { axes: Axis[]; extrinsic: boolean } {
  if (!/^([xyz]{3}|[XYZ]{3})$/.test(seq)) throw new OmaiError('value', `bad Euler sequence ${seq}`)
  const axes = [...seq.toLowerCase()] as Axis[]
  if (axes[0] === axes[1] || axes[1] === axes[2]) throw new OmaiError('value', 'consecutive Euler axes must differ')
  return { axes, extrinsic: seq === seq.toLowerCase() }
}

/**
 * The rotation matrix of Euler angles in any of the 24 conventions: extrinsic `xyz` is `R_z(c) R_y(b) R_x(a)` (each
 * rotation about a fixed axis acts on the left), intrinsic `XYZ` is `R_x(a) R_y(b) R_z(c)` (about the moving axes).
 * Matches scipy's `Rotation.from_euler`. Differentiable.
 *
 * @param seq - The sequence, such as 'xyz', 'ZYX' or 'zxz'.
 * @param angles - Angles `[..., 3]` in radians.
 * @returns Matrices `[..., 3, 3]`.
 * @throws OmaiError with code `value` for a malformed sequence.
 * @concept euler-angles
 * @example
 * ```ts
 * import { eulerToMatrix } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(eulerToMatrix('ZYX', [Math.PI / 2, 0, 0])).map((r) => r.map(Math.round)) // => [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
 * ```
 */
export function eulerToMatrix(seq: string, angles: TensorLike): Value {
  const { axes, extrinsic } = parseSequence(seq)
  const [a] = promoteFloat(angles) as [Value]
  const ms = axes.map((axis, i) => elementaryRotation(axis, part(a, i)))
  const [m0, m1, m2] = ms as [Value, Value, Value]
  return extrinsic ? matmul(m2, matmul(m1, m0)) : matmul(m0, matmul(m1, m2))
}

/**
 * Euler angles of a rotation matrix in a convention (scipy's `as_euler`): the first and third in `[-pi, pi]`, the
 * middle in `[-pi/2, pi/2]` (Tait-Bryan) or `[0, pi]` (proper). At gimbal lock the third angle is set to zero.
 *
 * @param seq - The sequence.
 * @param matrix - A rotation matrix `[3, 3]`.
 * @returns The three angles in radians.
 * @concept euler-angles
 * @example
 * ```ts
 * import { eulerToMatrix, matrixToEuler } from 'omai-engine/foundations'
 * matrixToEuler('xyz', eulerToMatrix('xyz', [0.1, 0.2, 0.3])).map((v) => Math.round(v * 1e12) / 1e12) // => [0.1, 0.2, 0.3]
 * ```
 */
export function matrixToEuler(seq: string, matrix: TensorLike): [number, number, number] {
  parseSequence(seq)
  return rotationFromMatrix(matrix).asEuler(seq)
}

/** Result of {@link gimbalLock}. */
export interface GimbalLock {
  /** Whether the middle angle is within `tol` of a singular value. */
  readonly locked: boolean
  /** The distance of the middle angle from the nearest singular value (radians). */
  readonly margin: number
  /** For a locked configuration, the combination of the outer angles that still matters (their sum or difference). */
  readonly effective: number | null
}

/**
 * Detect gimbal lock: the middle Euler angle at a singular value (`+-pi/2` for Tait-Bryan, `0` or `pi` for proper
 * sequences), where the first and third axes line up and one degree of freedom is lost: only the sum or difference
 * of the outer angles then affects the rotation.
 *
 * @param seq - The sequence.
 * @param angles - The three angles in radians.
 * @param tol - The lock tolerance (default 1e-7, as scipy's warning).
 * @returns Whether it is locked, how far from lock, and the surviving outer-angle combination.
 * @concept gimbal-lock
 * @example
 * ```ts
 * import { gimbalLock } from 'omai-engine/foundations'
 * gimbalLock('xyz', [0.3, Math.PI / 2, 0.2]).locked // => true
 * gimbalLock('zxz', [0.3, 1, 0.2]).margin // => 1
 * ```
 */
export function gimbalLock(seq: string, angles: readonly [number, number, number], tol = 1e-7): GimbalLock {
  const { axes } = parseSequence(seq)
  const proper = axes[0] === axes[2]
  const [a, b, c] = angles
  const wrapped = Math.atan2(Math.sin(b), Math.cos(b))
  const margin = proper
    ? Math.min(Math.abs(wrapped), Math.PI - Math.abs(wrapped))
    : Math.abs(Math.abs(wrapped) - Math.PI / 2)
  const locked = margin <= tol
  if (!locked) return { locked, margin, effective: null }
  // The outer axes now coincide, in the same or the opposite direction: shifting a and c in opposite directions leaves
  // the rotation unchanged exactly when their sum is what survives.
  const at = (x: number, z: number): number[] => toFlat(eulerToMatrix(seq, [x, b, z]))
  const base = at(a, c)
  const shifted = at(a + 0.25, c - 0.25)
  const same = base.every((v, i) => Math.abs(v - (shifted[i] ?? 0)) < 1e-6)
  return { locked, margin, effective: same ? a + c : a - c }
}

/** Quaternions are `[x, y, z, w]` (scalar last), as scipy. */
const quatParts = (q: Value): [Value, Value, Value, Value] => [part(q, 0), part(q, 1), part(q, 2), part(q, 3)]

/**
 * The Hamilton product `p q` of quaternions `[x, y, z, w]` (scalar last): the rotation q followed by p.
 * Differentiable.
 *
 * @param p - Quaternions `[..., 4]`.
 * @param q - Quaternions `[..., 4]`.
 * @returns `p q` `[..., 4]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatMultiply } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatMultiply([1, 0, 0, 0], [0, 1, 0, 0])) // => [0, 0, 1, 0]
 * ```
 */
export function quatMultiply(p: TensorLike, q: TensorLike): Value {
  const [pv, qv] = promoteFloat(p, q) as [Value, Value]
  const [px, py, pz, pw] = quatParts(pv)
  const [qx, qy, qz, qw] = quatParts(qv)
  return stack(
    [
      add(add(mul(pw, qx), mul(px, qw)), sub(mul(py, qz), mul(pz, qy))),
      add(sub(mul(pw, qy), mul(px, qz)), add(mul(py, qw), mul(pz, qx))),
      add(add(mul(pw, qz), mul(px, qy)), sub(mul(pz, qw), mul(py, qx))),
      sub(sub(mul(pw, qw), mul(px, qx)), add(mul(py, qy), mul(pz, qz))),
    ],
    -1,
  )
}

/**
 * The conjugate `[-x, -y, -z, w]`: the inverse rotation of a unit quaternion. Differentiable.
 *
 * @param q - Quaternions `[..., 4]`.
 * @returns Conjugates `[..., 4]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatConjugate } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatConjugate([1, 2, 3, 4])) // => [-1, -2, -3, 4]
 * ```
 */
export function quatConjugate(q: TensorLike): Value {
  const [v] = promoteFloat(q) as [Value]
  return mul(v, [-1, -1, -1, 1])
}

/**
 * The inverse `q* / |q|^2` of a nonzero quaternion. Differentiable.
 *
 * @param q - Quaternions `[..., 4]`.
 * @returns Inverses `[..., 4]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatInverse } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatInverse([0, 0, 0, 2])) // => [0, 0, 0, 0.5]
 * ```
 */
export function quatInverse(q: TensorLike): Value {
  const [v] = promoteFloat(q) as [Value]
  return div(quatConjugate(v), keep(sum(mul(v, v), -1)))
}

/**
 * Scale quaternions to unit length. Differentiable away from zero.
 *
 * @param q - Quaternions `[..., 4]`.
 * @returns Unit quaternions `[..., 4]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatNormalize } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatNormalize([0, 0, 3, 4])) // => [0, 0, 0.6, 0.8]
 * ```
 */
export function quatNormalize(q: TensorLike): Value {
  const [v] = promoteFloat(q) as [Value]
  return div(v, keep(sqrt(sum(mul(v, v), -1))))
}

/**
 * The quaternion of a rotation about an axis by an angle: `[sin(t/2) u, cos(t/2)]` with u the unit axis.
 * Differentiable.
 *
 * @param axis - Axes `[..., 3]` (nonzero).
 * @param angle - Angles `[...]` in radians.
 * @returns Unit quaternions `[..., 4]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatFromAxisAngle } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatFromAxisAngle([0, 0, 1], Math.PI)).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0, 1, 0]
 * ```
 */
export function quatFromAxisAngle(axis: TensorLike, angle: TensorLike): Value {
  const [a, t] = promoteFloat(axis, angle) as [Value, Value]
  const u = div(a, keep(sqrt(sum(mul(a, a), -1))))
  const half = mul(t, 0.5)
  return stack([...[0, 1, 2].map((i) => mul(part(u, i), sin(half))), cos(half)], -1)
}

/**
 * The axis and angle of a quaternion (normalised first; angle in `[0, pi]` after choosing `w >= 0`).
 *
 * @param q - A quaternion `[4]`.
 * @returns `{ axis, angle }`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatToAxisAngle } from 'omai-engine/foundations'
 * quatToAxisAngle([0, Math.SQRT1_2, 0, Math.SQRT1_2]).angle // => 1.5707963267948966
 * ```
 */
export function quatToAxisAngle(q: readonly number[]): { readonly axis: number[]; readonly angle: number } {
  const rv = rotationFromQuat(q).asRotvec()
  const angle = Math.hypot(...rv)
  return { axis: angle === 0 ? [1, 0, 0] : rv.map((v) => v / angle), angle }
}

/**
 * The rotation matrix of a quaternion (normalised first). Differentiable.
 *
 * @param q - Quaternions `[..., 4]`.
 * @returns Matrices `[..., 3, 3]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatToMatrix } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatToMatrix([0, 0, 0, 1])) // => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
 * ```
 */
export function quatToMatrix(q: TensorLike): Value {
  const [x, y, z, w] = quatParts(quatNormalize(q))
  const two = (a: Value, b: Value): Value => mul(2, mul(a, b))
  return rows3([
    [sub(1, add(two(y, y), two(z, z))), sub(two(x, y), two(z, w)), add(two(x, z), two(y, w))],
    [add(two(x, y), two(z, w)), sub(1, add(two(x, x), two(z, z))), sub(two(y, z), two(x, w))],
    [sub(two(x, z), two(y, w)), add(two(y, z), two(x, w)), sub(1, add(two(x, x), two(y, y)))],
  ])
}

/**
 * The unit quaternion of a rotation matrix (scipy's `from_matrix`), canonical with `w >= 0`.
 *
 * @param matrix - A rotation matrix `[3, 3]`.
 * @returns `[x, y, z, w]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { matrixToQuat } from 'omai-engine/foundations'
 * matrixToQuat([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) // => [0, 0, 0, 1]
 * ```
 */
export function matrixToQuat(matrix: TensorLike): [number, number, number, number] {
  return rotationFromMatrix(matrix).asQuat(true)
}

/**
 * Rotate vectors by quaternions, `q v q*` (normalised first), as `v + 2w (u x v) + 2 u x (u x v)`. Differentiable.
 *
 * @param q - Quaternions `[..., 4]`.
 * @param v - Vectors `[..., 3]`.
 * @returns Rotated vectors `[..., 3]`.
 * @concept quaternions
 * @example
 * ```ts
 * import { quatFromAxisAngle, quatRotate } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatRotate(quatFromAxisAngle([0, 0, 1], Math.PI / 2), [1, 0, 0])).map(Math.round) // => [0, 1, 0]
 * ```
 */
export function quatRotate(q: TensorLike, v: TensorLike): Value {
  const unit = quatNormalize(q)
  const [vv] = promoteFloat(v) as [Value]
  const u = sliceAxis(unit, -1, 0, 3)
  const w = sliceAxis(unit, -1, 3, 4)
  const uv = crossProduct(u, vv)
  return add(vv, add(mul(mul(w, 2), uv), mul(2, crossProduct(u, uv))))
}

/**
 * Spherical linear interpolation of unit quaternions along the shorter arc (q is negated when `p . q < 0`): constant
 * angular speed from p (t = 0) to q (t = 1). Falls back to normalised linear interpolation for nearly equal
 * rotations. Differentiable.
 *
 * @param p - Quaternions `[..., 4]` (unit).
 * @param q - Quaternions `[..., 4]` (unit).
 * @param t - Interpolation parameters `[...]`.
 * @returns Unit quaternions `[..., 4]`.
 * @concept slerp
 * @example
 * ```ts
 * import { quatSlerp } from 'omai-engine/foundations'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(quatSlerp([0, 0, 0, 1], [0, 0, 1, 0], 0.5)).map((v) => Math.round(v * 1e12) / 1e12) // => [0, 0, 0.707106781187, 0.707106781187]
 * ```
 */
export function quatSlerp(p: TensorLike, q: TensorLike, t: TensorLike): Value {
  const [pv, qv0, tv] = promoteFloat(p, q, t) as [Value, Value, Value]
  const d = sum(mul(pv, qv0), -1)
  const qv = mul(qv0, keep(where(less(d, 0), -1, 1)))
  // The angle between them, by Kahan's stable formula (half the angle of the rotation between them).
  const omega = mul(
    2,
    arctan2(sqrt(sum(mul(sub(pv, qv), sub(pv, qv)), -1)), sqrt(sum(mul(add(pv, qv), add(pv, qv)), -1))),
  )
  const small = less(omega, 1e-8)
  const safe = where(small, 1, omega)
  const sinO = sin(safe)
  const a = where(small, sub(1, tv), div(sin(mul(sub(1, tv), safe)), sinO))
  const b = where(small, tv, div(sin(mul(tv, safe)), sinO))
  return quatNormalize(add(mul(pv, keep(a)), mul(qv, keep(b))))
}

/**
 * The angle of the rotation that takes one orientation to another (the geodesic distance on SO(3)), from unit
 * quaternions: `2 atan2(|p - s q|, |p + s q|)` with `s` the sign of `p . q`, in `[0, pi]`. Differentiable.
 *
 * @param p - Quaternions `[..., 4]` (unit).
 * @param q - Quaternions `[..., 4]` (unit).
 * @returns Angles `[...]`.
 * @concept rotation-distance
 * @example
 * ```ts
 * import { quatFromAxisAngle, rotationDistance } from 'omai-engine/foundations'
 * import { item } from 'omai-engine/tensor'
 * item(rotationDistance([0, 0, 0, 1], quatFromAxisAngle([1, 0, 0], 0.5))) // => 0.5
 * ```
 */
export function rotationDistance(p: TensorLike, q: TensorLike): Value {
  const [pv, qv0] = promoteFloat(p, q) as [Value, Value]
  const qv = mul(qv0, keep(where(less(sum(mul(pv, qv0), -1), 0), -1, 1)))
  const minus = sub(pv, qv)
  const plus = add(pv, qv)
  return mul(4, arctan2(sqrt(sum(mul(minus, minus), -1)), sqrt(sum(mul(plus, plus), -1))))
}

const canonical = (q: readonly number[]): number[] => ((q[3] ?? 0) < 0 ? q.map((v) => -v) : [...q])

/**
 * The chordal L2 mean of rotations (scipy's `Rotation.mean`, Markley et al. 2007): the eigenvector of the largest
 * eigenvalue of `sum_i w_i q_i q_i^T`, which minimises the weighted sum of squared Frobenius distances between rotation
 * matrices. Insensitive to the sign of each quaternion.
 *
 * @param quats - Quaternions `[n, 4]`.
 * @param weights - Non-negative weights `[n]` (default equal).
 * @returns The mean quaternion `[x, y, z, w]` with `w >= 0`.
 * @concept rotation-averaging
 * @example
 * ```ts
 * import { meanRotation, quatFromAxisAngle } from 'omai-engine/foundations'
 * import { toFlat } from 'omai-engine/tensor'
 * const qs = [toFlat(quatFromAxisAngle([0, 0, 1], 0.2)), toFlat(quatFromAxisAngle([0, 0, 1], -0.2))]
 * meanRotation(qs) // => [0, 0, 0, 1]
 * ```
 */
export function meanRotation(quats: TensorLike, weights?: TensorLike): [number, number, number, number] {
  const [q] = promoteFloat(quats) as [Value]
  const qs = concreteOrThrow(q, 'meanRotation') as Tensor
  if (qs.shape.length !== 2 || qs.shape[1] !== 4) throw new OmaiError('shape', 'meanRotation: quaternions [n, 4]')
  const n = qs.shape[0] ?? 0
  const w = weights === undefined ? Array.from({ length: n }, () => 1) : toFlat(promoteFloat(weights)[0] as Value)
  const unit = toArray(quatNormalize(qs)) as number[][]
  const m = Array.from({ length: 4 }, (_, i) =>
    Array.from({ length: 4 }, (_, j) => unit.reduce((s, u, k) => s + (w[k] ?? 0) * (u[i] ?? 0) * (u[j] ?? 0), 0)),
  )
  const { eigenvectors } = eigh(m)
  const v = toArray(eigenvectors) as number[][]
  const top = v.map((row) => row[3] ?? 0)
  return canonical(top).map((x) => (Object.is(x, -0) || Math.abs(x) < 1e-300 ? 0 : x)) as [
    number,
    number,
    number,
    number,
  ]
}

/** Result of {@link karcherMean}. */
export interface KarcherMean {
  readonly quaternion: [number, number, number, number]
  /** The geodesic (Riemannian) mean's residual tangent step at each iteration. */
  readonly steps: number[]
  readonly converged: boolean
}

/**
 * The geodesic (Karcher, Riemannian L2) mean of rotations by fixed-point iteration: map every rotation into the tangent
 * space at the current mean with the logarithm (rotation vectors), average, and step along the exponential, until the
 * step is below `tol`. It minimises the sum of squared rotation angles, unlike the chordal {@link meanRotation}.
 *
 * @param quats - Quaternions `[n, 4]`.
 * @param options - Weights, start (default the chordal mean), tolerance (1e-12) and iteration cap (100).
 * @returns The mean, the step norms and whether it converged.
 * @concept rotation-averaging
 * @example
 * ```ts
 * import { karcherMean, quatFromAxisAngle } from 'omai-engine/foundations'
 * import { toFlat } from 'omai-engine/tensor'
 * const qs = [0.1, 0.4, 0.7].map((t) => toFlat(quatFromAxisAngle([1, 0, 0], t)))
 * Math.round(karcherMean(qs).quaternion[0] * 1e12) / 1e12 // => 0.198669330795
 * ```
 */
export function karcherMean(
  quats: TensorLike,
  options: { readonly weights?: TensorLike; readonly tol?: number; readonly maxIter?: number } = {},
): KarcherMean {
  const [q] = promoteFloat(quats) as [Value]
  const qs = toArray(q) as number[][]
  const w = options.weights === undefined ? qs.map(() => 1) : toFlat(promoteFloat(options.weights)[0] as Value)
  const total = w.reduce((s, x) => s + x, 0)
  const rotations = qs.map((x) => rotationFromQuat(x))
  let mean: Rotation = rotationFromQuat(meanRotation(qs, options.weights))
  const steps: number[] = []
  const tol = options.tol ?? 1e-12
  for (let it = 0; it < (options.maxIter ?? 100); it++) {
    const step = [0, 0, 0]
    for (const [k, r] of rotations.entries()) {
      const rv = mean.inv().compose(r).asRotvec()
      for (let i = 0; i < 3; i++) step[i] = (step[i] ?? 0) + ((w[k] ?? 0) * (rv[i] ?? 0)) / total
    }
    const size = Math.hypot(...step)
    steps.push(size)
    mean = mean.compose(rotationFromRotvec(step))
    if (size <= tol) return { quaternion: mean.asQuat(true), steps, converged: true }
  }
  return { quaternion: mean.asQuat(true), steps, converged: false }
}

/**
 * Whether a matrix is a rotation: orthogonal (`R^T R = I`) with determinant +1, within a tolerance.
 *
 * @param matrix - A square matrix `[n, n]`.
 * @param tol - Tolerance (default 1e-9).
 * @returns `true` for a proper rotation.
 * @concept rotation-matrix
 * @example
 * ```ts
 * import { isRotationMatrix } from 'omai-engine/foundations'
 * isRotationMatrix([[0, -1], [1, 0]]) // => true
 * isRotationMatrix([[1, 0], [0, -1]]) // => false
 * ```
 */
export function isRotationMatrix(matrix: TensorLike, tol = 1e-9): boolean {
  const [m] = promoteFloat(matrix) as [Value]
  const n = m.shape[0] ?? 0
  if (m.shape.length !== 2 || m.shape[1] !== n) return false
  const gram = toArray(matmul(swapaxes(m, 0, 1), m)) as number[][]
  const orthogonal = gram.every((row, i) => row.every((v, j) => Math.abs(v - (i === j ? 1 : 0)) <= tol))
  if (!orthogonal) return false
  // An orthogonal matrix has determinant +-1; the sign tells rotations from reflections.
  return item(det(m)) > 0
}
