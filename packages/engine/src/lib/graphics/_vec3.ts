// Number-level 3-vector kernels on `Vec3` tuples, for the per-ray and per-pixel loops of the renderers (the
// differentiable forms are the tensor operations and `foundations/vectors.ts`). Dot products and norms come from the
// shared BLAS kernels.
import { OmaiError, type Vec3 } from '../base/index.ts'
import { ddot, dnrm2 } from '../numerics/linalg/blas.ts'

export const dot3 = (a: Vec3, b: Vec3): number => ddot(a, b)
export const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const sub3 = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
export const mul3 = (a: Vec3, b: Vec3): Vec3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
/** `a + s b`. */
export const madd3 = (a: Vec3, b: Vec3, s: number): Vec3 => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
// Adding 0 turns -0 into 0, so frames built from cross products compare equal to their written forms.
export const cross3 = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1] + 0,
  a[2] * b[0] - a[0] * b[2] + 0,
  a[0] * b[1] - a[1] * b[0] + 0,
]
export const length3 = (a: Vec3): number => dnrm2(a)

/** The unit vector along `a`; throws for the zero vector (or a non-finite one). */
export function normalise3(a: Vec3, what = 'vector'): Vec3 {
  const n = dnrm2(a)
  if (!(n > 0 && Number.isFinite(n))) throw new OmaiError('value', `graphics: ${what} must be non-zero and finite`)
  return [a[0] / n + 0, a[1] / n + 0, a[2] / n + 0]
}

/** The mirror direction of `d` about the unit normal `n`: `d - 2 (d . n) n`. */
export function reflect3(d: Vec3, n: Vec3): Vec3 {
  return madd3(d, n, -2 * dot3(d, n))
}

/**
 * Snell refraction of the unit direction `d` through the unit normal `n` (facing against `d`) with `eta = n1 / n2`;
 * `undefined` on total internal reflection.
 */
export function refract3(d: Vec3, n: Vec3, eta: number): Vec3 | undefined {
  const cosi = -dot3(d, n)
  const k = 1 - eta * eta * (1 - cosi * cosi)
  if (k < 0) return undefined
  return madd3(scale3(d, eta), n, eta * cosi - Math.sqrt(k))
}

/** Two unit vectors completing a right-handed orthonormal basis `(t, b, n)` with the unit normal `n` (Frisvad, revised). */
export function basis3(n: Vec3): [Vec3, Vec3] {
  const s = n[2] >= 0 ? 1 : -1
  const a = -1 / (s + n[2])
  const b = n[0] * n[1] * a
  return [
    [1 + s * n[0] * n[0] * a, s * b, -s * n[0]],
    [b, s + n[1] * n[1] * a, -n[1]],
  ]
}

/** Read a vector from a flat array at `3 i`. */
export const at3 = (xs: ArrayLike<number>, i: number): Vec3 => [xs[3 * i] ?? 0, xs[3 * i + 1] ?? 0, xs[3 * i + 2] ?? 0]

/** Check a vector has three finite entries. */
export function check3(a: readonly number[], what: string): Vec3 {
  if (a.length !== 3 || !a.every(Number.isFinite)) {
    throw new OmaiError('value', `graphics: ${what} must be three finite numbers`)
  }
  return [a[0] ?? 0, a[1] ?? 0, a[2] ?? 0]
}
