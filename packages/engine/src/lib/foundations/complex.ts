/**
 * Complex analysis for teaching: polar form, principal logarithms, powers, roots and the trigonometric functions, n-th
 * roots and roots of unity, domain colouring of complex functions, Mobius transformations with composition, fixed
 * points and classification, standard conformal maps and the images of grids under them, the Cauchy-Riemann check, and
 * contour integrals with residues, Laurent coefficients, winding numbers and the argument principle.
 */

import { jvp } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import type { Pixels } from '../scene/index.ts'
import {
  abs,
  absComplex,
  add,
  angle,
  type ComplexLike,
  type ComplexValue,
  cDiv,
  cExp,
  cis,
  cMul,
  complex,
  cos,
  cosh,
  div,
  item,
  less,
  linspace,
  log,
  mul,
  neg,
  onesLike,
  promoteFloat,
  sin,
  sinh,
  sqrt,
  sub,
  sum,
  toArray,
  toFlat,
  type Value,
  where,
  zerosLike,
} from '../tensor/index.ts'

/** A complex function written with complex-pair operations, applied elementwise. */
export type ComplexFunction = (z: ComplexValue) => ComplexValue

const lit = (z: ComplexNumber | number): ComplexNumber => (typeof z === 'number' ? { re: z, im: 0 } : z)
const pair = (z: ComplexNumber | number): ComplexValue => {
  const w = lit(z)
  return complex(w.re, w.im)
}

/**
 * The polar form `z = r e^(i theta)`: modulus and principal argument in `(-pi, pi]`. Differentiable away from 0.
 *
 * @param z - Complex values (a pair or a real array).
 * @returns `{ r, theta }`.
 * @concept complex-polar-form
 * @example
 * ```ts
 * import { toPolarForm } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(toPolarForm({ re: [0], im: [2] }).theta) // => [1.5707963267948966]
 * ```
 */
export function toPolarForm(z: ComplexLike): { readonly r: Value; readonly theta: Value } {
  return { r: absComplex(z), theta: angle(z) }
}

/**
 * The principal logarithm `log z = log |z| + i arg z`, with the branch cut along the negative real axis.
 * Differentiable away from the cut and 0.
 *
 * @param z - Complex values.
 * @returns log z.
 * @concept complex-logarithm
 * @example
 * ```ts
 * import { complexLog } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(complexLog({ re: [-1], im: [0] }).im) // => [3.141592653589793]
 * ```
 */
export function complexLog(z: ComplexLike): ComplexValue {
  const c = complex(z)
  return { re: log(absComplex(c)), im: angle(c) }
}

/**
 * The principal power `z^w = exp(w log z)` (with `0^w = 0`). Differentiable away from the branch cut.
 *
 * @param z - Bases.
 * @param w - Exponents (complex or real).
 * @returns z^w.
 * @concept complex-powers
 * @example
 * ```ts
 * import { complexPow } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * // i^i = e^(-pi/2), a real number.
 * toArray(complexPow({ re: [0], im: [1] }, { re: [0], im: [1] }).re) // => [0.20787957635076193]
 * ```
 */
export function complexPow(z: ComplexLike, w: ComplexLike): ComplexValue {
  const c = complex(z)
  const zero = less(absComplex(c), 1e-300)
  const safe = { re: where(zero, 1, c.re), im: where(zero, 0, c.im) }
  const p = cExp(cMul(complex(w), complexLog(safe)))
  return { re: where(zero, 0, p.re), im: where(zero, 0, p.im) }
}

/**
 * The principal square root (real part >= 0), in the cancellation-free form. Differentiable away from the cut.
 *
 * @param z - Complex values.
 * @returns sqrt z.
 * @concept complex-powers
 * @example
 * ```ts
 * import { complexSqrt } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(complexSqrt({ re: [-4], im: [0] }).im) // => [2]
 * ```
 */
export function complexSqrt(z: ComplexLike): ComplexValue {
  const c = complex(z)
  const r = absComplex(c)
  // t = sqrt((r + |x|) / 2); for x >= 0: (t, y / 2t); for x < 0: (|y| / 2t, sign(y) t).
  const t = sqrt(mul(add(r, abs(c.re)), 0.5))
  const tSafe = where(less(t, 1e-300), 1, t)
  const neg0 = less(c.re, 0)
  const signY = where(less(c.im, 0), -1, 1)
  const re = where(neg0, div(abs(c.im), mul(tSafe, 2)), t)
  const im = where(neg0, mul(signY, t), div(c.im, mul(tSafe, 2)))
  return { re: where(less(t, 1e-300), 0, re), im: where(less(t, 1e-300), 0, im) }
}

/**
 * The complex sine `sin(x + iy) = sin x cosh y + i cos x sinh y`. Differentiable.
 *
 * @param z - Complex values.
 * @returns sin z.
 * @concept complex-functions
 * @example
 * ```ts
 * import { complexSin } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(complexSin({ re: [0], im: [1] }).im) // => [1.1752011936438014]
 * ```
 */
export function complexSin(z: ComplexLike): ComplexValue {
  const c = complex(z)
  return { re: mul(sin(c.re), cosh(c.im)), im: mul(cos(c.re), sinh(c.im)) }
}

/**
 * The complex cosine `cos(x + iy) = cos x cosh y - i sin x sinh y`. Differentiable.
 *
 * @param z - Complex values.
 * @returns cos z.
 * @concept complex-functions
 * @example
 * ```ts
 * import { complexCos } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(complexCos({ re: [0], im: [1] }).re) // => [1.5430806348152437]
 * ```
 */
export function complexCos(z: ComplexLike): ComplexValue {
  const c = complex(z)
  return { re: mul(cos(c.re), cosh(c.im)), im: neg(mul(sin(c.re), sinh(c.im))) }
}

/**
 * The n distinct n-th roots of a complex number by de Moivre: `|z|^(1/n) e^(i (arg z + 2 pi k) / n)`, k = 0..n-1,
 * evenly spaced on a circle.
 *
 * @param z - The number (complex or real).
 * @param n - The order (>= 1).
 * @returns The roots `[n]`, the principal root first.
 * @concept roots-of-unity
 * @example
 * ```ts
 * import { nthRoots } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(nthRoots(-8, 3).re).map((v) => Math.round(v * 1e12) / 1e12) // => [1, -2, 1]
 * ```
 */
export function nthRoots(z: ComplexNumber | number, n: number): ComplexValue {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'nthRoots: n must be a positive integer')
  const w = lit(z)
  const r = Math.hypot(w.re, w.im) ** (1 / n)
  const theta = Math.atan2(w.im, w.re)
  const angles = Array.from({ length: n }, (_, k) => (theta + 2 * Math.PI * k) / n)
  const c = cis(angles)
  return { re: mul(c.re, r), im: mul(c.im, r) }
}

/**
 * The n-th roots of unity `e^(2 pi i k / n)`, k = 0..n-1: the vertices of a regular n-gon, a cyclic group under
 * multiplication, summing to zero for n > 1.
 *
 * @param n - The order (>= 1).
 * @returns The roots `[n]`.
 * @concept roots-of-unity
 * @example
 * ```ts
 * import { rootsOfUnity } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(rootsOfUnity(4).im).map((v) => Math.round(v)) // => [0, 1, 0, -1]
 * ```
 */
export function rootsOfUnity(n: number): ComplexValue {
  return nthRoots(1, n)
}

/** A rectangle of the complex plane: `[[reMin, reMax], [imMin, imMax]]`. */
export type ComplexBox = readonly [readonly [number, number], readonly [number, number]]

/**
 * A grid of complex numbers over a rectangle, laid out like an image: rows run from the top (largest imaginary part)
 * down, columns from left to right.
 *
 * @param box - The rectangle.
 * @param size - Columns and rows `[nx, ny]` (default 200 x 200).
 * @returns The grid `[ny, nx]`.
 * @concept domain-colouring
 * @example
 * ```ts
 * import { complexGrid } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(complexGrid([[-1, 1], [-1, 1]], [3, 2]).im) // => [[1, 1, 1], [-1, -1, -1]]
 * ```
 */
export function complexGrid(box: ComplexBox, size: readonly [number, number] = [200, 200]): ComplexValue {
  const [nx, ny] = size
  const xs = toFlat(linspace(box[0][0], box[0][1], nx))
  const ys = toFlat(linspace(box[1][1], box[1][0], ny))
  return complex(
    ys.map(() => xs),
    ys.map((y) => xs.map(() => y)),
  )
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  // h in [0, 1).
  const k = (n: number): number => (n + h * 12) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number): number => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1))
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))]
}

/** Result of {@link domainColouring}. */
export interface DomainColouring {
  /** The picture: hue from the argument, brightness bands from log2 |f|. */
  readonly pixels: Pixels
  readonly modulus: number[][]
  readonly argument: number[][]
}

/**
 * Domain colouring of a complex function over a rectangle: each point coloured by the argument of f (the hue goes
 * once round the colour wheel per turn, so a zero of order k shows every colour k times counter-clockwise and a pole
 * clockwise) and shaded by `log2 |f|` (bands double the modulus), the standard picture of zeros, poles and branch
 * cuts. The pixels can be drawn with the `image` scene node.
 *
 * @param f - The function.
 * @param box - The rectangle.
 * @param options - Grid size `[nx, ny]` (default 200 x 200) and whether to shade modulus bands (default true).
 * @returns The pixels and the modulus and argument on the grid.
 * @concept domain-colouring
 * @example
 * ```ts
 * import { domainColouring } from 'mdml-engine/foundations'
 * import { cMul } from 'mdml-engine/tensor'
 * domainColouring((z) => cMul(z, z), [[-1, 1], [-1, 1]], { size: [8, 6] }).pixels.rgba.length // => 192
 * ```
 */
export function domainColouring(
  f: ComplexFunction,
  box: ComplexBox,
  options: { readonly size?: readonly [number, number]; readonly bands?: boolean } = {},
): DomainColouring {
  const [nx, ny] = options.size ?? [200, 200]
  const w = f(complexGrid(box, [nx, ny]))
  const modulus = toArray(absComplex(w)) as number[][]
  const argument = toArray(angle(w)) as number[][]
  const rgba = new Uint8ClampedArray(nx * ny * 4)
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nx; j++) {
      const m = modulus[i]?.[j] ?? Number.NaN
      const a = argument[i]?.[j] ?? 0
      const p = 4 * (i * nx + j)
      if (!Number.isFinite(m)) {
        rgba.set([255, 255, 255, 255], p)
        continue
      }
      const h = (((a / (2 * Math.PI)) % 1) + 1) % 1
      const band = options.bands === false ? 0 : Math.log2(Math.max(m, 1e-300)) % 1
      const l = options.bands === false ? 0.5 : 0.35 + 0.25 * ((band + 1) % 1)
      const [r, g, b] = hslToRgb(h, 0.9, l)
      rgba.set([r, g, b, 255], p)
    }
  }
  return { pixels: { width: nx, height: ny, rgba }, modulus, argument }
}

/** A Mobius transformation `z -> (a z + b) / (c z + d)` with `ad - bc != 0`. */
export interface Mobius {
  readonly a: ComplexNumber
  readonly b: ComplexNumber
  readonly c: ComplexNumber
  readonly d: ComplexNumber
}

/** The kinds of Mobius transformation, by the normalised trace. */
export type MobiusKind = 'identity' | 'elliptic' | 'parabolic' | 'hyperbolic' | 'loxodromic'

/**
 * A Mobius (linear fractional) transformation `(a z + b) / (c z + d)`: the conformal automorphisms of the Riemann
 * sphere, mapping circles and lines to circles and lines.
 *
 * @param a - Coefficient (complex or real).
 * @param b - Coefficient.
 * @param c - Coefficient.
 * @param d - Coefficient.
 * @returns The transformation.
 * @throws MdmlError with code `value` when `ad - bc = 0`.
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { mobius } from 'mdml-engine/foundations'
 * mobius(1, 0, 0, 2).d // => { re: 2, im: 0 }
 * ```
 */
export function mobius(
  a: ComplexNumber | number,
  b: ComplexNumber | number,
  c: ComplexNumber | number,
  d: ComplexNumber | number,
): Mobius {
  const m = { a: lit(a), b: lit(b), c: lit(c), d: lit(d) }
  const det = C.sub(C.mul(m.a, m.d), C.mul(m.b, m.c))
  if (Math.hypot(det.re, det.im) === 0) throw new MdmlError('value', 'mobius: ad - bc must not vanish')
  return m
}

/**
 * Apply a Mobius transformation to complex values (the pole `-d/c` maps to infinity). Differentiable.
 *
 * @param m - The transformation.
 * @param z - Complex values.
 * @returns `(a z + b) / (c z + d)`.
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { mobius, mobiusApply } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * // The Cayley map sends i to 0.
 * toArray(mobiusApply(mobius({ re: 1, im: 0 }, { re: 0, im: -1 }, 1, { re: 0, im: 1 }), { re: [0], im: [1] }).re) // => [0]
 * ```
 */
export function mobiusApply(m: Mobius, z: ComplexLike): ComplexValue {
  const w = complex(z)
  const num = add2(cMul(pair(m.a), w), pair(m.b))
  const den = add2(cMul(pair(m.c), w), pair(m.d))
  return cDiv(num, den)
}

const add2 = (p: ComplexValue, q: ComplexValue): ComplexValue => ({ re: add(p.re, q.re), im: add(p.im, q.im) })

/**
 * Compose Mobius transformations: `(m1 o m2)(z) = m1(m2(z))`, the product of their matrices `[[a, b], [c, d]]`.
 *
 * @param m1 - Applied second.
 * @param m2 - Applied first.
 * @returns The composition.
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { mobius, mobiusCompose } from 'mdml-engine/foundations'
 * mobiusCompose(mobius(2, 0, 0, 1), mobius(1, 1, 0, 1)).b // => { re: 2, im: 0 }
 * ```
 */
export function mobiusCompose(m1: Mobius, m2: Mobius): Mobius {
  return mobius(
    C.add(C.mul(m1.a, m2.a), C.mul(m1.b, m2.c)),
    C.add(C.mul(m1.a, m2.b), C.mul(m1.b, m2.d)),
    C.add(C.mul(m1.c, m2.a), C.mul(m1.d, m2.c)),
    C.add(C.mul(m1.c, m2.b), C.mul(m1.d, m2.d)),
  )
}

/**
 * The inverse Mobius transformation `(d z - b) / (-c z + a)`.
 *
 * @param m - The transformation.
 * @returns Its inverse.
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { mobius, mobiusInverse } from 'mdml-engine/foundations'
 * mobiusInverse(mobius(1, 3, 0, 1)).b // => { re: -3, im: 0 }
 * ```
 */
export function mobiusInverse(m: Mobius): Mobius {
  const neg1 = (z: ComplexNumber): ComplexNumber => ({ re: -z.re || 0, im: -z.im || 0 })
  return mobius(m.d, neg1(m.b), neg1(m.c), m.a)
}

/**
 * The fixed points of a Mobius transformation, the roots of `c z^2 + (d - a) z - b = 0` (one finite point and infinity
 * when c = 0 and a != d; just infinity for a translation; everything for the identity).
 *
 * @param m - The transformation.
 * @returns The finite fixed points (with `infinite: true` when infinity is fixed).
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { mobius, mobiusFixedPoints } from 'mdml-engine/foundations'
 * mobiusFixedPoints(mobius(0, 1, 1, 0)).points // => [{ re: -1, im: 0 }, { re: 1, im: 0 }]
 * ```
 */
export function mobiusFixedPoints(m: Mobius): { readonly points: ComplexNumber[]; readonly infinite: boolean } {
  const tiny = (z: ComplexNumber): boolean => Math.hypot(z.re, z.im) <= 1e-14
  const clean = (z: ComplexNumber): ComplexNumber => ({
    re: Math.abs(z.re) < 1e-15 ? 0 : z.re,
    im: Math.abs(z.im) < 1e-15 ? 0 : z.im,
  })
  const dma = C.sub(m.d, m.a)
  if (tiny(m.c)) {
    if (tiny(dma)) return { points: [], infinite: true }
    return { points: [clean(C.div(m.b, dma))], infinite: true }
  }
  // z = ((a - d) +- sqrt((d - a)^2 + 4 b c)) / 2c
  const disc = C.sqrt(C.add(C.mul(dma, dma), C.mul(C.mul(m.b, m.c), 4)))
  const two = C.mul(m.c, 2)
  const amd = C.sub(m.a, m.d)
  const pts = [C.div(C.sub(amd, disc), two), C.div(C.add(amd, disc), two)]
  const unique = tiny(disc) ? [pts[0] as ComplexNumber] : pts
  return { points: unique.map(clean).sort((p, q) => p.re - q.re || p.im - q.im), infinite: false }
}

/**
 * Classify a Mobius transformation by its normalised squared trace `t = (a + d)^2 / (ad - bc)`: parabolic when t = 4
 * (one fixed point), elliptic for real t in [0, 4) (a rotation about two fixed points), hyperbolic for real t > 4 (a
 * dilation between them), loxodromic otherwise.
 *
 * @param m - The transformation.
 * @param tol - Tolerance for the comparisons (default 1e-12).
 * @returns The kind and the normalised squared trace.
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { classifyMobius, mobius } from 'mdml-engine/foundations'
 * classifyMobius(mobius(1, 1, 0, 1)).kind // => 'parabolic'
 * classifyMobius(mobius(2, 0, 0, 1)).kind // => 'hyperbolic'
 * ```
 */
export function classifyMobius(m: Mobius, tol = 1e-12): { readonly kind: MobiusKind; readonly trace: ComplexNumber } {
  const det = C.sub(C.mul(m.a, m.d), C.mul(m.b, m.c))
  const tr = C.add(m.a, m.d)
  const t = C.div(C.mul(tr, tr), det)
  const identity =
    Math.hypot(m.b.re, m.b.im, m.c.re, m.c.im) <= tol && Math.hypot(C.sub(m.a, m.d).re, C.sub(m.a, m.d).im) <= tol
  if (identity) return { kind: 'identity', trace: t }
  const real = Math.abs(t.im) <= tol * Math.max(1, Math.abs(t.re))
  const kind: MobiusKind = !real
    ? 'loxodromic'
    : Math.abs(t.re - 4) <= tol * 4
      ? 'parabolic'
      : t.re >= 0 && t.re < 4
        ? 'elliptic'
        : t.re > 4
          ? 'hyperbolic'
          : 'loxodromic'
  return { kind, trace: t }
}

/**
 * The unique Mobius transformation sending three distinct points to three distinct points, through cross-ratios: the
 * map to `(0, 1, infinity)` from each triple, composed.
 *
 * @param from - Three distinct points z1, z2, z3.
 * @param to - Their images w1, w2, w3.
 * @returns The transformation.
 * @concept mobius-transformations
 * @example
 * ```ts
 * import { mobiusApply, mobiusFromPoints } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * const m = mobiusFromPoints([0, 1, 2], [1, 2, 3])
 * toArray(mobiusApply(m, [5]).re).map((v) => Math.round(v * 1e12) / 1e12) // => [6]
 * ```
 */
export function mobiusFromPoints(
  from: readonly [ComplexNumber | number, ComplexNumber | number, ComplexNumber | number],
  to: readonly [ComplexNumber | number, ComplexNumber | number, ComplexNumber | number],
): Mobius {
  // S(z) = ((z - z1)(z2 - z3)) / ((z - z3)(z2 - z1)) sends z1, z2, z3 to 0, 1, infinity.
  const toStandard = (p: readonly (ComplexNumber | number)[]): Mobius => {
    const [z1, z2, z3] = p.map(lit) as [ComplexNumber, ComplexNumber, ComplexNumber]
    const k = C.sub(z2, z3)
    const l = C.sub(z2, z1)
    return mobius(k, C.mul(C.mul(z1, k), -1), l, C.mul(C.mul(z3, l), -1))
  }
  return mobiusCompose(mobiusInverse(toStandard(to)), toStandard(from))
}

/**
 * Standard conformal maps of the plane, as complex functions: the square `z^2`, the exponential, the Joukowski map
 * `z + 1/z` (circles to aerofoils), the Cayley map `(z - i) / (z + i)` (upper half-plane to the unit disc) and the
 * inversion `1 / z`.
 *
 * @concept conformal-maps
 * @example
 * ```ts
 * import { conformalMaps } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(conformalMaps.joukowski({ re: [1], im: [0] }).re) // => [2]
 * ```
 */
export const conformalMaps: {
  readonly square: ComplexFunction
  readonly exp: ComplexFunction
  readonly joukowski: ComplexFunction
  readonly cayley: ComplexFunction
  readonly inversion: ComplexFunction
} = {
  square: (z) => cMul(z, z),
  exp: (z) => cExp(z),
  joukowski: (z) => add2(z, cDiv(complex(onesLike(z.re), zerosLike(z.im)), z)),
  cayley: (z) => cDiv({ re: z.re, im: sub(z.im, 1) }, { re: z.re, im: add(z.im, 1) }),
  inversion: (z) => cDiv(complex(onesLike(z.re), zerosLike(z.im)), z),
}

/** A grid and its image under a complex map, for drawing. */
export interface ConformalGrid {
  /** Grid lines (horizontal then vertical) as polylines of `[re, im]` points. */
  readonly before: number[][][]
  readonly after: number[][][]
}

/**
 * The image of a rectangular grid under a complex map, line by line: under a conformal map the image curves still meet
 * at right angles wherever the derivative is nonzero.
 *
 * @param f - The map.
 * @param box - The rectangle.
 * @param options - Lines per direction (default 9) and points per line (default 60).
 * @returns The grid lines and their images.
 * @concept conformal-maps
 * @example
 * ```ts
 * import { conformalGrid, conformalMaps } from 'mdml-engine/foundations'
 * conformalGrid(conformalMaps.square, [[0, 1], [0, 1]], { lines: 3, samples: 5 }).after.length // => 6
 * ```
 */
export function conformalGrid(
  f: ComplexFunction,
  box: ComplexBox,
  options: { readonly lines?: number; readonly samples?: number } = {},
): ConformalGrid {
  const k = options.lines ?? 9
  const s = options.samples ?? 60
  const xs = toFlat(linspace(box[0][0], box[0][1], k))
  const ys = toFlat(linspace(box[1][0], box[1][1], k))
  const tx = toFlat(linspace(box[0][0], box[0][1], s))
  const ty = toFlat(linspace(box[1][0], box[1][1], s))
  const before = [...ys.map((y) => tx.map((x) => [x, y])), ...xs.map((x) => ty.map((y) => [x, y]))]
  const flat = before.flat()
  const w = f(
    complex(
      flat.map((p) => p[0] ?? 0),
      flat.map((p) => p[1] ?? 0),
    ),
  )
  const re = toFlat(w.re)
  const im = toFlat(w.im)
  const after = before.map((line, i) => line.map((_, j) => [re[i * s + j] ?? Number.NaN, im[i * s + j] ?? Number.NaN]))
  return { before, after }
}

/**
 * The complex derivative of a function at a point and the Cauchy-Riemann check, by forward-mode differentiation of
 * its real and imaginary parts: f is complex-differentiable there exactly when `u_x = v_y` and `u_y = -v_x`, and
 * conformal (angle-preserving) when moreover `f'(z) != 0`.
 *
 * @param f - The function.
 * @param z - The point.
 * @returns `f'(z)`, the Cauchy-Riemann residual, and whether f is conformal at z.
 * @concept cauchy-riemann
 * @example
 * ```ts
 * import { cauchyRiemann, conformalMaps } from 'mdml-engine/foundations'
 * import { neg } from 'mdml-engine/tensor'
 * cauchyRiemann(conformalMaps.square, { re: 1, im: 2 }).derivative // => { re: 2, im: 4 }
 * cauchyRiemann((z) => ({ re: z.re, im: neg(z.im) }), { re: 1, im: 2 }).conformal // => false
 * ```
 */
export function cauchyRiemann(
  f: ComplexFunction,
  z: ComplexNumber | number,
): { readonly derivative: ComplexNumber; readonly residual: number; readonly conformal: boolean } {
  const p = pair(z)
  const zero = zerosLike(p.re)
  const one = onesLike(p.re)
  const alongX = jvp((q: ComplexValue) => f(q), p, { re: one, im: zero }).tangent
  const alongY = jvp((q: ComplexValue) => f(q), p, { re: zero, im: one }).tangent
  const [ux, vx, uy, vy] = [item(alongX.re), item(alongX.im), item(alongY.re), item(alongY.im)]
  const residual = Math.hypot(ux - vy, uy + vx)
  const scale = Math.max(1, Math.hypot(ux, vx, uy, vy))
  const derivative = { re: ux, im: vx }
  return { derivative, residual, conformal: residual <= 1e-10 * scale && Math.hypot(ux, vx) > 1e-12 * scale }
}

/** A contour in the complex plane: parameters `t` `[n]` to points. */
export type ComplexContour = (t: Value) => ComplexValue

/**
 * The circle `centre + radius e^(it)`, t in `[0, 2 pi]`, traversed counter-clockwise.
 *
 * @param centre - The centre.
 * @param radius - The radius.
 * @returns The contour.
 * @concept contour-integrals
 * @example
 * ```ts
 * import { circleContour } from 'mdml-engine/foundations'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(circleContour(0, 2)(tensor([0])).re) // => [2]
 * ```
 */
export function circleContour(centre: ComplexNumber | number, radius: number): ComplexContour {
  const c = lit(centre)
  return (t) => {
    const e = cis(t)
    return { re: add(mul(e.re, radius), c.re), im: add(mul(e.im, radius), c.im) }
  }
}

/**
 * A contour integral `int_gamma f(z) dz = int_a^b f(z(t)) z'(t) dt`, with `z'` by forward-mode differentiation. For a
 * closed smooth periodic contour the trapezoid rule on n points converges exponentially (the default 'trapezoid' for
 * closed contours); 'quad' integrates the real and imaginary parts adaptively.
 *
 * @param f - The integrand.
 * @param contour - The contour.
 * @param a - Start parameter (default 0).
 * @param b - End parameter (default 2 pi).
 * @param options - Method ('trapezoid' (default) for closed periodic contours, 'quad' for open ones) and points (256).
 * @returns The integral.
 * @concept contour-integrals
 * @example
 * ```ts
 * import { circleContour, contourIntegral } from 'mdml-engine/foundations'
 * import { cDiv, complex, onesLike } from 'mdml-engine/tensor'
 * // The integral of 1/z round the unit circle is 2 pi i.
 * Math.round(contourIntegral((z) => cDiv(complex(onesLike(z.re)), z), circleContour(0, 1)).im * 1e12) / 1e12 // => 6.283185307180
 * ```
 */
export function contourIntegral(
  f: ComplexFunction,
  contour: ComplexContour,
  a = 0,
  b: number = 2 * Math.PI,
  options: { readonly method?: 'trapezoid' | 'quad'; readonly n?: number } = {},
): ComplexNumber {
  const integrand = (t: Value): ComplexValue => {
    const z = contour(t)
    const dz = jvp((s: Value) => contour(s), t, onesLike(t)).tangent
    return cMul(f(z), dz)
  }
  if ((options.method ?? 'trapezoid') === 'trapezoid') {
    const n = options.n ?? 256
    const h = (b - a) / n
    const ts = Array.from({ length: n }, (_, k) => a + k * h)
    const w = integrand(promoteFloat(ts)[0] as Value)
    return { re: item(sum(w.re)) * h, im: item(sum(w.im)) * h }
  }
  const tol = { epsabs: 1e-12, epsrel: 1e-12, limit: 200 }
  return {
    re: item(quad((t) => integrand(t).re, a, b, tol).value),
    im: item(quad((t) => integrand(t).im, a, b, tol).value),
  }
}

/**
 * The residue of f at an isolated singularity: `(1 / 2 pi i) oint f(z) dz` round a small circle (no other singularity
 * inside), by the trapezoid rule.
 *
 * @param f - The function.
 * @param z0 - The singularity.
 * @param options - Circle radius (default 1e-2) and points (default 256).
 * @returns The residue.
 * @concept residues
 * @example
 * ```ts
 * import { residueAt } from 'mdml-engine/foundations'
 * import { cDiv, cMul, complex, onesLike, sub, type ComplexValue } from 'mdml-engine/tensor'
 * // 1 / (z (z - 1)) has residue -1 at 0.
 * const f = (z: ComplexValue) => cDiv(complex(onesLike(z.re)), cMul(z, { re: sub(z.re, 1), im: z.im }))
 * Math.round(residueAt(f, 0).re * 1e12) / 1e12 // => -1
 * ```
 */
export function residueAt(
  f: ComplexFunction,
  z0: ComplexNumber | number,
  options: { readonly radius?: number; readonly n?: number } = {},
): ComplexNumber {
  return laurentCoefficient(f, z0, -1, options)
}

/**
 * A coefficient of the Laurent series `f(z) = sum_k a_k (z - z0)^k` on an annulus round z0:
 * `a_k = (1 / 2 pi i) oint f(z) (z - z0)^(-k-1) dz` on a circle of the given radius, by the trapezoid rule (the
 * discrete Fourier transform of f on the circle). k = -1 is the residue.
 *
 * @param f - The function.
 * @param z0 - The centre.
 * @param k - The order.
 * @param options - Radius (default 1e-2) and points (default 256).
 * @returns `a_k`.
 * @concept laurent-series
 * @example
 * ```ts
 * import { laurentCoefficient } from 'mdml-engine/foundations'
 * import { cExp } from 'mdml-engine/tensor'
 * Math.round(laurentCoefficient(cExp, 0, 3, { radius: 1 }).re * 1e12) / 1e12 // => 0.166666666667
 * ```
 */
export function laurentCoefficient(
  f: ComplexFunction,
  z0: ComplexNumber | number,
  k: number,
  options: { readonly radius?: number; readonly n?: number } = {},
): ComplexNumber {
  const r = options.radius ?? 1e-2
  const n = options.n ?? 256
  const c = lit(z0)
  const ts = Array.from({ length: n }, (_, j) => (2 * Math.PI * j) / n)
  const e = cis(ts)
  const z = { re: add(mul(e.re, r), c.re), im: add(mul(e.im, r), c.im) }
  // a_k = (1/n) sum_j f(z_j) (r e^{i t_j})^{-k}.
  const w = cMul(f(z), cis(mul(promoteFloat(ts)[0] as Value, -k)))
  const scale = r ** -k / n
  return { re: item(sum(w.re)) * scale, im: item(sum(w.im)) * scale }
}

/**
 * The winding number of a closed contour about a point, `(1 / 2 pi i) oint dz / (z - z0)`, rounded to the nearest
 * integer (with the unrounded value).
 *
 * @param contour - A closed contour.
 * @param z0 - A point not on it.
 * @param options - Parameter range (default `[0, 2 pi]`) and points (default 512).
 * @returns The winding number and the raw integral.
 * @concept winding-number
 * @example
 * ```ts
 * import { circleContour, windingNumber } from 'mdml-engine/foundations'
 * windingNumber(circleContour(0, 1), 0.5).winding // => 1
 * windingNumber(circleContour(0, 1), 3).winding // => 0
 * ```
 */
export function windingNumber(
  contour: ComplexContour,
  z0: ComplexNumber | number,
  options: { readonly range?: readonly [number, number]; readonly n?: number } = {},
): { readonly winding: number; readonly value: number } {
  const c = lit(z0)
  const [a, b] = options.range ?? [0, 2 * Math.PI]
  const I = contourIntegral(
    (z) => cDiv(complex(onesLike(z.re), zerosLike(z.im)), { re: sub(z.re, c.re), im: sub(z.im, c.im) }),
    contour,
    a,
    b,
    { n: options.n ?? 512 },
  )
  const value = I.im / (2 * Math.PI)
  return { winding: Math.round(value) || 0, value }
}

/**
 * The argument principle: `(1 / 2 pi i) oint f'(z) / f(z) dz = Z - P`, the number of zeros minus poles of f inside a
 * closed contour (with multiplicity), f' by forward-mode differentiation.
 *
 * @param f - A meromorphic function with no zeros or poles on the contour.
 * @param contour - The closed contour.
 * @param options - Parameter range (default `[0, 2 pi]`) and points (default 1024).
 * @returns Zeros minus poles (rounded) and the raw value.
 * @concept argument-principle
 * @example
 * ```ts
 * import { argumentPrinciple, circleContour } from 'mdml-engine/foundations'
 * import { cMul } from 'mdml-engine/tensor'
 * argumentPrinciple((z) => cMul(z, cMul(z, z)), circleContour(0, 1)).count // => 3
 * ```
 */
export function argumentPrinciple(
  f: ComplexFunction,
  contour: ComplexContour,
  options: { readonly range?: readonly [number, number]; readonly n?: number } = {},
): { readonly count: number; readonly value: number } {
  const [a, b] = options.range ?? [0, 2 * Math.PI]
  const logDerivative: ComplexFunction = (z) => {
    const one = onesLike(z.re)
    const zero = zerosLike(z.re)
    const dz = jvp((q: ComplexValue) => f(q), z, { re: one, im: zero }).tangent
    return cDiv(dz, f(z))
  }
  const I = contourIntegral(logDerivative, contour, a, b, { n: options.n ?? 1024 })
  const value = I.im / (2 * Math.PI)
  return { count: Math.round(value) || 0, value }
}
