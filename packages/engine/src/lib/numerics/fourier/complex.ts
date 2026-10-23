import {
  add,
  arctan2,
  cos,
  exp,
  mul,
  neg,
  promoteFloat,
  sin,
  sqrt,
  sub,
  type TensorLike,
  type Value,
  zerosLike,
} from '../../tensor/index.ts'

/** A complex value as two real values of the same shape (the engine has no complex dtype). */
export interface ComplexValue {
  readonly re: Value
  readonly im: Value
}

/** A complex pair of arrays (not yet promoted). */
export interface ComplexPairLike {
  readonly re: TensorLike
  readonly im: TensorLike
}

/** A real array or a complex pair. */
export type ComplexLike = TensorLike | ComplexPairLike

/** Whether x is a complex pair. */
export function isComplex(x: unknown): x is ComplexPairLike {
  return typeof x === 'object' && x !== null && 're' in x && 'im' in x && !Array.isArray(x)
}

/**
 * A complex pair from a real array (zero imaginary part) or an existing pair.
 *
 * @param x - Real array or complex pair.
 * @returns The pair, floating point.
 * @example
 * ```ts
 * import { complex } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(complex([1, 2]).im) // => [0, 0]
 * ```
 */
export function complex(x: ComplexLike): ComplexValue {
  if (isComplex(x)) {
    const [re, im] = promoteFloat(x.re, x.im) as [Value, Value]
    return { re, im }
  }
  const [re] = promoteFloat(x) as [Value]
  return { re, im: zerosLike(re) }
}

/** (a + ib) + (c + id). */
export function cAdd(a: ComplexValue, b: ComplexValue): ComplexValue {
  return { re: add(a.re, b.re), im: add(a.im, b.im) }
}

/** (a + ib) - (c + id). */
export function cSub(a: ComplexValue, b: ComplexValue): ComplexValue {
  return { re: sub(a.re, b.re), im: sub(a.im, b.im) }
}

/**
 * The complex product.
 *
 * @param a - First factor.
 * @param b - Second factor.
 * @returns a b.
 * @example
 * ```ts
 * import { cMul, complex } from 'whitebox-engine/numerics/fourier'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(cMul({ re: [0], im: [1] }, { re: [0], im: [1] }).re) // => [-1]
 * ```
 */
export function cMul(a: ComplexValue, b: ComplexValue): ComplexValue {
  return { re: sub(mul(a.re, b.re), mul(a.im, b.im)), im: add(mul(a.re, b.im), mul(a.im, b.re)) }
}

/** The complex conjugate. */
export function cConj(a: ComplexValue): ComplexValue {
  return { re: a.re, im: neg(a.im) }
}

/** Multiply by a real value. */
export function cScale(a: ComplexValue, s: TensorLike): ComplexValue {
  return { re: mul(a.re, s), im: mul(a.im, s) }
}

/** The modulus |z| (differentiable away from zero). */
export function cAbs(a: ComplexValue): Value {
  return sqrt(add(mul(a.re, a.re), mul(a.im, a.im)))
}

/** The argument of z in (-pi, pi]. */
export function cAngle(a: ComplexValue): Value {
  return arctan2(a.im, a.re)
}

/** exp(z). */
export function cExp(a: ComplexValue): ComplexValue {
  const r = exp(a.re)
  return { re: mul(r, cos(a.im)), im: mul(r, sin(a.im)) }
}

/** exp(i theta) for real theta. */
export function cis(theta: TensorLike): ComplexValue {
  const [t] = promoteFloat(theta) as [Value]
  return { re: cos(t), im: sin(t) }
}
