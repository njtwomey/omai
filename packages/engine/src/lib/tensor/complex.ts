/**
 * Complex arithmetic on tensors, with a complex value held as a pair of real arrays of the same shape so that
 * everything stays differentiable: construction and real and imaginary parts, sums, products, quotients and conjugates,
 * modulus and argument, polar form (`torch.polar`), the complex exponential and `cis`.
 *
 * @index structure Complex tensors held as real and imaginary pairs, still differentiable.
 */

import { zerosLike } from './create.ts'
import { add, div, mul, neg, promoteFloat, sub } from './ops/arithmetic.ts'
import { arctan2, cos, exp, sin, sqrt } from './ops/math.ts'
import type { TensorLike } from './tensor.ts'
import type { Value } from './traced.ts'

/**
 * A complex value as two real values of the same shape. The engine has no complex dtype: a pair keeps every real
 * kernel, backend and derivative rule usable, and the derivative of a real function of a pair is the pair of partials
 * (the conjugate Wirtinger convention of JAX and PyTorch, read as two real gradients).
 *
 * @concept complex-numbers
 */
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

/**
 * Whether a value is a complex pair rather than a real array.
 *
 * @param x - Any value.
 * @returns `true` for objects with `re` and `im` fields.
 * @example
 * ```ts
 * import { isComplex } from 'omai-engine/tensor'
 * isComplex({ re: [1], im: [0] }) // => true
 * isComplex([1, 2]) // => false
 * ```
 */
export function isComplex(x: unknown): x is ComplexPairLike {
  return typeof x === 'object' && x !== null && 're' in x && 'im' in x && !Array.isArray(x)
}

/**
 * A complex value from its real and imaginary parts, from a real array (zero imaginary part) or from an existing pair.
 *
 * @param re - Real part, or a complex pair.
 * @param im - Imaginary part (zero when omitted).
 * @returns The pair, floating point, the parts broadcast to a common dtype.
 * @example
 * ```ts
 * import { complex, toArray } from 'omai-engine/tensor'
 * toArray(complex([1, 2]).im) // => [0, 0]
 * toArray(complex([1, 2], [3, 4]).im) // => [3, 4]
 * ```
 */
export function complex(re: ComplexLike, im?: TensorLike): ComplexValue {
  if (isComplex(re)) {
    const [r, i] = promoteFloat(re.re, re.im) as [Value, Value]
    return { re: r, im: i }
  }
  if (im !== undefined) {
    const [r, i] = promoteFloat(re, im) as [Value, Value]
    return { re: r, im: i }
  }
  const [r] = promoteFloat(re) as [Value]
  return { re: r, im: zerosLike(r) }
}

/**
 * The real part.
 *
 * @param z - A complex pair or a real array.
 * @returns Re z.
 * @example
 * ```ts
 * import { real, toArray } from 'omai-engine/tensor'
 * toArray(real({ re: [1, 2], im: [3, 4] })) // => [1, 2]
 * ```
 */
export function real(z: ComplexLike): Value {
  return complex(z).re
}

/**
 * The imaginary part (zero for a real array).
 *
 * @param z - A complex pair or a real array.
 * @returns Im z.
 * @example
 * ```ts
 * import { imag, toArray } from 'omai-engine/tensor'
 * toArray(imag([5])) // => [0]
 * ```
 */
export function imag(z: ComplexLike): Value {
  return complex(z).im
}

/**
 * The sum (a + ib) + (c + id).
 *
 * @param a - First term.
 * @param b - Second term.
 * @returns a + b.
 * @example
 * ```ts
 * import { cAdd, toArray } from 'omai-engine/tensor'
 * toArray(cAdd({ re: [1], im: [2] }, { re: [3], im: [4] }).im) // => [6]
 * ```
 */
export function cAdd(a: ComplexValue, b: ComplexValue): ComplexValue {
  return { re: add(a.re, b.re), im: add(a.im, b.im) }
}

/**
 * The difference (a + ib) - (c + id).
 *
 * @param a - First term.
 * @param b - Second term.
 * @returns a - b.
 * @example
 * ```ts
 * import { cSub, toArray } from 'omai-engine/tensor'
 * toArray(cSub({ re: [1], im: [2] }, { re: [3], im: [4] }).re) // => [-2]
 * ```
 */
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
 * import { cMul, toArray } from 'omai-engine/tensor'
 * toArray(cMul({ re: [0], im: [1] }, { re: [0], im: [1] }).re) // => [-1]
 * ```
 */
export function cMul(a: ComplexValue, b: ComplexValue): ComplexValue {
  return { re: sub(mul(a.re, b.re), mul(a.im, b.im)), im: add(mul(a.re, b.im), mul(a.im, b.re)) }
}

/**
 * The complex quotient, a conj(b) / |b|^2.
 *
 * @param a - Numerator.
 * @param b - Denominator.
 * @returns a / b.
 * @example
 * ```ts
 * import { cDiv, toArray } from 'omai-engine/tensor'
 * // (1 + 2i) / (1 + i) = 1.5 + 0.5i.
 * toArray(cDiv({ re: [1], im: [2] }, { re: [1], im: [1] }).re) // => [1.5]
 * ```
 */
export function cDiv(a: ComplexValue, b: ComplexValue): ComplexValue {
  const d = add(mul(b.re, b.re), mul(b.im, b.im))
  return {
    re: div(add(mul(a.re, b.re), mul(a.im, b.im)), d),
    im: div(sub(mul(a.im, b.re), mul(a.re, b.im)), d),
  }
}

/**
 * The complex conjugate.
 *
 * @param z - A complex pair or a real array.
 * @returns conj z.
 * @example
 * ```ts
 * import { conj, toArray } from 'omai-engine/tensor'
 * toArray(conj({ re: [1], im: [2] }).im) // => [-2]
 * ```
 */
export function conj(z: ComplexLike): ComplexValue {
  const c = complex(z)
  return { re: c.re, im: neg(c.im) }
}

/**
 * Multiply by a real value.
 *
 * @param a - The complex value.
 * @param s - A real scale, broadcast against it.
 * @returns s a.
 * @example
 * ```ts
 * import { cScale, toArray } from 'omai-engine/tensor'
 * toArray(cScale({ re: [1], im: [2] }, 3).im) // => [6]
 * ```
 */
export function cScale(a: ComplexValue, s: TensorLike): ComplexValue {
  return { re: mul(a.re, s), im: mul(a.im, s) }
}

/**
 * The modulus |z| = sqrt(re^2 + im^2), differentiable away from zero.
 *
 * @param z - A complex pair or a real array.
 * @returns |z|.
 * @example
 * ```ts
 * import { absComplex, toArray } from 'omai-engine/tensor'
 * toArray(absComplex({ re: [3], im: [4] })) // => [5]
 * ```
 */
export function absComplex(z: ComplexLike): Value {
  const c = complex(z)
  return sqrt(add(mul(c.re, c.re), mul(c.im, c.im)))
}

/**
 * The argument of z in (-pi, pi], as numpy's `angle`.
 *
 * @param z - A complex pair or a real array.
 * @returns arg z.
 * @example
 * ```ts
 * import { angle, toArray } from 'omai-engine/tensor'
 * toArray(angle({ re: [0, -1], im: [1, 0] })) // => [1.570796, 3.141593]
 * ```
 */
export function angle(z: ComplexLike): Value {
  const c = complex(z)
  return arctan2(c.im, c.re)
}

/**
 * The complex value with modulus r and argument theta, r (cos theta + i sin theta), as `torch.polar`.
 *
 * @param r - Modulus.
 * @param theta - Argument.
 * @returns The pair.
 * @example
 * ```ts
 * import { polar, toArray } from 'omai-engine/tensor'
 * toArray(polar([2], [0]).re) // => [2]
 * ```
 */
export function polar(r: TensorLike, theta: TensorLike): ComplexValue {
  const [m, t] = promoteFloat(r, theta) as [Value, Value]
  return { re: mul(m, cos(t)), im: mul(m, sin(t)) }
}

/**
 * The complex exponential exp(z).
 *
 * @param a - The exponent.
 * @returns exp(a).
 * @example
 * ```ts
 * import { cExp, toArray } from 'omai-engine/tensor'
 * toArray(cExp({ re: [0], im: [0] }).re) // => [1]
 * ```
 */
export function cExp(a: ComplexValue): ComplexValue {
  const r = exp(a.re)
  return { re: mul(r, cos(a.im)), im: mul(r, sin(a.im)) }
}

/**
 * exp(i theta) for real theta.
 *
 * @param theta - The angle.
 * @returns cos theta + i sin theta.
 * @example
 * ```ts
 * import { cis, toArray } from 'omai-engine/tensor'
 * toArray(cis([0]).re) // => [1]
 * ```
 */
export function cis(theta: TensorLike): ComplexValue {
  const [t] = promoteFloat(theta) as [Value]
  return { re: cos(t), im: sin(t) }
}
