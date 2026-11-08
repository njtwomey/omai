/**
 * Conversions between the representations of a linear system as plain coefficient data, scipy.signal style:
 * transfer-function normalisation, transfer function to state space (the controller canonical form) and back, transfer
 * function to zeros, poles and gain and back, and the routes between zeros-poles-gain and state space.
 */

import { OmaiError } from '../base/index.ts'
import { type ComplexArray, charpoly } from '../numerics/polynomial/index.ts'
import {
  add,
  concat,
  div,
  eye,
  flip,
  mul,
  neg,
  outer,
  reshape,
  sliceAxis,
  stack,
  sub,
  type TensorLike,
  tensor,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { asComplexArray, float, numbers, polyOf, rootsOf } from './_shared.ts'

// The conversions of scipy.signal (tf2ss, ss2tf, tf2zpk, zpk2tf, zpk2ss, ss2zpk), polynomials highest degree first.
// Transfer-function and state-space results are values, differentiable in the coefficients and matrices.

/** The four matrices of a state-space realisation x' = A x + B u, y = C x + D u. */
export interface StateSpaceMatrices {
  readonly a: Value
  readonly b: Value
  readonly c: Value
  readonly d: Value
}

/** A numerator and a denominator, highest degree first; the numerator is `[k]`, or `[outputs, k]` for one input. */
export interface Polynomials {
  readonly num: Value
  readonly den: Value
}

/** Zeros, poles (complex, as parallel parts) and a real gain. */
export interface ZerosPolesGainData {
  readonly zeros: ComplexArray
  readonly poles: ComplexArray
  readonly gain: number
}

/**
 * Normalise a transfer function (scipy.signal.normalize): strip the denominator's leading zeros, divide both by its
 * leading coefficient, and strip numerator columns that are zero (to 1e-14) in front, keeping at least one.
 *
 * @param num - Numerator `[k]` or `[outputs, k]`.
 * @param den - Denominator `[l]`.
 * @returns The normalised polynomials (a single-row numerator becomes 1-D).
 * @throws OmaiError with code `value` when the denominator is zero.
 * @example
 * ```ts
 * import { normalize } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(normalize([0, 2, 4], [0, 2, 6]).num) // => [1, 2]
 * ```
 */
export function normalize(num: TensorLike, den: TensorLike): Polynomials {
  let d = float(den)
  if (d.shape.length === 0) d = reshape(d, [1])
  if (d.shape.length !== 1) throw new OmaiError('shape', 'normalize: the denominator must be 1-D')
  const dn = numbers(d)
  const first = dn.findIndex((v) => v !== 0)
  if (first < 0) throw new OmaiError('value', 'normalize: the denominator must have a nonzero coefficient')
  d = sliceAxis(d, 0, first, dn.length)
  const lead = sliceAxis(d, 0, 0, 1)
  let n = float(num)
  if (n.shape.length === 0) n = reshape(n, [1, 1])
  else if (n.shape.length === 1) n = reshape(n, [1, n.shape[0] ?? 0])
  if (n.shape.length !== 2) throw new OmaiError('shape', 'normalize: the numerator must be 1-D or 2-D')
  n = div(n, lead)
  d = div(d, lead)
  const [p = 1, k = 0] = n.shape
  const nn = numbers(n)
  let zerosInFront = 0
  while (zerosInFront < k - 1) {
    let column = true
    for (let i = 0; i < p; i++) if (Math.abs(nn[i * k + zerosInFront] ?? 0) > 1e-14) column = false
    if (!column) break
    zerosInFront++
  }
  n = sliceAxis(n, 1, zerosInFront, k)
  return { num: p === 1 ? reshape(n, [k - zerosInFront]) : n, den: d }
}

/**
 * The controller canonical realisation of a transfer function (scipy.signal.tf2ss). A constant transfer function is
 * realised, as scipy, with one zero state.
 *
 * @param num - Numerator `[k]` or `[outputs, k]`.
 * @param den - Denominator `[l]` with l >= k.
 * @returns A, B, C and D.
 * @throws OmaiError with code `value` for an improper transfer function.
 * @concept state-space
 * @example
 * ```ts
 * import { tf2ss } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(tf2ss([1], [1, 3, 2]).a) // => [[-3, -2], [1, 0]]
 * ```
 */
export function tf2ss(num: TensorLike, den: TensorLike): StateSpaceMatrices {
  const normal = normalize(num, den)
  const n2 = normal.num.shape.length === 1 ? reshape(normal.num, [1, normal.num.shape[0] ?? 0]) : normal.num
  const [p = 1, m = 0] = n2.shape
  const k = normal.den.shape[0] ?? 0
  if (m > k) throw new OmaiError('value', 'tf2ss: improper transfer function (numerator longer than denominator)')
  const padded = m < k ? concat([zeros([p, k - m]), n2], 1) : n2
  const d = sliceAxis(padded, 1, 0, 1)
  if (k === 1) return { a: zeros([1, 1]), b: zeros([1, 1]), c: zeros([p, 1]), d }
  const tail = sliceAxis(normal.den, 0, 1, k)
  const frow = reshape(neg(tail), [1, k - 1])
  const a = k === 2 ? frow : concat([frow, eye(k - 2, k - 1)], 0)
  const c = sub(sliceAxis(padded, 1, 1, k), outer(reshape(d, [p]), tail))
  return { a, b: eye(k - 1, 1), c, d }
}

function checkMatrices(a: Value, b: Value, c: Value, d: Value, name: string): void {
  const [n = 0, n2 = 0] = a.shape
  const ok =
    a.shape.length === 2 &&
    n === n2 &&
    b.shape.length === 2 &&
    b.shape[0] === n &&
    c.shape.length === 2 &&
    c.shape[1] === n &&
    d.shape.length === 2 &&
    d.shape[0] === c.shape[0] &&
    d.shape[1] === b.shape[1]
  if (!ok) {
    throw new OmaiError('shape', `${name}: A [n, n], B [n, m], C [p, n] and D [p, m] do not fit`, {
      a: a.shape,
      b: b.shape,
      c: c.shape,
      d: d.shape,
    })
  }
}

/**
 * Check and complete state-space matrices (scipy.signal.abcd_normalize): float values, D zeros when omitted or
 * reshaped to `[p, m]` when given flat.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param c - C `[p, n]`.
 * @param d - D `[p, m]`, flat, or omitted.
 * @returns The four matrices.
 * @throws OmaiError with code `shape` when the shapes do not fit.
 * @example
 * ```ts
 * import { abcdNormalize } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(abcdNormalize([[1]], [[1, 2]], [[1]]).d) // => [[0, 0]]
 * ```
 */
export function abcdNormalize(a: TensorLike, b: TensorLike, c: TensorLike, d?: TensorLike): StateSpaceMatrices {
  const av = float(a)
  const bv = float(b)
  const cv = float(c)
  const p = cv.shape[0] ?? 0
  const m = bv.shape[1] ?? 0
  let dv = d === undefined ? zeros([p, m]) : float(d)
  if (dv.shape.length < 2 && numbers(dv).length === p * m) dv = reshape(dv, [p, m])
  checkMatrices(av, bv, cv, dv, 'state space')
  return { a: av, b: bv, c: cv, d: dv }
}

/**
 * The transfer function from one input of a state-space system (scipy.signal.ss2tf): den = det(sI - A) and, per
 * output k, num_k = det(sI - A + B C_k) + (D_k - 1) den. Differentiable in the matrices (through `charpoly`).
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param c - C `[p, n]`.
 * @param d - D `[p, m]` (zeros when omitted).
 * @param input - The input index.
 * @returns The numerator `[p, n + 1]` and denominator `[n + 1]`.
 * @throws OmaiError with code `value` when the input does not exist.
 * @concept transfer-functions
 * @example
 * ```ts
 * import { ss2tf } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(ss2tf([[-3, -2], [1, 0]], [[1], [0]], [[0, 1]], [[0]]).den) // => [1, 3, 2]
 * ```
 */
export function ss2tf(a: TensorLike, b: TensorLike, c: TensorLike, d?: TensorLike, input = 0): Polynomials {
  const s = abcdNormalize(a, b, c, d)
  const m = s.b.shape[1] ?? 0
  const p = s.c.shape[0] ?? 0
  if (input < 0 || input >= m) throw new OmaiError('value', `ss2tf: the system has no input ${input}`)
  const bi = sliceAxis(s.b, 1, input, input + 1)
  const den = flip(charpoly(s.a))
  const nums: Value[] = []
  for (let k = 0; k < p; k++) {
    const ck = sliceAxis(s.c, 0, k, k + 1)
    const dk = reshape(sliceAxis(sliceAxis(s.d, 0, k, k + 1), 1, input, input + 1), [])
    nums.push(add(flip(charpoly(sub(s.a, mul(bi, ck)))), mul(sub(dk, 1), den)))
  }
  return { num: stack(nums, 0), den }
}

/**
 * Zeros, poles and gain of a single-output transfer function (scipy.signal.tf2zpk).
 *
 * @param num - Numerator `[k]`.
 * @param den - Denominator `[l]`.
 * @returns Zeros and poles (sorted by real then imaginary part) and the gain.
 * @throws OmaiError with code `shape` for a numerator with several rows.
 * @concept transfer-functions
 * @example
 * ```ts
 * import { tf2zpk } from 'omai-engine/systems'
 * tf2zpk([2, 2], [1, 3, 2]) // => { zeros: { re: [-1], im: [0] }, poles: { re: [-2, -1], im: [0, 0] }, gain: 2 }
 * ```
 */
export function tf2zpk(num: TensorLike, den: TensorLike): ZerosPolesGainData {
  const normal = normalize(num, den)
  if (normal.num.shape.length !== 1) throw new OmaiError('shape', 'tf2zpk: one numerator row is supported')
  const b = numbers(normal.num)
  const gain = b[0] ?? 0
  const zeros = gain === 0 ? { re: [], im: [] } : rootsOf(b.map((v) => v / gain))
  return { zeros, poles: rootsOf(numbers(normal.den)), gain }
}

/**
 * The transfer function with given zeros, poles and gain (scipy.signal.zpk2tf); complex zeros and poles must come in
 * conjugate pairs.
 *
 * @param zeros - Zeros, real or as parallel parts.
 * @param poles - Poles, real or as parallel parts.
 * @param gain - The gain.
 * @returns The numerator and denominator.
 * @concept transfer-functions
 * @example
 * ```ts
 * import { zpk2tf } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(zpk2tf([-1], [-2, -3], 2).den) // => [1, 5, 6]
 * ```
 */
export function zpk2tf(
  zeros: readonly number[] | ComplexArray,
  poles: readonly number[] | ComplexArray,
  gain: number,
): Polynomials {
  return {
    num: tensor(polyOf(asComplexArray(zeros)).map((v) => v * gain)),
    den: tensor(polyOf(asComplexArray(poles))),
  }
}

/**
 * A state-space realisation of zeros, poles and gain (scipy.signal.zpk2ss), through the transfer function.
 *
 * @param zeros - Zeros.
 * @param poles - Poles.
 * @param gain - The gain.
 * @returns A, B, C and D (controller canonical form).
 * @example
 * ```ts
 * import { zpk2ss } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(zpk2ss([], [-1], 1).a) // => [[-1]]
 * ```
 */
export function zpk2ss(
  zeros: readonly number[] | ComplexArray,
  poles: readonly number[] | ComplexArray,
  gain: number,
): StateSpaceMatrices {
  const { num, den } = zpk2tf(zeros, poles, gain)
  return tf2ss(num, den)
}

/**
 * Zeros, poles and gain from one input to a single output of a state-space system (scipy.signal.ss2zpk).
 *
 * @param a - A.
 * @param b - B.
 * @param c - C (one row).
 * @param d - D.
 * @param input - The input index.
 * @returns Zeros, poles and gain.
 * @example
 * ```ts
 * import { ss2zpk } from 'omai-engine/systems'
 * ss2zpk([[-1]], [[1]], [[2]], [[0]]).gain // => 2
 * ```
 */
export function ss2zpk(a: TensorLike, b: TensorLike, c: TensorLike, d?: TensorLike, input = 0): ZerosPolesGainData {
  const { num, den } = ss2tf(a, b, c, d, input)
  if ((num.shape[0] ?? 0) !== 1) throw new OmaiError('shape', 'ss2zpk: one output is supported')
  return tf2zpk(reshape(num, [num.shape[1] ?? 0]), den)
}
