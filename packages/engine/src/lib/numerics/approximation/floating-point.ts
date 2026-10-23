import { grad } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import { item, tensor, type Value } from '../../tensor/index.ts'

const buffer = new DataView(new ArrayBuffer(8))

function bitsOf(x: number): bigint {
  buffer.setFloat64(0, x)
  return buffer.getBigInt64(0)
}

function fromBits(b: bigint): number {
  buffer.setBigInt64(0, b)
  return buffer.getFloat64(0)
}

/**
 * The next representable double after x in the direction of y (numpy.nextafter).
 *
 * @param x - Start.
 * @param y - Direction.
 * @returns The neighbour.
 * @concept floating-point
 * @example
 * ```ts
 * import { nextAfter } from 'whitebox-engine/numerics/approximation'
 * nextAfter(1, 2) - 1 // => 2.220446049250313e-16
 * ```
 */
export function nextAfter(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return Number.NaN
  if (x === y) return y
  if (x === 0) return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE
  const b = bitsOf(x)
  return fromBits(y > x === x > 0 ? b + 1n : b - 1n)
}

/**
 * The unit in the last place: the gap from |x| to the next larger double (numpy.spacing(abs(x))).
 *
 * @param x - A finite double.
 * @returns The spacing.
 * @example
 * ```ts
 * import { ulp } from 'whitebox-engine/numerics/approximation'
 * ulp(1) // => 2.220446049250313e-16
 * ```
 */
export function ulp(x: number): number {
  if (!Number.isFinite(x)) return Number.NaN
  const a = Math.abs(x)
  return nextAfter(a, Number.POSITIVE_INFINITY) - a
}

/**
 * Machine epsilon of a floating-point format: the gap between 1 and the next number (numpy.finfo(...).eps).
 *
 * @param dtype - float64 (default) or float32.
 * @returns Epsilon.
 * @example
 * ```ts
 * import { machineEpsilon } from 'whitebox-engine/numerics/approximation'
 * machineEpsilon('float32') // => 1.1920928955078125e-7
 * ```
 */
export function machineEpsilon(dtype: 'float64' | 'float32' = 'float64'): number {
  return dtype === 'float64' ? 2 ** -52 : 2 ** -23
}

/**
 * Round to the nearest float32 (round-half-even), returned as a double.
 *
 * @param x - A double.
 * @returns The float32 value.
 * @example
 * ```ts
 * import { toFloat32 } from 'whitebox-engine/numerics/approximation'
 * toFloat32(0.1) // => 0.10000000149011612
 * ```
 */
export function toFloat32(x: number): number {
  return Math.fround(x)
}

/**
 * The exact sum of two doubles as s + e, s = fl(a + b) (Knuth's TwoSum).
 *
 * @param a - First term.
 * @param b - Second term.
 * @returns [s, e] with a + b = s + e exactly.
 * @example
 * ```ts
 * import { twoSum } from 'whitebox-engine/numerics/approximation'
 * twoSum(1, 1e-16) // => [1, 1e-16]
 * ```
 */
export function twoSum(a: number, b: number): [number, number] {
  const s = a + b
  const bb = s - a
  return [s, a - (s - bb) + (b - bb)]
}

/** Veltkamp's split of a double into high and low halves of 26 bits. */
function split(a: number): [number, number] {
  const c = 134217729 * a
  const hi = c - (c - a)
  return [hi, a - hi]
}

/**
 * The exact product of two doubles as p + e, p = fl(a b) (Dekker's TwoProduct with Veltkamp splitting).
 *
 * @param a - First factor.
 * @param b - Second factor.
 * @returns [p, e] with a b = p + e exactly (barring overflow).
 * @example
 * ```ts
 * import { twoProduct } from 'whitebox-engine/numerics/approximation'
 * const [p, e] = twoProduct(0.1, 0.1)
 * e !== 0 // => true
 * ```
 */
export function twoProduct(a: number, b: number): [number, number] {
  const p = a * b
  const [ah, al] = split(a)
  const [bh, bl] = split(b)
  return [p, al * bl - (p - ah * bh - al * bh - ah * bl)]
}

/**
 * Kahan's compensated summation: the rounding error of each addition is carried into the next.
 *
 * @param xs - Terms.
 * @returns The sum, with error independent of n to first order.
 * @example
 * ```ts
 * import { kahanSum } from 'whitebox-engine/numerics/approximation'
 * kahanSum(new Array(10).fill(0.1)) // => 1
 * ```
 */
export function kahanSum(xs: readonly number[]): number {
  let sum = 0
  let c = 0
  for (const x of xs) {
    const y = x - c
    const t = sum + y
    c = t - sum - y
    sum = t
  }
  return sum
}

/**
 * Neumaier's improved compensated summation (also correct when a term exceeds the running sum).
 *
 * @param xs - Terms.
 * @returns The sum.
 * @example
 * ```ts
 * import { neumaierSum } from 'whitebox-engine/numerics/approximation'
 * neumaierSum([1, 1e100, 1, -1e100]) // => 2
 * ```
 */
export function neumaierSum(xs: readonly number[]): number {
  let sum = 0
  let c = 0
  for (const x of xs) {
    const t = sum + x
    c += Math.abs(sum) >= Math.abs(x) ? sum - t + x : x - t + sum
    sum = t
  }
  return sum + c
}

/**
 * Pairwise (cascade) summation, as numpy's sum: error O(log n) instead of O(n).
 *
 * @param xs - Terms.
 * @returns The sum.
 * @example
 * ```ts
 * import { pairwiseSum } from 'whitebox-engine/numerics/approximation'
 * pairwiseSum([1, 2, 3, 4]) // => 10
 * ```
 */
export function pairwiseSum(xs: readonly number[]): number {
  if (xs.length <= 8) return xs.reduce((s, x) => s + x, 0)
  const mid = Math.floor(xs.length / 2)
  return pairwiseSum(xs.slice(0, mid)) + pairwiseSum(xs.slice(mid))
}

/**
 * The relative condition number of a scalar function, |x f'(x) / f(x)|, with the derivative from autodiff: how much
 * relative input error is amplified.
 *
 * @param f - A differentiable function.
 * @param x - The point.
 * @returns The condition number.
 * @example
 * ```ts
 * import { conditionNumber } from 'whitebox-engine/numerics/approximation'
 * import { log } from 'whitebox-engine/tensor'
 * // log is ill-conditioned near 1: cond = 1 / |log x|.
 * Math.round(conditionNumber((x) => log(x), 1.001)) // => 1000
 * ```
 */
export function conditionNumber(f: (x: Value) => Value, x: number): number {
  const fx = item(f(tensor(x)))
  const d = item(grad(f)(tensor(x)) as Value)
  if (fx === 0) throw new WhiteboxError('value', 'conditionNumber: f(x) = 0')
  return Math.abs((x * d) / fx)
}

/**
 * First-order error propagation: the standard deviation of f(x) for independent input errors, sqrt(sum (df/dx_i
 * sigma_i)^2), with the gradient from autodiff.
 *
 * @param f - A differentiable scalar function of a vector.
 * @param x - The point.
 * @param sigma - Input standard deviations.
 * @returns The propagated standard deviation.
 * @example
 * ```ts
 * import { propagateError } from 'whitebox-engine/numerics/approximation'
 * import { prod } from 'whitebox-engine/tensor'
 * // The area of a 2 x 3 rectangle with sides known to 0.1: sqrt((3 * 0.1)^2 + (2 * 0.1)^2).
 * Math.abs(propagateError((v) => prod(v), [2, 3], [0.1, 0.1]) - Math.sqrt(0.13)) < 1e-12 // => true
 * ```
 */
export function propagateError(f: (x: Value) => Value, x: readonly number[], sigma: readonly number[]): number {
  const g = Array.from((grad(f)(tensor([...x])) as { data: ArrayLike<number> }).data)
  return Math.sqrt(g.reduce((s, gi, i) => s + (gi * (sigma[i] ?? 0)) ** 2, 0))
}

/**
 * The number of correct significant decimal digits of an approximation: -log10(|approx - exact| / |exact|).
 *
 * @param approx - The approximation.
 * @param exact - The reference.
 * @returns Digits (Infinity when equal).
 * @example
 * ```ts
 * import { significantDigits } from 'whitebox-engine/numerics/approximation'
 * Math.round(significantDigits(3.14159, Math.PI)) // => 6
 * ```
 */
export function significantDigits(approx: number, exact: number): number {
  if (approx === exact) return Number.POSITIVE_INFINITY
  return -Math.log10(Math.abs(approx - exact) / Math.abs(exact === 0 ? 1 : exact))
}
