/**
 * Floating-point arithmetic made visible: neighbours and gaps of doubles (`nextAfter`, `ulp`, machine epsilon),
 * rounding to float32, float16 and bfloat16 (emulated in doubles, with their ranges and gaps), error-free transformations (TwoSum and TwoProduct), summation algorithms to compare their
 * rounding errors, condition numbers and first-order error propagation from autodiff, and counts of correct significant
 * digits.
 */

import { grad } from '../../autodiff/index.ts'
import { kahanSum, OmaiError, pairwiseSum, roundHalfEven } from '../../base/index.ts'
import { item, type TensorLike, tensor, toVector, type Value } from '../../tensor/index.ts'

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
 * import { nextAfter } from 'omai-engine/numerics/approximation'
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
 * import { ulp } from 'omai-engine/numerics/approximation'
 * ulp(1) // => 2.220446049250313e-16
 * ```
 */
export function ulp(x: number): number {
  if (!Number.isFinite(x)) return Number.NaN
  const a = Math.abs(x)
  const above = nextAfter(a, Number.POSITIVE_INFINITY)
  // The largest finite double has no double above it: its spacing is the gap below (numpy.spacing).
  return Number.isFinite(above) ? above - a : a - nextAfter(a, Number.NEGATIVE_INFINITY)
}

/** The binary formats the module can name: IEEE double and single, IEEE half (float16) and Google's bfloat16. */
export type FloatFormatName = 'float64' | 'float32' | 'float16' | 'bfloat16'

/** The shape of a binary floating-point format. */
export interface FloatFormat {
  /** Bits of the fraction (the stored mantissa): 52, 23, 10, 7. */
  readonly fractionBits: number
  /** The smallest normal exponent: 2^minExponent is the smallest normal number. */
  readonly minExponent: number
  /** The largest finite number. */
  readonly max: number
}

/**
 * The binary floating-point formats by name: fraction bits, smallest normal exponent and largest finite value.
 *
 * @example
 * ```ts
 * import { FLOAT_FORMATS } from 'omai-engine/numerics/approximation'
 * FLOAT_FORMATS.float16.max // => 65504
 * ```
 */
export const FLOAT_FORMATS: Readonly<Record<FloatFormatName, FloatFormat>> = {
  float64: { fractionBits: 52, minExponent: -1022, max: Number.MAX_VALUE },
  float32: { fractionBits: 23, minExponent: -126, max: (2 - 2 ** -23) * 2 ** 127 },
  float16: { fractionBits: 10, minExponent: -14, max: 65504 },
  bfloat16: { fractionBits: 7, minExponent: -126, max: (2 - 2 ** -7) * 2 ** 127 },
}

/**
 * Machine epsilon of a floating-point format: the gap between 1 and the next number (numpy.finfo(...).eps).
 *
 * @param dtype - float64 (default), float32, float16 or bfloat16.
 * @returns Epsilon.
 * @example
 * ```ts
 * import { machineEpsilon } from 'omai-engine/numerics/approximation'
 * machineEpsilon('float32') // => 1.1920928955078125e-7
 * ```
 */
export function machineEpsilon(dtype: FloatFormatName = 'float64'): number {
  return 2 ** -FLOAT_FORMATS[dtype].fractionBits
}

/**
 * Round to the nearest float32 (round-half-even), returned as a double.
 *
 * @param x - A double.
 * @returns The float32 value.
 * @example
 * ```ts
 * import { toFloat32 } from 'omai-engine/numerics/approximation'
 * toFloat32(0.1) // => 0.10000000149011612
 * ```
 */
export function toFloat32(x: number): number {
  return Math.fround(x)
}

/** The binary exponent floor(log2 |x|) of a finite nonzero double, read from its bits (subnormals report -1023). */
function exponentOf(x: number): number {
  buffer.setFloat64(0, x)
  return ((buffer.getUint32(0) >>> 20) & 0x7ff) - 1023
}

/**
 * Round a double to the nearest value of a narrower binary format (round half to even; gradual underflow into
 * subnormals; overflow to infinity), returned as a double. Infinities and NaN pass through.
 *
 * @param x - A double.
 * @param format - The target format (default float16).
 * @returns The value of `format` nearest to `x`.
 * @example
 * ```ts
 * import { roundToFormat } from 'omai-engine/numerics/approximation'
 * roundToFormat(1 + 2 ** -11, 'float16') // => 1
 * ```
 */
export function roundToFormat(x: number, format: FloatFormatName = 'float16'): number {
  if (!Number.isFinite(x) || x === 0) return x
  const f = FLOAT_FORMATS[format]
  if (format === 'float64') return x
  const e = Math.max(exponentOf(x), f.minExponent)
  // The quantum is the spacing of the format at this magnitude (the subnormal spacing below the smallest normal).
  const quantum = 2 ** (e - f.fractionBits)
  const r = roundHalfEven(Math.abs(x) / quantum) * quantum
  const out = r > f.max ? Number.POSITIVE_INFINITY : r
  return x < 0 ? -out : out
}

/**
 * Round to the nearest IEEE half-precision (float16) value: a 10-bit fraction, normals down to 2^-14, subnormals to
 * 2^-24, largest finite value 65504, overflow to infinity. The way mixed-precision training loses small gradients.
 *
 * @param x - A double.
 * @returns The float16 value, as a double.
 * @example
 * ```ts
 * import { toFloat16 } from 'omai-engine/numerics/approximation'
 * toFloat16(0.1) // => 0.0999755859375
 * ```
 */
export function toFloat16(x: number): number {
  return roundToFormat(x, 'float16')
}

/**
 * Round to the nearest bfloat16 value: float32's 8-bit exponent (so its range) with a 7-bit fraction.
 *
 * @param x - A double.
 * @returns The bfloat16 value, as a double.
 * @example
 * ```ts
 * import { toBfloat16 } from 'omai-engine/numerics/approximation'
 * toBfloat16(0.1) // => 0.10009765625
 * ```
 */
export function toBfloat16(x: number): number {
  return roundToFormat(x, 'bfloat16')
}

/**
 * The exact sum of two doubles as s + e, s = fl(a + b) (Knuth's TwoSum).
 *
 * @param a - First term.
 * @param b - Second term.
 * @returns [s, e] with a + b = s + e exactly.
 * @example
 * ```ts
 * import { twoSum } from 'omai-engine/numerics/approximation'
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
 * import { twoProduct } from 'omai-engine/numerics/approximation'
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

/** The summation algorithms of {@link compensatedSum}. */
export type SumMethod = 'naive' | 'kahan' | 'neumaier' | 'pairwise'

/**
 * Sum with a chosen algorithm, to compare their rounding errors: naive left-to-right; Kahan's compensated summation
 * (each addition's rounding error carried into the next); Neumaier's variant (also correct when a term exceeds the
 * running sum; base's kahanSum); pairwise (cascade) summation as numpy (base's pairwiseSum). A kernel on plain numbers
 * (migration, D-009: kept native, because reading a long list as a tensor-like made it five times slower).
 *
 * @param xs - Terms.
 * @param method - The algorithm (default neumaier).
 * @returns The sum.
 * @concept floating-point
 * @example
 * ```ts
 * import { compensatedSum } from 'omai-engine/numerics/approximation'
 * compensatedSum([1, 1e100, 1, -1e100], 'neumaier') // => 2
 * compensatedSum([1, 1e100, 1, -1e100], 'kahan') // => 0
 * ```
 */
export function compensatedSum(xs: ArrayLike<number>, method: SumMethod = 'neumaier'): number {
  switch (method) {
    case 'naive': {
      let s = 0
      for (let i = 0; i < xs.length; i++) s += xs[i] ?? 0
      return s
    }
    case 'kahan': {
      let sum = 0
      let c = 0
      for (let i = 0; i < xs.length; i++) {
        const y = (xs[i] ?? 0) - c
        const t = sum + y
        c = t - sum - y
        sum = t
      }
      return sum
    }
    case 'neumaier':
      return kahanSum(xs)
    case 'pairwise':
      return pairwiseSum(xs)
  }
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
 * import { conditionNumber } from 'omai-engine/numerics/approximation'
 * import { log } from 'omai-engine/tensor'
 * // log is ill-conditioned near 1: cond = 1 / |log x|.
 * Math.round(conditionNumber((x) => log(x), 1.001)) // => 1000
 * ```
 */
export function conditionNumber(f: (x: Value) => Value, x: number): number {
  const fx = item(f(tensor(x)))
  const d = item(grad(f)(tensor(x)) as Value)
  if (fx === 0) throw new OmaiError('value', 'conditionNumber: f(x) = 0')
  return Math.abs((x * d) / fx)
}

/**
 * First-order error propagation: the standard deviation of f(x) for independent input errors, sqrt(sum (df/dx_i
 * sigma_i)^2), with the gradient from autodiff.
 *
 * @param f - A differentiable scalar function of a vector.
 * @param x - The point `[n]`.
 * @param sigma - Input standard deviations `[n]`.
 * @returns The propagated standard deviation.
 * @throws OmaiError with code `shape` unless `x` and `sigma` are vectors, and `not-differentiable` for a traced
 *   value.
 * @example
 * ```ts
 * import { propagateError } from 'omai-engine/numerics/approximation'
 * import { prod } from 'omai-engine/tensor'
 * // The area of a 2 x 3 rectangle with sides known to 0.1: sqrt((3 * 0.1)^2 + (2 * 0.1)^2).
 * Math.abs(propagateError((v) => prod(v), [2, 3], [0.1, 0.1]) - Math.sqrt(0.13)) < 1e-12 // => true
 * ```
 */
export function propagateError(f: (x: Value) => Value, x: TensorLike, sigma: TensorLike): number {
  const g = Array.from((grad(f)(tensor([...toVector(x, 'propagateError')])) as { data: ArrayLike<number> }).data)
  const spread = toVector(sigma, 'propagateError')
  return Math.sqrt(g.reduce((s, gi, i) => s + (gi * (spread[i] ?? 0)) ** 2, 0))
}

/**
 * The number of correct significant decimal digits of an approximation: -log10(|approx - exact| / |exact|).
 *
 * @param approx - The approximation.
 * @param exact - The reference.
 * @returns Digits (Infinity when equal).
 * @example
 * ```ts
 * import { significantDigits } from 'omai-engine/numerics/approximation'
 * Math.round(significantDigits(3.14159, Math.PI)) // => 6
 * ```
 */
export function significantDigits(approx: number, exact: number): number {
  if (approx === exact) return Number.POSITIVE_INFINITY
  return -Math.log10(Math.abs(approx - exact) / Math.abs(exact === 0 ? 1 : exact))
}
