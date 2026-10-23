import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  cumsum,
  div,
  mul,
  promoteFloat,
  sliceAxis,
  sub,
  sum,
  type TensorLike,
  type Value,
  zeros,
} from '../../tensor/index.ts'

/** Sample spacing: explicit sample points `x`, or a constant step `dx` (default 1). */
export interface SpacingOptions {
  readonly x?: TensorLike
  readonly dx?: number
  /** The axis of y to integrate along (default -1). */
  readonly axis?: number
}

function prepare(
  y: TensorLike,
  options: SpacingOptions,
): { y: Value; axis: number; n: number; dx: (start: number, stop: number) => Value | number } {
  const [yv] = promoteFloat(y) as [Value]
  const axis = (options.axis ?? -1) < 0 ? yv.shape.length + (options.axis ?? -1) : (options.axis ?? -1)
  const n = yv.shape[axis] ?? 0
  if (options.x === undefined) {
    const h = options.dx ?? 1
    return { y: yv, axis, n, dx: () => h }
  }
  const [xv] = promoteFloat(options.x) as [Value]
  if ((xv.shape[xv.shape.length - 1] ?? 0) !== n)
    throw new WhiteboxError('shape', 'integration: x must match y along the axis')
  const xs = (start: number, stop: number): Value => sliceAxis(xv, -1, start, stop)
  return { y: yv, axis, n, dx: (start, stop) => sub(xs(start + 1, stop + 1), xs(start, stop)) }
}

/**
 * The composite trapezoidal rule (numpy.trapezoid, scipy.integrate.trapezoid). Differentiable in y and x.
 *
 * @param y - Samples.
 * @param options - Sample points or spacing, and the axis.
 * @returns The integral estimate.
 * @concept quadrature
 * @example
 * ```ts
 * import { trapezoid } from 'whitebox-engine/numerics/quadrature'
 * import { item } from 'whitebox-engine/tensor'
 * item(trapezoid([0, 1, 4], { dx: 1 })) // => 3
 * ```
 */
export function trapezoid(y: TensorLike, options: SpacingOptions = {}): Value {
  const { y: yv, axis, n, dx } = prepare(y, options)
  const left = sliceAxis(yv, axis, 0, n - 1)
  const right = sliceAxis(yv, axis, 1, n)
  return sum(mul(mul(add(left, right), 0.5), dx(0, n - 1)), axis)
}

/**
 * The cumulative trapezoidal integral (scipy.integrate.cumulative_trapezoid): one fewer value than samples, or
 * `initial` prepended.
 *
 * @param y - Samples.
 * @param options - Sample points or spacing, the axis, and an initial value to prepend.
 * @returns The running integral.
 * @example
 * ```ts
 * import { cumulativeTrapezoid } from 'whitebox-engine/numerics/quadrature'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(cumulativeTrapezoid([0, 1, 4], { initial: 0 })) // => [0, 0.5, 3]
 * ```
 */
export function cumulativeTrapezoid(
  y: TensorLike,
  options: SpacingOptions & { readonly initial?: number } = {},
): Value {
  const { y: yv, axis, n, dx } = prepare(y, options)
  const pieces = mul(mul(add(sliceAxis(yv, axis, 0, n - 1), sliceAxis(yv, axis, 1, n)), 0.5), dx(0, n - 1))
  const running = cumsum(pieces, axis)
  if (options.initial === undefined) return running
  const shape = [...running.shape]
  shape[axis] = 1
  return concat([add(zeros(shape), options.initial), running], axis)
}

/**
 * Composite Simpson's rule (scipy.integrate.simpson): for an odd number of samples, Simpson on every pair of
 * intervals; for an even number, Simpson on all but the last interval, which gets Cartwright's correction (as scipy
 * since 1.11). Equal spacing or arbitrary sample points.
 *
 * @param y - Samples.
 * @param options - Sample points or spacing, and the axis.
 * @returns The integral estimate.
 * @concept quadrature
 * @example
 * ```ts
 * import { simpson } from 'whitebox-engine/numerics/quadrature'
 * import { item } from 'whitebox-engine/tensor'
 * item(simpson([0, 1, 4], { dx: 1 })) // => 2.6666666666666665
 * ```
 */
export function simpson(y: TensorLike, options: SpacingOptions = {}): Value {
  const { y: yv, axis, n } = prepare(y, options)
  if (n < 3) return trapezoid(y, options)
  const at = (i: number): Value => sliceAxis(yv, axis, i, i + 1)
  const xsOf = (): ((i: number) => number | Value) => {
    if (options.x === undefined) {
      const h = options.dx ?? 1
      return (i) => i * h
    }
    const [xv] = promoteFloat(options.x) as [Value]
    return (i) => sliceAxis(xv, -1, i, i + 1)
  }
  const xs = xsOf()
  const h = (i: number): Value => sub(promoteFloat(xs(i + 1))[0] as Value, promoteFloat(xs(i))[0] as Value)
  // Simpson's rule over [x_i, x_{i+2}] for (possibly) unequal steps h0, h1.
  const panel = (i: number): Value => {
    const h0 = h(i)
    const h1 = h(i + 1)
    const hsum = add(h0, h1)
    const hprod = mul(h0, h1)
    const h0divh1 = div(h0, h1)
    const t0 = mul(at(i), sub(2, div(1, h0divh1)))
    const t1 = mul(at(i + 1), div(mul(hsum, hsum), hprod))
    const t2 = mul(at(i + 2), sub(2, h0divh1))
    return mul(div(hsum, 6), add(add(t0, t1), t2))
  }
  const last = n % 2 === 0 ? n - 2 : n - 1
  let total: Value = mul(at(0), 0)
  for (let i = 0; i + 2 <= last; i += 2) total = add(total, panel(i))
  if (n % 2 === 0) {
    // Cartwright's correction for the final interval, from the last three points.
    const h0 = h(n - 3)
    const h1 = h(n - 2)
    const alpha = div(add(mul(mul(h1, h1), 2), mul(mul(h0, h1), 3)), mul(add(h0, h1), 6))
    const beta = div(add(mul(h1, h1), mul(mul(h0, h1), 3)), mul(h0, 6))
    const eta = div(mul(mul(h1, h1), h1), mul(mul(h0, add(h0, h1)), 6))
    total = add(total, sub(add(mul(alpha, at(n - 1)), mul(beta, at(n - 2))), mul(eta, at(n - 3))))
  }
  return sum(total, axis)
}

/** One row of a Romberg table: the trapezoid estimate at 2^k intervals and its Richardson extrapolations. */
export type RombergRow = readonly number[]

/**
 * Romberg integration: trapezoid estimates with 1, 2, 4, ... intervals, extrapolated by Richardson's method, until two
 * diagonal entries agree (the classical algorithm scipy.integrate.romberg implemented). The table is returned for
 * figures.
 *
 * @param f - The integrand (vectorised over a 1-D value of points).
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @param options - Absolute and relative tolerances (1.48e-8) and the maximum number of halvings (10).
 * @returns The estimate and the Romberg table.
 * @example
 * ```ts
 * import { romberg } from 'whitebox-engine/numerics/quadrature'
 * import { exp } from 'whitebox-engine/tensor'
 * Math.abs(romberg((x) => exp(x), 0, 1).value - (Math.E - 1)) < 1e-10 // => true
 * ```
 */
export function romberg(
  f: (x: Value) => Value,
  a: number,
  b: number,
  options: { readonly tol?: number; readonly rtol?: number; readonly divmax?: number } = {},
): { readonly value: number; readonly table: readonly RombergRow[]; readonly converged: boolean } {
  const tol = options.tol ?? 1.48e-8
  const rtol = options.rtol ?? 1.48e-8
  const evaluate = (xs: number[]): number[] =>
    Array.from((f(promoteFloat(xs)[0] as Value) as { data: ArrayLike<number> }).data)
  const [fa, fb] = evaluate([a, b]) as [number, number]
  let h = b - a
  let trap = (h * (fa + fb)) / 2
  const table: number[][] = [[trap]]
  for (let k = 1; k <= (options.divmax ?? 10); k++) {
    const m = 2 ** (k - 1)
    h /= 2
    const mids = evaluate(Array.from({ length: m }, (_, i) => a + (2 * i + 1) * h))
    trap = trap / 2 + h * mids.reduce((s, v) => s + v, 0)
    const row = [trap]
    for (let j = 1; j <= k; j++) {
      const prev = table[k - 1]?.[j - 1] ?? 0
      row.push((row[j - 1] ?? 0) + ((row[j - 1] ?? 0) - prev) / (4 ** j - 1))
    }
    table.push(row)
    const last = row[k] ?? 0
    const before = table[k - 1]?.[k - 1] ?? 0
    if (Math.abs(last - before) <= Math.max(tol, rtol * Math.abs(last))) return { value: last, table, converged: true }
  }
  return { value: table.at(-1)?.at(-1) ?? Number.NaN, table, converged: false }
}
