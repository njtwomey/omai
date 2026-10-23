import { maxOf, WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  concreteOrThrow,
  cumsum,
  div,
  fromStorage,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  searchsorted,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  take,
  tensor,
  toFlat,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'
import { eigvals } from '../linalg/index.ts'

/**
 * A piecewise polynomial in the power basis (scipy.interpolate.PPoly): on [x_i, x_{i+1}] it is
 * sum_j c[j, i] (x - x_i)^(k - j), highest power first. Trailing coefficient axes are separate output components.
 * Evaluation gathers the interval's coefficients and runs Horner's rule with operations, so values are differentiable
 * in the coefficients (hence in data an interpolant was built from) and in the query points.
 */
export interface PPoly {
  /** Breakpoints `[m + 1]`, increasing. */
  readonly breakpoints: Tensor
  /** Coefficients `[k + 1, m, ...]`. */
  readonly coefficients: Value
  /** The polynomial degree k. */
  readonly degree: number
  /** Extend the end pieces outside the breakpoints (true), return NaN there (false), or wrap around ('periodic'). */
  readonly extrapolate: Extrapolate
  /** Values at the query points: `[q, ...]` for a 1-D query, `[...]` for a scalar. */
  readonly evaluate: (x: TensorLike, nu?: number) => Value
  readonly derivative: (nu?: number) => PPoly
  readonly antiderivative: (nu?: number) => PPoly
  readonly integrate: (a: number, b: number) => Value
  /** Real roots (for one output component), sorted, within the breakpoints. */
  readonly roots: () => number[]
}

/** How a piecewise function behaves outside its breakpoints. */
export type Extrapolate = boolean | 'periodic'

function power(t: Value, trailing: number): Value {
  return trailing === 0 ? t : reshape(t, [...t.shape, ...new Array<number>(trailing).fill(1)])
}

/**
 * A piecewise polynomial from coefficients and breakpoints.
 *
 * @param coefficients - `[k + 1, m, ...]`, highest power first.
 * @param breakpoints - `[m + 1]`, increasing.
 * @param extrapolate - Extend the end pieces (default true), NaN outside (false) or wrap around ('periodic').
 * @returns The piecewise polynomial.
 * @concept piecewise-polynomials
 * @example
 * ```ts
 * import { ppoly } from 'whitebox-engine/numerics/interpolate'
 * import { toArray } from 'whitebox-engine/tensor'
 * // x^2 on [0, 1] and 1 + 2(x - 1) on [1, 2].
 * toArray(ppoly([[1, 0], [0, 2], [0, 1]], [0, 1, 2]).evaluate([0.5, 1.5])) // => [0.25, 2]
 * ```
 */
export function ppoly(coefficients: TensorLike, breakpoints: TensorLike, extrapolate: Extrapolate = true): PPoly {
  const [c] = promoteFloat(coefficients) as [Value]
  const x = concreteOrThrow(promoteFloat(breakpoints)[0] as Tensor, 'ppoly breakpoints')
  const xs = toFlat(x)
  const m = xs.length - 1
  if (c.shape.length < 2 || c.shape[1] !== m)
    throw new WhiteboxError('shape', 'ppoly: coefficients must be [k + 1, m, ...]')
  for (let i = 0; i < m; i++) {
    if (!((xs[i + 1] ?? 0) > (xs[i] ?? 0))) throw new WhiteboxError('value', 'ppoly: breakpoints must increase')
  }
  const k = (c.shape[0] ?? 1) - 1
  const trailing = c.shape.length - 2

  const evaluate = (query: TensorLike, nu = 0): Value => {
    if (nu > 0) return derivative(nu).evaluate(query)
    if (nu < 0) return antiderivative(-nu).evaluate(query)
    const [q0] = promoteFloat(query) as [Value]
    const scalar = q0.shape.length === 0
    let q = q0
    if (extrapolate === 'periodic') {
      // Shift each point into [x_0, x_m) by whole periods (the shift is a constant, so derivatives pass through).
      const lo = xs[0] ?? 0
      const period = (xs[m] ?? 0) - lo
      const shifts = toFlat(primalOf(reshape(q0, [-1]))).map((v) => -Math.floor((v - lo) / period) * period)
      q = add(reshape(q0, [-1]), fromStorage(Float64Array.from(shifts), [shifts.length], 'float64'))
    }
    const qf = reshape(q, [-1])
    const qp = primalOf(qf)
    const idx = toFlat(searchsorted(x, qp, 'right')).map((i) => Math.min(Math.max(i - 1, 0), m - 1))
    const at = fromStorage(Int32Array.from(idx), [idx.length], 'int32')
    const cg = take(c, at, 1)
    const t = power(sub(qf, take(x, at)), trailing)
    let acc = reshape(sliceAxis(cg, 0, 0, 1), cg.shape.slice(1))
    for (let j = 1; j <= k; j++) acc = add(mul(acc, t), reshape(sliceAxis(cg, 0, j, j + 1), cg.shape.slice(1)))
    if (extrapolate === false) {
      const lo = xs[0] ?? 0
      const hi = xs[m] ?? 0
      const outside = toFlat(qp).map((v) => v < lo || v > hi)
      if (outside.some(Boolean)) {
        const mask = power(fromStorage(Uint8Array.from(outside, Number), [outside.length], 'bool'), trailing)
        acc = where(mask, Number.NaN, acc)
      }
    }
    return scalar ? reshape(acc, acc.shape.slice(1)) : acc
  }

  const derivative = (nu = 1): PPoly => {
    if (nu === 0) return self
    if (nu > k) return ppoly(zeros([1, ...c.shape.slice(1)]), x, extrapolate)
    // d/dx sum c_j t^(k - j): row j becomes c_j (k - j)(k - j - 1)...(k - j - nu + 1), dropping the last nu rows.
    const factors = Array.from({ length: k + 1 - nu }, (_, j) => {
      let f = 1
      for (let r = 0; r < nu; r++) f *= k - j - r
      return f
    })
    const scale = power(reshape(tensor(factors), [k + 1 - nu, 1]), trailing)
    return ppoly(mul(sliceAxis(c, 0, 0, k + 1 - nu), scale), x, extrapolate)
  }

  const antiderivative = (nu = 1): PPoly => {
    if (nu === 0) return self
    if (nu > 1) return antiderivative(1).antiderivative(nu - 1)
    // Integrate each piece: c_j t^(k - j) -> c_j t^(k - j + 1) / (k - j + 1), with a constant so the result is
    // continuous and zero at the first breakpoint.
    const divisors = Array.from({ length: k + 1 }, (_, j) => k + 1 - j)
    const lifted = div(c, power(reshape(tensor(divisors), [k + 1, 1]), trailing))
    const h = power(tensor(Array.from({ length: m }, (_, i) => (xs[i + 1] ?? 0) - (xs[i] ?? 0))), trailing)
    // The integral over each whole interval: sum_j lifted_j h^(k + 1 - j).
    let whole: Value = mul(reshape(sliceAxis(lifted, 0, 0, 1), lifted.shape.slice(1)), h)
    for (let j = 1; j <= k; j++)
      whole = mul(add(whole, reshape(sliceAxis(lifted, 0, j, j + 1), lifted.shape.slice(1))), h)
    const running = cumsum(whole, 0)
    const constants = concat([zeros([1, ...running.shape.slice(1)]), sliceAxis(running, 0, 0, m - 1)], 0)
    // A periodic function's antiderivative is not periodic (scipy then leaves it undefined outside).
    return ppoly(
      concat([lifted, reshape(constants, [1, ...constants.shape])], 0),
      x,
      extrapolate === 'periodic' ? false : extrapolate,
    )
  }

  const integrate = (a: number, b: number): Value => {
    const anti = antiderivative(1)
    const ends = anti.evaluate([a, b])
    return sub(
      reshape(sliceAxis(ends, 0, 1, 2), ends.shape.slice(1)),
      reshape(sliceAxis(ends, 0, 0, 1), ends.shape.slice(1)),
    )
  }

  const roots = (): number[] => {
    if (trailing !== 0) throw new WhiteboxError('shape', 'ppoly roots: only for a single output component')
    const cs = toFlat(primalOf(c))
    const found: number[] = []
    for (let i = 0; i < m; i++) {
      const lo = xs[i] ?? 0
      const hi = xs[i + 1] ?? 0
      const coeffs = Array.from({ length: k + 1 }, (_, j) => cs[j * m + i] ?? 0)
      for (const t of realRoots(coeffs)) {
        const r = lo + t
        if (r >= lo - 1e-12 * Math.max(1, Math.abs(lo)) && r <= hi + 1e-12 * Math.max(1, Math.abs(hi)))
          found.push(Math.min(Math.max(r, lo), hi))
      }
    }
    found.sort((p, q) => p - q)
    return found.filter((r, i) => i === 0 || Math.abs(r - (found[i - 1] ?? 0)) > 1e-10 * Math.max(1, Math.abs(r)))
  }

  const self: PPoly = {
    breakpoints: x,
    coefficients: c,
    degree: k,
    extrapolate,
    evaluate,
    derivative,
    antiderivative,
    integrate,
    roots,
  }
  return self
}

/** Real roots of sum_j a_j t^(n - j) (highest first), by companion-matrix eigenvalues polished with Newton steps. */
export function realRoots(coefficients: readonly number[]): number[] {
  let a = [...coefficients]
  const scale = maxOf(a.map(Math.abs))
  if (scale === 0) return []
  while (a.length > 0 && Math.abs(a[0] ?? 0) <= 1e-14 * scale) a = a.slice(1)
  const n = a.length - 1
  if (n < 1) return []
  const lead = a[0] ?? 1
  if (n === 1) return [-(a[1] ?? 0) / lead]
  const companion = new Float64Array(n * n)
  for (let j = 0; j < n; j++) companion[j] = -(a[j + 1] ?? 0) / lead
  for (let i = 1; i < n; i++) companion[i * n + i - 1] = 1
  const { re, im } = eigvals(fromStorage(companion, [n, n], 'float64'))
  const rs = toFlat(re)
  const is = toFlat(im)
  const p = (t: number): [number, number] => {
    let v = 0
    let d = 0
    for (const coeff of a) {
      d = d * t + v
      v = v * t + coeff
    }
    return [v, d]
  }
  return rs
    .filter((_, i) => Math.abs(is[i] ?? 0) <= 1e-8 * Math.max(1, Math.abs(rs[i] ?? 0)))
    .map((r0) => {
      let r = r0
      for (let it = 0; it < 3; it++) {
        const [v, d] = p(r)
        if (d === 0) break
        r -= v / d
      }
      return r
    })
}
