/**
 * Chebyshev expansions, sums of Chebyshev polynomials on an interval and the standard near-best way to approximate a
 * smooth function: coefficients from values at Chebyshev points by a DCT, and `chebfun`, which samples on finer and
 * finer grids until the coefficients fall below machine precision (as the chebfun constructor).
 */

import { type Json, MdmlError, maxOf } from '../../base/index.ts'
import {
  add,
  fromStorage,
  mul,
  primalOf,
  promoteFloat,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  type Value,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  numberFromJson,
  numbersFromJson,
  numbersToJson,
  numberToJson,
  runAlgorithm,
} from '../../trace/index.ts'
import { vectorOf } from '../_vector.ts'
import { dct } from '../fourier/index.ts'
import { eigvals } from '../linalg/index.ts'
import { orthoval } from '../polynomial/index.ts'
import type { ApproximationPath } from './approximation-contract.ts'

/**
 * A Chebyshev expansion sum_k c_k T_k(t) of a function on [a, b], with t = (2x - a - b) / (b - a). Holds tensors
 * (migration, D-009: `coefficients`, the roots and the extrema points were plain arrays).
 */
export interface Chebyshev {
  /** The coefficients c_0, c_1, ..., `[k]`. */
  readonly coefficients: Tensor<'float64'>
  readonly domain: readonly [number, number]
  /** Values at points (Clenshaw by operations: differentiable in the points). */
  readonly evaluate: (x: TensorLike) => Value
  readonly derivative: (order?: number) => Chebyshev
  /** The indefinite integral, zero at a. */
  readonly cumsum: () => Chebyshev
  /** The definite integral over the domain (Clenshaw-Curtis). */
  readonly sum: () => number
  /** Real roots in the domain, sorted, as a `[r]` tensor. */
  readonly roots: () => Tensor<'float64'>
  /** Local extrema and the endpoints (`points`, a `[p]` tensor), with the global maximum and minimum. */
  readonly extrema: () => {
    readonly points: Tensor<'float64'>
    readonly max: readonly [number, number]
    readonly min: readonly [number, number]
  }
}

/**
 * A Chebyshev expansion from coefficients on a domain. Takes the coefficients as a tensor-like (migration, D-009: was
 * `readonly number[]`); the record holds tensors (it kept plain arrays).
 *
 * @param coefficients - c_0, c_1, ... in the Chebyshev basis `[k]`.
 * @param domain - The interval (default [-1, 1]).
 * @returns The expansion.
 * @throws MdmlError with code `value` for an empty domain, `shape` unless the coefficients are a vector, and
 *   `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { chebyshev } from 'mdml-engine/numerics/approximation'
 * import { item } from 'mdml-engine/tensor'
 * item(chebyshev([0, 0, 1]).evaluate(0.5)) // => -0.5
 * ```
 */
export function chebyshev(coefficients: TensorLike, domain: readonly [number, number] = [-1, 1]): Chebyshev {
  const [a, b] = domain
  if (!(b > a)) throw new MdmlError('value', 'chebyshev: the domain must be an increasing interval')
  const entries = toVector(coefficients, 'chebyshev')
  const c = entries.length === 0 ? [0] : [...entries]
  const toT = (x: Value): Value => mul(add(mul(x, 2), -(a + b)), 1 / (b - a))
  const evaluate = (x: TensorLike): Value => {
    const [xv] = promoteFloat(x) as [Value]
    return orthoval('chebyshev', toT(xv), c)
  }
  const derivative = (order = 1): Chebyshev => {
    let d = [...c]
    for (let r = 0; r < order; r++) {
      const n = d.length - 1
      if (n === 0) {
        d = [0]
        break
      }
      // c'_{k-1} = c'_{k+1} + 2 k c_k (numpy chebder), then the chain rule factor 2 / (b - a).
      const out = new Array<number>(n).fill(0)
      for (let k = n; k >= 1; k--) {
        out[k - 1] = 2 * k * (d[k] ?? 0) + (out[k + 1] ?? 0)
      }
      out[0] = (out[0] ?? 0) / 2
      d = out.map((v) => (v * 2) / (b - a))
    }
    return chebyshev(d, domain)
  }
  const cumsum = (): Chebyshev => {
    // integral T_k = T_{k+1} / (2(k+1)) - T_{k-1} / (2(k-1)) (numpy chebint), scaled by (b - a) / 2.
    const n = c.length
    const out = new Array<number>(n + 1).fill(0)
    out[1] = (c[0] ?? 0) - (n > 2 ? (c[2] ?? 0) / 2 : 0)
    for (let k = 2; k <= n; k++) out[k] = ((c[k - 1] ?? 0) - (c[k + 1] ?? 0)) / (2 * k)
    const scaled = out.map((v) => (v * (b - a)) / 2)
    // Make it zero at a (t = -1): T_k(-1) = (-1)^k.
    const at = scaled.reduce((s, v, k) => s + v * (k % 2 === 0 ? 1 : -1), 0)
    scaled[0] = (scaled[0] ?? 0) - at
    return chebyshev(scaled, domain)
  }
  const sum = (): number => {
    // integral of T_k over [-1, 1] is 2 / (1 - k^2) for even k, 0 for odd.
    let s = 0
    c.forEach((ck, k) => {
      if (k % 2 === 0) s += (ck * 2) / (1 - k * k)
    })
    return (s * (b - a)) / 2
  }
  const rootList = (): number[] => {
    const trimmed = trimCoefficients(c)
    const n = trimmed.length - 1
    if (n < 1) return []
    let ts: number[]
    if (n === 1) ts = [-(trimmed[0] ?? 0) / (trimmed[1] ?? 1)]
    else if (n > 50) {
      // Long expansions: split the interval and recurse (the colleague matrix would be large).
      const mid = (a + b) / 2
      const left = chebfun((x) => evaluate(x), [a, mid])
      const right = chebfun((x) => evaluate(x), [mid, b])
      const all = [...toFlat(left.roots()), ...toFlat(right.roots())].sort((p, q) => p - q)
      return all.filter((r, i) => i === 0 || Math.abs(r - (all[i - 1] ?? 0)) > 1e-10 * (b - a))
    } else {
      // The colleague matrix: its eigenvalues are the roots of sum c_k T_k.
      const m = new Float64Array(n * n)
      m[1] = 1
      for (let i = 1; i < n; i++) {
        m[i * n + i - 1] = 0.5
        if (i + 1 < n) m[i * n + i + 1] = 0.5
      }
      const lead = trimmed[n] ?? 1
      for (let j = 0; j < n; j++) m[(n - 1) * n + j] = (m[(n - 1) * n + j] ?? 0) - (trimmed[j] ?? 0) / (2 * lead)
      const { re, im } = eigvals(fromStorage(m, [n, n], 'float64'))
      const rs = toFlat(re)
      const is = toFlat(im)
      ts = rs.filter((r, i) => Math.abs(is[i] ?? 0) < 1e-8 && r >= -1 - 1e-8 && r <= 1 + 1e-8)
    }
    const xs = ts.map((t) => Math.min(b, Math.max(a, (a + b) / 2 + ((b - a) / 2) * t)))
    // Newton polish on the expansion (eigenvalues of the colleague matrix lose accuracy near the ends).
    const d = derivative()
    const polished = xs.map((x0) => {
      let x = x0
      for (let it = 0; it < 4; it++) {
        const fx = toFlat(primalOf(evaluate([x])))[0] ?? 0
        const dx = toFlat(primalOf(d.evaluate([x])))[0] ?? 0
        if (dx === 0 || fx === 0) break
        const next = Math.min(b, Math.max(a, x - fx / dx))
        if (next === x) break
        x = next
      }
      return x
    })
    polished.sort((p, q) => p - q)
    return polished.filter((r, i) => i === 0 || Math.abs(r - (polished[i - 1] ?? 0)) > 1e-10 * (b - a))
  }
  const roots = (): Tensor<'float64'> => vectorOf(rootList())
  const extrema = () => {
    const points = [
      a,
      ...toFlat(derivative().roots()).filter((x) => x > a && x < b),
      b,
    ]
    const values = toFlat(primalOf(evaluate(points)))
    let iMax = 0
    let iMin = 0
    values.forEach((v, i) => {
      if (v > (values[iMax] ?? 0)) iMax = i
      if (v < (values[iMin] ?? 0)) iMin = i
    })
    return {
      points: vectorOf(points),
      max: [points[iMax] ?? a, values[iMax] ?? 0] as const,
      min: [points[iMin] ?? a, values[iMin] ?? 0] as const,
    }
  }
  return { coefficients: vectorOf(c), domain, evaluate, derivative, cumsum, sum, roots, extrema }
}

function trimCoefficients(c: readonly number[]): number[] {
  const scale = Math.max(maxOf(c.map(Math.abs)), 0)
  let n = c.length
  while (n > 1 && Math.abs(c[n - 1] ?? 0) <= 1e-14 * scale) n--
  return c.slice(0, n)
}

/**
 * Chebyshev coefficients of the interpolant at n Chebyshev points of the second kind (numpy's chebinterpolate uses the
 * first kind): c_k = (2 / (n - 1)) sum'' f_j T_k(x_j), by a type-1 DCT.
 *
 * @param values - f at x_j = cos(j pi / (n - 1)), j = 0..n - 1 (from 1 down to -1), a vector `[n]`.
 * @returns The coefficients `[n]`.
 * @throws MdmlError with code `shape` unless `values` is a vector, and `not-differentiable` for a traced value.
 * @example
 * ```ts
 * import { chebyshevCoefficients } from 'mdml-engine/numerics/approximation'
 * import { toFlat } from 'mdml-engine/tensor'
 * // x^2 = (T_0 + T_2) / 2.
 * toFlat(chebyshevCoefficients([1, 0, 1])).map((v) => Math.round(v * 1e12) / 1e12) // => [0.5, 0, 0.5]
 * ```
 */
export function chebyshevCoefficients(values: TensorLike): Tensor<'float64', [number]> {
  return vectorOf(coefficientsOf(toVector(values, 'chebyshevCoefficients')))
}

function coefficientsOf(values: readonly number[]): number[] {
  const n = values.length
  if (n === 1) return [values[0] ?? 0]
  const d = toFlat(dct(values, 1)).map((v) => v / (n - 1))
  d[0] = (d[0] ?? 0) / 2
  d[n - 1] = (d[n - 1] ?? 0) / 2
  return d
}

/** Options of {@link chebfunAlgorithm} and {@link chebfun}. */
export interface ChebfunOptions {
  /** Tolerance relative to the largest coefficient (default 2^-52 times 10). */
  readonly tol?: number
  /** The largest sample size (default 2^16 + 1). */
  readonly maxPoints?: number
}

/** The state of {@link chebfunAlgorithm}: the next sample size, the coefficients of the last interpolant and whether they decayed. */
export interface ChebfunState {
  /** The sample size the next step takes (17, 33, 65, ...). */
  readonly points: number
  /** The Chebyshev coefficients of the last interpolant (empty before the first step). */
  readonly coefficients: number[]
  /** The largest of the last coefficients relative to the largest (Infinity before the first step). */
  readonly tail: number
  readonly converged: boolean
}

/**
 * Adaptive Chebyshev approximation (chebfun's constructor) as a stepped algorithm: each step samples the function at 17,
 * 33, 65, ... Chebyshev points (doubling), computes the coefficients and tests whether the last eighth of them (at
 * least four) are negligible relative to the largest. It logs the degree, the tail and the coefficients, and is done
 * (`converged`) when they are, and `failed` when the sample would pass `maxPoints`. The state serialises.
 *
 * @param f - The function, vectorised over a 1-D value.
 * @param domain - The interval (default [-1, 1]).
 * @param options - Tolerance and the largest sample size.
 * @returns The algorithm.
 * @throws MdmlError with code `value` (while stepping) when the function is not finite on the domain.
 * @concept chebyshev-approximation
 * @example
 * ```ts
 * import { chebfunAlgorithm } from 'mdml-engine/numerics/approximation'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { exp } from 'mdml-engine/tensor'
 * const run = runAlgorithm(chebfunAlgorithm((x) => exp(x)), { maxSteps: 20 })
 * [run.status, run.steps, run.state.coefficients.length] // => ['converged', 2, 33]
 * ```
 */
export function chebfunAlgorithm(
  f: (x: Value) => Value,
  domain: readonly [number, number] = [-1, 1],
  options: ChebfunOptions = {},
): Algorithm<ChebfunState, ApproximationPath> {
  const [a, b] = domain
  const tol = options.tol ?? 10 * 2.220446049250313e-16
  const maxPoints = options.maxPoints ?? 65537
  return {
    name: 'numerics/chebfun',
    config: { domain: [a, b], tol, maxPoints },
    init: () => ({ points: 17, coefficients: [], tail: Number.POSITIVE_INFINITY, converged: false }),
    step: (s, ctx) => {
      const n = s.points
      const xs = Array.from({ length: n }, (_, j) => (a + b) / 2 + ((b - a) / 2) * Math.cos((j * Math.PI) / (n - 1)))
      const values = toFlat(primalOf(f(fromStorage(Float64Array.from(xs), [n], 'float64'))))
      if (values.some((v) => !Number.isFinite(v)))
        throw new MdmlError('value', 'chebfun: the function is not finite on the domain')
      const c = coefficientsOf(values)
      const scale = Math.max(maxOf(c.map(Math.abs)), Number.MIN_VALUE)
      // Converged when the last few coefficients (an eighth of them, at least 4) are all negligible.
      const tailCount = Math.max(4, Math.floor(n / 8))
      const tail = maxOf(c.slice(-tailCount).map((v) => Math.abs(v))) / scale
      const converged = c.slice(-tailCount).every((v) => Math.abs(v) <= tol * scale)
      ctx.log('chebfun/degree', n - 1)
      ctx.log('chebfun/tail', tail)
      if (ctx.wants('chebfun/coefficients')) ctx.log('chebfun/coefficients', c)
      return { points: converged ? n : 2 * n - 1, coefficients: c, tail, converged }
    },
    done: (s) => (s.converged ? 'converged' : s.points > maxPoints ? 'failed' : undefined),
    serialise: (s) => ({
      points: s.points,
      coefficients: numbersToJson(s.coefficients),
      tail: numberToJson(s.tail),
      converged: s.converged,
    }),
    deserialise: (j) => {
      const o = j as unknown as Record<string, Json>
      return {
        points: o['points'] as number,
        coefficients: numbersFromJson(o['coefficients']),
        tail: numberFromJson(o['tail']),
        converged: o['converged'] as boolean,
      }
    },
  }
}

/**
 * Adaptive Chebyshev approximation of a smooth function on an interval (chebfun's constructor): sample at 17, 33,
 * 65, ... Chebyshev points until the coefficients decay below tol relative to their largest, then chop the negligible
 * tail. Smooth functions need few terms (analytic functions converge geometrically). {@link chebfunAlgorithm} run to
 * the end.
 *
 * @param f - The function, vectorised over a 1-D value.
 * @param domain - The interval (default [-1, 1]).
 * @param options - Tolerance (default 2^-52 times 10) and the largest sample size (default 2^16 + 1).
 * @returns The expansion (a record of tensors); its coefficients show the decay.
 * @throws MdmlError with code `convergence` when the coefficients do not decay within the largest sample size.
 * @concept chebyshev-approximation
 * @index function Adaptive Chebyshev approximation of a smooth function.
 * @example
 * ```ts
 * import { chebfun } from 'mdml-engine/numerics/approximation'
 * import { exp } from 'mdml-engine/tensor'
 * const f = chebfun((x) => exp(x))
 * f.coefficients.shape[0] < 20 && Math.abs(f.sum() - (Math.E - 1 / Math.E)) < 1e-14 // => true
 * ```
 */
export function chebfun(
  f: (x: Value) => Value,
  domain: readonly [number, number] = [-1, 1],
  options: ChebfunOptions = {},
): Chebyshev {
  const tol = options.tol ?? 10 * 2.220446049250313e-16
  const maxPoints = options.maxPoints ?? 65537
  const s = runAlgorithm(chebfunAlgorithm(f, domain, options), { maxSteps: Number.MAX_SAFE_INTEGER }).state
  if (!s.converged)
    throw new MdmlError('convergence', `chebfun: no convergence with ${maxPoints} points (is the function smooth?)`)
  const c = s.coefficients
  const scale = Math.max(maxOf(c.map(Math.abs)), Number.MIN_VALUE)
  let last = c.length - 1
  while (last > 0 && Math.abs(c[last] ?? 0) <= tol * scale) last--
  return chebyshev(c.slice(0, last + 1), domain)
}
