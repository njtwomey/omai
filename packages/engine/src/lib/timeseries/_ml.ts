/**
 * Maximum-likelihood fitting shared by the time-series models: a number-level log-likelihood becomes an `Objective`
 * whose derivative is the central difference (a custom vjp), so the library's traced BFGS minimises it, and the BFGS
 * algorithm gains the likelihood as a metric and a serialisable state.
 */
import { customVjp } from '../autodiff/index.ts'
import { inv, lstsq } from '../numerics/linalg/index.ts'
import { type BfgsState, bfgsAlgorithm, type Objective } from '../optim/index.ts'
import { fromStorage, mul, primalOf, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** A scalar function of a flat parameter vector as an `Objective` differentiated by central differences. */
export function finiteDifferenceObjective(f: (x: Float64Array) => number, name: string, eps = 1e-6): Objective {
  const fn = customVjp({
    name,
    forward: (x) => fromStorage(Float64Array.of(f(Float64Array.from(toFlat(primalOf(x))))), [], 'float64'),
    vjp: (g, [x]) => {
      const x0 = Float64Array.from(toFlat(primalOf(x as Value)))
      const grad = new Float64Array(x0.length)
      for (let i = 0; i < x0.length; i++) {
        const h = eps * Math.max(1, Math.abs(x0[i] ?? 0))
        const up = Float64Array.from(x0)
        const down = Float64Array.from(x0)
        up[i] = (x0[i] ?? 0) + h
        down[i] = (x0[i] ?? 0) - h
        grad[i] = (f(up) - f(down)) / (2 * h)
      }
      return [mul(g, fromStorage(grad, [grad.length], 'float64'))]
    },
  })
  return (x) => fn(x as Value)
}

/** Options of {@link likelihoodAlgorithm}. */
export interface LikelihoodOptions {
  /** The gradient tolerance of the per-observation negative log-likelihood (default 1e-6). */
  readonly gradTol?: number
}

/**
 * BFGS on a negative log-likelihood divided by the number of observations, as a traced algorithm that also reports the
 * log-likelihood and serialises its state.
 *
 * @param name - The algorithm's name.
 * @param config - Its JSON configuration.
 * @param negLogLikPerObs - The objective: minus the log-likelihood over `nobs`, at a flat parameter vector.
 * @param x0 - The start.
 * @param nobs - The number of observations.
 * @param options - The gradient tolerance.
 * @returns The algorithm.
 */
export function likelihoodAlgorithm(
  name: string,
  config: object,
  negLogLikPerObs: (x: Float64Array) => number,
  x0: ArrayLike<number>,
  nobs: number,
  options: LikelihoodOptions = {},
): Algorithm<BfgsState> {
  const base = bfgsAlgorithm(finiteDifferenceObjective(negLogLikPerObs, `${name}-nll`), Array.from(x0), {
    gradTol: options.gradTol ?? 1e-6,
  })
  const baseSchema = base.schema
  return {
    ...base,
    name,
    config: { ...config, nobs },
    schema: channels({
      metrics: {
        ...(baseSchema?.metrics ?? {}),
        logLikelihood: { description: 'the log-likelihood at the iterate', direction: 'higher' },
      },
      tensors: baseSchema?.tensors ?? {},
      messages: baseSchema?.messages ?? {},
    }),
    observe: (s, t) => {
      const o = base.observe?.(s, t) ?? {}
      return { ...o, metrics: { ...(o.metrics ?? {}), logLikelihood: -s.value * nobs } }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BfgsState,
  }
}

/** The least-squares coefficients of `y` on the given columns (empty for no columns). */
export function leastSquares(columns: readonly ArrayLike<number>[], y: ArrayLike<number>): number[] {
  if (columns.length === 0) return []
  const n = y.length
  const k = columns.length
  const a = new Float64Array(n * k)
  columns.forEach((c, j) => {
    for (let i = 0; i < n; i++) a[i * k + j] = c[i] ?? 0
  })
  return toFlat(lstsq(fromStorage(a, [n, k], 'float64'), fromStorage(Float64Array.from(y), [n], 'float64')))
}

/** The Hessian of a scalar function by central differences, `n x n` row-major. */
export function numericalHessian(f: (x: Float64Array) => number, x: ArrayLike<number>, rel = 1e-4): Float64Array {
  const n = x.length
  const x0 = Float64Array.from(x)
  const h = Float64Array.from(x0, (v) => rel * Math.max(1, Math.abs(v)))
  const at = (di: number, hi: number, dj: number, hj: number): number => {
    const p = Float64Array.from(x0)
    p[di] = (p[di] ?? 0) + hi
    p[dj] = (p[dj] ?? 0) + hj
    return f(p)
  }
  const out = new Float64Array(n * n)
  const f0 = f(x0)
  for (let i = 0; i < n; i++) {
    const hi = h[i] ?? 1e-4
    out[i * n + i] = (at(i, hi, i, 0) - 2 * f0 + at(i, -hi, i, 0)) / (hi * hi)
    for (let j = i + 1; j < n; j++) {
      const hj = h[j] ?? 1e-4
      const v = (at(i, hi, j, hj) - at(i, hi, j, -hj) - at(i, -hi, j, hj) + at(i, -hi, j, -hj)) / (4 * hi * hj)
      out[i * n + j] = v
      out[j * n + i] = v
    }
  }
  return out
}

/** The square roots of the diagonal of the inverse of a symmetric matrix (NaN where it is not positive). */
export function inverseDiagonalRoots(h: Float64Array, n: number): number[] {
  const covariance = toFlat(inv(fromStorage(h, [n, n], 'float64')))
  return Array.from({ length: n }, (_, i) => Math.sqrt(covariance[i * n + i] ?? Number.NaN))
}
