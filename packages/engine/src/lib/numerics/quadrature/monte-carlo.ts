/**
 * Integration by random sampling: plain Monte Carlo over a box with its standard error, randomised quasi-Monte Carlo
 * with randomly shifted Halton points, and importance sampling from a proposal with the effective sample size of its
 * weights.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, halton, type Key, uniform } from '../../random/index.ts'
import {
  add,
  div,
  exp,
  fromStorage,
  item,
  mean,
  mul,
  primalOf,
  type Tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'

/** A Monte Carlo estimate with its standard error. */
export interface MonteCarloResult {
  /** The estimate, differentiable in the integrand's parameters. */
  readonly value: Value
  readonly standardError: number
  readonly samples: number
}

function boxVolume(box: readonly (readonly [number, number])[]): number {
  return box.reduce((v, [lo, hi]) => v * (hi - lo), 1)
}

function scaleToBox(u: Float64Array, n: number, box: readonly (readonly [number, number])[]): Tensor {
  const d = box.length
  const p = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < d; k++) {
      const [lo, hi] = box[k] as readonly [number, number]
      p[i * d + k] = lo + (hi - lo) * (u[i * d + k] ?? 0)
    }
  }
  return fromStorage(p, [n, d], 'float64')
}

function standardError(values: Value, n: number): number {
  const v = toFlat(primalOf(values))
  const m = v.reduce((s, x) => s + x, 0) / n
  const variance = v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, n - 1)
  return Math.sqrt(variance / n)
}

/**
 * Plain Monte Carlo integration over a box: volume times the mean of f at n uniform points, with the standard error
 * volume * s / sqrt(n).
 *
 * @param f - The integrand on points `[n, d]`, returning `[n]`.
 * @param box - The interval of each variable.
 * @param key - Random key.
 * @param n - Number of samples.
 * @returns The estimate and its standard error.
 * @throws OmaiError with code `value` when `n` is not a positive integer.
 * @concept monte-carlo
 * @example
 * ```ts
 * import { monteCarlo } from 'omai-engine/numerics/quadrature'
 * import { key } from 'omai-engine/random'
 * import { item, sum, square } from 'omai-engine/tensor'
 * const r = monteCarlo((p) => sum(square(p), -1), [[0, 1], [0, 1]], key(0), 20000)
 * Math.abs(item(r.value) - 2 / 3) < 4 * r.standardError // => true
 * ```
 */
export function monteCarlo(
  f: (points: Value) => Value,
  box: readonly (readonly [number, number])[],
  key: Key,
  n: number,
): MonteCarloResult {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', 'monteCarlo: n must be a positive integer')
  const u = uniform(key, [n, box.length]).data as Float64Array
  const values = mul(f(scaleToBox(u, n, box)), boxVolume(box))
  return { value: mean(values), standardError: standardError(values, n), samples: n }
}

/**
 * Randomised quasi-Monte Carlo integration over a box: the Halton sequence under `replicates` independent
 * Cranley-Patterson random shifts (x + u mod 1); the estimate is the mean over replicates and the standard error comes
 * from their spread. Error close to O(1 / n) for smooth integrands, against O(1 / sqrt(n)) for plain Monte Carlo.
 *
 * @param f - The integrand on points `[n, d]`, returning `[n]`.
 * @param box - The interval of each variable.
 * @param key - Random key for the shifts.
 * @param n - Points per replicate.
 * @param replicates - Number of shifted replicates (default 16).
 * @returns The estimate and its standard error.
 * @example
 * ```ts
 * import { quasiMonteCarlo } from 'omai-engine/numerics/quadrature'
 * import { key } from 'omai-engine/random'
 * import { item, sum, square } from 'omai-engine/tensor'
 * Math.abs(item(quasiMonteCarlo((p) => sum(square(p), -1), [[0, 1], [0, 1]], key(0), 1024).value) - 2 / 3) < 1e-3 // => true
 * ```
 */
export function quasiMonteCarlo(
  f: (points: Value) => Value,
  box: readonly (readonly [number, number])[],
  key: Key,
  n: number,
  replicates = 16,
): MonteCarloResult {
  if (replicates < 2) throw new OmaiError('value', 'quasiMonteCarlo: at least two replicates are needed')
  const d = box.length
  const base = halton(n, d).data as Float64Array
  const estimates: Value[] = []
  for (let r = 0; r < replicates; r++) {
    const shift = uniform(foldIn(key, r), [d]).data as Float64Array
    const u = base.map((v, i) => (v + (shift[i % d] ?? 0)) % 1)
    estimates.push(mul(mean(f(scaleToBox(u, n, box))), boxVolume(box)))
  }
  const total = estimates.reduce((a, b) => add(a, b))
  const value = div(total, replicates)
  const means = estimates.map((e) => item(primalOf(e)))
  const m = means.reduce((s, x) => s + x, 0) / replicates
  const variance = means.reduce((s, x) => s + (x - m) ** 2, 0) / (replicates - 1)
  return { value, standardError: Math.sqrt(variance / replicates), samples: n * replicates }
}

/** A proposal for importance sampling: a sampler and its log density. */
export interface Proposal {
  readonly sample: (key: Key, n: number) => Value
  readonly logDensity: (points: Value) => Value
}

/** Result of {@link importanceIntegral}. */
export interface ImportanceResult extends MonteCarloResult {
  /** Kish's effective sample size (sum w)^2 / sum w^2 of the weights f / q. */
  readonly effectiveSampleSize: number
}

/**
 * Importance-sampled integration: the integral of f is E_q[f(X) / q(X)] for X from the proposal q, estimated by the
 * sample mean, with its standard error and the effective sample size of the weights.
 *
 * @param f - The integrand on points `[n, d]`, returning `[n]`.
 * @param proposal - The proposal's sampler and log density.
 * @param key - Random key.
 * @param n - Number of samples.
 * @returns The estimate, standard error and effective sample size.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importanceIntegral } from 'omai-engine/numerics/quadrature'
 * import { key, normal } from 'omai-engine/random'
 * import { exp, item, mul, neg, square, sum, sub } from 'omai-engine/tensor'
 * // The integral of exp(-x^2) over the line, with a standard normal proposal: sqrt(pi).
 * const proposal = { sample: (k, n) => normal(k, [n, 1]), logDensity: (p) => sub(mul(sum(square(p), -1), -0.5), 0.9189385332046727) }
 * const r = importanceIntegral((p) => exp(neg(sum(square(p), -1))), proposal, key(1), 20000)
 * Math.abs(item(r.value) - Math.sqrt(Math.PI)) < 4 * r.standardError // => true
 * ```
 */
export function importanceIntegral(
  f: (points: Value) => Value,
  proposal: Proposal,
  key: Key,
  n: number,
): ImportanceResult {
  const points = primalOf(proposal.sample(key, n))
  const ratios = mul(f(points), exp(mul(proposal.logDensity(points), -1)))
  const w = toFlat(primalOf(ratios))
  const s1 = w.reduce((s, x) => s + x, 0)
  const s2 = w.reduce((s, x) => s + x * x, 0)
  return {
    value: mean(ratios),
    standardError: standardError(ratios, n),
    samples: n,
    effectiveSampleSize: s2 === 0 ? 0 : (s1 * s1) / s2,
  }
}
