// Test helpers for samplers: moment checks in units of Monte Carlo standard error, and standard targets.
import { expect } from 'vitest'
import { ndtr } from '../lib/numerics/special/index.ts'
import { foldIn, type Key, key } from '../lib/random/index.ts'
import {
  type Chains,
  ess,
  type LogDensity,
  type MCMCKernel,
  type MCMCState,
  mcse,
  numericTarget,
  parameterDraws,
} from '../lib/sampling/index.ts'
import { item, type Tensor } from '../lib/tensor/index.ts'

/** Mean and standard deviation of the pooled draws of one coordinate. */
export function moments(x: ArrayLike<number>): { readonly mean: number; readonly sd: number } {
  let m = 0
  for (let i = 0; i < x.length; i++) m += x[i] ?? 0
  m /= x.length
  let s = 0
  for (let i = 0; i < x.length; i++) s += ((x[i] ?? 0) - m) ** 2
  return { mean: m, sd: Math.sqrt(s / (x.length - 1)) }
}

/**
 * Check that each coordinate's posterior mean (and sd, when given) is within `z` Monte Carlo standard errors of the
 * truth, with MCSE from the chains' own effective sample size.
 */
export function expectMoments(
  chains: Chains,
  truth: { readonly mean: readonly number[]; readonly sd?: readonly number[] },
  z = 4,
): void {
  truth.mean.forEach((mu, i) => {
    const x = parameterDraws(chains, i)
    const { mean, sd } = moments(x.data)
    const se = mcse(x)
    expect(Math.abs(mean - mu), `mean of ${chains.names[i]}: ${mean} vs ${mu} (mcse ${se})`).toBeLessThan(z * se)
    const s = truth.sd?.[i]
    if (s !== undefined) {
      const seSd = mcse(x, { method: 'sd' })
      expect(Math.abs(sd - s), `sd of ${chains.names[i]}: ${sd} vs ${s} (mcse ${seSd})`).toBeLessThan(z * seSd)
    }
  })
}

/** Check moments of independent draws `[n, dim]` (row-major) with the plain standard error. */
export function expectIidMoments(
  draws: ArrayLike<number>,
  dim: number,
  truth: { readonly mean: readonly number[]; readonly sd?: readonly number[] },
  z = 4,
): void {
  const n = draws.length / dim
  truth.mean.forEach((mu, i) => {
    const x = Float64Array.from({ length: n }, (_, k) => draws[k * dim + i] ?? 0)
    const { mean, sd } = moments(x)
    expect(Math.abs(mean - mu), `mean ${i}: ${mean} vs ${mu}`).toBeLessThan((z * sd) / Math.sqrt(n))
    const s = truth.sd?.[i]
    if (s !== undefined) {
      // The sd of a sample sd is about sd / sqrt(2 n) for light tails; allow kurtosis with a factor 2.
      expect(Math.abs(sd - s), `sd ${i}: ${sd} vs ${s}`).toBeLessThan((2 * z * s) / Math.sqrt(2 * n))
    }
  })
}

/**
 * Invariance of the target under a kernel: start from `draws` (exact independent draws of the target, `[n, dim]`),
 * apply `steps` transitions to each with independent keys, and check that the results still have the target's mean
 * and sd (they are still independent draws of the target when the kernel leaves it invariant).
 */
export function expectInvariant<S extends MCMCState>(
  kernel: MCMCKernel<S>,
  draws: Float64Array,
  truth: { readonly mean: readonly number[]; readonly sd?: readonly number[] },
  options: { readonly key?: Key; readonly steps?: number; readonly z?: number } = {},
): Float64Array {
  const dim = kernel.dim
  const n = draws.length / dim
  const out = new Float64Array(draws.length)
  const k = options.key ?? key(0)
  for (let i = 0; i < n; i++) {
    let s = kernel.init(draws.subarray(i * dim, (i + 1) * dim))
    for (let t = 0; t < (options.steps ?? 1); t++) s = kernel.step(foldIn(foldIn(k, i), t), s).state
    out.set(s.position, i * dim)
  }
  expectIidMoments(out, dim, truth, options.z ?? 4)
  return out
}

/**
 * The Kolmogorov-Smirnov check of independent draws against a CDF: sup |F_n - F| below the 0.1% critical value
 * 1.95 / sqrt(n) (fixed seeds make the test deterministic; the level only sets how strict it is).
 */
export function expectDistribution(draws: ArrayLike<number>, cdf: (x: number) => number): void {
  const x = Float64Array.from(draws).sort()
  const n = x.length
  let d = 0
  for (let i = 0; i < n; i++) {
    const f = cdf(x[i] ?? 0)
    d = Math.max(d, Math.abs((i + 1) / n - f), Math.abs(i / n - f))
  }
  expect(d, `KS distance ${d} with n = ${n}`).toBeLessThan(1.95 / Math.sqrt(n))
}

/** The standard normal CDF. */
export function normalCdf(x: number): number {
  return item(ndtr(x))
}

/** Bulk ESS of every coordinate is at least `min`. */
export function expectMixing(chains: Chains, min: number): void {
  for (let i = 0; i < chains.dim; i++) {
    const e = ess(parameterDraws(chains, i) as Tensor)
    expect(e, `ess of ${chains.names[i]}`).toBeGreaterThan(min)
  }
}

/** A correlated bivariate Gaussian with mean `mu`, sds `s` and correlation `rho`, with its exact gradient. */
export function gaussian2(mu: readonly [number, number], s: readonly [number, number], rho: number): LogDensity {
  const det = 1 - rho * rho
  const z = (x: Float64Array): [number, number] => [((x[0] ?? 0) - mu[0]) / s[0], ((x[1] ?? 0) - mu[1]) / s[1]]
  return numericTarget(
    (x) => {
      const [a, b] = z(x)
      return (-0.5 * (a * a - 2 * rho * a * b + b * b)) / det
    },
    2,
    {
      grad: (x) => {
        const [a, b] = z(x)
        return [-(a - rho * b) / det / s[0], -(b - rho * a) / det / s[1]]
      },
    },
  )
}

/** The standard normal in `dim` coordinates with its gradient. */
export function standardNormal(dim: number): LogDensity {
  return numericTarget(
    (x) => {
      let s = 0
      for (const v of x) s += v * v
      return -0.5 * s
    },
    dim,
    { grad: (x) => x.map((v) => -v) },
  )
}

/** An equal mixture of N(-d, 1) and N(d, 1) in one coordinate, with its gradient. */
export function bimodal(d: number): LogDensity {
  const lp = (x: number): number => {
    const a = -0.5 * (x + d) ** 2
    const b = -0.5 * (x - d) ** 2
    const m = Math.max(a, b)
    return m + Math.log(0.5 * Math.exp(a - m) + 0.5 * Math.exp(b - m))
  }
  return numericTarget((x) => lp(x[0] ?? 0), 1, {
    grad: (x) => {
      const v = x[0] ?? 0
      const a = -0.5 * (v + d) ** 2
      const b = -0.5 * (v - d) ** 2
      const m = Math.max(a, b)
      const wa = Math.exp(a - m)
      const wb = Math.exp(b - m)
      return [(wa * -(v + d) + wb * -(v - d)) / (wa + wb)]
    },
  })
}
