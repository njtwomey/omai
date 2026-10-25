/**
 * Rejection samplers that produce independent exact draws from an unnormalised density: classic rejection sampling
 * under an envelope M q(x), adaptive rejection sampling for log-concave densities (Gilks and Wild), whose
 * piecewise-exponential envelope tightens with each rejection, and the ratio-of-uniforms method.
 */

import { logAddExp, MdmlError } from '../base/index.ts'
import { foldIn, type Key, uniformStream } from '../random/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import type { LogDensity, Proposal } from './target.ts'

/** The result of the rejection samplers. */
export interface RejectionResult {
  /** The accepted draws `[n, dim]` (fewer than asked when the proposal budget ran out). */
  readonly samples: Tensor<'float64'>
  /** Proposals made. */
  readonly proposals: number
  /** Accepted / proposed. */
  readonly acceptanceRate: number
  /** Whether all n draws were obtained within the budget. */
  readonly complete: boolean
}

/** Options of {@link rejectionSample}. */
export interface RejectionOptions {
  readonly key: Key
  readonly n: number
  /** log M with pi(x) <= M q(x) everywhere (pi may be unnormalised). */
  readonly logEnvelope: number
  /** The most proposals (default 1000 n). */
  readonly maxProposals?: number
}

/**
 * Rejection sampling (von Neumann 1951): draw x ~ q and u ~ U(0, 1), and keep x when u M q(x) <= pi(x). Accepted
 * draws are exact samples of pi, and the acceptance rate is Z / M. Points where pi(x) > M q(x) break the guarantee;
 * they are counted in `envelopeViolations`.
 *
 * @param target - The target (unnormalised is fine).
 * @param q - The proposal.
 * @param options - Key, number of draws, log M and the proposal budget.
 * @returns The draws, proposals, acceptance rate, completeness and envelope violations.
 * @throws MdmlError with code `shape` when dimensions differ, and `value` for a bad number of draws.
 * @concept rejection-sampling
 * @reference J. von Neumann, Various techniques used in connection with random digits (1951).
 * @example
 * ```ts
 * import { numericTarget, proposalFromDistribution, rejectionSample } from 'mdml-engine/sampling'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * // pi(x) = exp(-x^2 / 2) and q = N(0, 2^2): pi / q <= 2 sqrt(2 pi).
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const r = rejectionSample(t, proposalFromDistribution(normal(0, 2)), { key: key(0), n: 500, logEnvelope: Math.log(2 * Math.sqrt(2 * Math.PI)) })
 * Math.abs(r.acceptanceRate - 0.5) < 0.06 // => true
 * ```
 */
export function rejectionSample(
  target: LogDensity,
  q: Proposal,
  options: RejectionOptions,
): RejectionResult & { readonly envelopeViolations: number } {
  if (q.dim !== target.dim) throw new MdmlError('shape', 'rejectionSample: proposal and target dimensions differ')
  const { n } = options
  if (!(Number.isInteger(n) && n >= 0))
    throw new MdmlError('value', 'rejectionSample: n must be a non-negative integer')
  const dim = target.dim
  const budget = options.maxProposals ?? 1000 * Math.max(1, n)
  const out = new Float64Array(n * dim)
  const rand = uniformStream(foldIn(options.key, 1), 64)
  let accepted = 0
  let proposals = 0
  let violations = 0
  let batch = 0
  while (accepted < n && proposals < budget) {
    const size = Math.min(budget - proposals, Math.max(16, 2 * (n - accepted)))
    const xs = q.sampleMany(foldIn(foldIn(options.key, 0), batch), size)
    const lq = q.logProbMany(xs)
    batch += 1
    for (let i = 0; i < size && accepted < n; i++) {
      proposals += 1
      const x = xs.subarray(i * dim, (i + 1) * dim)
      const logRatio = target.logDensity(x) - options.logEnvelope - (lq[i] ?? 0)
      if (logRatio > 1e-12) violations += 1
      if (Math.log(1 - rand()) <= logRatio) {
        out.set(x, accepted * dim)
        accepted += 1
      }
    }
  }
  return {
    samples: fromStorage(out.slice(0, accepted * dim), [accepted, dim], 'float64'),
    proposals,
    acceptanceRate: proposals === 0 ? Number.NaN : accepted / proposals,
    complete: accepted === n,
    envelopeViolations: violations,
  }
}

/** Options of {@link adaptiveRejectionSample}. */
export interface AdaptiveRejectionOptions {
  readonly key: Key
  readonly n: number
  /** Starting abscissae (at least two; default -1 and 1). With an infinite bound the outermost must slope the right way. */
  readonly initialPoints?: readonly number[]
  /** The support [low, high] (default the real line). */
  readonly domain?: readonly [number, number]
  /** The most abscissae in the hull (default 50). */
  readonly maxPoints?: number
  /** The most proposals (default 100 n). */
  readonly maxProposals?: number
}

/**
 * Adaptive rejection sampling (Gilks and Wild 1992) for a log-concave density on an interval: the tangents of
 * h = log f at the abscissae form a piecewise-exponential envelope and the chords a squeeze. Draws from the envelope
 * are accepted by the squeeze test when possible (no evaluation), otherwise by evaluating h; every evaluation
 * becomes a new abscissa, so the envelope tightens and the acceptance rate approaches 1.
 *
 * @param target - A one-dimensional log-concave target with a gradient.
 * @param options - Key, number of draws, starting points, domain, hull size and proposal budget.
 * @returns The draws `[n, 1]`, proposals, acceptance rate, completeness and the final abscissae.
 * @throws MdmlError with code `value` for a target that is not one-dimensional with a gradient, fewer than two
 *   starting points, points outside the domain, an envelope that cannot be normalised (wrong slopes at infinite
 *   bounds), or a density found not to be log-concave.
 * @concept adaptive-rejection-sampling
 * @reference W. R. Gilks and P. Wild, Adaptive rejection sampling for Gibbs sampling (Applied Statistics 1992).
 * @example
 * ```ts
 * import { adaptiveRejectionSample, numericTarget } from 'mdml-engine/sampling'
 * import { key } from 'mdml-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * const r = adaptiveRejectionSample(t, { key: key(0), n: 1000 })
 * r.acceptanceRate > 0.9 // => true
 * ```
 */
export function adaptiveRejectionSample(
  target: LogDensity,
  options: AdaptiveRejectionOptions,
): RejectionResult & { readonly abscissae: readonly number[] } {
  if (target.dim !== 1 || !target.hasGradient) {
    throw new MdmlError('value', 'adaptiveRejectionSample: need a one-dimensional target with a gradient')
  }
  const { n } = options
  const [lo, hi] = options.domain ?? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
  const maxPoints = options.maxPoints ?? 50
  const budget = options.maxProposals ?? 100 * Math.max(1, n)
  const init = [...(options.initialPoints ?? [-1, 1])].sort((a, b) => a - b)
  if (init.length < 2 || init.some((x) => !(x > lo && x < hi))) {
    throw new MdmlError('value', 'adaptiveRejectionSample: need two starting points inside the domain')
  }
  const xs: number[] = []
  const hs: number[] = []
  const ds: number[] = []
  const insert = (x: number): void => {
    const r = target.valueAndGrad([x])
    let i = 0
    while (i < xs.length && (xs[i] ?? 0) < x) i++
    if (xs[i] === x) return
    xs.splice(i, 0, x)
    hs.splice(i, 0, r.value)
    ds.splice(i, 0, r.grad[0] ?? 0)
    for (let j = 1; j < ds.length; j++) {
      if ((ds[j] ?? 0) > (ds[j - 1] ?? 0) + 1e-9 * (1 + Math.abs(ds[j - 1] ?? 0))) {
        throw new MdmlError('value', 'adaptiveRejectionSample: the density is not log-concave')
      }
    }
  }
  for (const x of init) insert(x)
  if (!hs.every(Number.isFinite))
    throw new MdmlError('value', 'adaptiveRejectionSample: the log density must be finite at the starting points')
  if (
    (lo === Number.NEGATIVE_INFINITY && !((ds[0] ?? 0) > 0)) ||
    (hi === Number.POSITIVE_INFINITY && !((ds[ds.length - 1] ?? 0) < 0))
  ) {
    throw new MdmlError('value', 'adaptiveRejectionSample: with an infinite bound the outer tangents must point down')
  }

  /** The envelope: breakpoints z_0..z_k and the log masses of the k tangent pieces. */
  const hull = (): { readonly z: number[]; readonly logMass: number[] } => {
    const k = xs.length
    const z: number[] = [lo]
    for (let j = 0; j < k - 1; j++) {
      const [x0, x1, h0, h1, d0, d1] = [
        xs[j] ?? 0,
        xs[j + 1] ?? 0,
        hs[j] ?? 0,
        hs[j + 1] ?? 0,
        ds[j] ?? 0,
        ds[j + 1] ?? 0,
      ]
      // Where the tangents at x0 and x1 meet (the midpoint when they are parallel).
      z.push(Math.abs(d0 - d1) < 1e-12 ? 0.5 * (x0 + x1) : (h1 - h0 - x1 * d1 + x0 * d0) / (d0 - d1))
    }
    z.push(hi)
    const logMass = xs.map((x, j) => logPieceMass(hs[j] ?? 0, ds[j] ?? 0, x, z[j] ?? lo, z[j + 1] ?? hi))
    return { z, logMass }
  }
  const upper = (x: number, z: readonly number[]): number => {
    let j = 0
    while (j < xs.length - 1 && x > (z[j + 1] ?? 0)) j++
    return (hs[j] ?? 0) + (ds[j] ?? 0) * (x - (xs[j] ?? 0))
  }
  const lower = (x: number): number => {
    if (x < (xs[0] ?? 0) || x > (xs[xs.length - 1] ?? 0)) return Number.NEGATIVE_INFINITY
    let j = 0
    while (j < xs.length - 2 && x > (xs[j + 1] ?? 0)) j++
    const [x0, x1, h0, h1] = [xs[j] ?? 0, xs[j + 1] ?? 0, hs[j] ?? 0, hs[j + 1] ?? 0]
    return ((x1 - x) * h0 + (x - x0) * h1) / (x1 - x0)
  }

  const rand = uniformStream(options.key, 64)
  const out = new Float64Array(n)
  let accepted = 0
  let proposals = 0
  let env = hull()
  while (accepted < n && proposals < budget) {
    proposals += 1
    const total = env.logMass.reduce((a, b) => logAddExp(a, b), Number.NEGATIVE_INFINITY)
    if (!Number.isFinite(total))
      throw new MdmlError('value', 'adaptiveRejectionSample: the envelope cannot be normalised')
    // Choose a piece by its mass, then invert its exponential CDF.
    let u = rand()
    let j = 0
    for (; j < env.logMass.length - 1; j++) {
      const p = Math.exp((env.logMass[j] ?? Number.NEGATIVE_INFINITY) - total)
      if (u < p) break
      u -= p
    }
    const x = samplePiece(ds[j] ?? 0, xs[j] ?? 0, env.z[j] ?? lo, env.z[j + 1] ?? hi, rand())
    const w = Math.log(1 - rand())
    const ux = upper(x, env.z)
    if (w <= lower(x) - ux) {
      out[accepted++] = x
      continue
    }
    const hx = target.logDensity([x])
    if (w <= hx - ux) out[accepted++] = x
    if (xs.length < maxPoints && Number.isFinite(hx)) {
      insert(x)
      env = hull()
    }
  }
  return {
    samples: fromStorage(out.slice(0, accepted), [accepted, 1], 'float64'),
    proposals,
    acceptanceRate: proposals === 0 ? Number.NaN : accepted / proposals,
    complete: accepted === n,
    abscissae: [...xs],
  }
}

/** log of the integral of exp(h + d (x - x0)) over [a, b]. */
function logPieceMass(h: number, d: number, x0: number, a: number, b: number): number {
  if (!(b > a)) return Number.NEGATIVE_INFINITY
  if (Math.abs(d) < 1e-12) return h + Math.log(b - a)
  // log (exp(d (b - x0)) - exp(d (a - x0))) / |d|, with the larger exponent factored out.
  const ea = d * (a - x0)
  const eb = d * (b - x0)
  const hiE = Math.max(ea, eb)
  const loE = Math.min(ea, eb)
  return h + hiE + Math.log1p(-Math.exp(loE - hiE)) - Math.log(Math.abs(d))
}

/** Invert the CDF of exp(d (x - x0)) on [a, b] at u. */
function samplePiece(d: number, x0: number, a: number, b: number, u: number): number {
  if (Math.abs(d) < 1e-12) return a + u * (b - a)
  const ea = d * (a - x0)
  const eb = d * (b - x0)
  // x = x0 + log(exp(ea) + u (exp(eb) - exp(ea))) / d, stably in log space.
  const m = Math.max(ea, eb)
  const v = Math.exp(ea - m) + u * (Math.exp(eb - m) - Math.exp(ea - m))
  const x = x0 + (m + Math.log(v)) / d
  return Math.min(Math.max(x, a), b)
}

/** Options of {@link ratioOfUniforms}. */
export interface RatioOfUniformsOptions {
  readonly key: Key
  readonly n: number
  /**
   * The bounding rectangle [0, uMax] x [vMin, vMax] of {(u, v): 0 < u <= sqrt(f(v / u))}: uMax = sup sqrt f, vMin and
   * vMax the inf and sup of x sqrt f(x). Default: found by a grid search over `range`, enlarged by 1%.
   */
  readonly bounds?: { readonly uMax: number; readonly vMin: number; readonly vMax: number }
  /** The interval searched for the bounds (default [-30, 30]). */
  readonly range?: readonly [number, number]
  /** The most proposals (default 1000 n). */
  readonly maxProposals?: number
}

/**
 * The ratio-of-uniforms method (Kinderman and Monahan 1977): for an unnormalised density f, (u, v) uniform on
 * C = {(u, v): 0 < u <= sqrt(f(v / u))} gives x = v / u ~ f. C is sampled by rejection from its bounding rectangle,
 * found here by a grid search when not given (densities with tails heavier than 1/x^2 have no bounded C).
 *
 * @param density - f(x) >= 0, unnormalised (log densities: pass exp of them).
 * @param options - Key, number of draws, bounds or search range, and the proposal budget.
 * @returns The draws `[n, 1]`, proposals, acceptance rate, completeness and the rectangle used.
 * @throws MdmlError with code `value` for a density that is zero on the search range or bounds that are not
 *   finite.
 * @concept ratio-of-uniforms
 * @reference A. J. Kinderman and J. F. Monahan, Computer generation of random variables using the ratio of uniform
 *   deviates (ACM TOMS 1977).
 * @example
 * ```ts
 * import { ratioOfUniforms } from 'mdml-engine/sampling'
 * import { key } from 'mdml-engine/random'
 * // The standard normal: the region has area sqrt(2 pi) / 2 inside a 1 x 2 sqrt(2 / e) box.
 * const r = ratioOfUniforms((x) => Math.exp(-0.5 * x * x), { key: key(0), n: 2000 })
 * Math.abs(r.acceptanceRate - Math.sqrt(Math.PI * Math.E) / 4) < 0.03 // => true
 * ```
 */
export function ratioOfUniforms(
  density: (x: number) => number,
  options: RatioOfUniformsOptions,
): RejectionResult & { readonly bounds: { readonly uMax: number; readonly vMin: number; readonly vMax: number } } {
  const { n } = options
  const f = (x: number): number => {
    const v = density(x)
    return v > 0 && Number.isFinite(v) ? v : 0
  }
  let bounds = options.bounds
  if (bounds === undefined) {
    const [a, b] = options.range ?? [-30, 30]
    let uMax = 0
    let vMin = 0
    let vMax = 0
    const grid = 20001
    for (let i = 0; i < grid; i++) {
      const x = a + ((b - a) * i) / (grid - 1)
      const s = Math.sqrt(f(x))
      uMax = Math.max(uMax, s)
      vMin = Math.min(vMin, x * s)
      vMax = Math.max(vMax, x * s)
    }
    if (!(uMax > 0)) throw new MdmlError('value', 'ratioOfUniforms: the density is zero on the search range')
    bounds = { uMax: 1.01 * uMax, vMin: 1.01 * vMin, vMax: 1.01 * vMax }
  }
  if (![bounds.uMax, bounds.vMin, bounds.vMax].every(Number.isFinite) || !(bounds.uMax > 0)) {
    throw new MdmlError('value', 'ratioOfUniforms: the bounds must be finite with uMax > 0')
  }
  const budget = options.maxProposals ?? 1000 * Math.max(1, n)
  const rand = uniformStream(options.key, 64)
  const out = new Float64Array(n)
  let accepted = 0
  let proposals = 0
  while (accepted < n && proposals < budget) {
    proposals += 1
    const u = bounds.uMax * (1 - rand())
    const v = bounds.vMin + (bounds.vMax - bounds.vMin) * rand()
    const x = v / u
    if (u * u <= f(x)) out[accepted++] = x
  }
  return {
    samples: fromStorage(out.slice(0, accepted), [accepted, 1], 'float64'),
    proposals,
    acceptanceRate: proposals === 0 ? Number.NaN : accepted / proposals,
    complete: accepted === n,
    bounds,
  }
}
