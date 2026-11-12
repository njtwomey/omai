/**
 * Gaussian process regression as a spec: twelve noisy observations of a smooth function, a squared-exponential kernel with an
 * unknown length scale (a `LogNormal(0, 0.5)` prior), a known amplitude and a known noise level, written with a latent
 * function vector `f ~ MVN(0, K(ell))` and `y ~ MVN(f, sigma² I)`. The covariance is the library's own Gram matrix
 * (`omai-engine/kernels`, differentiable in the length scale) supplied through a `custom` function, so it is opaque to
 * message passing and runs on every sampling engine. The truth is exact given the length scale (the function posterior is a
 * Gaussian whose mean and covariance are the textbook formulas) and one-dimensional quadrature over the length scale, with
 * the function values' marginals as the grid mixtures of those Gaussians, as in the eight-schools benchmark.
 */

import { logNormal as logNormalDist, multivariateNormal } from '../../dists/index.ts'
import { gram, rbfKernel } from '../../kernels/index.ts'
import { cholesky, dgemv, dpotrf, dpotrs } from '../../numerics/linalg/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, type Key, key as makeKey } from '../../random/index.ts'
import {
  add,
  clip,
  eye,
  full,
  matmul,
  mul,
  reshape,
  tensor,
  toArray,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel } from '../model/index.ts'
import { observe, sample } from '../primitives.ts'
import { gridPosterior, type MarginalOracle } from '../validation/index.ts'
import { type BenchmarkData, type BenchmarkModel, type BenchmarkTruth, defineBenchmark } from './benchmark.ts'
import { laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'
import { conditionalOracle } from './truth.ts'

const POINTS = 12
const AMPLITUDE = 1
const NOISE = 0.3
const JITTER = 1e-6

/**
 * The squared-exponential Gram matrix of the inputs at a length scale, with the amplitude and a little jitter. The length scale
 * is kept in [0.02, 20] (the prior puts no mass outside it to speak of), so a wild step of a sampler cannot make the matrix
 * non-finite.
 */
function kernelMatrix(x: Value, ell: Value): Value {
  const rows = reshape(x, [x.shape[0] ?? POINTS, 1])
  return gram(rbfKernel({ lengthScale: clip(ell, 0.02, 20) }), rows, { jitter: JITTER / AMPLITUDE ** 2 })
}

function generateGp(key: Key): BenchmarkData {
  const u = toFlat(drawUniform(foldIn(key, 0), [POINTS]))
  const e = toFlat(drawNormal(foldIn(key, 1), [POINTS]))
  const x = Array.from(u, (v) => -3 + 6 * v).sort((a, b) => a - b)
  const f = x.map((v) => Math.sin(1.2 * v) + 0.2 * v)
  return {
    data: { x, y: f.map((v, i) => v + NOISE * (e[i] ?? 0)) },
    generating: { amplitude: AMPLITUDE, noise: NOISE, f },
  }
}

/** The Gaussian process regression spec: the length scale, the function vector and the noisy observations. */
const gpSpec = defineModel('gaussian-process-regression', (b) => {
  const x = b.data('x', { dims: ['point'], role: 'covariate' })
  const y = b.data('y', { eventRank: 1, role: 'response' })
  const ell = b.latent('ell', d.logNormal(0, 0.5), { doc: 'the length scale of the squared-exponential kernel' })
  const K = b.custom(
    'rbfGram',
    (args) => {
      const [xs, scale] = args
      return kernelMatrix((xs as { value: Value }).value, (scale as { value: Value }).value)
    },
    [x, ell],
    { eventRank: 2 },
  )
  const f = b.latent('f', d.multivariateNormal(zeros([POINTS]), { covariance: K }), {
    doc: 'the function at the inputs',
  })
  b.observe(
    'y',
    d.multivariateNormal(f, {
      covariance: b.const(toArray(mul(eye(POINTS), NOISE * NOISE)) as number[][]),
    }),
    y,
  )
})

/**
 * The posterior of the function values given a length scale: the mean `K (K + sigma² I)^-1 y` and the marginal standard
 * deviations of `K - K (K + sigma² I)^-1 K`, the textbook formulas, by a Cholesky factorisation of `K + sigma² I`.
 *
 * @param x - The inputs.
 * @param y - The observations.
 * @param ell - The length scale.
 * @returns The mean and standard deviation of each function value, and the log marginal likelihood `log N(y; 0, K + sigma² I)`.
 * @concept gaussian-processes
 * @example
 * ```ts
 * import { gpPosteriorGivenLengthScale } from 'omai-engine/probprog/models'
 * const p = gpPosteriorGivenLengthScale([0, 1, 2], [0, 1, 0], 1)
 * [p.mean.length, p.sd.every((s) => s > 0 && s < 1)] // => [3, true]
 * ```
 */
export function gpPosteriorGivenLengthScale(
  x: readonly number[],
  y: readonly number[],
  ell: number,
): { readonly mean: number[]; readonly sd: number[]; readonly logMarginalLikelihood: number } {
  const n = x.length
  const K = toFlat(kernelMatrixOf(x, ell))
  const C = Float64Array.from(K)
  for (let i = 0; i < n; i++) C[i * n + i] = (C[i * n + i] ?? 0) + NOISE * NOISE
  const { factor } = dpotrf(C, n)
  const alpha = dpotrs(factor, y, n)
  const mean = Array.from(dgemv(K, n, n, alpha))
  const sd = Array.from({ length: n }, (_, i) => {
    const row = K.slice(i * n, (i + 1) * n)
    const solved = dpotrs(factor, row, n)
    let reduction = 0
    for (let j = 0; j < n; j++) reduction += (row[j] ?? 0) * (solved[j] ?? 0)
    return Math.sqrt(Math.max((K[i * n + i] ?? 0) - reduction, 1e-14))
  })
  let logDet = 0
  for (let i = 0; i < n; i++) logDet += 2 * Math.log(factor[i * n + i] ?? 1)
  const quad = y.reduce((a, v, i) => a + v * (alpha[i] ?? 0), 0)
  return { mean, sd, logMarginalLikelihood: -0.5 * (n * Math.log(2 * Math.PI) + logDet + quad) }
}

/** The hyperparameters of a Gaussian process regression on inputs of any dimension: an ARD squared-exponential kernel, its variance and the noise. */
export interface GpHyperparameters {
  /** The length scale: one number for every input, or one per input (an ARD kernel). */
  readonly lengthScale: number | readonly number[]
  /** The kernel variance (the amplitude squared). */
  readonly variance: number
  /** The sd of the observation noise. */
  readonly noise: number
}

/**
 * The posterior of the function values of a Gaussian process regression given every hyperparameter, in plain arithmetic: the
 * kernel `variance exp(-sum_k (x_k - x'_k)^2 / (2 ell_k^2))` (one length scale per input when `lengthScale` is an array, an ARD
 * kernel), a little jitter, the noise `sigma^2 I`; the textbook mean `K (K + sigma^2 I)^-1 y`, the marginal sds of
 * `K - K (K + sigma^2 I)^-1 K` and the log marginal likelihood `log N(y; 0, K + sigma^2 I)`. It reads nothing from the
 * library's kernels, so it is the reference of the grid over several hyperparameters (`epGridMethod`).
 *
 * @param x - The inputs, one row per point (`[[0], [1]]` for one input).
 * @param y - The observations.
 * @param h - The length scale (or one per input), the kernel variance and the noise sd.
 * @returns The mean and sd of each function value and the log marginal likelihood.
 * @concept gaussian-processes
 * @example
 * ```ts
 * import { gpPosteriorGivenHyperparameters } from 'omai-engine/probprog/models'
 * const p = gpPosteriorGivenHyperparameters([[0], [1], [2]], [0, 1, 0], { lengthScale: 1, variance: 2, noise: 0.3 })
 * [p.mean.length, p.sd.every((s) => s > 0 && s < 1.5)] // => [3, true]
 * ```
 */
export function gpPosteriorGivenHyperparameters(
  x: readonly (readonly number[])[],
  y: readonly number[],
  h: GpHyperparameters,
): { readonly mean: number[]; readonly sd: number[]; readonly logMarginalLikelihood: number } {
  const n = x.length
  const k = x[0]?.length ?? 0
  const scales = typeof h.lengthScale === 'number' ? toFlat(full([k], h.lengthScale)) : h.lengthScale
  const K = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let q = 0
      for (let c = 0; c < k; c++) q += (((x[i]?.[c] ?? 0) - (x[j]?.[c] ?? 0)) / (scales[c] ?? 1)) ** 2
      K[i * n + j] = h.variance * Math.exp(-0.5 * q) + (i === j ? JITTER : 0)
    }
  const C = Float64Array.from(K)
  for (let i = 0; i < n; i++) C[i * n + i] = (C[i * n + i] ?? 0) + h.noise * h.noise
  const { factor } = dpotrf(C, n)
  const alpha = dpotrs(factor, y, n)
  const mean = Array.from(dgemv(K, n, n, alpha))
  const sd = Array.from({ length: n }, (_, i) => {
    const row = K.slice(i * n, (i + 1) * n)
    const solved = dpotrs(factor, row, n)
    let reduction = 0
    for (let j = 0; j < n; j++) reduction += (row[j] ?? 0) * (solved[j] ?? 0)
    return Math.sqrt(Math.max((K[i * n + i] ?? 0) - reduction, 1e-14))
  })
  let logDet = 0
  for (let i = 0; i < n; i++) logDet += 2 * Math.log(factor[i * n + i] ?? 1)
  const quad = y.reduce((a, v, i) => a + v * (alpha[i] ?? 0), 0)
  return { mean, sd, logMarginalLikelihood: -0.5 * (n * Math.log(2 * Math.PI) + logDet + quad) }
}

function kernelMatrixOf(x: readonly number[], ell: number): Value {
  return kernelMatrix(tensor(x), tensor(ell))
}

const cache = new WeakMap<BenchmarkData, Map<string, BenchmarkTruth>>()

/** The truth: quadrature over the length scale, the function values as the grid mixture of their conditional normals. */
function gpTruth(data: BenchmarkData, quality: 'full' | 'fast'): BenchmarkTruth {
  const hit = cache.get(data)?.get(quality)
  if (hit !== undefined) return hit
  const x = data.data['x'] as number[]
  const y = data.data['y'] as number[]
  // The marginal model of the length scale: the function integrated out, y ~ N(0, K(ell) + sigma² I).
  const inputs = tensor(x)
  const observations = tensor(y)
  const priorMean = zeros([x.length])
  const noise = mul(eye(x.length), NOISE * NOISE)
  const marginal = (): void => {
    const ell = sample('ell', logNormalDist(0, 0.5))
    observe('y', multivariateNormal(priorMean, { covariance: add(kernelMatrix(inputs, ell), noise) }), observations)
  }
  const grid = gridPosterior(
    marginal,
    undefined,
    quality === 'fast' ? { points: 61, maxRefinements: 0 } : { points: 161 },
  )
  const at = new Map<number, ReturnType<typeof gpPosteriorGivenLengthScale>>()
  const given = (u: number): ReturnType<typeof gpPosteriorGivenLengthScale> => {
    let p = at.get(u)
    if (p === undefined) {
      p = gpPosteriorGivenLengthScale(x, y, Math.exp(u))
      at.set(u, p)
    }
    return p
  }
  const marginals: Record<string, MarginalOracle> = { ell: grid.marginal('ell') }
  x.forEach((_, i) => {
    marginals[`f[${i}]`] = conditionalOracle(
      `f[${i}]`,
      grid,
      (u) => {
        const p = given(u[0] ?? 0)
        return { mean: p.mean[i] as number, sd: p.sd[i] as number }
      },
      { points: quality === 'fast' ? 120 : 300 },
    )
  })
  const truth: BenchmarkTruth = {
    kind: 'quadrature',
    source: `quadrature over the length scale (${grid.points} points, evidence error ${grid.evidenceError.toExponential(1)}) of the marginal likelihood N(y; 0, K + sigma² I) with the function integrated out; each function value is the grid mixture of its exact Gaussian conditionals (mean K (K + sigma² I)^-1 y)`,
    logEvidence: grid.logEvidence,
    marginals,
  }
  const byQuality = cache.get(data) ?? new Map<string, BenchmarkTruth>()
  byQuality.set(quality, truth)
  cache.set(data, byQuality)
  return truth
}

/**
 * Gaussian process regression with an unknown length scale: twelve noisy samples of `sin(1.2 x) + 0.2 x` on `[-3, 3]`, a
 * squared-exponential kernel (amplitude 1, noise 0.3 known), the function vector as a latent with a multivariate normal
 * prior. Given the length scale the function posterior is Gaussian in closed form; the length scale's posterior is one
 * dimensional, so the truth is exact to the quadrature. NUTS samples the function and the length scale together, with the
 * covariance a `custom` function over the library's Gram matrix; the Laplace fit and the mean-field guide are the cheap
 * approximations, the latter too narrow where the function values are strongly correlated.
 *
 * @concept gaussian-processes
 * @index dataset Gaussian process regression with an unknown length scale; closed-form function posterior, quadrature over the length scale.
 * @example
 * ```ts
 * import { gaussianProcessRegression } from 'omai-engine/probprog/models'
 * [gaussianProcessRegression.bound().sizes, gaussianProcessRegression.truth(undefined, 'fast').kind] // => [{ point: 12 }, 'quadrature']
 * ```
 */
export const gaussianProcessRegression: BenchmarkModel = defineBenchmark({
  name: 'gaussian-process-regression',
  title: 'Gaussian process regression',
  description:
    'A latent function with a squared-exponential kernel of unknown length scale observed with known noise: the function posterior is Gaussian given the length scale, so the truth is a one-dimensional quadrature mixture.',
  tags: ['gaussian-process', 'kernel', 'regression', 'multivariate-normal', 'linear-gaussian', 'custom', 'quadrature'],
  references: [
    'Rasmussen and Williams (2006), Gaussian Processes for Machine Learning, chapters 2 and 5',
    'Williams and Rasmussen (1996), Gaussian processes for regression',
  ],
  definition: gpSpec,
  generate: (key) => generateGp(key),
  truth: (data, quality) => gpTruth(data, quality),
  plans: (scale) => [
    nutsPlan(scale, {
      warmup: 200,
      draws: 200,
      expect: 'differ',
      note: 'the centred form has a funnel between the length scale and the function: divergences, and the length scale is explored poorly (see the non-centred variant)',
    }),
    temperedPlan(scale, {
      expect: 'report',
      note: 'tempered particles in thirteen dimensions: consistent, with a noisy tail',
    }),
    laplacePlan(scale, {
      note: 'the joint mode of the function and the length scale of a hierarchy is degenerate (the density grows without bound as the function follows the data): no posterior',
    }),
    sviPlan(scale, {
      note: 'a mean-field normal ignores the correlation between neighbouring function values: spreads too small',
    }),
  ],
})

/** The non-centred spec: `f = L(ell) z` with `z` standard normal, so the sampler works in a space whose geometry does not depend on the length scale. */
const nonCentredSpec = defineModel('gaussian-process-non-centred', (b) => {
  const x = b.data('x', { dims: ['point'], role: 'covariate' })
  const y = b.data('y', { dims: ['point'], role: 'response' })
  const ell = b.latent('ell', d.logNormal(0, 0.5), { doc: 'the length scale of the squared-exponential kernel' })
  const z = b.latent('z', d.normal(0, 1), { dims: ['point'], doc: 'the whitened function: f = L z' })
  const f = b.derived(
    'f',
    b.custom(
      'rbfCholeskyApply',
      (args) => {
        const [xs, scale, white] = args
        const L = cholesky(kernelMatrix((xs as { value: Value }).value, (scale as { value: Value }).value))
        return matmul(L, (white as { value: Value }).value)
      },
      [x, ell, z],
      { dims: ['point'] },
    ),
    { doc: 'the function at the inputs' },
  )
  b.observe('y', d.normal(f, NOISE), y)
})

/** The function values of the draws of the whitened model, `f = L(ell) z`, for the canonical inputs (a method whose posterior holds `z` and `ell` and not the derived `f`). */
function functionDraws(draws: Readonly<Record<string, ArrayLike<number>>>): Record<string, ArrayLike<number>> {
  const ell = draws['ell']
  if (ell === undefined) return {}
  const x = tensor(generateGp(makeKey(0)).data['x'] as number[])
  const out = Array.from({ length: POINTS }, () => new Float64Array(ell.length))
  for (let i = 0; i < ell.length; i++) {
    const L = toFlat(cholesky(kernelMatrix(x, tensor(ell[i] as number))))
    for (let r = 0; r < POINTS; r++) {
      let v = 0
      for (let c = 0; c <= r; c++)
        v += (L[r * POINTS + c] as number) * ((draws[`z[${c}]`] as ArrayLike<number>)[i] as number)
      ;(out[r] as Float64Array)[i] = v
    }
  }
  return Object.fromEntries(out.map((v, r) => [`f[${r}]`, v]))
}

/**
 * Gaussian process regression, non-centred: the same model as `gaussianProcessRegression` written as `f = L(ell) z` with
 * `z ~ N(0, I)`, the standard cure for the funnel between the length scale and the function. The truth is the same, and NUTS
 * now agrees with it where the centred form diverges; the Laplace fit and the guide, which hold `z` and `ell`, are compared
 * through the function values computed from their draws (for the canonical inputs).
 *
 * @concept gaussian-processes
 * @index dataset Gaussian process regression, non-centred (f = L z); the same quadrature truth as the centred form.
 * @example
 * ```ts
 * import { gaussianProcessNonCentred } from 'omai-engine/probprog/models'
 * gaussianProcessNonCentred.bound().sizes // => { point: 12 }
 * ```
 */
export const gaussianProcessNonCentred: BenchmarkModel = defineBenchmark({
  name: 'gaussian-process-non-centred',
  title: 'Gaussian process regression, non-centred',
  description:
    'The Gaussian process regression written as f = L z with a whitened z: the geometry the sampler sees no longer depends on the length scale, and NUTS agrees with the quadrature truth.',
  tags: ['gaussian-process', 'kernel', 'regression', 'non-centred', 'custom', 'quadrature'],
  references: [
    'Rasmussen and Williams (2006), Gaussian Processes for Machine Learning, chapters 2 and 5',
    'Betancourt and Girolami (2015), Hamiltonian Monte Carlo for hierarchical models',
  ],
  definition: nonCentredSpec,
  generate: (key) => generateGp(key),
  summarise: functionDraws,
  truth: (data, quality) => gpTruth(data, quality),
  plans: (scale) => [
    nutsPlan(scale, {
      warmup: 200,
      draws: 200,
      note: 'whitened, so no funnel: agrees within Monte Carlo error',
      tolerances: { meanError: 0.2, meanZ: 6, sdRatio: [0.8, 1.2], ks: 0.2 },
    }),
    temperedPlan(scale, {
      expect: 'report',
      note: 'tempered particles in thirteen dimensions: consistent, with a noisy tail',
    }),
    laplacePlan(scale, {
      note: 'a Gaussian at the mode of the whitened density: close for the function, the length scale is skewed on the log scale',
    }),
    sviPlan(scale, {
      note: 'a mean-field normal on the whitened coordinates: the posterior of z is nearly independent, so the spreads are close',
    }),
  ],
})
