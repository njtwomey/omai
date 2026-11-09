/**
 * Generators of Gaussian data whose population covariance is known, the test bed for covariance, precision and
 * correlation estimators: a factor model (loadings and uniquenesses), a spiked covariance (a few strong directions in
 * isotropic noise, with the Baik-Ben Arous-Peche threshold and the limiting sample eigenvalues), correlated blocks,
 * Toeplitz covariances (autoregressive, moving-average and long-memory) and a Wishart-distributed population covariance.
 * Every truth carries the covariance, the correlation, the precision and the eigenvalues, so an estimator's error can be
 * measured against what generated the data; the targets are the latent draws the rows came from.
 */

import { OmaiError } from '../base/index.ts'
import { bbpThreshold, spikedEigenvalueLimit, spikedEigenvectorOverlap } from '../covariance/index.ts'
import { type DatasetGenerator, type RegressionData, xy } from '../data/index.ts'
import { cholesky, dgemm, eigvalsh, haarOrthogonal, inv, pinv } from '../numerics/linalg/index.ts'
import { foldIn, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { matrix, stamp } from './_util.ts'

/** The truth every generator shares: covariance, correlation, precision, eigenvalues and condition number. */
function covarianceTruth(cov: Float64Array, p: number, pseudo = false): Record<string, unknown> {
  const c = matrix(cov, p, p)
  const sd = Array.from({ length: p }, (_, i) => Math.sqrt(cov[i * p + i] ?? 1))
  const corr = Float64Array.from(cov, (v, k) => v / ((sd[Math.floor(k / p)] ?? 1) * (sd[k % p] ?? 1)))
  const eigenvalues = Array.from(toFlat(eigvalsh(c))).reverse()
  const smallest = eigenvalues[p - 1] ?? 0
  return {
    covariance: c,
    correlation: matrix(corr, p, p),
    precision: pseudo ? pinv(c) : inv(c),
    eigenvalues,
    conditionNumber:
      smallest > 1e-12 * (eigenvalues[0] ?? 1) ? (eigenvalues[0] ?? 1) / smallest : Number.POSITIVE_INFINITY,
  }
}

/** Rows `z A^T` for white `z` of `r` columns and `A` `[p, r]`. */
function mixRows(z: ArrayLike<number>, a: ArrayLike<number>, n: number, r: number, p: number): Float64Array {
  return dgemm(z, a, n, r, p, { transposeB: true })
}

/** `A diag(w) A^T` for `A` `[p, r]`. */
function gram(a: ArrayLike<number>, w: ArrayLike<number>, p: number, r: number): Float64Array {
  const scaled = Float64Array.from({ length: p * r }, (_, k) => (a[k] ?? 0) * (w[k % r] ?? 0))
  return dgemm(scaled, a, p, r, p, { transposeB: true })
}

const names = (p: number): string[] => Array.from({ length: p }, (_, i) => `x${i}`)

/**
 * Data from a factor model `x = Lambda f + e`: `factors` latent factors `f ~ N(0, I)` with Gaussian loadings and
 * per-variable uniquenesses (the variances of `e`), so the covariance is `Lambda Lambda^T + Psi`. The targets are the
 * factor scores; the truth has the loadings, the uniquenesses and the covariance, correlation, precision and eigenvalues.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { factorCovarianceData } from 'omai-engine/datasets'
 * generate(factorCovarianceData, { knobs: { n: 20, p: 6, factors: 2 } }).data.x.shape // => [20, 6]
 * ```
 */
export const factorCovarianceData: DatasetGenerator<RegressionData> = {
  name: 'factorCovarianceData',
  kind: 'regression',
  description: 'Gaussian data from a factor model, with the loadings, uniquenesses and covariance as truth.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 500 }),
    p: int(2, 2000, { label: 'variables', description: 'The number of variables.', default: 20 }),
    factors: int(1, 100, { label: 'factors', description: 'The number of latent factors.', default: 3 }),
    strength: real(0.1, 10, {
      label: 'loading scale',
      description: 'The standard deviation of the loadings (the factors’ strength against the noise).',
      default: 1,
    }),
    uniqueness: real(0.01, 10, {
      label: 'uniqueness',
      description: 'The mean variance of each variable’s own noise (each is drawn within a factor 3 of it).',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['p'] as number
    const r = knobs['factors'] as number
    const strength = knobs['strength'] as number
    const unique = knobs['uniqueness'] as number
    if (r > p) throw new OmaiError('value', `factorCovarianceData: ${r} factors for ${p} variables`)
    const loadings = Float64Array.from(normal(foldIn(k, 0), [p, r]).data, (v) => v * strength)
    const u = uniform(foldIn(k, 1), [p]).data
    const psi = Float64Array.from(u, (v) => unique * 3 ** (2 * v - 1))
    const f = normal(foldIn(k, 2), [n, r]).data
    const e = normal(foldIn(k, 3), [n, p]).data
    const x = mixRows(f, loadings, n, r, p)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < p; j++) x[i * p + j] = (x[i * p + j] ?? 0) + Math.sqrt(psi[j] ?? 1) * (e[i * p + j] ?? 0)
    const cov = gram(loadings, new Float64Array(r).fill(1), p, r)
    for (let j = 0; j < p; j++) cov[j * p + j] = (cov[j * p + j] ?? 0) + (psi[j] ?? 0)
    return {
      data: xy(matrix(x, n, p), matrix(f, n, r), {
        ...stamp('factorCovarianceData', factorCovarianceData.description, knobs),
        featureNames: names(p),
        targetNames: Array.from({ length: r }, (_, i) => `factor${i}`),
      }),
      truth: {
        latents: { factors: matrix(f, n, r) },
        extra: {
          ...covarianceTruth(cov, p),
          loadings: matrix(loadings, p, r),
          uniquenesses: Array.from(psi),
          aspectRatio: p / n,
        },
      },
    }
  },
}

/**
 * Data from a spiked covariance: isotropic noise of variance `variance` plus `spikes` strong directions (Haar-random
 * orthonormal), the population eigenvalues `variance (1 + (strength - 1) spread^k)` for `k = 0, 1, ...`. The targets are
 * the spike scores. The truth has the spike eigenvalues and directions, the Baik-Ben Arous-Peche threshold of the aspect
 * ratio `p / n`, which spikes lie above it, and the limiting sample eigenvalue and eigenvector overlap of each.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { spikedCovarianceData } from 'omai-engine/datasets'
 * generate(spikedCovarianceData, { knobs: { n: 50, p: 10, spikes: 2 } }).truth.extra?.['spikes'] // => [6, 4]
 * ```
 */
export const spikedCovarianceData: DatasetGenerator<RegressionData> = {
  name: 'spikedCovarianceData',
  kind: 'regression',
  description: 'Gaussian data from a spiked covariance, with the spikes and the phase-transition limits as truth.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 500 }),
    p: int(2, 2000, { label: 'variables', description: 'The number of variables.', default: 100 }),
    spikes: int(0, 50, { label: 'spikes', description: 'The number of strong directions.', default: 3 }),
    strength: real(1, 1000, {
      label: 'strength',
      description:
        'The largest spike as a multiple of the noise variance (above 1 + sqrt(p / n) it separates from the bulk).',
      default: 6,
    }),
    spread: real(0.05, 1, {
      label: 'spread',
      description: 'Each further spike’s excess over the noise as a fraction of the previous one’s.',
      default: 0.6,
    }),
    variance: real(0.01, 100, {
      label: 'noise variance',
      description: 'The variance of the isotropic noise.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['p'] as number
    const m = knobs['spikes'] as number
    const strength = knobs['strength'] as number
    const spread = knobs['spread'] as number
    const variance = knobs['variance'] as number
    if (m > p) throw new OmaiError('value', `spikedCovarianceData: ${m} spikes for ${p} variables`)
    const spikes = Array.from({ length: m }, (_, i) => variance * (1 + (strength - 1) * spread ** i))
    const ratio = p / n
    const e = normal(foldIn(k, 0), [n, p]).data
    const x = Float64Array.from(e, (v) => Math.sqrt(variance) * v)
    let scores = new Float64Array(0)
    let directions = new Float64Array(0)
    const cov = new Float64Array(p * p)
    for (let j = 0; j < p; j++) cov[j * p + j] = variance
    if (m > 0) {
      directions = Float64Array.from(haarOrthogonal(foldIn(k, 1), p, m).data as Float64Array)
      scores = Float64Array.from(normal(foldIn(k, 2), [n, m]).data)
      const amplitude = Float64Array.from(spikes, (l) => Math.sqrt(l - variance))
      const scaled = Float64Array.from(scores, (v, c) => v * (amplitude[c % m] ?? 0))
      const add = mixRows(scaled, directions, n, m, p)
      for (let c = 0; c < x.length; c++) x[c] = (x[c] ?? 0) + (add[c] ?? 0)
      const spikePart = gram(
        directions,
        Float64Array.from(spikes, (l) => l - variance),
        p,
        m,
      )
      for (let c = 0; c < cov.length; c++) cov[c] = (cov[c] ?? 0) + (spikePart[c] ?? 0)
    }
    return {
      data: xy(matrix(x, n, p), matrix(scores, n, m), {
        ...stamp('spikedCovarianceData', spikedCovarianceData.description, knobs),
        featureNames: names(p),
        targetNames: Array.from({ length: m }, (_, i) => `spike${i}`),
      }),
      truth: {
        latents: { scores: matrix(scores, n, m) },
        extra: {
          ...covarianceTruth(cov, p),
          spikes,
          directions: matrix(directions, p, m),
          noiseVariance: variance,
          aspectRatio: ratio,
          bbpThreshold: bbpThreshold(ratio, variance),
          detectable: spikes.map((l) => l > bbpThreshold(ratio, variance)),
          limitingEigenvalues: spikes.map((l) => spikedEigenvalueLimit(l, ratio, variance)),
          limitingOverlaps: spikes.map((l) => spikedEigenvectorOverlap(l, ratio, variance)),
        },
      },
    }
  },
}

/**
 * Data from correlated blocks of variables with unit variances: the variables are split into `blocks` near-equal groups,
 * correlated `within` inside a group and `between` across groups (a global factor of loading `sqrt(between)`, a factor
 * per block of loading `sqrt(within - between)`, and noise). The targets are the block factors; the truth has the block
 * of each variable and the covariance (equal to the correlation), precision and eigenvalues.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { blockCovarianceData } from 'omai-engine/datasets'
 * generate(blockCovarianceData, { knobs: { n: 20, p: 8, blocks: 2 } }).truth.extra?.['blockOf'] // => [0, 0, 0, 0, 1, 1, 1, 1]
 * ```
 */
export const blockCovarianceData: DatasetGenerator<RegressionData> = {
  name: 'blockCovarianceData',
  kind: 'regression',
  description: 'Gaussian data from correlated blocks of variables, with the block structure and covariance as truth.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 500 }),
    p: int(2, 2000, { label: 'variables', description: 'The number of variables.', default: 24 }),
    blocks: int(1, 200, { label: 'blocks', description: 'The number of correlated groups of variables.', default: 4 }),
    within: real(0, 0.99, {
      label: 'within-block correlation',
      description: 'The correlation inside a block.',
      default: 0.6,
    }),
    between: real(0, 0.99, {
      label: 'between-block correlation',
      description: 'The correlation across blocks (at most the within-block one).',
      default: 0.1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['p'] as number
    const b = knobs['blocks'] as number
    const within = knobs['within'] as number
    const between = knobs['between'] as number
    if (b > p) throw new OmaiError('value', `blockCovarianceData: ${b} blocks for ${p} variables`)
    if (between > within) throw new OmaiError('value', 'blockCovarianceData: between must not exceed within')
    const blockOf: number[] = []
    for (let g = 0; g < b; g++) {
      const size = Math.floor(p / b) + (g < p % b ? 1 : 0)
      for (let c = 0; c < size; c++) blockOf.push(g)
    }
    const global = normal(foldIn(k, 0), [n]).data
    const f = normal(foldIn(k, 1), [n, b]).data
    const e = normal(foldIn(k, 2), [n, p]).data
    const x = new Float64Array(n * p)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < p; j++)
        x[i * p + j] =
          Math.sqrt(between) * (global[i] ?? 0) +
          Math.sqrt(within - between) * (f[i * b + (blockOf[j] ?? 0)] ?? 0) +
          Math.sqrt(1 - within) * (e[i * p + j] ?? 0)
    const cov = new Float64Array(p * p)
    for (let i = 0; i < p; i++)
      for (let j = 0; j < p; j++) cov[i * p + j] = i === j ? 1 : blockOf[i] === blockOf[j] ? within : between
    return {
      data: xy(matrix(x, n, p), matrix(f, n, b), {
        ...stamp('blockCovarianceData', blockCovarianceData.description, knobs),
        featureNames: names(p),
        targetNames: Array.from({ length: b }, (_, i) => `block${i}`),
      }),
      truth: {
        latents: { blockFactors: matrix(f, n, b), global: Array.from(global) },
        extra: { ...covarianceTruth(cov, p), blockOf, aspectRatio: p / n },
      },
    }
  },
}

/** The autocovariance profile of {@link toeplitzCovarianceData}. */
export type ToeplitzProfile = 'ar1' | 'moving-average' | 'long-memory'

/**
 * Data whose covariance is a Toeplitz matrix of unit variance, `Sigma_ij = r(|i - j|)`: the autocorrelation of an
 * autoregressive process of order one (`rho^k`), of a moving average of order one with coefficient `rho` (only lag one,
 * `rho / (1 + rho^2)`), or the fractional Gaussian noise of a Hurst exponent (long memory, `r(k) = (|k+1|^2H - 2|k|^2H +
 * |k-1|^2H) / 2`). The targets are the white draws behind each row; the truth has the autocorrelation vector,
 * covariance, precision (tridiagonal for the autoregressive profile) and eigenvalues.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { toeplitzCovarianceData } from 'omai-engine/datasets'
 * generate(toeplitzCovarianceData, { knobs: { n: 20, p: 5, rho: 0.5 } }).truth.extra?.['autocorrelation'] // => [1, 0.5, 0.25, 0.125, 0.0625]
 * ```
 */
export const toeplitzCovarianceData: DatasetGenerator<RegressionData> = {
  name: 'toeplitzCovarianceData',
  kind: 'regression',
  description: 'Gaussian data with an autoregressive, moving-average or long-memory Toeplitz covariance as truth.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 500 }),
    p: int(2, 2000, {
      label: 'variables',
      description: 'The number of variables (the length of the process).',
      default: 30,
    }),
    profile: choice(['ar1', 'moving-average', 'long-memory'] as const, {
      label: 'profile',
      description: 'The autocorrelation: geometric, a single lag, or the power law of fractional Gaussian noise.',
      default: 'ar1',
    }),
    rho: real(-0.99, 0.99, {
      label: 'coefficient',
      description: 'The autoregressive coefficient, or the moving-average coefficient.',
      default: 0.6,
    }),
    hurst: real(0.5, 0.99, {
      label: 'Hurst exponent',
      description: 'The memory of the long-memory profile (0.5 is white noise).',
      default: 0.8,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['p'] as number
    const profile = knobs['profile'] as ToeplitzProfile
    const rho = knobs['rho'] as number
    const h = knobs['hurst'] as number
    const r = Array.from({ length: p }, (_, lag) => {
      if (profile === 'ar1') return rho ** lag
      if (profile === 'moving-average') return lag === 0 ? 1 : lag === 1 ? rho / (1 + rho * rho) : 0
      return lag === 0 ? 1 : 0.5 * ((lag + 1) ** (2 * h) - 2 * lag ** (2 * h) + (lag - 1) ** (2 * h))
    })
    const cov = new Float64Array(p * p)
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) cov[i * p + j] = r[Math.abs(i - j)] ?? 0
    const factor = toFlat(cholesky(matrix(cov, p, p)))
    const z = normal(foldIn(k, 0), [n, p]).data
    const x = mixRows(z, factor, n, p, p)
    return {
      data: xy(matrix(x, n, p), matrix(z, n, p), {
        ...stamp('toeplitzCovarianceData', toeplitzCovarianceData.description, knobs),
        featureNames: names(p),
        targetNames: Array.from({ length: p }, (_, i) => `z${i}`),
      }),
      truth: {
        latents: { white: matrix(z, n, p) },
        extra: { ...covarianceTruth(cov, p), autocorrelation: r, aspectRatio: p / n },
      },
    }
  },
}

/**
 * Data whose population covariance is itself a Wishart draw: `Sigma = G G^T / df` for a Gaussian `G` of `p x df`, the
 * covariance of a random cloud of `df` points, so its eigenvalues follow the Marchenko-Pastur law of the ratio `p / df`
 * (spread out, with a condition number that grows as `df` falls towards `p`). Rows are `N(0, Sigma)`; the targets are the
 * white draws behind them. The truth has `df`, `Sigma`, its correlation, pseudo-inverse and eigenvalues (the covariance is
 * singular when `df < p`).
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { wishartCovarianceData } from 'omai-engine/datasets'
 * generate(wishartCovarianceData, { knobs: { n: 20, p: 5, df: 15 } }).truth.extra?.['df'] // => 15
 * ```
 */
export const wishartCovarianceData: DatasetGenerator<RegressionData> = {
  name: 'wishartCovarianceData',
  kind: 'regression',
  description:
    'Gaussian data whose population covariance is a Wishart draw, with the covariance and spectrum as truth.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 500 }),
    p: int(2, 2000, { label: 'variables', description: 'The number of variables.', default: 20 }),
    df: int(1, 100_000, {
      label: 'degrees of freedom',
      description: 'The Wishart degrees of freedom (fewer than the variables make the covariance singular).',
      default: 40,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['p'] as number
    const df = knobs['df'] as number
    const g = Float64Array.from(normal(foldIn(k, 0), [p, df]).data, (v) => v / Math.sqrt(df))
    const cov = gram(g, new Float64Array(df).fill(1), p, df)
    const z = normal(foldIn(k, 1), [n, df]).data
    const x = mixRows(z, g, n, df, p)
    return {
      data: xy(matrix(x, n, p), matrix(z, n, df), {
        ...stamp('wishartCovarianceData', wishartCovarianceData.description, knobs),
        featureNames: names(p),
        targetNames: Array.from({ length: df }, (_, i) => `z${i}`),
      }),
      truth: {
        latents: { white: matrix(z, n, df) },
        extra: { ...covarianceTruth(cov, p, true), df, aspectRatio: p / n, populationRatio: p / df },
      },
    }
  },
}
