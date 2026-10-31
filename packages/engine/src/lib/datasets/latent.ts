/**
 * Latent-factor generators with their factors as truth: independent non-Gaussian sources mixed linearly (for
 * independent component analysis), two views of a shared latent signal (for canonical correlation and partial least
 * squares), sparse features with importances (the toy model of superposition), and sparse non-negative codes written
 * into a lower-dimensional space through a dictionary of unit directions (what a sparse autoencoder must recover).
 */

import { type DatasetGenerator, type RegressionData, xy } from '../data/index.ts'
import { inv, sqrtm, svdvals } from '../numerics/linalg/index.ts'
import { foldIn, laplace, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { add, eye, matmul, mul, toFlat, transpose, type Value } from '../tensor/index.ts'
import { matrix, stamp } from './_util.ts'

/** The source signals of {@link mixedSources}. */
export type SourceKind = 'laplace' | 'uniform' | 'sawtooth' | 'sine'

/**
 * Independent sources mixed linearly: `x = A s (+ noise)`, with unit-variance non-Gaussian sources (Laplace, uniform,
 * sawtooth or sine waves of different frequencies) and a random mixing matrix. Features are the mixtures and the
 * targets the sources, so a recovery can be scored; the truth has the sources, the mixing matrix and its inverse, and
 * the sources' kind (non-Gaussian sources are what make the mixing identifiable, up to order, sign and scale).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { mixedSources } from 'mdml-engine/datasets'
 * generate(mixedSources, { knobs: { n: 20, sources: 3 } }).data.x.shape // => [20, 3]
 * ```
 */
export const mixedSources: DatasetGenerator<RegressionData> = {
  name: 'mixedSources',
  kind: 'regression',
  description: 'Independent non-Gaussian sources mixed by a random matrix, the sources and the mixing as truth.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 1000 }),
    sources: int(2, 50, {
      label: 'sources',
      description: 'The number of independent sources (and of mixtures).',
      default: 3,
    }),
    kind: choice(['laplace', 'uniform', 'sawtooth', 'sine'], {
      label: 'source kind',
      description: 'The distribution or waveform of the sources.',
      default: 'laplace',
    }),
    noise: real(0, 5, {
      label: 'noise standard deviation',
      description: 'Gaussian noise added to the mixtures.',
      default: 0,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['sources'] as number
    const kind = knobs['kind'] as SourceKind
    const s = new Float64Array(n * m)
    for (let j = 0; j < m; j++) {
      const kj = foldIn(foldIn(k, 0), j)
      const column =
        kind === 'laplace'
          ? Array.from(laplace(kj, [n], 0, 1 / Math.SQRT2).data)
          : kind === 'uniform'
            ? Array.from(uniform(kj, [n], { low: -Math.sqrt(3), high: Math.sqrt(3) }).data)
            : kind === 'sawtooth'
              ? Array.from({ length: n }, (_, i) => Math.sqrt(3) * (2 * (((i * (j + 2)) / 29) % 1) - 1))
              : Array.from({ length: n }, (_, i) => Math.SQRT2 * Math.sin((2 * Math.PI * i * (j + 1.37)) / 41))
      for (let i = 0; i < n; i++) s[i * m + j] = column[i] ?? 0
    }
    // A mixing matrix with well separated singular values: random draws until the condition number is moderate.
    let a = normal(foldIn(k, 1), [m, m])
    for (let attempt = 0; attempt < 50; attempt++) {
      const sv = toFlat(svdvals(a))
      if ((sv[0] ?? 1) / Math.max(sv[sv.length - 1] ?? 1e-12, 1e-12) < 10) break
      a = normal(foldIn(foldIn(k, 2), attempt), [m, m])
    }
    const mix = matmul(matrix(s, n, m), transpose(a))
    const sd = knobs['noise'] as number
    const noise = normal(foldIn(k, 3), [n, m]).data
    const x = Float64Array.from(toFlat(mix), (v, i) => v + sd * (noise[i] ?? 0))
    return {
      data: xy(matrix(x, n, m), matrix(s, n, m), {
        ...stamp('mixedSources', mixedSources.description, knobs),
        featureNames: Array.from({ length: m }, (_, i) => `mixture${i}`),
        targetNames: Array.from({ length: m }, (_, i) => `source${i}`),
      }),
      truth: {
        noiseScale: sd,
        latents: { sources: matrix(s, n, m) },
        extra: { mixing: a, kind },
      },
    }
  },
}

/**
 * Two views of shared latent signals: `z ~ N(0, I)` with `latent` dimensions, a view of `viewA` features
 * `A z + private + noise` and a view of `viewB` features `B z + private + noise`, the private parts independent between
 * the views. Features are the first view and targets the second. The truth has the loadings and the population
 * canonical correlations (the singular values of the whitened cross-covariance), against which CCA's estimates and PLS's
 * covariance directions can be compared.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { twoViews } from 'mdml-engine/datasets'
 * generate(twoViews, { knobs: { n: 10, latent: 1, viewA: 3, viewB: 2 } }).data.y.shape // => [10, 2]
 * ```
 */
export const twoViews: DatasetGenerator<RegressionData> = {
  name: 'twoViews',
  kind: 'regression',
  description: 'Two views driven by shared latent signals plus private noise, with the canonical correlations known.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 1000 }),
    latent: int(1, 20, { label: 'shared factors', description: 'The number of shared latent signals.', default: 2 }),
    viewA: int(1, 200, {
      label: 'features in the first view',
      description: 'The dimension of the first view.',
      default: 5,
    }),
    viewB: int(1, 200, {
      label: 'features in the second view',
      description: 'The dimension of the second view.',
      default: 4,
    }),
    noise: real(0.01, 10, {
      label: 'private noise',
      description: 'The standard deviation of each view’s private noise.',
      default: 1,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const r = knobs['latent'] as number
    const da = knobs['viewA'] as number
    const db = knobs['viewB'] as number
    const sd = knobs['noise'] as number
    const z = normal(foldIn(k, 0), [n, r])
    const a = normal(foldIn(k, 1), [da, r])
    const b = normal(foldIn(k, 2), [db, r])
    const na = normal(foldIn(k, 3), [n, da]).data
    const nb = normal(foldIn(k, 4), [n, db]).data
    const xa = Float64Array.from(toFlat(matmul(z, transpose(a))), (v, i) => v + sd * (na[i] ?? 0))
    const xb = Float64Array.from(toFlat(matmul(z, transpose(b))), (v, i) => v + sd * (nb[i] ?? 0))
    // Population covariances: Saa = A A^T + sd^2 I, Sbb = B B^T + sd^2 I, Sab = A B^T.
    const covA = add(matmul(a, transpose(a)), mul(eye(da), sd * sd))
    const covB = add(matmul(b, transpose(b)), mul(eye(db), sd * sd))
    const white = matmul(matmul(inv(sqrtm(covA)), matmul(a, transpose(b))), inv(sqrtm(covB)))
    const correlations = toFlat(svdvals(white)).slice(0, Math.min(r, da, db))
    return {
      data: xy(matrix(xa, n, da), matrix(xb, n, db), {
        ...stamp('twoViews', twoViews.description, knobs),
        featureNames: Array.from({ length: da }, (_, i) => `a${i}`),
        targetNames: Array.from({ length: db }, (_, i) => `b${i}`),
      }),
      truth: {
        latents: { shared: z },
        extra: { loadingsA: a, loadingsB: b, canonicalCorrelations: correlations, noise: sd },
      },
    }
  },
}

/**
 * The toy model of superposition: `features` independent sparse features, each zero with probability `sparsity` and
 * otherwise uniform on [0, 1], with importances that decay geometrically (feature `i` has weight `decay^i`). The
 * dataset is an autoencoding problem: features are the vectors and targets the same vectors. The truth has the
 * importances, the feature probabilities, and the importance-weighted loss of a given reconstruction.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { toySuperposition } from 'mdml-engine/datasets'
 * (generate(toySuperposition, { knobs: { features: 4, decay: 0.5 } }).truth.extra?.['importance'] as number[]).slice(0, 3) // => [1, 0.5, 0.25]
 * ```
 */
export const toySuperposition: DatasetGenerator<RegressionData> = {
  name: 'toySuperposition',
  kind: 'regression',
  description: 'Sparse features with importances for autoencoders squeezing more features than dimensions.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of vectors.', default: 2000 }),
    features: int(2, 1000, { label: 'features', description: 'The number of features.', default: 20 }),
    sparsity: real(0, 1, {
      label: 'sparsity',
      description: 'The chance that a feature is zero in a vector.',
      default: 0.9,
    }),
    decay: real(0.01, 1, { label: 'importance decay', description: 'Feature i has importance decay^i.', default: 0.9 }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['features'] as number
    const sparsity = knobs['sparsity'] as number
    const importance = Array.from({ length: m }, (_, i) => (knobs['decay'] as number) ** i)
    const u = uniform(foldIn(k, 0), [n, m]).data
    const v = uniform(foldIn(k, 1), [n, m]).data
    const x = Float64Array.from(u, (a, i) => (a < sparsity ? 0 : (v[i] ?? 0)))
    const weighted = (reconstruction: ArrayLike<number>): number => {
      let total = 0
      for (let i = 0; i < n * m; i++) total += (importance[i % m] ?? 0) * ((x[i] ?? 0) - (reconstruction[i] ?? 0)) ** 2
      return total / n
    }
    return {
      data: xy(matrix(x, n, m), matrix(x, n, m), { ...stamp('toySuperposition', toySuperposition.description, knobs) }),
      truth: {
        extra: {
          importance,
          sparsity,
          activeProbability: 1 - sparsity,
          weightedLoss: weighted,
          meanSquare: weighted(new Float64Array(n * m)),
        },
      },
    }
  },
}

/**
 * Sparse codes through a dictionary: `features` unit-norm directions in `dim` dimensions (rows of the dictionary), and
 * vectors `x = sum_j c_j d_j (+ noise)` whose codes are zero with probability `sparsity` and otherwise uniform on
 * [0.5, 1.5]. This is the setting sparse autoencoders and dictionary learning are asked to invert. Features are the
 * vectors and targets the codes; the truth has the codes (latent `codes`), the dictionary (rows, as `dictionary`), the
 * sparsity and the largest cosine between two dictionary directions (how close to orthogonal the plant is).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { sparseDictionary } from 'mdml-engine/datasets'
 * generate(sparseDictionary, { knobs: { n: 5, features: 3, dim: 4 } }).data.x.shape // => [5, 4]
 * ```
 */
export const sparseDictionary: DatasetGenerator<RegressionData> = {
  name: 'sparseDictionary',
  kind: 'regression',
  description: 'Sparse non-negative codes written through a dictionary of unit directions, with the dictionary known.',
  knobs: dict({
    n: int(1, 10_000_000, { label: 'samples', description: 'The number of vectors.', default: 2000 }),
    features: int(1, 1000, { label: 'features', description: 'The number of dictionary directions.', default: 6 }),
    dim: int(1, 1000, { label: 'dimension', description: 'The dimension the directions live in.', default: 8 }),
    sparsity: real(0, 1, {
      label: 'sparsity',
      description: 'The chance that a code is zero in a vector.',
      default: 0.85,
    }),
    noise: real(0, 10, { label: 'noise', description: 'The standard deviation of the added noise.', default: 0 }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const m = knobs['features'] as number
    const d = knobs['dim'] as number
    const sparsity = knobs['sparsity'] as number
    const sd = knobs['noise'] as number
    const raw = normal(foldIn(k, 0), [m, d]).data
    const dictionary = Array.from({ length: m }, (_, j) => {
      const row = Array.from({ length: d }, (_, q) => raw[j * d + q] ?? 0)
      const norm = Math.hypot(...row) || 1
      return row.map((v) => v / norm)
    })
    const u = uniform(foldIn(k, 1), [n, m]).data
    const v = uniform(foldIn(k, 2), [n, m]).data
    const eps = normal(foldIn(k, 3), [n, d]).data
    const codes = Float64Array.from(u, (a, i) => (a < sparsity ? 0 : 0.5 + (v[i] ?? 0)))
    const x = new Float64Array(n * d)
    for (let i = 0; i < n; i++)
      for (let q = 0; q < d; q++) {
        let acc = sd * (eps[i * d + q] ?? 0)
        for (let j = 0; j < m; j++) acc += (codes[i * m + j] ?? 0) * (dictionary[j]?.[q] ?? 0)
        x[i * d + q] = acc
      }
    let coherence = 0
    for (let a = 0; a < m; a++)
      for (let b = a + 1; b < m; b++) {
        let dot = 0
        for (let q = 0; q < d; q++) dot += (dictionary[a]?.[q] ?? 0) * (dictionary[b]?.[q] ?? 0)
        coherence = Math.max(coherence, Math.abs(dot))
      }
    return {
      data: xy(matrix(x, n, d), matrix(codes, n, m), {
        ...stamp('sparseDictionary', sparseDictionary.description, knobs),
      }),
      truth: { latents: { codes: matrix(codes, n, m) }, extra: { dictionary, sparsity, coherence } },
    }
  },
}
