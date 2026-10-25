/**
 * Clustering datasets with known clusters: concentric rings, anisotropic and unequal-variance blobs (the cases of
 * scikit-learn's k-means assumptions example) and the Old Faithful geyser eruptions, a classic two-cluster real
 * dataset.
 */

import {
  type ClassificationData,
  type DatasetGenerator,
  type Generated,
  type Knobs,
  type Truth,
  take,
  xyClasses,
} from '../data/index.ts'
import { inv, slogdet } from '../numerics/linalg/index.ts'
import { foldIn, type Key, normal, permutation, uniform } from '../random/index.ts'
import { bool, dict, int, real } from '../space/index.ts'
import {
  add,
  item,
  matmul,
  mul,
  softmax,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { FAITHFUL_COLUMNS, FAITHFUL_SHA256 } from './_data/faithful.data.ts'
import { isotropicLogDensities, mixtureTruth, monteCarloBayesError, withLazyBayesError } from './_mixtures.ts'
import { evenSplit, labels, matrix, stamp } from './_util.ts'

/** Shuffle rows when asked, keeping the clusters aligned; the Bayes error is computed lazily. */
function clustered(
  name: string,
  description: string,
  knobs: Knobs,
  k: Key,
  x: Tensor,
  y: readonly number[],
  truth: Truth,
  bayes: () => number,
): Generated<ClassificationData> {
  const data = xyClasses(x, labels(y), stamp(name, description, knobs))
  const order = knobs['shuffle'] === false ? undefined : permutation(foldIn(k, 99), y.length)
  const rows = order === undefined ? data : take(data, order)
  const clusters = order === undefined ? labels(y) : labels(Array.from(order.data, (i) => y[i] ?? 0))
  return { data: rows, truth: withLazyBayesError({ ...truth, clusters }, bayes) }
}

const nKnob = int(2, 100_000, {
  description: 'The number of rows, split evenly between the rings.',
  default: 300,
  label: 'samples',
})
const shuffleKnob = bool({
  label: 'shuffle rows',
  description: 'Shuffle the rows; otherwise they come ring by ring.',
  default: true,
})

/**
 * Concentric rings: ring `k` has radius `k + 1`, angles uniform and radial Gaussian noise. The posterior depends only
 * on the distance from the centre, `p(k | x) ∝ π_k N(|x|; k + 1, σ²)` (exact for the folded radial model), so
 * centroid-based methods fail where spectral and density methods succeed.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { rings } from 'whitebox-engine/datasets'
 * generate(rings, { knobs: { n: 90, rings: 3 } }).data.classes // => ['0', '1', '2']
 * ```
 */
export const rings: DatasetGenerator<ClassificationData> = {
  name: 'rings',
  kind: 'classification',
  description: 'Concentric rings of radius 1, 2, ... with radial Gaussian noise.',
  knobs: dict({
    n: nKnob,
    rings: int(2, 20, { label: 'rings', description: 'The number of concentric rings (radii 1, 2, ...).', default: 2 }),
    noise: real(0.001, 2, {
      description: 'The standard deviation of the radial noise.',
      default: 0.1,
      label: 'radial noise',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nr = knobs['rings'] as number
    const sd = knobs['noise'] as number
    const counts = evenSplit(n, nr)
    const theta = uniform(foldIn(k, 0), [n], { low: 0, high: 2 * Math.PI }).data
    const e = normal(foldIn(k, 1), [n]).data
    const x = new Float64Array(2 * n)
    const y: number[] = []
    let r = 0
    for (const [c, m] of counts.entries()) {
      for (let i = 0; i < m; i++, r++) {
        const rad = c + 1 + sd * (e[r] ?? 0)
        x[2 * r] = rad * Math.cos(theta[r] ?? 0)
        x[2 * r + 1] = rad * Math.sin(theta[r] ?? 0)
        y.push(c)
      }
    }
    const logPrior = counts.map((m) => Math.log(m / n))
    const joint = (v: Value): Value => {
      const radius = sqrt(sum(square(v), 1))
      return stack(
        counts.map((_, c) =>
          add(
            mul(square(sub(radius, c + 1)), -0.5 / (sd * sd)),
            (logPrior[c] ?? 0) - Math.log(sd * Math.sqrt(2 * Math.PI)),
          ),
        ),
        1,
      )
    }
    const posterior = (v: Value): Value => softmax(joint(v), 1)
    const draw = (key: Key, m: number): Value => {
      const ring = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => Math.min(nr - 1, Math.floor(u * nr)))
      const t = uniform(foldIn(key, 1), [m], { low: 0, high: 2 * Math.PI }).data
      const z = normal(foldIn(key, 2), [m]).data
      return matrix(
        ring.flatMap((c, i) => {
          const rad = c + 1 + sd * (z[i] ?? 0)
          return [rad * Math.cos(t[i] ?? 0), rad * Math.sin(t[i] ?? 0)]
        }),
        m,
        2,
      )
    }
    const truth: Truth = { posterior, extra: { radii: counts.map((_, c) => c + 1) } }
    return clustered('rings', rings.description, knobs, k, matrix(x, n, 2), y, truth, () =>
      monteCarloBayesError(posterior, draw, foldIn(k, 2)),
    )
  },
}

/**
 * Anisotropic blobs (scikit-learn's k-means assumptions example): Gaussian blobs sheared by the linear map
 * `[[0.6, -0.6], [-0.4, 0.8]]`, so clusters are elongated and tilted. The posterior is the exact Gaussian mixture with
 * covariance `σ² Aᵀ A`.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { anisotropicBlobs } from 'whitebox-engine/datasets'
 * generate(anisotropicBlobs, { knobs: { n: 30 } }).data.x.shape // => [30, 2]
 * ```
 */
export const anisotropicBlobs: DatasetGenerator<ClassificationData> = {
  name: 'anisotropicBlobs',
  kind: 'classification',
  description: 'Gaussian blobs sheared by a linear map, so the clusters are elongated.',
  knobs: dict({
    n: nKnob,
    centers: int(1, 20, { label: 'centres', description: 'The number of blobs.', default: 3 }),
    clusterStd: real(0.01, 10, {
      label: 'cluster standard deviation',
      description: 'The standard deviation of each blob before shearing.',
      default: 1,
    }),
    boxSize: real(0.1, 100, {
      label: 'centre box half-width',
      description: 'Centres are drawn uniformly in [-boxSize, boxSize] along each axis.',
      default: 10,
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const c = knobs['centers'] as number
    const sd = knobs['clusterStd'] as number
    const box = knobs['boxSize'] as number
    const shear = [
      [0.6, -0.6],
      [-0.4, 0.8],
    ]
    const a = tensor(shear)
    const centres = Array.from({ length: c }, (_, i) =>
      Array.from(uniform(foldIn(foldIn(k, 0), i), [2], { low: -box, high: box }).data),
    )
    const counts = evenSplit(n, c)
    const z = normal(foldIn(k, 1), [n, 2]).data
    const raw = new Float64Array(2 * n)
    const y: number[] = []
    let r = 0
    for (const [ci, m] of counts.entries()) {
      for (let i = 0; i < m; i++, r++) {
        raw[2 * r] = (centres[ci]?.[0] ?? 0) + sd * (z[2 * r] ?? 0)
        raw[2 * r + 1] = (centres[ci]?.[1] ?? 0) + sd * (z[2 * r + 1] ?? 0)
        y.push(ci)
      }
    }
    const x = matmul(matrix(raw, n, 2), a) as Tensor
    const ainv = inv(a)
    const logDet = item(slogdet(a).logabsdet)
    const exact = mixtureTruth({
      logDensities: (v) => {
        const back = matmul(v, ainv)
        return sub(isotropicLogDensities(centres, sd)(back), logDet)
      },
      classOf: centres.map((_, i) => i),
      logWeights: counts.map((m) => Math.log(m / n)),
      nClasses: c,
    })
    const draw = (key: Key, m: number): Value => {
      const cls = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => Math.min(c - 1, Math.floor(u * c)))
      const e = normal(foldIn(key, 1), [m, 2]).data
      const pts = cls.flatMap((ci, i) => [
        (centres[ci]?.[0] ?? 0) + sd * (e[2 * i] ?? 0),
        (centres[ci]?.[1] ?? 0) + sd * (e[2 * i + 1] ?? 0),
      ])
      return matmul(matrix(pts, m, 2), a)
    }
    const truth: Truth = {
      ...exact,
      latents: { centers: centres.map((p) => toFlat(matmul(tensor([p]), a))) },
      extra: { transform: shear },
    }
    return clustered('anisotropicBlobs', anisotropicBlobs.description, knobs, k, x, y, truth, () =>
      monteCarloBayesError(exact.posterior, draw, foldIn(k, 2), c),
    )
  },
}

/**
 * Blobs of unequal spread (scikit-learn's k-means assumptions example): isotropic Gaussian clusters with standard
 * deviations `[1, 2.5, 0.5]` (cycled for more clusters, scaled by `spread`), with the exact mixture posterior.
 *
 * @example
 * ```ts
 * import { generate } from 'whitebox-engine/data'
 * import { unequalVarianceBlobs } from 'whitebox-engine/datasets'
 * generate(unequalVarianceBlobs).truth.extra?.['stds'] // => [1, 2.5, 0.5]
 * ```
 */
export const unequalVarianceBlobs: DatasetGenerator<ClassificationData> = {
  name: 'unequalVarianceBlobs',
  kind: 'classification',
  description: 'Isotropic Gaussian blobs with different standard deviations.',
  knobs: dict({
    n: nKnob,
    centers: int(1, 20, { label: 'centres', description: 'The number of blobs.', default: 3 }),
    spread: real(0.01, 10, {
      description: 'Scales the blobs’ standard deviations (0.5, 1.5 and 2.5 times, cycling).',
      default: 1,
      label: 'scale of the standard deviations',
    }),
    boxSize: real(0.1, 100, {
      label: 'centre box half-width',
      description: 'Centres are drawn uniformly in [-boxSize, boxSize] along each axis.',
      default: 10,
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const c = knobs['centers'] as number
    const spread = knobs['spread'] as number
    const box = knobs['boxSize'] as number
    const stds = Array.from({ length: c }, (_, i) => spread * ([1, 2.5, 0.5][i % 3] ?? 1))
    const centres = Array.from({ length: c }, (_, i) =>
      Array.from(uniform(foldIn(foldIn(k, 0), i), [2], { low: -box, high: box }).data),
    )
    const counts = evenSplit(n, c)
    const z = normal(foldIn(k, 1), [n, 2]).data
    const x = new Float64Array(2 * n)
    const y: number[] = []
    let r = 0
    for (const [ci, m] of counts.entries()) {
      for (let i = 0; i < m; i++, r++) {
        x[2 * r] = (centres[ci]?.[0] ?? 0) + (stds[ci] ?? 1) * (z[2 * r] ?? 0)
        x[2 * r + 1] = (centres[ci]?.[1] ?? 0) + (stds[ci] ?? 1) * (z[2 * r + 1] ?? 0)
        y.push(ci)
      }
    }
    const exact = mixtureTruth({
      logDensities: isotropicLogDensities(centres, stds),
      isotropic: { means: centres, sigma: stds },
      classOf: centres.map((_, i) => i),
      logWeights: counts.map((m) => Math.log(m / n)),
      nClasses: c,
    })
    const draw = (key: Key, m: number): Value => {
      const cls = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => Math.min(c - 1, Math.floor(u * c)))
      const e = normal(foldIn(key, 1), [m, 2]).data
      return matrix(
        cls.flatMap((ci, i) => [
          (centres[ci]?.[0] ?? 0) + (stds[ci] ?? 1) * (e[2 * i] ?? 0),
          (centres[ci]?.[1] ?? 0) + (stds[ci] ?? 1) * (e[2 * i + 1] ?? 0),
        ]),
        m,
        2,
      )
    }
    const truth: Truth = { ...exact, latents: { centers: centres }, extra: { stds } }
    return clustered(
      'unequalVarianceBlobs',
      unequalVarianceBlobs.description,
      knobs,
      k,
      matrix(x, n, 2),
      y,
      truth,
      () => monteCarloBayesError(exact.posterior, draw, foldIn(k, 2), c),
    )
  },
}

/**
 * Old Faithful geyser eruptions (Härdle, 1991; R's `datasets::faithful`): eruption length and waiting time to the next
 * eruption in minutes for 272 eruptions, a classic two-cluster example. The labels are the conventional split into
 * short (under 3 minutes) and long eruptions, which `truth.clusters` repeats.
 *
 * @returns The dataset and its two clusters.
 * @example
 * ```ts
 * import { oldFaithful } from 'whitebox-engine/datasets'
 * oldFaithful().data.x.shape // => [272, 2]
 * ```
 */
export function oldFaithful(): Generated<ClassificationData> {
  const e = FAITHFUL_COLUMNS['eruptions'] as readonly number[]
  const w = FAITHFUL_COLUMNS['waiting'] as readonly number[]
  const n = e.length
  const y = e.map((v) => (v < 3 ? 0 : 1))
  const data = xyClasses(
    matrix(
      e.flatMap((v, i) => [v, w[i] ?? 0]),
      n,
      2,
    ),
    labels(y),
    {
      name: 'oldFaithful',
      description: 'Eruption durations and waiting times of the Old Faithful geyser (Hardle, 1991).',
      source: 'R datasets::faithful via Rdatasets',
      url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html',
      citation: 'Hardle (1991), Smoothing Techniques with Implementation in S, Springer.',
      licence: 'Public domain data; table from R (GPL-2 | GPL-3)',
      sha256: FAITHFUL_SHA256,
      featureNames: ['eruptions', 'waiting'],
      classes: ['short', 'long'],
    },
  )
  return { data, truth: { clusters: labels(y), extra: { rule: 'eruptions < 3 minutes is short' } } }
}
