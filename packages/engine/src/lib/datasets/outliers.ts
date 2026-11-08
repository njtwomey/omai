/**
 * Contaminated datasets: clean data with a known fraction of outliers, labelled so detectors can be scored against the
 * truth. Gaussian blobs and two moons with scattered outliers for anomaly detection (as in scikit-learn's comparison),
 * and a noisy line with outliers for robust regression.
 */

import {
  type ClassificationData,
  type DatasetGenerator,
  type Generated,
  type GeneratorKnobs,
  type RegressionData,
  xy,
  xyClasses,
} from '../data/index.ts'
import { foldIn, type Key, normal, permutation, uniform } from '../random/index.ts'
import { bool, choice, dict, int, real } from '../space/index.ts'
import { add, mul, sliceAxis, squeeze, type Value } from '../tensor/index.ts'
import { labels, matrix, stamp, vector } from './_util.ts'
import { moonPoints } from './classification.ts'

const samples = int(4, 100_000, {
  default: 300,
  label: 'samples',
  description: 'The number of rows, inliers and outliers together.',
})
const contaminationKnob = real(0, 0.5, {
  default: 0.1,
  label: 'contamination',
  description: 'The fraction of rows that are outliers.',
})
const shuffleKnob = bool({
  default: true,
  label: 'shuffle rows',
  description: 'Shuffle the rows; otherwise the inliers come first and the outliers last.',
})

/** The number of outliers among `n` rows at a contamination fraction (rounded, at most `n - 2`). */
function outlierCount(n: number, contamination: number): number {
  return Math.min(n - 2, Math.round(contamination * n))
}

/** A row order: a keyed permutation, or the identity without shuffling. */
function rowOrder(k: Key, n: number, shuffle: boolean): readonly number[] {
  return shuffle ? Array.from(permutation(k, n).data) : Array.from({ length: n }, (_, i) => i)
}

/**
 * Inlier points with uniform outliers in a square, as two classes (0 inlier, 1 outlier), the anomaly-detection
 * setting of sklearn's "Comparing anomaly detection algorithms" example.
 */
function withUniformOutliers(
  name: string,
  description: string,
  knobs: GeneratorKnobs,
  k: Key,
  inliers: Float64Array,
  nOutliers: number,
  halfWidth: number,
): Generated<ClassificationData> {
  const nIn = inliers.length / 2
  const n = nIn + nOutliers
  const out = uniform(foldIn(k, 2), [nOutliers, 2], { low: -halfWidth, high: halfWidth }).data
  const order = rowOrder(foldIn(k, 3), n, knobs['shuffle'] !== false)
  const x = new Float64Array(2 * n)
  const y = new Int32Array(n)
  for (const [r, i] of order.entries()) {
    const outlier = i >= nIn
    const src = outlier ? out : inliers
    const at = outlier ? i - nIn : i
    x[2 * r] = src[2 * at] ?? 0
    x[2 * r + 1] = src[2 * at + 1] ?? 0
    y[r] = outlier ? 1 : 0
  }
  return {
    data: xyClasses(matrix(x, n, 2), labels(y), {
      ...stamp(name, description, knobs),
      featureNames: ['x1', 'x2'],
      classes: ['inlier', 'outlier'],
    }),
    truth: {
      latents: { outlier: labels(y) },
      extra: { contamination: nOutliers / n, nOutliers, outlierBox: [-halfWidth, halfWidth] },
    },
  }
}

/**
 * Gaussian blobs with uniformly scattered outliers, labelled (0 inlier, 1 outlier) so detectors can be scored against
 * the truth: `centers` blobs of standard deviation `clusterStd` evenly spaced on a circle of radius 2.5 (one blob sits
 * at the origin), and a `contamination` fraction of the rows drawn uniformly on `[-6, 6]^2`, as in sklearn's anomaly
 * detection comparison.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { blobsWithOutliers } from 'omai-engine/datasets'
 * const { data } = generate(blobsWithOutliers, { knobs: { n: 100, contamination: 0.2 } })
 * data.classes // => ['inlier', 'outlier']
 * Array.from(data.y.data).filter((v) => v === 1).length // => 20
 * ```
 */
export const blobsWithOutliers: DatasetGenerator<ClassificationData> = {
  name: 'blobsWithOutliers',
  kind: 'classification',
  description: 'Gaussian blobs contaminated by outliers scattered uniformly in a square, labelled inlier or outlier.',
  knobs: dict({
    n: samples,
    contamination: contaminationKnob,
    centers: int(1, 6, {
      default: 2,
      label: 'blobs',
      description: 'The number of Gaussian blobs of inliers, evenly spaced on a circle.',
    }),
    clusterStd: real(0.05, 3, {
      default: 0.5,
      label: 'blob standard deviation',
      description: 'The standard deviation of each blob along each axis.',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nOut = outlierCount(n, knobs['contamination'] as number)
    const nIn = n - nOut
    const c = knobs['centers'] as number
    const sd = knobs['clusterStd'] as number
    const z = normal(foldIn(k, 1), [nIn, 2]).data
    const inliers = new Float64Array(2 * nIn)
    for (let i = 0; i < nIn; i++) {
      const b = Math.floor((i * c) / nIn)
      const t = (2 * Math.PI * b) / c + Math.PI / 4
      const r = c === 1 ? 0 : 2.5
      inliers[2 * i] = r * Math.cos(t) + sd * (z[2 * i] ?? 0)
      inliers[2 * i + 1] = r * Math.sin(t) + sd * (z[2 * i + 1] ?? 0)
    }
    return withUniformOutliers('blobsWithOutliers', blobsWithOutliers.description, knobs, k, inliers, nOut, 6)
  },
}

/**
 * Two moons with uniformly scattered outliers, labelled (0 inlier, 1 outlier): sklearn's `make_moons` points with
 * Gaussian noise, scaled to `4 (x - [0.5, 0.25])` as in its anomaly detection comparison, and a `contamination`
 * fraction of the rows drawn uniformly on `[-6, 6]^2`.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { moonsWithOutliers } from 'omai-engine/datasets'
 * generate(moonsWithOutliers, { knobs: { n: 50 } }).data.x.shape // => [50, 2]
 * ```
 */
export const moonsWithOutliers: DatasetGenerator<ClassificationData> = {
  name: 'moonsWithOutliers',
  kind: 'classification',
  description:
    'Two interleaving half circles contaminated by outliers scattered uniformly, labelled inlier or outlier.',
  knobs: dict({
    n: samples,
    contamination: contaminationKnob,
    noise: real(0, 1, {
      default: 0.05,
      label: 'noise standard deviation',
      description: 'The standard deviation of the Gaussian noise on the moons (before scaling by 4).',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nOut = outlierCount(n, knobs['contamination'] as number)
    const nIn = n - nOut
    const noise = knobs['noise'] as number
    const clean = moonPoints(nIn).x.data
    const z = normal(foldIn(k, 1), [nIn, 2]).data
    const inliers = Float64Array.from(
      { length: 2 * nIn },
      (_, i) => 4 * ((clean[i] ?? 0) + noise * (z[i] ?? 0) - (i % 2 === 0 ? 0.5 : 0.25)),
    )
    return withUniformOutliers('moonsWithOutliers', moonsWithOutliers.description, knobs, k, inliers, nOut, 6)
  },
}

/**
 * A straight line with Gaussian noise and a `contamination` fraction of outliers, for robust regression (Huber,
 * RANSAC, Theil-Sen, quantile regression). Inliers are `y = intercept + slope x + noise e` with x uniform on [0, 10].
 * Outliers are `vertical` (y shifted up by `offset` noise-free units, x as the inliers), `leverage` (x on [9, 12],
 * y pulled below the line by `offset`), or `both` (half of each). The regression function is the clean line, and
 * `truth.latents.outlier` marks the outlying rows.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { linearOutliers } from 'omai-engine/datasets'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * const { truth } = generate(linearOutliers, { knobs: { slope: 2, intercept: 1 } })
 * toFlat(truth.regressionFunction?.(tensor([[0], [1]])) ?? tensor([])) // => [1, 3]
 * ```
 */
export const linearOutliers: DatasetGenerator<RegressionData> = {
  name: 'linearOutliers',
  kind: 'regression',
  description: 'A noisy straight line contaminated by outliers in y or at high leverage in x.',
  knobs: dict({
    n: int(4, 100_000, { default: 100, label: 'samples', description: 'The number of rows, outliers included.' }),
    contamination: real(0, 0.5, {
      default: 0.15,
      label: 'contamination',
      description: 'The fraction of rows that are outliers.',
    }),
    outliers: choice(['vertical', 'leverage', 'both'], {
      default: 'vertical',
      label: 'kind of outliers',
      description: 'Shifted in y only, at high leverage (large x, y below the line), or half of each.',
    }),
    offset: real(0, 100, {
      default: 15,
      label: 'outlier offset',
      description: 'How far the outliers sit from the line, in units of y.',
    }),
    slope: real(-10, 10, { default: 1.5, label: 'slope', description: 'The slope of the clean line.' }),
    intercept: real(-50, 50, { default: 2, label: 'intercept', description: 'The intercept of the clean line.' }),
    noise: real(0, 10, {
      default: 1,
      label: 'noise standard deviation',
      description: 'The standard deviation of the Gaussian noise on the inliers (and on the outliers).',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const nOut = outlierCount(n, knobs['contamination'] as number)
    const kind = knobs['outliers'] as string
    const offset = knobs['offset'] as number
    const slope = knobs['slope'] as number
    const intercept = knobs['intercept'] as number
    const noise = knobs['noise'] as number
    const xs = Float64Array.from(uniform(foldIn(k, 0), [n], { low: 0, high: 10 }).data)
    const e = normal(foldIn(k, 1), [n]).data
    const spread = uniform(foldIn(k, 2), [n], { low: 0.75, high: 1.25 }).data
    // The outliers are the last nOut rows of a keyed shuffle, so they spread along the line.
    const order = rowOrder(foldIn(k, 3), n, true)
    const outlier = new Int32Array(n)
    for (let r = n - nOut; r < n; r++) outlier[order[r] ?? 0] = 1
    let seen = 0
    const ys = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const isOut = outlier[i] === 1
      const leverage = isOut && (kind === 'leverage' || (kind === 'both' && seen++ % 2 === 1))
      if (leverage) xs[i] = 9 + 3 * (((xs[i] ?? 0) / 10) % 1)
      const line = intercept + slope * (xs[i] ?? 0) + noise * (e[i] ?? 0)
      const shift = isOut ? offset * (spread[i] ?? 1) : 0
      ys[i] = leverage ? line - shift : line + shift
    }
    const f = (v: Value): Value => add(mul(squeeze(sliceAxis(v, -1, 0, 1), -1), slope), intercept)
    return {
      data: xy(matrix(xs, n, 1), vector(ys), {
        ...stamp('linearOutliers', linearOutliers.description, knobs),
        featureNames: ['x'],
      }),
      truth: {
        regressionFunction: f,
        noiseScale: noise,
        relevantFeatures: [0],
        latents: { outlier: labels(outlier) },
        extra: { slope, intercept, contamination: nOut / n, nOutliers: nOut, outliers: kind },
      },
    }
  },
}
