/**
 * Synthetic classification generators with exact class posteriors and Bayes errors: Gaussian blobs, moons and circles
 * (scikit-learn's `make_blobs`, `make_moons`, `make_circles`), XOR, spirals, a checkerboard, Gaussian quantiles and
 * Hastie's problem 10.2. The noise-free point sets behind the sklearn generators are exported too.
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
import { gammaincinv, ndtr } from '../numerics/special/index.ts'
import { foldIn, type Key, normal, permutation, uniform } from '../random/index.ts'
import { bool, dict, int, real } from '../space/index.ts'
import {
  add,
  cast,
  floor,
  fromStorage,
  greater,
  item,
  mod,
  mul,
  oneHot,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  take as takeRows,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { isotropicLogDensities, mixtureTruth, monteCarloBayesError, withLazyBayesError } from './_mixtures.ts'
import { column, evenSplit, labels, matrix, stamp } from './_util.ts'

/** Shuffle rows of a generated dataset (and its row-aligned latents) when asked. */
function finish(
  name: string,
  description: string,
  knobs: Knobs,
  k: Key,
  x: Tensor,
  y: readonly number[],
  truth: Truth,
  latentRows: Readonly<Record<string, Tensor>> = {},
  classes?: readonly string[],
  bayes?: () => number,
): Generated<ClassificationData> {
  const data = xyClasses(x, labels(y), {
    ...stamp(name, description, knobs),
    ...(classes === undefined ? {} : { classes }),
  })
  const shuffle = knobs['shuffle'] !== false
  const n = y.length
  const order = shuffle ? permutation(foldIn(k, 99), n) : undefined
  const rows = order === undefined ? data : take(data, order)
  const latents: Record<string, unknown> = { ...(truth.latents ?? {}) }
  for (const [key, v] of Object.entries(latentRows)) {
    latents[key] = order === undefined ? v : (takeRows(v, order, 0) as Tensor)
  }
  const clusters =
    truth.clusters === undefined
      ? undefined
      : order === undefined
        ? truth.clusters
        : labels(Array.from(order.data, (i) => truth.clusters?.data[i] ?? 0))
  const out: Truth = { ...truth, ...(clusters === undefined ? {} : { clusters }), latents }
  return { data: rows, truth: bayes === undefined ? out : withLazyBayesError(out, bayes) }
}

const nKnob = int(2, 100_000, {
  description: 'The number of rows, split evenly between the centres.',
  default: 100,
  label: 'samples',
})
const shuffleKnob = bool({
  description: 'Shuffle the rows; otherwise they come grouped by blob.',
  default: true,
  label: 'shuffle rows',
})

// ---------------------------------------------------------------------------------------------------------------
// Blobs

/**
 * Isotropic Gaussian blobs (sklearn.datasets.make_blobs): centres uniform in a box, an even split of rows between
 * centres, and the exact class posterior of the mixture (with priors from the split), its log density and a Monte
 * Carlo Bayes error. The centres are in `truth.latents.centers`.
 *
 * @index dataset Isotropic Gaussian blobs with the exact class posterior.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { blobs } from 'mdml-engine/datasets'
 * generate(blobs, { knobs: { n: 30, centers: 3 } }).data.classes // => ['0', '1', '2']
 * ```
 */
export const blobs: DatasetGenerator<ClassificationData> = {
  name: 'blobs',
  kind: 'classification',
  description: 'Isotropic Gaussian blobs with centres uniform in a box (sklearn make_blobs).',
  knobs: dict({
    n: nKnob,
    nFeatures: int(1, 100, { description: 'The number of features (dimensions).', default: 2, label: 'features' }),
    centers: int(1, 50, { description: 'The number of Gaussian blobs.', default: 3, label: 'centres' }),
    clusterStd: real(0.01, 20, {
      description: 'The standard deviation of each blob along each axis.',
      default: 1,
      label: 'cluster standard deviation',
    }),
    boxSize: real(0.1, 100, {
      description: 'Centres are drawn uniformly in [-boxSize, boxSize] along each axis.',
      default: 10,
      label: 'centre box half-width',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['nFeatures'] as number
    const c = knobs['centers'] as number
    const sd = knobs['clusterStd'] as number
    const box = knobs['boxSize'] as number
    const centres = uniform(foldIn(k, 0), [c, d], { low: -box, high: box }).data
    const means = Array.from({ length: c }, (_, i) => Array.from(centres.slice(i * d, (i + 1) * d)))
    const counts = evenSplit(n, c)
    const z = normal(foldIn(k, 1), [n, d]).data
    const x = new Float64Array(n * d)
    const y: number[] = []
    let r = 0
    for (const [ci, m] of counts.entries()) {
      for (let s = 0; s < m; s++, r++) {
        for (let j = 0; j < d; j++) x[r * d + j] = (means[ci]?.[j] ?? 0) + sd * (z[r * d + j] ?? 0)
        y.push(ci)
      }
    }
    const model = {
      logDensities: isotropicLogDensities(means, sd),
      isotropic: { means, sigma: sd },
      classOf: means.map((_, i) => i),
      logWeights: counts.map((m) => Math.log(m / n)),
      nClasses: c,
    }
    const exact = mixtureTruth(model)
    const draw = (key: Key, m: number): Value => {
      const cls = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => {
        let acc = 0
        for (const [ci, cnt] of counts.entries()) {
          acc += cnt / n
          if (u < acc) return ci
        }
        return c - 1
      })
      const e = normal(foldIn(key, 1), [m, d]).data
      return matrix(
        Float64Array.from(e, (v, i) => (means[cls[Math.floor(i / d)] ?? 0]?.[i % d] ?? 0) + sd * v),
        m,
        d,
      )
    }
    const truth: Truth = {
      ...exact,
      clusters: labels(y),
      latents: { centers: means },
      extra: { priors: counts.map((m) => m / n) },
    }
    return finish('blobs', blobs.description, knobs, k, matrix(x, n, d), y, truth, {}, undefined, () =>
      monteCarloBayesError(exact.posterior, draw, foldIn(k, 2), c),
    )
  },
}

// ---------------------------------------------------------------------------------------------------------------
// Moons and circles: points at evenly spaced angles plus Gaussian noise, so the posterior is an exact mixture.

/**
 * The noise-free moons of sklearn.datasets.make_moons (unshuffled): `n // 2` points on the upper arc (class 0) and the
 * rest on the shifted lower arc (class 1).
 *
 * @param n - The number of points.
 * @returns The points `[n, 2]` and their classes.
 * @example
 * ```ts
 * import { moonPoints } from 'mdml-engine/datasets'
 * moonPoints(4).y // => [0, 0, 1, 1]
 * ```
 */
export function moonPoints(n: number): { readonly x: Tensor; readonly y: number[]; readonly angle: number[] } {
  const nOut = Math.floor(n / 2)
  const nIn = n - nOut
  const x = new Float64Array(2 * n)
  const y: number[] = []
  const angle: number[] = []
  const at = (m: number, i: number) => (m === 1 ? 0 : (Math.PI * i) / (m - 1))
  for (let i = 0; i < nOut; i++) {
    const t = at(nOut, i)
    x[2 * i] = Math.cos(t)
    x[2 * i + 1] = Math.sin(t)
    y.push(0)
    angle.push(t)
  }
  for (let i = 0; i < nIn; i++) {
    const t = at(nIn, i)
    x[2 * (nOut + i)] = 1 - Math.cos(t)
    x[2 * (nOut + i) + 1] = 1 - Math.sin(t) - 0.5
    y.push(1)
    angle.push(t)
  }
  return { x: matrix(x, n, 2), y, angle }
}

/**
 * The noise-free circles of sklearn.datasets.make_circles (unshuffled): `n // 2` points on the unit circle (class 0)
 * and the rest on a circle of radius `factor` (class 1), at evenly spaced angles.
 *
 * @param n - The number of points.
 * @param factor - The inner radius (default 0.8).
 * @returns The points `[n, 2]` and their classes.
 * @example
 * ```ts
 * import { circlePoints } from 'mdml-engine/datasets'
 * circlePoints(4, 0.5).y // => [0, 0, 1, 1]
 * ```
 */
export function circlePoints(
  n: number,
  factor = 0.8,
): { readonly x: Tensor; readonly y: number[]; readonly angle: number[] } {
  const nOut = Math.floor(n / 2)
  const nIn = n - nOut
  const x = new Float64Array(2 * n)
  const y: number[] = []
  const angle: number[] = []
  for (let i = 0; i < nOut; i++) {
    const t = (2 * Math.PI * i) / nOut
    x[2 * i] = Math.cos(t)
    x[2 * i + 1] = Math.sin(t)
    y.push(0)
    angle.push(t)
  }
  for (let i = 0; i < nIn; i++) {
    const t = (2 * Math.PI * i) / nIn
    x[2 * (nOut + i)] = factor * Math.cos(t)
    x[2 * (nOut + i) + 1] = factor * Math.sin(t)
    y.push(1)
    angle.push(t)
  }
  return { x: matrix(x, n, 2), y, angle }
}

/** Data and truth for points at fixed positions plus isotropic noise. */
function noisyPoints(
  name: string,
  description: string,
  knobs: Knobs,
  k: Key,
  base: { readonly x: Tensor; readonly y: number[]; readonly angle: number[] },
): Generated<ClassificationData> {
  const noise = knobs['noise'] as number
  const n = base.y.length
  const sigma = Math.max(noise, 1e-3)
  const means = Array.from({ length: n }, (_, i) => [base.x.data[2 * i] ?? 0, base.x.data[2 * i + 1] ?? 0])
  const model = {
    logDensities: isotropicLogDensities(means, sigma),
    isotropic: { means, sigma },
    classOf: base.y,
    logWeights: base.y.map(() => -Math.log(n)),
    nClasses: 2,
  }
  const exact = mixtureTruth(model)
  const z = normal(foldIn(k, 0), [n, 2]).data
  const x = matrix(
    Float64Array.from(base.x.data, (v, i) => v + noise * (z[i] ?? 0)),
    n,
    2,
  )
  const draw = (key: Key, m: number): Value => {
    const pick = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => Math.min(n - 1, Math.floor(u * n)))
    const e = normal(foldIn(key, 1), [m, 2]).data
    return matrix(
      Float64Array.from(e, (v, i) => (means[pick[Math.floor(i / 2)] ?? 0]?.[i % 2] ?? 0) + noise * v),
      m,
      2,
    )
  }
  const truth: Truth = {
    ...exact,
    noiseScale: noise,
  }
  return finish(
    name,
    description,
    knobs,
    k,
    x,
    base.y,
    truth,
    { angle: fromStorage(Float64Array.from(base.angle), [n], 'float64') },
    undefined,
    () => (noise === 0 ? 0 : monteCarloBayesError(exact.posterior, draw, foldIn(k, 1), n)),
  )
}

/**
 * Two interleaving half circles (sklearn.datasets.make_moons): points at evenly spaced angles on each arc plus
 * Gaussian noise, so the class posterior is an exact mixture over the arc points. The angle of each point is
 * `truth.latents.angle`.
 *
 * @index dataset Two interleaving half circles.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { moons } from 'mdml-engine/datasets'
 * generate(moons, { knobs: { n: 50, noise: 0.1 } }).data.x.shape // => [50, 2]
 * ```
 */
export const moons: DatasetGenerator<ClassificationData> = {
  name: 'moons',
  kind: 'classification',
  description: 'Two interleaving half circles with Gaussian noise (sklearn make_moons).',
  knobs: dict({
    n: nKnob,
    noise: real(0, 2, {
      description: 'The standard deviation of the Gaussian noise added to the points.',
      default: 0.1,
      label: 'noise standard deviation',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => noisyPoints('moons', moons.description, knobs, k, moonPoints(knobs['n'] as number)),
}

/**
 * A large circle containing a smaller one (sklearn.datasets.make_circles), with the exact mixture posterior.
 *
 * @index dataset A large circle containing a smaller one.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { circles } from 'mdml-engine/datasets'
 * generate(circles, { knobs: { n: 40, factor: 0.5 } }).data.classes // => ['0', '1']
 * ```
 */
export const circles: DatasetGenerator<ClassificationData> = {
  name: 'circles',
  kind: 'classification',
  description: 'A large circle containing a smaller circle, with Gaussian noise (sklearn make_circles).',
  knobs: dict({
    n: nKnob,
    noise: real(0, 2, {
      description: 'The standard deviation of the Gaussian noise added to the points.',
      default: 0.05,
      label: 'noise standard deviation',
    }),
    factor: real(0, 0.999, {
      description: 'The radius of the inner circle relative to the outer (1).',
      default: 0.8,
      label: 'inner radius',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) =>
    noisyPoints(
      'circles',
      circles.description,
      knobs,
      k,
      circlePoints(knobs['n'] as number, knobs['factor'] as number),
    ),
}

// ---------------------------------------------------------------------------------------------------------------
// XOR, spirals, checkerboards

/**
 * The XOR problem: four Gaussian blobs at `(±s, ±s)`, labelled by whether the signs of the two coordinates differ,
 * with the exact posterior and Bayes error (each coordinate's sign is misjudged with probability `Φ(-s / σ)`).
 *
 * @index dataset The XOR problem as four Gaussian blobs.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { xor } from 'mdml-engine/datasets'
 * generate(xor, { knobs: { separation: 1, noise: 0.5 } }).truth.bayesError // => 0.0445
 * ```
 */
export const xor: DatasetGenerator<ClassificationData> = {
  name: 'xor',
  kind: 'classification',
  description: 'Four Gaussian blobs at the corners of a square, labelled exclusive-or.',
  knobs: dict({
    n: nKnob,
    separation: real(0.1, 10, {
      description: 'The distance of each blob’s centre from the origin along each axis.',
      default: 1,
      label: 'corner distance',
    }),
    noise: real(0.01, 5, {
      description: 'The standard deviation of each blob.',
      default: 0.3,
      label: 'blob standard deviation',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const s = knobs['separation'] as number
    const sd = knobs['noise'] as number
    const corners = [
      [s, s],
      [-s, -s],
      [s, -s],
      [-s, s],
    ]
    const cls = [0, 0, 1, 1]
    const counts = evenSplit(n, 4)
    const z = normal(foldIn(k, 0), [n, 2]).data
    const x = new Float64Array(2 * n)
    const y: number[] = []
    let r = 0
    for (const [c, m] of counts.entries()) {
      for (let i = 0; i < m; i++, r++) {
        x[2 * r] = (corners[c]?.[0] ?? 0) + sd * (z[2 * r] ?? 0)
        x[2 * r + 1] = (corners[c]?.[1] ?? 0) + sd * (z[2 * r + 1] ?? 0)
        y.push(cls[c] ?? 0)
      }
    }
    const exact = mixtureTruth({
      logDensities: isotropicLogDensities(corners, sd),
      classOf: cls,
      logWeights: counts.map((m) => Math.log(m / n)),
      nClasses: 2,
    })
    // An error happens when exactly one coordinate's sign is wrong (the posterior is symmetric).
    const q = item(ndtr(-s / sd))
    const truth: Truth = { ...exact, bayesError: Math.round(2 * q * (1 - q) * 1e4) / 1e4, clusters: labels(y) }
    return finish('xor', xor.description, knobs, k, matrix(x, n, 2), y, truth)
  },
}

/**
 * Interleaved spiral arms (the "two spirals" benchmark, generalised to several arms): position `t` uniform on [0, 1],
 * radius `t`, angle `2π (turns t + arm / arms)`, plus Gaussian noise. The posterior integrates over `t` by a
 * 400-point quadrature along each arm, so it is exact to quadrature accuracy; `truth.latents.t` holds the positions.
 *
 * @index dataset Interleaved spiral arms.
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { spirals } from 'mdml-engine/datasets'
 * generate(spirals, { knobs: { n: 60, arms: 3 } }).data.classes // => ['0', '1', '2']
 * ```
 */
export const spirals: DatasetGenerator<ClassificationData> = {
  name: 'spirals',
  kind: 'classification',
  description: 'Interleaved spiral arms with Gaussian noise.',
  knobs: dict({
    n: nKnob,
    arms: int(2, 10, { label: 'arms', description: 'The number of spiral arms (one class each).', default: 2 }),
    turns: real(0.25, 5, { description: 'How many turns each arm makes.', default: 1.5, label: 'turns per arm' }),
    noise: real(0.005, 1, {
      description: 'The standard deviation of the Gaussian noise added to the points.',
      default: 0.05,
      label: 'noise standard deviation',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const arms = knobs['arms'] as number
    const turns = knobs['turns'] as number
    const sd = knobs['noise'] as number
    const at = (t: number, a: number): [number, number] => {
      const th = 2 * Math.PI * (turns * t + a / arms)
      return [t * Math.cos(th), t * Math.sin(th)]
    }
    const counts = evenSplit(n, arms)
    const tt = uniform(foldIn(k, 0), [n]).data
    const z = normal(foldIn(k, 1), [n, 2]).data
    const x = new Float64Array(2 * n)
    const y: number[] = []
    let r = 0
    for (const [a, m] of counts.entries()) {
      for (let i = 0; i < m; i++, r++) {
        const [px, py] = at(tt[r] ?? 0, a)
        x[2 * r] = px + sd * (z[2 * r] ?? 0)
        x[2 * r + 1] = py + sd * (z[2 * r + 1] ?? 0)
        y.push(a)
      }
    }
    const grid = 400
    const means: number[][] = []
    const classOf: number[] = []
    const logWeights: number[] = []
    for (const [a, m] of counts.entries()) {
      for (let g = 0; g < grid; g++) {
        means.push(at((g + 0.5) / grid, a))
        classOf.push(a)
        logWeights.push(Math.log(m / n / grid))
      }
    }
    const exact = mixtureTruth({
      logDensities: isotropicLogDensities(means, sd),
      isotropic: { means, sigma: sd },
      classOf,
      logWeights,
      nClasses: arms,
    })
    const draw = (key: Key, m: number): Value => {
      const arm = Array.from(uniform(foldIn(key, 0), [m]).data, (u) => Math.min(arms - 1, Math.floor(u * arms)))
      const t = uniform(foldIn(key, 1), [m]).data
      const e = normal(foldIn(key, 2), [m, 2]).data
      const out = new Float64Array(2 * m)
      for (let i = 0; i < m; i++) {
        const [px, py] = at(t[i] ?? 0, arm[i] ?? 0)
        out[2 * i] = px + sd * (e[2 * i] ?? 0)
        out[2 * i + 1] = py + sd * (e[2 * i + 1] ?? 0)
      }
      return matrix(out, m, 2)
    }
    const truth: Truth = {
      ...exact,
      noiseScale: sd,
    }
    return finish(
      'spirals',
      spirals.description,
      knobs,
      k,
      matrix(x, n, 2),
      y,
      truth,
      { t: fromStorage(Float64Array.from(tt), [n], 'float64') },
      undefined,
      () => monteCarloBayesError(exact.posterior, draw, foldIn(k, 2), means.length),
    )
  },
}

/**
 * A checkerboard: features uniform on `[0, cells)^2`, the class the parity of the cell, and labels flipped with
 * probability `labelNoise`, so the Bayes error is exactly `min(labelNoise, 1 - labelNoise)`.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { checkerboard } from 'mdml-engine/datasets'
 * generate(checkerboard, { knobs: { labelNoise: 0.1 } }).truth.bayesError // => 0.1
 * ```
 */
export const checkerboard: DatasetGenerator<ClassificationData> = {
  name: 'checkerboard',
  kind: 'classification',
  description: 'Uniform points on a checkerboard, labelled by the colour of their cell.',
  knobs: dict({
    n: nKnob,
    cells: int(1, 20, { description: 'The number of cells along each side.', default: 4, label: 'cells per side' }),
    labelNoise: real(0, 1, {
      description: 'The probability that each label is flipped.',
      default: 0,
      label: 'label flip probability',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const cells = knobs['cells'] as number
    const flip = knobs['labelNoise'] as number
    const x = uniform(foldIn(k, 0), [n, 2], { low: 0, high: cells })
    const u = uniform(foldIn(k, 1), [n]).data
    const y = Array.from({ length: n }, (_, i) => {
      const c = (Math.floor(x.data[2 * i] ?? 0) + Math.floor(x.data[2 * i + 1] ?? 0)) % 2
      return (u[i] ?? 1) < flip ? 1 - c : c
    })
    const posterior = (v: Value): Value => {
      const parity = mod(add(floor(column(v, 0)), floor(column(v, 1))), 2)
      const p1 = add(mul(parity, 1 - 2 * flip), flip)
      return stack([sub(1, p1), p1], 1)
    }
    const truth: Truth = { posterior, bayesError: Math.min(flip, 1 - flip), extra: { cells } }
    return finish('checkerboard', checkerboard.description, { ...knobs, shuffle: false }, k, x, y, truth)
  },
}

/**
 * Class labels of sklearn.datasets.make_gaussian_quantiles for given rows: rows sorted by squared distance from the
 * mean, then split into `nClasses` equal groups (the last takes the remainder).
 *
 * @param x - The rows `[n, d]`.
 * @param nClasses - The number of classes.
 * @param mean - The centre (default the origin).
 * @returns A class per row.
 * @example
 * ```ts
 * import { gaussianQuantileLabels } from 'mdml-engine/datasets'
 * gaussianQuantileLabels([[3], [0.1], [-1], [2]], 2) // => [1, 0, 0, 1]
 * ```
 */
export function gaussianQuantileLabels(
  x: Value | readonly (readonly number[])[],
  nClasses: number,
  mean?: readonly number[],
): number[] {
  const t = Array.isArray(x) ? tensor(x as number[][]) : (x as Value)
  const d = t.shape[1] ?? 1
  const flat = toFlat(t)
  const n = flat.length / d
  const dist = Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < d; j++) s += ((flat[i * d + j] ?? 0) - (mean?.[j] ?? 0)) ** 2
    return s
  })
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (dist[a] ?? 0) - (dist[b] ?? 0) || a - b)
  const step = Math.floor(n / nClasses)
  const out = new Array<number>(n).fill(nClasses - 1)
  for (const [rank, i] of order.entries()) out[i] = Math.min(nClasses - 1, Math.floor(rank / Math.max(1, step)))
  return out
}

/**
 * Nested classes by Gaussian quantiles (sklearn.datasets.make_gaussian_quantiles): `x ~ N(mean, σ² I)`, rows split
 * into equal groups by their distance from the mean. The population truth thresholds the squared distance at the
 * chi-squared quantiles `σ² χ²_d(k / K)`, so the Bayes error is 0.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { gaussianQuantiles } from 'mdml-engine/datasets'
 * generate(gaussianQuantiles, { knobs: { n: 30 } }).truth.bayesError // => 0
 * ```
 */
export const gaussianQuantiles: DatasetGenerator<ClassificationData> = {
  name: 'gaussianQuantiles',
  kind: 'classification',
  description:
    'Classes nested by quantiles of the distance from the centre of a Gaussian (sklearn make_gaussian_quantiles).',
  knobs: dict({
    n: nKnob,
    nFeatures: int(1, 100, { description: 'The number of feature columns.', default: 2, label: 'features' }),
    nClasses: int(2, 20, {
      description: 'The number of classes (nested shells of equal probability).',
      default: 3,
      label: 'classes',
    }),
    cov: real(0.01, 100, {
      description: 'The variance of the isotropic Gaussian along each axis.',
      default: 1,
      label: 'variance',
    }),
    shuffle: shuffleKnob,
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const d = knobs['nFeatures'] as number
    const nc = knobs['nClasses'] as number
    const v = knobs['cov'] as number
    const x = matrix(
      Float64Array.from(normal(foldIn(k, 0), [n, d]).data, (z) => Math.sqrt(v) * z),
      n,
      d,
    )
    const y = gaussianQuantileLabels(x, nc)
    const thresholds = Array.from({ length: nc - 1 }, (_, c) => v * 2 * item(gammaincinv(d / 2, (c + 1) / nc)))
    const posterior = (val: Value): Value => {
      const r2 = toFlat(sum(square(val), 1))
      const cls = r2.map((r) => thresholds.filter((t) => r > t).length)
      return oneHot(tensor(cls, { dtype: 'int32' }), nc)
    }
    const truth: Truth = { posterior, bayesError: 0, extra: { thresholds } }
    return finish('gaussianQuantiles', gaussianQuantiles.description, knobs, k, x, y, truth)
  },
}

/**
 * Hastie, Tibshirani and Friedman's binary problem 10.2 (sklearn.datasets.make_hastie_10_2): ten standard normal
 * features, class 1 when the squared norm exceeds 9.34 (the chi-squared median), so the Bayes error is 0.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { hastie10_2 } from 'mdml-engine/datasets'
 * generate(hastie10_2).data.x.shape // => [100, 10]
 * ```
 */
export const hastie10_2: DatasetGenerator<ClassificationData> = {
  name: 'hastie10_2',
  kind: 'classification',
  description: 'Ten standard normal features; class 1 when the squared norm exceeds 9.34 (sklearn make_hastie_10_2).',
  knobs: dict({ n: int(2, 100_000, { description: 'The number of rows.', default: 100, label: 'samples' }) }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const x = normal(k, [n, 10])
    const y = toFlat(greater(sum(square(x), 1), 9.34)).map((v) => (v ? 1 : 0))
    const posterior = (v: Value): Value => oneHot(cast(greater(sum(square(v), 1), 9.34), 'int32'), 2)
    return finish(
      'hastie10_2',
      hastie10_2.description,
      { ...knobs, shuffle: false },
      k,
      x,
      y,
      {
        posterior,
        bayesError: 0,
      },
      {},
      ['-1', '1'],
    )
  },
}
