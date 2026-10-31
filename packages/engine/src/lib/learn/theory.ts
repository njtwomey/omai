/**
 * Statistical learning theory as functions, each with its assumptions stated: PAC sample complexity and generalisation
 * bounds for finite classes (realisable and agnostic), the union bound, VC dimension (the Vapnik and Sauer-Shelah forms,
 * and the shattering count, growth function and VC dimension of simple classes by enumeration), Rademacher complexity
 * (the Monte Carlo estimate, Massart's finite-class bound, the linear-class closed form and the generalisation bound),
 * margin bounds, algorithmic stability (the uniform stability of ridge regression and the bound that follows), PAC-Bayes
 * (McAllester, Seeger and Catoni, with the Gaussian KL and the inversion of the binary KL by root finding), regret
 * bounds of Hedge and online gradient descent, the bias-variance decomposition (squared loss and Domingos' 0-1 loss) by
 * Monte Carlo over training sets, and degrees of freedom (Efron's covariance definition by Monte Carlo and the closed
 * forms for ridge, lasso and k-nearest neighbours). The bounds are for comparing with simulation: they hold for the
 * stated assumptions and are usually loose.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { binomialExact } from '../foundations/combinatorics.ts'
import { svd } from '../numerics/linalg/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { ndtr } from '../numerics/special/index.ts'
import { linprog } from '../optim/linear-programming.ts'
import { foldIn, type Key, key as makeKey, normal, randint } from '../random/index.ts'
import { asTensor, item, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'

const finite = (who: string, name: string, v: number, min = 0, open = false): void => {
  if (!Number.isFinite(v) || (open ? v <= min : v < min))
    throw new MdmlError('value', `${who}: ${name} must be ${open ? '>' : '>='} ${min}`)
}
const unit = (who: string, delta: number): void => {
  if (!(delta > 0 && delta < 1)) throw new MdmlError('value', `${who}: delta must be in (0, 1)`)
}

/** Whether a PAC statement is for a class that contains the target (realisable) or not (agnostic). */
export type PacSetting = 'realisable' | 'agnostic'

/**
 * Sample size for PAC learning a finite class by empirical risk minimisation. Realisable (some hypothesis has zero
 * risk): a consistent hypothesis has risk at most epsilon with probability 1 - delta once n >= (ln|H| + ln(1/delta)) /
 * epsilon. Agnostic: the ERM hypothesis has risk within epsilon of the best in the class once n >= 2 (ln|H| +
 * ln(2/delta)) / epsilon^2 (uniform convergence to epsilon / 2, by Hoeffding and the union bound; losses in [0, 1]).
 *
 * @param logSize - ln |H|, the log of the number of hypotheses (so that astronomically large classes can be stated).
 * @param epsilon - The accuracy.
 * @param delta - The failure probability.
 * @param setting - `'realisable'` (default) or `'agnostic'`.
 * @returns The smallest sufficient integer sample size.
 * @concept learning-theory
 * @example
 * ```ts
 * import { pacSampleComplexity } from 'mdml-engine/learn'
 * pacSampleComplexity(Math.log(1000), 0.1, 0.05) // => 100
 * pacSampleComplexity(Math.log(1000), 0.1, 0.05, 'agnostic') // => 2120
 * ```
 */
export function pacSampleComplexity(
  logSize: number,
  epsilon: number,
  delta: number,
  setting: PacSetting = 'realisable',
): number {
  finite('pacSampleComplexity', 'logSize', logSize)
  finite('pacSampleComplexity', 'epsilon', epsilon, 0, true)
  unit('pacSampleComplexity', delta)
  return Math.ceil(
    setting === 'realisable'
      ? (logSize + Math.log(1 / delta)) / epsilon
      : (2 * (logSize + Math.log(2 / delta))) / (epsilon * epsilon),
  )
}

/**
 * The PAC bound for a finite class at sample size n. Realisable: with probability 1 - delta every consistent hypothesis
 * has risk at most (ln|H| + ln(1/delta)) / n. Agnostic: with probability 1 - delta every hypothesis has risk at most its
 * training risk plus sqrt((ln|H| + ln(1/delta)) / (2n)) (Hoeffding with the union bound; losses in [0, 1]); the returned
 * value is that excess term, the bound on the generalisation gap.
 *
 * @param n - The sample size.
 * @param logSize - ln |H|.
 * @param delta - The failure probability.
 * @param setting - `'realisable'` (default; the bound on the risk itself) or `'agnostic'` (the bound on the gap).
 * @returns The bound.
 * @concept learning-theory
 * @example
 * ```ts
 * import { pacFiniteBound } from 'mdml-engine/learn'
 * pacFiniteBound(100, Math.log(1000), 0.05) // => 0.0990
 * pacFiniteBound(100, Math.log(1000), 0.05, 'agnostic') // => 0.2225
 * ```
 */
export function pacFiniteBound(n: number, logSize: number, delta: number, setting: PacSetting = 'realisable'): number {
  finite('pacFiniteBound', 'n', n, 0, true)
  finite('pacFiniteBound', 'logSize', logSize)
  unit('pacFiniteBound', delta)
  const c = logSize + Math.log(1 / delta)
  return setting === 'realisable' ? c / n : Math.sqrt(c / (2 * n))
}

/**
 * The union bound for uniform convergence over a finite class: P(some hypothesis deviates from its risk by more than
 * epsilon) <= 2 |H| exp(-2 n epsilon^2) (Hoeffding for losses in [0, 1]), capped at one.
 *
 * @param n - The sample size.
 * @param logSize - ln |H|.
 * @param epsilon - The deviation.
 * @returns The bound on the probability.
 * @concept learning-theory
 * @example
 * ```ts
 * import { unionBound } from 'mdml-engine/learn'
 * unionBound(1000, Math.log(100), 0.1) // => 4.1223e-7
 * ```
 */
export function unionBound(n: number, logSize: number, epsilon: number): number {
  finite('unionBound', 'n', n, 0, true)
  return Math.min(1, 2 * Math.exp(logSize - 2 * n * epsilon * epsilon))
}

/**
 * The Sauer-Shelah bound on the growth function of a class of VC dimension d: Pi(n) <= sum_{i=0}^{d} C(n, i), which is 2^n for
 * n <= d and at most (e n / d)^d beyond it.
 *
 * @param n - The number of points.
 * @param d - The VC dimension.
 * @returns The bound on the number of labellings of n points.
 * @concept learning-theory
 * @example
 * ```ts
 * import { sauerBound } from 'mdml-engine/learn'
 * sauerBound(5, 2) // => 16
 * sauerBound(3, 5) // => 8
 * ```
 */
export function sauerBound(n: number, d: number): number {
  if (!(Number.isInteger(n) && n >= 0 && Number.isInteger(d) && d >= 0))
    throw new MdmlError('value', 'sauerBound: n and d must be non-negative integers')
  let total = 0n
  for (let i = 0; i <= Math.min(d, n); i++) total += binomialExact(n, i)
  return Number(total)
}

/**
 * Vapnik's VC generalisation bound: with probability at least 1 - delta, every hypothesis of a class of VC dimension d
 * has risk at most its training risk plus sqrt((d (ln(2n / d) + 1) + ln(4 / delta)) / n) (losses in [0, 1], n >= d).
 * The returned value is the excess term. It is distribution free and so loose, but it has the right 1/sqrt(n) shape.
 *
 * @param n - The sample size.
 * @param d - The VC dimension.
 * @param delta - The failure probability.
 * @returns The bound on the generalisation gap.
 * @concept learning-theory
 * @example
 * ```ts
 * import { vcBound } from 'mdml-engine/learn'
 * vcBound(10000, 10, 0.05) // => 0.0951
 * ```
 */
export function vcBound(n: number, d: number, delta: number): number {
  finite('vcBound', 'n', n, 0, true)
  finite('vcBound', 'd', d, 0, true)
  unit('vcBound', delta)
  if (n < d) throw new MdmlError('value', 'vcBound: the bound needs n >= d')
  return Math.sqrt((d * (Math.log((2 * n) / d) + 1) + Math.log(4 / delta)) / n)
}

/**
 * The VC sample complexity of realisable PAC learning (Blumer, Ehrenfeucht, Haussler and Warmuth 1989): a consistent
 * hypothesis of a class with VC dimension d has risk at most epsilon with probability 1 - delta when n >= max((4 /
 * epsilon) log2(2 / delta), (8 d / epsilon) log2(13 / epsilon)).
 *
 * @param d - The VC dimension.
 * @param epsilon - The accuracy.
 * @param delta - The failure probability.
 * @returns A sufficient sample size.
 * @concept learning-theory
 * @example
 * ```ts
 * import { vcSampleComplexity } from 'mdml-engine/learn'
 * vcSampleComplexity(3, 0.1, 0.05) // => 1686
 * ```
 */
export function vcSampleComplexity(d: number, epsilon: number, delta: number): number {
  finite('vcSampleComplexity', 'd', d, 0, true)
  finite('vcSampleComplexity', 'epsilon', epsilon, 0, true)
  unit('vcSampleComplexity', delta)
  return Math.ceil(
    Math.max(
      ((4 / epsilon) * Math.log(2 / delta)) / Math.LN2,
      ((8 * d) / epsilon) * (Math.log(13 / epsilon) / Math.LN2),
    ),
  )
}

/**
 * A concept class on points of R^m, for counting the labellings it realises. `realises` says whether some member of the
 * class labels the points exactly as given (true = positive).
 */
export interface ConceptClass {
  readonly name: string
  /** The dimension m of the points. */
  readonly dimension: number
  /** The VC dimension, known in closed form (to check an enumeration against). */
  readonly vcDimension: number
  realises(points: readonly (readonly number[])[], labels: readonly boolean[]): boolean
}

const coordinate = (p: readonly number[] | undefined, j = 0): number => p?.[j] ?? 0

/** Thresholds on the line: positive iff x >= t. VC dimension 1. */
const halfLine: ConceptClass = {
  name: 'half-lines',
  dimension: 1,
  vcDimension: 1,
  realises: (points, labels) => {
    let maxNeg = Number.NEGATIVE_INFINITY
    let minPos = Number.POSITIVE_INFINITY
    points.forEach((p, i) => {
      if (labels[i] === true) minPos = Math.min(minPos, coordinate(p))
      else maxNeg = Math.max(maxNeg, coordinate(p))
    })
    return maxNeg < minPos
  },
}

/** Intervals on the line: positive iff a <= x <= b. VC dimension 2. */
const interval: ConceptClass = {
  name: 'intervals',
  dimension: 1,
  vcDimension: 2,
  realises: (points, labels) => {
    const pos = points.flatMap((p, i) => (labels[i] === true ? [coordinate(p)] : []))
    if (pos.length === 0) return true
    const lo = Math.min(...pos)
    const hi = Math.max(...pos)
    return points.every((p, i) => labels[i] === true || coordinate(p) < lo || coordinate(p) > hi)
  },
}

/** Axis-aligned boxes in R^m: positive iff inside. VC dimension 2m. */
function boxes(dimension: number): ConceptClass {
  return {
    name: `axis-aligned boxes in R^${dimension}`,
    dimension,
    vcDimension: 2 * dimension,
    realises: (points, labels) => {
      const pos = points.filter((_, i) => labels[i] === true)
      if (pos.length === 0) return true
      const lo = Array.from({ length: dimension }, (_, j) => Math.min(...pos.map((p) => coordinate(p, j))))
      const hi = Array.from({ length: dimension }, (_, j) => Math.max(...pos.map((p) => coordinate(p, j))))
      return points.every(
        (p, i) =>
          labels[i] === true || lo.some((l, j) => coordinate(p, j) < l) || hi.some((h, j) => coordinate(p, j) > h),
      )
    },
  }
}

/** Half-spaces in R^m with a bias: positive iff w . x + b >= 0, decided by a linear program. VC dimension m + 1. */
function halfspaces(dimension: number): ConceptClass {
  return {
    name: `half-spaces in R^${dimension}`,
    dimension,
    vcDimension: dimension + 1,
    realises: (points, labels) => {
      if (labels.every((l) => l) || labels.every((l) => !l)) return true
      // Strict separability with unit margin: y_i (w . x_i + b) >= 1, a feasibility problem in (w, b).
      const aUb = points.map((p, i) => {
        const y = labels[i] === true ? 1 : -1
        return [...Array.from({ length: dimension }, (_, j) => -y * coordinate(p, j)), -y]
      })
      const r = linprog({
        c: new Array<number>(dimension + 1).fill(0),
        aUb,
        bUb: new Array<number>(points.length).fill(-1),
        bounds: Array.from({ length: dimension + 1 }, () => [null, null] as const),
      })
      return r.status === 'optimal'
    },
  }
}

/**
 * The concept classes with a counting oracle: thresholds and intervals on the line, axis-aligned boxes and half-spaces in
 * R^m. Each carries its known VC dimension so an enumeration can be checked against it.
 *
 * @concept learning-theory
 * @example
 * ```ts
 * import { conceptClasses } from 'mdml-engine/learn'
 * conceptClasses.halfspaces(2).vcDimension // => 3
 * ```
 */
export const conceptClasses: {
  readonly halfLine: ConceptClass
  readonly interval: ConceptClass
  readonly boxes: (dimension: number) => ConceptClass
  readonly halfspaces: (dimension: number) => ConceptClass
} = { halfLine, interval, boxes, halfspaces }

/**
 * The number of the 2^k labellings of the given points that a concept class realises (k at most 16).
 *
 * @param points - The k points, each an array of the class's dimension.
 * @param cls - The class.
 * @returns How many labellings are realised.
 * @throws MdmlError with code `value` for more than 16 points or points of the wrong dimension.
 * @concept learning-theory
 * @example
 * ```ts
 * import { conceptClasses, shatterCount } from 'mdml-engine/learn'
 * shatterCount([[0], [1], [2]], conceptClasses.interval) // => 7
 * shatterCount([[0, 0], [1, 0], [0, 1]], conceptClasses.halfspaces(2)) // => 8
 * ```
 */
export function shatterCount(points: readonly (readonly number[])[], cls: ConceptClass): number {
  const k = points.length
  if (k > 16) throw new MdmlError('value', 'shatterCount: at most 16 points')
  if (points.some((p) => p.length !== cls.dimension))
    throw new MdmlError('shape', `shatterCount: ${cls.name} needs points of dimension ${cls.dimension}`)
  let count = 0
  for (let mask = 0; mask < 2 ** k; mask++) {
    const labels = Array.from({ length: k }, (_, i) => (mask >> i) % 2 === 1)
    if (cls.realises(points, labels)) count++
  }
  return count
}

/**
 * Whether the class shatters the points (realises all 2^k labellings).
 *
 * @param points - The points.
 * @param cls - The class.
 * @returns True if every labelling is realised.
 * @concept learning-theory
 * @example
 * ```ts
 * import { conceptClasses, isShattered } from 'mdml-engine/learn'
 * isShattered([[0, 0], [1, 0], [0, 1], [1, 1]], conceptClasses.halfspaces(2)) // => false
 * ```
 */
export function isShattered(points: readonly (readonly number[])[], cls: ConceptClass): boolean {
  return shatterCount(points, cls) === 2 ** points.length
}

/** Options of {@link growthFunction} and {@link vcDimensionByShattering}. */
export interface ShatteringOptions {
  /** Random point sets tried beyond the structured one (default 30). */
  readonly trials?: number
  readonly key?: Key
}

/** Points in convex position (on a circle at irregular angles) in R^m, or distinct points on the line; the usual maximisers. */
function structuredPoints(cls: ConceptClass, k: number): number[][] {
  if (cls.dimension === 1) return Array.from({ length: k }, (_, i) => [i])
  return Array.from({ length: k }, (_, i) => {
    const a = (2 * Math.PI * (i + 0.17 * Math.sin(3 * i + 1))) / k
    return [
      Math.cos(a),
      Math.sin(a),
      ...new Array<number>(Math.max(0, cls.dimension - 2)).fill(0).map((_, j) => Math.cos((i + 1) * (j + 2) * 0.7)),
    ].slice(0, cls.dimension)
  })
}

function randomPoints(cls: ConceptClass, k: number, key: Key): number[][] {
  const z = toFlat(normal(key, [k * cls.dimension]))
  return Array.from({ length: k }, (_, i) =>
    Array.from({ length: cls.dimension }, (_, j) => z[i * cls.dimension + j] ?? 0),
  )
}

/**
 * The growth function Pi_H(k) of a class: the most labellings it realises on any k points. Computed as the maximum over
 * a structured point set (convex position, the maximiser for half-spaces in the plane) and random ones, so it is exact
 * for half-lines, intervals and half-spaces in the plane (k^2 - k + 2) and a lower bound in general.
 *
 * @param cls - The class.
 * @param k - The number of points.
 * @param options - Extra random point sets and the key.
 * @returns The largest number of labellings found.
 * @concept learning-theory
 * @example
 * ```ts
 * import { conceptClasses, growthFunction } from 'mdml-engine/learn'
 * growthFunction(conceptClasses.interval, 4) // => 11
 * growthFunction(conceptClasses.halfspaces(2), 5) // => 22
 * ```
 */
export function growthFunction(cls: ConceptClass, k: number, options: ShatteringOptions = {}): number {
  const key = options.key ?? makeKey(0)
  let best = shatterCount(structuredPoints(cls, k), cls)
  for (let t = 0; t < (options.trials ?? 30) && best < 2 ** k; t++)
    best = Math.max(best, shatterCount(randomPoints(cls, k, foldIn(key, t)), cls))
  return best
}

/** The outcome of {@link vcDimensionByShattering}. */
export interface VcSearch {
  /** The largest k for which some tried point set was shattered. */
  readonly vcDimension: number
  /** The shattered configuration of that size. */
  readonly witness: readonly (readonly number[])[]
  /** Whether k + 1 points were tried and none was shattered (evidence for equality; a proof needs the class's theory). */
  readonly nextFailed: boolean
}

/**
 * The VC dimension by shattering enumeration: the largest k up to `maxK` for which a structured or random set of k points
 * is shattered. A shattered set proves VC >= k; that no tried set of k + 1 points is shattered is evidence that it is
 * exactly k (the class's theory proves it: 1, 2, 2m and m + 1 for the classes of {@link conceptClasses}).
 *
 * @param cls - The class.
 * @param maxK - The largest set size to try (default 6).
 * @param options - Random point sets tried per size, and the key.
 * @returns The dimension found, a shattered witness, and whether the next size failed.
 * @concept learning-theory
 * @example
 * ```ts
 * import { conceptClasses, vcDimensionByShattering } from 'mdml-engine/learn'
 * vcDimensionByShattering(conceptClasses.boxes(2)).vcDimension // => 4
 * ```
 */
export function vcDimensionByShattering(cls: ConceptClass, maxK = 6, options: ShatteringOptions = {}): VcSearch {
  const key = options.key ?? makeKey(0)
  const trials = options.trials ?? 60
  const attempt = (k: number): readonly (readonly number[])[] | undefined => {
    const s = structuredPoints(cls, k)
    if (isShattered(s, cls)) return s
    for (let t = 0; t < trials; t++) {
      const r = randomPoints(cls, k, foldIn(key, 1000 * k + t))
      if (isShattered(r, cls)) return r
    }
    return undefined
  }
  let witness: readonly (readonly number[])[] = []
  let k = 0
  while (k < maxK) {
    const w = attempt(k + 1)
    if (w === undefined) break
    witness = w
    k++
  }
  return { vcDimension: k, witness, nextFailed: k < maxK }
}

/** Options of the Monte Carlo estimators of this file. */
export interface MonteCarloOptions {
  /** Number of draws (default 200). */
  readonly draws?: number
  /** The key (default `key(0)`). */
  readonly key?: Key
}

/** State of {@link rademacherAlgorithm}: the supremum correlation found for each sign vector drawn so far. */
export interface RademacherState {
  readonly values: readonly number[]
}

/** The result of {@link empiricalRademacher}. */
export interface RademacherEstimate {
  /** The Monte Carlo mean of sup_h (1/n) sum_i sigma_i h(x_i). */
  readonly estimate: number
  /** The standard error of the mean over the draws. */
  readonly standardError: number
  readonly draws: number
  /** The supremum for each draw. */
  readonly values: readonly number[]
}

function matrixOf(
  who: string,
  x: TensorLike,
): { readonly rows: number; readonly cols: number; readonly data: Float64Array } {
  const t = asTensor(x)
  if (t.shape.length !== 2) throw new MdmlError('shape', `${who}: expects a matrix`)
  return { rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0, data: Float64Array.from(toFlat(t)) }
}

/**
 * Empirical Rademacher complexity by Monte Carlo as an algorithm with one sign vector per step. For a class given by a
 * finite set of predictors, the matrix holds each predictor's outputs on the sample (rows are hypotheses, columns the n
 * points); a step draws sigma in {-1, +1}^n and records sup_h (1/n) sum_i sigma_i h(x_i). With `norm` set the class is
 * instead the linear class {x -> w . x : |w|_2 <= norm} and the matrix holds the sample's features (rows are the points),
 * for which the supremum is norm |sum_i sigma_i x_i|_2 / n.
 *
 * @param predictions - The finite class (hypotheses x points), or with `norm` the feature matrix (points x features).
 * @param options - Number of draws, key and the norm bound of the linear class.
 * @returns The algorithm over {@link RademacherState}.
 * @concept learning-theory
 * @example
 * ```ts
 * import { rademacherAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = rademacherAlgorithm([[1, 1, 1, 1]], { draws: 3 })
 * runAlgorithm(alg, { maxSteps: 3 }).state.values.length // => 3
 * ```
 */
export function rademacherAlgorithm(
  predictions: TensorLike,
  options: MonteCarloOptions & { readonly norm?: number } = {},
): Algorithm<RademacherState> {
  const m = matrixOf('rademacher', predictions)
  const draws = options.draws ?? 200
  const linear = options.norm !== undefined
  const n = linear ? m.rows : m.cols
  return {
    name: 'rademacher',
    config: { draws, n, hypotheses: linear ? null : m.rows, norm: options.norm ?? null } as Json,
    schema: channels({
      metrics: {
        supremum: { description: 'sup over the class of the correlation with the drawn signs' },
        estimate: { description: 'running mean of the suprema' },
      },
    }),
    init: () => ({ values: [] }),
    step: (s, ctx) => {
      const sigma = Float64Array.from(toFlat(randint(ctx.key, [n], 0, 2)), (v) => 2 * v - 1)
      let sup: number
      if (linear) {
        let norm2 = 0
        for (let j = 0; j < m.cols; j++) {
          let a = 0
          for (let i = 0; i < n; i++) a += (sigma[i] ?? 0) * (m.data[i * m.cols + j] ?? 0)
          norm2 += a * a
        }
        sup = ((options.norm ?? 0) * Math.sqrt(norm2)) / n
      } else {
        sup = Number.NEGATIVE_INFINITY
        for (let h = 0; h < m.rows; h++) {
          let a = 0
          for (let i = 0; i < n; i++) a += (sigma[i] ?? 0) * (m.data[h * m.cols + i] ?? 0)
          sup = Math.max(sup, a / n)
        }
      }
      return { values: [...s.values, sup] }
    },
    done: (s) => (s.values.length >= draws ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        supremum: s.values[s.values.length - 1] ?? Number.NaN,
        estimate: s.values.reduce((a, b) => a + b, 0) / Math.max(1, s.values.length),
      },
    }),
    serialise: (s) => ({ values: [...s.values] }),
    deserialise: (j) => j as unknown as RademacherState,
  }
}

/**
 * The empirical Rademacher complexity of a class on a sample, by Monte Carlo over random sign vectors
 * ({@link rademacherAlgorithm} run to the end).
 *
 * @param predictions - The finite class (hypotheses x points) or, with `norm`, the feature matrix of a linear class.
 * @param options - Draws, key and the norm bound.
 * @returns The estimate with its standard error and the per-draw suprema.
 * @concept learning-theory
 * @example
 * ```ts
 * import { empiricalRademacher } from 'mdml-engine/learn'
 * empiricalRademacher([[1, 1, 1, 1]], { draws: 4000 }).estimate < 0.05 // => true
 * ```
 */
export function empiricalRademacher(
  predictions: TensorLike,
  options: MonteCarloOptions & { readonly norm?: number } = {},
): RademacherEstimate {
  const draws = options.draws ?? 200
  const result = runAlgorithm(rademacherAlgorithm(predictions, options), {
    maxSteps: draws,
    key: options.key ?? makeKey(0),
  })
  const v = result.state.values
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  const variance = v.length > 1 ? v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1) : 0
  return { estimate: mean, standardError: Math.sqrt(variance / v.length), draws: v.length, values: v }
}

/**
 * Massart's bound on the Rademacher complexity of a finite class: R_n <= r sqrt(2 ln|H|) / n where r = max_h |h(x_1..x_n)|_2;
 * for classifiers with values in {-1, +1} (r = sqrt(n)) it is sqrt(2 ln|H| / n).
 *
 * @param n - The sample size.
 * @param logSize - ln |H|.
 * @param maxNorm - r, the largest Euclidean norm of a hypothesis's outputs on the sample (default sqrt(n)).
 * @returns The bound.
 * @concept learning-theory
 * @example
 * ```ts
 * import { rademacherBoundFinite } from 'mdml-engine/learn'
 * rademacherBoundFinite(100, Math.log(1000)) // => 0.3717
 * ```
 */
export function rademacherBoundFinite(n: number, logSize: number, maxNorm: number = Math.sqrt(n)): number {
  finite('rademacherBoundFinite', 'n', n, 0, true)
  finite('rademacherBoundFinite', 'logSize', logSize)
  return (maxNorm * Math.sqrt(2 * logSize)) / n
}

/**
 * The Rademacher complexity bound of a norm-bounded linear class {x -> w . x : |w|_2 <= B} on a sample whose features have
 * |x_i|_2 <= R: the empirical complexity is at most B sqrt(sum |x_i|^2) / n <= B R / sqrt(n) (Cauchy-Schwarz and Jensen).
 * Given the feature matrix the first form is used; given only R, the second.
 *
 * @param norm - B, the bound on |w|_2.
 * @param n - The sample size.
 * @param radius - R, the bound on the features' norms (when no feature matrix is given).
 * @param x - The feature matrix (points x features), for the sample-specific form.
 * @returns The bound.
 * @concept learning-theory
 * @example
 * ```ts
 * import { rademacherBoundLinear } from 'mdml-engine/learn'
 * rademacherBoundLinear(2, 100, 1.5) // => 0.3
 * ```
 */
export function rademacherBoundLinear(norm: number, n: number, radius?: number, x?: TensorLike): number {
  finite('rademacherBoundLinear', 'norm', norm)
  finite('rademacherBoundLinear', 'n', n, 0, true)
  if (x !== undefined) {
    const m = matrixOf('rademacherBoundLinear', x)
    let s = 0
    for (const v of m.data) s += v * v
    return (norm * Math.sqrt(s)) / m.rows
  }
  if (radius === undefined) throw new MdmlError('value', 'rademacherBoundLinear: needs a radius or a feature matrix')
  return (norm * radius) / Math.sqrt(n)
}

/** Options of {@link rademacherGeneralisationBound}. */
export interface RademacherBoundOptions {
  /** Whether `complexity` is the empirical complexity on the sample (default true) or the expected complexity. */
  readonly empirical?: boolean
}

/**
 * The Rademacher generalisation bound (Bartlett and Mendelson 2002; Mohri et al., theorem 3.3) for a class of functions
 * with values in [0, 1] (the loss class): with probability at least 1 - delta every member has risk at most its training
 * risk plus 2 R + sqrt(ln(1/delta) / (2n)) for the expected complexity R, or plus 2 R-hat + 3 sqrt(ln(2/delta) / (2n)) for the
 * empirical one. For a 0-1 loss of a {-1, +1} classifier the complexity of the loss class is half that of the classifier
 * class. The returned value is the excess term.
 *
 * @param complexity - The Rademacher complexity of the loss class.
 * @param n - The sample size.
 * @param delta - The failure probability.
 * @param options - Whether the complexity is empirical.
 * @returns The bound on the generalisation gap.
 * @concept learning-theory
 * @example
 * ```ts
 * import { rademacherGeneralisationBound } from 'mdml-engine/learn'
 * rademacherGeneralisationBound(0.05, 1000, 0.05) // => 0.2288
 * ```
 */
export function rademacherGeneralisationBound(
  complexity: number,
  n: number,
  delta: number,
  options: RademacherBoundOptions = {},
): number {
  finite('rademacherGeneralisationBound', 'complexity', complexity)
  finite('rademacherGeneralisationBound', 'n', n, 0, true)
  unit('rademacherGeneralisationBound', delta)
  return options.empirical === false
    ? 2 * complexity + Math.sqrt(Math.log(1 / delta) / (2 * n))
    : 2 * complexity + 3 * Math.sqrt(Math.log(2 / delta) / (2 * n))
}

/** Options of {@link marginBound}. */
export interface MarginBoundOptions {
  /** The margin gamma (default 1: the margins are those of a classifier scaled to unit margin). */
  readonly gamma?: number
}

/**
 * The margin bound for a norm-bounded linear classifier (Bartlett and Mendelson 2002; Koltchinskii and Panchenko 2002;
 * Mohri et al., theorem 5.8): with probability at least 1 - delta, for the classifier sign(w . x) with |w|_2 <= B on
 * features with |x|_2 <= R, the error is at most the fraction of training points with margin y w . x below gamma plus
 * 2 B R / (gamma sqrt(n)) + 3 sqrt(ln(2 / delta) / (2n)). The ramp loss at scale gamma is 1/gamma-Lipschitz, which gives the
 * complexity term.
 *
 * @param margins - The training margins y_i (w . x_i), one per point.
 * @param norm - B, the bound on |w|_2.
 * @param radius - R, the bound on the features' norms.
 * @param delta - The failure probability.
 * @param options - The margin gamma.
 * @returns The bound on the error probability, the fraction below the margin and the complexity term.
 * @concept learning-theory
 * @example
 * ```ts
 * import { marginBound } from 'mdml-engine/learn'
 * marginBound([2, 3, 2.5, 4], 1, 1, 0.05, { gamma: 1 }).fraction // => 0
 * ```
 */
export function marginBound(
  margins: ArrayLike<number>,
  norm: number,
  radius: number,
  delta: number,
  options: MarginBoundOptions = {},
): { readonly bound: number; readonly fraction: number; readonly complexity: number } {
  const gamma = options.gamma ?? 1
  finite('marginBound', 'gamma', gamma, 0, true)
  unit('marginBound', delta)
  const n = margins.length
  if (n === 0) throw new MdmlError('value', 'marginBound: needs at least one margin')
  let below = 0
  for (let i = 0; i < n; i++) if ((margins[i] ?? 0) < gamma) below++
  const fraction = below / n
  const complexity = (2 * norm * radius) / (gamma * Math.sqrt(n)) + 3 * Math.sqrt(Math.log(2 / delta) / (2 * n))
  return { bound: Math.min(1, fraction + complexity), fraction, complexity }
}

/** Assumptions of the stability bounds of regularised least squares. */
export interface RidgeStabilityOptions {
  /** R, a bound on the norm of the features. */
  readonly radius: number
  /** Y, a bound on the magnitude of the responses. */
  readonly labelBound: number
}

/**
 * The uniform stability of ridge regression, in closed form. The algorithm minimises (1/n) sum (w . x_i - y_i)^2 +
 * lambda |w|^2 with |x| <= R and |y| <= Y. The objective is 2 lambda-strongly convex, and the minimiser has |w|
 * <= Y / sqrt(lambda) (compare with w = 0), on which the squared loss is G-Lipschitz in w with G = 2 R (R Y / sqrt(lambda) + Y).
 * Replacing one training example moves the minimiser by at most G / (lambda n) and so changes the loss at any point by at
 * most beta = G^2 / (lambda n): the stability is O(1 / (lambda n)).
 *
 * @param n - The sample size.
 * @param lambda - The ridge penalty (on the mean squared error scale above).
 * @param options - R and Y.
 * @returns beta, the uniform stability.
 * @concept learning-theory
 * @example
 * ```ts
 * import { ridgeUniformStability } from 'mdml-engine/learn'
 * ridgeUniformStability(100, 1, { radius: 1, labelBound: 1 }) // => 0.16
 * ```
 */
export function ridgeUniformStability(n: number, lambda: number, options: RidgeStabilityOptions): number {
  finite('ridgeUniformStability', 'n', n, 0, true)
  finite('ridgeUniformStability', 'lambda', lambda, 0, true)
  const { radius: r, labelBound: y } = options
  const g = 2 * r * ((r * y) / Math.sqrt(lambda) + y)
  return (g * g) / (lambda * n)
}

/**
 * The generalisation bound from uniform stability (Bousquet and Elisseeff 2002, theorem 12): for an algorithm with uniform
 * stability beta and a loss bounded by M, with probability at least 1 - delta the risk is at most the training risk
 * plus 2 beta + (4 n beta + M) sqrt(ln(1/delta) / (2n)). It is non-vacuous only when beta decays faster than 1 / sqrt(n).
 * The returned value is the excess term.
 *
 * @param beta - The uniform stability.
 * @param n - The sample size.
 * @param delta - The failure probability.
 * @param lossBound - M, the bound on the loss (default 1).
 * @returns The bound on the generalisation gap.
 * @concept learning-theory
 * @example
 * ```ts
 * import { stabilityBound } from 'mdml-engine/learn'
 * stabilityBound(0.0001, 1000, 0.05) // => 0.0544
 * ```
 */
export function stabilityBound(beta: number, n: number, delta: number, lossBound = 1): number {
  finite('stabilityBound', 'beta', beta)
  finite('stabilityBound', 'n', n, 0, true)
  unit('stabilityBound', delta)
  return 2 * beta + (4 * n * beta + lossBound) * Math.sqrt(Math.log(1 / delta) / (2 * n))
}

/** A learner for the stability measurement: fit on (x, y) and return the predictor. */
export type Learner = (x: Float64Array[], y: Float64Array) => (x: Float64Array) => number

/**
 * The empirical uniform stability of a learner, measured by replacement: for each training example in turn, replace it
 * with a fresh draw, refit, and record the largest change in the loss over a set of test points; the maximum over the
 * replaced positions is a lower bound on the uniform stability (it is sup over datasets and replacements in theory).
 *
 * @param fit - The learner.
 * @param train - The training features and responses.
 * @param replacements - The examples to swap in (one per position tried).
 * @param test - The test points the change is measured on.
 * @param loss - The loss of a prediction (default the squared error).
 * @returns The largest change in the loss found.
 * @concept learning-theory
 * @example
 * ```ts
 * import { empiricalStability } from 'mdml-engine/learn'
 * const meanFit = (x: Float64Array[], y: Float64Array) => { const m = y.reduce((a, b) => a + b, 0) / y.length; return () => m }
 * const train = { x: [[0], [1], [2], [3]].map((r) => Float64Array.from(r)), y: Float64Array.from([0, 1, 2, 3]) }
 * const swap = [{ x: Float64Array.from([9]), y: 10 }]
 * empiricalStability(meanFit, train, swap, { x: [Float64Array.from([0])], y: Float64Array.from([1.5]) }) // => 6.25
 * ```
 */
export function empiricalStability(
  fit: Learner,
  train: { readonly x: Float64Array[]; readonly y: Float64Array },
  replacements: readonly { readonly x: Float64Array; readonly y: number }[],
  test: { readonly x: Float64Array[]; readonly y: Float64Array },
  loss: (prediction: number, y: number) => number = (p, y) => (p - y) ** 2,
): number {
  const base = fit(train.x, train.y)
  const baseLoss = test.x.map((x, i) => loss(base(x), test.y[i] ?? 0))
  let worst = 0
  for (let j = 0; j < train.x.length; j++) {
    const swap = replacements[j % replacements.length]
    if (swap === undefined) continue
    const xs = train.x.map((x, i) => (i === j ? swap.x : x))
    const ys = Float64Array.from(train.y, (v, i) => (i === j ? swap.y : v))
    const g = fit(xs, ys)
    test.x.forEach((x, i) => {
      worst = Math.max(worst, Math.abs(loss(g(x), test.y[i] ?? 0) - (baseLoss[i] ?? 0)))
    })
  }
  return worst
}

/**
 * The binary KL divergence kl(q || p) = q ln(q/p) + (1-q) ln((1-q)/(1-p)).
 *
 * @param q - The first Bernoulli parameter.
 * @param p - The second.
 * @returns The divergence in nats (infinite when p is 0 or 1 and q is not).
 * @concept learning-theory
 * @example
 * ```ts
 * import { klBernoulli } from 'mdml-engine/learn'
 * klBernoulli(0.5, 0.5) // => 0
 * ```
 */
export function klBernoulli(q: number, p: number): number {
  const term = (a: number, b: number): number =>
    a === 0 ? 0 : b === 0 ? Number.POSITIVE_INFINITY : a * Math.log(a / b)
  return term(q, p) + term(1 - q, 1 - p)
}

/**
 * The upper inverse of the binary KL, sup {p >= q : kl(q || p) <= c}, found by root finding (Seeger's inversion of the
 * Chernoff bound); the largest risk consistent with an empirical risk q and a budget c.
 *
 * @param q - The empirical risk.
 * @param c - The KL budget.
 * @returns The largest p with kl(q || p) <= c.
 * @concept learning-theory
 * @example
 * ```ts
 * import { klInverse } from 'mdml-engine/learn'
 * klInverse(0.1, 0.05) // => 0.2201
 * ```
 */
export function klInverse(q: number, c: number): number {
  if (!(q >= 0 && q <= 1)) throw new MdmlError('value', 'klInverse: q must be in [0, 1]')
  finite('klInverse', 'c', c)
  if (q >= 1 || c === Number.POSITIVE_INFINITY) return 1
  if (c === 0) return q
  const f = (p: number): number => klBernoulli(q, p) - c
  if (f(1 - 1e-15) < 0) return 1
  return brentq(f, q, 1 - 1e-15, { xtol: 1e-14 }).root
}

/**
 * The lower inverse of the binary KL, inf {p <= q : kl(q || p) <= c}.
 *
 * @param q - The empirical value.
 * @param c - The KL budget.
 * @returns The smallest p with kl(q || p) <= c.
 * @concept learning-theory
 * @example
 * ```ts
 * import { klInverseLower } from 'mdml-engine/learn'
 * klInverseLower(0.5, 0.05) // => 0.3458
 * ```
 */
export function klInverseLower(q: number, c: number): number {
  return 1 - klInverse(1 - q, c)
}

/**
 * The KL divergence KL(Q || P) between Gaussian posterior and prior over the weights with diagonal covariances: sum_j
 * [ln(sP_j / sQ_j) + (sQ_j^2 + (mQ_j - mP_j)^2) / (2 sP_j^2) - 1/2], the complexity term of PAC-Bayes bounds for a stochastic
 * linear predictor. A scalar standard deviation means an isotropic covariance.
 *
 * @param meanQ - The posterior mean.
 * @param sdQ - The posterior standard deviations (one value, or one per weight).
 * @param meanP - The prior mean.
 * @param sdP - The prior standard deviations.
 * @returns The divergence in nats.
 * @concept learning-theory
 * @example
 * ```ts
 * import { gaussianKl } from 'mdml-engine/learn'
 * gaussianKl([1, 0], 1, [0, 0], 1) // => 0.5
 * ```
 */
export function gaussianKl(
  meanQ: ArrayLike<number>,
  sdQ: number | ArrayLike<number>,
  meanP: ArrayLike<number>,
  sdP: number | ArrayLike<number>,
): number {
  const d = meanQ.length
  if (meanP.length !== d) throw new MdmlError('shape', 'gaussianKl: means differ in length')
  const at = (s: number | ArrayLike<number>, j: number): number => (typeof s === 'number' ? s : (s[j] ?? 0))
  let kl = 0
  for (let j = 0; j < d; j++) {
    const sq = at(sdQ, j)
    const sp = at(sdP, j)
    if (!(sq > 0 && sp > 0)) throw new MdmlError('value', 'gaussianKl: standard deviations must be positive')
    kl += Math.log(sp / sq) + (sq * sq + ((meanQ[j] ?? 0) - (meanP[j] ?? 0)) ** 2) / (2 * sp * sp) - 0.5
  }
  return kl
}

/**
 * The Gibbs risk (expected 0-1 error) of a Gaussian-perturbed linear classifier in closed form: for w ~ N(mu, s^2 I), the
 * margin y w . x is normal with mean y mu . x and standard deviation s |x|, so the error probability on an example is
 * Phi(-y mu . x / (s |x|)) (Langford and Shawe-Taylor 2002). Averaged over the sample.
 *
 * @param mean - The posterior mean mu.
 * @param sd - The posterior standard deviation s.
 * @param x - The features (points x features).
 * @param y - Labels in {-1, +1}.
 * @returns The mean error probability.
 * @concept learning-theory
 * @example
 * ```ts
 * import { gaussianGibbsRisk } from 'mdml-engine/learn'
 * gaussianGibbsRisk([1], 1, [[1]], [1]) // => 0.1587
 * ```
 */
export function gaussianGibbsRisk(mean: ArrayLike<number>, sd: number, x: TensorLike, y: ArrayLike<number>): number {
  const m = matrixOf('gaussianGibbsRisk', x)
  if (m.rows !== y.length || m.cols !== mean.length) throw new MdmlError('shape', 'gaussianGibbsRisk: shapes disagree')
  finite('gaussianGibbsRisk', 'sd', sd, 0, true)
  let total = 0
  for (let i = 0; i < m.rows; i++) {
    let dot = 0
    let norm2 = 0
    for (let j = 0; j < m.cols; j++) {
      const v = m.data[i * m.cols + j] ?? 0
      dot += v * (mean[j] ?? 0)
      norm2 += v * v
    }
    total += item(ndtr((-(y[i] ?? 0) * dot) / (sd * Math.sqrt(norm2))))
  }
  return total / m.rows
}

/**
 * McAllester's PAC-Bayes bound (McAllester 1999, in Maurer's form): with probability at least 1 - delta over the sample,
 * for every posterior Q simultaneously, the Gibbs risk is at most the empirical Gibbs risk plus sqrt((KL(Q || P) + ln(2 sqrt(n) /
 * delta)) / (2n)); the prior P must not depend on the sample, losses in [0, 1].
 *
 * @param empiricalRisk - The empirical Gibbs risk of Q.
 * @param kl - KL(Q || P).
 * @param n - The sample size.
 * @param delta - The failure probability.
 * @returns The bound on the Gibbs risk.
 * @concept learning-theory
 * @example
 * ```ts
 * import { mcAllesterBound } from 'mdml-engine/learn'
 * mcAllesterBound(0.1, 5, 1000, 0.05) // => 0.1779
 * ```
 */
export function mcAllesterBound(empiricalRisk: number, kl: number, n: number, delta: number): number {
  finite('mcAllesterBound', 'kl', kl)
  finite('mcAllesterBound', 'n', n, 0, true)
  unit('mcAllesterBound', delta)
  return Math.min(1, empiricalRisk + Math.sqrt((kl + Math.log((2 * Math.sqrt(n)) / delta)) / (2 * n)))
}

/**
 * Seeger's (Langford-Seeger) PAC-Bayes bound: kl(empirical Gibbs risk || Gibbs risk) <= (KL(Q || P) + ln(2 sqrt(n) / delta)) /
 * n with probability at least 1 - delta, inverted by root finding ({@link klInverse}); tighter than McAllester's form,
 * especially at small risk.
 *
 * @param empiricalRisk - The empirical Gibbs risk of Q.
 * @param kl - KL(Q || P).
 * @param n - The sample size.
 * @param delta - The failure probability.
 * @returns The bound on the Gibbs risk.
 * @concept learning-theory
 * @example
 * ```ts
 * import { seegerBound } from 'mdml-engine/learn'
 * seegerBound(0.1, 5, 1000, 0.05) // => 0.1531
 * ```
 */
export function seegerBound(empiricalRisk: number, kl: number, n: number, delta: number): number {
  finite('seegerBound', 'kl', kl)
  finite('seegerBound', 'n', n, 0, true)
  unit('seegerBound', delta)
  return klInverse(empiricalRisk, (kl + Math.log((2 * Math.sqrt(n)) / delta)) / n)
}

/**
 * Catoni's PAC-Bayes bound (Catoni 2007; Germain et al. 2009): for a fixed c > 0, with probability at least 1 - delta the
 * Gibbs risk is at most (1 - exp(-c R-hat - (KL(Q || P) + ln(1/delta)) / n)) / (1 - exp(-c)). With `optimise` the constant c
 * is chosen from a grid of ten values by minimising the bound, paying a union bound over the grid (delta / 10); the
 * bound is monotone in R-hat and KL and is the form PAC-Bayes training objectives minimise.
 *
 * @param empiricalRisk - The empirical Gibbs risk of Q.
 * @param kl - KL(Q || P).
 * @param n - The sample size.
 * @param delta - The failure probability.
 * @param c - The constant (default: optimised over a grid).
 * @returns The bound on the Gibbs risk.
 * @concept learning-theory
 * @example
 * ```ts
 * import { catoniBound } from 'mdml-engine/learn'
 * catoniBound(0.1, 5, 1000, 0.05, 1) // => 0.1619
 * ```
 */
export function catoniBound(empiricalRisk: number, kl: number, n: number, delta: number, c?: number): number {
  finite('catoniBound', 'kl', kl)
  finite('catoniBound', 'n', n, 0, true)
  unit('catoniBound', delta)
  const at = (cc: number, d: number): number =>
    Math.min(1, (1 - Math.exp(-cc * empiricalRisk - (kl + Math.log(1 / d)) / n)) / (1 - Math.exp(-cc)))
  if (c !== undefined) {
    finite('catoniBound', 'c', c, 0, true)
    return at(c, delta)
  }
  const grid = Array.from({ length: 10 }, (_, i) => 2 ** (i - 3))
  return Math.min(...grid.map((cc) => at(cc, delta / grid.length)))
}

/**
 * The regret bound of Hedge (exponential weights) over N experts with losses in [0, 1] after T rounds: ln N / eta + eta T /
 * 8 for the learning rate eta, which is at its smallest, sqrt(T ln N / 2), at eta = sqrt(8 ln N / T) (Freund and
 * Schapire 1997; Cesa-Bianchi and Lugosi 2006, theorem 2.2).
 *
 * @param T - The number of rounds.
 * @param N - The number of experts.
 * @param eta - The learning rate (default the optimal one for T).
 * @returns The bound on the regret against the best expert.
 * @concept learning-theory
 * @example
 * ```ts
 * import { hedgeRegretBound } from 'mdml-engine/learn'
 * hedgeRegretBound(1000, 10) // => 33.9307
 * ```
 */
export function hedgeRegretBound(T: number, N: number, eta?: number): number {
  finite('hedgeRegretBound', 'T', T, 0, true)
  if (!(N >= 2)) throw new MdmlError('value', 'hedgeRegretBound: needs at least 2 experts')
  const e = eta ?? Math.sqrt((8 * Math.log(N)) / T)
  finite('hedgeRegretBound', 'eta', e, 0, true)
  return Math.log(N) / e + (e * T) / 8
}

/**
 * The regret bound of online (projected) gradient descent on convex losses (Zinkevich 2003): with the feasible set of
 * diameter D, gradients of norm at most G and the step eta, regret <= D^2 / (2 eta) + eta G^2 T / 2, which is D G sqrt(T) at
 * eta = D / (G sqrt(T)); with the anytime step eta_t = D / (G sqrt(t)) it is (3/2) D G sqrt(T).
 *
 * @param T - The number of rounds.
 * @param D - The diameter of the feasible set.
 * @param G - The bound on the gradient norms.
 * @param eta - The constant step (default the optimal one for T).
 * @returns The bound on the regret against the best fixed point.
 * @concept learning-theory
 * @example
 * ```ts
 * import { ogdRegretBound } from 'mdml-engine/learn'
 * ogdRegretBound(10000, 2, 1) // => 200
 * ```
 */
export function ogdRegretBound(T: number, D: number, G: number, eta?: number): number {
  finite('ogdRegretBound', 'T', T, 0, true)
  finite('ogdRegretBound', 'D', D, 0, true)
  finite('ogdRegretBound', 'G', G, 0, true)
  const e = eta ?? D / (G * Math.sqrt(T))
  return (D * D) / (2 * e) + (e * G * G * T) / 2
}

/**
 * The information-theoretic generalisation bound (Xu and Raginsky 2017): for a loss that is sigma-sub-Gaussian, the expected
 * generalisation gap is at most sqrt(2 sigma^2 I(S; W) / n), with I the mutual information between the training sample and
 * the learned hypothesis, in nats. For losses in [0, 1], sigma = 1/2.
 *
 * @param mutualInformation - I(S; W) in nats.
 * @param n - The sample size.
 * @param sigma - The sub-Gaussian scale of the loss (default 1/2, for losses in [0, 1]).
 * @returns The bound on the expected gap.
 * @concept learning-theory
 * @example
 * ```ts
 * import { mutualInformationBound } from 'mdml-engine/learn'
 * mutualInformationBound(10, 1000) // => 0.0707
 * ```
 */
export function mutualInformationBound(mutualInformation: number, n: number, sigma = 0.5): number {
  finite('mutualInformationBound', 'mutualInformation', mutualInformation)
  finite('mutualInformationBound', 'n', n, 0, true)
  return Math.sqrt((2 * sigma * sigma * mutualInformation) / n)
}

/** The bias-variance decomposition of a set of fits, averaged over the test points. */
export interface BiasVariance {
  /** The average squared bias (the mean fit against the truth). */
  readonly bias2: number
  /** The average variance of the fits around their mean. */
  readonly variance: number
  /** The noise variance given (0 if none), so that `expected` is the expected squared error against noisy targets. */
  readonly noise: number
  /** bias2 + variance + noise. */
  readonly expected: number
  /** The squared bias at each test point. */
  readonly bias2At: readonly number[]
  /** The variance at each test point. */
  readonly varianceAt: readonly number[]
}

/**
 * The bias-variance decomposition of squared loss from a set of fits: with R fits (each trained on its own sample) giving
 * predictions f_r(x) at the test points and the truth f(x), bias^2(x) = (mean_r f_r(x) - f(x))^2 and variance(x) = the variance of
 * f_r(x) over r (with the 1/R divisor, so that expected = bias^2 + variance is the exact mean squared error of the fits against
 * the truth), and the noise variance is added to give the expected squared error against noisy targets.
 *
 * @param predictions - R x m predictions of R fits at m test points.
 * @param truth - The m values of the true regression function.
 * @param noiseVariance - The variance of the target noise (default 0).
 * @returns The decomposition.
 * @concept learning-theory
 * @example
 * ```ts
 * import { biasVariance } from 'mdml-engine/learn'
 * biasVariance([[1, 2], [3, 2]], [1, 1]).variance // => 0.5
 * ```
 */
export function biasVariance(predictions: TensorLike, truth: ArrayLike<number>, noiseVariance = 0): BiasVariance {
  const p = matrixOf('biasVariance', predictions)
  if (p.cols !== truth.length) throw new MdmlError('shape', 'biasVariance: predictions and truth disagree in length')
  const bias2At: number[] = []
  const varianceAt: number[] = []
  for (let j = 0; j < p.cols; j++) {
    let mean = 0
    for (let r = 0; r < p.rows; r++) mean += (p.data[r * p.cols + j] ?? 0) / p.rows
    let v = 0
    for (let r = 0; r < p.rows; r++) v += ((p.data[r * p.cols + j] ?? 0) - mean) ** 2 / p.rows
    bias2At.push((mean - (truth[j] ?? 0)) ** 2)
    varianceAt.push(v)
  }
  const avg = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
  const bias2 = avg(bias2At)
  const variance = avg(varianceAt)
  return { bias2, variance, noise: noiseVariance, expected: bias2 + variance + noiseVariance, bias2At, varianceAt }
}

/** The 0-1 loss decomposition of Domingos (2000) for two classes. */
export interface BiasVarianceZeroOne {
  /** Average bias: the fraction of test points where the main (modal) prediction differs from the optimal one. */
  readonly bias: number
  /** Average variance: how often a fit disagrees with the main prediction. */
  readonly variance: number
  /** Variance on the unbiased points (adds to the loss). */
  readonly unbiasedVariance: number
  /** Variance on the biased points (subtracts from the loss). */
  readonly biasedVariance: number
  /** The net variance effect, unbiasedVariance - biasedVariance. */
  readonly netVariance: number
  /** The average noise N(x) = P(t != y*(x)). */
  readonly noise: number
  /** The expected 0-1 error against noisy targets, c1 N + B + c2 V averaged, with Domingos' coefficients. */
  readonly expected: number
}

/**
 * Domingos' bias-variance decomposition of the 0-1 loss (2000) for two classes, from R fits' predicted labels at m test
 * points. With the main prediction y_m(x) the modal label over the fits and y*(x) the Bayes-optimal label: bias B(x) = 1[y_m != y*],
 * variance V(x) = P(f_r(x) != y_m), noise N(x) = P(t != y*), and the expected 0-1 loss is (2 P(f = y*) - 1) N(x) + B(x) + (1 - 2B(x))
 * V(x): variance adds on unbiased points and subtracts on biased ones.
 *
 * @param predictions - R x m predicted labels (any two values) of R fits.
 * @param optimal - The m Bayes-optimal labels y*.
 * @param noise - The noise N(x) = P(target != y*) at each point, or one value for all (default 0).
 * @returns The decomposition.
 * @concept learning-theory
 * @example
 * ```ts
 * import { biasVarianceZeroOne } from 'mdml-engine/learn'
 * biasVarianceZeroOne([[1, 0], [1, 1], [0, 1]], [1, 1]).variance // => 0.3333
 * ```
 */
export function biasVarianceZeroOne(
  predictions: TensorLike,
  optimal: ArrayLike<number>,
  noise: number | ArrayLike<number> = 0,
): BiasVarianceZeroOne {
  const p = matrixOf('biasVarianceZeroOne', predictions)
  if (p.cols !== optimal.length)
    throw new MdmlError('shape', 'biasVarianceZeroOne: predictions and optimal labels disagree in length')
  let bias = 0
  let variance = 0
  let unbiased = 0
  let biased = 0
  let noiseTotal = 0
  let expected = 0
  for (let j = 0; j < p.cols; j++) {
    const counts = new Map<number, number>()
    for (let r = 0; r < p.rows; r++) {
      const v = p.data[r * p.cols + j] ?? 0
      counts.set(v, (counts.get(v) ?? 0) + 1)
    }
    if (counts.size > 2) throw new MdmlError('value', 'biasVarianceZeroOne: the decomposition is for two classes')
    let main = Number.NaN
    let best = -1
    for (const [label, c] of [...counts].sort((a, b) => a[0] - b[0])) {
      if (c > best) {
        best = c
        main = label
      }
    }
    const ystar = optimal[j] ?? 0
    const b = main === ystar ? 0 : 1
    const v = 1 - best / p.rows
    const n = typeof noise === 'number' ? noise : (noise[j] ?? 0)
    const pCorrect = (counts.get(ystar) ?? 0) / p.rows
    const c1 = 2 * pCorrect - 1
    bias += b / p.cols
    variance += v / p.cols
    if (b === 0) unbiased += v / p.cols
    else biased += v / p.cols
    noiseTotal += n / p.cols
    expected += (c1 * n + b + (1 - 2 * b) * v) / p.cols
  }
  return {
    bias,
    variance,
    unbiasedVariance: unbiased,
    biasedVariance: biased,
    netVariance: unbiased - biased,
    noise: noiseTotal,
    expected,
  }
}

/** A training set drawn for one repetition of {@link biasVarianceAlgorithm}. */
export interface TrainingDraw {
  readonly x: Float64Array[]
  readonly y: Float64Array
}

/** Options of {@link biasVarianceAlgorithm}. */
export interface BiasVarianceStudy {
  /** Draw a training set from the key. */
  readonly sample: (key: Key) => TrainingDraw
  /** Fit a predictor. */
  readonly fit: Learner
  /** The test points. */
  readonly testX: readonly Float64Array[]
  /** The true regression function (or the Bayes-optimal labels) at the test points. */
  readonly truth: ArrayLike<number>
  /** The number of resampled training sets (default 100). */
  readonly repeats?: number
  readonly noiseVariance?: number
  readonly key?: Key
}

/** State of {@link biasVarianceAlgorithm}: the predictions of each fit so far (one row per repetition). */
export interface BiasVarianceState {
  readonly predictions: readonly (readonly number[])[]
}

/**
 * The Monte Carlo bias-variance study as an algorithm: each step draws a fresh training set, fits the learner and records
 * its predictions at the test points; the decomposition of the rows so far ({@link biasVariance}) is what a page plots as the
 * repetitions accumulate.
 *
 * @param study - The sampler, the learner, the test points, the truth and the number of repetitions.
 * @returns The algorithm over {@link BiasVarianceState}.
 * @concept learning-theory
 * @example
 * ```ts
 * import { biasVarianceAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = biasVarianceAlgorithm({
 *   sample: () => ({ x: [Float64Array.from([0])], y: Float64Array.from([1]) }),
 *   fit: (_x, y) => () => y[0] ?? 0,
 *   testX: [Float64Array.from([0])],
 *   truth: [1],
 *   repeats: 2,
 * })
 * runAlgorithm(alg, { maxSteps: 2 }).state.predictions // => [[1], [1]]
 * ```
 */
export function biasVarianceAlgorithm(study: BiasVarianceStudy): Algorithm<BiasVarianceState> {
  const repeats = study.repeats ?? 100
  return {
    name: 'bias-variance',
    config: { repeats, testPoints: study.testX.length } as Json,
    schema: channels({
      metrics: {
        bias2: { description: 'average squared bias over the fits so far' },
        variance: { description: 'average variance over the fits so far' },
        expected: { description: 'bias^2 + variance + noise' },
      },
      tensors: { fit: { role: 'output', description: 'the predictions of the latest fit at the test points' } },
    }),
    init: () => ({ predictions: [] }),
    step: (s, ctx) => {
      const train = study.sample(ctx.key)
      const predict = study.fit(train.x, train.y)
      return { predictions: [...s.predictions, study.testX.map((x) => predict(x))] }
    },
    done: (s) => (s.predictions.length >= repeats ? 'completed' : undefined),
    observe: (s) => {
      const bv = biasVariance(s.predictions as number[][], study.truth, study.noiseVariance ?? 0)
      return {
        metrics: { bias2: bv.bias2, variance: bv.variance, expected: bv.expected },
        tensors: { fit: { role: 'output', value: s.predictions[s.predictions.length - 1] ?? [] } },
      }
    },
    serialise: (s) => ({ predictions: s.predictions.map((r) => [...r]) }),
    deserialise: (j) => j as unknown as BiasVarianceState,
  }
}

/**
 * Run the Monte Carlo bias-variance study to the end and decompose the fits ({@link biasVarianceAlgorithm}).
 *
 * @param study - The study.
 * @returns The decomposition of squared loss over the resampled training sets.
 * @concept learning-theory
 * @example
 * ```ts
 * import { monteCarloBiasVariance } from 'mdml-engine/learn'
 * const r = monteCarloBiasVariance({
 *   sample: () => ({ x: [Float64Array.from([0])], y: Float64Array.from([1]) }),
 *   fit: (_x, y) => () => y[0] ?? 0,
 *   testX: [Float64Array.from([0])],
 *   truth: [0],
 *   repeats: 3,
 * })
 * r.bias2 // => 1
 * ```
 */
export function monteCarloBiasVariance(study: BiasVarianceStudy): BiasVariance {
  const repeats = study.repeats ?? 100
  const result = runAlgorithm(biasVarianceAlgorithm(study), { maxSteps: repeats, key: study.key ?? makeKey(0) })
  return biasVariance(result.state.predictions as number[][], study.truth, study.noiseVariance ?? 0)
}

/** State of {@link degreesOfFreedomAlgorithm}: the sum over draws of sum_i yhat_i eps_i. */
export interface DegreesOfFreedomState {
  /** The per-draw values (1/sigma) sum_i yhat_i eps_i. */
  readonly values: readonly number[]
}

/**
 * Efron's covariance definition of the degrees of freedom, df = (1/sigma^2) sum_i Cov(yhat_i, y_i), by Monte Carlo as an
 * algorithm with one draw per step. With y = mu + sigma eps for a known mean mu, Cov(yhat_i, y_i) = sigma E[yhat_i eps_i], so a draw
 * contributes (1/sigma) sum_i yhat_i eps_i, whose mean is the degrees of freedom (and 2 sigma^2 df is the optimism of the
 * training error). For a linear smoother yhat = S y it is trace(S).
 *
 * @param fit - The fitting procedure: responses in, fitted values out.
 * @param mu - The true mean of the responses.
 * @param sigma - The noise standard deviation.
 * @param options - Draws and key.
 * @returns The algorithm over {@link DegreesOfFreedomState}.
 * @concept learning-theory
 * @example
 * ```ts
 * import { degreesOfFreedomAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = degreesOfFreedomAlgorithm((y) => y, [0, 0, 0], 1, { draws: 2 })
 * runAlgorithm(alg, { maxSteps: 2 }).state.values.length // => 2
 * ```
 */
export function degreesOfFreedomAlgorithm(
  fit: (y: Float64Array) => ArrayLike<number>,
  mu: ArrayLike<number>,
  sigma: number,
  options: MonteCarloOptions = {},
): Algorithm<DegreesOfFreedomState> {
  finite('degreesOfFreedom', 'sigma', sigma, 0, true)
  const draws = options.draws ?? 500
  const n = mu.length
  return {
    name: 'degrees-of-freedom',
    config: { draws, n, sigma } as Json,
    schema: channels({ metrics: { df: { description: 'running estimate of the degrees of freedom' } } }),
    init: () => ({ values: [] }),
    step: (s, ctx) => {
      const eps = toFlat(normal(ctx.key, [n]))
      const y = Float64Array.from(eps, (e, i) => (mu[i] ?? 0) + sigma * e)
      const yhat = fit(y)
      let acc = 0
      for (let i = 0; i < n; i++) acc += (yhat[i] ?? 0) * (eps[i] ?? 0)
      return { values: [...s.values, acc / sigma] }
    },
    done: (s) => (s.values.length >= draws ? 'completed' : undefined),
    observe: (s) => ({ metrics: { df: s.values.reduce((a, b) => a + b, 0) / Math.max(1, s.values.length) } }),
    serialise: (s) => ({ values: [...s.values] }),
    deserialise: (j) => j as unknown as DegreesOfFreedomState,
  }
}

/**
 * The degrees of freedom of a fitting procedure by Efron's covariance definition, estimated by Monte Carlo
 * ({@link degreesOfFreedomAlgorithm} run to the end).
 *
 * @param fit - Responses in, fitted values out.
 * @param mu - The true mean of the responses.
 * @param sigma - The noise standard deviation.
 * @param options - Draws and key.
 * @returns The estimate and its standard error.
 * @concept learning-theory
 * @example
 * ```ts
 * import { effectiveDegreesOfFreedom } from 'mdml-engine/learn'
 * effectiveDegreesOfFreedom((y) => y, [0, 0, 0, 0], 1, { draws: 2000 }).estimate > 3.5 // => true
 * ```
 */
export function effectiveDegreesOfFreedom(
  fit: (y: Float64Array) => ArrayLike<number>,
  mu: ArrayLike<number>,
  sigma: number,
  options: MonteCarloOptions = {},
): { readonly estimate: number; readonly standardError: number } {
  const draws = options.draws ?? 500
  const r = runAlgorithm(degreesOfFreedomAlgorithm(fit, mu, sigma, options), {
    maxSteps: draws,
    key: options.key ?? makeKey(0),
  })
  const v = r.state.values
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  const variance = v.length > 1 ? v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1) : 0
  return { estimate: mean, standardError: Math.sqrt(variance / v.length) }
}

/**
 * The degrees of freedom of ridge regression in closed form: trace of the hat matrix X (X'X + lambda I)^-1 X' =
 * sum_j s_j^2 / (s_j^2 + lambda) over the singular values s_j of X. It falls from the rank of X at lambda = 0 to zero as
 * lambda grows. Pass either the design matrix or its singular values.
 *
 * @param x - The design matrix (n x p), or the singular values when `singular` is true.
 * @param lambda - The ridge penalty in |y - X w|^2 + lambda |w|^2.
 * @param singular - Whether `x` already holds the singular values.
 * @returns The degrees of freedom.
 * @concept learning-theory
 * @example
 * ```ts
 * import { ridgeDegreesOfFreedom } from 'mdml-engine/learn'
 * ridgeDegreesOfFreedom([2, 1], 1, true) // => 1.3
 * ```
 */
export function ridgeDegreesOfFreedom(x: TensorLike, lambda: number, singular = false): number {
  finite('ridgeDegreesOfFreedom', 'lambda', lambda)
  const s = singular ? toFlat(asTensor(x)) : toFlat(svd(x).s)
  return s.reduce((a, v) => a + (v * v) / (v * v + lambda), 0)
}

/**
 * The degrees of freedom of the lasso (Zou, Hastie and Tibshirani 2007): in expectation the number of nonzero
 * coefficients of the fit, an unbiased estimate for any design (an intercept, if fitted, adds one).
 *
 * @param coefficients - The fitted coefficients.
 * @param intercept - Whether an unpenalised intercept was fitted (default false).
 * @returns The estimate.
 * @concept learning-theory
 * @example
 * ```ts
 * import { lassoDegreesOfFreedom } from 'mdml-engine/learn'
 * lassoDegreesOfFreedom([0, 1.5, 0, -2]) // => 2
 * ```
 */
export function lassoDegreesOfFreedom(coefficients: ArrayLike<number>, intercept = false): number {
  let nz = intercept ? 1 : 0
  for (let j = 0; j < coefficients.length; j++) if ((coefficients[j] ?? 0) !== 0) nz++
  return nz
}

/**
 * The degrees of freedom of k-nearest-neighbour regression: the trace of the smoother, n / k when each point counts among its
 * own k neighbours (every row of the smoother has k entries 1/k, with the diagonal entry 1/k).
 *
 * @param n - The sample size.
 * @param k - The number of neighbours.
 * @returns n / k.
 * @concept learning-theory
 * @example
 * ```ts
 * import { knnDegreesOfFreedom } from 'mdml-engine/learn'
 * knnDegreesOfFreedom(100, 5) // => 20
 * ```
 */
export function knnDegreesOfFreedom(n: number, k: number): number {
  finite('knnDegreesOfFreedom', 'n', n, 0, true)
  finite('knnDegreesOfFreedom', 'k', k, 0, true)
  return n / k
}

/**
 * The covariance penalty 2 sigma^2 df / n: the expected excess of test error over training error (the optimism) of a fit
 * with df degrees of freedom on n observations with noise variance sigma^2 (Efron 1986); AIC and Mallows' Cp make it explicit.
 *
 * @param sigma - The noise standard deviation.
 * @param df - The degrees of freedom.
 * @param n - The number of observations.
 * @returns The optimism.
 * @concept learning-theory
 * @example
 * ```ts
 * import { covariancePenalty } from 'mdml-engine/learn'
 * covariancePenalty(1, 5, 50) // => 0.2
 * ```
 */
export function covariancePenalty(sigma: number, df: number, n: number): number {
  finite('covariancePenalty', 'n', n, 0, true)
  return (2 * sigma * sigma * df) / n
}
