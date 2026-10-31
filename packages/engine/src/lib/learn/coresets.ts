/**
 * Coreset construction and weighted subset selection: Gonzalez's greedy k-center traversal (`kCenterGreedyAlgorithm`,
 * `kCenterGreedy`), Welling's kernel herding in RKHS (`herdingAlgorithm`, `kernelHerding`), statistical leverage score
 * and k-means sensitivity importance sampling (`leverageScoreCoreset`, `sensitivitySampling`), and Campbell and
 * Broderick's Frank-Wolfe Hilbert Bayesian coresets (`bayesianCoresetAlgorithm`, `bayesianCoreset`).
 */

import { MdmlError } from '../base/index.ts'
import { ddot, dnrm2, svd } from '../numerics/linalg/index.ts'
import { categorical, foldIn, type Key } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { keyFor, rbfGram, runOptionsFor, squaredDistances } from './_shared.ts'
import type { FitOptions } from './estimator.ts'
import { fromView, integerOption, matrixView, takeRows } from './validation.ts'

/** Options of {@link kCenterGreedyAlgorithm} and {@link kCenterGreedy}. */
export interface KCenterOptions extends FitOptions {
  /** Initial row index when `initialIndices` is empty (default 0). */
  readonly seedIndex?: number
  /** Pre-selected row indices already in the centre set. */
  readonly initialIndices?: readonly number[]
  /** Fixed centres outside `x` (already labelled rows, say) `[m, d]`: covered first, never returned among the picks. */
  readonly centres?: TensorLike
}

/** Serialisable state of {@link kCenterGreedyAlgorithm}. */
export interface KCenterState {
  readonly selected: readonly number[]
  readonly minDistances: Float64Array
  readonly radius: number
  readonly radii: readonly number[]
}

/** Result of {@link kCenterGreedy}. */
export interface KCenterResult {
  /** Selected row indices in greedy traversal order. */
  readonly indices: readonly number[]
  /** Final Euclidean distance from each row to its nearest selected centre. */
  readonly distances: readonly number[]
  /** Final covering radius `max_i distances[i]`. */
  readonly radius: number
  /** Covering radius after each step. */
  readonly radii: readonly number[]
}

/**
 * Gonzalez's farthest-first k-center traversal as a stepped {@link Algorithm} (Gonzalez 1985; Sener and Savarese 2018):
 * each step picks the point farthest from the current coreset and updates every point's distance to its nearest centre,
 * logging `coreset/radius` and `coreset/selected`.
 *
 * @param x - Data matrix `[n, d]`.
 * @param size - Number of coreset points to select.
 * @param options - Optional `seedIndex`, `initialIndices` and trace options.
 * @returns Stepped {@link Algorithm} over {@link KCenterState}.
 * @concept coresets
 * @example
 * ```ts
 * import { kCenterGreedyAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = kCenterGreedyAlgorithm([[0], [2], [10]], 2, { seedIndex: 0 })
 * runAlgorithm(alg, { maxSteps: 2 }).state.selected // => [0, 2]
 * ```
 */
export function kCenterGreedyAlgorithm(
  x: TensorLike,
  size: number,
  options: KCenterOptions = {},
): Algorithm<KCenterState> {
  const m = matrixView(x, 'kCenterGreedy')
  const k = Math.min(integerOption(size, 'size', 'kCenterGreedy', 1), m.n)
  const seed = Math.min(Math.max(0, options.seedIndex ?? 0), m.n - 1)
  const fixed =
    options.centres === undefined ? undefined : matrixView(options.centres, 'kCenterGreedy', { nFeatures: m.d })
  const covered = (fixed?.n ?? 0) > 0 || (options.initialIndices?.length ?? 0) > 0

  const updateDistTo = (
    dist: Float64Array,
    c: Float64Array,
  ): { readonly nextDist: Float64Array; readonly radius: number } => {
    const nextDist = Float64Array.from(dist)
    let radius = 0
    for (let i = 0; i < m.n; i++) {
      let sq = 0
      for (let j = 0; j < m.d; j++) sq += ((m.data[i * m.d + j] ?? 0) - (c[j] ?? 0)) ** 2
      const d = Math.sqrt(sq)
      if (d < (nextDist[i] ?? Number.POSITIVE_INFINITY)) nextDist[i] = d
      if ((nextDist[i] ?? 0) > radius) radius = nextDist[i] ?? 0
    }
    return { nextDist, radius }
  }
  const updateDist = (dist: Float64Array, idx: number): { readonly nextDist: Float64Array; readonly radius: number } =>
    updateDistTo(dist, m.data.subarray(idx * m.d, (idx + 1) * m.d))

  return {
    name: 'k-center-greedy',
    config: { n: m.n, d: m.d, size: k, seedIndex: seed, centres: fixed?.n ?? 0 },
    schema: channels({
      metrics: {
        'coreset/radius': { description: 'covering radius of the selected centres', direction: 'lower' },
        'coreset/size': { description: 'number of centres selected' },
      },
      tensors: {
        'coreset/selected': { role: 'state', description: 'indices of the selected coreset points' },
      },
    }),
    init: () => {
      let dist: Float64Array = new Float64Array(m.n).fill(Number.POSITIVE_INFINITY)
      let radius = Number.POSITIVE_INFINITY
      const initIdx = options.initialIndices ?? []
      const selected: number[] = []
      const radii: number[] = []
      for (let r = 0; fixed !== undefined && r < fixed.n; r++) {
        const rowDist = updateDistTo(dist, fixed.data.subarray(r * fixed.d, (r + 1) * fixed.d))
        dist = rowDist.nextDist
        radius = rowDist.radius
      }
      for (const idx of initIdx) {
        if (idx >= 0 && idx < m.n && !selected.includes(idx)) {
          const r = updateDist(dist, idx)
          dist = r.nextDist
          radius = r.radius
          selected.push(idx)
          radii.push(radius)
        }
      }
      return { selected, minDistances: dist, radius, radii }
    },
    step: (s) => {
      let pick = -1
      if (s.selected.length === 0 && !covered) {
        pick = seed
      } else {
        const used = new Set(s.selected)
        let best = Number.NEGATIVE_INFINITY
        for (let i = 0; i < m.n; i++) {
          if (!used.has(i) && (s.minDistances[i] ?? 0) > best) {
            best = s.minDistances[i] ?? 0
            pick = i
          }
        }
      }
      if (pick < 0) return s
      const { nextDist, radius } = updateDist(s.minDistances, pick)
      return {
        selected: [...s.selected, pick],
        minDistances: nextDist,
        radius,
        radii: [...s.radii, radius],
      }
    },
    done: (s) => (s.selected.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        'coreset/radius': Number.isFinite(s.radius) ? s.radius : 0,
        'coreset/size': s.selected.length,
      },
      tensors: {
        'coreset/selected': {
          role: 'state',
          value: Float64Array.from(s.selected),
          shape: [s.selected.length],
        },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as KCenterState,
  }
}

/**
 * Gonzalez's farthest-first greedy k-center coreset (Gonzalez 1985; Sener and Savarese 2018): select `size` points one
 * at a time to minimise the maximum distance from any data point to its nearest selected point (a 2-approximation to
 * the optimal k-center covering radius).
 *
 * @param x - Data matrix `[n, d]`.
 * @param size - Number of points to select.
 * @param options - Optional `seedIndex`, `initialIndices` and trace options.
 * @returns Selected `indices`, final nearest-centre `distances`, covering `radius` and per-step `radii`.
 * @concept coresets
 * @example
 * ```ts
 * import { kCenterGreedy } from 'mdml-engine/learn'
 * const r = kCenterGreedy([[0], [1], [5], [10]], 3, { seedIndex: 0 })
 * r.indices // => [0, 3, 2]
 * r.radius // => 1
 * ```
 */
export function kCenterGreedy(x: TensorLike, size: number, options: KCenterOptions = {}): KCenterResult {
  const m = matrixView(x, 'kCenterGreedy')
  const k = Math.min(integerOption(size, 'size', 'kCenterGreedy', 1), m.n)
  const alg = kCenterGreedyAlgorithm(x, k, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'kCenterGreedy'), maxSteps: k })
  return {
    indices: state.selected,
    distances: Array.from(state.minDistances),
    radius: state.radius,
    radii: state.radii,
  }
}

/** Options of {@link herdingAlgorithm} and {@link kernelHerding}. */
export interface KernelHerdingOptions extends FitOptions {
  /** RBF kernel bandwidth `gamma` in `k(a, b) = exp(-gamma ||a - b||^2)` (default `1 / d`). */
  readonly gamma?: number
  /** Allow selecting the same point multiple times as a super-sample sequence (default false). */
  readonly replace?: boolean
}

/** Serialisable state of {@link herdingAlgorithm}. */
export interface HerdingState {
  readonly selected: readonly number[]
  readonly kernelSum: Float64Array
  readonly withinSum: number
  readonly mmd: number
  readonly mmds: readonly number[]
}

/** Result of {@link kernelHerding}. */
export interface KernelHerdingResult {
  /** Selected row indices in herding order. */
  readonly indices: readonly number[]
  /** Final RKHS maximum mean discrepancy `MMD(X, S)` after selecting all points. */
  readonly mmd: number
  /** MMD after each herding step. */
  readonly mmds: readonly number[]
}

/**
 * Welling's kernel herding in RKHS as a stepped {@link Algorithm} (Chen, Welling and Smola 2010): at each step `t`,
 * greedily choose the point `x_i` that maximises `E_{x}[k(x, x_i)] - (1 / (t + 1)) sum_{s in S} k(x_s, x_i)`, driving
 * the RKHS mean discrepancy down at O(1/T) rate and logging `coreset/mmd`.
 *
 * @param x - Data matrix `[n, d]`.
 * @param size - Number of points to herd.
 * @param options - RBF `gamma` (default `1 / d`), `replace` (default false) and trace options.
 * @returns Stepped {@link Algorithm} over {@link HerdingState}.
 * @concept coresets
 * @example
 * ```ts
 * import { herdingAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = herdingAlgorithm([[-2], [-1.9], [2], [2.1]], 2, { gamma: 1 })
 * runAlgorithm(alg, { maxSteps: 2 }).state.selected.length // => 2
 * ```
 */
export function herdingAlgorithm(
  x: TensorLike,
  size: number,
  options: KernelHerdingOptions = {},
): Algorithm<HerdingState> {
  const m = matrixView(x, 'kernelHerding')
  const replace = options.replace ?? false
  const k = replace
    ? integerOption(size, 'size', 'kernelHerding', 1)
    : Math.min(integerOption(size, 'size', 'kernelHerding', 1), m.n)
  const gamma = options.gamma ?? 1 / Math.max(1, m.d)
  if (!(gamma > 0)) throw new MdmlError('value', 'kernelHerding: gamma must be > 0')

  const K = rbfGram(m, m, gamma)
  const meanK = new Float64Array(m.n)
  let kBar = 0
  for (let i = 0; i < m.n; i++) {
    let rowSum = 0
    for (let j = 0; j < m.n; j++) rowSum += K[i * m.n + j] ?? 0
    meanK[i] = rowSum / m.n
    kBar += (meanK[i] ?? 0) / m.n
  }

  return {
    name: 'kernel-herding',
    config: { n: m.n, d: m.d, size: k, gamma, replace },
    schema: channels({
      metrics: {
        'coreset/mmd': { description: 'RKHS maximum mean discrepancy to the full dataset', direction: 'lower' },
        'coreset/size': { description: 'number of herded points' },
      },
      tensors: {
        'coreset/selected': { role: 'state', description: 'indices of the herded points' },
      },
    }),
    init: () => ({
      selected: [],
      kernelSum: new Float64Array(m.n),
      withinSum: 0,
      mmd: Math.sqrt(Math.max(0, kBar)),
      mmds: [],
    }),
    step: (s) => {
      const used = replace ? undefined : new Set(s.selected)
      const nextCount = s.selected.length + 1
      let best = -1
      let bestObj = Number.NEGATIVE_INFINITY
      for (let i = 0; i < m.n; i++) {
        if (used?.has(i)) continue
        // Maximising 2 * meanK[i] - (2 * kernelSum[i] + K[i,i]) / nextCount minimises MMD^2 at step nextCount.
        const obj = (meanK[i] ?? 0) - (s.kernelSum[i] ?? 0) / nextCount
        if (obj > bestObj) {
          bestObj = obj
          best = i
        }
      }
      if (best < 0) return s
      const nextKernelSum = Float64Array.from(s.kernelSum)
      for (let i = 0; i < m.n; i++) nextKernelSum[i] = (nextKernelSum[i] ?? 0) + (K[i * m.n + best] ?? 0)
      const nextWithin = s.withinSum + 2 * (s.kernelSum[best] ?? 0) + (K[best * m.n + best] ?? 1)
      const nextSelected = [...s.selected, best]
      let cross = 0
      for (const idx of nextSelected) cross += meanK[idx] ?? 0
      const mmdSq = Math.max(0, kBar - (2 * cross) / nextCount + nextWithin / (nextCount * nextCount))
      const mmd = Math.sqrt(mmdSq)
      return {
        selected: nextSelected,
        kernelSum: nextKernelSum,
        withinSum: nextWithin,
        mmd,
        mmds: [...s.mmds, mmd],
      }
    },
    done: (s) => (s.selected.length >= k ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { 'coreset/mmd': s.mmd, 'coreset/size': s.selected.length },
      tensors: {
        'coreset/selected': {
          role: 'state',
          value: Float64Array.from(s.selected),
          shape: [s.selected.length],
        },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as HerdingState,
  }
}

/**
 * Welling's kernel herding in RKHS (Chen, Welling and Smola 2010): greedily select `size` points whose empirical kernel
 * mean embedding matches the full dataset's mean embedding with minimal maximum mean discrepancy (MMD).
 *
 * @param x - Data matrix `[n, d]`.
 * @param size - Number of points to select.
 * @param options - RBF `gamma` (default `1 / d`), `replace` and trace options.
 * @returns Selected `indices`, final `mmd` and per-step `mmds`.
 * @concept coresets
 * @example
 * ```ts
 * import { kernelHerding } from 'mdml-engine/learn'
 * const r = kernelHerding([[-2], [-1.9], [2], [2.1]], 2, { gamma: 1 })
 * r.mmds[1] < r.mmds[0] // => true
 * ```
 */
export function kernelHerding(x: TensorLike, size: number, options: KernelHerdingOptions = {}): KernelHerdingResult {
  const m = matrixView(x, 'kernelHerding')
  const k =
    options.replace === true
      ? integerOption(size, 'size', 'kernelHerding', 1)
      : Math.min(integerOption(size, 'size', 'kernelHerding', 1), m.n)
  const alg = herdingAlgorithm(x, k, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'kernelHerding'), maxSteps: k })
  return { indices: state.selected, mmd: state.mmd, mmds: state.mmds }
}

/** Result of importance-weighted coreset extraction ({@link leverageScoreCoreset}, {@link sensitivitySampling}). */
export interface WeightedCoresetResult {
  /** Selected row indices `[size]`. */
  readonly indices: readonly number[]
  /** Unbiased importance weights `w_r = 1 / (size * p_{i_r})` for each selected point `[size]`. */
  readonly weights: readonly number[]
  /** Per-row importance scores (leverage or sensitivity) across the full dataset `[n]`. */
  readonly scores: readonly number[]
  /** Normalised sampling distribution `p_i = scores[i] / sum_j scores[j]` across the full dataset `[n]`. */
  readonly probabilities: readonly number[]
}

/** Options of {@link leverageScoreCoreset}. */
export interface LeverageCoresetOptions {
  /** Ridge regularisation parameter `lambda >= 0` for ridge leverage scores (default 0: exact SVD subspace leverage). */
  readonly lambda?: number
  /** Pick the top-`size` highest-leverage rows deterministically instead of sampling (default false). */
  readonly deterministic?: boolean
  readonly key?: Key
}

function selectByImportance(
  scores: Float64Array,
  size: number,
  deterministic: boolean,
  key: Key,
): WeightedCoresetResult {
  const n = scores.length
  let sum = 0
  for (let i = 0; i < n; i++) sum += scores[i] ?? 0
  const probs = sum > 0 ? Array.from(scores, (s) => s / sum) : Array.from({ length: n }, () => 1 / n)
  let indices: number[]
  if (deterministic) {
    indices = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p || a.i - b.i)
      .slice(0, Math.min(size, n))
      .map((e) => e.i)
  } else {
    indices = toFlat(categorical(key, probs, [size]))
  }
  const m = indices.length
  const weights = indices.map((idx) => 1 / (m * Math.max(probs[idx] ?? 1 / n, 1e-15)))
  return {
    indices,
    weights,
    scores: Array.from(scores),
    probabilities: probs,
  }
}

/**
 * Statistical leverage score coreset (Drineas, Mahoney and Muthukrishnan 2006; Alaoui and Mahoney 2015): compute each
 * row's subspace leverage score `l_i = sum_j (sigma_j^2 / (sigma_j^2 + lambda)) U_{i,j}^2` from the SVD of `x` and
 * extract a weighted coreset of `size` rows with importance weights `1 / (size * p_i)`.
 *
 * @param x - Design matrix `[n, d]`.
 * @param size - Coreset size.
 * @param options - Ridge `lambda` (default 0), `deterministic` flag and PRNG `key`.
 * @returns Selected `indices`, importance `weights`, leverage `scores` and sampling `probabilities`.
 * @concept coresets
 * @example
 * ```ts
 * import { leverageScoreCoreset } from 'mdml-engine/learn'
 * // Row 3 is a high-leverage point along the second axis.
 * const c = leverageScoreCoreset([[1, 0], [1, 0], [1, 0], [0, 10]], 2, { deterministic: true })
 * c.indices[0] // => 3
 * ```
 */
export function leverageScoreCoreset(
  x: TensorLike,
  size: number,
  options: LeverageCoresetOptions = {},
): WeightedCoresetResult {
  const m = matrixView(x, 'leverageScoreCoreset')
  const k = integerOption(size, 'size', 'leverageScoreCoreset', 1)
  const lambda = options.lambda ?? 0
  if (!(lambda >= 0)) throw new MdmlError('value', 'leverageScoreCoreset: lambda must be >= 0')
  const { u, s } = svd(fromView(m))
  const uFlat = toFlat(u)
  const sFlat = toFlat(s)
  const rank = sFlat.length
  const cutoff = Math.max(m.n, m.d) * 2.220446049250313e-16 * (sFlat[0] ?? 0)
  const scores = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) {
    let lev = 0
    for (let j = 0; j < rank; j++) {
      const sj = sFlat[j] ?? 0
      if (sj <= cutoff) continue
      const weight = lambda > 0 ? (sj * sj) / (sj * sj + lambda) : 1
      const uij = uFlat[i * rank + j] ?? 0
      lev += weight * uij * uij
    }
    scores[i] = Math.max(lev, 1e-15)
  }
  const key = keyFor(options.key === undefined ? {} : { key: options.key }, 'leverageScoreCoreset')
  return selectByImportance(scores, k, options.deterministic ?? false, key)
}

/** Options of {@link sensitivitySampling}. */
export interface SensitivityCoresetOptions {
  /** Number of rough centres used for the sensitivity upper bound (default `min(5, size)`). */
  readonly k?: number
  /** Pick the top-`size` highest-sensitivity rows deterministically (default false). */
  readonly deterministic?: boolean
  readonly key?: Key
}

/**
 * Sensitivity-sampling coreset for clustering and empirical risk (Feldman and Langberg 2011; Bachem, Lucic and Krause
 * 2017): bound each point's sensitivity by `2 a d^2(x_i, B) / c + 4 a mean_{B_j} d^2 / c + 4 n / |B_j|` for its nearest
 * rough centre `B_j` (`c` the mean squared distance to the centres, `a = 16 (log2 k + 2)`; the centres are a farthest-first
 * traversal), then extract an importance-weighted coreset.
 *
 * @param x - Data matrix `[n, d]`.
 * @param size - Coreset size.
 * @param options - Number of rough centres `k`, `deterministic` flag and PRNG `key`.
 * @returns Selected `indices`, importance `weights`, sensitivity `scores` and sampling `probabilities`.
 * @concept coresets
 * @example
 * ```ts
 * import { sensitivitySampling } from 'mdml-engine/learn'
 * // Isolated outlier at [100] has the highest sensitivity score.
 * const c = sensitivitySampling([[0], [0.1], [-0.1], [0.05], [100]], 2, { k: 1, deterministic: true })
 * c.indices[0] // => 4
 * ```
 */
export function sensitivitySampling(
  x: TensorLike,
  size: number,
  options: SensitivityCoresetOptions = {},
): WeightedCoresetResult {
  const m = matrixView(x, 'sensitivitySampling')
  const coresetSize = integerOption(size, 'size', 'sensitivitySampling', 1)
  const numCentres = Math.min(integerOption(options.k ?? Math.min(5, coresetSize), 'k', 'sensitivitySampling', 1), m.n)
  const key = keyFor(options.key === undefined ? {} : { key: options.key }, 'sensitivitySampling')

  // The rough solution B: a Gonzalez k-center traversal (a bicriteria stand-in for the D^2 seeding of the paper).
  const rough = kCenterGreedy(x, numCentres, { seedIndex: 0 }).indices
  const sq = squaredDistances(m, takeRows(m, rough))
  const nc = rough.length
  const clusterOf = new Int32Array(m.n)
  const distSq = new Float64Array(m.n)
  const clusterSizes = new Int32Array(nc)
  const clusterDistSum = new Float64Array(nc)
  let totalDistSq = 0
  for (let i = 0; i < m.n; i++) {
    let bestC = 0
    for (let c = 1; c < nc; c++) if ((sq[i * nc + c] ?? 0) < (sq[i * nc + bestC] ?? 0)) bestC = c
    const best = sq[i * nc + bestC] ?? 0
    clusterOf[i] = bestC
    distSq[i] = best
    clusterSizes[bestC] = (clusterSizes[bestC] ?? 0) + 1
    clusterDistSum[bestC] = (clusterDistSum[bestC] ?? 0) + best
    totalDistSq += best
  }

  // Bachem, Lucic and Krause (2017), Algorithm 1: s(x) = 2 alpha d(x, B)^2 / c + 4 alpha mean_{B_i} d^2 / c + 4 n / |B_i|,
  // with c the mean cost of B and alpha = 16 (log2 k + 2).
  const meanDistSq = totalDistSq / m.n
  const alpha = 16 * (Math.log2(Math.max(2, nc)) + 2)
  const scores = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) {
    const c = clusterOf[i] ?? 0
    const cSize = Math.max(1, clusterSizes[c] ?? 1)
    const cMean = (clusterDistSum[c] ?? 0) / cSize
    const dTerm = meanDistSq > 0 ? (2 * alpha * (distSq[i] ?? 0) + 4 * alpha * cMean) / meanDistSq : 0
    scores[i] = dTerm + (4 * m.n) / cSize
  }
  return selectByImportance(scores, coresetSize, options.deterministic ?? false, foldIn(key, 1))
}

/** Options of {@link bayesianCoresetAlgorithm} and {@link bayesianCoreset}. */
export interface BayesianCoresetOptions extends FitOptions {
  /** Centre each log-likelihood draw column before projecting (default false). */
  readonly centre?: boolean
}

/** Serialisable state of {@link bayesianCoresetAlgorithm}. */
export interface BayesianCoresetState {
  readonly step: number
  readonly weights: Float64Array
  readonly approx: Float64Array
  readonly residualNorm: number
  readonly relativeError: number
}

/** Result of {@link bayesianCoreset}. */
export interface BayesianCoresetResult {
  /** Sparse nonnegative weights `[n]` across all observations. */
  readonly weights: readonly number[]
  /** Indices of observations with positive weight. */
  readonly support: readonly number[]
  /** Final Euclidean norm `||sum_i w_i g_i - sum_i g_i||_2`. */
  readonly residualNorm: number
  /** Final relative error `||sum_i w_i g_i - g||_2 / ||g||_2`. */
  readonly relativeError: number
}

/**
 * Campbell and Broderick's Frank-Wolfe Hilbert Bayesian coreset as a stepped {@link Algorithm} (Campbell and Broderick
 * 2018, 2019): find sparse nonnegative weights `w >= 0` with at most `size` non-zeros whose weighted sum of
 * per-observation log-likelihood vectors `sum_i w_i g_i` best approximates the full data log-likelihood `g = sum_i g_i`.
 *
 * @param logLikelihoods - Per-observation log-likelihood vectors `[n, m]` evaluated at `m` parameter draws.
 * @param size - Maximum number of Frank-Wolfe steps / active coreset points.
 * @param options - Optional column centring and trace options.
 * @returns Stepped {@link Algorithm} over {@link BayesianCoresetState}.
 * @concept coresets
 * @example
 * ```ts
 * import { bayesianCoresetAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const L = [[1, 0], [1, 0], [0, 1], [0, 1]]
 * const alg = bayesianCoresetAlgorithm(L, 2)
 * const one = runAlgorithm(bayesianCoresetAlgorithm(L, 1), { maxSteps: 1 }).state.relativeError
 * runAlgorithm(alg, { maxSteps: 2 }).state.relativeError < one // => true
 * ```
 */
export function bayesianCoresetAlgorithm(
  logLikelihoods: TensorLike,
  size: number,
  options: BayesianCoresetOptions = {},
): Algorithm<BayesianCoresetState> {
  const raw = matrixView(logLikelihoods, 'bayesianCoreset')
  const n = raw.n
  const d = raw.d
  const maxSteps = integerOption(size, 'size', 'bayesianCoreset', 1)

  const gMat = new Float64Array(n * d)
  if (options.centre === true) {
    for (let j = 0; j < d; j++) {
      let colMean = 0
      for (let i = 0; i < n; i++) colMean += (raw.data[i * d + j] ?? 0) / n
      for (let i = 0; i < n; i++) gMat[i * d + j] = (raw.data[i * d + j] ?? 0) - colMean
    }
  } else {
    gMat.set(raw.data)
  }

  const norms = new Float64Array(n)
  const fullG = new Float64Array(d)
  let totalScale = 0
  for (let i = 0; i < n; i++) {
    const row = gMat.subarray(i * d, (i + 1) * d)
    const nm = dnrm2(row)
    norms[i] = nm
    totalScale += nm
    for (let j = 0; j < d; j++) fullG[j] = (fullG[j] ?? 0) + (row[j] ?? 0)
  }
  const fullNorm = Math.max(dnrm2(fullG), 1e-15)

  const bestDirection = (target: Float64Array): number => {
    let best = 0
    let bestScore = Number.NEGATIVE_INFINITY
    for (let i = 0; i < n; i++) {
      const nm = norms[i] ?? 0
      if (nm <= 1e-15) continue
      const row = gMat.subarray(i * d, (i + 1) * d)
      const score = ddot(row, target) / nm
      if (score > bestScore) {
        bestScore = score
        best = i
      }
    }
    return best
  }

  return {
    name: 'bayesian-coreset',
    config: { n, m: d, size: maxSteps, centre: options.centre ?? false },
    schema: channels({
      metrics: {
        'coreset/residual': { description: 'relative RKHS/L2 log-likelihood error', direction: 'lower' },
        'coreset/supportSize': { description: 'number of active positive weights' },
      },
      tensors: {
        'coreset/weights': { role: 'state', description: 'sparse nonnegative coreset weights', shape: ['n'] },
      },
    }),
    init: () => ({
      step: 0,
      weights: new Float64Array(n),
      approx: new Float64Array(d),
      residualNorm: dnrm2(fullG),
      relativeError: totalScale > 0 ? 1 : 0,
    }),
    step: (s) => {
      if (totalScale <= 1e-15) return { ...s, step: s.step + 1, residualNorm: 0, relativeError: 0 }
      const weights = Float64Array.from(s.weights)
      const approx = Float64Array.from(s.approx)
      if (s.step === 0) {
        const first = bestDirection(fullG)
        const nm = norms[first] ?? 1
        // Optimal nonnegative scalar along the first chosen atom: <g, g_first> / ||g_first||^2 (or Frank-Wolfe vertex).
        const row = gMat.subarray(first * d, (first + 1) * d)
        const w0 = Math.max(0, totalScale / nm)
        weights[first] = w0
        for (let j = 0; j < d; j++) approx[j] = w0 * (row[j] ?? 0)
      } else {
        const residual = Float64Array.from({ length: d }, (_, j) => (fullG[j] ?? 0) - (approx[j] ?? 0))
        const idx = bestDirection(residual)
        const nm = norms[idx] ?? 1
        const vScale = totalScale / nm
        const row = gMat.subarray(idx * d, (idx + 1) * d)
        const dir = Float64Array.from({ length: d }, (_, j) => vScale * (row[j] ?? 0) - (approx[j] ?? 0))
        const denom = ddot(dir, dir)
        const gamma = denom > 1e-24 ? Math.min(1, Math.max(0, ddot(residual, dir) / denom)) : 0
        for (let i = 0; i < n; i++) weights[i] = (1 - gamma) * (weights[i] ?? 0)
        weights[idx] = (weights[idx] ?? 0) + gamma * vScale
        for (let j = 0; j < d; j++) approx[j] = (1 - gamma) * (approx[j] ?? 0) + gamma * vScale * (row[j] ?? 0)
      }

      // Optimal radial rescaling alpha >= 0 of the active nonnegative weights so ||alpha * approx - fullG||_2 is minimal.
      const approxNormSq = ddot(approx, approx)
      if (approxNormSq > 1e-24) {
        const alpha = Math.max(0, ddot(fullG, approx) / approxNormSq)
        for (let i = 0; i < n; i++) weights[i] = alpha * (weights[i] ?? 0)
        for (let j = 0; j < d; j++) approx[j] = alpha * (approx[j] ?? 0)
      }

      const diff = Float64Array.from({ length: d }, (_, j) => (fullG[j] ?? 0) - (approx[j] ?? 0))
      const resNorm = dnrm2(diff)
      return {
        step: s.step + 1,
        weights,
        approx,
        residualNorm: resNorm,
        relativeError: resNorm / fullNorm,
      }
    },
    done: (s) => (s.step >= maxSteps || s.relativeError <= 1e-14 ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        'coreset/residual': s.relativeError,
        'coreset/supportSize': Array.from(s.weights).filter((w) => w > 1e-12).length,
      },
      tensors: {
        'coreset/weights': { role: 'state', value: s.weights, shape: [n] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BayesianCoresetState,
  }
}

/**
 * Bayesian coreset via Frank-Wolfe sparse projection (Campbell and Broderick 2018, 2019): approximate the full data
 * log-likelihood vector `sum_i g_i` with a sparse nonnegative combination `sum_i w_i g_i` of at most `size` points.
 *
 * @param logLikelihoods - Per-observation log-likelihood matrix `[n, m]`.
 * @param size - Maximum coreset support size / Frank-Wolfe iterations.
 * @param options - Optional column centring and trace options.
 * @returns Sparse `weights`, active `support` indices, `residualNorm` and `relativeError`.
 * @concept coresets
 * @example
 * ```ts
 * import { bayesianCoreset } from 'mdml-engine/learn'
 * const r = bayesianCoreset([[2, 0], [2, 0], [0, 3], [0, 3]], 2)
 * r.support.length // => 2
 * r.relativeError < bayesianCoreset([[2, 0], [2, 0], [0, 3], [0, 3]], 1).relativeError // => true
 * ```
 */
export function bayesianCoreset(
  logLikelihoods: TensorLike,
  size: number,
  options: BayesianCoresetOptions = {},
): BayesianCoresetResult {
  const maxSteps = integerOption(size, 'size', 'bayesianCoreset', 1)
  const alg = bayesianCoresetAlgorithm(logLikelihoods, maxSteps, options)
  const { state } = runAlgorithm(alg, { ...runOptionsFor(options, 'bayesianCoreset'), maxSteps })
  const weights = Array.from(state.weights, (w) => (w > 1e-12 ? w : 0))
  const support = weights.flatMap((w, i) => (w > 0 ? [i] : []))
  return {
    weights,
    support,
    residualNorm: state.residualNorm,
    relativeError: state.relativeError,
  }
}
