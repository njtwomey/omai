/**
 * Split criteria and split search for decision trees: impurities for classification (Gini, entropy in bits) and
 * regression (squared error, absolute error, Poisson deviance), node values, and `searchSplits`, which scores every
 * midpoint between sorted feature values (scikit-learn's best splitter) or one random threshold per feature (the
 * extra-trees splitter), keeping the candidates for explanation.
 */

import type { SplitCandidate } from '../../lib/exposition/index.ts'
import type { MatrixView } from '../../lib/learn/index.ts'
import { foldIn, type Key, uniform } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'

// TODO: histogram split search, surrogate splits.

/** Impurity criteria for classification (entropy and log loss are the same, in bits as scikit-learn). */
export type ClassificationCriterion = 'gini' | 'entropy' | 'log_loss'
/** Impurity criteria for regression. */
export type RegressionCriterion = 'squared_error' | 'friedman_mse' | 'absolute_error' | 'poisson'
/** Any split criterion. */
export type Criterion = ClassificationCriterion | RegressionCriterion

/** What a split search needs: the data, the criterion and the leaf-size limits. */
export interface SplitProblem {
  readonly x: MatrixView
  /** Class codes (classification) or targets (regression). */
  readonly y: Float64Array
  readonly w: Float64Array
  /** Number of classes (0 for regression). */
  readonly k: number
  readonly criterion: Criterion
  readonly minSamplesLeaf: number
  /** The minimum weight of a leaf. */
  readonly minWeightLeaf: number
  /** The total training weight (decreases are weighted by node weight / total). */
  readonly totalWeight: number
}

/** The best split found at a node, with the rows it sends each way. */
export interface ChosenSplit extends SplitCandidate {
  readonly leftRows: Int32Array
  readonly rightRows: Int32Array
}

/** A node's split search: the best split (if any valid one exists) and the candidates evaluated. */
export interface SplitSearch {
  readonly best?: ChosenSplit
  readonly candidates: readonly SplitCandidate[]
}

const LOG2 = Math.log(2)
const FEATURE_THRESHOLD = 1e-7

function classImpurity(counts: ArrayLike<number>, total: number, criterion: Criterion): number {
  if (total <= 0) return 0
  let s = 0
  if (criterion === 'gini') {
    for (let c = 0; c < counts.length; c++) s += ((counts[c] ?? 0) / total) ** 2
    return 1 - s
  }
  for (let c = 0; c < counts.length; c++) {
    const p = (counts[c] ?? 0) / total
    if (p > 0) s -= (p * Math.log(p)) / LOG2
  }
  return s
}

function weightedMedian(values: readonly number[], weights: readonly number[]): number {
  const order = values.map((_, i) => i).sort((a, b) => (values[a] ?? 0) - (values[b] ?? 0))
  const total = weights.reduce((a, b) => a + b, 0)
  let acc = 0
  for (const [k, i] of order.entries()) {
    acc += weights[i] ?? 0
    // Exactly half the weight below: the midpoint of the two middle values (scikit-learn's WeightedMedianCalculator).
    if (acc === total / 2 && k + 1 < order.length) return ((values[i] ?? 0) + (values[order[k + 1] ?? 0] ?? 0)) / 2
    if (acc >= total / 2) return values[i] ?? 0
  }
  return values[order[order.length - 1] ?? 0] ?? 0
}

/**
 * The impurity of a set of rows under a criterion: Gini or entropy (bits) of the weighted class proportions, the
 * weighted variance, the weighted mean absolute deviation from the weighted median, or the half Poisson deviance.
 *
 * @param problem - The data and criterion.
 * @param rows - The rows.
 * @returns The impurity.
 */
export function impurityOf(problem: SplitProblem, rows: ArrayLike<number>): number {
  const { y, w, k, criterion } = problem
  if (k > 0) {
    const counts = new Float64Array(k)
    let total = 0
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i] ?? 0
      counts[y[r] ?? 0] = (counts[y[r] ?? 0] ?? 0) + (w[r] ?? 0)
      total += w[r] ?? 0
    }
    return classImpurity(counts, total, criterion)
  }
  let sw = 0
  let s = 0
  let s2 = 0
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i] ?? 0
    const wi = w[r] ?? 0
    sw += wi
    s += wi * (y[r] ?? 0)
    s2 += wi * (y[r] ?? 0) ** 2
  }
  if (sw <= 0) return 0
  const mean = s / sw
  if (criterion === 'absolute_error') {
    const vs = Array.from(rows, (r) => y[r] ?? 0)
    const ws = Array.from(rows, (r) => w[r] ?? 0)
    const med = weightedMedian(vs, ws)
    return vs.reduce((a, v, i) => a + (ws[i] ?? 0) * Math.abs(v - med), 0) / sw
  }
  if (criterion === 'poisson') {
    if (mean <= 0) return Number.POSITIVE_INFINITY
    let dev = 0
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i] ?? 0
      const v = y[r] ?? 0
      if (v > 0) dev += (w[r] ?? 0) * v * Math.log(v / mean)
    }
    return dev / sw
  }
  return Math.max(s2 / sw - mean * mean, 0)
}

/**
 * The value of a node: weighted class counts for classification, the weighted mean (the weighted median for absolute
 * error) for regression.
 *
 * @param problem - The data and criterion.
 * @param rows - The node's rows.
 * @returns The value.
 */
export function nodeValue(problem: SplitProblem, rows: ArrayLike<number>): number[] {
  const { y, w, k } = problem
  if (k > 0) {
    const counts = new Array<number>(k).fill(0)
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i] ?? 0
      counts[y[r] ?? 0] = (counts[y[r] ?? 0] ?? 0) + (w[r] ?? 0)
    }
    return counts
  }
  if (problem.criterion === 'absolute_error')
    return [
      weightedMedian(
        Array.from(rows, (r) => y[r] ?? 0),
        Array.from(rows, (r) => w[r] ?? 0),
      ),
    ]
  let sw = 0
  let s = 0
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i] ?? 0
    sw += w[r] ?? 0
    s += (w[r] ?? 0) * (y[r] ?? 0)
  }
  return [sw > 0 ? s / sw : 0]
}

/** Running sufficient statistics of one side of a split. */
class Side {
  readonly counts: Float64Array
  weight = 0
  n = 0
  sum = 0
  sum2 = 0
  constructor(k: number) {
    this.counts = new Float64Array(Math.max(k, 0))
  }
  add(problem: SplitProblem, r: number, sign: 1 | -1): void {
    const wi = (problem.w[r] ?? 0) * sign
    const v = problem.y[r] ?? 0
    this.weight += wi
    this.n += sign
    if (problem.k > 0) this.counts[v] = (this.counts[v] ?? 0) + wi
    else {
      this.sum += wi * v
      this.sum2 += wi * v * v
    }
  }
  impurity(problem: SplitProblem, rows: () => ArrayLike<number>): number {
    if (problem.k > 0) return classImpurity(this.counts, this.weight, problem.criterion)
    if (this.weight <= 0) return 0
    const mean = this.sum / this.weight
    if (problem.criterion === 'absolute_error' || problem.criterion === 'poisson') return impurityOf(problem, rows())
    return Math.max(this.sum2 / this.weight - mean * mean, 0)
  }
}

function evaluate(
  problem: SplitProblem,
  parent: number,
  wNode: number,
  feature: number,
  threshold: number,
  left: Side,
  right: Side,
  leftRows: () => ArrayLike<number>,
  rightRows: () => ArrayLike<number>,
): SplitCandidate | undefined {
  if (left.n < problem.minSamplesLeaf || right.n < problem.minSamplesLeaf) return undefined
  if (left.weight < problem.minWeightLeaf || right.weight < problem.minWeightLeaf) return undefined
  if (problem.criterion === 'poisson' && (left.sum <= 1e-12 || right.sum <= 1e-12)) return undefined
  const il = left.impurity(problem, leftRows)
  const ir = right.impurity(problem, rightRows)
  const decrease = (wNode / problem.totalWeight) * (parent - (left.weight / wNode) * il - (right.weight / wNode) * ir)
  return { feature, threshold, impurityLeft: il, impurityRight: ir, decrease }
}

/**
 * Friedman's improvement in scikit-learn's proxy form (W_r S_l - W_l S_r)^2 / (W_l W_r) = W_l W_r (mean_l - mean_r)^2,
 * the score friedman_mse ranks splits by (computed without divisions first, so exact ties stay ties).
 */
function friedman(left: Side, right: Side): number {
  const diff = right.weight * left.sum - left.weight * right.sum
  return (diff * diff) / (left.weight * right.weight)
}

/** Options of {@link searchSplits}. */
export interface SearchOptions {
  /** 'best' sweeps every threshold; 'random' draws one threshold per feature (extra trees). */
  readonly splitter: 'best' | 'random'
  readonly key?: Key
  /** Candidates kept per feature for the record: all, or the best few (default all). */
  readonly record?: number
}

/**
 * Search the splits of a node over some features: every midpoint between distinct sorted values (the best splitter,
 * scikit-learn's), or one uniform threshold per feature (the random splitter of extra trees). Splits leaving fewer
 * than `minSamplesLeaf` rows or `minWeightLeaf` weight on a side are invalid. Each valid candidate records the
 * children's impurities and the weighted impurity decrease (node weight / total weight times the decrease).
 *
 * @param problem - The data, criterion and limits.
 * @param rows - The node's rows.
 * @param features - The features to search, in order (ties go to the first).
 * @param options - Splitter, key and how many candidates per feature to record.
 * @returns The best split and the candidates evaluated.
 */
export function searchSplits(
  problem: SplitProblem,
  rows: Int32Array,
  features: readonly number[],
  options: SearchOptions,
): SplitSearch {
  const { x } = problem
  const parent = impurityOf(problem, rows)
  let wNode = 0
  for (const r of rows) wNode += problem.w[r] ?? 0
  const candidates: SplitCandidate[] = []
  let best: { candidate: SplitCandidate; score: number; feature: number; threshold: number } | undefined
  const consider = (c: SplitCandidate, score: number): void => {
    if (best === undefined || score > best.score)
      best = { candidate: c, score, feature: c.feature, threshold: c.threshold }
  }
  const scoreOf = (c: SplitCandidate, left: Side, right: Side): number =>
    problem.criterion === 'friedman_mse' ? friedman(left, right) : c.decrease
  for (const [fi, j] of features.entries()) {
    const kept: SplitCandidate[] = []
    if (options.splitter === 'random') {
      let lo = Number.POSITIVE_INFINITY
      let hi = Number.NEGATIVE_INFINITY
      for (const r of rows) {
        const v = x.data[r * x.d + j] ?? 0
        lo = Math.min(lo, v)
        hi = Math.max(hi, v)
      }
      if (hi <= lo + FEATURE_THRESHOLD) continue
      const u = options.key === undefined ? 0.5 : (toFlat(uniform(foldIn(options.key, fi), []))[0] ?? 0.5)
      let threshold = lo + u * (hi - lo)
      if (threshold >= hi) threshold = lo
      const left = new Side(problem.k)
      const right = new Side(problem.k)
      const l: number[] = []
      const rr: number[] = []
      for (const r of rows) {
        if ((x.data[r * x.d + j] ?? 0) <= threshold) {
          left.add(problem, r, 1)
          l.push(r)
        } else {
          right.add(problem, r, 1)
          rr.push(r)
        }
      }
      const c = evaluate(
        problem,
        parent,
        wNode,
        j,
        threshold,
        left,
        right,
        () => l,
        () => rr,
      )
      if (c !== undefined) {
        kept.push(c)
        consider(c, scoreOf(c, left, right))
      }
    } else {
      const sorted = Int32Array.from(rows).sort((a, b) => (x.data[a * x.d + j] ?? 0) - (x.data[b * x.d + j] ?? 0))
      const left = new Side(problem.k)
      const right = new Side(problem.k)
      for (const r of sorted) right.add(problem, r, 1)
      for (let p = 1; p < sorted.length; p++) {
        const prev = sorted[p - 1] ?? 0
        left.add(problem, prev, 1)
        right.add(problem, prev, -1)
        const a = x.data[prev * x.d + j] ?? 0
        const b = x.data[(sorted[p] ?? 0) * x.d + j] ?? 0
        if (b <= a + FEATURE_THRESHOLD) continue
        let threshold = a / 2 + b / 2
        if (threshold === b || !Number.isFinite(threshold)) threshold = a
        const c = evaluate(
          problem,
          parent,
          wNode,
          j,
          threshold,
          left,
          right,
          () => sorted.subarray(0, p),
          () => sorted.subarray(p),
        )
        if (c !== undefined) {
          kept.push(c)
          consider(c, scoreOf(c, left, right))
        }
      }
    }
    const limit = options.record
    candidates.push(...(limit === undefined ? kept : [...kept].sort((a, b) => b.decrease - a.decrease).slice(0, limit)))
  }
  if (best === undefined) return { candidates }
  const chosen = best as { candidate: SplitCandidate }
  const { feature, threshold } = chosen.candidate
  const leftRows = Int32Array.from(Array.from(rows).filter((r) => (x.data[r * x.d + feature] ?? 0) <= threshold))
  const rightRows = Int32Array.from(Array.from(rows).filter((r) => (x.data[r * x.d + feature] ?? 0) > threshold))
  return { best: { ...chosen.candidate, leftRows, rightRows }, candidates }
}
