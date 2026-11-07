/**
 * Split criteria and split search for decision trees: impurities for classification (Gini, entropy in bits, the same
 * as log loss) and regression (squared error, Friedman's improvement, absolute error, Poisson deviance), node values,
 * and `searchSplits`, which scores every threshold between consecutive sorted values (scikit-learn's best splitter) or
 * one random threshold per feature (the extra-trees splitter), splits categorical features by the best subset of codes
 * and sends missing values (NaN) to whichever side does better. The candidates evaluated are kept for explanation.
 */

import { maxOf } from '../base/index.ts'
import type { SplitCandidate } from '../exposition/index.ts'
import type { MatrixView } from '../learn/validation.ts'
import { weightedPercentile } from '../metrics/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'

/** Impurity criteria for classification (entropy and log loss are the same, in bits as scikit-learn). */
export type ClassificationCriterion = 'gini' | 'entropy' | 'log_loss'
/** Impurity criteria for regression. */
export type RegressionCriterion = 'squared_error' | 'friedman_mse' | 'absolute_error' | 'poisson'
/** Any split criterion; `gain_ratio` is C4.5's: entropy impurity, splits ranked by information gain over split info. */
export type Criterion = ClassificationCriterion | RegressionCriterion | 'gain_ratio'

/** Whether a criterion is a classification criterion. */
const CLASSIFICATION: ReadonlySet<string> = new Set(['gini', 'entropy', 'log_loss', 'gain_ratio'])

/**
 * Whether a criterion is for classification (otherwise regression).
 *
 * @param criterion - The criterion.
 * @returns `true` for Gini, entropy, log loss and gain ratio.
 * @example
 * ```ts
 * import { isClassificationCriterion } from 'mdml-engine/trees'
 * isClassificationCriterion('gini') // => true
 * isClassificationCriterion('poisson') // => false
 * ```
 */
export function isClassificationCriterion(criterion: Criterion): criterion is ClassificationCriterion | 'gain_ratio' {
  return CLASSIFICATION.has(criterion)
}

/** What a split search needs: the data, the criterion and the leaf-size limits. */
export interface SplitProblem {
  /** Features, NaN for missing values. */
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
  /** Per feature: whether its values are category codes (default none). */
  readonly categorical?: readonly boolean[]
}

/** The best split found at a node, with the rows it sends each way. */
export interface ChosenSplit extends SplitCandidate {
  readonly leftRows: Int32Array
  readonly rightRows: Int32Array
  /** Where missing values go at this split. */
  readonly missingLeft: boolean
}

/** A node's split search: the best split (if any valid one exists) and the candidates evaluated. */
export interface SplitSearch {
  readonly best?: ChosenSplit
  readonly candidates: readonly SplitCandidate[]
}

const LOG2 = Math.log(2)
const FEATURE_THRESHOLD = 1e-7
/** The threshold of a split that sends every present value left and only missing values right. */
export const ALL_PRESENT: number = Number.MAX_VALUE

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
  return weightedPercentile(values, weights, 50, true)
}

/** Running sufficient statistics of a set of rows. */
class Stats {
  readonly counts: Float64Array
  weight = 0
  n = 0
  sum = 0
  sum2 = 0
  sumYlogY = 0
  constructor(k: number) {
    this.counts = new Float64Array(Math.max(k, 0))
  }
  add(problem: SplitProblem, r: number, sign: 1 | -1 = 1): void {
    const wi = (problem.w[r] ?? 0) * sign
    const v = problem.y[r] ?? 0
    this.weight += wi
    this.n += sign
    if (problem.k > 0) this.counts[v] = (this.counts[v] ?? 0) + wi
    else {
      this.sum += wi * v
      this.sum2 += wi * v * v
      if (v > 0) this.sumYlogY += wi * v * Math.log(v)
    }
  }
  merge(other: Stats, sign: 1 | -1 = 1): void {
    this.weight += sign * other.weight
    this.n += sign * other.n
    this.sum += sign * other.sum
    this.sum2 += sign * other.sum2
    this.sumYlogY += sign * other.sumYlogY
    for (let c = 0; c < this.counts.length; c++) this.counts[c] = (this.counts[c] ?? 0) + sign * (other.counts[c] ?? 0)
  }
  copy(): Stats {
    const s = new Stats(this.counts.length)
    s.merge(this)
    return s
  }
  impurity(problem: SplitProblem, rows: () => ArrayLike<number>): number {
    if (problem.k > 0) return classImpurity(this.counts, this.weight, problem.criterion)
    if (this.weight <= 0) return 0
    const mean = this.sum / this.weight
    if (problem.criterion === 'absolute_error') return absoluteImpurity(problem, rows())
    if (problem.criterion === 'poisson') {
      if (mean <= 0) return Number.POSITIVE_INFINITY
      return Math.max((this.sumYlogY - this.sum * Math.log(mean)) / this.weight, 0)
    }
    return Math.max(this.sum2 / this.weight - mean * mean, 0)
  }
}

function absoluteImpurity(problem: SplitProblem, rows: ArrayLike<number>): number {
  const vs = Array.from(rows, (r) => problem.y[r] ?? 0)
  const ws = Array.from(rows, (r) => problem.w[r] ?? 0)
  const sw = ws.reduce((a, b) => a + b, 0)
  if (sw <= 0) return 0
  const med = weightedMedian(vs, ws)
  return vs.reduce((a, v, i) => a + (ws[i] ?? 0) * Math.abs(v - med), 0) / sw
}

function statsOf(problem: SplitProblem, rows: ArrayLike<number>): Stats {
  const s = new Stats(problem.k)
  for (let i = 0; i < rows.length; i++) s.add(problem, rows[i] ?? 0)
  return s
}

/**
 * The impurity of a set of rows under a criterion: Gini or entropy (bits) of the weighted class proportions, the
 * weighted variance, the weighted mean absolute deviation from the weighted median, or the half Poisson deviance.
 *
 * @param problem - The data and criterion.
 * @param rows - The rows.
 * @returns The impurity.
 * @example
 * ```ts
 * import { impurityOf } from 'mdml-engine/trees'
 * import { matrixView } from 'mdml-engine/learn'
 * const p = { x: matrixView([[0], [1]], 'demo'), y: Float64Array.of(0, 1), w: Float64Array.of(1, 1), k: 2, criterion: 'gini' as const, minSamplesLeaf: 1, minWeightLeaf: 0, totalWeight: 2 }
 * impurityOf(p, [0, 1]) // => 0.5
 * ```
 */
export function impurityOf(problem: SplitProblem, rows: ArrayLike<number>): number {
  return statsOf(problem, rows).impurity(problem, () => rows)
}

/**
 * The value of a node: weighted class counts for classification, the weighted mean (the weighted median for absolute
 * error) for regression.
 *
 * @param problem - The data and criterion.
 * @param rows - The node's rows.
 * @returns The value.
 * @example
 * ```ts
 * import { nodeValue } from 'mdml-engine/trees'
 * import { matrixView } from 'mdml-engine/learn'
 * const p = { x: matrixView([[0], [1], [2]], 'demo'), y: Float64Array.of(1, 2, 6), w: Float64Array.of(1, 1, 1), k: 0, criterion: 'squared_error' as const, minSamplesLeaf: 1, minWeightLeaf: 0, totalWeight: 3 }
 * nodeValue(p, [0, 1, 2]) // => [3]
 * ```
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
    sw += w[rows[i] ?? 0] ?? 0
    s += (w[rows[i] ?? 0] ?? 0) * (y[rows[i] ?? 0] ?? 0)
  }
  return [sw > 0 ? s / sw : 0]
}

/** Options of {@link searchSplits}. */
export interface SearchOptions {
  /** 'best' sweeps every threshold; 'random' draws one threshold per feature (extra trees). */
  readonly splitter: 'best' | 'random'
  readonly key?: Key
  /** Candidates kept per feature for the record: all (default), the best few, or none (0, faster). */
  readonly record?: number
}

interface Evaluated {
  readonly candidate: SplitCandidate
  readonly score: number
}

/**
 * Friedman's improvement in scikit-learn's proxy form (W_r S_l - W_l S_r)^2 / (W_l W_r) = W_l W_r (mean_l - mean_r)^2,
 * the score friedman_mse ranks splits by (computed without divisions first, so exact ties stay ties).
 */
function friedman(left: Stats, right: Stats): number {
  const diff = right.weight * left.sum - left.weight * right.sum
  return (diff * diff) / (left.weight * right.weight)
}

function splitInfo(left: Stats, right: Stats): number {
  const total = left.weight + right.weight
  let s = 0
  for (const side of [left, right]) {
    const p = side.weight / total
    if (p > 0) s -= (p * Math.log(p)) / LOG2
  }
  return s
}

/** Everything a search over one node shares: the totals and the parent's impurity. */
interface Node {
  readonly problem: SplitProblem
  readonly rows: Int32Array
  readonly parent: number
  readonly wNode: number
}

function evaluate(
  node: Node,
  base: Pick<SplitCandidate, 'feature' | 'threshold'> & Partial<Pick<SplitCandidate, 'categories' | 'missingLeft'>>,
  left: Stats,
  right: Stats,
  leftRows: () => ArrayLike<number>,
  rightRows: () => ArrayLike<number>,
): Evaluated | undefined {
  const { problem } = node
  if (left.n < problem.minSamplesLeaf || right.n < problem.minSamplesLeaf) return undefined
  if (left.weight < problem.minWeightLeaf || right.weight < problem.minWeightLeaf) return undefined
  if (left.weight <= 0 || right.weight <= 0) return undefined
  if (problem.criterion === 'poisson' && (left.sum <= 1e-12 || right.sum <= 1e-12)) return undefined
  const il = left.impurity(problem, leftRows)
  const ir = right.impurity(problem, rightRows)
  const decrease =
    (node.wNode / problem.totalWeight) *
    (node.parent - (left.weight / node.wNode) * il - (right.weight / node.wNode) * ir)
  const gain = node.parent - (left.weight / node.wNode) * il - (right.weight / node.wNode) * ir
  let score = decrease
  if (problem.criterion === 'friedman_mse') score = friedman(left, right)
  else if (problem.criterion === 'gain_ratio') {
    const info = splitInfo(left, right)
    if (!(info > 1e-12)) return undefined
    score = gain / info
  }
  const candidate: SplitCandidate = { ...base, impurityLeft: il, impurityRight: ir, decrease }
  return { candidate: score === decrease ? candidate : { ...candidate, score }, score }
}

/**
 * Search the splits of a node over some features: every midpoint between distinct sorted values (the best splitter,
 * scikit-learn's), or one uniform threshold per feature (the random splitter of extra trees); for a categorical
 * feature the best subset of its codes (categories ordered by their mean response, so that the optimal subset is a
 * prefix, or all subsets for few categories of a multiclass problem). Splits leaving fewer than `minSamplesLeaf` rows
 * or `minWeightLeaf` weight on a side are invalid. Rows with a missing value (NaN) are sent to the side that scores
 * better, and a split of the present values from the missing ones is also tried. Each valid candidate records the
 * children's impurities and the weighted impurity decrease (node weight / total weight times the decrease).
 *
 * @param problem - The data, criterion and limits.
 * @param rows - The node's rows.
 * @param features - The features to search, in order (ties go to the first).
 * @param options - Splitter, key and how many candidates per feature to record.
 * @returns The best split and the candidates evaluated.
 * @example
 * ```ts
 * import { searchSplits } from 'mdml-engine/trees'
 * import { matrixView } from 'mdml-engine/learn'
 * const p = { x: matrixView([[0], [1], [2], [3]], 'demo'), y: Float64Array.of(0, 0, 1, 1), w: Float64Array.of(1, 1, 1, 1), k: 2, criterion: 'gini' as const, minSamplesLeaf: 1, minWeightLeaf: 0, totalWeight: 4 }
 * searchSplits(p, Int32Array.of(0, 1, 2, 3), [0], { splitter: 'best' }).best?.threshold // => 1.5
 * ```
 */
export function searchSplits(
  problem: SplitProblem,
  rows: Int32Array,
  features: readonly number[],
  options: SearchOptions,
): SplitSearch {
  const { x } = problem
  const all = statsOf(problem, rows)
  const node: Node = {
    problem,
    rows,
    parent: all.impurity(problem, () => rows),
    wNode: all.weight,
  }
  const keepAll = options.record !== 0
  const candidates: SplitCandidate[] = []
  let best: Evaluated | undefined
  for (const [fi, j] of features.entries()) {
    const kept: SplitCandidate[] = []
    const consider = (e: Evaluated | undefined): void => {
      if (e === undefined) return
      if (keepAll) kept.push(e.candidate)
      if (best === undefined || e.score > best.score) best = e
    }
    const present: number[] = []
    const missing: number[] = []
    for (const r of rows) (Number.isNaN(x.data[r * x.d + j] ?? 0) ? missing : present).push(r)
    const miss = statsOf(problem, missing)
    const hasMissing = missing.length > 0
    const value = (r: number): number => x.data[r * x.d + j] ?? 0
    // A split with the missing rows on each side in turn.
    const withMissing = (
      base: Pick<SplitCandidate, 'feature' | 'threshold'> & Partial<Pick<SplitCandidate, 'categories'>>,
      left: Stats,
      right: Stats,
      leftRows: () => ArrayLike<number>,
      rightRows: () => ArrayLike<number>,
    ): void => {
      if (!hasMissing) {
        consider(evaluate(node, base, left, right, leftRows, rightRows))
        return
      }
      const rightPlus = right.copy()
      rightPlus.merge(miss)
      consider(
        evaluate(node, { ...base, missingLeft: false }, left, rightPlus, leftRows, () => [
          ...asArray(rightRows()),
          ...missing,
        ]),
      )
      const leftPlus = left.copy()
      leftPlus.merge(miss)
      consider(
        evaluate(
          node,
          { ...base, missingLeft: true },
          leftPlus,
          right,
          () => [...asArray(leftRows()), ...missing],
          rightRows,
        ),
      )
    }
    if (present.length >= 2 && problem.categorical?.[j] === true) {
      searchCategorical(node, j, present, options, fi, withMissing)
    } else if (present.length >= 2 && options.splitter === 'random') {
      let lo = Number.POSITIVE_INFINITY
      let hi = Number.NEGATIVE_INFINITY
      for (const r of present) {
        lo = Math.min(lo, value(r))
        hi = Math.max(hi, value(r))
      }
      if (hi > lo + FEATURE_THRESHOLD) {
        const u = options.key === undefined ? 0.5 : (toFlat(uniform(foldIn(options.key, fi), []))[0] ?? 0.5)
        let threshold = lo + u * (hi - lo)
        if (threshold >= hi) threshold = lo
        const left = new Stats(problem.k)
        const right = new Stats(problem.k)
        const l: number[] = []
        const rr: number[] = []
        for (const r of present) {
          if (value(r) <= threshold) {
            left.add(problem, r)
            l.push(r)
          } else {
            right.add(problem, r)
            rr.push(r)
          }
        }
        withMissing(
          { feature: j, threshold },
          left,
          right,
          () => l,
          () => rr,
        )
      }
    } else if (present.length >= 2) {
      const sorted = Int32Array.from(present).sort((a, b) => value(a) - value(b))
      const left = new Stats(problem.k)
      const right = statsOf(problem, sorted)
      for (let p = 1; p < sorted.length; p++) {
        const prev = sorted[p - 1] ?? 0
        left.add(problem, prev)
        right.add(problem, prev, -1)
        const a = value(prev)
        const b = value(sorted[p] ?? 0)
        if (b <= a + FEATURE_THRESHOLD) continue
        let threshold = a / 2 + b / 2
        if (threshold === b || !Number.isFinite(threshold)) threshold = a
        withMissing(
          { feature: j, threshold },
          left,
          right,
          () => sorted.subarray(0, p),
          () => sorted.subarray(p),
        )
      }
    }
    // Every present value left, only the missing ones right.
    if (hasMissing && present.length > 0 && problem.categorical?.[j] !== true)
      consider(
        evaluate(
          node,
          { feature: j, threshold: ALL_PRESENT, missingLeft: false },
          statsOf(problem, present),
          miss,
          () => present,
          () => missing,
        ),
      )
    const limit = options.record
    candidates.push(...(limit === undefined ? kept : [...kept].sort((a, b) => b.decrease - a.decrease).slice(0, limit)))
  }
  const found = best as Evaluated | undefined
  if (found === undefined) return { candidates }
  const chosen = found.candidate
  const { leftRows, rightRows } = partitionRows(problem, rows, chosen, chosen.missingLeft === true)
  const noMissing = !rows.some((r) => Number.isNaN(x.data[r * x.d + chosen.feature] ?? 0))
  // With no missing values seen, future ones follow the child with more samples (scikit-learn's rule).
  const missingLeft = noMissing ? leftRows.length > rightRows.length : chosen.missingLeft === true
  return { best: { ...chosen, leftRows, rightRows, missingLeft }, candidates }
}

const asArray = (r: ArrayLike<number>): number[] => Array.from(r)

/**
 * Send rows left or right by a split: a numeric split sends `x <= threshold` left, a categorical one the codes
 * listed in `categories`, and a missing value goes left when `missingLeft`.
 *
 * @param problem - The data.
 * @param rows - The rows to send.
 * @param split - The split's feature, threshold and categories.
 * @param missingLeft - Whether missing values go left.
 * @returns The rows each way.
 * @example
 * ```ts
 * import { partitionRows } from 'mdml-engine/trees'
 * import { matrixView } from 'mdml-engine/learn'
 * const p = { x: matrixView([[0], [1], [2]], 'demo'), y: Float64Array.of(0, 0, 1), w: Float64Array.of(1, 1, 1), k: 2, criterion: 'gini' as const, minSamplesLeaf: 1, minWeightLeaf: 0, totalWeight: 3 }
 * Array.from(partitionRows(p, Int32Array.of(0, 1, 2), { feature: 0, threshold: 1.5 }, false).leftRows) // => [0, 1]
 * ```
 */
export function partitionRows(
  problem: SplitProblem,
  rows: Int32Array,
  split: { readonly feature: number; readonly threshold: number; readonly categories?: readonly number[] },
  missingLeft: boolean,
): { readonly leftRows: Int32Array; readonly rightRows: Int32Array } {
  const { x } = problem
  const cats = split.categories ?? []
  const l: number[] = []
  const r: number[] = []
  for (const row of rows) {
    const v = x.data[row * x.d + split.feature] ?? 0
    const left = Number.isNaN(v) ? missingLeft : cats.length > 0 ? cats.includes(v) : v <= split.threshold
    ;(left ? l : r).push(row)
  }
  return { leftRows: Int32Array.from(l), rightRows: Int32Array.from(r) }
}

/** Subsets of `0..m-1` that contain element 0 (so each partition is listed once), except the whole set. */
function subsetsOf(m: number): number[][] {
  const out: number[][] = []
  for (let mask = 0; mask < 2 ** (m - 1) - 1; mask++) {
    const s = [0]
    for (let b = 0; b < m - 1; b++) if (mask & (1 << b)) s.push(b + 1)
    out.push(s)
  }
  return out
}

function searchCategorical(
  node: Node,
  j: number,
  present: readonly number[],
  options: SearchOptions,
  fi: number,
  withMissing: (
    base: Pick<SplitCandidate, 'feature' | 'threshold'> & Partial<Pick<SplitCandidate, 'categories'>>,
    left: Stats,
    right: Stats,
    leftRows: () => ArrayLike<number>,
    rightRows: () => ArrayLike<number>,
  ) => void,
): void {
  const { problem } = node
  const { x } = problem
  const byCode = new Map<number, number[]>()
  for (const r of present) {
    const c = x.data[r * x.d + j] ?? 0
    const list = byCode.get(c)
    if (list === undefined) byCode.set(c, [r])
    else list.push(r)
  }
  const codes = [...byCode.keys()].sort((a, b) => a - b)
  if (codes.length < 2) return
  const stats = codes.map((c) => statsOf(problem, byCode.get(c) ?? []))
  const rowsOf = (members: readonly number[]): number[] => members.flatMap((i) => byCode.get(codes[i] ?? 0) ?? [])
  const total = statsOf(problem, present)
  const emit = (leftIdx: readonly number[]): void => {
    const left = new Stats(problem.k)
    for (const i of leftIdx) left.merge(stats[i] as Stats)
    const right = total.copy()
    right.merge(left, -1)
    const inLeft = new Set(leftIdx)
    const rightIdx = codes.map((_, i) => i).filter((i) => !inLeft.has(i))
    withMissing(
      { feature: j, threshold: -2, categories: leftIdx.map((i) => codes[i] ?? 0).sort((a, b) => a - b) },
      left,
      right,
      () => rowsOf(leftIdx),
      () => rowsOf(rightIdx),
    )
  }
  if (options.splitter === 'random') {
    // A random non-trivial subset of the codes.
    const u = options.key === undefined ? [0.5] : Array.from(toFlat(uniform(foldIn(options.key, fi), [codes.length])))
    let left = codes.map((_, i) => i).filter((i) => (u[i] ?? 0.5) < 0.5)
    if (left.length === 0 || left.length === codes.length)
      left = [Math.min(codes.length - 1, Math.floor((u[0] ?? 0.5) * codes.length))]
    emit(left)
    return
  }
  if (problem.k > 2 && codes.length <= 10) {
    for (const s of subsetsOf(codes.length)) emit(s)
    return
  }
  // Order by the mean response (regression), the share of the last class (binary) or of the node's majority class.
  const majority = problem.k > 2 ? total.counts.indexOf(maxOf(total.counts)) : problem.k - 1
  const response = stats.map((s) =>
    problem.k > 0 ? (s.counts[majority] ?? 0) / (s.weight || 1) : s.sum / (s.weight || 1),
  )
  const order = codes.map((_, i) => i).sort((a, b) => (response[a] ?? 0) - (response[b] ?? 0))
  for (let p = 1; p < order.length; p++) emit(order.slice(0, p))
}
