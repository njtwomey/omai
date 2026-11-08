/**
 * Histogram gradient boosting (LightGBM's idea; scikit-learn's `HistGradientBoostingRegressor` and
 * `HistGradientBoostingClassifier`): the features are binned once into at most 255 bins, each tree is grown leaf-wise
 * (the open leaf with the largest gain next, up to a leaf limit) from per-bin sums of gradients and hessians instead of
 * sorted values, a node's histogram is its parent's minus its sibling's (the subtraction trick, so only the smaller
 * child is scanned), and missing values get a bin of their own and are sent to whichever side gains more. Splits and
 * leaf values are second-order, gain G_L^2 / (H_L + l2) + G_R^2 / (H_R + l2) - G^2 / (H + l2) and -G / (H + l2). The
 * boosting loop, losses and logs are those of gradient boosting; only the tree grower is different.
 */

import { maxOf } from '../base/index.ts'
import type { MatrixView } from '../learn/validation.ts'
import type { FittedTree } from '../trees/tree.ts'
import {
  type BoostEnv,
  boostingEstimator,
  type GradientBoostingEstimator,
  type GradientBoostingHyper,
  type RegressionLoss,
  type StageGrower,
  type StageTree,
} from './_boosting.ts'

/** The features cut into bins: per feature the thresholds between bins, and the bin of every training value. */
export interface Binning {
  /** `edges[j]` are the thresholds of feature j, ascending; bin b holds values in (edges[b - 1], edges[b]]. */
  readonly edges: readonly (readonly number[])[]
  /** `codes[i * d + j]` is the bin of row i on feature j; missing values are in bin `edges[j].length + 1`. */
  readonly codes: Uint16Array
}

/** The percentile of a sorted column by numpy's `averaged_inverted_cdf` method (the one scikit-learn bins with). */
function invertedCdfPercentile(sorted: ArrayLike<number>, percent: number): number {
  const n = sorted.length
  const virtual = (n * percent) / 100 - 1
  const lo = Math.floor(virtual)
  const gamma = virtual - lo > 0 ? 1 : 0.5
  const a = sorted[Math.min(Math.max(lo, 0), n - 1)] ?? 0
  const b = sorted[Math.min(Math.max(lo + 1, 0), n - 1)] ?? 0
  return a + (b - a) * gamma
}

/**
 * Bin the features as scikit-learn's `_BinMapper`: a feature with at most `maxBins` distinct values gets a bin per value
 * (thresholds at the midpoints), otherwise thresholds at `maxBins - 1` evenly spaced percentiles (numpy's `averaged_inverted_cdf`); a value belongs to the first bin whose threshold it does not exceed, and a missing value to a last,
 * extra bin.
 *
 * @param m - The features, NaN for missing.
 * @param maxBins - The most bins per feature, missing excluded (default 255).
 * @returns The thresholds and the binned data.
 * @example
 * ```ts
 * import { binFeatures } from 'omai-engine/ensembles'
 * import { matrixView } from 'omai-engine/learn'
 * const b = binFeatures(matrixView([[1], [2], [4], [Number.NaN]], 'demo', { allowNaN: true }), 255)
 * b.edges[0] // => [1.5, 3]
 * Array.from(b.codes) // => [0, 1, 2, 3]
 * ```
 */
export function binFeatures(m: MatrixView, maxBins = 255): Binning {
  const edges: number[][] = []
  for (let j = 0; j < m.d; j++) {
    const column = Float64Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0)
      .filter((v) => !Number.isNaN(v))
      .sort()
    const distinct = column.filter((v, i) => i === 0 || v !== column[i - 1])
    let cuts: number[]
    if (distinct.length <= maxBins) {
      cuts = Array.from({ length: Math.max(0, distinct.length - 1) }, (_, i) => {
        return ((distinct[i] ?? 0) + (distinct[i + 1] ?? 0)) / 2
      })
    } else {
      const percents = Array.from({ length: maxBins - 1 }, (_, i) => ((i + 1) * 100) / maxBins)
      cuts = [...new Set(percents.map((p) => invertedCdfPercentile(column, p)))].sort((a, b) => a - b)
    }
    edges.push(cuts)
  }
  const codes = new Uint16Array(m.n * m.d)
  for (let j = 0; j < m.d; j++) {
    const cuts = edges[j] as number[]
    for (let i = 0; i < m.n; i++) {
      const v = m.data[i * m.d + j] ?? 0
      if (Number.isNaN(v)) {
        codes[i * m.d + j] = cuts.length + 1
        continue
      }
      let lo = 0
      let hi = cuts.length
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if ((cuts[mid] ?? 0) < v) lo = mid + 1
        else hi = mid
      }
      codes[i * m.d + j] = lo
    }
  }
  return { edges, codes }
}

/** Per-bin sums over a node's rows: gradient, hessian and count, for every feature. */
interface Histogram {
  readonly g: Float64Array[]
  readonly h: Float64Array[]
  readonly n: Float64Array[]
}

interface BestSplit {
  readonly feature: number
  /** Split after this bin: bins up to it go left. */
  readonly bin: number
  readonly missingLeft: boolean
  readonly gain: number
}

interface HNode {
  readonly id: number
  readonly depth: number
  readonly rows: readonly number[]
  readonly hist: Histogram
  readonly g: number
  readonly h: number
  readonly best: BestSplit | undefined
  left: number
  right: number
  split: BestSplit | undefined
  leftCount: number
  rightCount: number
}

/** The histogram grower for boosting: leaf-wise trees from binned data. */
function histogramGrower({ m, w, loss, h }: BoostEnv): StageGrower {
  const d = m.d
  const binning = binFeatures(m, h.maxBins ?? 255)
  const { edges, codes } = binning
  const missingBin = (j: number): number => (edges[j]?.length ?? 0) + 1
  const lambda = h.l2
  const minLeaf = h.minSamplesLeaf
  const minHessian = h.minHessian ?? 1e-3
  const maxLeaves = h.maxLeafNodes ?? 31
  const refit = h.loss === 'absolute_error' || h.loss === 'quantile' || h.loss === 'huber'
  const sizes = Array.from({ length: d }, (_, j) => missingBin(j) + 1)
  const emptyHistogram = (): Histogram => ({
    g: sizes.map((s) => new Float64Array(s)),
    h: sizes.map((s) => new Float64Array(s)),
    n: sizes.map((s) => new Float64Array(s)),
  })
  return ({ c, bag, residual, column }) => {
    const hess = loss.hessian?.(column, c)
    // scikit-learn keeps gradients and hessians in single precision.
    const grad = Float64Array.from(residual, (r, i) => Math.fround(-r * (w[i] ?? 0)))
    const hessian = Float64Array.from(w, (wi, i) => Math.fround(wi * (hess?.[i] ?? 1)))
    const build = (rows: readonly number[]): Histogram => {
      const hist = emptyHistogram()
      for (const r of rows)
        for (let j = 0; j < d; j++) {
          const b = codes[r * d + j] ?? 0
          ;(hist.g[j] as Float64Array)[b] = ((hist.g[j] as Float64Array)[b] ?? 0) + (grad[r] ?? 0)
          ;(hist.h[j] as Float64Array)[b] = ((hist.h[j] as Float64Array)[b] ?? 0) + (hessian[r] ?? 0)
          ;(hist.n[j] as Float64Array)[b] = ((hist.n[j] as Float64Array)[b] ?? 0) + 1
        }
      return hist
    }
    const bestSplit = (hist: Histogram, total: { g: number; h: number; n: number }): BestSplit | undefined => {
      let best: BestSplit | undefined
      const current = (total.g * total.g) / (total.h + lambda)
      for (let j = 0; j < d; j++) {
        const gj = hist.g[j] as Float64Array
        const hj = hist.h[j] as Float64Array
        const nj = hist.n[j] as Float64Array
        const real = missingBin(j) // bins 0 .. real - 1 hold values
        const hasMissing = (nj[real] ?? 0) > 0
        let gl = 0
        let hl = 0
        let nl = 0
        // Missing values go right: scan the bins left to right.
        for (let b = 0; b < real - 1; b++) {
          gl += gj[b] ?? 0
          hl += hj[b] ?? 0
          nl += nj[b] ?? 0
          if (nl < minLeaf || hl < minHessian) continue
          const nr = total.n - nl
          const hr = total.h - hl
          if (nr < minLeaf || hr < minHessian) break
          const gr = total.g - gl
          const gain = (gl * gl) / (hl + lambda) + (gr * gr) / (hr + lambda) - current
          if (best === undefined ? gain > -1 : gain > best.gain) best = { feature: j, bin: b, missingLeft: false, gain }
        }
        if (!hasMissing) continue
        // Missing values go left: scan from the right.
        let gr = 0
        let hr = 0
        let nr = 0
        for (let b = real - 1; b >= 1; b--) {
          gr += gj[b] ?? 0
          hr += hj[b] ?? 0
          nr += nj[b] ?? 0
          if (nr < minLeaf || hr < minHessian) continue
          const nl2 = total.n - nr
          const hl2 = total.h - hr
          if (nl2 < minLeaf || hl2 < minHessian) break
          const gl2 = total.g - gr
          const gain = (gl2 * gl2) / (hl2 + lambda) + (gr * gr) / (hr + lambda) - current
          if (best === undefined ? gain > -1 : gain > best.gain)
            best = { feature: j, bin: b - 1, missingLeft: true, gain }
        }
      }
      return best !== undefined && best.gain > 0 ? best : undefined
    }
    const make = (id: number, depth: number, rows: readonly number[], hist: Histogram): HNode => {
      let g = 0
      let hh = 0
      for (const r of rows) {
        g += grad[r] ?? 0
        hh += hessian[r] ?? 0
      }
      const best = depth >= h.maxDepth ? undefined : bestSplit(hist, { g, h: hh, n: rows.length })
      return {
        id,
        depth,
        rows,
        hist,
        g,
        h: hh,
        best,
        left: -1,
        right: -1,
        split: undefined,
        leftCount: 0,
        rightCount: 0,
      }
    }
    const nodes: HNode[] = [make(0, 0, bag, build(bag))]
    const open: number[] = nodes[0]?.best === undefined ? [] : [0]
    let leaves = 1
    while (open.length > 0 && leaves < maxLeaves) {
      let pick = 0
      for (let i = 1; i < open.length; i++)
        if ((nodes[open[i] ?? 0]?.best?.gain ?? 0) > (nodes[open[pick] ?? 0]?.best?.gain ?? 0)) pick = i
      const id = open.splice(pick, 1)[0] ?? 0
      const node = nodes[id] as HNode
      const split = node.best as BestSplit
      const miss = missingBin(split.feature)
      const goesLeft = (r: number): boolean => {
        const b = codes[r * d + split.feature] ?? 0
        return b === miss ? split.missingLeft : b <= split.bin
      }
      const leftRows = node.rows.filter(goesLeft)
      const rightRows = node.rows.filter((r) => !goesLeft(r))
      // The histogram subtraction trick: scan the smaller child, subtract it from the parent for the other.
      const small = leftRows.length <= rightRows.length ? leftRows : rightRows
      const smallHist = build(small)
      const bigHist = emptyHistogram()
      for (let j = 0; j < d; j++)
        for (let b = 0; b < (sizes[j] ?? 0); b++) {
          ;(bigHist.g[j] as Float64Array)[b] = (node.hist.g[j]?.[b] ?? 0) - (smallHist.g[j]?.[b] ?? 0)
          ;(bigHist.h[j] as Float64Array)[b] = (node.hist.h[j]?.[b] ?? 0) - (smallHist.h[j]?.[b] ?? 0)
          ;(bigHist.n[j] as Float64Array)[b] = (node.hist.n[j]?.[b] ?? 0) - (smallHist.n[j]?.[b] ?? 0)
        }
      const left = make(nodes.length, node.depth + 1, leftRows, small === leftRows ? smallHist : bigHist)
      const right = make(nodes.length + 1, node.depth + 1, rightRows, small === leftRows ? bigHist : smallHist)
      node.split = split
      node.left = left.id
      node.right = right.id
      node.leftCount = leftRows.length
      node.rightCount = rightRows.length
      nodes.push(left, right)
      leaves++
      for (const k of [left, right]) if (k.best !== undefined) open.push(k.id)
    }
    // Leaves of every training row, walking the bins.
    const leafOf = new Int32Array(m.n)
    for (let r = 0; r < m.n; r++) {
      let id = 0
      for (;;) {
        const node = nodes[id] as HNode
        if (node.split === undefined) break
        const b = codes[r * d + node.split.feature] ?? 0
        id = (b === missingBin(node.split.feature) ? node.split.missingLeft : b <= node.split.bin)
          ? node.left
          : node.right
      }
      leafOf[r] = id
    }
    const value = nodes.map((node) => {
      if (refit && node.split === undefined) {
        const inLeaf = bag.filter((r) => leafOf[r] === node.id)
        if (inLeaf.length > 0) return [loss.leaf(inLeaf, residual, column, c)]
      }
      return [-node.g / (node.h + lambda || 1e-12)]
    })
    const tree: FittedTree = {
      task: 'regression',
      classes: [],
      nFeatures: d,
      feature: nodes.map((n) => n.split?.feature ?? -2),
      threshold: nodes.map((n) => (n.split === undefined ? -2 : (edges[n.split.feature]?.[n.split.bin] ?? 0))),
      left: nodes.map((n) => n.left),
      right: nodes.map((n) => n.right),
      value,
      impurity: nodes.map(() => 0),
      nSamples: nodes.map((n) => n.rows.length),
      weightedNSamples: nodes.map((n) => n.rows.length),
      // With no missing value seen at a split, future ones follow the child with more samples.
      missingLeft: nodes.map((n) => {
        if (n.split === undefined) return false
        const seen = (n.hist.n[n.split.feature]?.[missingBin(n.split.feature)] ?? 0) > 0
        return seen ? n.split.missingLeft : n.leftCount > n.rightCount
      }),
      categories: nodes.map(() => []),
    }
    const bins = maxOf(edges.map((e) => e.length + 1))
    return { tree, leaves: leafOf, bins } satisfies StageTree
  }
}

/** Hyperparameters of the histogram boosters: those of gradient boosting, with leaf-wise growth by default. */
export type HistGradientBoostingHyper = GradientBoostingHyper

const histDefaults = (
  hyper: Partial<GradientBoostingHyper>,
  loss: GradientBoostingHyper['loss'],
): GradientBoostingHyper => ({
  nEstimators: hyper.nEstimators ?? 100,
  learningRate: hyper.learningRate ?? 0.1,
  maxDepth: hyper.maxDepth ?? 1000,
  minSamplesLeaf: hyper.minSamplesLeaf ?? 20,
  minSamplesSplit: 2,
  minImpurityDecrease: 0,
  subsample: 1,
  maxFeatures: 'all',
  maxLeafNodes: hyper.maxLeafNodes ?? 31,
  loss: hyper.loss ?? loss,
  alpha: hyper.alpha ?? 0.5,
  leaves: 'newton',
  l2: hyper.l2 ?? 0,
  validationFraction: hyper.validationFraction ?? 0,
  ...(hyper.nIterNoChange === undefined ? {} : { nIterNoChange: hyper.nIterNoChange }),
  tol: hyper.tol ?? 1e-7,
  maxBins: hyper.maxBins ?? 255,
  minHessian: hyper.minHessian ?? 1e-3,
})

/**
 * Histogram gradient boosting for regression (sklearn.ensemble.HistGradientBoostingRegressor): features binned once,
 * leaf-wise trees from gradient histograms with the subtraction trick, missing values sent to the better side, losses
 * squared, absolute, quantile (`alpha`) and Poisson. A session adds one tree per step; the logs and probes are those of
 * {@link gradientBoostingRegressor} (`boost/trainLoss`, `eval/loss`, `boost/residuals`, `boost/stopped`, ...) and
 * `boost/bins`, the number of bins per feature.
 *
 * @param hyper - Iterations (default 100), learning rate (0.1), leaf limit (31), minimum leaf size (20), L2
 *   penalty, bins (255), loss and early stopping.
 * @returns The estimator.
 * @concept boosting
 * @example
 * ```ts
 * import { histGradientBoostingRegressor } from 'omai-engine/ensembles'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 60])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 1 : 3))
 * const fitted = histGradientBoostingRegressor({ nEstimators: 30, minSamplesLeaf: 5 }).fit({ x, y })
 * toFlat(fitted.predict([[0.2], [0.8]])).map(Math.round) // => [1, 3]
 * ```
 */
export function histGradientBoostingRegressor(
  hyper: Partial<GradientBoostingHyper & { readonly loss: RegressionLoss }> = {},
): GradientBoostingEstimator {
  return boostingEstimator(
    'hist-gradient-boosting-regressor',
    false,
    histDefaults(hyper, 'squared_error'),
    histogramGrower,
  )
}

/**
 * Histogram gradient boosting for classification (sklearn.ensemble.HistGradientBoostingClassifier): the logistic loss,
 * one tree per class per iteration for several classes.
 *
 * @param hyper - As {@link histGradientBoostingRegressor}.
 * @returns The estimator.
 * @concept boosting
 * @index estimator Histogram gradient boosting with leaf-wise growth.
 * @example
 * ```ts
 * import { histGradientBoostingClassifier } from 'omai-engine/ensembles'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 60 }, (_, i) => [i / 60])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * toFlat(histGradientBoostingClassifier({ nEstimators: 20, minSamplesLeaf: 5 }).fit({ x, y }).predict([[0.1], [0.9]])) // => [0, 1]
 * ```
 */
export function histGradientBoostingClassifier(hyper: Partial<GradientBoostingHyper> = {}): GradientBoostingEstimator {
  return boostingEstimator('hist-gradient-boosting-classifier', true, histDefaults(hyper, 'log_loss'), histogramGrower)
}
