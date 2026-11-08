// The machinery shared by the gradient-boosting estimators (exact trees and histograms): the losses, the boosting
// algorithm that adds one stage per step, a stage grower to plug in, the verbs of a fitted model and the estimator.
import { argmaxOf, OmaiError, quantileOf } from '../base/index.ts'
import { categorical, type Distribution } from '../dists/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import { type FittedFacts, type LearnData, type Verbs, withDefaultScore } from '../learn/estimator.ts'
import {
  encodeLabels,
  type MatrixView,
  matrixView,
  rowsOf,
  sampleWeightsOf,
  takeRows,
  targetVector,
} from '../learn/validation.ts'
import { weightedPercentile } from '../metrics/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  event,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { type GrowOptions, growTree, type MaxFeatures, type TreeData } from '../trees/grow.ts'
import { applyTree, type FittedTree, treeFeatureImportances } from '../trees/tree.ts'
import { stagesFromRun } from './_stages.ts'

/** Losses for regression. */
export type RegressionLoss = 'squared_error' | 'absolute_error' | 'huber' | 'quantile' | 'poisson'
/** Losses for classification. */
export type ClassificationLoss = 'log_loss' | 'exponential'

/** Hyperparameters of the gradient-boosting estimators. */
export interface GradientBoostingHyper {
  readonly nEstimators: number
  readonly learningRate: number
  readonly maxDepth: number
  readonly minSamplesLeaf: number
  readonly minSamplesSplit: number
  readonly minImpurityDecrease: number
  /** Fraction of rows each stage's tree sees (stochastic gradient boosting when below 1). */
  readonly subsample: number
  readonly maxFeatures: MaxFeatures
  readonly maxLeafNodes?: number
  readonly loss: RegressionLoss | ClassificationLoss
  /** The quantile of the quantile loss and the Huber cutoff quantile (default 0.9). */
  readonly alpha: number
  /** `friedman`: leaf values by the loss's own line search; `newton`: second-order leaves with penalty `l2`. */
  readonly leaves: 'friedman' | 'newton'
  /** The L2 penalty on leaf values in Newton mode (default 0). */
  readonly l2: number
  /** Fraction of rows held out to stop early (0: none), rounds without improvement allowed, and the tolerance. */
  readonly validationFraction: number
  readonly nIterNoChange?: number
  readonly tol: number
  /** Histogram boosting: the most bins per feature (default 255) and the smallest hessian sum in a leaf. */
  readonly maxBins?: number
  readonly minHessian?: number
}

/** Parameters of a fitted gradient-boosting model: the initial raw predictions and the trees of each stage. */
export type GradientBoostingParams = {
  /** One raw value per output (one for regression and binary problems, K otherwise). */
  readonly init: number[]
  /** `stages[m][k]`: the tree for output k at stage m (leaf values final, before the learning rate). */
  readonly stages: FittedTree[][]
  readonly learningRate: number
  readonly loss: string
  readonly classes: number[]
  readonly nFeatures: number
}

/** Verbs and facts of a fitted gradient-boosting model. */
export interface GradientBoostingVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  /** The raw predictions (log odds or scores for classifiers). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** The raw predictions after each stage, `[stages][m * K]`. */
  readonly stagedDecisionFunction: (x: TensorLike) => Float64Array[]
  readonly nFeaturesIn: number
  readonly featureImportances: readonly number[]
}

/** The state of the boosting algorithm. */
export interface BoostingState {
  /** Raw predictions on all training rows, `[n * k]`. */
  readonly f: Float64Array
  readonly stages: readonly (readonly FittedTree[])[]
  /** After each stage: the loss on the rows the stage was fitted to, on the held-out rows, out of bag. */
  readonly trainLoss: readonly number[]
  readonly validLoss: readonly number[]
  readonly oobLoss: readonly number[]
  readonly bestValid: number
  readonly sinceBest: number
  readonly stopped: boolean
  /** The rows held out for early stopping (1), fixed by the run's key. */
  readonly held: Uint8Array
}

const expit = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))
const EPS = Number.EPSILON

/** The probabilities of raw predictions `[m * k]`: expit for one output (binary), softmax otherwise. */
function probabilities(f: ArrayLike<number>, m: number, k: number, classes: number, loss: string): Float64Array {
  const out = new Float64Array(m * classes)
  for (let i = 0; i < m; i++) {
    if (k === 1) {
      const q = expit((loss === 'exponential' ? 2 : 1) * (f[i] ?? 0))
      out[i * 2] = 1 - q
      out[i * 2 + 1] = q
    } else {
      let max = Number.NEGATIVE_INFINITY
      for (let c = 0; c < k; c++) max = Math.max(max, f[i * k + c] ?? 0)
      let z = 0
      for (let c = 0; c < k; c++) z += Math.exp((f[i * k + c] ?? 0) - max)
      for (let c = 0; c < k; c++) out[i * k + c] = Math.exp((f[i * k + c] ?? 0) - max) / z
    }
  }
  return out
}

/** The raw predictions of the first `stages` stages on rows of x. */
export function rawOf(
  p: GradientBoostingParams,
  x: TensorLike,
  stages: number = p.stages.length,
): { readonly m: number; readonly f: Float64Array } {
  const m = matrixView(x, 'gradientBoosting', { nFeatures: p.nFeatures, allowNaN: true }).n
  const k = p.init.length
  const f = new Float64Array(m * k)
  for (let i = 0; i < m; i++) for (let c = 0; c < k; c++) f[i * k + c] = p.init[c] ?? 0
  for (const stage of p.stages.slice(0, stages))
    for (const [c, tree] of stage.entries()) {
      const leaves = applyTree(tree, x)
      for (let i = 0; i < m; i++)
        f[i * k + c] = (f[i * k + c] ?? 0) + p.learningRate * (tree.value[leaves[i] ?? 0]?.[0] ?? 0)
    }
  return { m, f }
}

export function boostingVerbs(p: GradientBoostingParams): GradientBoostingVerbs {
  const classifier = p.classes.length > 0
  const k = p.init.length
  const proba = (x: TensorLike): Tensor => {
    const { m, f } = rawOf(p, x)
    return fromStorage(probabilities(f, m, k, p.classes.length, p.loss), [m, p.classes.length], 'float64')
  }
  const importances = new Array<number>(p.nFeatures).fill(0)
  const trees = p.stages.flat()
  for (const t of trees)
    for (const [j, v] of treeFeatureImportances(t).entries()) importances[j] = (importances[j] ?? 0) + v / trees.length
  const common = {
    decisionFunction: (x: TensorLike) => {
      const { m, f } = rawOf(p, x)
      return k === 1 ? fromStorage(f, [m], 'float64') : fromStorage(f, [m, k], 'float64')
    },
    stagedDecisionFunction: (x: TensorLike) => p.stages.map((_, s) => rawOf(p, x, s + 1).f),
    nFeaturesIn: p.nFeatures,
    featureImportances: importances,
  }
  if (!classifier)
    return withDefaultScore('regressor', {
      ...common,
      predict: (x: TensorLike) => {
        const { m, f } = rawOf(p, x)
        return fromStorage(p.loss === 'poisson' ? f.map(Math.exp) : f, [m], 'float64')
      },
    })
  return withDefaultScore('classifier', {
    ...common,
    predict: (x: TensorLike) => {
      const { m, f } = rawOf(p, x)
      const pr = probabilities(f, m, k, p.classes.length, p.loss)
      const kc = p.classes.length
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(pr.subarray(i * kc, (i + 1) * kc))] ?? 0),
        [m],
        'float64',
      )
    },
    predictProba: proba,
    predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }),
    classes: p.classes,
  })
}

/** A loss: its raw outputs, initial prediction, negative gradient, hessian, leaf value and value. */
export interface LossSpec {
  /** Raw outputs per row. */
  readonly k: number
  init(): number[]
  /** The negative gradient for output c at the raw predictions `f`. */
  gradient(f: Float64Array, c: number): Float64Array
  /** The hessian for output c (undefined: the loss has no Newton form). */
  hessian?(f: Float64Array, c: number): Float64Array
  /** The leaf value Friedman's line search gives for the in-bag rows of a leaf. */
  leaf(rows: readonly number[], residual: Float64Array, f: Float64Array, c: number): number
  /** The mean loss over some rows. */
  loss(f: Float64Array, rows: readonly number[]): number
}

export function lossSpec(
  h: GradientBoostingHyper,
  y: ArrayLike<number>,
  w: ArrayLike<number>,
  classes: number,
): LossSpec {
  const n = y.length
  const kind = h.loss
  const ws = (rows: readonly number[]): number[] => rows.map((r) => w[r] ?? 0)
  const sumW = (rows: readonly number[]): number => rows.reduce((a, r) => a + (w[r] ?? 0), 0)
  const unweighted = Array.from(w).every((v) => v === w[0])
  if (classes > 0) {
    const k = classes === 2 ? 1 : classes
    if (kind === 'exponential') {
      if (classes !== 2) throw new OmaiError('value', 'gradientBoosting: the exponential loss needs two classes')
      const sign = (i: number): number => 2 * (y[i] ?? 0) - 1
      return {
        k,
        init: () => {
          let pos = 0
          let total = 0
          for (let i = 0; i < n; i++) {
            total += w[i] ?? 0
            if (y[i] === 1) pos += w[i] ?? 0
          }
          const p = Math.min(Math.max(pos / total, EPS), 1 - EPS)
          return [0.5 * Math.log(p / (1 - p))]
        },
        gradient: (f) => Float64Array.from({ length: n }, (_, i) => sign(i) * Math.exp(-sign(i) * (f[i] ?? 0))),
        hessian: (f) => Float64Array.from({ length: n }, (_, i) => Math.exp(-sign(i) * (f[i] ?? 0))),
        leaf: (rows, _r, f) => {
          let num = 0
          let den = 0
          for (const r of rows) {
            const e = Math.exp(-sign(r) * (f[r] ?? 0))
            num += (w[r] ?? 0) * sign(r) * e
            den += (w[r] ?? 0) * e
          }
          return den < 1e-150 ? 0 : num / den
        },
        loss: (f, rows) => {
          let s = 0
          for (const r of rows) s += (w[r] ?? 0) * Math.exp(-sign(r) * (f[r] ?? 0))
          return s / sumW(rows)
        },
      }
    }
    return {
      k,
      init: () => {
        const p = new Array<number>(classes).fill(0)
        let total = 0
        for (let i = 0; i < n; i++) {
          p[y[i] ?? 0] = (p[y[i] ?? 0] ?? 0) + (w[i] ?? 0)
          total += w[i] ?? 0
        }
        const probs = p.map((v) => Math.min(Math.max(v / total, EPS), 1 - EPS))
        return k === 1 ? [Math.log((probs[1] ?? 0.5) / (probs[0] ?? 0.5))] : probs.map(Math.log)
      },
      gradient: (f, c) => {
        const pr = probabilities(f, n, k, classes, kind)
        return Float64Array.from({ length: n }, (_, i) => {
          const target = k === 1 ? (y[i] === 1 ? 1 : 0) : y[i] === c ? 1 : 0
          return target - (pr[i * classes + (k === 1 ? 1 : c)] ?? 0)
        })
      },
      hessian: (f, c) => {
        const pr = probabilities(f, n, k, classes, kind)
        return Float64Array.from({ length: n }, (_, i) => {
          const q = pr[i * classes + (k === 1 ? 1 : c)] ?? 0
          return q * (1 - q)
        })
      },
      leaf: (rows, residual) => {
        let num = 0
        let den = 0
        for (const r of rows) {
          const res = residual[r] ?? 0
          num += (w[r] ?? 0) * res
          den += (w[r] ?? 0) * Math.abs(res) * (1 - Math.abs(res))
        }
        if (k > 1) num *= (k - 1) / k
        return Math.abs(den) < 1e-150 ? 0 : num / den
      },
      loss: (f, rows) => {
        const pr = probabilities(f, n, k, classes, kind)
        let s = 0
        for (const r of rows) s -= (w[r] ?? 0) * Math.log(Math.max(pr[r * classes + (y[r] ?? 0)] ?? 0, 1e-300))
        return s / sumW(rows)
      },
    }
  }
  // Regression.
  const alpha = h.alpha
  let gamma = 0
  const diffs = (rows: readonly number[], f: Float64Array): number[] => rows.map((r) => (y[r] ?? 0) - (f[r] ?? 0))
  return {
    k: 1,
    init: () => {
      const all = Array.from(y)
      const wl = Array.from(w)
      const total = wl.reduce((a, b) => a + b, 0)
      if (kind === 'squared_error') return [all.reduce((a, v, i) => a + v * (wl[i] ?? 0), 0) / total]
      if (kind === 'poisson') return [Math.log(all.reduce((a, v, i) => a + v * (wl[i] ?? 0), 0) / total)]
      const level = kind === 'quantile' ? alpha : 0.5
      return [unweighted ? quantileOf(all, level) : weightedPercentile(all, wl, level * 100, false)]
    },
    gradient: (f) => {
      const d = Float64Array.from({ length: n }, (_, i) => (y[i] ?? 0) - (f[i] ?? 0))
      if (kind === 'squared_error') return d
      if (kind === 'absolute_error') return d.map(Math.sign)
      if (kind === 'quantile') return d.map((v) => (v > 0 ? alpha : -(1 - alpha)))
      if (kind === 'poisson') return Float64Array.from({ length: n }, (_, i) => (y[i] ?? 0) - Math.exp(f[i] ?? 0))
      gamma = weightedPercentile(Array.from(d, Math.abs), Array.from(w), alpha * 100, false)
      return d.map((v) => (Math.abs(v) <= gamma ? v : gamma * Math.sign(v)))
    },
    ...(kind === 'squared_error'
      ? { hessian: () => new Float64Array(n).fill(1) }
      : kind === 'poisson'
        ? { hessian: (f: Float64Array) => Float64Array.from(f, (v) => Math.exp(v)) }
        : {}),
    leaf: (rows, residual, f) => {
      const weights = ws(rows)
      if (kind === 'squared_error')
        return (
          rows.reduce((a, r, i) => a + (residual[r] ?? 0) * (weights[i] ?? 0), 0) / weights.reduce((a, b) => a + b, 0)
        )
      if (kind === 'poisson') {
        let num = 0
        let den = 0
        for (const r of rows) {
          num += (w[r] ?? 0) * (residual[r] ?? 0)
          den += (w[r] ?? 0) * Math.exp(f[r] ?? 0)
        }
        return den < 1e-150 ? 0 : num / den
      }
      const d = diffs(rows, f)
      // Leaf values use scikit-learn's weighted percentile (the first value reaching the rank), not the interpolation.
      const at = (level: number): number => weightedPercentile(d, weights, level * 100, false)
      if (kind === 'absolute_error') return at(0.5)
      if (kind === 'quantile') return at(alpha)
      const median = at(0.5)
      const total = weights.reduce((a, b) => a + b, 0)
      return (
        median +
        d.reduce(
          (a, v, i) => a + (weights[i] ?? 0) * Math.sign(v - median) * Math.min(Math.abs(v - median), gamma),
          0,
        ) /
          total
      )
    },
    loss: (f, rows) => {
      let s = 0
      let sw = 0
      for (const r of rows) {
        const d = (y[r] ?? 0) - (f[r] ?? 0)
        const wi = w[r] ?? 0
        sw += wi
        if (kind === 'squared_error') s += wi * d * d
        else if (kind === 'absolute_error') s += wi * Math.abs(d)
        else if (kind === 'quantile') s += wi * (d > 0 ? alpha * d : (alpha - 1) * d)
        else if (kind === 'poisson') s += wi * (Math.exp(f[r] ?? 0) - (y[r] ?? 0) * (f[r] ?? 0))
        else s += wi * (Math.abs(d) <= gamma ? 0.5 * d * d : gamma * (Math.abs(d) - gamma / 2))
      }
      return s / sw
    },
  }
}

/** What a stage's tree is grown from: the output, the rows in the bag, the gradient, the raw predictions, a key. */
export interface StageInput {
  readonly c: number
  readonly bag: readonly number[]
  /** The negative gradient of every row. */
  readonly residual: Float64Array
  /** The raw predictions of all rows and outputs. */
  readonly column: Float64Array
  readonly key: Key
}

/** The tree of a stage with its final leaf values (before the learning rate), and the leaf of every training row. */
export interface StageTree {
  readonly tree: FittedTree
  readonly leaves: Int32Array
  /** Histogram growers: the bins per feature, for the log `boost/bins`. */
  readonly bins?: number
}

/** Grows one stage's tree. */
export type StageGrower = (input: StageInput) => StageTree

/** What a grower is built from. */
export interface BoostEnv {
  readonly m: MatrixView
  readonly w: Float64Array
  readonly y: ArrayLike<number>
  readonly loss: LossSpec
  readonly h: GradientBoostingHyper
}

/** Builds the grower of a run. */
export type GrowerFactory = (env: BoostEnv) => StageGrower

/** The exact grower: the library's tree grower on the gradient, then the loss's own leaf values (or Newton's). */
const exactGrower: GrowerFactory = ({ m, w, loss, h }) => {
  const n = m.n
  const xs = fromStorage(m.data, [n, m.d], 'float64')
  const grow: GrowOptions = {
    criterion: 'friedman_mse',
    splitter: 'best',
    growth: 'depth',
    maxDepth: h.maxDepth,
    minSamplesSplit: h.minSamplesSplit,
    minSamplesLeaf: h.minSamplesLeaf,
    minWeightFractionLeaf: 0,
    maxFeatures: h.maxFeatures,
    ...(h.maxLeafNodes === undefined ? {} : { maxLeafNodes: h.maxLeafNodes }),
    minImpurityDecrease: h.minImpurityDecrease,
    record: 0,
  }
  return ({ c, bag, residual, column, key }) => {
    const hess = h.leaves === 'newton' ? (loss.hessian as NonNullable<LossSpec['hessian']>)(column, c) : undefined
    const target = hess === undefined ? residual : residual.map((g, i) => g / Math.max(hess[i] ?? 0, 1e-12))
    const weight = hess === undefined ? w : Float64Array.from(w, (v, i) => v * Math.max(hess[i] ?? 0, 1e-12))
    const sub: TreeData =
      bag.length === n
        ? { x: m, y: target, w: weight, classes: [] }
        : {
            x: takeRows(m, bag),
            y: Float64Array.from(bag, (r) => target[r] ?? 0),
            w: Float64Array.from(bag, (r) => weight[r] ?? 0),
            classes: [],
          }
    const grown = growTree(sub, grow, { key }).tree
    const leaves = applyTree(grown, xs)
    const byLeaf = new Map<number, number[]>()
    for (const r of bag) {
      const l = leaves[r] ?? 0
      const list = byLeaf.get(l)
      if (list === undefined) byLeaf.set(l, [r])
      else list.push(r)
    }
    const value = grown.value.map((v, node) => {
      const rows = byLeaf.get(node)
      if (rows === undefined) return v
      if (h.leaves === 'newton') {
        let g = 0
        let hh = 0
        for (const r of rows) {
          g += (w[r] ?? 0) * (residual[r] ?? 0)
          hh += (w[r] ?? 0) * (hess?.[r] ?? 0)
        }
        return [g / (hh + h.l2 || 1e-12)]
      }
      return [loss.leaf(rows, residual, column, c)]
    })
    return { tree: { ...grown, value }, leaves }
  }
}

/** Boosting as an algorithm, one stage per step. */
export function boostingAlgorithm(
  data: LearnData,
  h: GradientBoostingHyper,
  classifier: boolean,
  name: string,
  makeGrower: GrowerFactory = exactGrower,
): Algorithm<BoostingState, BoostingPath> {
  const m = matrixView(data.x, name, { allowNaN: true })
  const n = m.n
  if (data.y === undefined) throw new OmaiError('value', `${name}: needs targets y`)
  const labels = classifier ? encodeLabels(data.y, name, n) : undefined
  const y = labels === undefined ? targetVector(data.y, name, n) : Float64Array.from(labels.codes)
  const w = sampleWeightsOf(data.weights, n, name)
  const classes = labels?.classes.length ?? 0
  if (classifier && classes < 2) throw new OmaiError('value', `${name}: needs two classes`)
  const loss = lossSpec(h, y, w, classes)
  if (h.leaves === 'newton' && loss.hessian === undefined)
    throw new OmaiError('value', `${name}: Newton leaves need a smooth loss, not ${h.loss}`)
  const k = loss.k
  const init = loss.init()
  const grower = makeGrower({ m, w, y, loss, h })
  const early = h.nIterNoChange !== undefined && h.validationFraction > 0
  return {
    name,
    config: { nEstimators: h.nEstimators, learningRate: h.learningRate, loss: h.loss, leaves: h.leaves },
    init: (key) => {
      const f = new Float64Array(n * k)
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) f[i * k + c] = init[c] ?? 0
      const held = new Uint8Array(n)
      if (early)
        for (const r of toFlat(permutation(foldIn(key, 7), n)).slice(0, Math.floor(h.validationFraction * n)))
          held[r] = 1
      return {
        f,
        stages: [],
        trainLoss: [],
        validLoss: [],
        oobLoss: [],
        bestValid: Number.MAX_VALUE,
        sinceBest: 0,
        stopped: false,
        held,
      }
    },
    step: (s, ctx) => {
      const held = s.held
      const eligible = Array.from({ length: n }, (_, i) => i).filter((i) => !held[i])
      const nBag = Math.max(1, Math.floor(h.subsample * eligible.length))
      const bag =
        h.subsample < 1
          ? Array.from(
              toFlat(permutation(foldIn(ctx.key, 0), eligible.length)).slice(0, nBag),
              (j) => eligible[j] ?? 0,
            ).sort((a, b) => a - b)
          : eligible
      const inBag = new Uint8Array(n)
      for (const r of bag) inBag[r] = 1
      const f = Float64Array.from(s.f)
      const stage: FittedTree[] = []
      let firstResidual: Float64Array | undefined
      let leafValues: number[] = []
      let bins: number | undefined
      for (let c = 0; c < k; c++) {
        const column = s.f
        const residual = loss.gradient(column, c)
        if (c === 0) firstResidual = residual
        const { tree, leaves, bins: binsUsed } = grower({ c, bag, residual, column, key: foldIn(ctx.key, c + 1) })
        if (binsUsed !== undefined) bins = binsUsed
        const value = tree.value
        stage.push(tree)
        if (c === 0) leafValues = value.filter((_, node) => tree.left[node] === -1).map((v) => v[0] ?? 0)
        for (let i = 0; i < n; i++)
          f[i * k + c] = (f[i * k + c] ?? 0) + h.learningRate * (value[leaves[i] ?? 0]?.[0] ?? 0)
      }
      const rowsOf = (pred: (i: number) => boolean): number[] => Array.from({ length: n }, (_, i) => i).filter(pred)
      const fitRows = rowsOf((i) => inBag[i] === 1)
      const outRows = rowsOf((i) => !held[i] && !inBag[i])
      const heldRows = rowsOf((i) => held[i] === 1)
      const train = loss.loss(f, fitRows)
      const valid = heldRows.length > 0 ? loss.loss(f, heldRows) : Number.NaN
      const oob = outRows.length > 0 ? loss.loss(f, outRows) : Number.NaN
      const round = s.stages.length + 1
      const improved = early && valid < s.bestValid - h.tol
      const sinceBest = early ? (improved ? 0 : s.sinceBest + 1) : 0
      const stopped = early && sinceBest >= (h.nIterNoChange ?? Number.POSITIVE_INFINITY)
      if (ctx.wants('boost/round')) ctx.log('boost/round', round)
      if (ctx.wants('ensemble/weight')) ctx.log('ensemble/weight', h.learningRate)
      if (ctx.wants('boost/trainLoss')) ctx.log('boost/trainLoss', train)
      if (ctx.wants('boost/validLoss') && (Number.isFinite(valid) || Number.isFinite(oob)))
        ctx.log('boost/validLoss', Number.isFinite(valid) ? valid : oob)
      if (ctx.wants('eval/loss'))
        ctx.log('eval/loss', Number.isFinite(valid) ? valid : Number.isFinite(oob) ? oob : train)
      if (ctx.wants('boost/residuals') && firstResidual !== undefined) ctx.log('boost/residuals', firstResidual)
      if (ctx.wants('boost/leafValues')) ctx.log('boost/leafValues', leafValues)
      if (ctx.wants('boost/bins') && bins !== undefined) ctx.log('boost/bins', bins)
      if (stopped) ctx.emit('boost/stopped', { round, validLoss: valid, bestValid: Math.min(s.bestValid, valid) })
      return {
        f,
        stages: [...s.stages, stage],
        trainLoss: [...s.trainLoss, train],
        validLoss: Number.isFinite(valid) ? [...s.validLoss, valid] : s.validLoss,
        oobLoss: Number.isFinite(oob) ? [...s.oobLoss, oob] : s.oobLoss,
        bestValid: improved ? valid : s.bestValid,
        sinceBest,
        stopped,
        held,
      }
    },
    done: (s) => (s.stopped || s.stages.length >= h.nEstimators ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BoostingState,
  }
}

/**
 * The paths of gradient boosting (all the variants): the stage's loss, residuals, leaf values and bins, early stopping
 * (`boost/stopped`) and the derived `model/stagedPrediction`.
 *
 * @example
 * ```ts
 * import type { BoostingPath } from 'omai-engine/ensembles'
 * const path: BoostingPath = 'boost/trainLoss'
 * path // => 'boost/trainLoss'
 * ```
 */
export type BoostingPath =
  | 'boost/round'
  | 'ensemble/weight'
  | 'boost/trainLoss'
  | 'boost/validLoss'
  | 'eval/loss'
  | 'boost/residuals'
  | 'boost/bins'
  | 'boost/leafValues'
  | 'boost/stopped'
  | 'model/stagedPrediction'

/** A gradient-boosting estimator. */
export type GradientBoostingEstimator = IterativeEstimator<
  GradientBoostingHyper,
  BoostingState,
  GradientBoostingParams,
  GradientBoostingVerbs,
  BoostingPath
>

export function boostingEstimator(
  name: string,
  classifier: boolean,
  defaults: GradientBoostingHyper,
  makeGrower: GrowerFactory = exactGrower,
): GradientBoostingEstimator {
  const classesOf = (data: LearnData): readonly number[] => (classifier ? encodeLabels(data.y ?? [], name).classes : [])
  return iterative<GradientBoostingHyper, BoostingState, GradientBoostingParams, GradientBoostingVerbs, BoostingPath>(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      defaults,
      hyperSpace: (h) =>
        dict({
          nEstimators: int(1, Math.max(500, h.nEstimators), { default: h.nEstimators, label: 'stages' }),
          learningRate: real(0.01, Math.max(1, h.learningRate), {
            log: true,
            default: h.learningRate,
            label: 'learning rate',
          }),
          maxDepth: int(1, Math.max(8, h.maxDepth), { default: h.maxDepth, label: 'tree depth' }),
          subsample: real(0.1, 1, {
            label: 'subsample',
            description: 'The fraction of rows each tree is fitted on (below 1: stochastic gradient boosting).',
            default: h.subsample,
          }),
          leaves: choice(['friedman', 'newton'] as const, { default: h.leaves, label: 'leaf values' }),
          loss: classifier
            ? choice(['log_loss', 'exponential'] as const, { default: h.loss as ClassificationLoss })
            : choice(['squared_error', 'absolute_error', 'huber', 'quantile', 'poisson'] as const, {
                default: h.loss as RegressionLoss,
              }),
        }),
      algorithm: (data, h) => boostingAlgorithm(data, h, classifier, name, makeGrower),
      params: (state, data, h) => {
        const w = sampleWeightsOf(data.weights, rowsOf(data.x), name)
        const labels = classifier ? encodeLabels(data.y ?? [], name) : undefined
        const y = labels === undefined ? targetVector(data.y ?? [], name) : Float64Array.from(labels.codes)
        return {
          init: lossSpec(h, y, w, classesOf(data).length).init(),
          stages: state.stages.map((s) => [...s]),
          learningRate: h.learningRate,
          loss: h.loss,
          classes: [...classesOf(data)],
          nFeatures: matrixView(data.x, name, { allowNaN: true }).d,
        }
      },
      verbs: (p) => boostingVerbs(p),
      maxSteps: (h) => h.nEstimators,
      contract: (data, h): Contract<BoostingState, BoostingPath> =>
        defineContract({
          'boost/round': scalar({ description: 'stages so far' }),
          'ensemble/weight': scalar({ description: 'the learning rate each stage is added with' }),
          'boost/trainLoss': scalar({
            description: 'the loss on the rows the stage was fitted to',
            direction: 'lower',
          }),
          'boost/validLoss': scalar({
            description: 'the loss on held-out rows (or out of bag, when subsampling)',
            direction: 'lower',
          }),
          'eval/loss': scalar({
            description: 'the held-out loss when there is one, the training loss otherwise',
            direction: 'lower',
          }),
          'boost/residuals': tensor({
            role: 'output',
            shape: ['n'],
            description: 'the negative gradient the stage’s tree was fitted to',
          }),
          'boost/bins': scalar({ description: 'bins per feature (histogram boosting)' }),
          'boost/leafValues': tensor({
            role: 'output',
            shape: ['leaves'],
            description: 'the leaf values of the stage’s tree',
          }),
          'boost/stopped': event({ description: 'early stopping: the held-out loss stopped improving' }),
          'model/stagedPrediction': derived<BoostingState>({
            description:
              'the model’s raw prediction at this stage on given points (args: x, default the training rows)',
            compute: (state, args) => {
              const a = (args ?? {}) as { readonly x?: number[][] }
              const y = classifier
                ? Float64Array.from(encodeLabels(data.y ?? [], name).codes)
                : targetVector(data.y ?? [], name)
              const p: GradientBoostingParams = {
                init: lossSpec(h, y, sampleWeightsOf(data.weights, y.length, name), classesOf(data).length).init(),
                stages: state.stages.map((s) => [...s]),
                learningRate: h.learningRate,
                loss: h.loss,
                classes: [...classesOf(data)],
                nFeatures: matrixView(data.x, name, { allowNaN: true }).d,
              }
              return Array.from(rawOf(p, a.x ?? data.x).f)
            },
          }),
        }),
      projections: {
        'ensemble/stages': (run) =>
          stagesFromRun(run, {
            method: name,
            scoreName: 'training loss',
            higherIsBetter: false,
            weight: 'ensemble/weight',
            train: 'boost/trainLoss',
            valid: 'boost/validLoss',
            target: 'boost/residuals',
          }) as Exposition | undefined,
      },
      logs: {
        'boost/round': Log(),
        'ensemble/weight': Log(),
        'boost/trainLoss': Log(),
        'boost/validLoss': Log(),
        'eval/loss': Log(),
        'boost/residuals': Log({ detail: 'full' }),
        'boost/stopped': Log(),
      },
    },
    defaults,
  )
}
