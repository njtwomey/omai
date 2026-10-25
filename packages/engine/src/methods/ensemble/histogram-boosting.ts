/**
 * Histogram gradient boosting (the basics of scikit-learn's `HistGradientBoosting*`, the LightGBM approach): features
 * are binned once into at most 255 quantile bins so split finding works on histograms, missing values get their own bin
 * and a learned direction, and trees are grown with second-order (Newton) leaf values, traced iteration by iteration.
 */

import { quantileOf, WhiteboxError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type EnsembleStages, type Exposition, ensembleStages } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  type MatrixView,
  matrixView,
  sampleWeightsOf,
  stagedSteps,
  takeRows,
  targetVector,
} from '../../lib/learn/index.ts'
import { permutation } from '../../lib/random/index.ts'
import { bool, dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'

// TODO: categorical features, monotonic constraints, interaction constraints.

/** Hyperparameters of the histogram gradient-boosting estimators. */
export interface HistBoostingHyper {
  readonly maxIter: number
  readonly learningRate: number
  readonly maxLeafNodes: number
  readonly maxDepth?: number
  readonly minSamplesLeaf: number
  readonly l2Regularization: number
  /** Bins per feature (at most 255; missing values get one more). */
  readonly maxBins: number
  /** Hold out a keyed validation set and stop when its loss stops improving. */
  readonly earlyStopping: boolean
  readonly validationFraction: number
  readonly nIterNoChange: number
  readonly tol: number
}

/** A histogram tree: thresholds on raw values, where missing values go, and leaf values (shrinkage applied). */
export type HistTree = {
  readonly feature: number[]
  readonly threshold: number[]
  readonly missingLeft: boolean[]
  readonly left: number[]
  readonly right: number[]
  readonly value: number[]
}

/** Parameters of a fitted histogram booster. */
export type HistBoostingParams = {
  readonly baseline: number[]
  /** iterations[m][k]: the tree for output k at iteration m. */
  readonly iterations: HistTree[][]
  readonly classes: number[]
  readonly nFeatures: number
}

/** Verbs of a fitted histogram booster. */
export interface HistBoostingVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly classes?: number[]
  readonly nFeaturesIn: number
}

/** A fitted histogram booster. */
export type FittedHistBoosting = Fitted<HistBoostingParams> & HistBoostingVerbs

/**
 * Bin edges per feature (scikit-learn's `_BinMapper`): midpoints of the distinct values when there are at most
 * `maxBins` of them, else interior quantiles.
 */
function binEdges(x: MatrixView, maxBins: number): number[][] {
  return Array.from({ length: x.d }, (_, j) => {
    const col: number[] = []
    for (let i = 0; i < x.n; i++) {
      const v = x.data[i * x.d + j] ?? Number.NaN
      if (!Number.isNaN(v)) col.push(v)
    }
    const distinct = Array.from(new Set(col)).sort((a, b) => a - b)
    if (distinct.length <= maxBins) return distinct.slice(1).map((v, i) => ((distinct[i] ?? 0) + v) / 2)
    const edges = Array.from({ length: maxBins - 1 }, (_, b) => quantileOf(col, (b + 1) / maxBins))
    return Array.from(new Set(edges)).sort((a, b) => a - b)
  })
}

function binOf(edges: readonly number[], v: number): number {
  if (Number.isNaN(v)) return -1
  let lo = 0
  let hi = edges.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (v <= (edges[mid] ?? 0)) hi = mid
    else lo = mid + 1
  }
  return lo
}

function predictTree(t: HistTree, row: ArrayLike<number>): number {
  let node = 0
  while ((t.left[node] ?? -1) >= 0) {
    const v = row[t.feature[node] ?? 0] ?? Number.NaN
    const left = Number.isNaN(v) ? Boolean(t.missingLeft[node]) : v <= (t.threshold[node] ?? 0)
    node = left ? (t.left[node] ?? 0) : (t.right[node] ?? 0)
  }
  return t.value[node] ?? 0
}

function rawOf(p: HistBoostingParams, x: TensorLike): { readonly m: number; readonly f: Float64Array } {
  const v = matrixView(x, 'histGradientBoosting', { nFeatures: p.nFeatures, allowNaN: true })
  const k = p.baseline.length
  const f = new Float64Array(v.n * k)
  for (let i = 0; i < v.n; i++) {
    const row = v.data.subarray(i * v.d, (i + 1) * v.d)
    for (let c = 0; c < k; c++) {
      let s = p.baseline[c] ?? 0
      for (const it of p.iterations) s += predictTree(it[c] as HistTree, row)
      f[i * k + c] = s
    }
  }
  return { m: v.n, f }
}

/** Gradient and hessian of the loss, and the loss itself, for raw predictions `f` `[n * k]`. */
interface HistLoss {
  readonly k: number
  baseline(y: Float64Array, w: Float64Array): number[]
  grad(f: Float64Array, y: Float64Array, c: number, out: { g: Float64Array; h: Float64Array }): void
  loss(f: Float64Array, y: Float64Array, w: Float64Array, rows: readonly number[]): number
}

const softmaxRow = (f: Float64Array, i: number, k: number): number[] => {
  let max = Number.NEGATIVE_INFINITY
  for (let c = 0; c < k; c++) max = Math.max(max, f[i * k + c] ?? 0)
  const e = Array.from({ length: k }, (_, c) => Math.exp((f[i * k + c] ?? 0) - max))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

function lossOf(classes: number): HistLoss {
  if (classes === 0)
    return {
      k: 1,
      baseline: (y, w) => [y.reduce((a, v, i) => a + v * (w[i] ?? 0), 0) / w.reduce((a, b) => a + b, 0)],
      grad: (f, y, _c, out) => {
        for (let i = 0; i < y.length; i++) {
          out.g[i] = (f[i] ?? 0) - (y[i] ?? 0)
          out.h[i] = 1
        }
      },
      loss: (f, y, w, rows) => {
        let s = 0
        let sw = 0
        for (const r of rows) {
          s += (w[r] ?? 0) * 0.5 * ((f[r] ?? 0) - (y[r] ?? 0)) ** 2
          sw += w[r] ?? 0
        }
        return s / sw
      },
    }
  const k = classes === 2 ? 1 : classes
  return {
    k,
    baseline: (y, w) => {
      const p = new Array<number>(classes).fill(0)
      let total = 0
      y.forEach((c, i) => {
        p[c] = (p[c] ?? 0) + (w[i] ?? 0)
        total += w[i] ?? 0
      })
      const q = p.map((v) => Math.min(Math.max(v / total, 1e-15), 1 - 1e-15))
      return k === 1 ? [Math.log((q[1] ?? 0.5) / (q[0] ?? 0.5))] : q.map(Math.log)
    },
    grad: (f, y, c, out) => {
      for (let i = 0; i < y.length; i++) {
        const p = k === 1 ? 1 / (1 + Math.exp(-(f[i] ?? 0))) : (softmaxRow(f, i, k)[c] ?? 0)
        const target = k === 1 ? (y[i] === 1 ? 1 : 0) : y[i] === c ? 1 : 0
        out.g[i] = p - target
        out.h[i] = Math.max(p * (1 - p), 1e-16)
      }
    },
    loss: (f, y, w, rows) => {
      let s = 0
      let sw = 0
      for (const r of rows) {
        const c = y[r] ?? 0
        const p = k === 1 ? (c === 1 ? 1 : -1) * (f[r] ?? 0) : 0
        const nll = k === 1 ? Math.log1p(Math.exp(-p)) : -Math.log(Math.max(softmaxRow(f, r, k)[c] ?? 0, 1e-300))
        s += (w[r] ?? 0) * nll
        sw += w[r] ?? 0
      }
      return s / sw
    },
  }
}

interface Split {
  readonly gain: number
  readonly feature: number
  readonly bin: number
  readonly missingLeft: boolean
}

/** Grow one histogram tree best-first on gradients and hessians of the training rows. */
function growHistTree(
  bins: Int32Array,
  d: number,
  edges: readonly number[][],
  rows: readonly number[],
  g: Float64Array,
  h: Float64Array,
  w: Float64Array,
  hp: HistBoostingHyper,
): HistTree {
  const lambda = hp.l2Regularization
  const score = (G: number, H: number): number => (G * G) / (H + lambda)
  const tree = {
    feature: [] as number[],
    threshold: [] as number[],
    missingLeft: [] as boolean[],
    left: [] as number[],
    right: [] as number[],
    value: [] as number[],
  }
  const sums = (rs: readonly number[]): { G: number; H: number } => {
    let G = 0
    let H = 0
    for (const r of rs) {
      G += (w[r] ?? 0) * (g[r] ?? 0)
      H += (w[r] ?? 0) * (h[r] ?? 0)
    }
    return { G, H }
  }
  const bestSplit = (rs: readonly number[]): Split | undefined => {
    const { G, H } = sums(rs)
    let best: Split | undefined
    for (let j = 0; j < d; j++) {
      const nb = (edges[j]?.length ?? 0) + 1
      const hg = new Float64Array(nb)
      const hh = new Float64Array(nb)
      const hc = new Int32Array(nb)
      let mg = 0
      let mh = 0
      let mc = 0
      for (const r of rs) {
        const b = bins[r * d + j] ?? -1
        const gw = (w[r] ?? 0) * (g[r] ?? 0)
        const hw = (w[r] ?? 0) * (h[r] ?? 0)
        if (b < 0) {
          mg += gw
          mh += hw
          mc++
        } else {
          hg[b] = (hg[b] ?? 0) + gw
          hh[b] = (hh[b] ?? 0) + hw
          hc[b] = (hc[b] ?? 0) + 1
        }
      }
      let gl = 0
      let hl = 0
      let cl = 0
      for (let b = 0; b < nb - 1; b++) {
        gl += hg[b] ?? 0
        hl += hh[b] ?? 0
        cl += hc[b] ?? 0
        for (const missingLeft of mc > 0 ? [false, true] : [false]) {
          const GL = gl + (missingLeft ? mg : 0)
          const HL = hl + (missingLeft ? mh : 0)
          const CL = cl + (missingLeft ? mc : 0)
          const CR = rs.length - CL
          if (CL < hp.minSamplesLeaf || CR < hp.minSamplesLeaf) continue
          const gain = 0.5 * (score(GL, HL) + score(G - GL, H - HL) - score(G, H))
          if (gain > 1e-12 && (best === undefined || gain > best.gain)) best = { gain, feature: j, bin: b, missingLeft }
        }
      }
    }
    return best
  }
  const addLeaf = (rs: readonly number[]): number => {
    const { G, H } = sums(rs)
    tree.feature.push(-1)
    tree.threshold.push(Number.NaN)
    tree.missingLeft.push(false)
    tree.left.push(-1)
    tree.right.push(-1)
    tree.value.push((-hp.learningRate * G) / (H + lambda))
    return tree.value.length - 1
  }
  interface Open {
    readonly id: number
    readonly rows: readonly number[]
    readonly depth: number
    readonly split?: Split
  }
  const open = (id: number, rs: readonly number[], depth: number): Open => {
    const s =
      rs.length >= 2 * hp.minSamplesLeaf && depth < (hp.maxDepth ?? Number.POSITIVE_INFINITY)
        ? bestSplit(rs)
        : undefined
    return { id, rows: rs, depth, ...(s === undefined ? {} : { split: s }) }
  }
  const frontier: Open[] = [open(addLeaf(rows), rows, 0)]
  let leaves = 1
  while (leaves < hp.maxLeafNodes) {
    let pick = -1
    for (const [i, o] of frontier.entries())
      if (o.split !== undefined && (pick < 0 || o.split.gain > (frontier[pick]?.split?.gain ?? 0))) pick = i
    if (pick < 0) break
    const node = frontier.splice(pick, 1)[0] as Open
    const s = node.split as Split
    const goesLeft = (r: number): boolean => {
      const b = bins[r * d + s.feature] ?? -1
      return b < 0 ? s.missingLeft : b <= s.bin
    }
    const lr = node.rows.filter(goesLeft)
    const rr = node.rows.filter((r) => !goesLeft(r))
    const l = addLeaf(lr)
    const r = addLeaf(rr)
    tree.feature[node.id] = s.feature
    tree.threshold[node.id] = edges[s.feature]?.[s.bin] ?? 0
    tree.missingLeft[node.id] = s.missingLeft
    tree.left[node.id] = l
    tree.right[node.id] = r
    leaves++
    frontier.push(open(l, lr, node.depth + 1), open(r, rr, node.depth + 1))
  }
  return tree
}

interface HistState {
  readonly f: Float64Array
  readonly iterations: HistTree[][]
  readonly train: number[]
  readonly validation: number[]
  readonly best: number
  readonly stall: number
}

function histAlgorithm(
  x: MatrixView,
  y: Float64Array,
  w: Float64Array,
  loss: HistLoss,
  hp: HistBoostingHyper,
  trainRows: readonly number[],
  validRows: readonly number[],
): { readonly algorithm: Algorithm<HistState>; readonly baseline: number[] } {
  const n = x.n
  const k = loss.k
  const edges = binEdges(takeRows(x, trainRows), Math.min(hp.maxBins, 255))
  const bins = new Int32Array(n * x.d)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < x.d; j++) bins[i * x.d + j] = binOf(edges[j] ?? [], x.data[i * x.d + j] ?? Number.NaN)
  const baseline = loss.baseline(
    Float64Array.from(trainRows, (r) => y[r] ?? 0),
    Float64Array.from(trainRows, (r) => w[r] ?? 0),
  )
  const algorithm: Algorithm<HistState> = {
    name: 'histGradientBoosting',
    config: { maxIter: hp.maxIter, learningRate: hp.learningRate, maxLeafNodes: hp.maxLeafNodes },
    schema: channels({
      metrics: {
        trainLoss: { description: 'training loss', direction: 'lower' },
        validationLoss: { description: 'validation loss (early stopping)', direction: 'lower' },
      },
    }),
    init: () => {
      const f = new Float64Array(n * k)
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) f[i * k + c] = baseline[c] ?? 0
      return { f, iterations: [], train: [], validation: [], best: Number.POSITIVE_INFINITY, stall: 0 }
    },
    step: (s) => {
      const f = Float64Array.from(s.f)
      const trees: HistTree[] = []
      for (let c = 0; c < k; c++) {
        const g = new Float64Array(n)
        const h = new Float64Array(n)
        loss.grad(k === 1 ? Float64Array.from({ length: n }, (_, i) => s.f[i] ?? 0) : s.f, y, c, { g, h })
        const tree = growHistTree(bins, x.d, edges, trainRows, g, h, w, hp)
        trees.push(tree)
        for (let i = 0; i < n; i++)
          f[i * k + c] = (f[i * k + c] ?? 0) + predictTree(tree, x.data.subarray(i * x.d, (i + 1) * x.d))
      }
      const train = loss.loss(f, y, w, trainRows)
      const valid = validRows.length > 0 ? loss.loss(f, y, w, validRows) : Number.NaN
      const improved = validRows.length > 0 && valid < s.best - hp.tol
      return {
        f,
        iterations: [...s.iterations, trees],
        train: [...s.train, train],
        validation: validRows.length > 0 ? [...s.validation, valid] : s.validation,
        best: improved ? valid : s.best,
        stall: improved ? 0 : s.stall + 1,
      }
    },
    done: (s) => (validRows.length > 0 && s.stall >= hp.nIterNoChange ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        trainLoss: s.train[s.train.length - 1] ?? Number.NaN,
        ...(s.validation.length > 0 ? { validationLoss: s.validation[s.validation.length - 1] ?? Number.NaN } : {}),
      },
    }),
  }
  return { algorithm, baseline }
}

function histVerbs(p: HistBoostingParams): HistBoostingVerbs {
  const classifier = p.classes.length > 0
  const k = p.baseline.length
  const proba = (x: TensorLike): Tensor => {
    const { m, f } = rawOf(p, x)
    const kc = p.classes.length
    const out = new Float64Array(m * kc)
    for (let i = 0; i < m; i++) {
      if (k === 1) {
        const q = 1 / (1 + Math.exp(-(f[i] ?? 0)))
        out[i * 2] = 1 - q
        out[i * 2 + 1] = q
      } else for (const [c, v] of softmaxRow(f, i, k).entries()) out[i * kc + c] = v
    }
    return fromStorage(out, [m, kc], 'float64')
  }
  return {
    predict: (x) => {
      if (!classifier) {
        const { m, f } = rawOf(p, x)
        return fromStorage(f, [m], 'float64')
      }
      const pr = proba(x)
      const kc = p.classes.length
      const m = pr.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => {
          let best = 0
          for (let c = 1; c < kc; c++) if ((pr.data[i * kc + c] ?? 0) > (pr.data[i * kc + best] ?? 0)) best = c
          return p.classes[best] ?? 0
        }),
        [m],
        'float64',
      )
    },
    ...(classifier
      ? {
          predictProba: proba,
          predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }),
          classes: p.classes,
        }
      : {}),
    decisionFunction: (x) => {
      const { m, f } = rawOf(p, x)
      return k === 1 ? fromStorage(f, [m], 'float64') : fromStorage(f, [m, k], 'float64')
    },
    nFeaturesIn: p.nFeatures,
  }
}

function histEstimator(
  name: string,
  classifier: boolean,
  hyper: HistBoostingHyper,
): ExplainedEstimator<HistBoostingHyper, FittedHistBoosting> {
  const est: ExplainedEstimator<HistBoostingHyper, FittedHistBoosting> = defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: HistBoostingHyper): Space =>
        dict({
          maxIter: int(1, 500, { default: h.maxIter, label: 'iterations' }),
          learningRate: real(0.01, 1, {
            label: 'learning rate',
            description: 'The shrinkage of each tree’s contribution.',
            log: true,
            default: h.learningRate,
          }),
          maxLeafNodes: int(2, 64, { default: h.maxLeafNodes, label: 'leaves per tree' }),
          l2Regularization: real(0, 10, {
            label: 'L2 regularisation',
            description: 'The L2 penalty on the leaf values.',
            default: h.l2Regularization,
          }),
          earlyStopping: bool({
            label: 'early stopping',
            description: 'Stop when the validation score stops improving.',
            default: h.earlyStopping,
          }),
        }),
      fit: (data: LearnData, h: HistBoostingHyper, options: FitOptions) => {
        const x = matrixView(data.x, name, { allowNaN: true })
        const w = sampleWeightsOf(data.weights, x.n, name)
        const classes = classifier ? [...encodeLabels(data.y ?? [], name, x.n).classes] : []
        const y = classifier
          ? Float64Array.from(encodeLabels(data.y ?? [], name, x.n).codes)
          : targetVector(data.y ?? [], name, x.n)
        if (classifier && classes.length < 2) throw new WhiteboxError('value', `${name}: needs two classes`)
        const key = resolveKey(options, name)
        const order = Array.from(toFlat(permutation(key, x.n)))
        const nValid = h.earlyStopping ? Math.max(1, Math.floor(h.validationFraction * x.n)) : 0
        const validRows = order.slice(0, nValid).sort((a, b) => a - b)
        const trainRows = order.slice(nValid).sort((a, b) => a - b)
        const loss = lossOf(classes.length)
        const { algorithm, baseline } = histAlgorithm(x, y, w, loss, h, trainRows, validRows)
        const r = runAlgorithm(algorithm, {
          maxSteps: h.maxIter,
          key,
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        })
        const params: HistBoostingParams = { baseline, iterations: r.state.iterations, classes, nFeatures: x.d }
        const features = featureNamesOf(data, x.d)
        const exposition: Exposition[] = [
          ensembleStages({
            method: name,
            weights: r.state.iterations.map(() => h.learningRate),
            trainScores: r.state.train,
            scoreName: 'loss',
            higherIsBetter: false,
            ...(r.state.validation.length > 0 ? { validationScores: r.state.validation } : {}),
          }),
        ]
        if (x.d === 2) {
          // The model after each recorded iteration.
          const fitted = est.revive(params)
          const its = r.state.iterations
          const M = its.length
          const steps =
            options.explain === false
              ? undefined
              : stagedSteps(M, (m) => (m === M ? fitted : est.revive({ ...params, iterations: its.slice(0, m) })), {
                  objective: r.state.train,
                  misclassified: classifier,
                })
          const surface = decisionSurfaceOf(fitted, data.x, {
            featureNames: features,
            ...(classifier ? { classNames: classNamesOf(data, classes) } : {}),
            ...(data.y === undefined ? {} : { y: data.y }),
            ...(steps === undefined
              ? {}
              : {
                  steps,
                  stepName: 'iteration',
                  objective: {
                    name: 'training loss',
                    iterations: r.state.train.map((_, m) => m + 1),
                    values: r.state.train,
                  },
                }),
          })
          exposition[0] = { ...(exposition[0] as EnsembleStages), surface }
          exposition.push(surface)
        }
        return { params, exposition }
      },
      methods: (p: HistBoostingParams) => histVerbs(p),
    },
    hyper,
  )
  return est
}

const histDefaults = (hyper: Partial<HistBoostingHyper>): HistBoostingHyper => ({
  maxIter: hyper.maxIter ?? 100,
  learningRate: hyper.learningRate ?? 0.1,
  maxLeafNodes: hyper.maxLeafNodes ?? 31,
  ...(hyper.maxDepth === undefined ? {} : { maxDepth: hyper.maxDepth }),
  minSamplesLeaf: hyper.minSamplesLeaf ?? 20,
  l2Regularization: hyper.l2Regularization ?? 0,
  maxBins: hyper.maxBins ?? 255,
  earlyStopping: hyper.earlyStopping ?? false,
  validationFraction: hyper.validationFraction ?? 0.1,
  nIterNoChange: hyper.nIterNoChange ?? 10,
  tol: hyper.tol ?? 1e-7,
})

/**
 * Histogram gradient boosting for regression (sklearn.ensemble.HistGradientBoostingRegressor, basics): features
 * binned once into at most 255 quantile bins (missing values in their own bin, sent to the better side at each
 * split), Newton trees grown leaf-wise on gradient and hessian histograms with L2 regularisation, and optional early
 * stopping on a keyed validation split, one traced iteration per step. Exposition: the stages (training and
 * validation loss per iteration) and, for two features, the prediction surface.
 *
 * @param hyper - Iterations (default 100), learning rate (0.1), leaves (31), depth, leaf size (20), L2, bins and
 * early stopping.
 * @returns The estimator.
 */
export function histGradientBoostingRegressor(
  hyper: Partial<HistBoostingHyper> = {},
): ExplainedEstimator<HistBoostingHyper, FittedHistBoosting> {
  return histEstimator('hist-gradient-boosting-regressor', false, histDefaults(hyper))
}

/**
 * Histogram gradient boosting for classification with the log loss (one tree per iteration for two classes, one per
 * class otherwise); exposition as {@link histGradientBoostingRegressor} with class probabilities on the surface.
 *
 * @param hyper - As {@link histGradientBoostingRegressor}.
 * @returns The estimator.
 */
export function histGradientBoostingClassifier(
  hyper: Partial<HistBoostingHyper> = {},
): ExplainedEstimator<HistBoostingHyper, FittedHistBoosting> {
  return histEstimator('hist-gradient-boosting-classifier', true, histDefaults(hyper))
}

/** {@link histGradientBoostingRegressor} as a class. */
export class HistGradientBoostingRegressor extends ExplainedEstimatorClass<HistBoostingHyper, FittedHistBoosting> {
  /** @param hyper - As {@link histGradientBoostingRegressor}. */
  constructor(hyper: Partial<HistBoostingHyper> = {}) {
    super(histGradientBoostingRegressor(hyper))
  }
}

/** {@link histGradientBoostingClassifier} as a class. */
export class HistGradientBoostingClassifier extends ExplainedEstimatorClass<HistBoostingHyper, FittedHistBoosting> {
  /** @param hyper - As {@link histGradientBoostingClassifier}. */
  constructor(hyper: Partial<HistBoostingHyper> = {}) {
    super(histGradientBoostingClassifier(hyper))
  }
}
