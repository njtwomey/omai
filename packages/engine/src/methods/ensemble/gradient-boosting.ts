import { quantileOf, WhiteboxError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type EnsembleStages, type Exposition, ensembleStages, treeRecord } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  stagedSteps,
  takeRows,
} from '../../lib/learn/index.ts'
import { foldIn, permutation } from '../../lib/random/index.ts'
import { choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import {
  applyTree,
  featureImportancesOf,
  type GrowOptions,
  growTree,
  type TreeData,
  type TreeParams,
  treeData,
} from '../trees/index.ts'

/** Losses of gradient boosting for regression. */
export type RegressionLoss = 'squared_error' | 'absolute_error' | 'huber' | 'quantile'

/** Hyperparameters of the gradient-boosting estimators. */
export interface GradientBoostingHyper {
  readonly nEstimators: number
  readonly learningRate: number
  readonly maxDepth: number
  readonly minSamplesLeaf: number
  readonly minSamplesSplit: number
  /** Fraction of rows each stage's tree sees (stochastic gradient boosting when below 1). */
  readonly subsample: number
  readonly maxFeatures: GrowOptions['maxFeatures']
  /** The regression loss (ignored by the classifier, which uses the log loss). */
  readonly loss: RegressionLoss
  /** The quantile of the quantile and Huber losses (default 0.9). */
  readonly alpha: number
}

/** Parameters of a fitted gradient-boosting model: the initial raw predictions and the trees of each stage. */
export type GradientBoostingParams = {
  /** One raw value per output (one for regression and binary problems, K otherwise). */
  readonly init: number[]
  /** stages[m][k]: the tree for output k at stage m (leaf values already updated). */
  readonly stages: TreeParams[][]
  readonly learningRate: number
  readonly classes: number[]
  readonly nFeatures: number
}

/** Verbs of a fitted gradient-boosting model. */
export interface GradientBoostingVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  /** The raw predictions (log odds or scores for classifiers). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** The raw predictions after each stage, `[stages][m * K]`. */
  readonly stagedDecisionFunction: (x: TensorLike) => Float64Array[]
  readonly classes?: number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted gradient-boosting model. */
export type FittedGradientBoosting = Fitted<GradientBoostingParams> & GradientBoostingVerbs

const expit = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))

/** The weighted percentile: the first value whose cumulative weight reaches q times the total (scikit-learn's). */
function weightedQuantile(values: readonly number[], weights: readonly number[], q: number): number {
  const order = values.map((_, i) => i).sort((a, b) => (values[a] ?? 0) - (values[b] ?? 0))
  const total = weights.reduce((a, b) => a + b, 0)
  let acc = 0
  for (const i of order) {
    acc += weights[i] ?? 0
    if (acc >= q * total) return values[i] ?? 0
  }
  return values[order[order.length - 1] ?? 0] ?? 0
}

function raw(
  p: GradientBoostingParams,
  x: TensorLike,
  stages = p.stages.length,
): { readonly m: number; readonly f: Float64Array } {
  const m = matrixView(x, 'gradientBoosting', { nFeatures: p.nFeatures }).n
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

function probabilities(f: Float64Array, m: number, k: number, classes: number): Float64Array {
  const out = new Float64Array(m * classes)
  for (let i = 0; i < m; i++) {
    if (k === 1) {
      const q = expit(f[i] ?? 0)
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

function boostingVerbs(p: GradientBoostingParams): GradientBoostingVerbs {
  const classifier = p.classes.length > 0
  const k = p.init.length
  const proba = (x: TensorLike): Tensor => {
    const { m, f } = raw(p, x)
    return fromStorage(probabilities(f, m, k, p.classes.length), [m, p.classes.length], 'float64')
  }
  const importances = new Array<number>(p.nFeatures).fill(0)
  const trees = p.stages.flat()
  for (const t of trees)
    for (const [j, v] of featureImportancesOf(t).entries()) importances[j] = (importances[j] ?? 0) + v / trees.length
  return {
    predict: (x) => {
      const { m, f } = raw(p, x)
      if (!classifier) return fromStorage(f, [m], 'float64')
      const pr = probabilities(f, m, k, p.classes.length)
      const kc = p.classes.length
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => {
          let best = 0
          for (let c = 1; c < kc; c++) if ((pr[i * kc + c] ?? 0) > (pr[i * kc + best] ?? 0)) best = c
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
      const { m, f } = raw(p, x)
      return k === 1 ? fromStorage(f, [m], 'float64') : fromStorage(f, [m, k], 'float64')
    },
    stagedDecisionFunction: (x) => p.stages.map((_, s) => raw(p, x, s + 1).f),
    nFeaturesIn: p.nFeatures,
    featureImportances: importances,
  }
}

/** The state of the boosting algorithm. */
interface BoostState {
  readonly f: Float64Array
  readonly stages: TreeParams[][]
  readonly losses: number[]
  readonly oob: number[]
  readonly residuals: number[][]
  readonly firstTree?: ReturnType<typeof growTree>
}

interface LossSpec {
  /** Raw outputs per row. */
  readonly k: number
  init(data: TreeData): number[]
  /** Negative gradient for output c. */
  gradient(f: Float64Array, data: TreeData, c: number): Float64Array
  /** The leaf value from the rows of a leaf. */
  leaf(rows: readonly number[], residual: Float64Array, f: Float64Array, data: TreeData, c: number): number
  /** The loss, averaged with weights, over some rows. */
  loss(f: Float64Array, data: TreeData, rows: readonly number[]): number
}

function regressionLoss(kind: RegressionLoss, alpha: number): LossSpec {
  let gamma = 0
  const diffs = (rows: readonly number[], f: Float64Array, data: TreeData): number[] =>
    rows.map((r) => (data.y[r] ?? 0) - (f[r] ?? 0))
  const ws = (rows: readonly number[], data: TreeData): number[] => rows.map((r) => data.w[r] ?? 0)
  return {
    k: 1,
    init: (data) => {
      const all = Array.from(data.y)
      const w = Array.from(data.w)
      if (kind === 'squared_error')
        return [all.reduce((a, v, i) => a + v * (w[i] ?? 0), 0) / w.reduce((a, b) => a + b, 0)]
      // The initial model: numpy's interpolated percentile without weights, as scikit-learn's.
      const level = kind === 'quantile' ? alpha : 0.5
      return [w.every((v) => v === w[0]) ? quantileOf(all, level) : weightedQuantile(all, w, level)]
    },
    gradient: (f, data) => {
      const n = data.y.length
      const d = Float64Array.from({ length: n }, (_, i) => (data.y[i] ?? 0) - (f[i] ?? 0))
      if (kind === 'squared_error') return d
      if (kind === 'absolute_error') return d.map(Math.sign)
      if (kind === 'quantile') return d.map((v) => (v > 0 ? alpha : -(1 - alpha)))
      gamma = weightedQuantile(Array.from(d, Math.abs), Array.from(data.w), alpha)
      return d.map((v) => (Math.abs(v) <= gamma ? v : gamma * Math.sign(v)))
    },
    leaf: (rows, residual, f, data) => {
      if (kind === 'squared_error') {
        const w = ws(rows, data)
        return rows.reduce((a, r, i) => a + (residual[r] ?? 0) * (w[i] ?? 0), 0) / w.reduce((a, b) => a + b, 0)
      }
      const d = diffs(rows, f, data)
      const w = ws(rows, data)
      if (kind === 'absolute_error') return weightedQuantile(d, w, 0.5)
      if (kind === 'quantile') return weightedQuantile(d, w, alpha)
      const median = weightedQuantile(d, w, 0.5)
      const shifted = d.map((v) => v - median)
      return median + shifted.reduce((a, v) => a + Math.sign(v) * Math.min(Math.abs(v), gamma), 0) / shifted.length
    },
    loss: (f, data, rows) => {
      let s = 0
      let sw = 0
      for (const r of rows) {
        const d = (data.y[r] ?? 0) - (f[r] ?? 0)
        const w = data.w[r] ?? 0
        sw += w
        if (kind === 'squared_error') s += w * d * d
        else if (kind === 'absolute_error') s += w * Math.abs(d)
        else if (kind === 'quantile') s += w * (d > 0 ? alpha * d : (alpha - 1) * d)
        else s += w * (Math.abs(d) <= gamma ? 0.5 * d * d : gamma * (Math.abs(d) - gamma / 2))
      }
      return s / sw
    },
  }
}

function logLoss(classes: number): LossSpec {
  const k = classes === 2 ? 1 : classes
  return {
    k,
    init: (data) => {
      const p = new Array<number>(classes).fill(0)
      let total = 0
      data.y.forEach((c, i) => {
        p[c] = (p[c] ?? 0) + (data.w[i] ?? 0)
        total += data.w[i] ?? 0
      })
      const probs = p.map((v) => Math.min(Math.max(v / total, 2.220446049250313e-16), 1 - 2.220446049250313e-16))
      return k === 1 ? [Math.log((probs[1] ?? 0.5) / (probs[0] ?? 0.5))] : probs.map(Math.log)
    },
    gradient: (f, data, c) => {
      const n = data.y.length
      const pr = probabilities(f, n, k, classes)
      return Float64Array.from({ length: n }, (_, i) => {
        const target = k === 1 ? (data.y[i] === 1 ? 1 : 0) : data.y[i] === c ? 1 : 0
        return target - (pr[i * classes + (k === 1 ? 1 : c)] ?? 0)
      })
    },
    leaf: (rows, residual, _f, data) => {
      let num = 0
      let den = 0
      for (const r of rows) {
        const res = residual[r] ?? 0
        const w = data.w[r] ?? 0
        num += w * res
        den += k === 1 ? w * Math.abs(res) * (1 - Math.abs(res)) : w * Math.abs(res) * (1 - Math.abs(res))
      }
      if (k > 1) num *= (k - 1) / k
      return Math.abs(den) < 1e-150 ? 0 : num / den
    },
    loss: (f, data, rows) => {
      const n = data.y.length
      const pr = probabilities(f, n, k, classes)
      let s = 0
      let sw = 0
      for (const r of rows) {
        const w = data.w[r] ?? 0
        sw += w
        s -= w * Math.log(Math.max(pr[r * classes + (data.y[r] ?? 0)] ?? 0, 1e-300))
      }
      return s / sw
    },
  }
}

function boostingAlgorithm(data: TreeData, loss: LossSpec, h: GradientBoostingHyper): Algorithm<BoostState> {
  const n = data.x.n
  const k = loss.k
  const init = loss.init(data)
  const grow: GrowOptions = {
    criterion: 'friedman_mse',
    splitter: 'best',
    maxDepth: h.maxDepth,
    minSamplesSplit: h.minSamplesSplit,
    minSamplesLeaf: h.minSamplesLeaf,
    minWeightFractionLeaf: 0,
    maxFeatures: h.maxFeatures,
    minImpurityDecrease: 0,
    record: 3,
  }
  const nInBag = Math.max(1, Math.floor(h.subsample * n))
  return {
    name: 'gradientBoosting',
    config: { nEstimators: h.nEstimators, learningRate: h.learningRate, outputs: k },
    schema: channels({
      metrics: {
        trainLoss: { description: 'training loss after the stage (in-bag rows)', direction: 'lower' },
        oobLoss: { description: 'out-of-bag loss (subsampling only)', direction: 'lower' },
      },
    }),
    init: () => {
      const f = new Float64Array(n * k)
      for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) f[i * k + c] = init[c] ?? 0
      return { f, stages: [], losses: [], oob: [], residuals: [] }
    },
    step: (s, ctx) => {
      const inBag = new Uint8Array(n)
      if (h.subsample < 1) for (const r of toFlat(permutation(foldIn(ctx.key, 0), n)).slice(0, nInBag)) inBag[r] = 1
      else inBag.fill(1)
      const bagRows = Array.from({ length: n }, (_, i) => i).filter((i) => inBag[i])
      const f = Float64Array.from(s.f)
      const stage: TreeParams[] = []
      let first: ReturnType<typeof growTree> | undefined
      let residualsShown: number[] = []
      for (let c = 0; c < k; c++) {
        const column = Float64Array.from({ length: n }, (_, i) => s.f[i * k + c] ?? 0)
        const fk = k === 1 ? column : s.f
        const residual = loss.gradient(fk, data, c)
        if (c === 0) residualsShown = Array.from(residual)
        // The tree sees only the in-bag rows (zero-weight rows are not samples, as in scikit-learn).
        const sub: TreeData =
          h.subsample < 1
            ? {
                x: takeRows(data.x, bagRows),
                y: Float64Array.from(bagRows, (r) => residual[r] ?? 0),
                w: Float64Array.from(bagRows, (r) => data.w[r] ?? 0),
                classes: [],
              }
            : { x: data.x, y: residual, w: data.w, classes: [] }
        const grown = growTree(sub, grow, { key: foldIn(ctx.key, c + 1) })
        if (c === 0) first = grown
        const leaves = applyTree(grown.tree, fromStorage(data.x.data, [n, data.x.d], 'float64'))
        const byLeaf = new Map<number, number[]>()
        for (const r of bagRows) {
          const l = leaves[r] ?? 0
          const list = byLeaf.get(l) ?? []
          list.push(r)
          byLeaf.set(l, list)
        }
        const value = grown.tree.value.map((v, node) => {
          const rows = byLeaf.get(node)
          return rows === undefined ? v : [loss.leaf(rows, residual, column, data, c)]
        })
        const tree: TreeParams = { ...grown.tree, value }
        stage.push(tree)
        for (let i = 0; i < n; i++)
          f[i * k + c] = (f[i * k + c] ?? 0) + h.learningRate * (value[leaves[i] ?? 0]?.[0] ?? 0)
      }
      const oobRows = Array.from({ length: n }, (_, i) => i).filter((i) => !inBag[i])
      return {
        f,
        stages: [...s.stages, stage],
        losses: [...s.losses, loss.loss(f, data, bagRows)],
        oob: h.subsample < 1 && oobRows.length > 0 ? [...s.oob, loss.loss(f, data, oobRows)] : s.oob,
        residuals: [...s.residuals, residualsShown],
        ...((s.firstTree ?? first) ? { firstTree: s.firstTree ?? (first as ReturnType<typeof growTree>) } : {}),
      }
    },
    observe: (s) => ({
      metrics: {
        trainLoss: s.losses[s.losses.length - 1] ?? Number.NaN,
        ...(s.oob.length > 0 ? { oobLoss: s.oob[s.oob.length - 1] ?? Number.NaN } : {}),
      },
    }),
  }
}

function boostingEstimator(
  name: string,
  classifier: boolean,
  hyper: GradientBoostingHyper,
): ExplainedEstimator<GradientBoostingHyper, FittedGradientBoosting> {
  const est: ExplainedEstimator<GradientBoostingHyper, FittedGradientBoosting> = defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: GradientBoostingHyper): Space =>
        dict({
          nEstimators: int(1, 500, { default: h.nEstimators, label: 'stages' }),
          learningRate: real(0.01, 1, { log: true, default: h.learningRate, label: 'learning rate' }),
          maxDepth: int(1, 8, { default: h.maxDepth, label: 'tree depth' }),
          subsample: real(0.1, 1, { default: h.subsample }),
          ...(classifier
            ? {}
            : { loss: choice(['squared_error', 'absolute_error', 'huber', 'quantile'], { default: h.loss }) }),
        }),
      fit: (data: LearnData, h: GradientBoostingHyper, options: FitOptions) => {
        const prepared = treeData(data, classifier, name)
        if (classifier && prepared.classes.length < 2) throw new WhiteboxError('value', `${name}: needs two classes`)
        const loss = classifier ? logLoss(prepared.classes.length) : regressionLoss(h.loss, h.alpha)
        const r = runAlgorithm(boostingAlgorithm(prepared, loss, h), {
          maxSteps: h.nEstimators,
          key: resolveKey(options, name),
          ...(options.parent !== undefined
            ? { parent: options.parent }
            : {
                ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
                ...(options.policy === undefined ? {} : { policy: options.policy }),
              }),
        })
        const s = r.state
        const params: GradientBoostingParams = {
          init: loss.init(prepared),
          stages: s.stages,
          learningRate: h.learningRate,
          classes: [...prepared.classes],
          nFeatures: prepared.x.d,
        }
        const features = featureNamesOf(data, prepared.x.d)
        const classNames = classifier ? classNamesOf(data, prepared.classes) : undefined
        const exposition: Exposition[] = [
          ensembleStages({
            method: name,
            weights: s.stages.map(() => h.learningRate),
            trainScores: s.losses,
            scoreName: 'training loss',
            higherIsBetter: false,
            ...(s.oob.length === s.losses.length && s.oob.length > 0 ? { validationScores: s.oob } : {}),
            targets: s.residuals,
          }),
        ]
        if (s.firstTree !== undefined)
          exposition.push(treeRecord({ criterion: 'friedman_mse', features, nodes: s.firstTree.nodes }))
        if (prepared.x.d === 2) {
          // The model after each recorded stage, with the residuals its newest tree was fitted to.
          const fitted = est.revive(params)
          const M = s.stages.length
          const steps =
            options.explain === false
              ? undefined
              : stagedSteps(M, (m) => (m === M ? fitted : est.revive({ ...params, stages: s.stages.slice(0, m) })), {
                  objective: s.losses,
                  rowWeights: s.residuals,
                  misclassified: classifier,
                })
          const surface = decisionSurfaceOf(fitted, data.x, {
            featureNames: features,
            ...(classNames === undefined ? {} : { classNames }),
            ...(data.y === undefined ? {} : { y: data.y }),
            ...(steps === undefined
              ? {}
              : {
                  steps,
                  stepName: 'stage',
                  objective: { name: 'training loss', iterations: s.losses.map((_, m) => m + 1), values: s.losses },
                }),
          })
          exposition[0] = { ...(exposition[0] as EnsembleStages), surface }
          exposition.push(surface)
        }
        return { params, exposition }
      },
      methods: (p: GradientBoostingParams) => boostingVerbs(p),
    },
    hyper,
  )
  return est
}

const boostingDefaults = (hyper: Partial<GradientBoostingHyper>): GradientBoostingHyper => ({
  nEstimators: hyper.nEstimators ?? 100,
  learningRate: hyper.learningRate ?? 0.1,
  maxDepth: hyper.maxDepth ?? 3,
  minSamplesLeaf: hyper.minSamplesLeaf ?? 1,
  minSamplesSplit: hyper.minSamplesSplit ?? 2,
  subsample: hyper.subsample ?? 1,
  maxFeatures: hyper.maxFeatures ?? 'all',
  loss: hyper.loss ?? 'squared_error',
  alpha: hyper.alpha ?? 0.9,
})

/**
 * Gradient boosting for regression (sklearn.ensemble.GradientBoostingRegressor), as a traced algorithm with one stage
 * per step: each stage fits a Friedman-MSE tree to the negative gradient of the loss (squared, absolute, Huber or
 * quantile), re-estimates its leaf values for the loss, and adds them scaled by the learning rate; subsampling gives
 * stochastic gradient boosting with out-of-bag losses. Exposition: the stages (training loss per stage, out-of-bag
 * loss when subsampling, the residuals each tree was fitted to), the first tree's record and, for two features, the
 * prediction surface.
 *
 * @param hyper - Stages (default 100), learning rate (0.1), tree depth (3), leaf limits, subsample, loss and alpha.
 * @returns The estimator.
 */
export function gradientBoostingRegressor(
  hyper: Partial<GradientBoostingHyper> = {},
): ExplainedEstimator<GradientBoostingHyper, FittedGradientBoosting> {
  return boostingEstimator('gradient-boosting-regressor', false, boostingDefaults(hyper))
}

/**
 * Gradient boosting for classification with the log loss (sklearn.ensemble.GradientBoostingClassifier): one tree per
 * stage for two classes (Newton leaf values on the log odds), one per class otherwise (softmax). Exposition as
 * {@link gradientBoostingRegressor}, with the log loss and the decision surface with class probabilities.
 *
 * @param hyper - As {@link gradientBoostingRegressor} (the loss is fixed).
 * @returns The estimator.
 */
export function gradientBoostingClassifier(
  hyper: Partial<GradientBoostingHyper> = {},
): ExplainedEstimator<GradientBoostingHyper, FittedGradientBoosting> {
  return boostingEstimator('gradient-boosting-classifier', true, boostingDefaults(hyper))
}

/** {@link gradientBoostingRegressor} as a class. */
export class GradientBoostingRegressor extends ExplainedEstimatorClass<GradientBoostingHyper, FittedGradientBoosting> {
  /** @param hyper - As {@link gradientBoostingRegressor}. */
  constructor(hyper: Partial<GradientBoostingHyper> = {}) {
    super(gradientBoostingRegressor(hyper))
  }
}

/** {@link gradientBoostingClassifier} as a class. */
export class GradientBoostingClassifier extends ExplainedEstimatorClass<GradientBoostingHyper, FittedGradientBoosting> {
  /** @param hyper - As {@link gradientBoostingClassifier}. */
  constructor(hyper: Partial<GradientBoostingHyper> = {}) {
    super(gradientBoostingClassifier(hyper))
  }
}
