import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
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
import { choice as draw, foldIn } from '../../lib/random/index.ts'
import { dict, int, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import {
  featureImportancesOf,
  type GrowOptions,
  growTree,
  type TreeData,
  type TreeParams,
  treeData,
  treePredict,
} from '../trees/index.ts'
import { scoreOf } from './_stages.ts'

/** Hyperparameters of the AdaBoost estimators. */
export interface AdaBoostHyper {
  readonly nEstimators: number
  readonly learningRate: number
  /** Depth of each weak tree (default 1, stumps, for classification; 3 for regression). */
  readonly maxDepth: number
  /** The regression loss of AdaBoost.R2. */
  readonly loss: 'linear' | 'square' | 'exponential'
}

/** Parameters of a fitted AdaBoost model. */
export type AdaBoostParams = {
  readonly trees: TreeParams[]
  readonly weights: number[]
  readonly classes: number[]
  readonly nFeatures: number
}

/** Verbs of a fitted AdaBoost model. */
export interface AdaBoostVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly decisionFunction?: (x: TensorLike) => Tensor
  readonly classes?: number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted AdaBoost model. */
export type FittedAdaBoost = Fitted<AdaBoostParams> & AdaBoostVerbs

/** SAMME's decision: sum of member weights voting for each class, over the total weight, `[m, k]`. */
function decision(
  p: AdaBoostParams,
  x: TensorLike,
  stages = p.trees.length,
): { readonly m: number; readonly d: Float64Array } {
  const m = matrixView(x, 'adaBoost', { nFeatures: p.nFeatures }).n
  const k = p.classes.length
  const d = new Float64Array(m * k)
  let total = 0
  for (const [s, tree] of p.trees.slice(0, stages).entries()) {
    const w = p.weights[s] ?? 0
    total += w
    const pred = toFlat(treePredict(tree, x))
    for (let i = 0; i < m; i++) {
      const c = p.classes.indexOf(pred[i] ?? 0)
      d[i * k + c] = (d[i * k + c] ?? 0) + w
    }
  }
  for (let i = 0; i < d.length; i++) d[i] = (d[i] ?? 0) / (total || 1)
  return { m, d }
}

function proba(p: AdaBoostParams, x: TensorLike): Tensor {
  const { m, d } = decision(p, x)
  const k = p.classes.length
  const out = new Float64Array(m * k)
  for (let i = 0; i < m; i++) {
    // scikit-learn's SAMME probabilities: softmax of the decision / (K - 1) (binary: of [-f, f] / 2).
    const row = Array.from({ length: k }, (_, c) => (d[i * k + c] ?? 0) / (k - 1))
    const z = k === 2 ? [((row[0] ?? 0) - (row[1] ?? 0)) / 2, ((row[1] ?? 0) - (row[0] ?? 0)) / 2] : row
    const max = Math.max(...z)
    const e = z.map((v) => Math.exp(v - max))
    const s = e.reduce((a, b) => a + b, 0)
    for (let c = 0; c < k; c++) out[i * k + c] = (e[c] ?? 0) / s
  }
  return fromStorage(out, [m, k], 'float64')
}

function importances(p: AdaBoostParams): number[] {
  const out = new Array<number>(p.nFeatures).fill(0)
  const total = p.weights.reduce((a, b) => a + b, 0) || 1
  for (const [s, t] of p.trees.entries())
    for (const [j, v] of featureImportancesOf(t).entries()) out[j] = (out[j] ?? 0) + ((p.weights[s] ?? 0) * v) / total
  return out
}

interface BoostState {
  readonly sampleWeight: Float64Array
  readonly trees: TreeParams[]
  readonly weights: number[]
  readonly targets: number[][]
  readonly nodes?: ReturnType<typeof growTree>['nodes']
  readonly stop: boolean
}

function growOptions(maxDepth: number, criterion: 'gini' | 'squared_error'): GrowOptions {
  return {
    criterion,
    splitter: 'best',
    maxDepth,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    minWeightFractionLeaf: 0,
    maxFeatures: 'all',
    minImpurityDecrease: 0,
    record: 3,
  }
}

/** AdaBoost's SAMME rounds as an algorithm: fit a weighted tree, weigh it by its error, reweight the misclassified. */
function sammeAlgorithm(data: TreeData, h: AdaBoostHyper): Algorithm<BoostState> {
  const n = data.x.n
  const k = data.classes.length
  const grow = growOptions(h.maxDepth, 'gini')
  const xs = fromStorage(data.x.data, [n, data.x.d], 'float64')
  return {
    name: 'adaBoostSamme',
    config: { nEstimators: h.nEstimators, learningRate: h.learningRate },
    schema: channels({
      metrics: {
        error: { description: 'weighted error of the round', direction: 'lower' },
        alpha: { description: 'member weight' },
      },
    }),
    init: () => {
      const total = data.w.reduce((a, b) => a + b, 0)
      return {
        sampleWeight: Float64Array.from(data.w, (w) => w / total),
        trees: [],
        weights: [],
        targets: [],
        stop: false,
      }
    },
    step: (s, ctx) => {
      const g = growTree({ ...data, w: s.sampleWeight }, grow, { key: ctx.key })
      const pred = toFlat(treePredict(g.tree, xs))
      let err = 0
      let total = 0
      const wrong = new Uint8Array(n)
      for (let i = 0; i < n; i++) {
        wrong[i] = data.classes.indexOf(pred[i] ?? 0) !== (data.y[i] ?? 0) ? 1 : 0
        err += (s.sampleWeight[i] ?? 0) * (wrong[i] ?? 0)
        total += s.sampleWeight[i] ?? 0
      }
      err /= total
      const base = {
        trees: [...s.trees, g.tree],
        targets: [...s.targets, Array.from(s.sampleWeight)],
        nodes: s.nodes ?? g.nodes,
      }
      if (err <= 0) return { ...s, ...base, weights: [...s.weights, 1], stop: true }
      if (err >= 1 - 1 / k) {
        if (s.trees.length === 0)
          throw new WhiteboxError('value', 'adaBoost: the first weak learner is no better than chance')
        return { ...s, stop: true }
      }
      const alpha = h.learningRate * (Math.log((1 - err) / err) + Math.log(k - 1))
      const next = Float64Array.from(s.sampleWeight, (w, i) => (wrong[i] && w > 0 ? Math.exp(Math.log(w) + alpha) : w))
      const sum = next.reduce((a, b) => a + b, 0)
      return { ...base, sampleWeight: next.map((w) => w / sum), weights: [...s.weights, alpha], stop: false }
    },
    done: (s) => (s.stop ? 'converged' : undefined),
    observe: (s) => ({ metrics: { alpha: s.weights[s.weights.length - 1] ?? 0, error: Number.NaN } }),
  }
}

/** AdaBoost.R2's rounds: fit a tree to a weighted resample, weigh it by its average loss, reweight by the loss. */
function r2Algorithm(data: TreeData, h: AdaBoostHyper): Algorithm<BoostState> {
  const n = data.x.n
  const grow = growOptions(h.maxDepth, 'squared_error')
  const xs = fromStorage(data.x.data, [n, data.x.d], 'float64')
  return {
    name: 'adaBoostR2',
    config: { nEstimators: h.nEstimators, learningRate: h.learningRate, loss: h.loss },
    init: () => {
      const total = data.w.reduce((a, b) => a + b, 0)
      return {
        sampleWeight: Float64Array.from(data.w, (w) => w / total),
        trees: [],
        weights: [],
        targets: [],
        stop: false,
      }
    },
    step: (s, ctx) => {
      const rows = Array.from(
        toFlat(draw(foldIn(ctx.key, 0), n, n, { replace: true, weights: Array.from(s.sampleWeight) })),
      )
      const g = growTree(
        {
          x: takeRows(data.x, rows),
          y: Float64Array.from(rows, (r) => data.y[r] ?? 0),
          w: new Float64Array(n).fill(1),
          classes: [],
        },
        grow,
        { key: foldIn(ctx.key, 1) },
      )
      const pred = toFlat(treePredict(g.tree, xs))
      const abs = pred.map((p, i) => Math.abs(p - (data.y[i] ?? 0)))
      const max = Math.max(...abs)
      const lossVec = abs.map((e) => {
        const r = max > 0 ? e / max : 0
        return h.loss === 'square' ? r * r : h.loss === 'exponential' ? 1 - Math.exp(-r) : r
      })
      const err = lossVec.reduce((a, l, i) => a + l * (s.sampleWeight[i] ?? 0), 0)
      const base = { trees: [...s.trees, g.tree], targets: [...s.targets, Array.from(s.sampleWeight)] }
      if (err <= 0) return { ...s, ...base, weights: [...s.weights, 1], stop: true }
      if (err >= 0.5) {
        if (s.trees.length === 0) throw new WhiteboxError('value', 'adaBoost: the first weak learner is too weak')
        return { ...s, stop: true }
      }
      const beta = err / (1 - err)
      const alpha = h.learningRate * Math.log(1 / beta)
      const next = Float64Array.from(s.sampleWeight, (w, i) => w * beta ** ((1 - (lossVec[i] ?? 0)) * h.learningRate))
      const sum = next.reduce((a, b) => a + b, 0)
      return { ...base, sampleWeight: next.map((w) => w / sum), weights: [...s.weights, alpha], stop: false }
    },
    done: (s) => (s.stop ? 'converged' : undefined),
  }
}

/** The weighted median of member predictions (AdaBoost.R2's combination). */
function weightedMedianPredict(p: AdaBoostParams, x: TensorLike, stages = p.trees.length): Float64Array {
  const preds = p.trees.slice(0, stages).map((t) => toFlat(treePredict(t, x)))
  const m = preds[0]?.length ?? 0
  const w = p.weights.slice(0, stages)
  const total = w.reduce((a, b) => a + b, 0)
  return Float64Array.from({ length: m }, (_, i) => {
    const order = preds.map((_, s) => s).sort((a, b) => (preds[a]?.[i] ?? 0) - (preds[b]?.[i] ?? 0))
    let acc = 0
    for (const s of order) {
      acc += w[s] ?? 0
      if (acc >= 0.5 * total) return preds[s]?.[i] ?? 0
    }
    return preds[order[order.length - 1] ?? 0]?.[i] ?? 0
  })
}

function adaEstimator(
  name: string,
  classifier: boolean,
  hyper: AdaBoostHyper,
): ExplainedEstimator<AdaBoostHyper, FittedAdaBoost> {
  const est: ExplainedEstimator<AdaBoostHyper, FittedAdaBoost> = defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: AdaBoostHyper): Space =>
        dict({
          nEstimators: int(1, 300, { default: h.nEstimators, label: 'rounds' }),
          learningRate: real(0.01, 2, {
            label: 'learning rate',
            description: 'How much each weak learner’s vote counts.',
            log: true,
            default: h.learningRate,
          }),
          maxDepth: int(1, 6, { default: h.maxDepth, label: 'weak tree depth' }),
        }),
      fit: (data: LearnData, h: AdaBoostHyper, options: FitOptions) => {
        const prepared = treeData(data, classifier, name)
        const r = runAlgorithm(classifier ? sammeAlgorithm(prepared, h) : r2Algorithm(prepared, h), {
          maxSteps: h.nEstimators,
          key: resolveKey(options, name),
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        })
        const s = r.state
        const params: AdaBoostParams = {
          trees: s.trees,
          weights: s.weights,
          classes: [...prepared.classes],
          nFeatures: prepared.x.d,
        }
        const k = prepared.classes.length
        const trainScores = s.trees.map((_, m) =>
          classifier
            ? scoreOf(decision(params, data.x, m + 1).d, 1, prepared.y, k)
            : scoreOf(weightedMedianPredict(params, data.x, m + 1), 1, prepared.y, 0),
        )
        const features = featureNamesOf(data, prepared.x.d)
        const classNames = classifier ? classNamesOf(data, prepared.classes) : undefined
        const exposition: Exposition[] = [
          ensembleStages({
            method: name,
            weights: s.weights,
            trainScores,
            targets: s.targets,
            scoreName: classifier ? 'training accuracy' : 'training R²',
            higherIsBetter: true,
          }),
        ]
        if (s.nodes !== undefined)
          exposition.push(
            treeRecord({
              criterion: 'gini',
              features,
              ...(classNames === undefined ? {} : { classes: classNames }),
              nodes: s.nodes,
            }),
          )
        if (prepared.x.d === 2) {
          // The model after each recorded stage, with the sample weights its newest member was fitted to.
          const fitted = est.revive(params)
          const M = s.trees.length
          const steps =
            options.explain === false
              ? undefined
              : stagedSteps(
                  M,
                  (m) =>
                    m === M
                      ? fitted
                      : est.revive({ ...params, trees: s.trees.slice(0, m), weights: s.weights.slice(0, m) }),
                  { objective: trainScores, rowWeights: s.targets, misclassified: classifier },
                )
          const surface = decisionSurfaceOf(fitted, data.x, {
            featureNames: features,
            ...(classNames === undefined ? {} : { classNames }),
            ...(data.y === undefined ? {} : { y: data.y }),
            ...(steps === undefined
              ? {}
              : {
                  steps,
                  stepName: 'stage',
                  objective: {
                    name: classifier ? 'training accuracy' : 'training R²',
                    iterations: trainScores.map((_, m) => m + 1),
                    values: trainScores,
                    lowerIsBetter: false,
                  },
                }),
          })
          exposition[0] = { ...(exposition[0] as EnsembleStages), surface }
          exposition.push(surface)
        }
        return { params, exposition }
      },
      methods: (p: AdaBoostParams): AdaBoostVerbs =>
        classifier
          ? {
              predict: (x) => {
                const { m, d } = decision(p, x)
                const k = p.classes.length
                return fromStorage(
                  Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(d.subarray(i * k, (i + 1) * k))] ?? 0),
                  [m],
                  'float64',
                )
              },
              predictProba: (x) => proba(p, x),
              predictDistribution: (x) => categorical({ probs: proba(p, x) }),
              decisionFunction: (x) => {
                const { m, d } = decision(p, x)
                return fromStorage(d, [m, p.classes.length], 'float64')
              },
              classes: p.classes,
              nFeaturesIn: p.nFeatures,
              featureImportances: importances(p),
            }
          : {
              predict: (x) => {
                const v = weightedMedianPredict(p, x)
                return fromStorage(v, [v.length], 'float64')
              },
              nFeaturesIn: p.nFeatures,
              featureImportances: importances(p),
            },
    },
    hyper,
  )
  return est
}

/**
 * AdaBoost for classification by SAMME (sklearn.ensemble.AdaBoostClassifier), one boosting round per traced step:
 * a weighted tree (a stump by default), its weight log((1 - err) / err) + log(K - 1) times the learning rate, and
 * the misclassified rows reweighted. Exposition: the stages (member weights, training accuracy after each round and
 * the sample weights each member was fitted to), the first member's tree and, for two features, the decision surface.
 *
 * @param hyper - Rounds (default 50), learning rate (1) and weak-tree depth (1).
 * @returns The estimator.
 */
export function adaBoostClassifier(
  hyper: Partial<AdaBoostHyper> = {},
): ExplainedEstimator<AdaBoostHyper, FittedAdaBoost> {
  return adaEstimator('ada-boost-classifier', true, {
    nEstimators: hyper.nEstimators ?? 50,
    learningRate: hyper.learningRate ?? 1,
    maxDepth: hyper.maxDepth ?? 1,
    loss: hyper.loss ?? 'linear',
  })
}

/**
 * AdaBoost.R2 for regression (sklearn.ensemble.AdaBoostRegressor): trees on keyed weighted resamples, weighted by
 * log(1 / beta), combined by the weighted median. Exposition: the stages (weights, training R^2, sample weights).
 *
 * @param hyper - Rounds (default 50), learning rate (1), weak-tree depth (3) and loss (linear, square, exponential).
 * @returns The estimator.
 */
export function adaBoostRegressor(
  hyper: Partial<AdaBoostHyper> = {},
): ExplainedEstimator<AdaBoostHyper, FittedAdaBoost> {
  return adaEstimator('ada-boost-regressor', false, {
    nEstimators: hyper.nEstimators ?? 50,
    learningRate: hyper.learningRate ?? 1,
    maxDepth: hyper.maxDepth ?? 3,
    loss: hyper.loss ?? 'linear',
  })
}

/** {@link adaBoostClassifier} as a class. */
export class AdaBoostClassifier extends ExplainedEstimatorClass<AdaBoostHyper, FittedAdaBoost> {
  /** @param hyper - As {@link adaBoostClassifier}. */
  constructor(hyper: Partial<AdaBoostHyper> = {}) {
    super(adaBoostClassifier(hyper))
  }
}

/** {@link adaBoostRegressor} as a class. */
export class AdaBoostRegressor extends ExplainedEstimatorClass<AdaBoostHyper, FittedAdaBoost> {
  /** @param hyper - As {@link adaBoostRegressor}. */
  constructor(hyper: Partial<AdaBoostHyper> = {}) {
    super(adaBoostRegressor(hyper))
  }
}
