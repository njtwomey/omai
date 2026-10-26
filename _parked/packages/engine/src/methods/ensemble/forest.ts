/**
 * Random forests and extremely randomised trees (as in scikit-learn): many decision trees, each grown on a bootstrap
 * sample with a random subset of features per split (or random thresholds, for extra trees), averaged into one
 * prediction. Classifiers and regressors with their classes; the regressor's predictive distribution comes from the
 * spread of the trees.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution, normal } from '../../lib/dists/index.ts'
import {
  type EnsembleStages,
  type Exposition,
  ensembleStages,
  type TreeNode,
  treeRecord,
} from '../../lib/exposition/index.ts'
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
  stagedSteps,
  takeRows,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, randint } from '../../lib/random/index.ts'
import { bool, choice, dict, int, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import {
  type Criterion,
  featureImportancesOf,
  type GrowOptions,
  growTree,
  type MaxFeatures,
  type TreeData,
  type TreeParams,
  treeData,
  treeOutputs,
} from '../trees/index.ts'
import { stagedAverageScores, stagedOobScores } from './_stages.ts'

// TODO: random trees embedding.

/** Hyperparameters of the forests. */
export interface ForestHyper {
  readonly nEstimators: number
  readonly criterion: Criterion
  readonly maxDepth?: number
  readonly minSamplesSplit: number
  readonly minSamplesLeaf: number
  readonly maxFeatures: MaxFeatures
  readonly maxLeafNodes?: number
  /** Draw a bootstrap sample per tree (random forests) or use every row (extra trees). */
  readonly bootstrap: boolean
  /** Rows per bootstrap, as a count or a fraction (default all). */
  readonly maxSamples?: number
  readonly splitter: 'best' | 'random'
}

/** Parameters of a fitted forest. */
export type ForestParams = {
  readonly trees: TreeParams[]
  readonly classes: number[]
  readonly nFeatures: number
  /** The out-of-bag score (accuracy or R^2) when bootstrapping. */
  readonly oobScore: number | null
}

/** Verbs of a fitted forest. */
export interface ForestVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Categorical averaged probabilities, or normal with the trees' mean and spread. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly classes?: number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted forest. */
export type FittedForest = Fitted<ForestParams> & ForestVerbs

function forestVerbs(p: ForestParams): ForestVerbs {
  const classifier = p.classes.length > 0
  const average = (x: TensorLike): { readonly mean: Float64Array; readonly sd: Float64Array; readonly m: number } => {
    const outs = p.trees.map((t) => Float64Array.from(toFlat(treeOutputs(t, x))))
    const len = outs[0]?.length ?? 0
    const mean = new Float64Array(len)
    for (const o of outs) for (let i = 0; i < len; i++) mean[i] = (mean[i] ?? 0) + (o[i] ?? 0) / outs.length
    const sd = new Float64Array(len)
    for (const o of outs)
      for (let i = 0; i < len; i++) sd[i] = (sd[i] ?? 0) + ((o[i] ?? 0) - (mean[i] ?? 0)) ** 2 / outs.length
    return { mean, sd: sd.map(Math.sqrt), m: classifier ? len / p.classes.length : len }
  }
  const proba = (x: TensorLike): Tensor => {
    const a = average(x)
    return fromStorage(a.mean, [a.m, p.classes.length], 'float64')
  }
  const importances = new Array<number>(p.nFeatures).fill(0)
  for (const t of p.trees)
    for (const [j, v] of featureImportancesOf(t).entries()) importances[j] = (importances[j] ?? 0) + v / p.trees.length
  return {
    predict: (x) => {
      if (!classifier) return fromStorage(average(x).mean, [average(x).m], 'float64')
      const pr = proba(x)
      const k = p.classes.length
      const m = pr.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => {
          let best = 0
          for (let c = 1; c < k; c++) if ((pr.data[i * k + c] ?? 0) > (pr.data[i * k + best] ?? 0)) best = c
          return p.classes[best] ?? 0
        }),
        [m],
        'float64',
      )
    },
    ...(classifier ? { predictProba: proba, classes: p.classes } : {}),
    predictDistribution: (x) => {
      if (classifier) return categorical({ probs: proba(x) })
      const a = average(x)
      return normal(
        fromStorage(a.mean, [a.m], 'float64'),
        fromStorage(
          a.sd.map((v) => Math.max(v, 1e-12)),
          [a.m],
          'float64',
        ),
      )
    },
    nFeaturesIn: p.nFeatures,
    featureImportances: importances,
  }
}

/** A tree grown on bootstrap counts: rows in the bag with their counts as weights, record rows mapped back. */
export function bootstrapTree(
  data: TreeData,
  options: GrowOptions,
  key: Key,
  bootstrap: boolean,
  size: number,
): {
  readonly tree: TreeParams
  readonly nodes: readonly TreeNode[]
  readonly inBag: Uint8Array
  readonly rows: number[]
} {
  const n = data.x.n
  const inBag = new Uint8Array(n)
  if (!bootstrap) {
    inBag.fill(1)
    const g = growTree(data, options, { key: foldIn(key, 1) })
    return { ...g, inBag, rows: Array.from({ length: n }, (_, i) => i) }
  }
  const counts = new Float64Array(n)
  for (const i of toFlat(randint(foldIn(key, 0), [size], 0, n))) counts[i] = (counts[i] ?? 0) + 1
  const rows = Array.from({ length: n }, (_, i) => i).filter((i) => (counts[i] ?? 0) > 0)
  for (const r of rows) inBag[r] = 1
  const sub: TreeData = {
    x: takeRows(data.x, rows),
    y: Float64Array.from(rows, (r) => data.y[r] ?? 0),
    w: Float64Array.from(rows, (r) => (data.w[r] ?? 0) * (counts[r] ?? 0)),
    classes: data.classes,
  }
  const g = growTree(sub, options, { key: foldIn(key, 1) })
  return { tree: g.tree, nodes: g.nodes.map((nd) => ({ ...nd, rows: nd.rows.map((r) => rows[r] ?? r) })), inBag, rows }
}

function forestEstimator(
  name: string,
  classifier: boolean,
  hyper: ForestHyper,
): ExplainedEstimator<ForestHyper, FittedForest> {
  const est: ExplainedEstimator<ForestHyper, FittedForest> = defineExplainedEstimator(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      hyperSpace: (h: ForestHyper): Space =>
        dict({
          nEstimators: int(1, 300, { default: h.nEstimators, label: 'trees' }),
          maxDepth: int(1, 20, { default: h.maxDepth ?? 8, label: 'maximum depth' }),
          maxFeatures: choice(['sqrt', 'log2', 'all'], {
            label: 'features per split',
            description: 'How many features each split considers: √d, log₂ d or all of them.',
            default: typeof h.maxFeatures === 'string' ? h.maxFeatures : 'all',
          }),
          bootstrap: bool({
            label: 'bootstrap',
            description: 'Fit each tree on a bootstrap resample of the rows (otherwise on all of them).',
            default: h.bootstrap,
          }),
        }),
      fit: (data: LearnData, h: ForestHyper, options: FitOptions) => {
        const prepared = treeData(data, classifier, name)
        const n = prepared.x.n
        if (!(h.nEstimators >= 1)) throw new MdmlError('value', `${name}: nEstimators must be at least 1`)
        const size =
          h.maxSamples === undefined ? n : h.maxSamples < 1 ? Math.max(1, Math.round(h.maxSamples * n)) : h.maxSamples
        const key = resolveKey(options, name)
        const grow: GrowOptions = {
          criterion: h.criterion,
          splitter: h.splitter,
          ...(h.maxDepth === undefined ? {} : { maxDepth: h.maxDepth }),
          minSamplesSplit: h.minSamplesSplit,
          minSamplesLeaf: h.minSamplesLeaf,
          minWeightFractionLeaf: 0,
          maxFeatures: h.maxFeatures,
          ...(h.maxLeafNodes === undefined ? {} : { maxLeafNodes: h.maxLeafNodes }),
          minImpurityDecrease: 0,
          record: 3,
        }
        const members = Array.from({ length: h.nEstimators }, (_, m) =>
          bootstrapTree(prepared, grow, foldIn(key, m), h.bootstrap, size),
        )
        const k = prepared.classes.length
        const outputs = members.map((m) => Float64Array.from(toFlat(treeOutputs(m.tree, data.x))))
        const trainScores = stagedAverageScores(outputs, prepared.y, k)
        const oob = h.bootstrap
          ? stagedOobScores(
              outputs,
              members.map((m) => m.inBag),
              prepared.y,
              k,
            )
          : undefined
        const params: ForestParams = {
          trees: members.map((m) => m.tree),
          classes: [...prepared.classes],
          nFeatures: prepared.x.d,
          oobScore: oob === undefined ? null : (oob[oob.length - 1] ?? null),
        }
        const features = featureNamesOf(data, prepared.x.d)
        const classNames = classifier ? classNamesOf(data, prepared.classes) : undefined
        const exposition: Exposition[] = [
          ensembleStages({
            method: name,
            weights: members.map(() => 1),
            trainScores,
            scoreName: classifier ? 'accuracy' : 'R²',
            higherIsBetter: true,
            ...(oob === undefined ? {} : { validationScores: oob }),
            ...(h.bootstrap ? { bootstraps: members.map((m) => m.rows) } : {}),
          }),
          treeRecord({
            criterion: h.criterion,
            features,
            ...(classNames === undefined ? {} : { classes: classNames }),
            nodes: members[0]?.nodes ?? [],
          }),
        ]
        if (prepared.x.d === 2) {
          // The forest as its trees are added: the average of the first m.
          const fitted = est.revive(params)
          const M = members.length
          const steps =
            options.explain === false
              ? undefined
              : stagedSteps(M, (m) => (m === M ? fitted : est.revive({ ...params, trees: params.trees.slice(0, m) })), {
                  objective: trainScores,
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
                  stepName: 'trees',
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
      methods: (p: ForestParams) => forestVerbs(p),
    },
    hyper,
  )
  return est
}

function forestDefaults(hyper: Partial<ForestHyper>, base: ForestHyper): ForestHyper {
  return { ...base, ...hyper } as ForestHyper
}

const base = (
  criterion: Criterion,
  maxFeatures: MaxFeatures,
  bootstrap: boolean,
  splitter: 'best' | 'random',
): ForestHyper => ({
  nEstimators: 100,
  criterion,
  minSamplesSplit: 2,
  minSamplesLeaf: 1,
  maxFeatures,
  bootstrap,
  splitter,
})

/**
 * A random forest classifier (sklearn.ensemble.RandomForestClassifier): trees on keyed bootstrap samples, sqrt(d)
 * features per split, probabilities averaged. Exposition: the ensemble's stages (training accuracy and out-of-bag
 * accuracy as trees are added, each tree's bootstrap rows), the first tree's record, and for two features the
 * decision surface.
 *
 * @param hyper - Trees (default 100), criterion (default gini), depth and leaf limits, features per split, bootstrap.
 * @returns The estimator.
 */
export function randomForestClassifier(
  hyper: Partial<ForestHyper> = {},
): ExplainedEstimator<ForestHyper, FittedForest> {
  return forestEstimator('random-forest-classifier', true, forestDefaults(hyper, base('gini', 'sqrt', true, 'best')))
}

/**
 * A random forest regressor (sklearn.ensemble.RandomForestRegressor): all features per split by default, predictions
 * averaged; `predictDistribution` is normal with the trees' mean and standard deviation. Exposition as
 * {@link randomForestClassifier} with R^2.
 *
 * @param hyper - As {@link randomForestClassifier} (criterion default squared error).
 * @returns The estimator.
 */
export function randomForestRegressor(hyper: Partial<ForestHyper> = {}): ExplainedEstimator<ForestHyper, FittedForest> {
  return forestEstimator(
    'random-forest-regressor',
    false,
    forestDefaults(hyper, base('squared_error', 'all', true, 'best')),
  )
}

/**
 * Extremely randomised trees for classification (sklearn.ensemble.ExtraTreesClassifier): every row, one random
 * threshold per feature, sqrt(d) features per split. Exposition as {@link randomForestClassifier} (no out-of-bag
 * curve unless bootstrapping).
 *
 * @param hyper - As {@link randomForestClassifier} (bootstrap default false).
 * @returns The estimator.
 */
export function extraTreesClassifier(hyper: Partial<ForestHyper> = {}): ExplainedEstimator<ForestHyper, FittedForest> {
  return forestEstimator('extra-trees-classifier', true, forestDefaults(hyper, base('gini', 'sqrt', false, 'random')))
}

/**
 * Extremely randomised trees for regression (sklearn.ensemble.ExtraTreesRegressor).
 *
 * @param hyper - As {@link randomForestRegressor} (bootstrap default false).
 * @returns The estimator.
 */
export function extraTreesRegressor(hyper: Partial<ForestHyper> = {}): ExplainedEstimator<ForestHyper, FittedForest> {
  return forestEstimator(
    'extra-trees-regressor',
    false,
    forestDefaults(hyper, base('squared_error', 'all', false, 'random')),
  )
}

/** {@link randomForestClassifier} as a class. */
export class RandomForestClassifier extends ExplainedEstimatorClass<ForestHyper, FittedForest> {
  /** @param hyper - As {@link randomForestClassifier}. */
  constructor(hyper: Partial<ForestHyper> = {}) {
    super(randomForestClassifier(hyper))
  }
}

/** {@link randomForestRegressor} as a class. */
export class RandomForestRegressor extends ExplainedEstimatorClass<ForestHyper, FittedForest> {
  /** @param hyper - As {@link randomForestRegressor}. */
  constructor(hyper: Partial<ForestHyper> = {}) {
    super(randomForestRegressor(hyper))
  }
}

/** {@link extraTreesClassifier} as a class. */
export class ExtraTreesClassifier extends ExplainedEstimatorClass<ForestHyper, FittedForest> {
  /** @param hyper - As {@link extraTreesClassifier}. */
  constructor(hyper: Partial<ForestHyper> = {}) {
    super(extraTreesClassifier(hyper))
  }
}

/** {@link extraTreesRegressor} as a class. */
export class ExtraTreesRegressor extends ExplainedEstimatorClass<ForestHyper, FittedForest> {
  /** @param hyper - As {@link extraTreesRegressor}. */
  constructor(hyper: Partial<ForestHyper> = {}) {
    super(extraTreesRegressor(hyper))
  }
}
