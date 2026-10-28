/**
 * The decision tree estimators on the learn contract, built on the iterative base: `decisionTreeClassifier` and
 * `decisionTreeRegressor` (CART as scikit-learn's), and the extremely randomised `extraTreeClassifier` and
 * `extraTreeRegressor`. A fit is a session of the growth algorithm, one node decided per step, so `.method(data)` and
 * `.session(data)` exist beside `.fit`, and `fittedAt(t)` is the tree as it stood after t nodes. The fitted value
 * predicts, gives probabilities, the leaf each row reaches (`apply`), its decision path, the tree as a structure or as
 * text, and the impurity importances.
 */

import type { Json } from '../base/index.ts'
import { categorical, type Distribution, normal } from '../dists/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import { type FittedFacts, type LearnData, type Verbs, withDefaultScore } from '../learn/estimator.ts'
import { classNamesOf, featureNamesOf } from '../learn/explained.ts'
import { rowsOf } from '../learn/validation.ts'
import type { Structure } from '../scene/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { derived, Log, message, scalar, tensor } from '../trace/index.ts'
import type { ClassificationCriterion, Criterion, RegressionCriterion } from './criteria.ts'
import {
  type GrowOptions,
  type GrowState,
  growTreeAlgorithm,
  splitProblem,
  type TreeData,
  type TreeInfo,
  treeData,
  treeOfState,
  treeRecordOf,
} from './grow.ts'
import { costComplexityPath, pruneTree } from './prune.ts'
import {
  applyTree,
  decisionPath,
  exportText,
  predictTree,
  type Tree,
  treeFeatureImportances,
  treeOutputs,
  treeToStructure,
} from './tree.ts'

/** Hyperparameters of the decision trees (scikit-learn's, with `growth` and `categorical` added). */
export interface DecisionTreeHyper extends GrowOptions {
  /** Minimal cost-complexity pruning (default 0, none). */
  readonly ccpAlpha: number
  /** Indices of the features whose values are category codes (split by subsets). */
  readonly categorical: readonly number[]
}

/** The verbs and facts of a fitted tree. */
export interface TreeVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Categorical leaf proportions (classifiers), or normal with the leaf mean and standard deviation (regressors). */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The leaf each row reaches. */
  readonly apply: (x: TensorLike) => Int32Array
  /** The nodes each row passes, root to leaf. */
  readonly decisionPath: (x: TensorLike) => number[][]
  /** The tree as a structure for drawing. */
  readonly describe: (features?: readonly string[], classNames?: readonly string[]) => Structure
  readonly exportText: (features?: readonly string[]) => string
  readonly nFeaturesIn: number
  readonly featureImportances: readonly number[]
}

/**
 * The verbs of a tree from its arrays (what every tree estimator and ensemble of trees builds on).
 *
 * @param tree - The tree.
 * @returns Predictions, probabilities, apply, decision path, structure, text and importances.
 * @example
 * ```ts
 * import { stumpTree, treeVerbs } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * toFlat(treeVerbs(stumpTree({ feature: 0, threshold: 0.5, left: 0, right: 1 })).predict([[0], [1]])) // => [0, 1]
 * ```
 */
export function treeVerbs(tree: Tree): TreeVerbs {
  const classifier = tree.task === 'classification'
  return {
    predict: (x) => predictTree(tree, x),
    ...(classifier ? { predictProba: (x: TensorLike) => treeOutputs(tree, x), classes: tree.classes } : {}),
    predictDistribution: (x) => {
      if (classifier) return categorical({ probs: treeOutputs(tree, x) })
      const leaves = applyTree(tree, x)
      const loc = Float64Array.from(leaves, (l) => tree.value[l]?.[0] ?? 0)
      const scale = Float64Array.from(leaves, (l) => Math.sqrt(Math.max(tree.impurity[l] ?? 0, 1e-12)))
      return normal(fromStorage(loc, [loc.length], 'float64'), fromStorage(scale, [scale.length], 'float64'))
    },
    apply: (x) => applyTree(tree, x),
    decisionPath: (x) => decisionPath(tree, x),
    describe: (features, classNames) => treeToStructure(tree, features, classNames),
    exportText: (features) => exportText(tree, features),
    nFeaturesIn: tree.nFeatures,
    featureImportances: treeFeatureImportances(tree),
  }
}

/** The tree estimators' type: the iterative base over growth states, with a tree as parameters. */
export type TreeEstimator = IterativeEstimator<DecisionTreeHyper, GrowState, Tree, TreeVerbs>

/** The tree's data and facts, prepared once per dataset. */
interface Prepared {
  readonly data: TreeData
  readonly info: TreeInfo
}

function treeEstimator(name: string, classifier: boolean, defaults: DecisionTreeHyper): TreeEstimator {
  const prepared = new WeakMap<LearnData, Prepared>()
  const prepare = (learn: LearnData): Prepared => {
    let p = prepared.get(learn)
    if (p === undefined) {
      const data = treeData(learn, classifier, name)
      p = {
        data,
        info: { task: classifier ? 'classification' : 'regression', classes: data.classes, nFeatures: data.x.d },
      }
      prepared.set(learn, p)
    }
    return p
  }
  const criteria: readonly Criterion[] = classifier
    ? ['gini', 'entropy', 'log_loss', 'gain_ratio']
    : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
  return iterative<DecisionTreeHyper, GrowState, Tree, TreeVerbs>({
    name,
    kind: classifier ? 'classifier' : 'regressor',
    defaults,
    hyperSpace: (h) =>
      dict({
        criterion: choice(criteria, {
          label: 'criterion',
          description: 'The impurity a split minimises.',
          default: h.criterion,
        }),
        growth: choice(['depth', 'breadth', 'best'] as const, {
          label: 'growth order',
          description: 'Which open node is split next: deepest, shallowest, or the largest impurity decrease.',
          default: h.growth ?? 'depth',
        }),
        minSamplesLeaf: int(1, Math.max(50, h.minSamplesLeaf), {
          default: h.minSamplesLeaf,
          label: 'minimum leaf size',
        }),
        ccpAlpha: real(0, 1, { default: h.ccpAlpha, label: 'pruning alpha' }),
        ...(h.maxDepth === undefined
          ? {}
          : { maxDepth: int(1, Math.max(30, h.maxDepth), { default: h.maxDepth, label: 'maximum depth' }) }),
      }),
    algorithm: (learn, h) => {
      const { data, info } = prepare(learn)
      return growTreeAlgorithm(splitProblem(data, h, h.categorical), h, info)
    },
    params: (state, learn, h) => pruneTree(treeOfState(state, prepare(learn).info), h.ccpAlpha),
    verbs: (tree) => withDefaultScore(classifier ? 'classifier' : 'regressor', treeVerbs(tree)),
    maxSteps: (_h, learn) => 4 * rowsOf(learn.x) + 10,
    contract: (learn, h) => {
      const { info } = prepare(learn)
      const names = featureNamesOf(learn, info.nFeatures)
      const classNames = info.task === 'classification' ? classNamesOf(learn, info.classes) : undefined
      return {
        'tree/split': message({
          description: 'the node just decided: its feature, threshold and impurity decrease, or that it is a leaf',
          payload: {
            node: 'node id',
            feature: 'feature index',
            threshold: 'cut',
            decrease: 'weighted impurity decrease',
          },
        }),
        'tree/candidates': tensor({
          role: 'output',
          shape: ['candidates', 5],
          description: 'every candidate split of the node: feature, threshold, left and right impurity, decrease',
        }),
        'tree/leaves': scalar({ description: 'leaves so far' }),
        'tree/depth': scalar({ description: 'depth of the deepest node so far' }),
        'tree/impurity': scalar({ description: 'weighted impurity of the leaves', direction: 'lower' }),
        [info.task === 'classification' ? 'eval/accuracy' : 'eval/r2']: scalar({
          description: `training ${info.task === 'classification' ? 'accuracy' : 'R²'} of the tree so far`,
          direction: 'higher',
        }),
        'model/structure': derived<GrowState>({
          description: 'the tree grown so far, with every candidate split considered at each node',
          compute: (state) =>
            treeRecordOf(state, {
              criterion: h.criterion,
              features: names,
              ...(classNames === undefined ? {} : { classes: classNames }),
            }),
        }),
        'model/importances': derived<GrowState>({
          description: 'the impurity importance of each feature in the tree so far',
          compute: (state) => treeFeatureImportances(pruneTree(treeOfState(state, info), h.ccpAlpha)) as Json,
        }),
      }
    },
    records: (state, learn, h): readonly Exposition[] => {
      if (h.record === 0) return []
      const { info } = prepare(learn)
      const names = featureNamesOf(learn, info.nFeatures)
      const classNames = info.task === 'classification' ? classNamesOf(learn, info.classes) : undefined
      const grown = treeOfState(state, info)
      return [
        treeRecordOf(state, {
          criterion: h.criterion,
          features: names,
          ...(classNames === undefined ? {} : { classes: classNames }),
        }),
        costComplexityPath(grown),
      ]
    },
    logs: {
      'tree/split': Log(),
      'tree/leaves': Log(),
      'tree/impurity': Log(),
      [classifier ? 'eval/accuracy' : 'eval/r2']: Log(),
    },
  })
}

function growDefaults(hyper: Partial<DecisionTreeHyper>, criterion: Criterion): DecisionTreeHyper {
  return {
    criterion: hyper.criterion ?? criterion,
    splitter: hyper.splitter ?? 'best',
    growth: hyper.growth ?? 'depth',
    ...(hyper.maxDepth === undefined ? {} : { maxDepth: hyper.maxDepth }),
    minSamplesSplit: hyper.minSamplesSplit ?? 2,
    minSamplesLeaf: hyper.minSamplesLeaf ?? 1,
    minWeightFractionLeaf: hyper.minWeightFractionLeaf ?? 0,
    maxFeatures: hyper.maxFeatures ?? 'all',
    ...(hyper.maxLeafNodes === undefined ? {} : { maxLeafNodes: hyper.maxLeafNodes }),
    minImpurityDecrease: hyper.minImpurityDecrease ?? 0,
    ccpAlpha: hyper.ccpAlpha ?? 0,
    categorical: hyper.categorical ?? [],
    ...(hyper.record === undefined ? {} : { record: hyper.record }),
  }
}

/**
 * A CART classification tree (sklearn.tree.DecisionTreeClassifier), grown by a stepped algorithm, one node decided per
 * step: the session logs `tree/split`, `tree/candidates` (every candidate split of the node), `tree/leaves`,
 * `tree/depth`, `tree/impurity` and `eval/accuracy`, and answers the probes `model/structure` (a tree record with the
 * candidates of every node), `model/importances`, `model/params`, `model/predictions` and `model/decision-surface`.
 * Categorical features (`categorical`), missing values (NaN), sample weights and best-first growth (`maxLeafNodes`) are
 * supported.
 *
 * @param hyper - Criterion (default gini), splitter, growth order, depth and leaf limits, feature subsampling,
 *   minimum decrease, pruning alpha, categorical features and how many candidates per feature to record.
 * @returns The estimator, with `method(data)` and `session(data)` beside `fit`.
 * @concept decision-trees
 * @example
 * ```ts
 * import { decisionTreeClassifier } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * const data = { x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] }
 * const fitted = decisionTreeClassifier().fit(data)
 * fitted.params.threshold // => [1.5, -2, -2]
 * toFlat(fitted.predict([[0.2], [2.7]])) // => [0, 1]
 * ```
 */
export function decisionTreeClassifier(
  hyper: Partial<DecisionTreeHyper & { readonly criterion: ClassificationCriterion | 'gain_ratio' }> = {},
): TreeEstimator {
  return treeEstimator('decision-tree-classifier', true, growDefaults(hyper, 'gini'))
}

/**
 * A CART regression tree (sklearn.tree.DecisionTreeRegressor); logging and probes as {@link decisionTreeClassifier},
 * with leaf means as values and `eval/r2` for the training R². `predictDistribution` is normal with each leaf's mean
 * and standard deviation.
 *
 * @param hyper - Criterion (default squared error) and the options of {@link decisionTreeClassifier}.
 * @returns The estimator.
 * @concept decision-trees
 * @example
 * ```ts
 * import { decisionTreeRegressor } from 'mdml-engine/trees'
 * import { toFlat } from 'mdml-engine/tensor'
 * const fitted = decisionTreeRegressor({ maxDepth: 1 }).fit({ x: [[0], [1], [2], [3]], y: [0, 0, 5, 5] })
 * toFlat(fitted.predict([[0.5], [2.5]])) // => [0, 5]
 * ```
 */
export function decisionTreeRegressor(
  hyper: Partial<DecisionTreeHyper & { readonly criterion: RegressionCriterion }> = {},
): TreeEstimator {
  return treeEstimator('decision-tree-regressor', false, growDefaults(hyper, 'squared_error'))
}

/**
 * An extremely randomised classification tree (sklearn.tree.ExtraTreeClassifier): one random threshold per feature,
 * sqrt(d) features per split.
 *
 * @param hyper - As {@link decisionTreeClassifier} (splitter random, maxFeatures sqrt by default).
 * @returns The estimator.
 * @concept decision-trees
 * @example
 * ```ts
 * import { extraTreeClassifier } from 'mdml-engine/trees'
 * import { key } from 'mdml-engine/random'
 * extraTreeClassifier().fit({ x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] }, { key: key(0) }).params.task // => 'classification'
 * ```
 */
export function extraTreeClassifier(hyper: Partial<DecisionTreeHyper> = {}): TreeEstimator {
  return treeEstimator(
    'extra-tree-classifier',
    true,
    growDefaults({ splitter: 'random', maxFeatures: 'sqrt', ...hyper }, 'gini'),
  )
}

/**
 * An extremely randomised regression tree (sklearn.tree.ExtraTreeRegressor): one random threshold per feature, all
 * features per split.
 *
 * @param hyper - As {@link decisionTreeRegressor} (splitter random by default).
 * @returns The estimator.
 * @concept decision-trees
 * @example
 * ```ts
 * import { extraTreeRegressor } from 'mdml-engine/trees'
 * import { key } from 'mdml-engine/random'
 * extraTreeRegressor().fit({ x: [[0], [1], [2], [3]], y: [0, 0, 5, 5] }, { key: key(0) }).params.task // => 'regression'
 * ```
 */
export function extraTreeRegressor(hyper: Partial<DecisionTreeHyper> = {}): TreeEstimator {
  return treeEstimator('extra-tree-regressor', false, growDefaults({ splitter: 'random', ...hyper }, 'squared_error'))
}
