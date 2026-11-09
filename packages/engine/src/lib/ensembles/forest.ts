/**
 * Random forests and extremely randomised trees (as scikit-learn's), and the random trees embedding, on the iterative
 * base: many decision trees, each grown by the library's grower on a bootstrap sample of the rows with a random subset
 * of the features at every split (or a random threshold per feature, for extra trees), averaged into one prediction.
 * A fit is a session with one tree per step, logging the out-of-bag error as the forest grows; the forest also gives the
 * leaf each row reaches in every tree, the proximity between rows, and the impurity importances. The embedding maps a row
 * to the leaves it falls in across totally random trees grown without targets.
 */

import { argmaxOf, OmaiError } from '../base/index.ts'
import { categorical, type Distribution, normal } from '../dists/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import { type FittedFacts, type LearnData, type Verbs, withDefaultScore } from '../learn/estimator.ts'
import { type BalancedBagOptions, balancedBag } from '../learn/preprocessing/resampling.ts'
import { matrixView, sampleWeightsOf, takeRows } from '../learn/validation.ts'
import { foldIn, uniform } from '../random/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import type { Criterion, RegressionCriterion } from '../trees/criteria.ts'
import { type GrowOptions, growTree, type MaxFeatures, type TreeData, treeData } from '../trees/grow.ts'
import { applyTree, type FittedTree, leafCount, treeFeatureImportances, treeOutputs } from '../trees/tree.ts'
import { stagesFromRun } from './_stages.ts'
import {
  type AveragingMember,
  type AveragingPath,
  type AveragingState,
  averagingAlgorithm,
  averagingSources,
} from './averaging.ts'

/** Hyperparameters of the forests. */
export interface ForestHyper {
  readonly nEstimators: number
  readonly criterion: Criterion
  readonly maxDepth?: number
  readonly minSamplesSplit: number
  readonly minSamplesLeaf: number
  readonly maxFeatures: MaxFeatures
  readonly maxLeafNodes?: number
  readonly minImpurityDecrease: number
  /** Draw a bootstrap sample per tree (random forests) or use every row (extra trees). */
  readonly bootstrap: boolean
  /** Rows per bootstrap, as a fraction when at most 1, else a count (default all). */
  readonly maxSamples?: number
  readonly splitter: 'best' | 'random'
  /** Indices of the categorical features. */
  readonly categorical: readonly number[]
  /** Balance each tree's sample by random under-sampling (as imbalanced-learn's balanced forest), see {@link balancedBag}. */
  readonly balance?: BalancedBagOptions
}

/** Parameters of a fitted forest. */
export type ForestParams = {
  readonly trees: FittedTree[]
  readonly classes: number[]
  readonly nFeatures: number
  /** The out-of-bag score (accuracy or R^2) when trees were sampled. */
  readonly oobScore: number | null
}

/** Verbs and facts of a fitted forest. */
export interface ForestVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  /** Categorical averaged probabilities, or normal with the trees' mean and spread. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The leaf each row reaches in each tree, `[m, trees]`. */
  readonly apply: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureImportances: readonly number[]
}

/**
 * The paths a forest logs and declares: the averaging ensemble's, the importances so far (`ensemble/importances`) and
 * the derived `model/proximity` and `model/importances`.
 *
 * @example
 * ```ts
 * import type { ForestPath } from 'omai-engine/ensembles'
 * const path: ForestPath = 'model/proximity'
 * path // => 'model/proximity'
 * ```
 */
export type ForestPath = AveragingPath | 'ensemble/importances' | 'model/proximity' | 'model/importances'

/** A forest estimator. */
export type ForestEstimator = IterativeEstimator<
  ForestHyper,
  AveragingState<FittedTree>,
  ForestParams,
  ForestVerbs,
  ForestPath
>

function forestVerbs(p: ForestParams): ForestVerbs {
  const classifier = p.classes.length > 0
  const k = p.classes.length
  const outputs = (x: TensorLike): Float64Array[] => p.trees.map((t) => Float64Array.from(toFlat(treeOutputs(t, x))))
  const average = (x: TensorLike): { readonly mean: Float64Array; readonly sd: Float64Array } => {
    const outs = outputs(x)
    const len = outs[0]?.length ?? 0
    const mean = new Float64Array(len)
    for (const o of outs) for (let i = 0; i < len; i++) mean[i] = (mean[i] ?? 0) + (o[i] ?? 0) / outs.length
    const sd = new Float64Array(len)
    for (const o of outs)
      for (let i = 0; i < len; i++) sd[i] = (sd[i] ?? 0) + ((o[i] ?? 0) - (mean[i] ?? 0)) ** 2 / outs.length
    return { mean, sd: sd.map(Math.sqrt) }
  }
  const proba = (x: TensorLike): Tensor => {
    const { mean } = average(x)
    return fromStorage(mean, [mean.length / k, k], 'float64')
  }
  const importances = new Array<number>(p.nFeatures).fill(0)
  for (const t of p.trees)
    for (const [j, v] of treeFeatureImportances(t).entries())
      importances[j] = (importances[j] ?? 0) + v / p.trees.length
  const common = {
    apply: (x: TensorLike) => {
      const leaves = p.trees.map((t) => applyTree(t, x))
      const m = leaves[0]?.length ?? 0
      const out = new Float64Array(m * p.trees.length)
      leaves.forEach((l, t) => {
        for (let i = 0; i < m; i++) out[i * p.trees.length + t] = l[i] ?? 0
      })
      return fromStorage(out, [m, p.trees.length], 'float64')
    },
    nFeaturesIn: p.nFeatures,
    featureImportances: importances,
  }
  if (classifier)
    return withDefaultScore('classifier', {
      ...common,
      predict: (x: TensorLike) => {
        const pr = proba(x)
        const m = pr.shape[0] ?? 0
        return fromStorage(
          Float64Array.from({ length: m }, (_, i) => p.classes[argmaxOf(pr.data.subarray(i * k, (i + 1) * k))] ?? 0),
          [m],
          'float64',
        )
      },
      predictProba: proba,
      predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }),
      classes: p.classes,
    })
  return withDefaultScore('regressor', {
    ...common,
    predict: (x: TensorLike) => {
      const { mean } = average(x)
      return fromStorage(mean, [mean.length], 'float64')
    },
    predictDistribution: (x: TensorLike) => {
      const { mean, sd } = average(x)
      return normal(
        fromStorage(mean, [mean.length], 'float64'),
        fromStorage(
          sd.map((v) => Math.max(v, 1e-12)),
          [mean.length],
          'float64',
        ),
      )
    },
  })
}

/** The fraction of trees in which each pair of training rows share a leaf, `[n, n]`. */
function proximityOf(trees: readonly FittedTree[], x: TensorLike): number[][] {
  const leaves = trees.map((t) => applyTree(t, x))
  const n = leaves[0]?.length ?? 0
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => leaves.filter((l) => l[i] === l[j]).length / Math.max(1, trees.length)),
  )
}

/**
 * The forest estimator under a name: the engine of the random forests, the extra trees and the balanced forest.
 *
 * @param name - The estimator's name.
 * @param classifier - Classification or regression.
 * @param defaults - The hyperparameters in full.
 * @returns The estimator.
 * @example
 * ```ts
 * import { forestEstimator } from 'omai-engine/ensembles'
 * const estimator = forestEstimator('my-forest', true, {
 *   nEstimators: 3,
 *   criterion: 'gini',
 *   minSamplesSplit: 2,
 *   minSamplesLeaf: 1,
 *   maxFeatures: 'sqrt',
 *   minImpurityDecrease: 0,
 *   bootstrap: true,
 *   splitter: 'best',
 *   categorical: [],
 * })
 * estimator.name // => 'my-forest'
 * ```
 */
export function forestEstimator(name: string, classifier: boolean, defaults: ForestHyper): ForestEstimator {
  const prepare = (learn: LearnData): TreeData => treeData(learn, classifier, name)
  const growOf = (h: ForestHyper): GrowOptions => ({
    criterion: h.criterion,
    splitter: h.splitter,
    growth: 'depth',
    ...(h.maxDepth === undefined ? {} : { maxDepth: h.maxDepth }),
    minSamplesSplit: h.minSamplesSplit,
    minSamplesLeaf: h.minSamplesLeaf,
    minWeightFractionLeaf: 0,
    maxFeatures: h.maxFeatures,
    ...(h.maxLeafNodes === undefined ? {} : { maxLeafNodes: h.maxLeafNodes }),
    minImpurityDecrease: h.minImpurityDecrease,
    record: 0,
  })
  return iterative<ForestHyper, AveragingState<FittedTree>, ForestParams, ForestVerbs, ForestPath>(
    {
      name,
      kind: classifier ? 'classifier' : 'regressor',
      defaults,
      hyperSpace: (h) =>
        dict({
          nEstimators: int(1, Math.max(300, h.nEstimators), { default: h.nEstimators, label: 'trees' }),
          maxFeatures: choice<string | number>(
            ['sqrt', 'log2', 'all', ...(typeof h.maxFeatures === 'number' ? [h.maxFeatures] : [])],
            {
              label: 'features per split',
              description: 'How many features each split considers: √d, log₂ d, all of them, or a count or fraction.',
              default: h.maxFeatures,
            },
          ),
          minSamplesLeaf: int(1, Math.max(50, h.minSamplesLeaf), {
            default: h.minSamplesLeaf,
            label: 'minimum leaf size',
          }),
          ...(h.maxDepth === undefined
            ? {}
            : { maxDepth: int(1, Math.max(30, h.maxDepth), { default: h.maxDepth, label: 'maximum depth' }) }),
        }),
      algorithm: (learn, h): Algorithm<AveragingState<FittedTree>, ForestPath> => {
        const data = prepare(learn)
        const n = data.x.n
        if (!(h.nEstimators >= 1)) throw new OmaiError('value', `${name}: nEstimators must be at least 1`)
        const options = growOf(h)
        const member: AveragingMember<FittedTree, 'ensemble/importances'> = {
          fit: (counts, key) => {
            const rows = Array.from({ length: n }, (_, i) => i).filter((i) => (counts[i] ?? 0) > 0)
            const sub: TreeData =
              rows.length === n && counts.every((c) => c === 1)
                ? data
                : {
                    x: takeRows(data.x, rows),
                    y: Float64Array.from(rows, (r) => data.y[r] ?? 0),
                    w: Float64Array.from(rows, (r) => (data.w[r] ?? 0) * (counts[r] ?? 0)),
                    classes: data.classes,
                  }
            return growTree(sub, options, { key, categorical: h.categorical }).tree
          },
          outputs: (tree) =>
            Float64Array.from(toFlat(treeOutputs(tree, fromStorage(data.x.data, [n, data.x.d], 'float64')))),
          after: (trees, ctx) => {
            if (!ctx.wants('ensemble/importances')) return
            const total = new Array<number>(data.x.d).fill(0)
            for (const t of trees)
              for (const [j, v] of treeFeatureImportances(t).entries()) total[j] = (total[j] ?? 0) + v / trees.length
            ctx.log('ensemble/importances', total)
          },
        }
        return averagingAlgorithm(
          name,
          { y: data.y, k: data.classes.length },
          {
            nEstimators: h.nEstimators,
            bootstrap: h.bootstrap,
            ...(h.maxSamples === undefined ? {} : { maxSamples: h.maxSamples <= 1 ? h.maxSamples * n : h.maxSamples }),
            ...(h.balance === undefined
              ? {}
              : {
                  balance: (counts, key) =>
                    balancedBag(
                      Array.from(data.y, (code) => data.classes[code] ?? code),
                      counts,
                      h.balance as BalancedBagOptions,
                      key,
                    ),
                }),
          },
          member,
        )
      },
      params: (state, learn) => {
        const data = prepare(learn)
        return {
          trees: [...state.members],
          classes: [...data.classes],
          nFeatures: data.x.d,
          oobScore: Number.isFinite(state.oobScores.at(-1)) ? (state.oobScores.at(-1) as number) : null,
        }
      },
      verbs: (p) => forestVerbs(p),
      maxSteps: (h) => h.nEstimators,
      contract: (learn): Contract<AveragingState<FittedTree>, ForestPath> =>
        defineContract({
          ...averagingSources(classifier),
          'ensemble/importances': tensor({
            role: 'output',
            shape: ['features'],
            description: 'the impurity importance of each feature, averaged over the trees so far',
          }),
          'model/proximity': derived<AveragingState<FittedTree>>({
            description: 'the fraction of trees in which each pair of training rows share a leaf',
            cost: 'dear',
            compute: (state) => proximityOf(state.members, learn.x),
          }),
          'model/importances': derived<AveragingState<FittedTree>>({
            description: 'the impurity importance of each feature over the trees so far',
            compute: (state) => {
              const total = new Array<number>(matrixView(learn.x, name, { allowNaN: true }).d).fill(0)
              for (const t of state.members)
                for (const [j, v] of treeFeatureImportances(t).entries())
                  total[j] = (total[j] ?? 0) + v / state.members.length
              return total
            },
          }),
        }),
      projections: {
        'ensemble/stages': (run) =>
          stagesFromRun(run, {
            method: name,
            scoreName: classifier ? 'accuracy' : 'R²',
            higherIsBetter: true,
            weight: 'ensemble/weight',
            train: 'ensemble/trainScore',
            valid: 'ensemble/oobScore',
            bootstrap: 'ensemble/bootstrap',
          }) as Exposition | undefined,
      },
      logs: {
        'ensemble/member': Log(),
        'ensemble/weight': Log(),
        'ensemble/trainScore': Log(),
        'ensemble/oobScore': Log(),
        'ensemble/oobError': Log(),
        'ensemble/bootstrap': Log({ detail: 'full' }),
      },
    },
    defaults,
  )
}

function forestDefaults(hyper: Partial<ForestHyper>, base: ForestHyper): ForestHyper {
  return {
    ...base,
    ...hyper,
    categorical: hyper.categorical ?? [],
  }
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
  minImpurityDecrease: 0,
  bootstrap,
  splitter,
  categorical: [],
})

/**
 * A random forest classifier (sklearn.ensemble.RandomForestClassifier): trees grown on keyed bootstrap samples with
 * sqrt(d) features per split, probabilities averaged. A session adds one tree per step and logs `ensemble/member`,
 * `ensemble/oobError`, `ensemble/memberError`, `ensemble/bootstrap`, `ensemble/importances` and the training and
 * out-of-bag scores; the probes are `model/proximity` and `model/importances`, and the fit records the stages.
 *
 * @param hyper - Trees (default 100), criterion (default gini), depth and leaf limits, features per split, bootstrap.
 * @returns The estimator.
 * @concept ensembles
 * @index estimator Random forest; regressor and extra-trees twins beside it.
 * @example
 * ```ts
 * import { randomForestClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40, ((i * 7) % 11) / 11])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * randomForestClassifier({ nEstimators: 8 }).fit({ x, y }, { key: key(0) }).params.trees.length // => 8
 * ```
 */
export function randomForestClassifier(hyper: Partial<ForestHyper> = {}): ForestEstimator {
  return forestEstimator('random-forest-classifier', true, forestDefaults(hyper, base('gini', 'sqrt', true, 'best')))
}

/**
 * A random forest regressor (sklearn.ensemble.RandomForestRegressor): all features per split by default, predictions
 * averaged; `predictDistribution` is normal with the trees' mean and standard deviation.
 *
 * @param hyper - As {@link randomForestClassifier} (criterion default squared error).
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { randomForestRegressor } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 30])
 * randomForestRegressor({ nEstimators: 5 }).fit({ x, y: x.map(([a = 0]) => a * a) }, { key: key(0) }).params.trees.length // => 5
 * ```
 */
export function randomForestRegressor(
  hyper: Partial<ForestHyper & { readonly criterion: RegressionCriterion }> = {},
): ForestEstimator {
  return forestEstimator(
    'random-forest-regressor',
    false,
    forestDefaults(hyper, base('squared_error', 'all', true, 'best')),
  )
}

/**
 * Extremely randomised trees for classification (sklearn.ensemble.ExtraTreesClassifier): every row, one random
 * threshold per feature, sqrt(d) features per split; no bootstrap by default, so no out-of-bag score.
 *
 * @param hyper - As {@link randomForestClassifier} (bootstrap default false).
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { extraTreesClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40, ((i * 7) % 11) / 11])
 * const y = x.map(([a = 0]) => (a < 0.5 ? 0 : 1))
 * extraTreesClassifier({ nEstimators: 6 }).fit({ x, y }, { key: key(0) }).params.oobScore // => null
 * ```
 */
export function extraTreesClassifier(hyper: Partial<ForestHyper> = {}): ForestEstimator {
  return forestEstimator('extra-trees-classifier', true, forestDefaults(hyper, base('gini', 'sqrt', false, 'random')))
}

/**
 * Extremely randomised trees for regression (sklearn.ensemble.ExtraTreesRegressor).
 *
 * @param hyper - As {@link randomForestRegressor} (bootstrap default false).
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { extraTreesRegressor } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 30])
 * extraTreesRegressor({ nEstimators: 5 }).fit({ x, y: x.map(([a = 0]) => a) }, { key: key(0) }).params.trees.length // => 5
 * ```
 */
export function extraTreesRegressor(
  hyper: Partial<ForestHyper & { readonly criterion: RegressionCriterion }> = {},
): ForestEstimator {
  return forestEstimator(
    'extra-trees-regressor',
    false,
    forestDefaults(hyper, base('squared_error', 'all', false, 'random')),
  )
}

/** Options of {@link randomTreesEmbedding}. */
export interface EmbeddingHyper {
  readonly nEstimators: number
  readonly maxDepth: number
  readonly minSamplesSplit: number
}

/** The trees of a fitted embedding, and where each tree's leaves start in the embedding's columns. */
export type EmbeddingParams = {
  readonly trees: FittedTree[]
  /** `offsets[t]` is the first column of tree t's leaves; the last entry is the number of columns. */
  readonly offsets: number[]
  readonly nFeatures: number
}

/** The state of the embedding algorithm: the trees so far. */
export interface EmbeddingState {
  readonly trees: readonly FittedTree[]
}

/** Verbs of a fitted embedding. */
export interface EmbeddingVerbs extends Verbs, FittedFacts {
  readonly transform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * The paths the random trees embedding logs: trees so far and leaves in all of them.
 *
 * @example
 * ```ts
 * import type { EmbeddingPath } from 'omai-engine/ensembles'
 * const path: EmbeddingPath = 'ensemble/leaves'
 * path // => 'ensemble/leaves'
 * ```
 */
export type EmbeddingPath = 'ensemble/member' | 'ensemble/leaves'

/** An embedding estimator. */
export type EmbeddingEstimator = IterativeEstimator<
  EmbeddingHyper,
  EmbeddingState,
  EmbeddingParams,
  EmbeddingVerbs,
  EmbeddingPath
>

/**
 * The random trees embedding as an algorithm: one totally random tree per step, grown on uniformly random targets with
 * one random feature and one random threshold per split (sklearn's `RandomTreesEmbedding`), so its leaves partition the
 * space without looking at any label. Logs `ensemble/member` and `ensemble/leaves` (leaves in all trees so far).
 *
 * @param data - Rows `[n, d]` (targets are ignored).
 * @param options - Trees, depth and minimum samples to split.
 * @returns The algorithm.
 * @concept ensembles
 * @example
 * ```ts
 * import { randomTreesEmbeddingAlgorithm } from 'omai-engine/ensembles'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 20 }, (_, i) => [i / 20, (i * 3) % 7])
 * const alg = randomTreesEmbeddingAlgorithm({ x }, { nEstimators: 3, maxDepth: 2, minSamplesSplit: 2 })
 * runAlgorithm(alg, { maxSteps: 10, seed: 0 }).state.trees.length // => 3
 * ```
 */
export function randomTreesEmbeddingAlgorithm(
  data: LearnData,
  options: EmbeddingHyper,
): Algorithm<EmbeddingState, EmbeddingPath> {
  const m = matrixView(data.x, 'randomTreesEmbedding', { allowNaN: true })
  const w = sampleWeightsOf(undefined, m.n, 'randomTreesEmbedding')
  const grow: GrowOptions = {
    criterion: 'squared_error',
    splitter: 'random',
    growth: 'depth',
    maxDepth: options.maxDepth,
    minSamplesSplit: options.minSamplesSplit,
    minSamplesLeaf: 1,
    minWeightFractionLeaf: 0,
    maxFeatures: 1,
    minImpurityDecrease: 0,
    record: 0,
  }
  return {
    name: 'randomTreesEmbedding',
    config: { nEstimators: options.nEstimators, maxDepth: options.maxDepth },
    init: () => ({ trees: [] }),
    step: (s, ctx) => {
      const y = toFlat(uniform(foldIn(ctx.key, 0), [m.n]))
      const tree = growTree({ x: m, y: Float64Array.from(y), w, classes: [] }, grow, { key: foldIn(ctx.key, 1) }).tree
      const trees = [...s.trees, tree]
      if (ctx.wants('ensemble/member')) ctx.log('ensemble/member', trees.length)
      if (ctx.wants('ensemble/leaves'))
        ctx.log(
          'ensemble/leaves',
          trees.reduce((a, t) => a + leafCount(t), 0),
        )
      return { trees }
    },
    done: (s) => (s.trees.length >= options.nEstimators ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EmbeddingState,
  }
}

/**
 * The random trees embedding (sklearn.ensemble.RandomTreesEmbedding) as an estimator: `transform` maps each row to a
 * one-hot indicator of the leaf it reaches in each tree (a dense `[m, leaves]` matrix), an unsupervised, non-linear,
 * high-dimensional representation in which linear models do well.
 *
 * @param hyper - Trees (default 10), depth (5) and minimum samples to split (2).
 * @returns The estimator.
 * @concept ensembles
 * @example
 * ```ts
 * import { randomTreesEmbedding } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 20 }, (_, i) => [i / 20])
 * const f = randomTreesEmbedding({ nEstimators: 3, maxDepth: 2 }).fit({ x }, { key: key(0) })
 * // One leaf per tree is hot in every row.
 * toFlat(f.transform([[0.3]])).reduce((a, b) => a + b, 0) // => 3
 * ```
 */
export function randomTreesEmbedding(hyper: Partial<EmbeddingHyper> = {}): EmbeddingEstimator {
  const defaults: EmbeddingHyper = {
    nEstimators: hyper.nEstimators ?? 10,
    maxDepth: hyper.maxDepth ?? 5,
    minSamplesSplit: hyper.minSamplesSplit ?? 2,
  }
  return iterative<EmbeddingHyper, EmbeddingState, EmbeddingParams, EmbeddingVerbs, EmbeddingPath>(
    {
      name: 'random-trees-embedding',
      kind: 'transformer',
      defaults,
      hyperSpace: (h) =>
        dict({
          nEstimators: int(1, Math.max(100, h.nEstimators), { default: h.nEstimators, label: 'trees' }),
          maxDepth: int(1, Math.max(12, h.maxDepth), { default: h.maxDepth, label: 'maximum depth' }),
        }),
      algorithm: (data, h) => randomTreesEmbeddingAlgorithm(data, h),
      params: (state, data) => {
        const offsets = [0]
        for (const t of state.trees) offsets.push((offsets.at(-1) ?? 0) + leafCount(t))
        return {
          trees: [...state.trees],
          offsets,
          nFeatures: matrixView(data.x, 'randomTreesEmbedding', { allowNaN: true }).d,
        }
      },
      verbs: (p) => {
        // The column of each leaf of each tree, in node order.
        const columns = p.trees.map((t, i) => {
          const at = new Map<number, number>()
          t.left.forEach((l, node) => {
            if (l < 0) at.set(node, (p.offsets[i] ?? 0) + at.size)
          })
          return at
        })
        return {
          transform: (x: TensorLike) => {
            const leaves = p.trees.map((t) => applyTree(t, x))
            const m = leaves[0]?.length ?? 0
            const width = p.offsets.at(-1) ?? 0
            const out = new Float64Array(m * width)
            leaves.forEach((l, t) => {
              for (let i = 0; i < m; i++) out[i * width + (columns[t]?.get(l[i] ?? 0) ?? 0)] = 1
            })
            return fromStorage(out, [m, width], 'float64')
          },
          nFeaturesIn: p.nFeatures,
          featureNamesOut: () => columns.flatMap((c, i) => [...c.keys()].map((node) => `tree${i}_leaf${node}`)),
        }
      },
      maxSteps: (h) => h.nEstimators,
      contract: (): Contract<EmbeddingState, EmbeddingPath> =>
        defineContract({
          'ensemble/member': scalar({ description: 'trees so far' }),
          'ensemble/leaves': scalar({ description: 'leaves in all the trees so far (the embedding’s width)' }),
        }),
      logs: { 'ensemble/member': Log(), 'ensemble/leaves': Log() },
    },
    defaults,
  )
}
