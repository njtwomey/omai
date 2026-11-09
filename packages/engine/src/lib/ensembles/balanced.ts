/**
 * The balanced ensembles of imbalanced-learn, built on the ensembles of this folder: each member is fitted on a sample
 * in which every class is cut, by random under-sampling, to the size of the smallest. Balanced bagging, the balanced
 * random forest (a balanced bootstrap per tree), EasyEnsemble (AdaBoost members on balanced samples) and RUSBoost
 * (AdaBoost whose every round sees a balanced random sample). They are the same session as their plain counterparts
 * (one member or round per step, the same paths and stages), so the out-of-bag curves and the stages show the balanced
 * samples; a row a member's balanced sample left out counts as out of bag for it.
 */

import { decisionTreeClassifier } from '../trees/decision-tree.ts'
import { type AdaBoostEstimator, type AdaBoostHyper, adaBoostClassifier, adaBoostEstimator } from './adaboost.ts'
import { type BaggingEstimator, type BaggingHyper, baggingEstimator } from './bagging.ts'
import { type ForestEstimator, type ForestHyper, forestEstimator } from './forest.ts'

/**
 * Balanced bagging (imblearn BalancedBaggingClassifier): bagging in which each member sees a bootstrap sample whose
 * classes have been cut by random under-sampling (without replacement unless `balance.replacement`) to the size of the
 * smallest class in it. Probabilities are averaged as in {@link baggingClassifier}.
 *
 * @param hyper - As {@link baggingClassifier} (default a decision tree, 10 members, bootstrap) plus `balance`
 *   (default `'auto'`, no replacement).
 * @returns The estimator.
 * @concept ensembles
 * @index estimator Bagging of members fitted on class-balanced bootstrap samples.
 * @example
 * ```ts
 * import { balancedBaggingClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i < 34 ? i / 34 : 0.8 + (i - 34) / 20])
 * const y = x.map((_, i) => (i < 34 ? 0 : 1))
 * const fitted = balancedBaggingClassifier({ nEstimators: 5 }).fit({ x, y }, { key: key(0) })
 * toFlat(fitted.predict([[0.1], [1.05]])) // => [0, 1]
 * ```
 */
export function balancedBaggingClassifier(hyper: Partial<BaggingHyper> = {}): BaggingEstimator {
  return baggingEstimator('balanced-bagging-classifier', true, {
    estimator: hyper.estimator ?? decisionTreeClassifier({ record: 0 }),
    nEstimators: hyper.nEstimators ?? 10,
    maxSamples: hyper.maxSamples ?? 1,
    maxFeatures: hyper.maxFeatures ?? 1,
    bootstrap: hyper.bootstrap ?? true,
    balance: hyper.balance ?? {},
  })
}

/**
 * The balanced random forest (Chen, Liaw and Breiman 2004; imblearn BalancedRandomForestClassifier): a random forest in
 * which each tree is grown on a balanced sample, every class drawn with replacement down to the size of the smallest
 * (`balance` defaults to `{ samplingStrategy: 'all', replacement: true }`, and the usual bootstrap is off).
 *
 * @param hyper - As {@link randomForestClassifier} (100 trees, gini, sqrt features per split) plus `balance`.
 * @returns The estimator.
 * @concept ensembles
 * @index estimator Random forest whose every tree sees a class-balanced sample.
 * @example
 * ```ts
 * import { balancedRandomForestClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i < 34 ? i / 34 : 0.8 + (i - 34) / 20])
 * const y = x.map((_, i) => (i < 34 ? 0 : 1))
 * const fitted = balancedRandomForestClassifier({ nEstimators: 8 }).fit({ x, y }, { key: key(0) })
 * toFlat(fitted.predict([[0.1], [1.05]])) // => [0, 1]
 * ```
 */
export function balancedRandomForestClassifier(hyper: Partial<ForestHyper> = {}): ForestEstimator {
  return forestEstimator('balanced-random-forest-classifier', true, {
    nEstimators: 100,
    criterion: 'gini',
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    maxFeatures: 'sqrt',
    minImpurityDecrease: 0,
    bootstrap: false,
    splitter: 'best',
    ...hyper,
    categorical: hyper.categorical ?? [],
    balance: hyper.balance ?? { samplingStrategy: 'all', replacement: true },
  })
}

/**
 * EasyEnsemble (Liu, Wu and Zhou 2009; imblearn EasyEnsembleClassifier): an ensemble of AdaBoost classifiers, each
 * fitted on a different balanced random sample (every class cut to the smallest, without replacement), so that all the
 * majority samples are used by some member while each member sees a balanced problem. Probabilities are averaged.
 *
 * @param hyper - Members (default 10), the member estimator (default {@link adaBoostClassifier} of 50 rounds) and
 *   `balance`.
 * @returns The estimator.
 * @concept ensembles
 * @index estimator Ensemble of AdaBoost members on balanced random samples.
 * @example
 * ```ts
 * import { easyEnsembleClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i < 34 ? i / 34 : 0.8 + (i - 34) / 20])
 * const y = x.map((_, i) => (i < 34 ? 0 : 1))
 * const fitted = easyEnsembleClassifier({ nEstimators: 3 }).fit({ x, y }, { key: key(0) })
 * toFlat(fitted.predict([[0.1], [1.05]])) // => [0, 1]
 * ```
 */
export function easyEnsembleClassifier(hyper: Partial<BaggingHyper> = {}): BaggingEstimator {
  return baggingEstimator('easy-ensemble-classifier', true, {
    estimator: hyper.estimator ?? adaBoostClassifier(),
    nEstimators: hyper.nEstimators ?? 10,
    maxSamples: 1,
    maxFeatures: 1,
    bootstrap: false,
    balance: hyper.balance ?? {},
  })
}

/**
 * RUSBoost (Seiffert et al. 2010; imblearn RUSBoostClassifier): AdaBoost in which each round fits its weak learner on a
 * random under-sample of the classes (every class cut to the smallest), carrying the rows' boosting weights, while the
 * weighted error and the reweighting use all the rows.
 *
 * @param hyper - As {@link adaBoostClassifier} (a stump, 50 rounds, SAMME) plus `balance` (default `'auto'`, no
 *   replacement).
 * @returns The estimator.
 * @concept boosting
 * @index estimator AdaBoost whose rounds each see a random under-sample.
 * @example
 * ```ts
 * import { rusBoostClassifier } from 'omai-engine/ensembles'
 * import { key } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [i < 34 ? i / 34 : 0.8 + (i - 34) / 20])
 * const y = x.map((_, i) => (i < 34 ? 0 : 1))
 * const fitted = rusBoostClassifier({ nEstimators: 10 }).fit({ x, y }, { key: key(0) })
 * toFlat(fitted.predict([[0.1], [1.05]])) // => [0, 1]
 * ```
 */
export function rusBoostClassifier(hyper: Partial<AdaBoostHyper> = {}): AdaBoostEstimator {
  return adaBoostEstimator('rus-boost-classifier', true, {
    estimator: hyper.estimator ?? decisionTreeClassifier({ maxDepth: 1, record: 0 }),
    nEstimators: hyper.nEstimators ?? 50,
    learningRate: hyper.learningRate ?? 1,
    algorithm: hyper.algorithm ?? 'SAMME',
    loss: hyper.loss ?? 'linear',
    balance: hyper.balance ?? {},
  })
}
