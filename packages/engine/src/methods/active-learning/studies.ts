/**
 * Pool-based active-learning studies (Settles, 2009; as in modAL): start from a few labelled rows per class, let a
 * strategy choose which pool rows an oracle labels next, refit after each query, and record accuracy against the number
 * of labels, compared with random sampling.
 */

// TODO: stream-based selective sampling studies (`streamActiveLearning` of the core) and several strategies averaged
// over seeds with spreads.
import { MdmlError } from '../../lib/base/index.ts'
import {
  type ClusteringSteps,
  clusteringSteps,
  type Explained,
  type Exposition,
  learningCurve,
} from '../../lib/exposition/index.ts'
import {
  activeLearning,
  classNamesOf,
  decisionSurfaceOf,
  type Estimator,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  requireVerb,
  takeData,
  targetVector,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { queryStrategy, type StrategyName } from './strategies.ts'

/** Options of {@link activeLearningStudy}. */
export interface ActiveLearningStudyOptions {
  /** The classifier (default logistic regression); it must predict probabilities. */
  readonly estimator?: Estimator
  readonly strategy: StrategyName
  /** Labelled rows per class at the start (default 2). */
  readonly initialPerClass?: number
  /** Labels bought (default 30, at most the pool). */
  readonly budget?: number
  /** Rows queried per round (default 1). */
  readonly batchSize?: number
  /** Committee members for committee strategies (default 5). */
  readonly committeeSize?: number
  /** Pool rows expected error reduction considers per round (default 20). */
  readonly candidates?: number
  /** Also run random sampling on the same start, for comparison (default true). */
  readonly baseline?: boolean
  readonly key?: Key
}

/** The result of {@link activeLearningStudy}. */
export interface ActiveLearningStudy {
  /** The model fitted on all labels bought. */
  readonly fitted: Fitted
  /** Dataset rows labelled at the start. */
  readonly initial: readonly number[]
  /** Dataset rows queried, in order. */
  readonly queried: readonly number[]
  /** Labelled rows after each round, the start first. */
  readonly labelled: readonly number[]
  /** Accuracy on every row after each round, the start first. */
  readonly accuracy: readonly number[]
  /** Random sampling's accuracy at the same budgets (when run). */
  readonly baseline?: readonly number[]
}

/** A keyed start with `perClass` labelled rows of each class (the rest is the pool). */
function initialRows(y: readonly number[], perClass: number, key: Key): number[] {
  const classes = [...new Set(y)].sort((a, b) => a - b)
  const order = Array.from(toFlat(permutation(key, y.length)))
  return classes.flatMap((c) => order.filter((i) => y[i] === c).slice(0, perClass)).sort((a, b) => a - b)
}

/**
 * A pool-based active-learning study on a labelled dataset (Settles 2009; modAL): start from a few labelled rows per
 * class, treat the rest as the unlabelled pool whose labels an oracle reveals on request, and let a query strategy
 * (uncertainty, margin, entropy, query by committee, expected error reduction, ranked batch, or random) choose which
 * rows to buy, round by round, through the core's traced `activeLearning` loop. Accuracy is measured on every row of
 * the dataset. Exposition: a `learningCurve` of accuracy against labels bought (with random sampling's curve beside it
 * when `baseline`), a `clusteringSteps` record of the labelled rows after each round (their labels; -1 still in the
 * pool) with the accuracy as its objective, and the final decision surface for two features.
 *
 * @param data - Labelled classification data.
 * @param options - The estimator, strategy, start, budget, batch size, committee, candidates, baseline and key.
 * @returns The final model, the queried rows and the curves, with the exposition records.
 * @throws MdmlError `value` when the data has no labels or the start leaves no pool.
 */
export function activeLearningStudy(
  data: LearnData,
  options: ActiveLearningStudyOptions,
): Explained<ActiveLearningStudy> {
  if (data.y === undefined) throw new MdmlError('value', 'activeLearningStudy: the data needs labels')
  const m = matrixView(data.x, 'activeLearningStudy')
  const y = Array.from(targetVector(data.y, 'activeLearningStudy', m.n))
  const key = options.key ?? makeKey(0)
  const estimator = options.estimator ?? logisticRegression()
  const batch = options.batchSize ?? 1
  const initial = initialRows(y, options.initialPerClass ?? 2, foldIn(key, 0))
  const start = new Set(initial)
  const poolRows = Array.from({ length: m.n }, (_, i) => i).filter((i) => !start.has(i))
  if (poolRows.length === 0) throw new MdmlError('value', 'activeLearningStudy: no pool left after the start')
  const budget = Math.min(options.budget ?? 30, poolRows.length)
  const labeled = takeData({ x: data.x, y }, initial)
  const pool = takeData({ x: data.x }, poolRows).x
  const accuracyOf = (f: Fitted): number => {
    const pred = toFlat(requireVerb(f, 'predict', 'activeLearningStudy')(data.x))
    return pred.filter((p, i) => p === y[i]).length / m.n
  }
  const run = (strategy: StrategyName, k: Key) =>
    activeLearning(estimator, {
      labeled,
      pool,
      oracle: (idx) => idx.map((i) => y[poolRows[i] ?? 0] ?? 0),
      strategy: queryStrategy(strategy, options.candidates === undefined ? {} : { candidates: options.candidates }),
      batchSize: batch,
      budget,
      ...(options.committeeSize === undefined ? {} : { committeeSize: options.committeeSize }),
      evaluation: { x: data.x, y },
      scoring: 'accuracy',
      key: k,
    })
  const startScore = accuracyOf(estimator.fit(labeled, { key: foldIn(key, 1) }))
  const main = run(options.strategy, foldIn(key, 2))
  const baseline =
    (options.baseline ?? true) && options.strategy !== 'random' ? run('random', foldIn(key, 2)) : undefined
  const queried = main.queried.map((i) => poolRows[i] ?? 0)
  const rounds = main.scores.length
  const labelled = [
    initial.length,
    ...Array.from({ length: rounds }, (_, r) => initial.length + Math.min(budget, (r + 1) * batch)),
  ]
  const accuracy = [startScore, ...main.scores]
  const baselineCurve = baseline === undefined ? undefined : [startScore, ...baseline.scores]
  // The labelled rows after each round: their labels, -1 while still in the pool.
  const assignments = labelled.map((count) => {
    const row = y.map(() => -1)
    for (const i of initial) row[i] = y[i] ?? 0
    for (const i of queried.slice(0, count - initial.length)) row[i] = y[i] ?? 0
    return row
  })
  const steps: ClusteringSteps = clusteringSteps({
    method: `active-learning/${options.strategy}`,
    assignments,
    objective: accuracy,
    // The rows labelled at the start, and the ones each round bought.
    seeds: [...initial],
    added: labelled.map((count, r) =>
      r === 0 ? [] : queried.slice((labelled[r - 1] ?? initial.length) - initial.length, count - initial.length),
    ),
    addedName: 'queried',
  })
  const curve = learningCurve({
    sizes: labelled,
    train: accuracy,
    // Without a random baseline the strategy's curve stands alone.
    validation: baselineCurve ?? [],
    resource: 'labels bought',
    labels: { train: options.strategy, validation: baselineCurve === undefined ? options.strategy : 'random' },
  })
  const records: Exposition[] = [curve, steps]
  if (m.d === 2)
    records.push(
      decisionSurfaceOf(main.fitted, data.x, {
        featureNames: featureNamesOf(data, m.d),
        classNames: classNamesOf(data, main.fitted.classes ?? []),
      }),
    )
  return {
    fitted: main.fitted,
    initial,
    queried,
    labelled,
    accuracy,
    ...(baselineCurve === undefined ? {} : { baseline: baselineCurve }),
    exposition: records,
  }
}
