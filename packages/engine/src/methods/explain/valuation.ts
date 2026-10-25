/**
 * Which training rows matter, for a fitted pipeline: a mislabelled-row study that flips some labels and asks whether
 * k-NN Shapley, data Banzhaf and Beta Shapley values find them, inspecting the lowest-valued rows first, as `curves`.
 */

import { MdmlError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import { curveSet, type Explained } from '../../lib/exposition/index.ts'
import { type Estimator, type LearnData, matrixView, targetVector } from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../lib/random/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'

/** The valuations a mislabelled-row study compares. */
export type ValuationMethod = 'knn shapley' | 'data banzhaf' | 'beta shapley'

/** Options of {@link mislabelledRowStudy}. */
export interface MislabelledStudyOptions {
  /** The share of training labels flipped (default 0.1). */
  readonly flipFraction?: number
  /** The valuations compared (default all three). */
  readonly methods?: readonly ValuationMethod[]
  /** Training rows used (default 100; the refit-based values cost one fit per sampled subset). */
  readonly trainRows?: number
  /** Rows held out for the utility (default 60). */
  readonly testRows?: number
  /** Subsets sampled by data Banzhaf and permutations by Beta Shapley (default 60 and 15). */
  readonly nSamples?: number
  readonly nPermutations?: number
  /** Neighbours of k-NN Shapley (default 5). */
  readonly k?: number
  readonly key?: Key
}

/** The values each method gave and which rows were flipped. */
export interface MislabelledStudyResult {
  readonly flipped: readonly number[]
  readonly values: Readonly<Record<string, readonly number[]>>
}

/**
 * A mislabelled-row study (Ghorbani and Zou 2019; Kwon and Zou 2022; Wang and Jia 2023): flip the labels of a share
 * of the training rows of a classification dataset, value every training row by each method against held-out rows,
 * and inspect rows from the lowest value up; the `curves` record shows the share of flipped rows found against the
 * share inspected, beside random inspection.
 *
 * @param estimator - The classifier refit by the subset utilities.
 * @param data - A classification dataset.
 * @param options - flipFraction, methods, trainRows, testRows, nSamples, nPermutations, k and key.
 * @returns The flipped rows and the values with a `curves` record.
 */
export function mislabelledRowStudy(
  estimator: Estimator,
  data: LearnData,
  options: MislabelledStudyOptions = {},
): Explained<MislabelledStudyResult> {
  const x = matrixView(data.x, 'mislabelledRowStudy')
  if (data.y === undefined) throw new MdmlError('value', 'mislabelledRowStudy: needs labels')
  const key = options.key ?? makeKey(0)
  const order = Array.from(permutation(foldIn(key, 0), x.n).data as Int32Array)
  const nTrain = Math.min(options.trainRows ?? 100, Math.floor(x.n * 0.7))
  const nTest = Math.min(options.testRows ?? 60, x.n - nTrain)
  const trainIdx = order.slice(0, nTrain)
  const testIdx = order.slice(nTrain, nTrain + nTest)
  const yAll = Array.from(targetVector(data.y, 'mislabelledRowStudy', x.n))
  const classes = Array.from(new Set(yAll)).sort((a, b) => a - b)
  if (classes.length < 2) throw new MdmlError('value', 'mislabelledRowStudy: needs two or more classes')
  const nFlip = Math.max(1, Math.round((options.flipFraction ?? 0.1) * nTrain))
  const flipped = Array.from(permutation(foldIn(key, 1), nTrain).data as Int32Array).slice(0, nFlip)
  const yTrain = trainIdx.map((i) => yAll[i] ?? 0)
  for (const f of flipped) {
    const c = classes.indexOf(yTrain[f] ?? 0)
    yTrain[f] = classes[(c + 1) % classes.length] ?? 0
  }
  const rowsOf = (idx: readonly number[]): number[][] =>
    idx.map((i) => Array.from(x.data.subarray(i * x.d, (i + 1) * x.d)))
  const train = { x: rowsOf(trainIdx), y: yTrain }
  const test = { x: rowsOf(testIdx), y: testIdx.map((i) => yAll[i] ?? 0) }
  const methods = options.methods ?? ['knn shapley', 'data banzhaf', 'beta shapley']
  const utility = explain.subsetUtility(estimator, train, test, { emptyUtility: 1 / classes.length })
  const values: Record<string, number[]> = {}
  for (const m of methods) {
    if (m === 'knn shapley')
      values[m] = explain.knnShapley(
        fromStorage(Float64Array.from(train.x.flat()), [nTrain, x.d], 'float64'),
        train.y,
        fromStorage(Float64Array.from(test.x.flat()), [nTest, x.d], 'float64'),
        test.y,
        options.k ?? 5,
      )
    else if (m === 'data banzhaf')
      values[m] = [
        ...explain.dataBanzhaf(utility, nTrain, { nSamples: options.nSamples ?? 60, key: foldIn(key, 2) }).values,
      ]
    else
      values[m] = [
        ...explain.betaShapley(utility, nTrain, { nPermutations: options.nPermutations ?? 15, key: foldIn(key, 3) })
          .values,
      ]
  }
  const fractions = Array.from({ length: 11 }, (_, i) => i / 10)
  const isFlipped = new Set(flipped)
  const found = (v: readonly number[]): number[] => {
    const rank = v.map((s, i) => ({ s, i })).sort((a, b) => a.s - b.s || a.i - b.i)
    return fractions.map((f) => {
      const seen = rank.slice(0, Math.round(f * nTrain))
      return seen.filter((e) => isFlipped.has(e.i)).length / nFlip
    })
  }
  return {
    flipped,
    values,
    exposition: [
      curveSet({
        x: fractions,
        series: [
          ...methods.map((m) => ({ name: m, y: found(values[m] ?? []) })),
          { name: 'random inspection', y: fractions },
        ],
        xLabel: 'share of training rows inspected, lowest value first',
        yLabel: 'share of flipped labels found',
      }),
    ],
  }
}
