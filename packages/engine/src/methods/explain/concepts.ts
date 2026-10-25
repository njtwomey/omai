/**
 * Concept explanations of a fitted network (TCAV on the LeNet of `methods/neural`): concepts given as sets of example
 * rows (images of a digit, say), tested at the network's hidden layer against random rows, as a `concepts` record.
 */

import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import { conceptScores, type Explained } from '../../lib/exposition/index.ts'
import { classNamesOf, type Fitted, type LearnData, matrixView } from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { compose, networkOf } from './_networks.ts'

/** A concept: a name and the rows of the data that show it. */
export interface ConceptSet {
  readonly name: string
  readonly rows: readonly number[]
}

/** Options of {@link tcavStudy}. */
export interface TcavStudyOptions {
  /** The class explained (an index into the network's classes; default the class most predicted for `inputs`). */
  readonly classIndex?: number
  /** Rows of the class's examples (default every row the network assigns to the class, at most 60). */
  readonly inputs?: readonly number[]
  /** Random sets drawn (default 8) and their size (default 30). */
  readonly nRandom?: number
  readonly randomSize?: number
  readonly key?: Key
}

/**
 * TCAV (Kim et al. 2018) for a fitted LeNet: each concept's activation vector at the hidden fully connected layer
 * (against several random sets of rows), the share of the class's examples whose logit rises along it, and the t test
 * of those scores against random directions' scores.
 *
 * @param fitted - A fitted LeNet.
 * @param data - The images.
 * @param concepts - The concepts, as rows of the data.
 * @param options - classIndex, inputs, nRandom, randomSize and key.
 * @returns The TCAV results per concept with a `concepts` record.
 */
export function tcavStudy(
  fitted: Fitted,
  data: LearnData,
  concepts: readonly ConceptSet[],
  options: TcavStudyOptions = {},
): Explained<{ readonly results: readonly explain.TcavResult[] }> {
  const x = matrixView(data.x, 'tcavStudy')
  const net = networkOf(fitted)
  const head = compose(net.layers.slice(0, net.hiddenLayer + 1))
  const tail = compose(net.layers.slice(net.hiddenLayer + 1))
  const rowsOf = (idx: readonly number[]): Tensor =>
    fromStorage(
      Float64Array.from(idx.flatMap((i) => Array.from(x.data.subarray(i * x.d, (i + 1) * x.d)))),
      [idx.length, x.d],
      'float64',
    )
  const acts = (idx: readonly number[]): number[][] => {
    const h = head(rowsOf(idx)) as Tensor
    const w = h.shape[1] ?? 1
    const v = toFlat(h)
    return idx.map((_, i) => v.slice(i * w, (i + 1) * w))
  }
  const all = Array.from({ length: x.n }, (_, i) => i)
  const logits = compose(net.layers)(rowsOf(all)) as Tensor
  const k = logits.shape[1] ?? 1
  const lv = toFlat(logits)
  const predicted = all.map((i) => argmaxOf(lv.slice(i * k, (i + 1) * k)))
  const target =
    options.classIndex ?? argmaxOf(Array.from({ length: k }, (_, c) => predicted.filter((p) => p === c).length))
  const inputs = options.inputs ?? all.filter((i) => predicted[i] === target).slice(0, 60)
  if (inputs.length === 0) throw new WhiteboxError('value', 'tcavStudy: no examples of the class')
  const key = options.key ?? makeKey(0)
  const nRandom = Math.max(2, options.nRandom ?? 8)
  const size = Math.min(x.n, options.randomSize ?? 30)
  const randoms = Array.from({ length: nRandom }, (_, r) =>
    acts(toFlat(permutation(foldIn(key, r), x.n)).slice(0, size)),
  )
  const classActs = acts(inputs)
  const results = concepts.map((c) => explain.tcav(tail, classActs, acts(c.rows), randoms, { target }))
  const name = classNamesOf(data, net.classes)[target] ?? String(target)
  return {
    results,
    exposition: [
      conceptScores({
        method: 'TCAV',
        target: name,
        layer: 'hidden layer',
        concepts: concepts.map((c, i) => {
          const r = results[i] as explain.TcavResult
          return {
            name: c.name,
            scores: r.scores,
            randomScores: r.randomScores,
            mean: r.mean,
            pValue: r.pValue,
            accuracy: r.accuracies.reduce((a, b) => a + b, 0) / Math.max(1, r.accuracies.length),
          }
        }),
      }),
    ],
  }
}
