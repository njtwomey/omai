// The run of a neural-network hub entry (private to collections/neural): fit an estimator whose hyperparameters take an
// `onProgress` callback, passing the page's progress sink so the decision surface and the loss curve draw live while
// the network trains.
import type { Estimator } from 'whitebox-engine/learn'
import { type MlpActivation, mlpNetwork } from 'whitebox-engine/methods/neural/foundations'
import { describe } from 'whitebox-engine/nn'
import { key as makeKey } from 'whitebox-engine/random'
import type { HubEntry, ViewSlot } from '../../entry/entry.ts'
import { type FitEstimatorOptions, fitEstimator } from '../../entry/run.ts'

/**
 * An entry's `params` and `run` from a trainable estimator: as {@link fitEstimator}, with the run's progress reports
 * passed to the estimator's `onProgress`.
 *
 * @param estimator - The estimator with its demo hyperparameters.
 * @param options - As {@link fitEstimator}.
 * @returns The `params` and `run` of a hub entry.
 */
export function trainingRun(estimator: Estimator, options: FitEstimatorOptions = {}): Pick<HubEntry, 'params' | 'run'> {
  const plain = fitEstimator(estimator, options)
  return {
    params: plain.params,
    run: (input) =>
      input.progress === undefined
        ? plain.run(input)
        : fitEstimator(estimator.with({ onProgress: input.progress }), options).run(input),
  }
}

/**
 * A slot drawing an MLP's architecture (render's `StructureView`) beside a record, from the page's parameters and
 * rows: the input width from the rows, the output from the labels (one unit for two classes or a regression).
 *
 * @param title - The slot's title.
 * @returns The slot.
 */
export function mlpArchitecture(title = 'The network'): ViewSlot {
  return {
    record: 'learningCurve',
    view: 'StructureView',
    area: 'below',
    title,
    props: (_, context) => {
      const d = context.points?.[0]?.length ?? 2
      const classes = context.labels === undefined ? 1 : new Set(context.labels).size
      const sizes = String(context.params['hiddenLayerSizes'] ?? '16')
        .split(',')
        .map(Number)
        .filter((w) => Number.isInteger(w) && w > 0)
      const activation = (context.params['activation'] ?? 'relu') as MlpActivation
      const net = mlpNetwork([d, ...sizes, classes > 2 ? classes : 1], activation, makeKey(0))
      return { structure: describe(net, [1, d]) }
    },
  }
}
