// The run of a neural-network hub entry (private to collections/neural): fit an estimator whose hyperparameters take an
// `onProgress` callback, passing the page's progress sink so the decision surface and the loss curve draw live while
// the network trains.
import type { Estimator } from 'whitebox-engine/learn'
import type { HubEntry } from '../../entry/entry.ts'
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
