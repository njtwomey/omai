// Semi-supervised hub entries (private to this collection): the method's controls plus how many labels each class
// keeps, and a run that hides the rest of the dataset's labels (keyed) before fitting.
import type { Dataset } from 'mdml-engine/data'
import type { Estimator } from 'mdml-engine/learn'
import { hideLabels } from 'mdml-engine/methods/semi-supervised'
import { foldIn } from 'mdml-engine/random'
import { dict, int } from 'mdml-engine/space'
import type { HubEntry } from '../../../entry/entry.ts'
import { type FitEstimatorOptions, fitEstimator, learnDataOf } from '../../../entry/run.ts'

/**
 * An entry's `params` and `run` for a semi-supervised estimator: its controls with `labelled` (labels kept per
 * class), and a run that keeps that many labels per class and marks the others unlabelled (-1).
 *
 * @param estimator - The estimator with its demo hyperparameters.
 * @param options - As `fitEstimator`.
 * @returns The `params` and `run` of a hub entry.
 */
export function withHiddenLabels(
  estimator: Estimator,
  options: FitEstimatorOptions = {},
): Pick<HubEntry, 'params' | 'run'> {
  const inner = fitEstimator(estimator, options)
  return {
    params: dict({ labelled: int(1, 30, { default: 3, label: 'labels kept per class' }), ...inner.params.spaces }),
    run: (input) => {
      const { labelled, ...params } = input.params
      const learn = learnDataOf(input.data)
      const y = hideLabels(learn.y ?? [], {
        perClass: typeof labelled === 'number' ? labelled : 3,
        key: foldIn(input.key, 1),
      })
      return inner.run({ ...input, params, data: { ...(input.data as Dataset), y } as unknown as Dataset })
    },
  }
}
