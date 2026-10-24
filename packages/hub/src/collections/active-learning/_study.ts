// Active-learning hub entries (private to this collection): a study's controls (the strategies of one family, the
// start, the budget and the batch size) and a run on the chosen dataset.

import type { Estimator } from 'whitebox-engine/learn'
import { activeLearningStudy, type StrategyName } from 'whitebox-engine/methods/active-learning'
import { choice, dict, int } from 'whitebox-engine/space'
import type { HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Options of {@link studyEntry}. */
export interface StudyEntryOptions {
  /** The family's strategies; the first is the default. */
  readonly strategies: readonly StrategyName[]
  /** The default budget (labels bought). */
  readonly budget?: number
  /** Pool rows expected error reduction considers per round. */
  readonly candidates?: number
  /** The classifier (default the study's, logistic regression). */
  readonly estimator?: Estimator
}

/**
 * The `params` and `run` of an active-learning entry.
 *
 * @param options - The strategies, default budget and candidates.
 * @returns The `params` and `run` of a hub entry.
 */
export function studyEntry(options: StudyEntryOptions): Pick<HubEntry, 'params' | 'run'> {
  const budget = options.budget ?? 30
  return {
    params: dict({
      strategy: choice([...options.strategies], { default: options.strategies[0] ?? 'random' }),
      initialPerClass: int(1, 10, { default: 2, label: 'labels per class at the start' }),
      budget: int(1, 100, { default: budget, label: 'labels bought' }),
      batchSize: int(1, 10, { default: 1, label: 'labels per round' }),
    }),
    run: ({ data, params, key }) =>
      activeLearningStudy(learnDataOf(data), {
        strategy: params['strategy'] as StrategyName,
        initialPerClass: params['initialPerClass'] as number,
        budget: params['budget'] as number,
        batchSize: params['batchSize'] as number,
        ...(options.candidates === undefined ? {} : { candidates: options.candidates }),
        ...(options.estimator === undefined ? {} : { estimator: options.estimator }),
        key,
      }),
  }
}
