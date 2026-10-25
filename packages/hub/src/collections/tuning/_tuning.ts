// The hub entry of a hyperparameter search (private to collections/tuning): the search's own settings as controls,
// a run that reports the trials so far, and the views of the trials, the surface and the best-so-far curve.
import { MdmlError } from 'mdml-engine/base'
import type { Estimator } from 'mdml-engine/learn'
import { type PrunerName, type SamplerName, tuneEstimator } from 'mdml-engine/methods/tuning'
import { choice, dict, int, type Space } from 'mdml-engine/space'
import { defineEntry, type HubEntry, type MethodRef, type ViewSlot } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** What a tuning entry declares. */
export interface TuningEntrySpec {
  readonly name: string
  readonly title: string
  readonly summary: string
  readonly tags: readonly string[]
  readonly datasets: readonly string[]
  readonly dataset?: string
  /** The estimator with its fixed hyperparameters; the search sets the ones in `space`. */
  readonly estimator: Estimator
  readonly space: Space
  readonly sampler: SamplerName
  readonly trials: number
  readonly cv: number
  readonly views: readonly ViewSlot[]
  readonly estimatorRef: MethodRef
}

/**
 * The hub entry of a hyperparameter search over an estimator.
 *
 * @param spec - The estimator, its search space, the search's demo settings and the views.
 * @returns The entry.
 */
export function tuningEntry(spec: TuningEntrySpec): HubEntry {
  return defineEntry({
    id: `tuning/${spec.name}`,
    title: spec.title,
    summary: spec.summary,
    collection: 'tuning',
    tags: ['hyperparameter optimisation', 'cross-validation', ...spec.tags],
    datasets: spec.datasets,
    ...(spec.dataset === undefined ? {} : { dataset: spec.dataset }),
    params: dict({
      sampler: choice(['grid', 'random', 'tpe', 'gp'], { default: spec.sampler, label: 'sampler' }),
      trials: int(4, 60, { default: spec.trials, label: 'trials' }),
      cv: int(2, 10, { default: spec.cv, label: 'folds' }),
      pruner: choice(['none', 'median', 'halving'], { default: 'none', label: 'pruner' }),
    }),
    run: ({ data, params, key, progress }) => {
      if (data === undefined) throw new MdmlError('value', `tuning/${spec.name}: needs a dataset`)
      const trials = Number(params['trials'])
      return tuneEstimator({
        estimator: spec.estimator,
        data: learnDataOf(data),
        space: spec.space,
        sampler: params['sampler'] as SamplerName,
        trials,
        cv: Number(params['cv']),
        pruner: params['pruner'] as PrunerName,
        key,
        ...(progress === undefined
          ? {}
          : {
              onTrial: (t) =>
                progress({ fraction: t.fraction, label: `trial ${t.trials} of ${trials}`, exposition: t.exposition() }),
            }),
      })
    },
    views: spec.views,
    methods: [{ file: 'methods/tuning/search.ts', name: 'tuneEstimator' }, spec.estimatorRef],
  })
}
