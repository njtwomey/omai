import { randomForestClassifier } from 'whitebox-engine/methods/ensemble'
import { dict, int } from 'whitebox-engine/space'
import type { HubEntry } from '../../entry/entry.ts'
import { tuningEntry } from './_tuning.ts'

/** Tuning a random forest's depth and leaf size by cross-validation. */
export const randomForestSearch: HubEntry = tuningEntry({
  name: 'random-forest',
  title: 'Tuning a random forest',
  summary: 'Tree-structured Parzen estimators search a forest’s maximum depth and minimum leaf size.',
  tags: ['ensembles', 'trees', 'TPE'],
  datasets: ['classification2d', 'classificationTabular'],
  estimator: randomForestClassifier({ nEstimators: 20 }),
  space: dict({
    maxDepth: int(1, 12, { label: 'maximum depth' }),
    minSamplesLeaf: int(1, 30, { log: true, label: 'minimum leaf size' }),
  }),
  sampler: 'tpe',
  trials: 25,
  cv: 3,
  views: [
    { record: 'hyperparameterSurface', title: 'Cross-validated accuracy over depth and leaf size' },
    { record: 'searchTrials', title: 'Trials', area: 'side' },
    { record: 'learningCurve', title: 'Each trial and the best so far', area: 'below' },
  ],
  estimatorRef: { file: 'methods/ensemble/forest.ts', name: 'randomForestClassifier' },
})
