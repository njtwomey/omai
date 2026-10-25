/**
 * The trust score entry: how far a classifier's held-out predictions agree with their neighbourhoods, against the
 * model's own confidence.
 */

import { trainTestSplit } from 'whitebox-engine/data'
import { trustScoreStudy } from 'whitebox-engine/methods/explain'
import { dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** Trust scores. */
export const trustScores: HubEntry = defineEntry({
  id: 'explainability/trust-scores',
  title: 'Trust scores',
  summary:
    'Whether to trust each prediction: closer to its class’s dense core than to any other’s, judged by held-out accuracy.',
  collection: 'explainability',
  tags: ['trust scores', 'example-based', 'confidence', 'nearest neighbours'],
  datasets: ['classificationTabular'],
  dataset: 'breastCancer',
  params: dict({
    model: modelControl(['random forest', 'gradient boosting', 'k-nearest neighbours'], 'random forest'),
    alpha: real(0, 0.5, { default: 0.1, label: 'density filter (alpha)' }),
    kFilter: int(1, 30, { default: 10, label: 'neighbours for density' }),
  }),
  run: ({ data, params, key }) => {
    const { train, test } = trainTestSplit(data as NonNullable<typeof data>, { testSize: 0.4, key })
    const { fitted, learn } = fitModel(train, params['model'] as ExplainedModel, key)
    return trustScoreStudy(fitted, learn, learnDataOf(test), {
      alpha: Number(params['alpha']),
      kFilter: Number(params['kFilter']),
    })
  },
  views: [{ record: 'curves', title: 'Accuracy as fewer rows are trusted' }],
  methods: [{ file: 'methods/explain/counterfactuals.ts', name: 'trustScoreStudy' }],
  references: ['Jiang, Kim, Guan and Gupta (2018), To trust or not to trust a classifier.'],
})
