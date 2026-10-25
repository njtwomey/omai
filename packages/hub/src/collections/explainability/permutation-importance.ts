/**
 * Hub entry for permutation importance, the drop in a model's held-out score when each feature is shuffled, on tabular
 * classification and regression data.
 */

import { trainTestSplit } from 'whitebox-engine/data'
import { featureImportance } from 'whitebox-engine/methods/explain'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** Permutation importance: the drop in a held-out score when a feature is shuffled. */
export const permutationImportance: HubEntry = defineEntry({
  id: 'explainability/permutation-importance',
  title: 'Permutation importance',
  summary: 'How much a model’s held-out score drops when each feature is shuffled, repeat by repeat.',
  collection: 'explainability',
  tags: ['feature importance', 'permutation importance', 'model-agnostic', 'inspection'],
  datasets: ['classificationTabular', 'regressionTabular'],
  dataset: 'breastCancer',
  params: dict({
    model: modelControl(['random forest', 'gradient boosting', 'k-nearest neighbours'], 'random forest'),
    measure: choice(['held out', 'training', 'impurity'], { default: 'held out', label: 'measured on' }),
    repeats: int(1, 30, { default: 8, label: 'repeats' }),
  }),
  run: ({ data, params, key }) => {
    const { train, test } = trainTestSplit(data as NonNullable<typeof data>, { testSize: 0.3, key })
    const { fitted, learn } = fitModel(train, params['model'] as ExplainedModel, key)
    const measure = params['measure']
    if (measure === 'impurity') return featureImportance(fitted, learn, { method: 'impurity', on: 'training' })
    const on = measure === 'held out' ? learnDataOf(test) : learn
    return featureImportance(fitted, on, { nRepeats: Number(params['repeats']), on: String(measure), key })
  },
  views: [{ record: 'importances', title: 'Importances, every repeat a dot' }],
  methods: [{ file: 'methods/explain/inspection.ts', name: 'featureImportance' }],
  references: ['Breiman (2001), Random forests.'],
})
