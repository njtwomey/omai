/**
 * The feature importance entry: permutation (marginal or conditional), drop-column and impurity importances.
 */

import { trainTestSplit } from 'mdml-engine/data'
import { dropColumnImportances, featureImportance } from 'mdml-engine/methods/explain'
import { choice, dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'
import { type ExplainedModel, fitModel, modelControl, modelOf } from './_explain.ts'

/** Permutation importance: the drop in a held-out score when a feature is shuffled. */
export const permutationImportance: HubEntry = defineEntry({
  id: 'explainability/permutation-importance',
  title: 'Permutation importance',
  summary:
    'How much a model’s score drops when each feature is shuffled (marginally or within strata of correlated features) or dropped and the model refit.',
  collection: 'explainability',
  tags: ['feature importance', 'permutation importance', 'model-agnostic', 'inspection'],
  datasets: ['classificationTabular', 'regressionTabular'],
  dataset: 'breastCancer',
  params: dict({
    model: modelControl(['random forest', 'gradient boosting', 'k-nearest neighbours'], 'random forest'),
    measure: choice(['held out', 'training', 'conditional', 'drop column', 'impurity'], {
      default: 'held out',
      label: 'measure',
    }),
    repeats: int(1, 30, { default: 8, label: 'repeats' }),
  }),
  run: ({ data, params, key }) => {
    const { train, test } = trainTestSplit(data as NonNullable<typeof data>, { testSize: 0.3, key })
    const { fitted, learn } = fitModel(train, params['model'] as ExplainedModel, key)
    const measure = params['measure']
    if (measure === 'drop column')
      return dropColumnImportances(modelOf(params['model'] as ExplainedModel, data?.kind === 'classification'), learn, {
        test: learnDataOf(test),
      })
    if (measure === 'conditional')
      return featureImportance(fitted, learnDataOf(test), {
        method: 'conditional',
        nRepeats: Number(params['repeats']),
        on: 'held out',
        key,
      })
    if (measure === 'impurity') return featureImportance(fitted, learn, { method: 'impurity', on: 'training' })
    const on = measure === 'held out' ? learnDataOf(test) : learn
    return featureImportance(fitted, on, { nRepeats: Number(params['repeats']), on: String(measure), key })
  },
  views: [{ record: 'importances', title: 'Importances, every repeat a dot' }],
  methods: [
    { file: 'methods/explain/inspection.ts', name: 'featureImportance' },
    { file: 'methods/explain/inspection.ts', name: 'dropColumnImportances' },
  ],
  references: [
    'Breiman (2001), Random forests.',
    'Strobl, Boulesteix, Kneib, Augustin and Zeileis (2008), Conditional variable importance for random forests.',
  ],
})
