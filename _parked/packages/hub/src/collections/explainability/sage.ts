/**
 * The SAGE entry: global importance as each feature's Shapley share of the loss a fitted model removes.
 */
import { trainTestSplit } from 'mdml-engine/data'
import { sageImportance } from 'mdml-engine/methods/explain'
import { dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** SAGE: Shapley additive global importance. */
export const sage: HubEntry = defineEntry({
  id: 'explainability/sage',
  title: 'SAGE global importance',
  summary:
    'How much of the held-out loss each feature removes, shared out fairly among the features by Shapley values.',
  collection: 'explainability',
  tags: ['SAGE', 'Shapley values', 'global importance', 'feature importance', 'model-agnostic'],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  params: dict({
    model: modelControl(['gradient boosting', 'random forest', 'k-nearest neighbours'], 'gradient boosting'),
    permutations: int(20, 1000, { default: 150, log: true, label: 'permutations' }),
    background: int(5, 100, { default: 20, label: 'background rows' }),
  }),
  run: ({ data, params, key }) => {
    const { train, test } = trainTestSplit(data as NonNullable<typeof data>, { testSize: 0.3, key })
    const { fitted } = fitModel(train, params['model'] as ExplainedModel, key)
    return sageImportance(fitted, learnDataOf(test), {
      nPermutations: Number(params['permutations']),
      background: Number(params['background']),
      maxRows: 150,
      key,
    })
  },
  views: [{ record: 'importances', title: 'Loss removed by each feature, with its standard error' }],
  methods: [{ file: 'methods/explain/inspection.ts', name: 'sageImportance' }],
  references: [
    'Covert, Lundberg and Lee (2020), Understanding global feature contributions with additive importance measures.',
  ],
})
