/**
 * Hub entry for Tree SHAP, exact Shapley values of a gradient-boosted or random forest model, as one prediction's
 * waterfall and a beeswarm of every row, on tabular regression and classification data.
 */

import { MdmlError } from 'mdml-engine/base'
import { explainDataset, explainPrediction } from 'mdml-engine/methods/explain'
import { dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { type ExplainedModel, fitModel, modelControl, rowControl } from './_explain.ts'

/** Tree SHAP: exact Shapley values of a tree ensemble, one prediction and the whole dataset. */
export const treeShap: HubEntry = defineEntry({
  id: 'explainability/tree-shap',
  title: 'Tree SHAP',
  summary: 'Exact Shapley values of a forest or a booster: one prediction’s waterfall and every row’s beeswarm.',
  collection: 'explainability',
  tags: ['SHAP', 'Shapley values', 'feature attribution', 'trees', 'ensembles'],
  datasets: ['regressionTabular', 'classificationTabular'],
  params: dict({
    model: modelControl(['gradient boosting', 'random forest'], 'gradient boosting'),
    row: rowControl,
    rows: int(20, 400, { default: 200, label: 'rows summarised' }),
  }),
  run: ({ data, params, key }) => {
    const { fitted, learn, n } = fitModel(data, params['model'] as ExplainedModel, key)
    if (n === 0) throw new MdmlError('value', 'tree-shap: no rows')
    const row = Math.min(Number(params['row']), n - 1)
    const one = explainPrediction(fitted, learn, row, { method: 'tree' })
    const all = explainDataset(fitted, learn, { method: 'tree', maxRows: Number(params['rows']), key })
    return { exposition: [...one.exposition, ...all.exposition] }
  },
  views: [
    { record: 'featureAttributions', view: 'WaterfallView', title: 'Why this prediction', index: 0 },
    { record: 'featureAttributions', view: 'ForcePlotView', title: 'As forces', index: 0 },
    { record: 'featureAttributions', view: 'BeeswarmView', title: 'Every row', index: 1, area: 'side' },
    { record: 'featureAttributions', view: 'AttributionBarView', title: 'Mean |SHAP|', index: 1, area: 'below' },
    {
      record: 'featureAttributions',
      view: 'DependenceScatterView',
      title: 'The most important feature',
      index: 1,
      area: 'below',
    },
  ],
  methods: [
    { file: 'methods/explain/attributions.ts', name: 'explainPrediction' },
    { file: 'methods/explain/attributions.ts', name: 'explainDataset' },
  ],
  references: ['Lundberg, Erion and Lee (2018), Consistent individualized feature attribution for tree ensembles.'],
})
