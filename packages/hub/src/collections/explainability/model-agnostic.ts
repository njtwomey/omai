/**
 * The model-agnostic attribution entry: Kernel SHAP, exact and sampled Shapley values and LIME for any fitted model.
 */

import { WhiteboxError } from 'whitebox-engine/base'
import { type AttributionMethod, explainPrediction } from 'whitebox-engine/methods/explain'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { type ExplainedModel, fitModel, modelControl, rowControl } from './_explain.ts'

/** Model-agnostic attributions: Kernel SHAP, exact and sampled Shapley values and LIME for any model. */
export const modelAgnostic: HubEntry = defineEntry({
  id: 'explainability/model-agnostic',
  title: 'Kernel, partition and conditional SHAP, sampling and LIME',
  summary:
    'Explain any model’s prediction by querying it: Shapley values exactly, by regression, by sampling, over a feature hierarchy or conditionally, or LIME.',
  collection: 'explainability',
  tags: [
    'SHAP',
    'Kernel SHAP',
    'LIME',
    'Shapley values',
    'Owen values',
    'conditional SHAP',
    'feature attribution',
    'model-agnostic',
  ],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  maxRows: 600,
  params: dict({
    model: modelControl(['k-nearest neighbours', 'random forest', 'gradient boosting'], 'k-nearest neighbours'),
    method: choice(['kernel', 'exact', 'sampling', 'lime', 'partition', 'conditional'], {
      default: 'sampling',
      label: 'explainer',
    }),
    row: rowControl,
    background: int(5, 100, { default: 30, label: 'background rows' }),
    permutations: int(10, 1000, { default: 300, log: true, label: 'permutations (sampling)' }),
  }),
  run: ({ data, params, key }) => {
    const { fitted, learn, n } = fitModel(data, params['model'] as ExplainedModel, key)
    if (n === 0) throw new WhiteboxError('value', 'model-agnostic: no rows')
    const method = params['method'] as AttributionMethod
    return explainPrediction(fitted, learn, Math.min(Number(params['row']), n - 1), {
      method,
      background: Number(params['background']),
      nPermutations: Number(params['permutations']),
      key,
    })
  },
  views: [
    { record: 'featureAttributions', view: 'WaterfallView', title: 'Why this prediction' },
    { record: 'featureAttributions', view: 'ForcePlotView', title: 'The same as forces' },
    { record: 'curves', title: 'The sampled estimate settling', area: 'side' },
  ],
  methods: [{ file: 'methods/explain/attributions.ts', name: 'explainPrediction' }],
  references: [
    'Lundberg and Lee (2017), A unified approach to interpreting model predictions.',
    'Strumbelj and Kononenko (2014), Explaining prediction models and individual predictions with feature contributions.',
    'Ribeiro, Singh and Guestrin (2016), “Why should I trust you?”: explaining the predictions of any classifier.',
    'Owen (1977), Values of games with a priori unions.',
    'Aas, Jullum and Løland (2021), Explaining individual predictions when features are dependent.',
  ],
})
