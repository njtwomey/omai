/**
 * The explanation-quality entry: several explainers of one fitted model scored on faithfulness, robustness,
 * complexity and completeness, with their agreement and deletion and insertion curves.
 */

import { evaluateExplanations } from 'whitebox-engine/methods/explain'
import { dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** Evaluating explanations (Quantus's metrics). */
export const explanationQuality: HubEntry = defineEntry({
  id: 'explainability/explanation-quality',
  title: 'Evaluating explanations',
  summary:
    'Which explainer to believe: deletion and insertion curves, faithfulness, sensitivity, sparseness and agreement.',
  collection: 'explainability',
  tags: ['evaluation of explanations', 'faithfulness', 'deletion curves', 'robustness', 'Quantus', 'disagreement'],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  maxRows: 400,
  params: dict({
    model: modelControl(['random forest', 'gradient boosting', 'k-nearest neighbours'], 'random forest'),
    rows: int(2, 30, { default: 8, label: 'rows explained' }),
    background: int(5, 60, { default: 20, label: 'background rows' }),
  }),
  run: ({ data, params, key }) => {
    const { fitted, learn } = fitModel(data, params['model'] as ExplainedModel, key)
    return evaluateExplanations(fitted, learn, {
      rows: Number(params['rows']),
      background: Number(params['background']),
      key,
    })
  },
  views: [
    { record: 'explanationQuality', view: 'ExplanationQualityView', title: 'Metrics and agreement' },
    { record: 'curves', title: 'Deletion and insertion curves', area: 'below' },
  ],
  methods: [{ file: 'methods/explain/evaluation.ts', name: 'evaluateExplanations' }],
  references: [
    'Hedström et al. (2023), Quantus: an explainable AI toolkit for responsible evaluation of neural network explanations.',
    'Petsiuk, Das and Saenko (2018), RISE: randomized input sampling for explanation of black-box models.',
    'Bhatt, Weller and Moura (2020), Evaluating and aggregating feature-based model explanations.',
    'Yeh et al. (2019), On the (in)fidelity and sensitivity of explanations.',
    'Krishna et al. (2022), The disagreement problem in explainable machine learning.',
  ],
})
