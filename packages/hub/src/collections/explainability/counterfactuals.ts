/**
 * The counterfactual entry: what a credit applicant would have to change, by growing spheres, FACE, Wachter, DiCE,
 * prototype-guided search or CEM, under actionability constraints, with robustness to refitting.
 */

import { WhiteboxError } from 'whitebox-engine/base'
import { matrixView } from 'whitebox-engine/learn'
import { randomForestClassifier } from 'whitebox-engine/methods/ensemble'
import { type CounterfactualMethod, explainCounterfactuals } from 'whitebox-engine/methods/explain'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Counterfactual explanations and recourse. */
export const counterfactuals: HubEntry = defineEntry({
  id: 'explainability/counterfactuals',
  title: 'Counterfactuals and recourse',
  summary:
    'The smallest changes that would flip a classifier’s decision, kept actionable, with how often they survive a refit.',
  collection: 'explainability',
  tags: ['counterfactuals', 'recourse', 'DiCE', 'CEM', 'growing spheres', 'FACE', 'actionability', 'contrastive'],
  datasets: ['fairnessTabular', 'classificationTabular'],
  dataset: 'germanCredit',
  maxRows: 1000,
  params: dict({
    model: choice(['logistic regression', 'random forest'], { default: 'logistic regression', label: 'model' }),
    method: choice(['growing spheres', 'face', 'wachter', 'dice', 'prototype', 'cem'], {
      default: 'dice',
      label: 'method (Wachter, DiCE, prototype and CEM need the logistic model)',
    }),
    row: int(0, 999, { default: 1, label: 'row explained' }),
    fixed: int(-1, 30, { default: 0, label: 'immutable feature (-1 for none)' }),
    refits: int(0, 10, { default: 4, label: 'refits for robustness' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const n = matrixView(learn.x, 'counterfactuals').n
    if (n === 0) throw new WhiteboxError('value', 'counterfactuals: no rows')
    const forest = params['model'] === 'random forest'
    const est = forest ? randomForestClassifier({ nEstimators: 30, maxDepth: 6 }) : logisticRegression({ C: 1 })
    const fitted = est.fit(learn, { key, explain: false })
    const fixed = Number(params['fixed'])
    return explainCounterfactuals(fitted, learn, Math.min(Number(params['row']), n - 1), {
      method: params['method'] as CounterfactualMethod,
      count: 3,
      ...(fixed >= 0 ? { constraints: { immutable: [fixed] } } : {}),
      estimator: est,
      refits: Number(params['refits']),
      key,
    })
  },
  views: [{ record: 'counterfactuals', view: 'CounterfactualView', title: 'The row and what would change' }],
  methods: [{ file: 'methods/explain/counterfactuals.ts', name: 'explainCounterfactuals' }],
  references: [
    'Wachter, Mittelstadt and Russell (2017), Counterfactual explanations without opening the black box.',
    'Mothilal, Sharma and Tan (2020), Explaining machine learning classifiers through diverse counterfactual explanations.',
    'Laugel et al. (2018), Comparison-based inverse classification for interpretability in machine learning.',
    'Dhurandhar et al. (2018), Explanations based on the missing: towards contrastive explanations with pertinent negatives.',
    'Van Looveren and Klaise (2021), Interpretable counterfactual explanations guided by prototypes.',
    'Ustun, Spangher and Liu (2019), Actionable recourse in linear classification.',
  ],
})
