/**
 * The rules entry: a model or a prediction as readable rules (RuleFit's rule ensemble, an anchor around one row, a
 * logistic model's points scorecard).
 */

import { WhiteboxError } from 'whitebox-engine/base'
import { matrixView } from 'whitebox-engine/learn'
import { randomForestClassifier } from 'whitebox-engine/methods/ensemble'
import { anchorExplanation, logisticScorecard, ruleFitExplanation } from 'whitebox-engine/methods/explain'
import { logisticRegression } from 'whitebox-engine/methods/linear'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Rule-based explanations. */
export const rules: HubEntry = defineEntry({
  id: 'explainability/rules',
  title: 'RuleFit, anchors and scorecards',
  summary:
    'Explanations as rules: a sparse rule ensemble for the whole model, an anchor for one prediction, points per bin.',
  collection: 'explainability',
  tags: ['RuleFit', 'anchors', 'scorecards', 'rules', 'interpretable models'],
  datasets: ['classificationTabular'],
  dataset: 'breastCancer',
  maxRows: 600,
  params: dict({
    kind: choice(['RuleFit', 'anchor', 'scorecard'], { default: 'anchor', label: 'explanation' }),
    row: int(0, 599, { default: 7, label: 'row (anchor)' }),
    trees: int(2, 60, { default: 20, label: 'trees (RuleFit)' }),
  }),
  run: ({ data, params, key }) => {
    const learn = learnDataOf(data)
    const n = matrixView(learn.x, 'rules').n
    if (n === 0) throw new WhiteboxError('value', 'rules: no rows')
    if (params['kind'] === 'RuleFit') return ruleFitExplanation(learn, { nTrees: Number(params['trees']), key })
    if (params['kind'] === 'scorecard')
      return logisticScorecard(logisticRegression({ C: 1 }).fit(learn), learn, { bins: 3 })
    const forest = randomForestClassifier({ nEstimators: 30, maxDepth: 5 }).fit(learn, { key, explain: false })
    return anchorExplanation(forest, learn, Math.min(Number(params['row']), n - 1), { key })
  },
  views: [{ record: 'rules', view: 'RuleListView', title: 'The rules' }],
  methods: [
    { file: 'methods/explain/glassbox.ts', name: 'ruleFitExplanation' },
    { file: 'methods/explain/glassbox.ts', name: 'anchorExplanation' },
    { file: 'methods/explain/glassbox.ts', name: 'logisticScorecard' },
  ],
  references: [
    'Friedman and Popescu (2008), Predictive learning via rule ensembles.',
    'Ribeiro, Singh and Guestrin (2018), Anchors: high-precision model-agnostic explanations.',
    'Siddiqi (2006), Credit risk scorecards.',
  ],
})
