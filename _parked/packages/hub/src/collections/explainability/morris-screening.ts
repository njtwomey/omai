/**
 * The Morris screening entry: elementary effects of a fitted model over the box its data span.
 */
import { morrisScreening as screen } from 'mdml-engine/methods/explain'
import { dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** Morris elementary effects as a global screening. */
export const morrisScreening: HubEntry = defineEntry({
  id: 'explainability/morris-screening',
  title: 'Morris screening',
  summary:
    'A cheap global sensitivity screen: features moved one at a time along random trajectories through the data’s box.',
  collection: 'explainability',
  tags: ['Morris', 'elementary effects', 'global sensitivity', 'screening', 'perturbation'],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  params: dict({
    model: modelControl(['gradient boosting', 'random forest', 'k-nearest neighbours'], 'gradient boosting'),
    trajectories: int(2, 200, { default: 40, log: true, label: 'trajectories' }),
    levels: int(2, 12, { default: 4, label: 'grid levels' }),
  }),
  run: ({ data, params, key }) => {
    const { fitted, learn } = fitModel(data, params['model'] as ExplainedModel, key)
    return screen(fitted, learn, {
      nTrajectories: Number(params['trajectories']),
      levels: Number(params['levels']),
      key,
    })
  },
  views: [{ record: 'importances', title: 'mu* with sigma as the spread, every trajectory a dot' }],
  methods: [{ file: 'methods/explain/perturbation.ts', name: 'morrisScreening' }],
  references: [
    'Morris (1991), Factorial sampling plans for preliminary computational experiments.',
    'Campolongo, Cariboni and Saltelli (2007), An effective screening design for sensitivity analysis of large models.',
  ],
})
