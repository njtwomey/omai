/**
 * The explainable boosting entry: a glass-box model fitted to the data, read by its shape functions, its pairwise
 * terms and their importances.
 */

import { explainableBoostingMachine } from 'mdml-engine/methods/explain'
import { dict, int, real } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Explainable boosting machines (GA2M). */
export const explainableBoosting: HubEntry = defineEntry({
  id: 'explainability/explainable-boosting',
  title: 'Explainable boosting (glass-box)',
  summary:
    'A model that is its own explanation: a shape function per feature and a few pairwise terms, boosted in turn.',
  collection: 'explainability',
  tags: ['EBM', 'GA2M', 'glass-box', 'generalised additive models', 'shape functions', 'interpretable models'],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  maxRows: 800,
  params: dict({
    rounds: int(10, 500, { default: 120, log: true, label: 'boosting rounds' }),
    learningRate: real(0.01, 0.5, { default: 0.1, log: true, label: 'learning rate' }),
    bins: int(4, 64, { default: 24, label: 'bins per feature' }),
    interactions: int(0, 4, { default: 2, label: 'pairwise terms' }),
  }),
  run: ({ data, params }) =>
    explainableBoostingMachine(learnDataOf(data), {
      rounds: Number(params['rounds']),
      learningRate: Number(params['learningRate']),
      bins: Number(params['bins']),
      interactions: Number(params['interactions']),
    }),
  views: [
    { record: 'dependence', view: 'DependenceView', title: 'Shape functions', props: () => ({ all: true }) },
    { record: 'importances', title: 'Term importances', area: 'side' },
    { record: 'dependence', view: 'DependenceSurfaceView', title: 'The strongest pair', area: 'side' },
  ],
  methods: [{ file: 'methods/explain/glassbox.ts', name: 'explainableBoostingMachine' }],
  references: [
    'Lou, Caruana, Gehrke and Hooker (2013), Accurate intelligible models with pairwise interactions.',
    'Nori, Jenkins, Koch and Caruana (2019), InterpretML: a unified framework for machine learning interpretability.',
  ],
})
