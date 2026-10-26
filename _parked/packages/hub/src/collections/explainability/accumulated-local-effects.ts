/**
 * The ALE entry: a fitted model's first-order accumulated local effect of a feature and the pure interaction of a pair.
 */
import { MdmlError } from 'mdml-engine/base'
import { matrixView } from 'mdml-engine/learn'
import { aleEffects } from 'mdml-engine/methods/explain'
import { dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** Accumulated local effects of first and second order. */
export const accumulatedLocalEffects: HubEntry = defineEntry({
  id: 'explainability/accumulated-local-effects',
  title: 'Accumulated local effects',
  summary:
    'Effects built from local changes within quantile bins, faithful when features are correlated; and a pair’s pure interaction.',
  collection: 'explainability',
  tags: ['ALE', 'global effects', 'interactions', 'model-agnostic', 'inspection'],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  maxRows: 800,
  params: dict({
    model: modelControl(['gradient boosting', 'random forest', 'k-nearest neighbours'], 'random forest'),
    feature: int(0, 30, { default: 2, label: 'feature' }),
    partner: int(0, 30, { default: 8, label: 'paired with' }),
    bins: int(3, 30, { default: 10, label: 'bins' }),
  }),
  run: ({ data, params, key }) => {
    const { fitted, learn, n } = fitModel(data, params['model'] as ExplainedModel, key)
    if (n === 0) throw new MdmlError('value', 'accumulated-local-effects: no rows')
    const d = matrixView(learn.x, 'explainability', { allowNaN: true }).d
    const f = Math.min(Number(params['feature']), d - 1)
    const g = Math.min(Number(params['partner']), d - 1)
    return aleEffects(fitted, learn, {
      features: [f],
      ...(g === f ? {} : { pairs: [[f, g] as const] }),
      bins: Number(params['bins']),
    })
  },
  views: [
    { record: 'dependence', view: 'DependenceView', title: 'First-order effect' },
    { record: 'dependence', view: 'DependenceSurfaceView', title: 'Second-order interaction', area: 'side' },
  ],
  methods: [{ file: 'methods/explain/inspection.ts', name: 'aleEffects' }],
  references: [
    'Apley and Zhu (2020), Visualizing the effects of predictor variables in black box supervised learning models.',
  ],
})
