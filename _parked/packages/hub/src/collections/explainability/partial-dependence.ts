/**
 * The partial dependence entry: a feature's average effect over every row's ICE curve, and a pair's two-way surface.
 */
import { MdmlError } from 'mdml-engine/base'
import { matrixView } from 'mdml-engine/learn'
import { dependencePlot } from 'mdml-engine/methods/explain'
import { choice, dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { type ExplainedModel, fitModel, modelControl } from './_explain.ts'

/** Partial dependence with ICE curves (raw, centred or derivative) and a two-way surface. */
export const partialDependence: HubEntry = defineEntry({
  id: 'explainability/partial-dependence',
  title: 'Partial dependence and ICE',
  summary: 'How the prediction moves as one feature is swept in every row: the average, each row’s curve, and a pair.',
  collection: 'explainability',
  tags: ['partial dependence', 'ICE', 'global effects', 'model-agnostic', 'inspection'],
  datasets: ['regressionTabular', 'classificationTabular'],
  dataset: 'diabetes',
  maxRows: 500,
  params: dict({
    model: modelControl(['gradient boosting', 'random forest', 'k-nearest neighbours'], 'gradient boosting'),
    feature: int(0, 30, { default: 2, label: 'feature' }),
    partner: int(0, 30, { default: 8, label: 'paired with' }),
    transform: choice(['none', 'centred', 'derivative'], { default: 'centred', label: 'ICE curves' }),
    rows: int(0, 200, { default: 60, label: 'ICE rows' }),
  }),
  run: ({ data, params, key }) => {
    const { fitted, learn, n } = fitModel(data, params['model'] as ExplainedModel, key)
    if (n === 0) throw new MdmlError('value', 'partial-dependence: no rows')
    const d = matrixView(learn.x, 'explainability', { allowNaN: true }).d
    const f = Math.min(Number(params['feature']), d - 1)
    const g = Math.min(Number(params['partner']), d - 1)
    return dependencePlot(fitted, learn, {
      features: [f],
      ...(g === f ? {} : { pairs: [[f, g] as const] }),
      transform: params['transform'] as 'none' | 'centred' | 'derivative',
      maxRows: Number(params['rows']),
      gridResolution: 25,
    })
  },
  views: [
    { record: 'dependence', view: 'DependenceView', title: 'Average and individual curves' },
    { record: 'dependence', view: 'DependenceSurfaceView', title: 'The pair together', area: 'side' },
  ],
  methods: [{ file: 'methods/explain/inspection.ts', name: 'dependencePlot' }],
  references: [
    'Friedman (2001), Greedy function approximation: a gradient boosting machine.',
    'Goldstein, Kapelner, Bleich and Pitkin (2015), Peeking inside the black box: visualizing statistical learning with plots of individual conditional expectation.',
  ],
})
