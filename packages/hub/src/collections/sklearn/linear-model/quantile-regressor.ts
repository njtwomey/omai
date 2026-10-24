import { quantileRegressor as quantileRegressorMethod } from 'whitebox-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Quantile regression: linear fits to a conditional quantile by the pinball loss. */
export const quantileRegressor: HubEntry = defineEntry({
  id: 'sklearn/linear-model/quantile-regressor',
  title: 'Quantile regression',
  summary: 'Linear fits to a conditional quantile by the pinball loss.',
  collection: 'sklearn/linear-model',
  tags: ['regression', 'linear', 'quantiles'],
  datasets: ['regression1d', 'regressionTabular'],
  dataset: 'quantileCurves',
  ...fitEstimator(quantileRegressorMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Fitted function' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/robust.ts', name: 'quantileRegressor' }],
})
