import { linearDiscriminantAnalysis as linearDiscriminantAnalysisMethod } from 'whitebox-engine/methods/discriminant'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Linear discriminant analysis: gaussian classes sharing one covariance: linear boundaries. */
export const linearDiscriminantAnalysis: HubEntry = defineEntry({
  id: 'sklearn/discriminant-analysis/linear-discriminant-analysis',
  title: 'Linear discriminant analysis',
  summary: 'Gaussian classes sharing one covariance: linear boundaries.',
  collection: 'sklearn/discriminant-analysis',
  tags: ['classification', 'generative', 'linear'],
  datasets: ['classification2d', 'classificationTabular'],
  ...fitEstimator(linearDiscriminantAnalysisMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'components', title: 'Components', area: 'side' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/discriminant/discriminant.ts', name: 'linearDiscriminantAnalysis' }],
})
