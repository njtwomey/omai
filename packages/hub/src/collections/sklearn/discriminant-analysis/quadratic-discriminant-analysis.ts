/**
 * Hub entry for quadratic discriminant analysis, Gaussian classes with their own covariances and so quadratic
 * boundaries, on two-dimensional classification data and tabular classification data.
 */

import { quadraticDiscriminantAnalysis as quadraticDiscriminantAnalysisMethod } from 'whitebox-engine/methods/discriminant'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Quadratic discriminant analysis: gaussian classes with their own covariances: quadratic boundaries. */
export const quadraticDiscriminantAnalysis: HubEntry = defineEntry({
  id: 'sklearn/discriminant-analysis/quadratic-discriminant-analysis',
  title: 'Quadratic discriminant analysis',
  summary:
    'Gaussian classes with their own covariances: quadratic boundaries, here nested ellipses around shells of one Gaussian.',
  collection: 'sklearn/discriminant-analysis',
  tags: ['classification', 'generative'],
  datasets: ['classification2d', 'classificationTabular'],
  dataset: 'gaussianQuantiles',
  ...fitEstimator(quadraticDiscriminantAnalysisMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/discriminant/discriminant.ts', name: 'quadraticDiscriminantAnalysis' }],
})
