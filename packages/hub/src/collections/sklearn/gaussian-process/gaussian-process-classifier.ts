/**
 * Hub entry for Gaussian process classification, a latent Gaussian process squashed to probabilities and approximated
 * by Laplace's method, on two-dimensional classification data and tabular classification data.
 */

import { gaussianProcessClassifier as gaussianProcessClassifierMethod } from 'mdml-engine/methods/gp'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Gaussian process classification: a latent Gaussian process squashed to probabilities, approximated by Laplace’s method. */
export const gaussianProcessClassifier: HubEntry = defineEntry({
  id: 'sklearn/gaussian-process/gaussian-process-classifier',
  title: 'Gaussian process classification',
  summary: 'A latent Gaussian process squashed to probabilities, approximated by Laplace’s method.',
  collection: 'sklearn/gaussian-process',
  tags: ['classification', 'bayesian', 'kernels'],
  datasets: ['classification2d', 'classificationTabular'],
  maxRows: 80,
  ...fitEstimator(gaussianProcessClassifierMethod({ maxIter: 30, surfaceResolution: 6 }), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Predictive probabilities' },
    { record: 'hyperparameterSurface', title: 'Marginal likelihood', area: 'side' },
    { record: 'coefficientTable', title: 'Hyperparameters', area: 'side' },
  ],
  methods: [{ file: 'methods/gp/classification.ts', name: 'gaussianProcessClassifier' }],
})
