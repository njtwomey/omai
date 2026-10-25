/**
 * Hub entry for probit regression, a linear model through the normal CDF (the latent-variable view of a binary
 * outcome), on two-dimensional classification data.
 */

import { probitRegression as probitRegressionMethod } from 'mdml-engine/methods/linear'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Probit regression: a linear model through the normal CDF: the latent-variable view of a binary outcome. */
export const probitRegression: HubEntry = defineEntry({
  id: 'sklearn/linear-model/probit-regression',
  title: 'Probit regression',
  summary: 'A linear model through the normal CDF: the latent-variable view of a binary outcome.',
  collection: 'sklearn/linear-model',
  tags: ['classification', 'linear', 'probabilistic'],
  datasets: ['classification2d'],
  ...fitEstimator(probitRegressionMethod(), { surface: true }),
  views: [
    { record: 'decisionSurface', title: 'Decision surface' },
    { record: 'coefficientTable', title: 'Coefficients', area: 'side' },
  ],
  methods: [{ file: 'methods/linear/classification.ts', name: 'probitRegression' }],
})
