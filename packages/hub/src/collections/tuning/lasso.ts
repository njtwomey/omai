/**
 * Hub entry for a grid search over the lasso's penalty by cross-validated R squared with its standard errors, on
 * tabular regression data.
 */

import { lasso } from 'mdml-engine/methods/linear'
import { dict, real } from 'mdml-engine/space'
import type { HubEntry } from '../../entry/entry.ts'
import { tuningEntry } from './_tuning.ts'

/** Choosing the lasso's penalty by cross-validation over a grid. */
export const lassoSearch: HubEntry = tuningEntry({
  name: 'lasso',
  title: 'Choosing the lasso’s alpha',
  summary: 'A grid search over the lasso penalty: cross-validated R² with its standard errors, too little to too much.',
  tags: ['linear models', 'regularisation', 'grid search'],
  datasets: ['regressionTabular'],
  dataset: 'friedman1',
  // The regularisation path is the lasso entry's; a search needs only the fit.
  estimator: lasso({ pathLength: 0 }),
  space: dict({ alpha: real(1e-3, 100, { log: true, label: 'alpha' }) }),
  sampler: 'grid',
  trials: 15,
  cv: 5,
  views: [
    { record: 'crossValidationCurve', title: 'Cross-validated R² against alpha' },
    { record: 'searchTrials', title: 'Trials', area: 'side' },
    { record: 'learningCurve', title: 'Each trial and the best so far', area: 'below' },
  ],
  estimatorRef: { file: 'methods/linear/regression.ts', name: 'lasso' },
})
