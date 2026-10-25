/**
 * Hub entry for conformal intervals for a regressor (split, normalised and CV+) that cover a chosen fraction of new
 * rows, on one-dimensional regression curves with noise that varies.
 */

import { kNeighborsRegressor } from 'whitebox-engine/methods/neighbours'
import { type ConformalRegressionMethod, conformalRegression as conformal } from 'whitebox-engine/methods/trust'
import { choice, dict, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { learnDataOf } from '../../entry/run.ts'

/** Split, normalised and CV+ conformal intervals for a regressor. */
export const conformalRegression: HubEntry = defineEntry({
  id: 'trust/conformal-regression',
  title: 'Conformal intervals',
  summary: 'Intervals that cover 1 − α of new rows for any regressor: split, normalised and CV+.',
  collection: 'trust',
  tags: ['conformal prediction', 'intervals', 'coverage', 'uncertainty'],
  datasets: ['regression1d'],
  dataset: 'heteroscedastic',
  knobs: { n: 600 },
  params: dict({
    method: choice(['split', 'normalised', 'cv-plus'], { default: 'split' }),
    alpha: real(0.01, 0.5, { default: 0.1, label: 'α (miscoverage)' }),
  }),
  run: (input) =>
    conformal(kNeighborsRegressor({ nNeighbors: 15 }), learnDataOf(input.data), {
      method: input.params['method'] as ConformalRegressionMethod,
      alpha: input.params['alpha'] as number,
      key: input.key,
    }),
  views: [{ record: 'conformal', title: 'Test rows with their intervals' }],
  methods: [{ file: 'methods/trust/conformal.ts', name: 'conformalRegression' }],
  references: ['Angelopoulos and Bates (2021), A gentle introduction to conformal prediction', 'MAPIE'],
})
