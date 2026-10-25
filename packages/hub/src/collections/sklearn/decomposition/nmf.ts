/**
 * Hub entry for non-negative matrix factorisation, parts-based factors with non-negative weights and parts, on tabular
 * classification data.
 */

import { nmf as nmfMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Non-negative matrix factorisation: parts-based factors: non-negative weights times non-negative parts. */
export const nmf: HubEntry = defineEntry({
  id: 'sklearn/decomposition/nmf',
  title: 'Non-negative matrix factorisation',
  summary: 'Parts-based factors: non-negative weights times non-negative parts.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'non-negative'],
  datasets: ['classificationTabular'],
  ...fitEstimator(nmfMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/nmf.ts', name: 'nmf' }],
})
