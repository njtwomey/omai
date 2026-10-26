/**
 * Hub entry for incremental PCA, principal components updated batch by batch, on three-dimensional manifolds such as
 * the Swiss roll and tabular data.
 */

import { incrementalPca as incrementalPcaMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Incremental PCA: pCA updated batch by batch, for data that does not fit at once. */
export const incrementalPca: HubEntry = defineEntry({
  id: 'sklearn/decomposition/incremental-pca',
  title: 'Incremental PCA',
  summary: 'PCA updated batch by batch, for data that does not fit at once.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'online'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(incrementalPcaMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/pca.ts', name: 'incrementalPca' }],
})
