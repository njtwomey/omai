/**
 * Hub entry for FastICA, which unmixes signals into maximally non-Gaussian independent sources, on three-dimensional
 * manifolds such as the Swiss roll and tabular data.
 */

import { fastIca as fastIcaMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** FastICA: unmix signals into maximally non-Gaussian, independent sources. */
export const fastIca: HubEntry = defineEntry({
  id: 'sklearn/decomposition/fast-ica',
  title: 'FastICA',
  summary: 'Unmix signals into maximally non-Gaussian, independent sources.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'independent components'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(fastIcaMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/ica.ts', name: 'fastIca' }],
})
