/**
 * Hub entry for probabilistic PCA, PCA as a latent Gaussian model with isotropic noise, on three-dimensional manifolds
 * such as the Swiss roll and tabular data.
 */

import { probabilisticPca as probabilisticPcaMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Probabilistic PCA: pCA as a latent Gaussian model with isotropic noise. */
export const probabilisticPca: HubEntry = defineEntry({
  id: 'sklearn/decomposition/probabilistic-pca',
  title: 'Probabilistic PCA',
  summary: 'PCA as a latent Gaussian model with isotropic noise.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'probabilistic', 'latent variables'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(probabilisticPcaMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/probabilistic.ts', name: 'probabilisticPca' }],
})
