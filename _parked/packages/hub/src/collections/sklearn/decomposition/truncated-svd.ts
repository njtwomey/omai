/**
 * Hub entry for truncated SVD, the leading singular vectors without centring (as in latent semantic analysis), on
 * three-dimensional manifolds such as the Swiss roll and tabular data.
 */

import { truncatedSvd as truncatedSvdMethod } from 'mdml-engine/methods/decomposition'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Truncated SVD: the leading singular vectors without centring (latent semantic analysis). */
export const truncatedSvd: HubEntry = defineEntry({
  id: 'sklearn/decomposition/truncated-svd',
  title: 'Truncated SVD',
  summary: 'The leading singular vectors without centring (latent semantic analysis).',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'linear algebra'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(truncatedSvdMethod({ nComponents: 2 }), { unsupervised: true }),
  views: [{ record: 'components', title: 'Components' }],
  methods: [{ file: 'methods/decomposition/svd.ts', name: 'truncatedSvd' }],
})
