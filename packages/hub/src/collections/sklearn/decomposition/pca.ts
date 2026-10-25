/**
 * Hub entry for principal component analysis (as scikit-learn's `PCA`), rotating onto the directions of largest
 * variance, on three-dimensional manifolds such as the Swiss roll and tabular data.
 */

import { pca as pcaEstimator } from 'whitebox-engine/methods/decomposition'
import { int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Principal component analysis (sklearn.decomposition.PCA): the axes of largest variance. */
export const pca: HubEntry = defineEntry({
  id: 'sklearn/decomposition/pca',
  title: 'PCA',
  summary: 'Rotate onto the directions of largest variance and keep the first few.',
  collection: 'sklearn/decomposition',
  tags: ['decomposition', 'dimensionality reduction', 'linear'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(pcaEstimator({ nComponents: 2 }), {
    unsupervised: true,
    override: { nComponents: int(1, 3, { default: 2, label: 'components' }) },
  }),
  views: [{ record: 'components', title: 'Components and explained variance' }],
  methods: [{ file: 'methods/decomposition/pca.ts', name: 'pca' }],
})
