/**
 * Hub entry for HDBSCAN, a hierarchy of density clusters cut where the clusters are most stable, on two-dimensional
 * clustering datasets.
 */

import { hdbscan as hdbscanMethod } from 'mdml-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** HDBSCAN: a hierarchy of density clusters, cut where clusters are most stable. */
export const hdbscan: HubEntry = defineEntry({
  id: 'sklearn/cluster/hdbscan',
  title: 'HDBSCAN',
  summary:
    'A hierarchy of density clusters, cut where clusters are most stable: a tight blob and a diffuse one are found alike.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'density', 'hierarchical'],
  datasets: ['clustering2d'],
  dataset: 'unequalVarianceBlobs',
  ...fitEstimator(hdbscanMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/density.ts', name: 'hdbscan' }],
})
