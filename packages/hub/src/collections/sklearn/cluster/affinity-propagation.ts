/**
 * Hub entry for affinity propagation, where points pass responsibility and availability messages until a few emerge as
 * exemplars, on two-dimensional clustering datasets.
 */

import { affinityPropagation as affinityPropagationMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Affinity propagation: points pass messages until a few emerge as exemplars. */
export const affinityPropagation: HubEntry = defineEntry({
  id: 'sklearn/cluster/affinity-propagation',
  title: 'Affinity propagation',
  summary:
    'Points trade responsibility and availability messages until a few emerge as exemplars: step through to watch the clusters settle.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'exemplars', 'message passing'],
  datasets: ['clustering2d'],
  ...fitEstimator(affinityPropagationMethod({ damping: 0.8 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/exemplar.ts', name: 'affinityPropagation' }],
})
