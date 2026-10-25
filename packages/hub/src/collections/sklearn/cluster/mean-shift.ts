/**
 * Hub entry for mean shift, every point climbing a kernel density estimate to a mode, on two-dimensional clustering
 * datasets.
 */

import { meanShift as meanShiftMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Mean shift: move every point uphill on a kernel density until it reaches a mode. */
export const meanShift: HubEntry = defineEntry({
  id: 'sklearn/cluster/mean-shift',
  title: 'Mean shift',
  summary:
    'Every point climbs a kernel density estimate to a mode; the trails show the climb, and points sharing a mode share a cluster.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'density', 'modes'],
  datasets: ['clustering2d'],
  ...fitEstimator(meanShiftMethod({ bandwidth: 2 }), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/density.ts', name: 'meanShift' }],
})
