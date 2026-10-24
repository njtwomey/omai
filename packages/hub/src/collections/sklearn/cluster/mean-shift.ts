import { meanShift as meanShiftMethod } from 'whitebox-engine/methods/cluster'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Mean shift: move every point uphill on a kernel density until it reaches a mode. */
export const meanShift: HubEntry = defineEntry({
  id: 'sklearn/cluster/mean-shift',
  title: 'Mean shift',
  summary: 'Move every point uphill on a kernel density until it reaches a mode.',
  collection: 'sklearn/cluster',
  tags: ['clustering', 'density', 'modes'],
  datasets: ['clustering2d'],
  ...fitEstimator(meanShiftMethod(), { unsupervised: true }),
  views: [{ record: 'clusteringSteps', title: 'Clustering steps' }],
  methods: [{ file: 'methods/cluster/density.ts', name: 'meanShift' }],
})
