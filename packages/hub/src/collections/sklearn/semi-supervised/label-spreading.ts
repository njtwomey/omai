/**
 * Hub entry for label spreading (as scikit-learn's `LabelSpreading`), label propagation on the normalised graph with
 * soft clamping of the given labels, on two-dimensional classification data.
 */

import { labelSpreading as labelSpreadingEstimator } from 'whitebox-engine/methods/semi-supervised'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { withHiddenLabels } from './_hidden.ts'

/** Label spreading (sklearn.semi_supervised.LabelSpreading): soft clamping on the normalised graph. */
export const labelSpreading: HubEntry = defineEntry({
  id: 'sklearn/semi-supervised/label-spreading',
  title: 'Label spreading',
  summary: 'Like label propagation on the normalised graph, but each point keeps only part of its own label.',
  collection: 'sklearn/semi-supervised',
  tags: ['semi-supervised', 'classification', 'graph', 'transductive'],
  datasets: ['classification2d'],
  knobs: { noise: 0.08 },
  ...withHiddenLabels(labelSpreadingEstimator({ kernel: 'knn', nNeighbors: 7, maxIter: 100, alpha: 0.9 })),
  views: [
    { record: 'clusteringSteps', title: 'Labels spreading through the graph' },
    { record: 'decisionSurface', title: 'Decision surface', area: 'side' },
  ],
  methods: [{ file: 'methods/semi-supervised/graph-based.ts', name: 'labelSpreading' }],
  references: ['Zhou, Bousquet, Lal, Weston and Schölkopf (2004), Learning with local and global consistency'],
})
