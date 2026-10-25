/**
 * Hub entry for label propagation (as scikit-learn's `LabelPropagation`), a few labels flowing along a similarity graph
 * to the unlabelled points, on two-dimensional classification data.
 */

import { labelPropagation as labelPropagationEstimator } from 'whitebox-engine/methods/semi-supervised'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { withHiddenLabels } from './_hidden.ts'

/** Label propagation (sklearn.semi_supervised.LabelPropagation): labels flow along a similarity graph. */
export const labelPropagation: HubEntry = defineEntry({
  id: 'sklearn/semi-supervised/label-propagation',
  title: 'Label propagation',
  summary: 'A few labels flow along a similarity graph to every unlabelled point, the labelled ones held fixed.',
  collection: 'sklearn/semi-supervised',
  tags: ['semi-supervised', 'classification', 'graph', 'transductive'],
  datasets: ['classification2d'],
  knobs: { noise: 0.08 },
  ...withHiddenLabels(labelPropagationEstimator({ kernel: 'knn', nNeighbors: 7 })),
  views: [
    { record: 'clusteringSteps', title: 'Labels spreading through the graph' },
    { record: 'decisionSurface', title: 'Decision surface', area: 'side' },
  ],
  methods: [{ file: 'methods/semi-supervised/graph-based.ts', name: 'labelPropagation' }],
  references: ['Zhu and Ghahramani (2002), Learning from labeled and unlabeled data with label propagation'],
})
