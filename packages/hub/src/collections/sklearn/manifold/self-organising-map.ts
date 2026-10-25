/**
 * Hub entry for the self-organising map, a grid of prototypes pulled towards the data with neighbours moving together,
 * on two-dimensional clusters, three-dimensional manifolds and tabular data.
 */

import { selfOrganisingMap as selfOrganisingMapMethod } from 'mdml-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Self-organising map: a grid of prototypes pulled towards the data, neighbours together. */
export const selfOrganisingMap: HubEntry = defineEntry({
  id: 'sklearn/manifold/self-organising-map',
  title: 'Self-organising map',
  summary: 'A grid of prototypes pulled towards the data, neighbours together.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'neural', 'prototypes'],
  datasets: ['clustering2d', 'manifold3d', 'classificationTabular'],
  ...fitEstimator(selfOrganisingMapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/som.ts', name: 'selfOrganisingMap' }],
})
