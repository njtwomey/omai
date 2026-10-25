/**
 * Hub entry for diffusion maps, coordinates from a random walk on the data's similarity graph, on three-dimensional
 * manifolds such as the Swiss roll and tabular data.
 */

import { diffusionMap as diffusionMapMethod } from 'mdml-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'
import { rowColours } from '../../../entry/slots.ts'

/** Diffusion map: coordinates from a random walk on the data’s similarity graph. */
export const diffusionMap: HubEntry = defineEntry({
  id: 'sklearn/manifold/diffusion-map',
  title: 'Diffusion map',
  summary: 'Coordinates from a random walk on the data’s similarity graph.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'spectral'],
  datasets: ['classificationTabular', 'manifold3d'],
  ...fitEstimator(diffusionMapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding', view: 'EmbeddingPlayer', props: (_, c) => rowColours(c) }],
  methods: [{ file: 'methods/manifold/spectral.ts', name: 'diffusionMap' }],
})
