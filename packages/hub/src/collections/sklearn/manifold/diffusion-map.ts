import { diffusionMap as diffusionMapMethod } from 'whitebox-engine/methods/manifold'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Diffusion map: coordinates from a random walk on the data’s similarity graph. */
export const diffusionMap: HubEntry = defineEntry({
  id: 'sklearn/manifold/diffusion-map',
  title: 'Diffusion map',
  summary: 'Coordinates from a random walk on the data’s similarity graph.',
  collection: 'sklearn/manifold',
  tags: ['manifold learning', 'embedding', 'spectral'],
  datasets: ['manifold3d', 'classificationTabular'],
  ...fitEstimator(diffusionMapMethod(), { unsupervised: true }),
  views: [{ record: 'embeddingSteps', title: 'Embedding' }],
  methods: [{ file: 'methods/manifold/spectral.ts', name: 'diffusionMap' }],
})
