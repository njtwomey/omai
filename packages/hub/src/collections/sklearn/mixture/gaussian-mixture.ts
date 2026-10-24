import { gaussianMixture as gaussianMixtureMethod } from 'whitebox-engine/methods/mixture'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { fitEstimator } from '../../../entry/run.ts'

/** Gaussian mixture: soft clusters as Gaussian components, fitted by expectation maximisation. */
export const gaussianMixture: HubEntry = defineEntry({
  id: 'sklearn/mixture/gaussian-mixture',
  title: 'Gaussian mixture',
  summary:
    'Soft clusters as Gaussian components fitted by expectation maximisation: full covariances stretch to fit the slanted blobs.',
  collection: 'sklearn/mixture',
  tags: ['clustering', 'density estimation', 'probabilistic', 'EM'],
  datasets: ['clustering2d', 'density2d'],
  dataset: 'anisotropicBlobs',
  ...fitEstimator(gaussianMixtureMethod({ nComponents: 3 }), { unsupervised: true }),
  views: [
    { record: 'mixtureSteps', title: 'EM steps' },
    { record: 'decisionSurface', title: 'Density', area: 'side' },
  ],
  methods: [{ file: 'methods/mixture/gaussian.ts', name: 'gaussianMixture' }],
})
